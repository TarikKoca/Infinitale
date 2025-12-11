from __future__ import annotations

from celery import shared_task
from celery.schedules import crontab
from typing import Optional
import logging

from django.db import transaction

from user import services as svc
from user.models import GameTurn
from django.conf import settings
from django.db import transaction
from user.models import User as ProfileUser, Game, GameStory, GameChapter, GameConstants


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=5,
)
def process_turn_task(self, turn_id: str) -> Optional[str]:
    """Background job to process a reserved turn."""
    logger = logging.getLogger(__name__)
    # Import LLM functions fresh for each task execution to avoid coroutine reuse
    from user.prompts import llm_next
    logger.info(f"Processing turn task {turn_id}")
    sfx = GameConstants.get_sound_effects()

    # Run potentially async-capable pipeline without blocking main worker loop
    import asyncio
    out = asyncio.run(
        svc.a_process_turn_job(
            str(turn_id),
            llm_next=llm_next,
            tts_narration=svc.tts_narration,
            tts_character=svc.tts_character,
            tti_generate=svc.tti_generate,
            sound_effects=sfx,
        )
    )
    logger.info(f"Turn task {turn_id} completed successfully")
    return str(out.id)

@shared_task(
    bind=True,
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=getattr(settings, "CELERY_TASK_SOFT_TIME_LIMIT", 1140),
    time_limit=getattr(settings, "CELERY_TASK_TIME_LIMIT", 1200),
)
def create_game_task(self, profile_user_id: str, *, genre: str, main_character: str, visual_style: str, difficulty: str, story_length: str, extra_requests: str) -> dict:
    """Background job to create a game and its assets with progress updates.
    
    Timeouts:
    - soft_time_limit: 4 minutes (raises SoftTimeLimitExceeded for graceful handling)
    - time_limit: 5 minutes (force kills the task)
    """
    import logging
    logger = logging.getLogger(__name__)
    # Debug: detect if running eagerly in web process vs. worker
    try:
        logger.info(
            "create_game_task start: is_eager=%s hostname=%s pid=%s",
            getattr(self.request, "is_eager", None),
            getattr(self.request, "hostname", None),
            getattr(self.request, "pid", None),
        )
    except Exception:
        pass
    from celery.exceptions import SoftTimeLimitExceeded
    # Import LLM functions fresh for each task execution to avoid coroutine reuse
    from user.prompts import llm_story, llm_plan_chapters, llm_next
    
    logger.info(f"Starting game creation for user {profile_user_id} - genre: {genre}, character: {main_character}")
    logger.info(f"Task ID: {self.request.id}, Retries: {self.request.retries}")
    
    game = None  # Track game for cleanup
    
    try:
        self.update_state(state="PROGRESS", meta={"step": "initializing", "pct": 5})

        with transaction.atomic():
            prof = ProfileUser.objects.select_for_update().get(pk=profile_user_id)
            logger.debug(f"Creating game with genre={genre}, character={main_character}, visual={visual_style}")
            # Default to user's preferred language if available
            lang = getattr(prof, 'preferred_language', 'en') or 'en'
            game = Game.objects.create(
                user=prof,
                genre_name=genre,
                main_character_name=main_character,
                visual_style=visual_style,
                difficulty=difficulty,
                story_length=story_length,
                extra_requests=extra_requests or "",
                status='creating',
                error_count=0,
                language_code=lang,
            )

        names = GameConstants.get_names()
        voices = GameConstants.get_voices()
        sfx = GameConstants.get_sound_effects()

        # Story generation with retry-aware error handling
        try:
            self.update_state(state="PROGRESS", meta={"step": "story", "pct": 20})
            logger.debug(f"Generating story for game {game.id}")
            svc.generate_story(game_id=str(game.id), llm_story=llm_story, extra_requests=extra_requests, names_by_gender=names)
        except Exception as story_error:
            logger.error(f"Story generation failed for game {game.id}: {story_error}")
            raise ValueError(f"Failed to generate story: {str(story_error)}")

        # Character generation with retry-aware error handling
        try:
            self.update_state(state="PROGRESS", meta={"step": "characters", "pct": 60})
            logger.debug(f"Generating characters for game {game.id}")
            svc.generate_characters(
                game_id=str(game.id),
                character_count=8,
                tti_generate=svc.tti_generate,
                names_by_gender=names,
                voices_by_gender=voices,
                overwrite=False,
            )
        except Exception as char_error:
            logger.error(f"Character generation failed for game {game.id}: {char_error}")
            raise ValueError(f"Failed to generate characters: {str(char_error)}")

        # Chapter planning with retry-aware error handling
        try:
            self.update_state(state="PROGRESS", meta={"step": "chapters", "pct": 85})
            logger.debug(f"Planning chapters for game {game.id}")
            # Use free models for chapter planning if free plan
            svc.plan_chapters(game_id=str(game.id), llm_plan_chapters=llm_plan_chapters)
        except Exception as chapter_error:
            logger.error(f"Chapter planning failed for game {game.id}: {chapter_error}")
            raise ValueError(f"Failed to plan chapters: {str(chapter_error)}")

        # Bootstrap gameplay: generate intro + first real turn so users can play immediately
        try:
            self.update_state(state="PROGRESS", meta={"step": "initial_turns", "pct": 92})
            logger.debug(f"Generating initial turns for game {game.id}")
            # Intro (world narration)
            svc.generate_turn(
                game_id=str(game.id),
                user_text=None,
                llm_next=llm_next,
                tts_narration=svc.tts_narration,
                tts_character=svc.tts_character,
                tti_generate=svc.tti_generate,
                sound_effects=sfx,
            )
            # First actionable turn with automatic "Let's start." message
            svc.generate_turn(
                game_id=str(game.id),
                user_text="Let's start.",
                llm_next=llm_next,
                tts_narration=svc.tts_narration,
                tts_character=svc.tts_character,
                tti_generate=svc.tti_generate,
                sound_effects=sfx,
            )
            logger.debug(f"Initial turns generated successfully for game {game.id}")
        except Exception as turn_error:
            # Initial turns are optional - log warning but don't fail the task
            logger.warning(f"Initial turns generation failed for game {game.id}: {str(turn_error)}", exc_info=True)

        self.update_state(state="PROGRESS", meta={"step": "finalizing", "pct": 95})
        
        # Final validation before returning
        game.refresh_from_db()
        if not GameStory.objects.filter(game=game).exists():
            raise ValueError(f"Game {game.id} created but story is missing")
        if not GameChapter.objects.filter(game=game).exists():
            raise ValueError(f"Game {game.id} created but chapters are missing")
        
        # Mark game as ready to play
        game.status = 'ongoing'
        game.save(update_fields=['status'])
        
        logger.info(f"Game {game.id} created successfully for user {profile_user_id}")
        # Explicitly return SUCCESS state
        result = {"ok": True, "game_id": str(game.id)}
        self.update_state(state="SUCCESS", meta=result)
        return result
        
    except SoftTimeLimitExceeded:
        # Determine configured soft time limit (prefer per-call setting if available)
        try:
            timelimit = getattr(self.request, "timelimit", None)
            soft_limit = None
            if isinstance(timelimit, (list, tuple)) and len(timelimit) == 2:
                soft_limit = timelimit[0]
            if not soft_limit:
                soft_limit = getattr(settings, "CELERY_TASK_SOFT_TIME_LIMIT", 1140)
        except Exception:
            soft_limit = getattr(settings, "CELERY_TASK_SOFT_TIME_LIMIT", 1140)
        logger.error(f"Game creation timed out after {int(soft_limit)} seconds for user {profile_user_id}")
        # Cleanup game on timeout
        _cleanup_failed_game(game, f"timed out after {int(soft_limit)} seconds")
        self.update_state(
            state='FAILURE',
            meta={'exc_type': 'Timeout', 'exc_message': 'Game creation timed out. Please try again.'}
        )
        raise
    except Exception as e:
        error_msg = f"Game creation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Record error on game before cleanup
        if game:
            try:
                from user.models import GameErrorType
                game.status = 'failed'
                game.record_error(GameErrorType.CREATION, error_msg, is_critical=True)
            except Exception as record_error:
                logger.error(f"Failed to record error on game {game.id}: {record_error}")
        
        # Cleanup game on any failure
        _cleanup_failed_game(game, str(e))
        # Store generic error message for frontend, keep details in logs
        self.update_state(
            state='FAILURE',
            meta={'exc_type': type(e).__name__, 'exc_message': 'An error occurred while creating your game. Please try again.'}
        )
        raise

def _cleanup_failed_game(game, reason: str):
    """Clean up a failed game and all related data."""
    if not game:
        return
        
    logger = logging.getLogger(__name__)
    try:
        game_id = game.id
        # Game deletion will cascade to all related objects:
        # - GameStory, GameChapter, GameTurn (via foreign keys)
        # - GameCharacter (via ManyToMany relationship)
        # - Any media files are cleaned up by the model's delete methods
        game.delete()
        logger.info(f"Cleaned up failed game {game_id} - reason: {reason}")
    except Exception as cleanup_error:
        logger.error(f"Failed to cleanup game {game.id if game else 'Unknown'}: {cleanup_error}")


@shared_task(bind=True)
def cleanup_stuck_games_task(self):
    """
    Periodic task to clean up games stuck in creating status.
    This task should be run every 15 minutes via celery beat.
    """
    logger = logging.getLogger(__name__)
    try:
        cleaned_count = svc.check_and_cleanup_stuck_games(timeout_minutes=10)
        logger.info(f"Stuck games cleanup task completed: {cleaned_count} games cleaned")
        return f"Cleaned up {cleaned_count} stuck games"
    except Exception as e:
        logger.error(f"Error in stuck games cleanup task: {e}")
        raise self.retry(countdown=300, max_retries=3)  # Retry after 5 minutes
