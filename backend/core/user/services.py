# user/services.py
from __future__ import annotations

from typing import Callable, Iterable, Sequence, Optional, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import logging
from django.db import transaction
from django.core.exceptions import ValidationError
from django.db.models import Max
from django.db import IntegrityError

from django.utils import timezone
from django.conf import settings
from pathlib import Path

from user.models import Game, GameStory, GameChapter, GameTurn, GameCharacter, TurnStatus, UsageTracking, User
from decimal import Decimal

from openai import OpenAI, AsyncOpenAI
import replicate
import requests
import hashlib
import uuid
import os
from user.circuit_breaker import openai_breaker, replicate_breaker, fal_breaker
import asyncio
from asgiref.sync import sync_to_async
import httpx

def _create_story_summary(messages: list[dict], current_summary: str, game, chapter) -> str:
    """
    Helper function to create story summaries.
    Falls back to simple concatenation if summarization fails.
    
    Args:
        messages: Message history to summarize
        current_summary: Existing realized_story content
        game: Game object for context
        chapter: Chapter object for context
        
    Returns:
        String summary of the story progression
    """
    try:
        from user.prompts import llm_summarize_story
        
        # Get character list for context
        character_names = list(game.characters.values_list('name', flat=True))
        characters_str = ", ".join(character_names)
        
        # Create summary from message history, merging with previous continuity log if present
        summary_result = llm_summarize_story(
            messages=messages,
            current_summary=current_summary,
        )
        
        logger.info(f"Story summarization completed")
        return summary_result
        
    except Exception as e:
        logger.error(f"Story summarization failed: {e}", exc_info=True)
        raise

logger = logging.getLogger("user.services")


def cleanup_stuck_game_creation(game_id: str, timeout_minutes: int = 10) -> bool:
    """
    Clean up a stuck game creation process.
    
    Args:
        game_id: The ID of the game to clean up
        timeout_minutes: How long to wait before considering a game creation stuck
        
    Returns:
        True if cleanup was performed, False if game wasn't stuck or cleanup failed
    """
    try:
        game = Game.objects.get(id=game_id)
    except Game.DoesNotExist:
        logger.warning(f"Game {game_id} not found for cleanup")
        return False
    
    if game.status != 'creating':
        logger.info(f"Game {game_id} is not in creating status, no cleanup needed")
        return False
    
    # Check if game creation has been stuck for more than timeout
    time_since_creation = timezone.now() - game.created_at
    if time_since_creation.total_seconds() < (timeout_minutes * 60):
        logger.info(f"Game {game_id} creation still within timeout ({time_since_creation.total_seconds():.0f}s)")
        return False
    
    logger.warning(f"Cleaning up stuck game {game_id} (stuck for {time_since_creation.total_seconds()/60:.1f} minutes)")
    
    try:
        # Clean up Redis locks and task state
        import redis
        redis_client = redis.Redis.from_url(settings.REDIS_URL)
        
        # Remove game creation lock and task state
        lock_keys = [
            f"game_creation_lock:{game_id}",
            f"game_creation_task:{game_id}",
            f"game_creation_status:{game_id}",
        ]
        
        for key in lock_keys:
            redis_client.delete(key)
        
        # Remove any Celery task result keys
        task_keys = redis_client.keys(f"celery-task-meta-*{game_id}*")
        if task_keys:
            redis_client.delete(*task_keys)
        
        # Update game status to failed
        game.status = 'failed'
        game.error_message = f"Game creation timed out after {time_since_creation.total_seconds()/60:.1f} minutes"
        game.save(update_fields=['status', 'error_message'])
        
        logger.info(f"Successfully cleaned up stuck game {game_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning up stuck game {game_id}: {e}")
        return False


def check_and_cleanup_stuck_games(timeout_minutes: int = 10) -> int:
    """
    Check for and clean up all stuck game creation processes.
    
    Args:
        timeout_minutes: How long to wait before considering a game creation stuck
        
    Returns:
        Number of games cleaned up
    """
    from datetime import timedelta
    
    cutoff_time = timezone.now() - timedelta(minutes=timeout_minutes)
    stuck_games = Game.objects.filter(
        status='creating',
        created_at__lt=cutoff_time
    )
    
    cleaned_count = 0
    for game in stuck_games:
        if cleanup_stuck_game_creation(str(game.id), timeout_minutes):
            cleaned_count += 1
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} stuck game(s)")
    
    return cleaned_count

# ==============================
# Circuit Breaker Wrapped API Calls
# ==============================

@fal_breaker
def _fal_api_call(model: str, arguments: dict, timeout: int = 30):
    """Wrapped fal.ai API call with circuit breaker."""
    import fal_client
    handler = fal_client.submit(model, arguments=arguments)
    return handler.get()

@replicate_breaker
def _replicate_api_call(model: str, input_data: dict):
    """Wrapped Replicate API call with circuit breaker."""
    return replicate.run(model, input=input_data)



# Async adapters for network-bound SDKs that are sync
async def a_fal_api_call(model: str, arguments: dict, timeout: int = 30):
    api_key = os.getenv("FAL_KEY", "").strip() or os.getenv("FAL_API_KEY", "").strip()
    api_url = os.getenv("FAL_API_URL", "").strip()
    if not api_key or not api_url:
        return await asyncio.to_thread(_fal_api_call, model, arguments, timeout)
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    url = f"{api_url.rstrip('/')}/{model.lstrip('/')}"
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        r = await client.post(url, headers=headers, json=arguments)
        r.raise_for_status()
        return r.json()

async def a_replicate_api_call(model: str, input_data: dict):
    """
    Async Replicate prediction: if model contains a version (owner/model:version),
    use HTTP API directly; else fall back to thread-wrapped SDK call.
    """
    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if ":" not in (model or "") or not token:
        return await asyncio.to_thread(_replicate_api_call, model, input_data)

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    payload = {"version": model, "input": input_data}
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        r = await client.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
        if r.status_code >= 400:
            try:
                data = r.json()
            except Exception:
                data = {}
            raise ValueError(f"Replicate create failed: {data}")
        data = r.json()
        status = data.get("status")
        get_url = data.get("urls", {}).get("get")
        if not get_url:
            raise ValueError("Replicate response missing get URL")
        # Poll until terminal
        import math
        backoff = 0.5
        for _ in range(120):  # up to ~60s
            if status in {"succeeded", "failed", "canceled"}:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.2, 2.5)
            pr = await client.get(get_url, headers=headers)
            pr.raise_for_status()
            data = pr.json()
            status = data.get("status")
        if status != "succeeded":
            raise ValueError(f"Replicate prediction ended with status={status}")
        return data.get("output")



# ==============================
# Ports / Types
# ==============================

# LLM
LLMStoryFn = Callable[[str, str, str, str, str], dict]
LLMPlanChaptersFn = Callable[[str, str, Iterable[tuple[str, str]], int, str], Sequence[str]]

class LLMNextOutput(Protocol):
    scene: str
    voice_name: str
    image_description: str
    choices: Sequence[str]
    voice_description: Optional[str]
    selected_sound_effect: str

LLMNextFn = Callable[[list[dict], Game, GameChapter, GameStory], LLMNextOutput]

# TTS (final CDN URL ve duration döner)
TTSNarrationFn = Callable[[str, str, str, Optional[Game]], tuple[str, float]]        # (text, who, user_id, game) -> (cdn_audio_url, duration)
TTSCharacterFn = Callable[[str, str], tuple[str, float]]  # (text, voice_name) -> (cdn_audio_url, duration)

# TTI (final CDN URL döner)
TTIFn = Callable[[str, Optional[str]], str]                          # (image_description, reference_image_url|None) -> cdn_image_url


# ==============================
# Low-level helpers
# ==============================

# --- Character reference prompt template + renderer ---
CHAR_REF_TMPL = (
    "a {visual_style} full body portrait of a {gender}. "
    "full body character reference. good looking shoes."
)

def _make_idempotency_key(game_id: str, chapter_index: int, user_text: str | None) -> str:
    # In DEBUG mode, include timestamp to ensure uniqueness for synchronous generation
    from django.conf import settings
    if settings.DEBUG:
        from django.utils import timezone
        raw = f"{game_id}:{chapter_index}:{user_text or ''}:{timezone.now().isoformat()}"
    else:
        # Include more context to prevent collisions
        # Get the current turn count for this chapter to make key unique
        from user.models import GameTurn, GameChapter
        try:
            chapter = GameChapter.objects.get(game_id=game_id, index=chapter_index)
            turn_count = GameTurn.objects.filter(chapter=chapter).count()
            raw = f"{game_id}:{chapter_index}:{turn_count}:{user_text or ''}"
        except GameChapter.DoesNotExist:
            # Fallback if chapter doesn't exist yet
            raw = f"{game_id}:{chapter_index}:0:{user_text or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _with_retry(func: Callable[[], str], *, attempts: int = 3, delay_seconds: float = 1.0) -> str:
    """Run func, retrying on Exception a few times. Returns func() result or raises last error."""
    from user.security import with_exponential_backoff
    return with_exponential_backoff(func, max_attempts=attempts, initial_delay=delay_seconds)

def _should_retry_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry."""
    from user.circuit_breaker import CircuitBreakerError
    
    # Don't retry circuit breaker errors (they indicate API is down)
    if isinstance(error, CircuitBreakerError):
        return False
    
    # Don't retry validation/programming errors
    if isinstance(error, (ValidationError, ValueError, TypeError)):
        return False
    
    # Retry network/API errors
    import requests
    if isinstance(error, (requests.RequestException, ConnectionError, TimeoutError)):
        return True
    
    # Retry other generic exceptions that might be transient
    return True

def _with_smart_retry(func: Callable, *, max_attempts: int = 5, operation_type: str = "api_call") -> any:
    """
    Enhanced retry mechanism with circuit breaker integration and smart error handling.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of retry attempts
        operation_type: Type of operation for logging
        
    Returns:
        Result of the function call
        
    Raises:
        Last exception if all retries fail
    """
    import time
    import random
    from user.circuit_breaker import CircuitBreakerError
    
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            logger.warning(f"{operation_type} attempt {attempt}/{max_attempts} failed: {str(e)[:200]}")
            
            # Don't retry if this error type shouldn't be retried
            if not _should_retry_error(e):
                logger.info(f"Not retrying {operation_type} due to non-retryable error: {type(e).__name__}")
                raise e
            
            # Don't retry on the last attempt
            if attempt == max_attempts:
                break
            
            # Calculate exponential backoff with jitter
            base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, 16s
            jitter = random.uniform(0.8, 1.2)  # ±20% jitter
            delay = min(30, base_delay * jitter)  # Max 30 seconds
            
            logger.info(f"Retrying {operation_type} in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
    
    # All attempts failed
    logger.error(f"{operation_type} failed after {max_attempts} attempts with final error: {str(last_error)[:500]}")
    raise last_error or Exception(f"{operation_type} failed after {max_attempts} attempts")

def _next_turn_index_locked(chapter: GameChapter) -> int:
    last = GameTurn.objects.filter(chapter=chapter).only("index").order_by("-index").first()
    return (last.index if last else -1) + 1

def render_char_ref_prompt(tmpl: str, visual_style: str, gender: str) -> str:
    return tmpl.format(visual_style=visual_style, gender=gender)

def enhance_visual_prompt(image_description: str, visual_style: str) -> str:
    """
    Enhance image description with style-specific suffixes for improved TTI generation quality.
    
    Args:
        image_description: Original image description from LLM
        visual_style: Visual style from game settings (realistic, illustration, anime, pixel_art)
    
    Returns:
        Enhanced image description with style-specific quality modifiers
    """
    from user.models import GameConstants
    
    if not image_description or not image_description.strip():
        return image_description
    
    # Get style-specific enhancement suffix
    enhancement = GameConstants.get_visual_style_enhancement(visual_style)
    if not enhancement:
        logger.debug(f"No enhancement found for visual style: {visual_style}")
        return image_description
    
    # Combine original description with enhancement suffix
    enhanced_prompt = f"{image_description.strip()}{enhancement}"
    
    logger.debug(f"Enhanced visual prompt for {visual_style}: {len(enhanced_prompt)} chars")
    return enhanced_prompt

def _trim_messages_by_token_cap(messages: list[dict], token_cap: int, encoding_name: str = "cl100k_base") -> list[dict]:
    """Trim a chat message list to fit within a token cap, preserving the first system message.

    Strategy:
    - Keep the first message if it has role==system
    - Then add messages from the end backwards until the token cap is reached
    - If the system message alone exceeds the cap, return only the system message
    """
    if token_cap <= 0:
        return []
    if not messages:
        return []

    # Lazy import to avoid hard dependency at import time
    encoder = None
    try:
        import tiktoken  # type: ignore
        try:
            encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoder = None

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception:
                pass
        # Fallback heuristic: ~4 chars per token
        return max(1, len(text) // 4)

    # Separate out leading system message if present
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        other_msgs = messages[1:]
    else:
        system_msg = None
        other_msgs = messages

    # Initialize token accumulator including system content if present
    acc_tokens = 0
    kept: list[dict] = []

    if system_msg:
        sys_content = system_msg.get("content", "")
        if isinstance(sys_content, str):
            acc_tokens += count_tokens(sys_content)
        # If system alone exceeds cap, return only system
        if acc_tokens >= token_cap:
            return [system_msg]

    # Walk from the end, prepend messages that fit
    for m in reversed(other_msgs):
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        need = count_tokens(content)
        if acc_tokens + need > token_cap:
            break
        kept.insert(0, m)
        acc_tokens += need

    if system_msg:
        result = [system_msg] + kept
    else:
        result = kept
    logger.debug(f"Trimmed messages from {len(messages)} to {len(result)} to fit {token_cap} token cap")
    return result

def _trim_messages_by_char_cap(messages: list[dict], cap: int) -> list[dict]:
    """Removes messages from the beginning (except system message) to fit within character cap."""
    if cap <= 0:
        return []
    if not messages:
        return []
    
    # Check if first message is a system message
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        other_msgs = messages[1:]
    else:
        # No system message in the persisted messages (it's added dynamically in llm_next)
        system_msg = None
        other_msgs = messages
    
    # Calculate total characters starting from the end
    acc = 0
    kept: list[dict] = []
    
    # Account for system message length if present
    if system_msg:
        system_content = system_msg.get("content", "")
        if isinstance(system_content, str):
            acc = len(system_content)
    
    # Iterate from the end, keeping messages that fit
    for m in reversed(other_msgs):
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        ln = len(content)
        if acc + ln > cap:
            break
        kept.insert(0, m)
        acc += ln
    
    # Include system message at the beginning if it exists
    if system_msg:
        result = [system_msg] + kept
    else:
        result = kept
    
    logger.debug(f"Trimmed messages from {len(messages)} to {len(result)} to fit {cap} char cap")
    return result

def _get_character_voice_and_ref(game: Game, name: str) -> tuple[Optional[str], Optional[str]]:
    ch = game.characters.filter(name=name).only("tts_voice", "reference_image_url").first()
    if not ch:
        return None, None
    return ch.tts_voice, ch.reference_image_url

def _should_generate_image(next_turn_index: int, every: int = 6, first: bool = True) -> bool:
    if every == 0:
        return False  # No images for plans without visuals
    if first and next_turn_index == 0:
        return True
    return every > 0 and ((next_turn_index + 1) % every == 0)

def _persist_remote(temp_url: str, kind: str) -> str:
    # Use the secure version from security module
    from user.security import secure_persist_remote
    return secure_persist_remote(temp_url, kind)

async def _persist_remote_async(temp_url: str, kind: str) -> str:
    try:
        from user.security import secure_persist_remote_async
        return await secure_persist_remote_async(temp_url, kind)
    except Exception:
        return await asyncio.to_thread(_persist_remote, temp_url, kind)

def _delete_media_url(url: Optional[str]) -> None:
    # Use the secure version from security module
    from user.security import secure_delete_media_url
    secure_delete_media_url(url)

def _is_http_url(url: Optional[str]) -> bool:
    u = (url or "").strip()
    return u.startswith("http://") or u.startswith("https://")

def _media_url_to_data_uri(url: Optional[str]) -> Optional[str]:
    """Convert local media URL (/media/...) to base64 data URI. Return None if not local or file missing."""
    if not url:
        return None
    u = str(url).strip()
    if u.startswith("data:"):
        return u
    if _is_http_url(u):
        return None
    media_url = getattr(settings, "MEDIA_URL", "/media/")
    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root or not u.startswith(media_url):
        return None
    path = (Path(media_root) / u[len(media_url):]).resolve()
    if not path.exists() or not path.is_file():
        return None
    try:
        import base64, mimetypes
        mime, _ = mimetypes.guess_type(path.name)
        mime = mime or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _tti_generate(
    image_description: str,
    reference_image_url: Optional[str] = None,
    image_reference_url: Optional[str] = None,
    aspect_ratio: str = "1:1",
    image_type: str = "general",
    visual_style: Optional[str] = None
) -> str:
    """
    Base TTI generation function that handles all cases.
    
    Args:
        image_description: The prompt for image generation
        reference_image_url: Optional character reference image URL
        aspect_ratio: Aspect ratio for the image (default: "1:1")
        image_type: Type of image being generated (for logging)
        visual_style: Visual style from game settings for enhancement
    
    Returns:
        CDN URL of the generated image
    """
    # Apply visual style enhancement if provided
    if visual_style:
        image_description = enhance_visual_prompt(image_description, visual_style)
        logger.debug(f"Applied {visual_style} enhancement to {image_type} image")
    logger.debug(f"Generating {image_type} image with character_reference={bool(reference_image_url)} image_reference={bool(image_reference_url)}")

    # Provider selection:
    # - If character reference provided → use Replicate luma/photon-flash (supports refs)
    # - Else → use fal.ai Flux Schnell; fallback to Replicate if fal fails

    def _call_replicate() -> str:
        inp = {
            "prompt": image_description,
            "aspect_ratio": aspect_ratio,
        }
        if reference_image_url:
            # Prefer data URI for local media; else pass http(s) URL as-is
            ref_data = _media_url_to_data_uri(reference_image_url)
            inp["character_reference"] = ref_data or (reference_image_url if _is_http_url(reference_image_url) else None)
        if image_reference_url:
            img_ref_data = _media_url_to_data_uri(image_reference_url)
            inp["image_reference"] = img_ref_data or (image_reference_url if _is_http_url(image_reference_url) else None)

        out = _replicate_api_call("luma/photon-flash", input_data=inp)
        item = out[0] if isinstance(out, list) and out else out
        temp_url = (
            (callable(getattr(item, "url", None)) and item.url())
            or getattr(item, "url", None)
            or (str(item) if isinstance(item, (str, bytes)) else None)
        )
        if not temp_url:
            raise ValueError("TTI output did not contain a URL.")
        return str(temp_url)

    def _call_fal() -> str:
        # fal.ai Flux Schnell; map aspect ratio to width/height
        def _ratio_to_hw(r: str) -> tuple[int, int]:
            mapping = {
                "1:1": (1024, 1024),
                "16:9": (1280, 720),
                "9:16": (720, 1280),
                "4:3": (1024, 768),
                "3:4": (768, 1024),
                "3:2": (1200, 800),
                "2:3": (800, 1200),
            }
            return mapping.get((r or "").strip(), (1024, 1024))

        width, height = _ratio_to_hw(aspect_ratio)
        args = {
            "prompt": image_description,
            "num_images": 1,
            "width": width,
            "height": height,
        }
        model_id = os.getenv("FAL_TTI_MODEL", "fal-ai/flux/schnell")
        out = _fal_api_call(model_id, arguments=args, timeout=60)
        temp_url = None
        try:
            if isinstance(out, dict):
                imgs = out.get("images") or out.get("output") or out.get("data")
                if isinstance(imgs, list) and imgs:
                    first = imgs[0]
                    if isinstance(first, dict):
                        temp_url = first.get("url") or first.get("image_url")
                    elif isinstance(first, str):
                        temp_url = first
                elif isinstance(out.get("image"), str):
                    temp_url = out.get("image")
            elif isinstance(out, (list, tuple)) and out:
                first = out[0]
                if isinstance(first, dict):
                    temp_url = first.get("url") or first.get("image_url")
                elif isinstance(first, str):
                    temp_url = first
            if not temp_url:
                temp_url = getattr(out, "url", None) or (callable(getattr(out, "url", None)) and out.url()) or None
        except Exception:
            pass
        if not temp_url:
            raise ValueError("fal.ai flux-schnell did not return an image URL")
        return str(temp_url)

    if reference_image_url:
        # Replicate path requires token
        token = os.getenv("REPLICATE_API_TOKEN", "").strip()
        if not token:
            logger.error("REPLICATE_API_TOKEN is not configured for reference-based TTI")
            # Dev placeholder output
            img_dir = Path(settings.MEDIA_ROOT) / "image"
            img_dir.mkdir(parents=True, exist_ok=True)
            ph = img_dir / f"{uuid.uuid4().hex}.txt"
            meta = [f"CHAR_REF={reference_image_url}"]
            if image_reference_url:
                meta.append(f"IMG_REF={image_reference_url}")
            ph.write_text(f"[{" ".join(meta)}]\n{image_description}", encoding="utf-8")
            rel = ph.relative_to(settings.MEDIA_ROOT).as_posix()
            return f"{settings.MEDIA_URL}{rel}"
        temp_url = _with_smart_retry(_call_replicate, max_attempts=5, operation_type="tti_generate_replicate")
    else:
        # Prefer fal.ai when no character reference
        fal_key = os.getenv("FAL_KEY", "").strip() or os.getenv("FAL_API_KEY", "").strip()
        temp_url = None
        if fal_key:
            try:
                temp_url = _with_smart_retry(_call_fal, max_attempts=5, operation_type="tti_generate_fal")
            except Exception as e:
                logger.warning(f"fal.ai flux-schnell failed, falling back to replicate: {e}")
        else:
            logger.info("FAL_KEY not configured; using replicate fallback")
        if not temp_url:
            # Fallback to replicate without references
            token = os.getenv("REPLICATE_API_TOKEN", "").strip()
            if not token:
                logger.error("REPLICATE_API_TOKEN is not configured (fal failed or not configured)")
                img_dir = Path(settings.MEDIA_ROOT) / "image"
                img_dir.mkdir(parents=True, exist_ok=True)
                ph = img_dir / f"{uuid.uuid4().hex}.txt"
                ph.write_text(image_description, encoding="utf-8")
                rel = ph.relative_to(settings.MEDIA_ROOT).as_posix()
                return f"{settings.MEDIA_URL}{rel}"
            temp_url = _with_smart_retry(_call_replicate, max_attempts=5, operation_type="tti_generate_replicate_fallback")

    # Persist asynchronously via a fresh event loop in this thread
    result = asyncio.run(_persist_remote_async(str(temp_url), "image"))
    logger.debug(f"{image_type} image generated and saved: {result}")
    return result


def tti_generate_char_ref(image_description: str, visual_style: Optional[str] = None) -> str:
    """Karakter referansı: luma/photon-flash (always)."""
    return _tti_generate(image_description, None, None, "1:1", "character reference", visual_style)


def tti_generate_turn_image(image_description: str, reference_image_url: str | None = None, image_reference_url: str | None = None, visual_style: Optional[str] = None) -> str:
    """Oyun içi görsel: luma/photon-flash; karakter ve önceki görsel referansları desteklenir."""
    return _tti_generate(image_description, reference_image_url, image_reference_url, "1:1", "turn image", visual_style)

def _batch_tti_with(
    func: Callable[[str], str],
    prompts: list[str],
    max_workers: int = 6,
) -> list[str]:
    """
    Batch TTI generation with parallel processing and proper error handling.
    Uses circuit breakers for each API call automatically.
    """
    logger.info(f"Starting batch TTI generation for {len(prompts)} images with {max_workers} workers")
    urls: list[Optional[str]] = [None] * len(prompts)
    workers = min(max_workers, max(1, len(prompts)))
    
    # Track statistics
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(func, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                urls[i] = fut.result(timeout=60)
                successful += 1
                logger.debug(f"Successfully generated image {i+1}/{len(prompts)}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to generate image {i+1}/{len(prompts)}: {e}")
                urls[i] = None
    
    logger.info(f"Batch TTI completed: {successful} successful, {failed} failed out of {len(prompts)}")
    
    # Filter out None values but ensure we have at least some results
    valid_urls = [u for u in urls if u]
    if not valid_urls:
        raise ValueError("All image generation attempts failed")
    
    # Return valid URLs, maintaining order where possible
    return valid_urls  # type: ignore


async def a_generate_characters(
    *,
    game_id: str,
    character_count: int,
    names_by_gender: dict[str, Sequence[str]],
    voices_by_gender: dict[str, Sequence[str]],
    balanced_gender: bool = True,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> list[GameCharacter]:
    # Defer to existing sync implementation in a worker thread for now
    return await asyncio.to_thread(
        generate_characters,
        game_id=game_id,
        character_count=character_count,
        tti_generate=tti_generate,  # signature requires it, unused internally
        names_by_gender=names_by_gender,
        voices_by_gender=voices_by_gender,
        balanced_gender=balanced_gender,
        overwrite=overwrite,
        max_workers=max_workers,
    )

def process_turn_job(
    turn_id: str,
    *,
    llm_next: LLMNextFn,
    tts_narration: TTSNarrationFn,
    tts_character: TTSCharacterFn,
    tti_generate: TTIFn,
    messages_char_cap: int = 50_000,
    image_every: int = 6,
    chapter_advance_every: int = 100,
    sound_effects: Optional[Sequence[dict]] = None,
) -> GameTurn:
    """
    PENDING turn'ü işler: LLM → TTS (+6'da 1 TTI) → DONE/FAILED.
    Uzun işlemler transaction DIŞINDA yapılır.
    """
    logger.info(f"Processing turn {turn_id}")
    # ---- 1) Snapshot + PROCESSING'e çek ----
    with transaction.atomic():
        turn = GameTurn.objects.select_for_update().select_related("chapter__game").get(pk=turn_id)
        if turn.status not in (TurnStatus.PENDING, TurnStatus.FAILED):
            logger.debug(f"Turn {turn_id} already processed with status {turn.status}")
            return turn  # zaten işlenmiş/işleniyor

        chapter = turn.chapter
        game = chapter.game
        story = game.story  # yoksa DoesNotExist
        
        # Determine chapter_advance_every based on story_length
        if game.story_length == "short":
            chapter_advance_every = 40
        else:  # standard (default)
            chapter_advance_every = 60
        
        # Determine image_every based on user's subscription plan
        user = game.user
        if user.subscription_plan == User.SubscriptionPlan.PRO:
            image_every = 1  # Every 3 turns for Pro
        elif user.subscription_plan == User.SubscriptionPlan.PLUS:
            image_every = 2  # Every 6 turns for Plus
        else:
            image_every = 0  # No visuals for other plans

        # LLM inputlarına lazım olacak snapshot
        idx = story.ongoing_chapter_index
        if idx != chapter.index:
            logger.warning("Story ongoing_chapter_index=%s != chapter.index=%s (turn_id=%s)", idx, chapter.index, turn_id)

        msgs = list(story.messages or [])
        if turn.user_text:
            msgs.append({"role": "user", "content": turn.user_text})

        do_image = _should_generate_image(turn.index, image_every, first=True)
        user_id_str = str(game.user_id)

        # status → PROCESSING
        turn.status = TurnStatus.PROCESSING
        turn.attempts += 1
        turn.started_at = timezone.now()
        turn.error_message = ""
        turn.save(update_fields=["status", "attempts", "started_at", "error_message"])

    tts_url = None
    tts_duration = 0.0
    image_url = None
    audio_type = "narration"
    try:
        # ---- 2) İlk tur intro mu? ----
        is_intro = (turn.index == 0) and not (story.messages or [])
        # Check plan restriction for TTS and image
        plan = game.user.subscription_plan
        # Lite should allow TTS like Plus/Pro; only FREE is restricted for TTS
        is_restricted = plan == User.SubscriptionPlan.FREE
        if is_intro:
            logger.debug(f"Processing intro turn for game {game.id}")
            scene_text = (story.introductory_text or "").strip()
            if not scene_text:
                raise ValidationError("World description is empty.")
            who = "narrator"
            choices = []
            sel_sfx = ""
            voice_desc = None
            # For FREE and LITE plans, skip the first-turn visual structured prompt entirely
            if plan in (User.SubscriptionPlan.FREE, User.SubscriptionPlan.LITE):
                img_desc = ""
            else:
                from user.prompts import llm_first_turn_visual
                img_desc = llm_first_turn_visual(
                    world_description=scene_text,
                    genre=game.genre_name,
                    main_character=game.main_character_name,
                    use_free_models=False,
                    language=getattr(game, 'language_code', 'en'),
                )
            if is_restricted:
                tts_url, tts_duration = None, 0.0
                audio_type = "narration"
                do_image = False
            else:
                tts_url, tts_duration = tts_narration(scene_text, "narrator", user_id_str)
                audio_type = "narration"
                try:
                    if plan in (User.SubscriptionPlan.PLUS, User.SubscriptionPlan.PRO):
                        do_image = True
                    else:
                        do_image = False
                except Exception:
                    do_image = False
        else:
            # ---- 2b) LLM normal akış ----
            out = llm_next(msgs, game, chapter, story)
            scene_text = out.scene.strip()
            if not scene_text:
                raise ValidationError("LLM returned empty scene.")
            if not out.image_description or not out.image_description.strip():
                raise ValidationError("image_description is required.")
            who = out.voice_name.strip().lower()
            img_desc = out.image_description.strip()
            choices = list(out.choices or [])
            sel_sfx = (out.selected_sound_effect or "").strip()
            if is_restricted:
                tts_url, tts_duration = None, 0.0
                audio_type = "narration"
            else:
                # ---- 3) TTS ----
                if who in ("narrator", "random"):
                    tts_url, tts_duration = tts_narration(scene_text, who, user_id_str, game)
                    audio_type = "narration"
                else:
                    char_voice, _ref = _get_character_voice_and_ref(game, who)
                    if not char_voice:
                        tts_url, tts_duration = tts_narration(scene_text, "narrator", user_id_str, game)
                        audio_type = "narration"
                    else:
                        # Respect character voices only for English; otherwise use default language voice
                        if getattr(game, 'language_code', 'en').lower() == 'en':
                            tts_url, tts_duration = tts_character(scene_text, char_voice)
                            audio_type = "character"
                        else:
                            tts_url, tts_duration = tts_narration(scene_text, "narrator", user_id_str, game)
                            audio_type = "narration"
            # ---- 4) TTI (first actionable turn or 6'da bir) ----
            if plan in (User.SubscriptionPlan.PLUS, User.SubscriptionPlan.PRO):
                if turn.index == 1:
                    do_image = True
                else:
                    do_image = _should_generate_image(turn.index, image_every)
            else:
                # No visuals for Lite/Free
                do_image = False
        # ---- 4b) Generate image if needed ----
        if do_image and img_desc:
            _, ref_url = _get_character_voice_and_ref(game, who)
            image_url = tti_generate_turn_image(img_desc, ref_url, None, game.visual_style)
        elif is_restricted:
            image_url = None

        # ---- 5) Kısa transaction: sonuçları yaz ----
        with transaction.atomic():
            turn = GameTurn.objects.select_for_update().select_related("chapter__game").get(pk=turn_id)
            story = turn.chapter.game.story  # taze oku

            # sound effect doğrula (varsa) ve FREE planda temizle
            if 'is_restricted' in locals() and is_restricted:
                selected_sound_effect = ""
            else:
                if sound_effects and sel_sfx:
                    valid = any(s.get("name") == sel_sfx for s in sound_effects)
                    selected_sound_effect = sel_sfx if valid else ""
                else:
                    selected_sound_effect = sel_sfx

            # Map from new field names to database field names
            turn.new_lines = scene_text  # scene -> new_lines (database field)
            turn.choices = choices
            turn.generated_tts_url = tts_url or ""
            turn.generated_visual_url = image_url or ""
            turn.selected_sound_effect = selected_sound_effect
            turn.audio_duration_seconds = Decimal(str(tts_duration))
            turn.status = TurnStatus.DONE
            turn.completed_at = timezone.now()
            turn.save(update_fields=[
                "new_lines","choices","generated_tts_url","generated_visual_url",
                "selected_sound_effect","audio_duration_seconds","status","completed_at"
            ])
            
            # Track usage
            user = game.user
            from user.models import User as _U
            if user.subscription_plan == _U.SubscriptionPlan.FREE:
                # For FREE plan, always deduct a fixed 30 seconds per turn
                fixed_seconds = Decimal("30.00")
                UsageTracking.objects.create(
                    user=user,
                    game_turn=turn,
                    audio_duration_seconds=fixed_seconds,
                    text_characters=len(scene_text),
                    audio_type="narration",
                )
                from django.db.models import F
                User.objects.filter(pk=user.pk).update(
                    total_audio_seconds_generated=F('total_audio_seconds_generated') + fixed_seconds
                )
            elif tts_url and tts_duration > 0:
                UsageTracking.objects.create(
                    user=user,
                    game_turn=turn,
                    audio_duration_seconds=Decimal(str(tts_duration)),
                    text_characters=len(scene_text),
                    audio_type=audio_type
                )
                
                # Update user's total audio seconds
                from django.db.models import F
                User.objects.filter(pk=user.pk).update(
                    total_audio_seconds_generated=F('total_audio_seconds_generated') + Decimal(str(tts_duration))
                )

            # messages güncelle (token cap)
            msgs2 = list(story.messages or [])
            if turn.user_text:
                msgs2.append({"role": "user", "content": turn.user_text})
            msgs2.append({"role": "assistant", "content": scene_text})
            # Use 8000-token cap regardless of char cap value
            story.messages = _trim_messages_by_token_cap(msgs2, 8000)
            story.save(update_fields=["messages"])

            # chapter ilerletme
            if chapter_advance_every > 0 and ((turn.index + 1) % chapter_advance_every == 0):
                logger.info(f"Advancing to next chapter for game {game.id} after turn {turn.index}")
                
                # Create story summary using helper function
                try:
                    story.realized_story = _create_story_summary(
                        messages=list(story.messages or []),
                        current_summary=story.realized_story or "",
                        game=game,
                        chapter=chapter
                    )
                except Exception as summarization_error:
                    logger.error(f"Failed to create story summary during chapter advancement: {summarization_error}")
                    # If we can't summarize, we cannot safely advance the chapter
                    # This preserves the current context instead of breaking the story
                    raise ValueError(f"Cannot advance chapter: story summarization failed - {summarization_error}")
                
                # Continue with chapter advancement
                story.ongoing_chapter_index = turn.chapter.index + 1
                story.messages = []  # context reset
                story.save(update_fields=["realized_story", "ongoing_chapter_index", "messages"])

        logger.info(f"Turn {turn_id} processed successfully")
        return turn

    except Exception as e:
        logger.exception("process_turn_job failed turn_id=%s", turn_id)
        # medya temizlik
        _delete_media_url(tts_url)
        _delete_media_url(image_url)
        
        # Check if we should retry
        with transaction.atomic():
            turn = GameTurn.objects.select_for_update().get(pk=turn_id)
            max_attempts = 5  # Increased from 3 to 5 attempts
            
            if turn.attempts < max_attempts:
                # Reset to PENDING for automatic retry
                turn.status = TurnStatus.PENDING
                turn.error_message = f"Attempt {turn.attempts} failed: {str(e)[:500]}. Retrying..."
                turn.save(update_fields=["status", "error_message"])
                
                # Enhanced exponential backoff with jitter
                from user.tasks import process_turn_task
                import random
                base_delay = 2 ** (turn.attempts - 1) * 3  # 3s, 6s, 12s, 24s, 48s
                jitter = random.uniform(0.8, 1.2)  # ±20% jitter to avoid thundering herd
                retry_delay = min(120, int(base_delay * jitter))  # Max 2 minutes
                
                logger.info(f"Scheduling retry for turn {turn_id} in {retry_delay}s (attempt {turn.attempts}/{max_attempts})")
                process_turn_task.apply_async(args=[str(turn_id)], countdown=retry_delay)
            else:
                # Max attempts reached, mark as FAILED with detailed error tracking
                turn.status = TurnStatus.FAILED
                turn.error_message = f"Failed after {max_attempts} attempts. Final error: {str(e)[:800]}"
                turn.completed_at = timezone.now()
                turn.save(update_fields=["status","error_message","completed_at"])
                logger.error(f"Turn {turn_id} failed after {max_attempts} attempts with final error: {str(e)}")
                
                # Log critical failure for monitoring
                logger.critical(f"Turn processing completely failed - turn_id: {turn_id}, game_id: {turn.chapter.game_id}, user_id: {turn.chapter.game.user_id}")
        
        return turn


# ==============================
# TTS / TTI implementations
# ==============================



_KOKORO_MODEL = "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
_KOKORO_ALL_VOICES = [
    "af_heart","af_alloy","af_aoede","af_bella","af_jessica","af_kore","af_nicole",
    "af_nova","af_river","af_sarah","af_sky","am_adam","am_echo","am_eric",
    "am_fenrir","am_liam","am_michael","am_onyx","am_puck","am_santa",
]

def _get_available_random_voices(game: Game) -> list[str]:
    """Get available kokoro voices that are not assigned to any character in the game."""
    # Get all character voices in this game
    assigned_voices = set(game.characters.values_list('tts_voice', flat=True))
    
    # Filter out assigned voices from the available pool
    available_voices = [voice for voice in _KOKORO_ALL_VOICES if voice not in assigned_voices]
    
    # If no voices are available (unlikely), fall back to the full pool
    if not available_voices:
        available_voices = _KOKORO_ALL_VOICES
    
    return available_voices

def _get_character_voice_and_ref(game: Game, name: str) -> tuple[Optional[str], Optional[str]]:
    """Get character's assigned voice and reference image by name."""
    ch = game.characters.filter(name=name).only("tts_voice", "reference_image_url").first()
    if not ch:
        return None, None
    return ch.tts_voice, ch.reference_image_url

def _get_user_narrator_voice(user_id: str) -> str:
    # TODO: kullanıcı tercihlerinden çek
    return "bm_lewis"

def tts_narration(text: str, who: str, user_id: str, game: Optional[Game] = None) -> tuple[str, float]:
    """Language-aware TTS narration via fal.ai Kokoro. English keeps narrator/random voice logic; other locales use fixed voices.
    FREE plan: do not generate audio; return empty URL and 0s.
    """
    logger.debug(f"Generating TTS narration for {who}")
    # Disable Kokoro TTS for FREE plan
    try:
        user = User.objects.get(pk=user_id)
        if user.subscription_plan == User.SubscriptionPlan.FREE:
            return "", 0.0
    except Exception:
        pass
    language = (getattr(game, 'language_code', 'en') if game else 'en')
    language = (language or 'en').lower()
    model_map = {
        'en': 'fal-ai/kokoro/american-english',
        'fr': 'fal-ai/kokoro/french',
        'es': 'fal-ai/kokoro/spanish',
        'pt-br': 'fal-ai/kokoro/brazilian-portuguese',
    }
    voice_map = {
        'fr': 'ff_siwis',
        'es': 'em_alex',
        'pt-br': 'pf_dora',
    }

    if language == 'en':
        # Preserve existing narrator/random selection for English only
        if who == "narrator":
            voice = _get_user_narrator_voice(user_id)
        elif who == "random":
            import random
            if game:
                available_voices = _get_available_random_voices(game)
                voice = random.choice(available_voices)
            else:
                logger.error("No game provided for random voice selection")
                voice = _get_user_narrator_voice(user_id)
    else:
        # Non-English: fixed voice
        voice = voice_map.get(language, voice_map.get('es', 'em_alex'))

    def _call_fal() -> str:
        args = {"prompt": text, "voice": voice}
        out = _fal_api_call(model_map.get(language, model_map['en']), arguments=args, timeout=60)
        # Try to normalize URLs from fal response
        temp_url = None
        if isinstance(out, dict):
            temp_url = out.get('audio_url') or out.get('url') or out.get('output')
            if isinstance(temp_url, (list, tuple)) and temp_url:
                temp_url = temp_url[0]
        if not temp_url:
            temp_url = getattr(out, 'url', None) or (callable(getattr(out, 'url', None)) and out.url()) or None
        if not temp_url:
            raise ValueError("fal.ai Kokoro did not return an audio URL")
        return str(temp_url)

    temp_url = _with_smart_retry(_call_fal, max_attempts=5, operation_type="tts_narration")
    # Prefer async persist if available
    try:
        from user.security import secure_persist_remote_async
        cdn_url = asyncio.run(secure_persist_remote_async(temp_url, "audio"))
    except Exception:
        cdn_url = _persist_remote(temp_url, "audio")
    
    # Calculate actual duration from the saved audio file
    duration = 0.0
    try:
        # Extract the file path from the CDN URL
        from pathlib import Path
        import wave
        from mutagen.mp3 import MP3
        from mutagen.oggvorbis import OggVorbis
        
        # cdn_url format: /media/audio/filename.wav
        if cdn_url.startswith(settings.MEDIA_URL):
            rel_path = cdn_url[len(settings.MEDIA_URL):]
            file_path = Path(settings.MEDIA_ROOT) / rel_path
            
            if file_path.exists():
                suffix = file_path.suffix.lower()
                if suffix == '.wav':
                    with wave.open(str(file_path), 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration = frames / float(rate)
                        logger.debug(f"Calculated WAV duration: {duration:.2f} seconds")
                elif suffix == '.mp3':
                    audio = MP3(str(file_path))
                    duration = audio.info.length
                    logger.debug(f"Calculated MP3 duration: {duration:.2f} seconds")
                elif suffix == '.ogg':
                    audio = OggVorbis(str(file_path))
                    duration = audio.info.length
                    logger.debug(f"Calculated OGG duration: {duration:.2f} seconds")
                else:
                    # Unknown format, use estimate
                    duration = len(text) / 12.5
                    logger.debug(f"Using estimated duration: {duration:.2f} seconds (unknown format: {suffix})")
            else:
                # File doesn't exist, use estimate
                duration = len(text) / 12.5
                logger.debug(f"Using estimated duration: {duration:.2f} seconds (file not found)")
        else:
            # Fallback to estimate
            duration = len(text) / 12.5
            logger.debug(f"Using estimated duration: {duration:.2f} seconds (URL format issue)")
    except ImportError:
        # mutagen not installed, fallback to WAV-only or estimate
        try:
            if file_path.exists() and file_path.suffix.lower() == '.wav':
                with wave.open(str(file_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
            else:
                duration = len(text) / 12.5
        except:
            duration = len(text) / 12.5
        logger.warning("mutagen not installed, limited audio format support")
    except Exception as e:
        logger.warning(f"Could not calculate audio duration: {e}. Using estimate.")
        duration = len(text) / 12.5
    
    return cdn_url, duration

def tts_character(text: str, voice_name: str) -> tuple[str, float]:
    """Character TTS for English via fal.ai Kokoro American-English with provided voice_name.
    Note: Character TTS is only called for non-FREE plans in the pipeline.
    """
    logger.debug(f"Generating TTS for character voice {voice_name}")

    def _call_fal() -> str:
        model = 'fal-ai/kokoro/american-english'
        args = {"prompt": text, "voice": voice_name}
        out = _fal_api_call(model, arguments=args, timeout=60)
        temp_url = None
        if isinstance(out, dict):
            temp_url = out.get('audio_url') or out.get('url') or out.get('output')
            if isinstance(temp_url, (list, tuple)) and temp_url:
                temp_url = temp_url[0]
        if not temp_url:
            temp_url = getattr(out, 'url', None) or (callable(getattr(out, 'url', None)) and out.url()) or None
        if not temp_url:
            raise ValueError("fal.ai Kokoro did not return an audio URL")
        return str(temp_url)

    temp_url = _with_smart_retry(_call_fal, max_attempts=5, operation_type="tts_character")
    
    # Persist using the same logic as tts_narration
    try:
        from user.security import secure_persist_remote_async
        cdn_url = asyncio.run(secure_persist_remote_async(temp_url, "audio"))
    except Exception:
        cdn_url = _persist_remote(temp_url, "audio")
    
    # Calculate actual duration from the saved audio file
    duration = 0.0
    try:
        # Extract the file path from the CDN URL
        from pathlib import Path
        if cdn_url.startswith(settings.MEDIA_URL):
            rel_path = cdn_url[len(settings.MEDIA_URL):]
            file_path = Path(settings.MEDIA_ROOT) / rel_path
            
            # Try to determine duration from audio file
            try:
                import wave
                with wave.open(str(file_path), 'rb') as wav_file:
                    duration = wav_file.getnframes() / float(wav_file.getframerate())
            except Exception:
                # Try with mutagen as fallback
                try:
                    from mutagen.wave import WAVE
                    audio_file = WAVE(str(file_path))
                    if audio_file.info:
                        duration = audio_file.info.length
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        
    if duration <= 0.0:
        # Fallback to estimated duration
        duration = len(text) / 12.5
    
    return cdn_url, duration



def tti_generate(image_description: str, reference_image_url: str | None = None, visual_style: Optional[str] = None) -> str:
    """
    TTI generation: luma/photon-flash (always). Returns a persistent CDN URL.
    """
    return _tti_generate(image_description, reference_image_url, None, "1:1", "general", visual_style)


# ==============================
# Services
# ==============================

def generate_story(
    *,
    game_id: str,
    llm_story: LLMStoryFn,
    extra_requests: str = "",
    overwrite: bool = False,
    names_by_gender: Optional[dict[str, list[str]]] = None,
) -> GameStory:
    logger.info(f"Generating story for game {game_id}")
    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        if GameStory.objects.filter(game=game).exists() and not overwrite:
            raise ValidationError("Story already exists. Set overwrite=True to replace.")

    # Generate character names list - select 8 random names from JSON
    if names_by_gender:
        import random
        # Get 4 random male names and 4 random female names from the JSON lists
        male_list = names_by_gender.get("male", [])
        female_list = names_by_gender.get("female", [])
        
        # Select 4 random names from each gender
        male_names = random.sample(male_list, min(4, len(male_list)))
        female_names = random.sample(female_list, min(4, len(female_list)))
        all_names = male_names + female_names
        
        # Shuffle all names together
        random.shuffle(all_names)
        characters = f"{game.main_character_name}, " + ", ".join(all_names)
    else:
        # This should never happen since we always load from JSON
        raise ValueError("Names JSON not loaded")
    
    # Use free LLM model sequence for FREE plan
    from user.models import User as _U
    use_free_models = (game.user.subscription_plan == _U.SubscriptionPlan.FREE)
    out = llm_story(
        game.genre_name,
        game.main_character_name,
        game.difficulty,
        extra_requests,
        characters,
        use_free_models=use_free_models,
        language=getattr(game, 'language_code', 'en'),
    )
    planned = (out.get("story") or "").strip()
    world = (out.get("introductory_text") or "").strip()
    if not planned or not world:
        # Provide actionable details for monitoring and debugging
        planned_len = len(planned)
        world_len = len(world)
        keys = sorted(list(out.keys())) if isinstance(out, dict) else []
        detail = (
            f"LLM story output is invalid: story_len={planned_len}, intro_len={world_len}, keys={keys}"
        )
        raise ValidationError(detail)

    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        story, created = GameStory.objects.get_or_create(
            game=game,
            defaults=dict(
                story=planned,
                introductory_text=world,
                realized_story="",
                messages=[],
                ongoing_chapter_index=0,
            ),
        )
        if not created and overwrite:
            story.story = planned
            story.introductory_text = world
            story.messages = []
            story.realized_story = ""
            story.ongoing_chapter_index = 0
            story.save(update_fields=[
                "story", "introductory_text",
                "messages", "realized_story", "ongoing_chapter_index"
            ])
        logger.info(f"Story generated for game {game_id}")
        return story


def generate_characters(
    *,
    game_id: str,
    character_count: int,
    tti_generate: TTIFn,  # KULLANILMAYACAK; imza korunuyor
    names_by_gender: dict[str, Sequence[str]],
    voices_by_gender: dict[str, Sequence[str]],
    balanced_gender: bool = True,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> list[GameCharacter]:
    logger.info(f"Generating {character_count} characters for game {game_id}")
    if character_count <= 0:
        raise ValidationError("character_count must be > 0")

    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        if not GameStory.objects.filter(game=game).exists():
            raise ValidationError("Generate story first.")
        # if existing and overwrite logic
        existing = list(game.characters.all())
        if existing and not overwrite:
            raise ValidationError("Characters already exist. Set overwrite=True to replace.")
        if existing and overwrite:
            game.characters.all().delete() 
    # cinsiyet dağılımı
    if balanced_gender:
        half = character_count // 2
        genders = ["male"] * half + ["female"] * (character_count - half)
    else:
        from random import choice
        genders = [choice(["male", "female"]) for _ in range(character_count)]

    # Build simplified prompts (TX dışında)
    import random, os, time
    start = time.monotonic()
    max_workers = max_workers or int(os.getenv("TTI_MAX_WORKERS", "6"))

    prompts: list[str] = []
    for g in genders:
        prompts.append(render_char_ref_prompt(CHAR_REF_TMPL, game.visual_style, g))

    # paralel TTI (fal.ai flux/schnell with photon-flash fallback)
    # For 8 characters, use optimal worker count (default 6, max 8)
    optimal_workers = min(max_workers, len(prompts))
    logger.info(f"Generating {len(prompts)} character images with {optimal_workers} parallel workers")
    
    # Create lambda to include visual_style for character references
    char_ref_with_style = lambda prompt: tti_generate_char_ref(prompt, game.visual_style)
    # Disable character TTI for FREE and LITE plans
    from user.models import User as _U
    if game.user.subscription_plan in {_U.SubscriptionPlan.FREE, _U.SubscriptionPlan.LITE}:
        cdn_urls = [""] * len(prompts)
    else:
        cdn_urls = _batch_tti_with(char_ref_with_style, prompts, max_workers=optimal_workers)

    # DB yaz (tek TX)
    created: list[GameCharacter] = []
    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        # Match genders with cdn_urls, handling potential length mismatch
        for g, img_url in zip(genders[:len(cdn_urls)], cdn_urls):
            # Get random name from the JSON list for this gender
            gender_names = names_by_gender.get(g, [])
            if not gender_names:
                raise ValueError(f"No names found for gender {g} in names.json" )
            name = random.choice(gender_names)
            
            # Get random voice from the JSON list for this gender
            gender_voices = voices_by_gender.get(g, [])
            if not gender_voices:
                raise ValueError(f"No voices found for gender {g} in voices.json")
            tts_voice = random.choice(gender_voices)
            ch = GameCharacter.objects.create(
                name=name,
                gender=g,
                reference_image_url=img_url or "",
                tts_voice=tts_voice,
            )
            game.characters.add(ch)
            created.append(ch)

    logger.info("generate_characters: %d images in %.1fs (workers=%d)",
                len(created), time.monotonic() - start, max_workers)
    return created


def plan_chapters(
    *,
    game_id: str,
    llm_plan_chapters: LLMPlanChaptersFn,
    overwrite: bool = False,
    start_index: int = 0,
) -> list[int]:
    logger.info(f"Planning chapters for game {game_id}")

    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        try:
            story = game.story
        except GameStory.DoesNotExist:
            raise ValidationError("Generate story first.")
        if game.characters.count() == 0:
            raise ValidationError("Generate characters first.")

        has_existing = GameChapter.objects.filter(game=game).exists()
        if has_existing and not overwrite:
            raise ValidationError("Chapters already exist. Use overwrite=True.")
        if has_existing and overwrite:
            GameChapter.objects.filter(game=game).delete()  # turns cascade

    # Use free LLM model sequence for FREE plan
    from user.models import User as _U
    use_free_models = (game.user.subscription_plan == _U.SubscriptionPlan.FREE)
    plans = llm_plan_chapters(story.story, use_free_models=use_free_models, language=getattr(game, 'language_code', 'en'))
    if not isinstance(plans, (list, tuple)):
        raise ValidationError(f"LLM returned invalid chapter plan type: {type(plans).__name__}")
    # Normalize and validate contents
    normalized_plans = [(p or "").strip() for p in plans]
    empty_indexes = [i for i, p in enumerate(normalized_plans) if not p]
    if empty_indexes:
        detail = (
            f"LLM returned invalid chapter plan list: non_empty={len(normalized_plans) - len(empty_indexes)} "
            f"total={len(normalized_plans)} empty_indexes={empty_indexes}"
        )
        raise ValidationError(detail)
    plans = normalized_plans

    with transaction.atomic():
        game = Game.objects.select_for_update().get(pk=game_id)
        objs = [GameChapter(game=game, index=i, planned_chapter_story=plan.strip())
                for i, plan in enumerate(plans, start=start_index)]
        GameChapter.objects.bulk_create(objs)

        story = game.story
        story.ongoing_chapter_index = start_index
        story.save(update_fields=["ongoing_chapter_index"])

    logger.info(f"Created {len(plans)} chapters for game {game_id}")
    return [c.index for c in GameChapter.objects.filter(game_id=game_id).order_by("index")]


@dataclass
class TurnResult:
    turn: GameTurn
    tts_url: Optional[str]
    image_url: Optional[str]

def reserve_turn(
    *,
    game_id: str,
    user_text: str | None = None,
    chapter_index: int | None = None,
    idempotency_key: str | None = None,
) -> GameTurn:
    logger.debug(f"Reserving turn for game {game_id}")
    # Avoid locking the Game row to reduce deadlock risk; lock only the target chapter
    with transaction.atomic():
        # Read story without row-lock; determine target chapter index
        story = GameStory.objects.select_related("game").get(game_id=game_id)
        # Block playing archived games at the service layer for safety
        if getattr(story.game, "is_archived", False):
            raise ValidationError("Game is archived and cannot be played.")
        idx = story.ongoing_chapter_index if chapter_index is None else chapter_index

        # Lock the chapter row deterministically
        chapter = GameChapter.objects.select_for_update().get(game_id=game_id, index=idx)

        # Idempotency fast-path before attempting insert
        idem = idempotency_key or _make_idempotency_key(str(story.game_id), idx, user_text)
        existing = GameTurn.objects.filter(idempotency_key=idem).first()
        if existing:
            logger.debug(f"Turn already exists with idempotency key {idem}")
            return existing

        # Retry on unique-index races when computing next index concurrently
        for _ in range(3):
            next_idx = _next_turn_index_locked(chapter)
            try:
                return GameTurn.objects.create(
                    chapter=chapter,
                    index=next_idx,
                    status=TurnStatus.PENDING,
                    reserved_at=timezone.now(),
                    user_text=user_text or "",
                    idempotency_key=idem,
                )
            except IntegrityError:
                continue

        raise ValidationError("Could not reserve next turn index (concurrent writers).")

def generate_turn(
    *,
    game_id: str,
    user_text: Optional[str],
    llm_next: LLMNextFn,
    tts_narration: TTSNarrationFn,
    tts_character: TTSCharacterFn,
    tti_generate: TTIFn,
    messages_char_cap: int = 50_000,
    image_every: int = 6,
    chapter_advance_every: int = 100,
    chapter_index: Optional[int] = None,
    sound_effects: Optional[Sequence[dict]] = None,
) -> TurnResult:
    logger.info(f"Generating turn for game {game_id}")
    t = reserve_turn(
        game_id=game_id,
        user_text=user_text,
        chapter_index=chapter_index,
    )
    t2 = process_turn_job(
        str(t.id),
        llm_next=llm_next,
        tts_narration=tts_narration,
        tts_character=tts_character,
        tti_generate=tti_generate,
        messages_char_cap=messages_char_cap,
        image_every=image_every,
        chapter_advance_every=chapter_advance_every,
        sound_effects=sound_effects,
    )
    return TurnResult(
        turn=t2,
        tts_url=t2.generated_tts_url or None,
        image_url=t2.generated_visual_url or None,
    )


# ==============================
# Async wrappers (non-blocking)
# ==============================

async def a_process_turn_job(
    turn_id: str,
    *,
    llm_next: LLMNextFn,
    tts_narration: TTSNarrationFn,
    tts_character: TTSCharacterFn,
    tti_generate: TTIFn,
    messages_char_cap: int = 50_000,
    image_every: int = 6,
    chapter_advance_every: int = 100,
    sound_effects: Optional[Sequence[dict]] = None,
) -> GameTurn:
    logger.info(f"[async] Processing turn {turn_id}")

    # 1) Initial snapshot and mark PROCESSING (short DB txn)
    async def _initial_snapshot() -> dict:
        def _sync():
            with transaction.atomic():
                t = GameTurn.objects.select_for_update().select_related("chapter__game", "chapter", "chapter__game__user").get(pk=turn_id)
                if t.status not in (TurnStatus.PENDING, TurnStatus.FAILED):
                    return {"already": True, "turn_id": str(t.id)}
                chapter = t.chapter
                game = chapter.game
                story = game.story
                msgs = list(story.messages or [])
                if t.user_text:
                    msgs.append({"role": "user", "content": t.user_text})
                # Set PROCESSING
                t.status = TurnStatus.PROCESSING
                t.attempts += 1
                t.started_at = timezone.now()
                t.error_message = ""
                t.save(update_fields=["status", "attempts", "started_at", "error_message"])
                # Determine plan-based image cadence
                plan = game.user.subscription_plan
                img_every = 3 if plan == User.SubscriptionPlan.PRO else 6 if plan == User.SubscriptionPlan.PLUS else 0
                # Determine chapter cadence by story length
                chap_every = 40 if game.story_length == "short" else 60
                return {
                    "already": False,
                    "turn_index": t.index,
                    "chapter_index": chapter.index,
                    "game_id": str(game.id),
                    "user_id": str(game.user_id),
                    "story_msgs": msgs,
                    "intro": (story.introductory_text or "").strip(),
                    "is_intro": (t.index == 0) and not (story.messages or []),
                    "image_every": img_every,
                    "chapter_every": chap_every,
                    "visual_style": game.visual_style,
                    "genre": game.genre_name,
                    "main_character": game.main_character_name,
                    "user_text": t.user_text or "",
                    "plan": plan,
                }
        return await asyncio.to_thread(_sync)

    snap = await _initial_snapshot()
    if snap.get("already"):
        # Return the turn as-is
        return await asyncio.to_thread(lambda: GameTurn.objects.get(pk=turn_id))

    # 2) LLM and media generation
    do_image = snap["is_intro"] or (True if snap["turn_index"] == 1 else False)
    image_every_eff = snap["image_every"]
    if not do_image and image_every_eff > 0:
        do_image = ((snap["turn_index"] + 1) % image_every_eff) == 0
    # Disable all TTI for FREE and LITE plans
    try:
        from user.models import User as _U
        if snap.get("plan") in {_U.SubscriptionPlan.FREE, _U.SubscriptionPlan.LITE}:
            do_image = False
    except Exception:
        pass

    if snap["is_intro"]:
        scene_text = snap["intro"]
        who = "narrator"
        choices: list[str] = []
        sel_sfx = ""
        voice_desc: Optional[str] = None
        # First-turn visual prompt
        def _first_visual_sync():
            from user.prompts import llm_first_turn_visual
            return llm_first_turn_visual(
                world_description=snap["intro"],
                genre=snap["genre"],
                main_character=snap["main_character"],
                use_free_models=(snap.get("plan") == User.SubscriptionPlan.FREE),
            )
        img_desc = await asyncio.to_thread(_first_visual_sync)
        # TTS narration (respect FREE: return empty)
        try:
            from user.models import User as _U
            if snap.get("plan") == _U.SubscriptionPlan.FREE:
                tts_url, tts_duration = "", 0.0
            else:
                tts_url, tts_duration = await a_tts_narration(scene_text, "narrator", snap["user_id"])
        except Exception:
            tts_url, tts_duration = await a_tts_narration(scene_text, "narrator", snap["user_id"])
        audio_type = "narration"
    else:
        # Call llm_next synchronously in thread (it may use sync SDKs internally)
        def _llm_next_sync():
            # Load fresh related instances for llm_next
            chapter = GameChapter.objects.select_related("game", "game__story").get(game_id=snap["game_id"], index=snap["chapter_index"])  # type: ignore
            game = chapter.game
            story = game.story
            return llm_next(list(snap["story_msgs"]), game, chapter, story)
        out = await asyncio.to_thread(_llm_next_sync)
        scene_text = out.scene.strip()
        img_desc = (out.image_description or "").strip()
        who = (out.voice_name or "narrator").strip().lower()
        choices = list(out.choices or [])
        sel_sfx = (out.selected_sound_effect or "").strip()
        # TTS selection (respect FREE plan)
        try:
            from user.models import User as _U
            is_free = (snap.get("plan") == _U.SubscriptionPlan.FREE)
        except Exception:
            is_free = False
        if is_free:
            tts_url, tts_duration = "", 0.0
            audio_type = "narration"
        else:
            if who in ("narrator", "random"):
                # Get game context for better random voice selection
                def _get_game():
                    return Game.objects.get(pk=snap["game_id"])  # type: ignore
                game_ctx = await asyncio.to_thread(_get_game)
                tts_url, tts_duration = await a_tts_narration(scene_text, who, snap["user_id"], game_ctx)
                audio_type = "narration"
            else:
                # fetch character voice
                def _get_voice_sync():
                    game = Game.objects.get(pk=snap["game_id"])  # type: ignore
                    return _get_character_voice_and_ref(game, who)
                char_voice, _ref = await asyncio.to_thread(_get_voice_sync)
                if not char_voice:
                    def _get_game():
                        return Game.objects.get(pk=snap["game_id"])  # type: ignore
                    game_ctx = await asyncio.to_thread(_get_game)
                    tts_url, tts_duration = await a_tts_narration(scene_text, "narrator", snap["user_id"], game_ctx)
                    audio_type = "narration"
                else:
                    tts_url, tts_duration = await a_tts_character(scene_text, char_voice)
                    audio_type = "character"

    # Image generation if needed
    image_url: Optional[str] = None
    if do_image and img_desc:
        # get character reference only; do not use previous image as reference
        async def _get_char_ref() -> Optional[str]:
            def _sync():
                game = Game.objects.get(pk=snap["game_id"])  # type: ignore
                _voice, ref_url = _get_character_voice_and_ref(game, who)
                return ref_url
            return await asyncio.to_thread(_sync)
        ref_url = await _get_char_ref()
        # TTI generation is disabled for FREE and LITE by do_image flag; only run when allowed
        image_url = await a_tti_generate(
            img_desc,
            reference_image_url=ref_url,
            image_reference_url=None,
            aspect_ratio="1:1",
            image_type="turn image" if not snap["is_intro"] else "intro image",
            visual_style=snap["visual_style"],
        )

    # 3) Final DB write (short txn)
    def _finalize_sync():
        with transaction.atomic():
            t = GameTurn.objects.select_for_update().select_related("chapter__game").get(pk=turn_id)
            story = t.chapter.game.story
            # Validate sfx and enforce plan gating (FREE: clear)
            try:
                from user.models import User as _U
                is_free = (snap.get("plan") == _U.SubscriptionPlan.FREE)
            except Exception:
                is_free = False
            if is_free:
                selected_sound_effect = ""
            else:
                if sound_effects and sel_sfx:
                    valid = any(s.get("name") == sel_sfx for s in (sound_effects or []))
                    selected_sound_effect = sel_sfx if valid else ""
                else:
                    selected_sound_effect = sel_sfx
            t.new_lines = scene_text
            t.choices = choices
            t.generated_tts_url = tts_url or ""
            t.generated_visual_url = image_url or ""
            t.selected_sound_effect = selected_sound_effect
            t.audio_duration_seconds = Decimal(str(tts_duration))
            t.status = TurnStatus.DONE
            t.completed_at = timezone.now()
            t.save(update_fields=[
                "new_lines","choices","generated_tts_url","generated_visual_url",
                "selected_sound_effect","audio_duration_seconds","status","completed_at"
            ])
            # Track usage
            user = t.chapter.game.user
            from user.models import User as _U
            if snap.get("plan") == _U.SubscriptionPlan.FREE:
                fixed_seconds = Decimal("30.00")
                UsageTracking.objects.create(
                    user=user,
                    game_turn=t,
                    audio_duration_seconds=fixed_seconds,
                    text_characters=len(scene_text),
                    audio_type="narration",
                )
                from django.db.models import F
                User.objects.filter(pk=user.pk).update(
                    total_audio_seconds_generated=F('total_audio_seconds_generated') + fixed_seconds
                )
            elif tts_url and tts_duration > 0:
                UsageTracking.objects.create(
                    user=user,
                    game_turn=t,
                    audio_duration_seconds=Decimal(str(tts_duration)),
                    text_characters=len(scene_text),
                    audio_type=audio_type,
                )
                from django.db.models import F
                User.objects.filter(pk=user.pk).update(
                    total_audio_seconds_generated=F('total_audio_seconds_generated') + Decimal(str(tts_duration))
                )
            # Update messages with token cap
            msgs2 = list(story.messages or [])
            if snap["user_text"]:
                msgs2.append({"role": "user", "content": snap["user_text"]})
            msgs2.append({"role": "assistant", "content": scene_text})
            story.messages = _trim_messages_by_token_cap(msgs2, 8000)
            story.save(update_fields=["messages"])
            # Chapter advancement
            if snap["chapter_every"] > 0 and ((t.index + 1) % snap["chapter_every"] == 0):
                try:
                    story.realized_story = _create_story_summary(
                        messages=list(story.messages or []),
                        current_summary=story.realized_story or "",
                        game=t.chapter.game,
                        chapter=t.chapter,
                    )
                except Exception as summarization_error:
                    raise ValueError(f"Cannot advance chapter: story summarization failed - {summarization_error}")
                story.ongoing_chapter_index = t.chapter.index + 1
                story.messages = []
                story.save(update_fields=["realized_story", "ongoing_chapter_index", "messages"])
            return t
    try:
        t2 = await asyncio.to_thread(_finalize_sync)
        logger.info(f"[async] Turn {turn_id} processed successfully")
        return t2
    except Exception as e:
        logger.exception("[async] process_turn_job failed turn_id=%s", turn_id)
        # Cleanup media if we produced any
        _delete_media_url(tts_url)
        _delete_media_url(image_url)
        # Mark for retry/failure similar to sync path
        def _mark_failed():
            with transaction.atomic():
                t = GameTurn.objects.select_for_update().get(pk=turn_id)
                max_attempts = 5
                if t.attempts < max_attempts:
                    t.status = TurnStatus.PENDING
                    t.error_message = f"Attempt {t.attempts} failed: {str(e)[:500]}. Retrying..."
                    t.save(update_fields=["status", "error_message"])
                    from user.tasks import process_turn_task
                    import random
                    base_delay = 2 ** (t.attempts - 1) * 3
                    jitter = random.uniform(0.8, 1.2)
                    retry_delay = min(120, int(base_delay * jitter))
                    process_turn_task.apply_async(args=[str(turn_id)], countdown=retry_delay)
                else:
                    t.status = TurnStatus.FAILED
                    t.error_message = f"Failed after {max_attempts} attempts. Final error: {str(e)[:800]}"
                    t.completed_at = timezone.now()
                    t.save(update_fields=["status","error_message","completed_at"])
            return t
        await asyncio.to_thread(_mark_failed)
        return await asyncio.to_thread(lambda: GameTurn.objects.get(pk=turn_id))


async def a_generate_turn(
    *,
    game_id: str,
    user_text: Optional[str],
    llm_next: LLMNextFn,
    tts_narration: TTSNarrationFn,
    tts_character: TTSCharacterFn,
    tti_generate: TTIFn,
    messages_char_cap: int = 50_000,
    image_every: int = 6,
    chapter_advance_every: int = 100,
    chapter_index: Optional[int] = None,
    sound_effects: Optional[Sequence[dict]] = None,
) -> TurnResult:
    # Reserve turn (DB offloaded)
    t = await a_reserve_turn(
        game_id=game_id,
        user_text=user_text,
        chapter_index=chapter_index,
    )
    # Process turn asynchronously
    t2 = await a_process_turn_job(
        str(t.id),
        llm_next=llm_next,
        tts_narration=tts_narration,
        tts_character=tts_character,
        tti_generate=tti_generate,
        messages_char_cap=messages_char_cap,
        image_every=image_every,
        chapter_advance_every=chapter_advance_every,
        sound_effects=sound_effects,
    )
    return TurnResult(turn=t2, tts_url=t2.generated_tts_url or None, image_url=t2.generated_visual_url or None)


async def a_reserve_turn(
    *,
    game_id: str,
    user_text: str | None = None,
    chapter_index: int | None = None,
    idempotency_key: str | None = None,
) -> GameTurn:
    return await asyncio.to_thread(
        reserve_turn,
        game_id=game_id,
        user_text=user_text,
        chapter_index=chapter_index,
        idempotency_key=idempotency_key,
    )


async def a_tts_narration(text: str, who: str, user_id: str, game: Optional[Game] = None) -> tuple[str, float]:
    logger.debug(f"[async] Generating TTS narration for {who}")
    # Voice selection mirrors sync logic
    # Disable Kokoro TTS for FREE plan
    try:
        u = await asyncio.to_thread(lambda: User.objects.get(pk=user_id))
        if u.subscription_plan == User.SubscriptionPlan.FREE:
            return "", 0.0
    except Exception:
        pass
    if who == "narrator":
        voice = _get_user_narrator_voice(user_id)
    elif who == "random":
        import random
        if game:
            available_voices = _get_available_random_voices(game)
            voice = random.choice(available_voices)
        else:
            # Fallback when no game context is available
            voice = random.choice(_KOKORO_ALL_VOICES)
    else:
        voice = "af_alloy"

    # Call Replicate via async path when possible
    async def _call() -> str:
        out = await a_replicate_api_call(_KOKORO_MODEL, {"text": text, "speed": 1.0, "voice": voice})
        item = out[0] if isinstance(out, list) and out else out
        temp_url = (
            (callable(getattr(item, "url", None)) and item.url())
            or getattr(item, "url", None)
            or (str(item) if isinstance(item, (str, bytes)) else None)
        )
        if not temp_url:
            raise ValueError("Kokoro TTS did not return a URL")
        return str(temp_url)

    try:
        temp_url = await _call()
    except Exception:
        # Fallback to thread if async fails
        return await asyncio.to_thread(tts_narration, text, who, user_id, game)

    # Persist asynchronously
    try:
        from user.security import secure_persist_remote_async
        cdn_url = await secure_persist_remote_async(temp_url, "audio")
    except Exception:
        cdn_url = await asyncio.to_thread(_persist_remote, temp_url, "audio")

    # Compute duration (use to_thread for wave/mutagen)
    async def _compute_duration() -> float:
        try:
            from pathlib import Path as _P
            import wave
            from mutagen.mp3 import MP3
            from mutagen.oggvorbis import OggVorbis
            if cdn_url.startswith(settings.MEDIA_URL):
                rel_path = cdn_url[len(settings.MEDIA_URL):]
                file_path = _P(settings.MEDIA_ROOT) / rel_path
                if file_path.exists():
                    suffix = file_path.suffix.lower()
                    if suffix == '.wav':
                        def _wav_len():
                            with wave.open(str(file_path), 'rb') as wav_file:
                                return wav_file.getnframes() / float(wav_file.getframerate())
                        return await asyncio.to_thread(_wav_len)
                    elif suffix == '.mp3':
                        def _mp3_len():
                            return MP3(str(file_path)).info.length
                        return await asyncio.to_thread(_mp3_len)
                    elif suffix == '.ogg':
                        def _ogg_len():
                            return OggVorbis(str(file_path)).info.length
                        return await asyncio.to_thread(_ogg_len)
        except Exception:
            pass
        return len(text) / 12.5

    duration = await _compute_duration()
    return cdn_url, duration


async def a_tts_character(text: str, voice_name: str) -> tuple[str, float]:
    logger.debug(f"[async] Generating TTS for character voice {voice_name}")

    # Use async Kokoro TTS call like in a_tts_narration
    async def _call() -> str:
        out = await a_replicate_api_call(_KOKORO_MODEL, {"text": text, "speed": 1.0, "voice": voice_name})
        item = out[0] if isinstance(out, list) and out else out
        temp_url = (
            (callable(getattr(item, "url", None)) and item.url())
            or getattr(item, "url", None)
            or (str(item) if isinstance(item, (str, bytes)) else None)
        )
        if not temp_url:
            raise ValueError("Kokoro TTS did not return a URL")
        return str(temp_url)

    try:
        temp_url = await _call()
    except Exception:
        # Fallback to thread if async fails
        return await asyncio.to_thread(tts_character, text, voice_name)

    # Persist asynchronously
    try:
        from user.security import secure_persist_remote_async
        cdn_url = await secure_persist_remote_async(temp_url, "audio")
    except Exception:
        cdn_url = await asyncio.to_thread(_persist_remote, temp_url, "audio")

    # Compute duration (use to_thread for wave/mutagen)
    async def _compute_duration():
        try:
            from pathlib import Path
            if cdn_url.startswith(settings.MEDIA_URL):
                rel_path = cdn_url[len(settings.MEDIA_URL):]
                file_path = Path(settings.MEDIA_ROOT) / rel_path
                
                # Try wave first
                try:
                    import wave
                    with wave.open(str(file_path), 'rb') as wav_file:
                        return wav_file.getnframes() / float(wav_file.getframerate())
                except Exception:
                    # Try mutagen as fallback
                    try:
                        from mutagen.wave import WAVE
                        audio_file = WAVE(str(file_path))
                        if audio_file.info:
                            return audio_file.info.length
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Fallback to estimated duration
        return len(text) / 12.5

    duration = await asyncio.to_thread(_compute_duration)
    return cdn_url, duration


async def a_tti_generate_char_ref(image_description: str, visual_style: Optional[str] = None) -> str:
    return await asyncio.to_thread(tti_generate_char_ref, image_description, visual_style)


async def a_tti_generate_turn_image(
    image_description: str,
    reference_image_url: str | None = None,
    image_reference_url: str | None = None,
    visual_style: Optional[str] = None,
) -> str:
    return await asyncio.to_thread(
        tti_generate_turn_image,
        image_description,
        reference_image_url,
        image_reference_url,
        visual_style,
    )


async def a_tti_generate(
    image_description: str,
    reference_image_url: Optional[str] = None,
    image_reference_url: Optional[str] = None,
    aspect_ratio: str = "1:1",
    image_type: str = "general",
    visual_style: Optional[str] = None,
) -> str:
    # Async wrapper for _tti_generate using thread to avoid blocking
    return await asyncio.to_thread(
        _tti_generate,
        image_description,
        reference_image_url,
        image_reference_url,
        aspect_ratio,
        image_type,
        visual_style,
    )


async def a_batch_tti_with(func: Callable[[str], 'asyncio.Future[str]'] | Callable[[str], str], prompts: list[str], max_concurrency: int = 6) -> list[str]:
    sem = asyncio.Semaphore(max(1, max_concurrency))
    results: list[Optional[str]] = [None] * len(prompts)

    async def runner(i: int, prompt: str):
        async with sem:
            try:
                maybe_coro = func(prompt)
                if asyncio.iscoroutine(maybe_coro):
                    results[i] = await maybe_coro  # type: ignore
                else:
                    results[i] = await asyncio.to_thread(func, prompt)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}/{len(prompts)}: {e}")
                results[i] = None

    await asyncio.gather(*(runner(i, p) for i, p in enumerate(prompts)))
    valid = [u for u in results if u]
    if not valid:
        raise ValueError("All image generation attempts failed")
    return valid  # type: ignore