from __future__ import annotations
import os
import asyncio
from typing import Optional, List
import logging
import concurrent.futures
import re

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from user.llm import cheap_llmPrompt, cheap_llmMessages, expensive_llmPrompt

# -------------------------------------------------
# Env + LLM clients
# -------------------------------------------------
load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Markdown stripping utility
# -------------------------------------------------
def strip_markdown(text: str | None) -> str:
    """Convert simple Markdown to plain text.

    - Removes code fences and inline code marks while preserving content
    - Converts links/images to their alt/text
    - Removes emphasis markers (*, _, **, __, ~~)
    - Strips heading markers (#), blockquotes (>), and list bullets (-, *, digits.)
    - Flattens tables by removing pipes and header separators
    - Normalizes whitespace
    """
    if not text:
        return ""

    t = str(text)

    # Code fences: keep inner content
    t = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?\n([\s\S]*?)```", r"\1", t)
    t = t.replace("```", "")

    # Inline code: `code` -> code
    t = re.sub(r"`([^`]*)`", r"\1", t)

    # Images: ![alt](url) -> alt
    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", t)

    # Links: [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)

    # Emphasis and strikethrough
    t = re.sub(r"(\*\*|__)(.*?)\1", r"\2", t)
    t = re.sub(r"(\*|_)(.*?)\1", r"\2", t)
    t = re.sub(r"~~(.*?)~~", r"\1", t)

    # Headings: remove leading #'s
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.MULTILINE)

    # Blockquotes: remove leading '>'
    t = re.sub(r"^\s*>\s?", "", t, flags=re.MULTILINE)

    # Lists: bullets and numbered
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)

    # Tables: remove pipes and header separators
    t = re.sub(r"^\s*\|\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s*\|\s*", " ", t)
    t = re.sub(r"^\s*:-{2,,}:?\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*-{3,}\s*$", "", t, flags=re.MULTILINE)

    # Collapse excessive whitespace
    t = re.sub(r"[\t\x0b\x0c\r]", " ", t)
    t = re.sub(r"\u00A0", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    return t.strip()

# -------------------------------------------------
# Helper to run async code safely from sync context
# -------------------------------------------------
def run_async_safe(coro_func, timeout=60):
    """Run async code safely from sync contexts, with reliable loop lifecycle.

    Uses asyncio.Runner() to ensure proper cleanup, avoiding "Event loop is closed"
    errors that can occur when transports attempt to schedule work during loop teardown.

    Args:
        coro_func: A callable that returns a new coroutine each call
        timeout: Timeout in seconds
    """
    import logging
    logger = logging.getLogger(__name__)

    def _run_with_runner() -> object:
        # Create a fresh coroutine instance
        coro = coro_func()
        if not asyncio.iscoroutine(coro):
            raise TypeError(f"coro_func must return a coroutine, got {type(coro)}")
        # Use asyncio.Runner for robust loop management (Python 3.11+)
        with asyncio.Runner() as runner:
            # Enforce timeout at the coroutine level to prevent indefinite hangs
            result = runner.run(asyncio.wait_for(coro, timeout))
            # Give the loop a final turn to finish anyio/httpx cleanup callbacks
            runner.run(asyncio.sleep(0))
            return result

    try:
        try:
            # If a loop is already running (e.g., inside ASGI), offload to a worker thread
            asyncio.get_running_loop()
            logger.info("Found running event loop, executing in a worker thread via Runner")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_with_runner)
                return future.result(timeout=timeout)
        except RuntimeError:
            # No loop running in this thread; run directly with Runner (timeout enforced inside)
            logger.info("No running event loop, executing directly via Runner")
            return _run_with_runner()
    except Exception as e:
        # Propagate timeouts immediately rather than retrying, to respect caller's limit
        if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
            logger.error(f"run_async_safe timed out after {timeout}s")
            raise
        logger.error(f"Error in run_async_safe: {type(e).__name__}: {str(e)}", exc_info=True)
        # Final fallback: new thread with Runner (isolated environment)
        logger.warning("Using fallback worker thread with Runner due to error")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_with_runner)
            return future.result(timeout=timeout)

# -------------------------------------------------
# STORY: Pydantic şemaları
# -------------------------------------------------
class GameStoryGenerationInput(BaseModel):
    genre: str
    player_name: str
    game_extra_requests: str = ""
    characters: str

class GameStoryGenerationOutput(BaseModel):
    introductory_text: str = Field(..., description="introductory text")
    story: str = Field(..., description="Story")

# -------------------------------------------------
# STORY: PromptTemplate + senkron LLM fonksiyonu
# -------------------------------------------------
story_generation_prompt = PromptTemplate.from_template(
    (
        "First, write an introductory text for the player.\n"
        "This introduction should be short (one paragraph) and immersive.\n"
        "Introduce {player_name}, their background, personality, and role.\n"
        "Describe the world they live in: its culture, history, dangers, and mysteries while not overwhelming the player with too much information.\n"
        "The introduction should make the player feel ready to start a long adventure. Avoid being cliche. Avoid spoiling the story.\n\n"

        "Then, write a very long and detailed 6-paragraph {genre} story summary about a person named {player_name}. In addition to {player_name}, there are the following characters: {characters}.\n"
        "The summary should be as long as it covers enough events for a 6-hour game, it should be a long script of a journey.\n"
        "Each paragraph should represent a different stage of the story, moving from beginning to end.\n"
        "Include major chapters, challenges, twists, and how the characters evolve through the plot.\n"
        "Use simple language.\n"
        "Make it read like a summary of a high-quality story.\n\n"

        "Don't use markdown.\n"
        "Follow these extra requests if provided: {game_extra_requests}\n\n"
    )
)

def llm_story(genre: str, main_char: str, difficulty: str, extra: str, characters: str, *, use_free_models: bool = False, language: str = 'en') -> dict:
    """Services.generate_story ile uyumlu: (genre, main_char, difficulty, extra) -> {'introductory_text', 'story'}"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"llm_story called with genre={genre}, main_char={main_char}")
    
    payload = GameStoryGenerationInput(
        genre=genre, player_name=main_char, game_extra_requests=extra or "", characters=characters
    )
    # Prepend language instruction
    lang_instruction = {
        'en': 'Write outputs in English.',
        'es': 'Escribe las salidas en español.',
        'fr': 'Rédige les sorties en français.',
        'pt-br': 'Escreva as saídas em português do Brasil.',
    }.get((language or 'en').lower(), 'Write outputs in English.')
    prompt_str = f"{lang_instruction}\n\n" + story_generation_prompt.format(
        genre=payload.genre,
        player_name=payload.player_name,
        game_extra_requests=payload.game_extra_requests,
        characters=payload.characters,
    )
    if not use_free_models:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
    
    # Create a closure that returns a fresh coroutine each time
    def create_coroutine():
        logger.debug("Creating fresh coroutine for story generation")
        if use_free_models:
            return cheap_llmPrompt(prompt_str, GameStoryGenerationOutput, use_free_models=True)
        return expensive_llmPrompt(prompt_str, GameStoryGenerationOutput)
    
    try:
        logger.debug("Calling run_async_safe for story generation")
        # 10x timeout for free-model path
        result = run_async_safe(create_coroutine, timeout=(1200 if use_free_models else 120))
        logger.debug(f"run_async_safe returned result type: {type(result)}")
        parsed = result if isinstance(result, GameStoryGenerationOutput) else GameStoryGenerationOutput.model_validate(result)
        return {
            "introductory_text": strip_markdown(parsed.introductory_text.strip()),
            "story": strip_markdown(parsed.story.strip()),
        }
    except Exception as e:
        logger.error(f"llm_story failed: {type(e).__name__}: {str(e)}", exc_info=True)
        raise
# -------------------------------------------------
# CHAPTERS: Pydantic şemaları
# -------------------------------------------------
from typing import List
from pydantic import BaseModel, Field

class GameChapterGenerationInput(BaseModel):
    game_story: str                 # story

class GameChapter(BaseModel):
    chapter_name: str
    chapter_content: str

class GameChaptersGenerationOutput(BaseModel):
    chapters: List[GameChapter] = Field(default_factory=list)

# -------------------------------------------------
# CHAPTERS: PromptTemplate + senkron LLM fonksiyonu
# -------------------------------------------------
chapter_generation_prompt = PromptTemplate.from_template(
	(
		"Take the following story summary and expand it into a detailed chapter outline.\n"
		"The story summary is divided into numbered parts.\n"
		"For each part, break it down into factual, scene-level events, like in a game script.\n"
		"Describe exactly what happens (who does what, where, when, how).\n"
		"Do not just summarize — write concrete events, e.g., \"Ahmet meets a sorcerer, the sorcerer casts a spell, they flee into the forest, Ahmet is wounded, Mehmet cries.\"\n\n"
		"Use 12 chapters total.\n"
		"Each chapter must have: (1) a short, informative title; (2) 8–10 bullet-point events.\n"
		"Each bullet must be a single concrete event.\n"
		"Keep bullets concise (one line each). Avoid prose paragraphs, keep to bullets only.\n\n"
		"Here is the story summary to expand:\n"
		"{story}\n"
	)
)

def llm_plan_chapters(story: str, *, use_free_models: bool = False, language: str = 'en') -> list[str]:
    """
    Chapter planları döner.
    """
    _input = GameChapterGenerationInput(
        game_story=story,
    )

    lang_instruction = {
        'en': 'Write outputs in English.',
        'es': 'Escribe las salidas en español.',
        'fr': 'Rédige les sorties en français.',
        'pt-br': 'Escreva as saídas em português do Brasil.',
    }.get((language or 'en').lower(), 'Write outputs in English.')
    prompt_str = (lang_instruction + "\n\n") + chapter_generation_prompt.format(
        story=_input.game_story,
    )

    if not use_free_models:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
    # Create a closure that returns a fresh coroutine each time
    def create_coroutine():
        return cheap_llmPrompt(prompt_str, GameChaptersGenerationOutput, use_free_models=use_free_models)
    # 10x timeout for free-model path
    result = run_async_safe(create_coroutine, timeout=(600 if use_free_models else 60))
    parsed = result if isinstance(result, GameChaptersGenerationOutput) else GameChaptersGenerationOutput.model_validate(result)

    total_chapters = len(getattr(parsed, "chapters", []) or [])
    empty_idx = [i for i, ch in enumerate(parsed.chapters or []) if not (getattr(ch, "chapter_content", "") or "").strip()]
    nonempty_count = total_chapters - len(empty_idx)

    plans = [
        strip_markdown((ch.chapter_content or "").strip())
        for ch in (parsed.chapters or [])
        if (ch.chapter_content or "").strip()
    ]

    if not plans:
        # Provide a concrete reason to aid debugging/monitoring
        reason_parts: list[str] = [f"model returned 0 non-empty chapter contents"]
        reason_parts.append(f"chapters={total_chapters}")
        if total_chapters > 0:
            reason_parts.append(f"empty_contents={len(empty_idx)} at indexes={empty_idx}")
        # The prompt requests 12 chapters; surface if the count deviates
        if total_chapters != 12:
            reason_parts.append(f"expected=12 (as instructed in prompt)")
        reason = "; ".join(reason_parts)
        logger.error(f"Chapter planning failed: {reason}")
        raise ValueError(f"Chapter planning failed: {reason}")

    # Soft warning if chapter count deviates from the prompt's instruction
    if total_chapters != 12:
        logger.warning(f"Chapter planning returned {total_chapters} chapters (expected 12 per prompt)")

    return plans

# -------------------------------------------------
# TURN: Pydantic şemaları
# -------------------------------------------------
from typing import Optional, List
from pydantic import BaseModel, Field

class GamePartGenerationInput(BaseModel):
    game_story: str               # story (+ istersen world'u ekleyebilirsin)
    sound_effects: str            # "name - description" satırları; tek string
    game_chapter_summary: str     # aktif chapter planı
    events_summary: str           # realized_story özeti (LLM yönetecek)
    game_extra_requests: str = ""


class GamePartGenerationOutput(BaseModel):
    scene: str = Field(..., description="4-6 sentences of narration")
    voice_name: str = Field(..., description="narrator/character name/random")
    choices: List[str] = Field(default_factory=list, description="3 short action options")
    image_description: str = Field(..., description="Visual description for AI, no names, specify lighting/time")
    selected_sound_effect: str = Field("", description="One from the list")

# -------------------------------------------------
# FIRST TURN VISUAL: Pydantic şemaları
# -------------------------------------------------
class FirstTurnVisualInput(BaseModel):
    world_description: str
    genre: str
    main_character: str
    setting_details: str = ""

class FirstTurnVisualOutput(BaseModel):
    image_description: str = Field(..., description="Detailed visual description of the world/setting for image generation")

# -------------------------------------------------
# FIRST TURN VISUAL: PromptTemplate + senkron LLM fonksiyonu
# -------------------------------------------------
first_turn_visual_prompt = PromptTemplate.from_template(
    (
        "You are creating the opening visual for a {genre} story. Based on the world description below, "
        "create a vivid, atmospheric image description that captures the essence of this world and sets the mood "
        "for the story. The image should establish the setting and atmosphere without showing specific action.\n\n"
        "World Description:\n{world_description}\n\n"
        "Main Character: {main_character}\n\n"
        "Additional Details: {setting_details}\n\n"
        "Create a detailed image description focusing on the environment, atmosphere, and mood. "
        "This should be a establishing shot that introduces the world to the player. "
        "Be specific about lighting, colors, architectural details, and environmental elements."
    )
)

def llm_first_turn_visual(world_description: str, genre: str, main_character: str, *, use_free_models: bool = False, language: str = 'en') -> str:
    """Generate an image description for the first turn based on world description."""
    payload = FirstTurnVisualInput(
        world_description=world_description,
        genre=genre,
        main_character=main_character,
        setting_details=""
    )
    lang_instruction = {
        'en': 'Write outputs in English.',
        'es': 'Escribe las salidas en español.',
        'fr': 'Rédige les sorties en français.',
        'pt-br': 'Escreva as saídas em português do Brasil.',
    }.get((language or 'en').lower(), 'Write outputs in English.')
    prompt_str = (lang_instruction + "\n\n") + first_turn_visual_prompt.format(
        world_description=payload.world_description,
        genre=payload.genre,
        main_character=payload.main_character,
        setting_details=payload.setting_details
    )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    # Create a closure that returns a fresh coroutine each time
    def create_coroutine():
        return cheap_llmPrompt(prompt_str, FirstTurnVisualOutput, use_free_models=use_free_models)
    # 10x timeout for free-model path
    result = run_async_safe(create_coroutine, timeout=(600 if use_free_models else 60))
    parsed = result if isinstance(result, FirstTurnVisualOutput) else FirstTurnVisualOutput.model_validate(result)
    return strip_markdown(parsed.image_description.strip())

# -------------------------------------------------
# TURN: PromptTemplate + senkron LLM fonksiyonu
# -------------------------------------------------
from langchain_core.prompts import PromptTemplate
import json
from pathlib import Path
from django.conf import settings

# Load sound effects from class constants instead of JSON file
def _load_sound_effects_string() -> str:
    """Load sound effects from GameConstants class."""
    try:
        from .models import GameConstants
        sound_effects = GameConstants.get_sound_effects()
        lines = [f"{sfx['name']} - {sfx.get('description', '')}" for sfx in sound_effects]
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to load sound effects: {e}")
        return ""

part_generation_prompt = PromptTemplate.from_template(
    (
        "You are Eternalore, an AI designed to immerse users in an interactive visual story game. Upon starting, you immediately create a vivid text depicting the {genre} genre. Vividly describe the scene with a long text. Remember to write an accurate, consistent and rich text possibly detailed. User responses guide the story, with you generating new text representing the consequences of their choices, thus evolving the narrative. Do present to user choices. For each user answer, focus on accurately and realistically interpreting and expanding user choices to maintain a coherent, fun, and engaging story. Don't talk personally to the user, he is inside the game.\n\nDetails:\n"
        "Keep the difficulty level {difficulty}\n"
        "User's In-Game Name is {player_name}\n"
        "This is the Game Story we planned for you to follow: {game_story}\nEven if you know the whole story from start to finish, user doesn't. So you must go through the story with him slowly, and don't reveal the whole story easily, make them experience the story as if they are in the game.\nDelay the stuff that will progress the plot too much. Also, describe who other characters in detail, and give the user a picture of the daily life and setting of the world while progressing the plot.\nDon't include omniscient commentaries from the narrator's perspective such as 'Change was comging, whether X was ready for it or not.'\n\nMake some characters say funny jokes, but avoid having them have a goofy personality, except perhaps 1 or 2 characters.\nEach character should have a unique personality and background, and character traits about them.\n"
        "This is the Game Chapter Summary: {game_chapter_summary}\nThe main focus is to use this chapter summary. Although I gave you the full summary of the story above, the part you will actually play is this chapter summary. Go through this summary slowly, in a fun and realistic way.\n"
        "This is the Events Summary: {events_summary}\nEvents Summary, unlike the summaries I have provided above, is a summary of ‘events that have actually occurred’; it is not a plan like the others. The summary written here describes events that have actually happened so far.\n"
        "Follow the Extra Requests: {game_extra_requests}\n\n"
        "Your role is to respond to user's choices by revealing consequences, evolving relationships, and maintaining a breathing world where every action ripples forward.\n\n"
        "Follow the instructions below:\n"
        "1. Show consequence of the choice (emotional, physical, or both)\n"
        "2. Evolve world state (time passes, characters act, environment changes)\n"
        "3. Offer 3 meaningful choices reflecting different values. Don't be repetitive.\n"
        "4. Maintain relationship dynamics based on history\n"        
        "5. Everything has a consequence, for example if the user breaks a vase, the user will have to pay for it, or if the user kills a character, the user will have to deal with the police, or if the user steals something, the user may have to deal with the consequences of being caught.\n"
        "6. Apply the user's response, immediately and strictly.\n"
        "7. Make the world alive and convincing. For example, all characters do things independently and want things.\n"
        "8. Adapt to user.\n"
        "9. Make the experience {difficulty} difficulty\n"
        "If the difficulty is Easy: Obvious hints, multiple chances, resources common\n"
        "If the difficulty is Normal: Subtle clues, limited retries, resources need planning\n"
        "If the difficulty is Hard: Hidden information, permanent consequences, scarce resources\n"
        "10. Be creative. Have fun.\n"
        "11. Show, don't tell.\n"
        "12. NEVER use markdown. Even if it's simple.\n"
        "13. The game should not be boring in any way. Keep the experience alive through both choices and storytelling.\n"
        "14. Avoid being restrictive. Make the user feel like they are in an alive game, not a script.\n"
        "15. Choices should always be physical actions. For example, punching is physical, waiting is physical, thinking is not physical, remembering is not physical.\n"
        "16. Don't overdo the poetic and literary talk. Mostly just describe physical, objective things. Avoid rambling. Talk in detail. It should be an exciting, passionate, fun, and engaging game.\n"
        "17. Every choice should have a consequence. Don't let the user choose something that doesn't have a consequence. For example, if the user chooses to wait, the user will have to wait until something. If the user chooses to punch, the user will experience the consequence of punching immediately.\n"
        "18. Ensure all choices presented to the player are logical and reasonable actions that a person might realistically consider taking in the given situation, avoiding absurd or self-destructive options unless specifically relevant to the narrative.\n"
        "19. Avoid common clichés. Create fresh and creative narratives.\n"
        "20. Introduce the characters with a natural pace while progressing the story.\n"
        "20. Let the story unfold slowly and naturally. Don't rush to reveal plot points. Keep the flow unhurried and organic. Allow the narrative to breathe between major events, giving players time to absorb each development before introducing the next. Give the user space and time to enjoy, feel, think, act, interact, and explore the world.\n"
        "21. Maintain narrative consistency by ensuring items, characters, and plot developments follow logical cause and effect, with elements having clear origins or explanations. While creative surprises and developments are encouraged, avoid contradictions or sudden appearances that break immersion without proper context or foreshadowing.\n"
        "22. When introducing new elements, ensure they feel natural and earned within the story's context. Choices should reference only what the player knows, while narrative should expand their knowledge organically.\n"
        "22. You are the master of this game. Everything is under your control. Be creative and clever in creating an enjoyable and immersive experience."
        "Outputs:\n"
        "1. Scene: Write 4-6 sentences. Users can only read the scene.\n"
        "2. Image Description: describe the appearance of the scene. Describe the location details such as lighting, time, weather, atmosphere, etc. Don't include any characters in the images. Keep in mind that user can't read image descriptions. Keep it simple and short. It should be from the user's pov/fps/perspective.\n"
        "3. selected_sound_effect: choose a sound effect from the list:\n"
        "{sound_effects}\nI provided you the list of sound effects above, choose one from the list.\n"
        "4. choices: [3 distinct value-based actions]\n"
        "5. voice_name: [narrator/character name/random]. If you are narrating the scene, write 'narrator', if a character is speaking, write that character's name, if a random and unimportant character is speaking, write 'random'. Always separate character dialogue from the narrator. If any character is talking, the scene output should contain only what that character says, same goes for random.\n"
    )
)

def llm_next(messages: list[dict], game, chapter, story) -> GamePartGenerationOutput:
    """
    Services.generate_turn ile uyumlu structured çıktı döner (Pydantic instance).
    messages: mevcut geçmiş (user/assistant rolleri), biz başa sistem benzeri bir çerçeve ekliyoruz.
    """
    # Girdi derle
    sfx_str = _load_sound_effects_string()
    # Dünya bilgisini ve varsa realized_story özetini dahil et
    combined_story_block = (
        f"{(story.story or '').strip()}\n\nIntroduction:\n{(story.introductory_text or '').strip()}"
    ).strip()

    payload = GamePartGenerationInput(
        game_story=combined_story_block,
        sound_effects=sfx_str,
        game_chapter_summary=chapter.planned_chapter_story,
        events_summary=(story.realized_story or ""),
        game_extra_requests=(getattr(game, "extra_requests", "") or ""),
    )
    lang_instruction = {
        'en': 'Write outputs in English.',
        'es': 'Escribe todas las salidas en español.',
        'fr': 'Rédige toutes les sorties en français.',
        'pt-br': 'Escreva todas as saídas em português do Brasil.',
    }.get(getattr(game, 'language_code', 'en').lower(), 'Write outputs in English.')
    system_block = (lang_instruction + "\n\n") + part_generation_prompt.format(
        genre=game.genre_name,
        difficulty=game.difficulty,
        player_name=game.main_character_name,
        game_story=payload.game_story,
        sound_effects=payload.sound_effects,
        game_chapter_summary=payload.game_chapter_summary,
        events_summary=payload.events_summary,
        game_extra_requests=payload.game_extra_requests,
    )

    # System içeriğini başa koyup sonra mevcut mesajları ekle
    msgs = [{"role": "system", "content": system_block}] + (messages or [])

    # Structured output → GamePartGenerationOutput
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    # Create a closure that returns a fresh coroutine each time
    def create_coroutine():
        # Use free model sequence for FREE plan users
        use_free = False
        try:
            plan = getattr(getattr(game, "user", None), "subscription_plan", "") or ""
            use_free = (str(plan).lower() == "free")
        except Exception:
            use_free = False
        return cheap_llmMessages(msgs, GamePartGenerationOutput, use_free_models=use_free)
    # 10x timeout for free-model path
    result = run_async_safe(create_coroutine, timeout=(600 if (getattr(getattr(game, "user", None), "subscription_plan", "") == "free") else 60))
    parsed = (
        result if isinstance(result, GamePartGenerationOutput)
        else GamePartGenerationOutput.model_validate(result)
    )
    # Sanitize fields against markdown before returning
    try:
        cleaned = GamePartGenerationOutput(
            scene=strip_markdown((parsed.scene or "").strip()),
            voice_name=strip_markdown((parsed.voice_name or "").strip()),
            choices=[strip_markdown((c or "").strip()) for c in list(parsed.choices or [])],
            image_description=strip_markdown((parsed.image_description or "").strip()),
            selected_sound_effect=strip_markdown((parsed.selected_sound_effect or "").strip()),
        )
        return cleaned
    except Exception:
        return parsed

# -------------------------------------------------
# STORY SUMMARIZATION: PromptTemplate + function
# -------------------------------------------------
story_summarization_prompt = PromptTemplate.from_template(
    (
        "You are a continuity memory logger.\n\n"
        "TASK\n"
        "Read the input story below. Produce a detailed continuity log that will keep the story consistent over a long game.\n\n"
        "STYLE\n"
        "- Simple language. Short, clear sentences. Past tense. Neutral voice.\n"
        "- No narrator/player lines. No scripts. No dialogue unless a short quote is vital.\n"
        "- Be concrete and factual: who did what, where, when, how, with what result.\n\n"
        "OUTPUT\n\n"
        "Primer (short, only if useful)\n"
        "- World: setting, threats, mysteries.\n"
        "- Cast: names, roles, look, traits, current goals, fears, secrets (if any).\n"
        "- Key places: names, purpose, hazards, routes.\n"
        "- Notable items: name, type, owner/holder, container, location, condition (mint/good/worn/damaged/broken), durability %, age/origin (or “unknown”), properties (magical/tech), charges/uses left, markings/inscriptions, maintenance done, risks (poison/curse/etc.), known history.\n\n"
        "Continuity Log\n"
        "- Break into chapters and scenes as needed (choose what makes sense).\n"
        "- For each scene, log in this order:\n\n"
        "Time:\n"
        "Place: (name; environment: weather, light, noise; changes since last scene)\n"
        "Cast present: (names with current state: injuries, conditions, emotions)\n"
        "Goals: (what each side wants right now)\n\n"
        "Events (numbered, atomic facts):\n"
        "  1) [actor] does [action] to/with [target/object], at [place], using [method/tool].\n"
        "  2) Immediate outcome and state change.\n"
        "  3) Keep clear cause → effect links.\n\n"
        "Items (track every change, every time an item appears):\n"
        "  - Name:\n"
        "    • Holder: previous → new (or dropped/stored)\n"
        "    • Container: previous → new\n"
        "    • Location: previous → new\n"
        "    • Condition: previous → new (note damage: cut/burn/corrosion/etc.)\n"
        "    • Durability: previous% → new%\n"
        "    • Age/Origin: updates and source of info\n"
        "    • Properties/Material: updates (e.g., “salty residue detected”)\n"
        "    • Charges/Uses: previous → new; maintenance done? (Y/N)\n"
        "    • Markings/Inscription: newly observed?\n"
        "    • Safety/Legal: changes or risks\n\n"
        "Relationship dynamics (no scores):\n"
        "  - Describe shifts with reasons and effects.\n"
        "  - Example: “After Aria inspected the parchment for salt, Ahmet grew angry. Tension rises; they avoid eye contact.”\n\n"
        "Knowledge & clues:\n"
        "  - New facts learned (source, confidence 0–1), false beliefs, unanswered questions.\n\n"
        "Environment changes:\n"
        "  - Map/state updates, hazards added/cleared, access opened/blocked.\n\n"
        "State after scene:\n"
        "  - Per character: health/status, emotions, inventory delta, position.\n"
        "  - Party: resources, route/plan, promises or debts.\n\n"
        "Open threads:\n"
        "  - Tasks started/advanced/resolved; timers/deadlines; what to verify later.\n\n"
        "RULES\n"
        "- Be specific. Prefer small steps over vague summaries.\n"
        "- Track item condition/durability/age every time it appears. Record changes.\n"
        "- If something is unknown, write “unknown” and add it to Open threads.\n"
        "- Keep cause → effect explicit to protect continuity.\n\n"
        "MERGING WITH PRIOR LOGS\n"
        "- If a previous continuity log or summary is provided, merge it with the new events.  \n"
        "- Keep all earlier details intact, update them only if the new events change the facts.  \n"
        "- Append new events as continuation of the log.  \n"
        "- If no prior log is provided, create a new one from scratch.\n\n"

        "INPUT\n"
        "{conversation_text}\n"
    )
)

def llm_summarize_story(
    messages: List[dict],
    current_summary: str = "",
) -> str:
    """
    Create a summary of story progression from message history.
    
    Args:
        messages: List of user/assistant message dictionaries
        current_summary: Existing story summary to build upon
        game_context: Game genre, main character, world info
        chapter_context: Current chapter information
        
    Returns:
        String summary of the story progression
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Creating story summary from {len(messages)} messages")
    
    # Build input solely from conversation messages
    conversation_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            conversation_parts.append(f"PLAYER ACTION: {content}")
        elif role == "assistant":
            conversation_parts.append(f"STORY RESPONSE: {content}")
    conversation_text = "\n\n".join(conversation_parts)
    if current_summary:
        combined_input = (
            f"Previous continuity log:\n{current_summary}\n\nNew events:\n{conversation_text}"
        )
    else:
        combined_input = conversation_text

    # Format the prompt (template expects one text block)
    prompt_str = story_summarization_prompt.format(
        conversation_text=combined_input or "",
    )
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not configured - cannot perform summarization")
        raise ValueError("Cannot summarize story: OPENAI_API_KEY not configured")
    
    # Create a closure that returns a fresh coroutine each time
    def create_coroutine():
        logger.debug("Creating fresh coroutine for story summarization")
        return cheap_llmPrompt(prompt_str)
    
    try:
        logger.debug("Calling run_async_safe for story summarization")
        result = run_async_safe(create_coroutine, timeout=120)
        logger.debug(f"Story summarization completed, result type: {type(result)}")
        # Try to normalize to string
        if isinstance(result, str):
            text = result
        else:
            text = getattr(result, "content", None) or str(result)
        logger.info(f"Story summarized: {len(text)} chars")
        return strip_markdown(text.strip())
        
    except Exception as e:
        logger.error(f"Story summarization failed: {type(e).__name__}: {str(e)}", exc_info=True)
        raise
