"""
Consolidated security functions for production-ready code.
"""
import os
import json
import hashlib
import mimetypes
import logging
import bleach
import string
import time
import random
from pathlib import Path
from typing import Optional
from functools import wraps

from django.conf import settings
from django.core.exceptions import ValidationError, SuspiciousOperation, PermissionDenied
from django.core.cache import cache
from django.http import JsonResponse
import httpx
import aiofiles
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

# ============================================
# FILE HANDLING SECURITY
# ============================================

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_AUDIO_TYPES = {
    'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 
    'audio/webm', 'audio/x-wav', 'audio/wave'
}
ALLOWED_IMAGE_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 
    'image/gif', 'image/bmp'
}

def validate_file_type(content_type: str, kind: str) -> bool:
    """Validate that the content type is allowed for the given kind."""
    if kind == "audio":
        return content_type.lower() in ALLOWED_AUDIO_TYPES
    elif kind == "image":
        return content_type.lower() in ALLOWED_IMAGE_TYPES
    return False

def secure_persist_remote(temp_url: str, kind: str, user_id: Optional[str] = None) -> str:
    """
    Securely download and persist a remote file with validation.
    """
    import uuid
    
    if kind not in ("audio", "image"):
        raise ValidationError(f"Invalid file kind: {kind}")
    
    # Validate URL is HTTPS (except for local development)
    if not settings.DEBUG and not temp_url.startswith("https://"):
        raise SuspiciousOperation(f"Insecure URL protocol: {temp_url}")
    
    try:
        # Download with retry logic and configurable timeout
        max_retries = 3
        base_timeout = 45  # Increased from 30 for large images
        last_exc = None
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff: 45s, 60s, 75s
                timeout = base_timeout + (attempt * 15)
                headers = {'User-Agent': 'Eternalore/1.0'}
                
                with httpx.stream("GET", temp_url, timeout=timeout, headers=headers, follow_redirects=True) as response:
                    response.raise_for_status()
                    # Check content length
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        raise ValidationError(f"File too large: {content_length} bytes")
                    
                    # Validate content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not validate_file_type(content_type, kind):
                        raise SuspiciousOperation(f"Invalid content type: {content_type}")
                    
                    # Determine file extension safely
                    if kind == "audio":
                        ext = ".wav" if "wav" in content_type else \
                              ".mp3" if "mpeg" in content_type or "mp3" in content_type else \
                              ".ogg" if "ogg" in content_type else ".audio"
                    else:
                        ext = ".png" if "png" in content_type else \
                              ".webp" if "webp" in content_type else \
                              ".jpg" if "jpeg" in content_type or "jpg" in content_type else \
                              ".gif" if "gif" in content_type else ".image"
                    
                    # Create output directory
                    sub_dir = "audio" if kind == "audio" else "image"
                    out_dir = Path(settings.MEDIA_ROOT) / sub_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate secure filename using hash
                    file_hash = hashlib.sha256(f"{uuid.uuid4().hex}{user_id or ''}".encode()).hexdigest()[:16]
                    out_path = out_dir / f"{file_hash}{ext}"
                    
                    # Download and write with size checking
                    total_size = 0
                    with open(out_path, 'wb') as f:
                        for chunk in response.iter_bytes():
                            if chunk:
                                total_size += len(chunk)
                                if total_size > MAX_FILE_SIZE:
                                    # Clean up partial file
                                    f.close()
                                    out_path.unlink(missing_ok=True)
                                    raise ValidationError(f"File exceeds maximum size during download")
                                f.write(chunk)
                    
                    # Verify file was written
                    if not out_path.exists() or out_path.stat().st_size == 0:
                        raise ValidationError("File write failed")
                    
                    # Additional validation for images
                    if kind == "image":
                        try:
                            from PIL import Image
                            with Image.open(out_path) as img:
                                img.verify()
                                if img.width > 8192 or img.height > 8192:
                                    out_path.unlink(missing_ok=True)
                                    raise ValidationError("Image dimensions too large")
                        except Exception as e:
                            out_path.unlink(missing_ok=True)
                            raise ValidationError(f"Invalid image file: {e}")
                    
                    # Return the media URL
                    rel_path = out_path.relative_to(settings.MEDIA_ROOT).as_posix()
                    media_url = f"{settings.MEDIA_URL}{rel_path}"
                    
                    logger.info(f"Persisted {kind} file: {media_url} (user: {user_id})")
                    return media_url
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Download attempt {attempt + 1} failed for {kind} from {temp_url}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                continue
        
        # If loop exits without return
        raise ValidationError(f"Failed to download file: {last_exc}")
        
    except httpx.RequestError as e:
        logger.error(f"Failed to download {kind} from {temp_url}: {e}")
        raise ValidationError(f"Failed to download file: {e}")

def secure_delete_media_url(url: Optional[str]) -> bool:
    """
    Securely delete a media file by its URL.
    """
    if not url:
        return False
    
    # Only delete files from our media URL
    if not url.startswith(settings.MEDIA_URL):
        logger.warning(f"Attempted to delete non-media URL: {url}")
        return False
    
    # Extract relative path
    rel_path = url[len(settings.MEDIA_URL):]
    
    # Prevent path traversal attacks
    if ".." in rel_path or rel_path.startswith("/"):
        logger.error(f"Path traversal attempt detected: {rel_path}")
        raise SuspiciousOperation(f"Invalid file path")
    
    # Construct full path and resolve it
    full_path = (Path(settings.MEDIA_ROOT) / rel_path).resolve()
    
    # Ensure the resolved path is still within MEDIA_ROOT
    try:
        full_path.relative_to(Path(settings.MEDIA_ROOT).resolve())
    except ValueError:
        logger.error(f"Path escape attempt: {full_path}")
        raise SuspiciousOperation(f"Path outside media directory")
    
    # Delete if exists
    try:
        if full_path.is_file():
            full_path.unlink()
            logger.info(f"Deleted media file: {url}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete {url}: {e}")
    
    return False

# ============================================
# ASYNC FILE PERSISTENCE
# ============================================

async def secure_persist_remote_async(temp_url: str, kind: str, user_id: Optional[str] = None) -> str:
    """
    Async version of secure_persist_remote using httpx + aiofiles.
    """
    import uuid
    if kind not in ("audio", "image"):
        raise ValidationError(f"Invalid file kind: {kind}")

    if not settings.DEBUG and not temp_url.startswith("https://"):
        raise SuspiciousOperation(f"Insecure URL protocol: {temp_url}")

    max_retries = 3
    base_timeout = 45
    last_exc: Exception | None = None
    headers = {'User-Agent': 'Eternalore/1.0'}

    for attempt in range(max_retries):
        try:
            timeout = base_timeout + (attempt * 15)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", temp_url, headers=headers) as response:
                    response.raise_for_status()
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        raise ValidationError(f"File too large: {content_length} bytes")
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not validate_file_type(content_type, kind):
                        raise SuspiciousOperation(f"Invalid content type: {content_type}")

                    if kind == "audio":
                        ext = ".wav" if "wav" in content_type else \
                              ".mp3" if "mpeg" in content_type or "mp3" in content_type else \
                              ".ogg" if "ogg" in content_type else ".audio"
                    else:
                        ext = ".png" if "png" in content_type else \
                              ".webp" if "webp" in content_type else \
                              ".jpg" if "jpeg" in content_type or "jpg" in content_type else \
                              ".gif" if "gif" in content_type else ".image"

                    sub_dir = "audio" if kind == "audio" else "image"
                    out_dir = Path(settings.MEDIA_ROOT) / sub_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    file_hash = hashlib.sha256(f"{uuid.uuid4().hex}{user_id or ''}".encode()).hexdigest()[:16]
                    out_path = out_dir / f"{file_hash}{ext}"

                    total_size = 0
                    async with aiofiles.open(out_path, 'wb') as f:
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                total_size += len(chunk)
                                if total_size > MAX_FILE_SIZE:
                                    await f.close()
                                    try:
                                        out_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                    raise ValidationError("File exceeds maximum size during download")
                                await f.write(chunk)

                    if not out_path.exists() or out_path.stat().st_size == 0:
                        raise ValidationError("File write failed")

                    if kind == "image":
                        try:
                            from PIL import Image
                            def _verify():
                                with Image.open(out_path) as img:
                                    img.verify()
                                    return img.size
                            width, height = await sync_to_async(_verify)()
                            if width > 8192 or height > 8192:
                                out_path.unlink(missing_ok=True)
                                raise ValidationError("Image dimensions too large")
                        except Exception as e:
                            out_path.unlink(missing_ok=True)
                            raise ValidationError(f"Invalid image file: {e}")

                    rel_path = out_path.relative_to(settings.MEDIA_ROOT).as_posix()
                    media_url = f"{settings.MEDIA_URL}{rel_path}"
                    logger.info(f"Persisted {kind} file (async): {media_url} (user: {user_id})")
                    return media_url
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                break
            wait_time = 2 ** attempt
            logger.warning(f"Async download attempt {attempt+1} failed for {kind} from {temp_url}, retrying in {wait_time}s: {e}")
            await sync_to_async(time.sleep)(wait_time)

    logger.error(f"Failed to download {kind} from {temp_url}: {last_exc}")
    raise ValidationError(f"Failed to download file: {last_exc}")

# ============================================
# INPUT VALIDATION & SANITIZATION
# ============================================

# Input validation constants
MAX_TEXT_LENGTH = 5000
MAX_NAME_LENGTH = 50
MAX_GENRE_LENGTH = 30
ALLOWED_VISUAL_STYLES = {'realistic', 'pixel_art', 'illustration', 'anime'}
ALLOWED_DIFFICULTIES = {'easy', 'normal', 'hard'}
ALLOWED_STORY_LENGTHS = {'short', 'standard'}

def sanitize_prompt(prompt: str, max_length: int = 1000) -> str:
    """
    Sanitize user prompts to prevent injection attacks.
    """
    if not prompt:
        return ""
    
    # Remove any potential injection patterns
    dangerous_patterns = [
        "ignore previous", "forget everything", "system:", 
        "assistant:", "user:", "\\n\\n", "```"
    ]
    
    sanitized = prompt.strip()
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern.lower(), "")
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def sanitize_user_input(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Sanitize user input to prevent XSS and injection attacks.
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = bleach.clean(text, tags=[], strip=True)
    
    # Limit length
    text = text[:max_length]
    
    # Remove control characters
    allowed_chars = string.printable
    text = ''.join(c for c in text if c in allowed_chars)
    
    return text.strip()

def validate_game_params(data: dict) -> dict:
    """
    Validate and sanitize game creation parameters.
    """
    cleaned = {}
    
    # Genre validation
    genre = sanitize_user_input(data.get('genre', ''), MAX_GENRE_LENGTH)
    if not genre:
        raise ValidationError("Genre is required")
    cleaned['genre'] = genre
    
    # Character name validation
    character = sanitize_user_input(data.get('mainCharacter', ''), MAX_NAME_LENGTH)
    if not character:
        raise ValidationError("Character name is required")
    if len(character) < 2:
        raise ValidationError("Character name too short")
    cleaned['mainCharacter'] = character
    
    # Visual style validation
    visual_style = data.get('visualStyle', 'illustration').lower()
    if visual_style not in ALLOWED_VISUAL_STYLES:
        visual_style = 'illustration'
    cleaned['visualStyle'] = visual_style
    
    # Difficulty validation
    difficulty = data.get('difficulty', 'normal').lower()
    if difficulty not in ALLOWED_DIFFICULTIES:
        difficulty = 'normal'
    cleaned['difficulty'] = difficulty
    
    # Story length validation
    story_length = data.get('storyLength', 'standard').lower()
    if story_length not in ALLOWED_STORY_LENGTHS:
        story_length = 'standard'
    cleaned['storyLength'] = story_length
    
    # Extra requests (optional)
    extra = sanitize_user_input(data.get('extraRequests', ''), 500)
    cleaned['extraRequests'] = extra
    
    return cleaned

def validate_json_request(request):
    """
    Safely parse JSON request body.
    """
    try:
        if request.body:
            return json.loads(request.body.decode('utf-8'))
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Invalid JSON in request: {e}")
        raise ValidationError("Invalid JSON in request body")

def validate_request_data(request):
    """
    Safely parse request data, handling both JSON and FormData.
    Returns a dictionary with validated data.
    """
    try:
        content_type = request.META.get('CONTENT_TYPE', '')
        
        if 'application/json' in content_type:
            # Handle JSON requests
            if request.body:
                return json.loads(request.body.decode('utf-8'))
            return {}
        elif 'multipart/form-data' in content_type or 'application/x-www-form-urlencoded' in content_type:
            # Handle FormData/URL-encoded requests
            # Convert QueryDict to regular dict for consistent handling
            return dict(request.POST.items())
        else:
            # Fallback - try to detect based on request body
            if request.body:
                try:
                    return json.loads(request.body.decode('utf-8'))
                except:
                    # If JSON parsing fails, try form data
                    return dict(request.POST.items())
            return {}
    except Exception as e:
        logger.warning(f"Invalid request data: {e}")
        raise ValidationError("Invalid request data")

# ============================================
# RATE LIMITING & QUOTA MANAGEMENT
# ============================================

def rate_limit_api(max_calls=10, window=60):
    """
    Decorator for rate limiting API endpoints per user.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({'error': 'Authentication required'}, status=401)
            
            cache_key = f"api_rate_limit:{request.user.id}:{func.__name__}"
            current_count = cache.get(cache_key, 0)
            
            if current_count >= max_calls:
                return JsonResponse({
                    'error': 'Rate limit exceeded',
                    'retry_after': window
                }, status=429)
            
            # Increment counter
            cache.set(cache_key, current_count + 1, window)
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator

def check_user_quota(user, required_seconds: float = 0) -> tuple[bool, str]:
    """
    Check if user has sufficient quota for audio generation.
    """
    from user.views import _plan_limits
    
    try:
        effective_plan = user.get_effective_plan_for_limits()
    except Exception:
        effective_plan = getattr(user, 'subscription_plan', 'free')
    limits = _plan_limits(effective_plan)
    try:
        usage = user.current_cycle_usage()
    except Exception:
        usage = user.current_month_usage()
    
    audio_limit = limits.get('monthly_audio_seconds', 0)
    audio_used = float(usage.get('total_audio_seconds', 0))
    
    if audio_limit == 0:
        return False, "Audio generation not available on free plan"
    
    remaining = audio_limit - audio_used
    if remaining < required_seconds:
        return False, f"Insufficient audio quota. {remaining:.1f} seconds remaining, {required_seconds:.1f} required"
    
    return True, "Quota available"

# ============================================
# OWNERSHIP & PERMISSION VALIDATION
# ============================================

def validate_ownership(user, obj, field='user'):
    """
    Validate that the user owns the object.
    """
    # Prefer FK id comparison to avoid lazy relation loads in async contexts
    owner_id_attr = f"{field}_id"
    try:
        obj_owner_id = getattr(obj, owner_id_attr, None)
        user_id = getattr(user, "id", None)
        if obj_owner_id is not None and user_id is not None:
            # Compare as strings to be robust across UUID/str types
            if str(obj_owner_id) == str(user_id):
                return
            # Fast fail if ids don't match
            raise PermissionDenied("You don't have permission to access this resource")
    except Exception:
        # Fall back to relation access below if attributes are not present
        pass

    # Fallback: access related owner object (may hit DB in sync contexts)
    owner = getattr(obj, field, None)
    if not owner:
        raise PermissionDenied("Object has no owner")

    owner_email = getattr(owner, 'email', str(owner))
    user_email = getattr(user, 'email', str(user))
    if owner_email != user_email:
        raise PermissionDenied("You don't have permission to access this resource")

# ============================================
# RETRY & ERROR HANDLING
# ============================================

def with_exponential_backoff(
    func,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0
):
    """
    Retry a function with exponential backoff.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                # Calculate next delay with jitter
                jitter = random.uniform(0, delay * 0.1)
                actual_delay = min(delay + jitter, max_delay)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {actual_delay:.1f}s"
                )
                
                time.sleep(actual_delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")
                raise
    
    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed without exception")

# ============================================
# LOGGING & MONITORING
# ============================================

def log_suspicious_activity(request, reason):
    """
    Log suspicious activity for monitoring.
    """
    logger.warning(
        f"Suspicious activity: {reason} | "
        f"User: {request.user.id if request.user.is_authenticated else 'Anonymous'} | "
        f"IP: {request.META.get('REMOTE_ADDR')} | "
        f"Path: {request.path} | "
        f"Method: {request.method}"
    )