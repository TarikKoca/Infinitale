"""LLM module: provides async LLM wrappers with OpenRouter as primary and OpenAI as fallback.

This module provides the async functions used by prompts.py for LLM calls.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio
from contextlib import contextmanager

load_dotenv()
logger = logging.getLogger(__name__)

# Honor DISABLE_LLM_PROXIES by removing proxy environment variables for LLM calls
def _disable_llm_proxies_if_requested() -> None:
    try:
        flag = (os.getenv("DISABLE_LLM_PROXIES", "").strip().lower() in {"1", "true", "yes", "on"})
        if not flag:
            return
        # Compute hosts of LLM providers and add them to NO_PROXY so only LLM calls bypass proxies
        from urllib.parse import urlparse
        openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        hosts = []
        try:
            hosts.append(urlparse(openrouter_base).hostname or "openrouter.ai")
        except Exception:
            hosts.append("openrouter.ai")
        try:
            hosts.append(urlparse(openai_base).hostname or "api.openai.com")
        except Exception:
            hosts.append("api.openai.com")
        # Deduplicate and merge with existing NO_PROXY
        existing_no_proxy = os.getenv("NO_PROXY", os.getenv("no_proxy", ""))
        existing_hosts = [h.strip() for h in existing_no_proxy.split(",") if h.strip()]
        merged = existing_hosts + [h for h in hosts if h and h not in existing_hosts]
        if merged:
            os.environ["NO_PROXY"] = ",".join(merged)
            os.environ["no_proxy"] = ",".join(merged)
        # Permanently clear generic proxy env vars to avoid SDK validation errors
        for key in [
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
            # Provider-specific known variables
            "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY",
        ]:
            if key in os.environ:
                os.environ.pop(key, None)
        # Ensure a strong default no-proxy behavior
        if not os.getenv("NO_PROXY"):
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
        logger.info("DISABLE_LLM_PROXIES is set; proxy env cleared and NO_PROXY updated: %s", os.getenv("NO_PROXY"))
    except Exception:
        # Never fail import due to env manipulation
        logger.warning("Failed to apply DISABLE_LLM_PROXIES settings", exc_info=True)

# Apply at import-time to affect any client creations in this module
_disable_llm_proxies_if_requested()

def _get_llm_hostnames() -> list[str]:
    from urllib.parse import urlparse
    hosts: list[str] = []
    try:
        hosts.append(urlparse(os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).hostname or "openrouter.ai")
    except Exception:
        hosts.append("openrouter.ai")
    try:
        hosts.append(urlparse(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).hostname or "api.openai.com")
    except Exception:
        hosts.append("api.openai.com")
    # Dedupe while preserving order
    result: list[str] = []
    seen: set[str] = set()
    for h in hosts:
        if h and h not in seen:
            result.append(h)
            seen.add(h)
    return result

@contextmanager
def _temporarily_disable_proxies_env():
    """Temporarily remove proxy-related env vars while creating LLM clients.

    This avoids errors like "Unknown scheme for proxy URL URL('socks://...')" originating
    from provider SDKs or httpx during client construction, while restoring the original
    environment immediately after the client is created.
    """
    if not (os.getenv("DISABLE_LLM_PROXIES", "").strip().lower() in {"1", "true", "yes", "on"}):
        # No-op
        yield
        return
    keys_to_clear = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY",
    ]
    backup: dict[str, str | None] = {}
    try:
        # Backup and clear
        for k in keys_to_clear:
            backup[k] = os.environ.get(k)
            if k in os.environ:
                os.environ.pop(k, None)
        # Extend NO_PROXY with LLM hostnames so later network code still ignores proxies for them
        current_no_proxy = os.getenv("NO_PROXY", os.getenv("no_proxy", ""))
        existing = [h.strip() for h in current_no_proxy.split(",") if h.strip()]
        for h in _get_llm_hostnames():
            if h not in existing:
                existing.append(h)
        if existing:
            joined = ",".join(existing)
            os.environ["NO_PROXY"] = joined
            os.environ["no_proxy"] = joined
        yield
    finally:
        # Restore
        for k, v in backup.items():
            if v is None:
                if k in os.environ:
                    os.environ.pop(k, None)
            else:
                os.environ[k] = v

# Client factory functions to avoid coroutine reuse issues
def create_openrouter_client():
    """Create a fresh OpenRouter client instance with sane timeouts."""
    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    with _temporarily_disable_proxies_env():
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemini-2.5-flash"),
            timeout=timeout,
            max_retries=max_retries,
        )

def create_openrouter_expensive_client():
    """Create a fresh OpenRouter expensive client instance with stricter limits."""
    timeout = float(os.getenv("LLM_EXPENSIVE_REQUEST_TIMEOUT", os.getenv("LLM_REQUEST_TIMEOUT", "90")))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    with _temporarily_disable_proxies_env():
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_EXPENSIVE_MODEL", "google/gemini-2.5-pro"),
            timeout=timeout,
            max_retries=max_retries,
        )


def _parse_free_model_sequence() -> list[str]:
    """Parse FREE_OPENROUTER_MODELS env into an ordered unique list.
    Supports:
    - Comma-separated: modelA,modelB,modelC
    - Newline-separated
    - JSON array: ["modelA","modelB","modelC"]
    """
    raw = os.getenv("FREE_OPENROUTER_MODELS", "")
    if not raw:
        return []
    raw = raw.strip()
    models: list[str] = []
    # JSON array support
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json
            arr = json.loads(raw)
            if isinstance(arr, list):
                models = [str(x).strip().strip('"').strip("'") for x in arr if str(x).strip()]
        except Exception:
            models = []
    if not models:
        # Split on commas and newlines
        import re
        parts = re.split(r"[,\n]", raw)
        models = [p.strip().strip('"').strip("'") for p in parts if p and p.strip()]
    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


async def _invoke_with_openrouter_fallbacks(input_data, *, output_model=None, request_timeout: float = 60.0):
    """Try OpenRouter free models in order; ALWAYS use structured outputs when provided.

    input_data can be messages (list) or a prompt (str). If output_model is provided,
    we use with_structured_output. If a model fails, move to the next model.
    Per-model attempts are capped to a strict timeout (default 120s).
    """
    sequence = _parse_free_model_sequence()
    last_error = None
    if not sequence:
        raise RuntimeError("FREE_OPENROUTER_MODELS is empty; cannot use free model sequence")
    # Per-model attempt timeout (strict): default 120s regardless of request_timeout
    try:
        per_attempt_timeout = float(os.getenv("FREE_MODEL_PER_ATTEMPT_TIMEOUT", "120"))
    except Exception:
        per_attempt_timeout = 120.0
    # Free-tier retry limits remain higher, but each attempt is bounded by per_attempt_timeout
    base_retries_env = int(os.getenv("LLM_MAX_RETRIES", "2"))
    free_retries = max(1, base_retries_env * 10)

    for idx, model_name in enumerate(sequence):
        try:
            with _temporarily_disable_proxies_env():
                client = ChatOpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                    model=model_name,
                    timeout=per_attempt_timeout,
                    max_retries=free_retries,
                )
            # Structured output requested
            if output_model is not None:
                structured = client.with_structured_output(output_model)
                sresp = await asyncio.wait_for(structured.ainvoke(input_data), per_attempt_timeout)
                if isinstance(sresp, output_model):
                    return sresp
                return output_model.model_validate(sresp)
            # No structure requested; return raw response
            return await asyncio.wait_for(client.ainvoke(input_data), per_attempt_timeout)
        except Exception as e:
            last_error = e
            logger.warning(f"Free model '{model_name}' failed: {type(e).__name__}: {e}")
            # If rate-limited (429), wait a bit before trying the next model
            try:
                from httpx import HTTPStatusError
                status = getattr(getattr(e, "response", None), "status_code", 0) if isinstance(e, HTTPStatusError) else 0
            except Exception:
                status = 0
            if status == 429:
                import random
                # Small backoff with jitter; capped to 2s
                delay = min(2.0, 0.5 + 0.3 * idx + random.uniform(0, 0.3))
                try:
                    await asyncio.sleep(delay)
                except Exception:
                    pass
            continue
    # If we exhausted all free models
    raise last_error or RuntimeError("All free models failed")

def create_openai_client():
    """Create a fresh OpenAI client instance with timeouts."""
    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    with _temporarily_disable_proxies_env():
        return ChatOpenAI(model="gpt-4o-mini", timeout=timeout, max_retries=max_retries)

def create_openai_expensive_client():
    """Create a fresh OpenAI expensive client instance with timeouts."""
    timeout = float(os.getenv("LLM_EXPENSIVE_REQUEST_TIMEOUT", os.getenv("LLM_REQUEST_TIMEOUT", "90")))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    with _temporarily_disable_proxies_env():
        return ChatOpenAI(model="gpt-4o", timeout=timeout, max_retries=max_retries)


async def cheap_llmMessages(messages, output_model=None, *, use_free_models: bool = False):
    """Use OpenRouter by default (or free model sequence), fallback to OpenAI after 3 retries."""
    request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
    if use_free_models:
        try:
            return await _invoke_with_openrouter_fallbacks(messages, output_model=output_model, request_timeout=request_timeout)
        except Exception as e:
            logger.warning(f"All free OpenRouter models failed: {e}")
            # Do not fall back to OpenAI for free tier
            raise
    for attempt in range(3):
        try:
            openrouter_client = create_openrouter_client()
            if output_model is not None:
                structured_model = openrouter_client.with_structured_output(output_model)
                response = await asyncio.wait_for(structured_model.ainvoke(messages), request_timeout)
                # Try to validate/convert response to expected model
                if isinstance(response, output_model):
                    return response
                else:
                    # Try to parse/validate the response as the expected model
                    try:
                        return output_model.model_validate(response)
                    except Exception as validation_error:
                        raise ValueError(f"Invalid structured response: {validation_error}")
            else:
                response = await asyncio.wait_for(openrouter_client.ainvoke(messages), request_timeout)
            return response
        except Exception as e:
            logger.warning(f"OpenRouter attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            logger.warning(f"Exception details: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
            if attempt == 2:
                logger.info("Falling back to OpenAI after 3 OpenRouter failures")
                # Only try OpenAI after all OpenRouter attempts failed
                try:
                    openai_client = create_openai_client()
                    if output_model is not None:
                        structured_model = openai_client.with_structured_output(output_model)
                        response = await asyncio.wait_for(structured_model.ainvoke(messages), request_timeout)
                    else:
                        response = await asyncio.wait_for(openai_client.ainvoke(messages), request_timeout)
                    return response
                except Exception as openai_error:
                    logger.error(f"OpenAI fallback also failed: {openai_error}")
                    raise


async def cheap_llmPrompt(prompt, output_model=None, *, use_free_models: bool = False):
    """Use OpenRouter by default (or free model sequence), fallback to OpenAI after 3 retries."""
    request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
    if use_free_models:
        try:
            return await _invoke_with_openrouter_fallbacks(prompt, output_model=output_model, request_timeout=request_timeout)
        except Exception as e:
            logger.warning(f"All free OpenRouter models failed: {e}")
            # Do not fall back to OpenAI for free tier
            raise
    for attempt in range(3):
        try:
            openrouter_client = create_openrouter_client()
            if output_model is not None:
                structured_model = openrouter_client.with_structured_output(output_model)
                response = await asyncio.wait_for(structured_model.ainvoke(prompt), request_timeout)
                # Try to validate/convert response to expected model
                if isinstance(response, output_model):
                    return response
                else:
                    # Try to parse/validate the response as the expected model
                    try:
                        return output_model.model_validate(response)
                    except Exception as validation_error:
                        raise ValueError(f"Invalid structured response: {validation_error}")
            else:
                response = await asyncio.wait_for(openrouter_client.ainvoke(prompt), request_timeout)
            return response
        except Exception as e:
            logger.warning(f"OpenRouter attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            logger.warning(f"Exception details: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
            if attempt == 2:
                logger.info("Falling back to OpenAI after 3 OpenRouter failures")
                # Only try OpenAI after all OpenRouter attempts failed
                try:
                    openai_client = create_openai_client()
                    if output_model is not None:
                        structured_model = openai_client.with_structured_output(output_model)
                        response = await asyncio.wait_for(structured_model.ainvoke(prompt), request_timeout)
                    else:
                        response = await asyncio.wait_for(openai_client.ainvoke(prompt), request_timeout)
                    return response
                except Exception as openai_error:
                    logger.error(f"OpenAI fallback also failed: {openai_error}")
                    raise


async def expensive_llmPrompt(prompt, output_model=None):
    """Use OpenRouter by default, fallback to OpenAI after 3 retries."""
    request_timeout = float(os.getenv("LLM_EXPENSIVE_REQUEST_TIMEOUT", os.getenv("LLM_REQUEST_TIMEOUT", "90")))
    for attempt in range(3):
        try:
            openrouter_expensive_client = create_openrouter_expensive_client()
            if output_model is not None:
                structured_model = openrouter_expensive_client.with_structured_output(output_model)
                response = await asyncio.wait_for(structured_model.ainvoke(prompt), request_timeout)
                # Try to validate/convert response to expected model
                if isinstance(response, output_model):
                    return response
                else:
                    # Try to parse/validate the response as the expected model
                    try:
                        return output_model.model_validate(response)
                    except Exception as validation_error:
                        raise ValueError(f"Invalid structured response: {validation_error}")
            else:
                response = await asyncio.wait_for(openrouter_expensive_client.ainvoke(prompt), request_timeout)
            return response
        except Exception as e:
            logger.warning(f"OpenRouter attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            logger.warning(f"Exception details: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
            if attempt == 2:
                logger.info("Falling back to OpenAI after 3 OpenRouter failures")
                # Only try OpenAI after all OpenRouter attempts failed
                try:
                    openai_expensive_client = create_openai_expensive_client()
                    if output_model is not None:
                        structured_model = openai_expensive_client.with_structured_output(output_model)
                        response = await asyncio.wait_for(structured_model.ainvoke(prompt), request_timeout)
                    else:
                        response = await asyncio.wait_for(openai_expensive_client.ainvoke(prompt), request_timeout)
                    return response
                except Exception as openai_error:
                    logger.error(f"OpenAI fallback also failed: {openai_error}")
                    raise