from __future__ import annotations

import json
import logging
from typing import Any

from django.contrib import messages
from django.utils.translation import gettext as _
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from functools import wraps
import asyncio
from django.contrib.auth import password_validation
from django.contrib import auth as django_auth
from asgiref.sync import sync_to_async
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.conf import settings
from django.utils.decorators import method_decorator
from django.utils import timezone
from django.utils import translation
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Count, Q
from django.core.exceptions import ValidationError, PermissionDenied
from decimal import Decimal, ROUND_DOWN

from rest_framework import viewsets

from user.models import (
    Game,
    GameCharacter,
    GameStory,
    GameChapter,
    GameTurn,
    User as ProfileUser,
    VisualStyleChoices,
    DifficultyChoices,
    UsageTracking,
    GameDeletionLog,
    TurnStatus,
    GameConstants,
)
from user.serializers import (
    GameSerializer,
    GameCharacterSerializer,
    GameStorySerializer,
    GameChapterSerializer,
    GameTurnSerializer,
)

from user import services as svc
from user.prompts import llm_story, llm_plan_chapters, llm_next
from user.tasks import process_turn_task, create_game_task


# ----------------------------
# Helper utilities
# ----------------------------


def custom_404(request, exception=None):
    """Custom 404 handler that doesn't expose URLs."""
    prof = None
    if request.user.is_authenticated:
        prof = _ensure_profile_for_auth(request.user)
    return render(request, '404.html', {'prof': prof}, status=404)



def _ensure_profile_for_auth(auth_user) -> ProfileUser:
    """Map django auth user to domain `user.User` by email; create if missing."""
    email = (getattr(auth_user, "email", "") or "").strip().lower()
    name = getattr(auth_user, "username", "") or (email.split("@")[0] if email else "Player")
    prof = ProfileUser.objects.filter(email=email).first()
    if prof:
        # Enforce end-of-period downgrade for canceled/non-renewed subscriptions without schema changes
        try:
            _maybe_auto_downgrade_if_cycle_expired(prof)
        except Exception:
            pass
        return prof
    # Create minimal profile
    return ProfileUser.objects.create(name=name[:15] or "Player", email=email or f"{name}@local")


async def _a_ensure_profile_for_auth(auth_user) -> ProfileUser:
    """Async-safe: evaluate lazy auth user and ORM inside a thread."""

    def _sync_get_or_create():
        # Accessing request.user may hit the DB; do it in a thread
        email = (getattr(auth_user, "email", "") or "").strip().lower()
        name = getattr(auth_user, "username", "") or (email.split("@")[0] if email else "Player")
        prof = ProfileUser.objects.filter(email=email).first()
        if prof:
            # Enforce end-of-period downgrade for canceled/non-renewed subscriptions without schema changes
            try:
                _maybe_auto_downgrade_if_cycle_expired(prof)
            except Exception:
                pass
            return prof
        created = ProfileUser.objects.create(name=name[:15] or "Player", email=email or f"{name}@local")
        try:
            _maybe_auto_downgrade_if_cycle_expired(created)
        except Exception:
            pass
        return created

    return await asyncio.to_thread(_sync_get_or_create)
# ----------------------------
# Helper for usage data
# ----------------------------

def _maybe_auto_downgrade_if_cycle_expired(prof: ProfileUser) -> None:
    """Downgrade paid plans to FREE when the billing cycle has elapsed without renewal.
    Assumes monthly billing; uses `plan_cycle_started_at + 30 days` as the period end.
    Renewal events (transaction.completed / subscription.activated/updated) reset cycle start.
    """
    try:
        # Only applicable to paid plans
        if prof.subscription_plan in {
            ProfileUser.SubscriptionPlan.LITE,
            ProfileUser.SubscriptionPlan.PLUS,
            ProfileUser.SubscriptionPlan.PRO,
        }:
            start = getattr(prof, "plan_cycle_started_at", None)
            if not start:
                return
            end = start + timezone.timedelta(days=30)
            if timezone.now() >= end:
                prof.subscription_plan = ProfileUser.SubscriptionPlan.FREE
                prof.plan_cycle_started_at = None
                prof.save(update_fields=["subscription_plan", "plan_cycle_started_at"])
    except Exception:
        # Fail-open: never block the request due to billing checks
        pass








# ----------------------------
# Helper for usage data
# ----------------------------

def _get_usage_context(prof: ProfileUser) -> dict:
    """Get usage data for templates."""
    try:
        effective_plan = prof.get_effective_plan_for_limits()
    except Exception:
        effective_plan = prof.subscription_plan
    limits = _plan_limits(effective_plan)
    # Use cycle-based window for paid plans; monthly for free
    try:
        usage_stats = prof.current_cycle_usage()
    except Exception:
        usage_stats = prof.current_month_usage()
    # Keep precision with Decimal to avoid float jitter in display
    audio_used_val = usage_stats.get('total_audio_seconds', 0) or Decimal('0')
    audio_used = audio_used_val if isinstance(audio_used_val, Decimal) else Decimal(str(audio_used_val))
    plan_limit_val = limits.get('monthly_audio_seconds', 0) or 0
    audio_limit = Decimal(str(plan_limit_val))
    audio_percentage = float((audio_used / audio_limit * Decimal('100'))) if audio_limit > 0 else 0.0
    
    return {
        # Raw seconds (floats) if needed elsewhere
        "audio_used": float(audio_used),
        "audio_limit": float(audio_limit),
        # Percentage for progress bar
        "audio_percentage": min(audio_percentage, 100.0),
        # Display hours rounded down to 1 decimal to avoid decreasing values
        "audio_hours_used": float((audio_used / Decimal('3600')).quantize(Decimal('0.0'), rounding=ROUND_DOWN)),
        "audio_hours_limit": float((audio_limit / Decimal('3600')).quantize(Decimal('0.0'), rounding=ROUND_DOWN)),
    }


def _get_deletion_window_start(prof: ProfileUser):
    """Deletion window aligns with usage window: monthly for free, billing cycle for paid."""
    try:
        return prof.get_usage_window_start()
    except Exception:
        now = timezone.now()
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _plan_max_games(plan: str) -> int:
    game_limits = {
        ProfileUser.SubscriptionPlan.LITE: 2,
        ProfileUser.SubscriptionPlan.PLUS: 5,
        ProfileUser.SubscriptionPlan.PRO: 20,
        ProfileUser.SubscriptionPlan.FREE: 1,
    }
    return game_limits.get(plan, 0)


def _get_deletion_context(prof: ProfileUser) -> dict:
    """Return deletion stats for confirmation UI and enforcement logic."""
    # Use effective plan to prevent toggling to FREE from resetting deletion caps this month
    try:
        effective_plan = prof.get_effective_plan_for_limits()
    except Exception:
        effective_plan = prof.subscription_plan
    max_games = _plan_max_games(effective_plan)
    if max_games <= 0:
        return {"deletion_allowed": 0, "deletion_used": 0, "deletion_remaining": 0}
    allowed = max_games * 4
    window_start = _get_deletion_window_start(prof)
    used = GameDeletionLog.objects.filter(user=prof, created_at__gte=window_start).count()
    remaining = max(allowed - used, 0)
    return {
        "deletion_allowed": allowed,
        "deletion_used": used,
        "deletion_remaining": remaining,
    }

# ----------------------------
# Plan/rate limits
# ----------------------------

def _plan_limits(subscription_plan: str) -> dict[str, int | float]:
    """Return simple per-plan limits.
    monthly_turns: hard cap per calendar month
    per_minute_turns: soft rate limit per minute
    monthly_audio_seconds: monthly audio generation limit in seconds
    """
    plan = (subscription_plan or "").lower()
    if plan == ProfileUser.SubscriptionPlan.PLUS:
        return {
            "monthly_turns": 1200, 
            "per_minute_turns": 20,
            "monthly_audio_seconds": 108000.0  # 30 hours
        }
    if plan == ProfileUser.SubscriptionPlan.PRO:
        return {
            "monthly_turns": 5000, 
            "per_minute_turns": 60,
            "monthly_audio_seconds": 324000.0  # 90 hours
        }
    if plan == ProfileUser.SubscriptionPlan.LITE:
        return {
            "monthly_turns": 300, 
            "per_minute_turns": 10,
            "monthly_audio_seconds": 43200.0  # 12 hours
        }
    # FREE default
    return {
        "monthly_turns": 0, 
        "per_minute_turns": 0,
        "monthly_audio_seconds": 14400.0  # 4 hours
    }


# ----------------------------
# Pages
# ----------------------------

@require_GET
def landing_page(request: HttpRequest) -> HttpResponse:
    return render(request, "landing_page.html")
@require_GET
def pricing_page(request: HttpRequest) -> HttpResponse:
    context = {}
    if request.user.is_authenticated:
        prof = _ensure_profile_for_auth(request.user)
        context['current_plan'] = prof.subscription_plan
        context['prof'] = prof
    return render(request, "pricing.html", context)

@require_GET
def privacy_page(request: HttpRequest) -> HttpResponse:
    return render(request, "privacy.html")

@require_GET
def terms_page(request: HttpRequest) -> HttpResponse:
    return render(request, "terms.html")

@require_GET
def support_page(request: HttpRequest) -> HttpResponse:
    return render(request, "support.html")

@require_GET
def updates_page(request: HttpRequest) -> HttpResponse:
    # If moved updates.md to a safe location, load and display
    updates_path = settings.BASE_DIR.parent.parent / "updates.md"
    text = ""
    try:
        text = updates_path.read_text(encoding="utf-8")
    except Exception:
        text = ""
    return render(request, "updates.html", {"updates": text})


# demo removed


def login_view(request: HttpRequest) -> HttpResponse:
    # If already logged in, redirect based on plan
    if request.user.is_authenticated:
        prof = _ensure_profile_for_auth(request.user)
        # If user has a game, go to latest game, else go to new game
        latest = Game.objects.filter(user=prof).order_by("-last_played_at", "-created_at").first()
        if latest:
            return redirect("in_game", game_id=str(latest.id))
        return redirect("new_game")
    # Clear any stale messages on fresh GET to avoid showing unrelated notices
    if request.method == "GET":
        list(messages.get_messages(request))
    if request.method == "POST":
        identifier = request.POST.get("username") or request.POST.get("email") or request.POST.get("username_or_email") or ""
        password = request.POST.get("password") or ""
        user = authenticate(request, username=identifier, password=password)
        if not user:
            # try resolve by email
            from django.contrib.auth import get_user_model

            AU = get_user_model()
            candidate = AU.objects.filter(email__iexact=identifier).first()
            if candidate:
                user = authenticate(request, username=candidate.username, password=password)
        if user:
            login(request, user)
            prof = _ensure_profile_for_auth(user)
            # Prefer current URL language for consistency; fall back to user's preference, then 'en'
            try:
                lang = getattr(request, 'LANGUAGE_CODE', None) or getattr(prof, 'preferred_language', None) or 'en'
                translation.activate(lang)
                request.session[translation.LANGUAGE_SESSION_KEY] = lang
                # Persist user's preference to current active language for future sessions
                try:
                    if hasattr(prof, 'preferred_language') and prof.preferred_language != lang:
                        prof.preferred_language = lang
                        prof.save(update_fields=['preferred_language'])
                except Exception:
                    pass
            except Exception:
                pass
            latest = Game.objects.filter(user=prof).order_by("-last_played_at", "-created_at").first()
            if latest:
                return redirect("in_game", game_id=str(latest.id))
            return redirect("new_game")
        messages.error(request, _("Invalid credentials."))
    return render(request, "login.html")


logger = logging.getLogger(__name__)


def api_login_required(view_func):
    """Custom login_required decorator for API endpoints that returns JSON instead of redirecting.
    Async-aware: awaits async views, calls sync views directly.
    """
    if asyncio.iscoroutinefunction(view_func):
        @wraps(view_func)
        async def wrapped_view(request, *args, **kwargs):
            # Resolve user safely in async context
            try:
                user = await sync_to_async(django_auth.get_user)(request)
            except Exception:
                user = None
            if not getattr(user, "is_authenticated", False):
                return JsonResponse({"ok": False, "error": _("Authentication required")}, status=401)
            # Ensure request.user is populated for downstream usage
            try:
                request.user = user  # type: ignore
            except Exception:
                pass
            return await view_func(request, *args, **kwargs)
        return wrapped_view
    else:
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({"ok": False, "error": _("Authentication required")}, status=401)
            return view_func(request, *args, **kwargs)
        return wrapped_view


def require_http_methods_async(methods):
    """Async-aware HTTP method decorator for function-based views."""
    def decorator(view_func):
        if asyncio.iscoroutinefunction(view_func):
            @wraps(view_func)
            async def _wrapped(request, *args, **kwargs):
                if request.method not in methods:
                    from django.http import HttpResponseNotAllowed
                    return HttpResponseNotAllowed(methods)
                return await view_func(request, *args, **kwargs)
            return _wrapped
        # Fallback to Django's sync decorator
        return require_http_methods(methods)(view_func)
    return decorator

require_post_async = require_http_methods_async({"POST"})
require_post_or_delete_async = require_http_methods_async({"POST", "DELETE"})
require_get_async = require_http_methods_async({"GET"})


def register_view(request: HttpRequest) -> HttpResponse:
    # If already logged in, redirect based on plan
    if request.user.is_authenticated:
        prof = _ensure_profile_for_auth(request.user)
        latest = Game.objects.filter(user=prof).order_by("-last_played_at", "-created_at").first()
        if latest:
            return redirect("in_game", game_id=str(latest.id))
        return redirect("new_game")
    if request.method == "POST":
        email = (request.POST.get("email") or "").strip()
        password = request.POST.get("password") or ""
        confirm = request.POST.get("confirmPassword") or request.POST.get("password2") or ""
        accept_terms = request.POST.get("acceptTerms") in {"on", "true", "1"}
        if not email:
            messages.error(request, _("Email is required."))
            return render(request, "register.html")
        if password != confirm:
            messages.error(request, _("Passwords do not match."))
            return render(request, "register.html")
        if not accept_terms:
            messages.error(request, _("You must accept Terms and Privacy Policy."))
            return render(request, "register.html")
        from django.contrib.auth import get_user_model

        AU = get_user_model()
        # Prevent duplicate emails
        if AU.objects.filter(email__iexact=email).exists():
            messages.error(request, _("An account with this email already exists. Please sign in."))
            return render(request, "register.html")
        username = email.split("@")[0]
        if AU.objects.filter(username=username).exists():
            # ensure unique username
            from uuid import uuid4

            username = f"{username}-{uuid4().hex[:6]}"
        # Validate password against Django validators
        try:
            password_validation.validate_password(password)
        except Exception as e:
            messages.error(request, _("Weak password: %(error)s") % {"error": e})
            return render(request, "register.html")
        try:
            user = AU.objects.create_user(username=username, email=email, password=password)
            # Ensure an auth backend is set before login (multiple backends configured)
            auth_user = authenticate(request, username=username, password=password) or user
            # create or sync profile; default new users to FREE
            ProfileUser.objects.get_or_create(
                email=email,
                defaults={"name": (email.split("@")[0])[:15], "subscription_plan": ProfileUser.SubscriptionPlan.FREE},
            )
            if getattr(auth_user, "backend", None):
                login(request, auth_user)
            else:
                # Fall back to explicit backend if authenticate didn't set it
                login(request, auth_user, backend='django.contrib.auth.backends.ModelBackend')
            # Redirect to latest game if exists, else to new game
            prof = _ensure_profile_for_auth(auth_user)
            latest = Game.objects.filter(user=prof).order_by("-last_played_at", "-created_at").first()
            if latest:
                return redirect("in_game", game_id=str(latest.id))
            return redirect("new_game")
        except Exception:
            logger.exception("Registration failed for email=%s", email)
            messages.error(request, _("Registration failed. Please try again."))
            return render(request, "register.html")
    return render(request, "register.html")


@login_required
@require_POST
def upgrade_plan(request: HttpRequest, plan: str) -> HttpResponse:
    # Disable direct mutations to avoid limit-reset abuse; redirect to billing checkout
    plan = (plan or "").lower()
    valid_paid = {ProfileUser.SubscriptionPlan.LITE, ProfileUser.SubscriptionPlan.PLUS, ProfileUser.SubscriptionPlan.PRO}
    if plan not in valid_paid:
        messages.error(request, _("Invalid plan."))
        return redirect("pricing")
    # Delegate to billing checkout which will create Paddle session; actual plan updates occur via webhook
    return redirect("billing_checkout", plan=plan)


@login_required
@require_POST
async def billing_checkout(request: HttpRequest, plan: str) -> JsonResponse | HttpResponse:
    """Create a Paddle checkout session/link for the requested plan.
    Returns JSON with {url} to redirect the user.
    """
    plan = (plan or "").lower()
    if plan not in {ProfileUser.SubscriptionPlan.LITE, ProfileUser.SubscriptionPlan.PLUS, ProfileUser.SubscriptionPlan.PRO}:
        return JsonResponse({"ok": False, "error": _("Invalid plan")}, status=400)

    # Ensure billing is configured
    if not getattr(settings, 'PADDLE_API_KEY', '').strip():
        return JsonResponse({"ok": False, "error": _("Billing is not configured")}, status=503)

    # Select price id
    if plan == ProfileUser.SubscriptionPlan.LITE:
        price_id = settings.PADDLE_PRICE_ID_LITE
    elif plan == ProfileUser.SubscriptionPlan.PLUS:
        price_id = settings.PADDLE_PRICE_ID_PLUS
    else:  # PRO
        price_id = getattr(settings, 'PADDLE_PRICE_ID_PRO', None)
    
    if not price_id:
        return JsonResponse({"ok": False, "error": _("Price ID not configured")}, status=500)

    # Build Paddle request using official SDK
    prof = await _a_ensure_profile_for_auth(request.user)
    try:
        from paddle_billing import Client, Environment, Options
        api_key = settings.PADDLE_API_KEY
        env = (settings.PADDLE_ENV or "sandbox").lower()
        options = Options(Environment.SANDBOX) if env == "sandbox" else Options(Environment.LIVE)
        paddle = Client(api_key, options=options)

        def _create_transaction():
            return paddle.transactions.create({
                "items": [
                    {"price_id": price_id, "quantity": 1},
                ],
                "customer": {"email": (getattr(request.user, "email", "") or prof.email) or ""},
                "success_url": settings.PADDLE_RETURN_URL,
                "cancel_url": settings.PADDLE_CANCEL_URL,
            })

        tx = await asyncio.to_thread(_create_transaction)
        # Try common shapes: object with attribute, SDK model with .data.checkout_url, or dict
        checkout_url = None
        if hasattr(tx, "checkout_url") and isinstance(getattr(tx, "checkout_url"), str):
            checkout_url = tx.checkout_url
        elif hasattr(tx, "data") and isinstance(getattr(tx, "data"), object) and hasattr(tx.data, "checkout_url"):
            checkout_url = tx.data.checkout_url
        elif isinstance(tx, dict):
            checkout_url = (
                tx.get("checkout_url")
                or (tx.get("data") or {}).get("checkout_url")
                or (tx.get("attributes") or {}).get("checkout_url")
            )
        if not checkout_url:
            return JsonResponse({"ok": False, "error": _("Checkout URL missing")}, status=502)
        return JsonResponse({"ok": True, "url": checkout_url})
    except ImportError:
        # Fallback to HTTP if SDK not installed
        import httpx
        api_base = "https://api-sandbox.paddle.com" if settings.PADDLE_ENV == "sandbox" else "https://api.paddle.com"
        headers = {
            "Authorization": f"Bearer {settings.PADDLE_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "items": [
                {"price_id": price_id, "quantity": 1},
            ],
            "customer": {
                "email": (getattr(request.user, "email", "") or prof.email) or "",
            },
            "success_url": settings.PADDLE_RETURN_URL,
            "cancel_url": settings.PADDLE_CANCEL_URL,
        }

        def _extract_checkout_url(data: dict) -> str | None:
            if not isinstance(data, dict):
                return None
            for key in ("url", "checkout_url"):
                val = data.get(key)
                if isinstance(val, str) and val.startswith("http"):
                    return val
            inner = data.get("data") or data.get("attributes")
            if isinstance(inner, dict):
                return _extract_checkout_url(inner)
            return None

        async with httpx.AsyncClient(timeout=20.0, http2=True) as client:
            resp = await client.post(f"{api_base}/checkout/sessions", json=payload, headers=headers)
            if resp.status_code == 404:
                resp = await client.post(f"{api_base}/v1/checkout", json=payload, headers=headers)
        content_type = (resp.headers.get("content-type", "") or "").split(";")[0].strip()
        data = resp.json() if content_type == "application/json" else {}
        checkout_url = _extract_checkout_url(data)
        if resp.status_code >= 400 or not checkout_url:
            err_msg = (
                (isinstance(data, dict) and (data.get("error", {}) or {}).get("message"))
                or (isinstance(data, dict) and data.get("message"))
                or "Checkout failed"
            )
            logger.warning("Paddle checkout create failed: status=%s body=%s", resp.status_code, data)
            return JsonResponse({"ok": False, "error": err_msg}, status=resp.status_code or 500)
        return JsonResponse({"ok": True, "url": checkout_url})
    try:
        async with httpx.AsyncClient(timeout=20.0, http2=True) as client:
            # Try modern endpoint first
            resp = await client.post(f"{api_base}/checkout/sessions", json=payload, headers=headers)
            if resp.status_code == 404:
                # Fallback to legacy path if enabled on account
                resp = await client.post(f"{api_base}/v1/checkout", json=payload, headers=headers)
        content_type = (resp.headers.get("content-type", "") or "").split(";")[0].strip()
        data = resp.json() if content_type == "application/json" else {}
        checkout_url = _extract_checkout_url(data)
        if resp.status_code >= 400 or not checkout_url:
            err_msg = (
                (isinstance(data, dict) and (data.get("error", {}) or {}).get("message"))
                or (isinstance(data, dict) and data.get("message"))
                or "Checkout failed"
            )
            logger.warning("Paddle checkout create failed: status=%s body=%s", resp.status_code, data)
            return JsonResponse({"ok": False, "error": err_msg}, status=resp.status_code or 500)
        return JsonResponse({"ok": True, "url": checkout_url})
    except Exception:
        logger.exception("Exception while creating Paddle checkout")
        return JsonResponse({"ok": False, "error": _("Could not create checkout")}, status=500)


@require_GET
@login_required
def billing_return(request: HttpRequest) -> HttpResponse:
    """Handle return from billing and redirect user to latest game or new game."""
    prof = _ensure_profile_for_auth(request.user)
    latest = Game.objects.filter(user=prof).order_by("-last_played_at", "-created_at").first()
    if latest:
        return redirect("in_game", game_id=str(latest.id))
    return redirect("new_game")


def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect("landing_page")


@login_required
def player_home(request: HttpRequest) -> HttpResponse:
    prof = _ensure_profile_for_auth(request.user)
    
    # Get game limits and usage
    max_games = _plan_max_games(prof.subscription_plan)
    
    # Get both active and archived games
    active_games = Game.objects.filter(user=prof, is_archived=False).order_by("-last_played_at")[:12]
    archived_games = Game.objects.filter(user=prof, is_archived=True).order_by("-last_played_at")[:12]
    active_count = Game.objects.filter(user=prof, is_archived=False).count()
    can_create_new = active_count < max_games
    
    # Combine all games for image fetching
    all_games_list = list(active_games) + list(archived_games)
    
    # Build a map: game_id -> last generated image url
    last_images: dict[str, str] = {}
    turns = (
        GameTurn.objects
        .filter(chapter__game__in=all_games_list, status="done")
        .only("generated_visual_url", "chapter__game_id", "completed_at")
        .order_by("-completed_at")
    )
    for t in turns:
        gid = str(t.chapter.game_id)
        if gid not in last_images and t.generated_visual_url:
            last_images[gid] = t.generated_visual_url
        if len(last_images) >= len(all_games_list):
            break
    
    # attach cover_url to each game instance for easier template usage
    for g in all_games_list:
        setattr(g, "cover_url", last_images.get(str(g.id), ""))
    
    context = {
        "games": active_games,
        "archived_games": archived_games,
        "prof": prof,
        "can_create_new": can_create_new,
        "max_games": max_games,
        "active_count": active_count,
    }
    context.update(_get_usage_context(prof))
    context.update(_get_deletion_context(prof))
    return render(request, "player_home.html", context)


@login_required
def account_settings(request: HttpRequest) -> HttpResponse:
    prof = _ensure_profile_for_auth(request.user)
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "update_profile":
            # No editable nickname; only allow theme etc. handled elsewhere
            messages.success(request, _("Profile updated successfully!"))
            return redirect("account_settings")
        
        elif action == "delete_account":
            confirm_delete = request.POST.get("confirm_delete", "").strip()
            if confirm_delete == "DELETE":
                # Guard: prevent deletion if subscription payment is still active (not canceled)
                if prof.has_paid_plan and not getattr(prof, 'billing_is_canceled', False):
                    messages.error(request, _("Account deletion is blocked while your subscription is active. Please cancel your subscription first."))
                    return redirect("account_settings")
                # Delete all user's games and related data
                Game.objects.filter(user=prof).delete()
                # Delete the user profile
                user = request.user
                prof.delete()
                # Delete Django user account
                user.delete()
                messages.success(request, _("Your account has been deleted."))
                return redirect("landing_page")
            else:
                messages.error(request, _("Please type DELETE to confirm account deletion."))
            return redirect("account_settings")
    
    context = {"prof": prof}
    context.update(_get_usage_context(prof))
    return render(request, "account_settings.html", context)


@login_required
@require_GET
def all_games(request: HttpRequest) -> HttpResponse:
    """Paginated list of all games for the current user.
    Includes lightweight cover resolution similar to player_home.
    """
    prof = _ensure_profile_for_auth(request.user)
    
    # Get game limits
    max_games = _plan_max_games(prof.subscription_plan)
    active_count = Game.objects.filter(user=prof, is_archived=False).count()
    can_create_new = active_count < max_games

    # Sorting & basic search (optional, safe defaults)
    sort = request.GET.get("sort") or "-last_played_at"
    if sort not in {"-last_played_at", "created_at", "-created_at"}:
        sort = "-last_played_at"
    q = (request.GET.get("q") or "").strip()
    
    # Include archived filter
    show_archived = request.GET.get("archived", "false").lower() == "true"

    qs = Game.objects.filter(user=prof)
    if q:
        from django.db.models import Q
        qs = qs.filter(Q(main_character_name__icontains=q) | Q(genre_name__icontains=q))
    qs = qs.order_by(sort)

    # Pagination
    page_size = 24
    paginator = Paginator(qs, page_size)
    page = request.GET.get("page") or 1
    try:
        page_obj = paginator.page(page)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    games = list(page_obj.object_list)

    # Resolve last image per game for the current page
    last_images: dict[str, str] = {}
    turns = (
        GameTurn.objects
        .filter(chapter__game__in=games, status="done")
        .only("generated_visual_url", "chapter__game_id", "completed_at")
        .order_by("-completed_at")
    )
    for t in turns:
        gid = str(t.chapter.game_id)
        if gid not in last_images and t.generated_visual_url:
            last_images[gid] = t.generated_visual_url
        if len(last_images) >= len(games):
            break
    for g in games:
        setattr(g, "cover_url", last_images.get(str(g.id), ""))

    # Separate active and archived games
    active_games = [g for g in games if not g.is_archived]
    archived_games = [g for g in games if g.is_archived]

    # compute total and plan limit (for informational display)
    total_games = Game.objects.filter(user=prof, is_archived=False).count()
    total_archived = Game.objects.filter(user=prof, is_archived=True).count()

    context = {
        "prof": prof,
        "games": active_games,
        "archived_games": archived_games,
        "page_obj": page_obj,
        "total_games": total_games,
        "total_archived": total_archived,
        "plan_limit": max_games,
        "can_create_new": can_create_new,
        "q": q,
        "sort": sort,
    }
    context.update(_get_usage_context(prof))
    context.update(_get_deletion_context(prof))
    return render(request, "all_games.html", context)


@login_required
def new_game(request: HttpRequest) -> HttpResponse:
    if request.method == "GET":
        prof = _ensure_profile_for_auth(request.user)
        # Clear any stale create-game task from session to force a fresh enqueue
        try:
            request.session.pop("create_game_task_id", None)
            request.session.pop("create_game_task_time", None)
        except Exception:
            pass
        context = {"prof": prof}
        context.update(_get_usage_context(prof))
        return render(request, "new_game.html", context)
    # Fallback: accept standard form POST
    return api_create_game(request)


@login_required
def in_game(request: HttpRequest, game_id) -> HttpResponse:
    prof = _ensure_profile_for_auth(request.user)
    # Keep this view synchronous; avoid await usage
    try:
        game = Game.objects.get(pk=game_id, user_id=prof.id)
    except Game.DoesNotExist:
        return redirect('all_games')
    # Block entering gameplay page for archived games
    if getattr(game, "is_archived", False):
        messages.warning(request, _("This game is archived and cannot be played."))
        return redirect('all_games')
    
    # Check if game has any turns (done or pending)
    any_turns = GameTurn.objects.filter(chapter__game=game).exists()
    
    # If no turns exist at all, the game creation might still be in progress
    # or there was an error during creation
    if not any_turns:
        # Check if game has chapters (required for turns)
        has_chapters = GameChapter.objects.filter(game=game).exists()
        if not has_chapters:
            # Game creation is likely still in progress or failed
            messages.warning(request, _("Game is still being created. Please wait a moment and refresh the page."))
            return redirect('all_games')
    
    # Get only completed turns for display
    turns_qs = GameTurn.objects.filter(chapter__game=game, status="done").order_by("chapter__index", "index")
    
    # In DEBUG mode, if no completed turns yet, try to generate them synchronously
    if settings.DEBUG and not turns_qs.exists() and any_turns:
        # There are turns but they're not done yet - process them
        pending_turns = GameTurn.objects.filter(
            chapter__game=game, 
            status__in=["pending", "processing"]
        ).order_by("chapter__index", "index")[:2]  # Process first 2 turns only
        
        for turn in pending_turns:
            try:
                sfx = GameConstants.get_sound_effects()
                svc.process_turn_job(
                    turn_id=str(turn.id),
                    llm_next=llm_next,
                    tts_narration=svc.tts_narration,
                    tts_character=svc.tts_character,
                    tti_generate=svc.tti_generate,
                    sound_effects=sfx,
                    # chapter_advance_every is now determined inside process_turn_job based on game.story_length
                )
            except Exception as e:
                logger.warning(f"Failed to process turn {turn.id}: {e}")
        
        # Re-fetch completed turns
        turns_qs = GameTurn.objects.filter(chapter__game=game, status="done").order_by("chapter__index", "index")
    
    last_turn = (
        GameTurn.objects
        .filter(chapter__game=game, status="done")
        .order_by("-chapter__index", "-index")
        .first()
    )
    
    context = {
        "prof": prof,
        "game": game,
        "turns": list(turns_qs),
        "last_turn": last_turn,
        "use_sync_process": settings.DEBUG,
        "has_pending_turns": GameTurn.objects.filter(
            chapter__game=game, 
            status__in=["pending", "processing"]
        ).exists(),
    }
    context.update(_get_usage_context(prof))
    return render(request, "in_game.html", context)


# ----------------------------
# API endpoints (JSON)
# ----------------------------

@require_POST
@api_login_required
def api_create_game(request: HttpRequest) -> JsonResponse:
    from user.security import validate_request_data, validate_game_params, rate_limit_api
    
    # Check rate limit: 5 game creations per hour
    from django.core.cache import cache
    cache_key = f"api_rate_limit:{request.user.id}:create_game"
    current_count = cache.get(cache_key, 0)
    if current_count >= 5:
        logger.warning(f"Rate limit exceeded for user {request.user.id} on game creation")
        return JsonResponse({'error': _("Rate limit exceeded. Maximum 5 games per hour."), 'retry_after': 3600}, status=429)
    cache.set(cache_key, current_count + 1, 3600)
    try:
        data = validate_request_data(request)
    except Exception as e:
        logger.error(f"Failed to validate request data: {e}")
        return JsonResponse({"ok": False, "error": _("Invalid request data")}, status=400)
    
    # Validate and sanitize inputs
    try:
        cleaned = validate_game_params(data)
        genre = cleaned.get("genre", "mystery")
        main_character = cleaned.get("mainCharacter", "Alex")
        visual = cleaned.get("visualStyle", "realistic")
        difficulty = cleaned.get("difficulty", "normal")
        story_length = cleaned.get("storyLength", "standard")
        extra_requests = cleaned.get("extraRequests", "")
    except ValidationError as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    prof = _ensure_profile_for_auth(request.user)
    # Determine per-plan Celery time limits (hard and soft)
    try:
        paid_plans = {
            ProfileUser.SubscriptionPlan.LITE,
            ProfileUser.SubscriptionPlan.PLUS,
            ProfileUser.SubscriptionPlan.PRO,
        }
        if prof.subscription_plan == ProfileUser.SubscriptionPlan.FREE:
            hard_time_limit = 1200
        else:
            hard_time_limit = 600
        soft_time_limit = max(60, hard_time_limit - 60)
    except Exception:
        # Fallback to global defaults if anything goes wrong
        from django.conf import settings as _s
        hard_time_limit = getattr(_s, "CELERY_TASK_TIME_LIMIT", 1200)
        soft_time_limit = getattr(_s, "CELERY_TASK_SOFT_TIME_LIMIT", max(60, hard_time_limit - 60))
    # Prevent duplicate in-flight creations per session/user
    try:
        from celery.result import AsyncResult as _AR
        existing_task_id = request.session.get("create_game_task_id")
        if existing_task_id:
            ar = _AR(existing_task_id)
            st = ar.state
            
            # Check task age - consider stale close to the soft time limit to avoid duplicates
            task_created = request.session.get("create_game_task_time")
            if task_created:
                from datetime import datetime
                task_age = (datetime.now().timestamp() - task_created)
                stale_after = max(120, soft_time_limit - 60)
                if task_age > stale_after:
                    logger.warning(f"Task {existing_task_id} is {task_age:.0f}s old (state: {st}), clearing it")
                    ar.forget()
                    request.session.pop("create_game_task_id", None)
                    request.session.pop("create_game_task_time", None)
                    # Continue to create new task
                elif st == "PROGRESS" and isinstance(ar.info, dict):
                    # Check if task is stuck at finalizing
                    if ar.info.get("step") == "finalizing" and ar.info.get("pct") == 95:
                        # Task seems stuck, forget it and continue
                        logger.warning(f"Task {existing_task_id} appears stuck at finalizing, clearing it")
                        ar.forget()
                        request.session.pop("create_game_task_id", None)
                        request.session.pop("create_game_task_time", None)
                    else:
                        return JsonResponse({"ok": True, "task_id": existing_task_id})
                elif st in {"PENDING", "STARTED", "RETRY", "PROGRESS"}:
                    return JsonResponse({"ok": True, "task_id": existing_task_id})
                else:
                    # Task is in SUCCESS, FAILURE, or REVOKED state - cleanup
                    request.session.pop("create_game_task_id", None)
                    request.session.pop("create_game_task_time", None)
    except Exception as e:
        logger.error(f"Error checking existing task: {e}")
        # Clear session on error to prevent getting stuck
        request.session.pop("create_game_task_id", None)
        request.session.pop("create_game_task_time", None)

    # Get game limits based on effective plan
    try:
        effective_plan = prof.get_effective_plan_for_limits()
    except Exception:
        effective_plan = prof.subscription_plan
    game_limits = {
        ProfileUser.SubscriptionPlan.LITE: 2,
        ProfileUser.SubscriptionPlan.PLUS: 5,
        ProfileUser.SubscriptionPlan.PRO: 20,
        ProfileUser.SubscriptionPlan.FREE: 1
    }
    max_games = game_limits.get(effective_plan, 0)
    
    # Count only non-archived games
    active_games = Game.objects.filter(user=prof, is_archived=False).order_by('last_played_at')
    active_count = active_games.count()
    
    # Clean up any stuck games before checking limits
    from user.services import check_and_cleanup_stuck_games
    check_and_cleanup_stuck_games(timeout_minutes=10)
    
    # Refresh active count after cleanup
    active_count = Game.objects.filter(user=prof, is_archived=False, status__in=['ongoing', 'completed']).count()
    
    if active_count >= max_games and max_games > 0:
        # Auto-archive oldest games if over limit
        if active_count > max_games:
            games_to_archive = active_games[:active_count - max_games + 1]
            for game in games_to_archive:
                game.is_archived = True
                game.save(update_fields=['is_archived'])
            active_count = max_games - 1  # Account for the new game about to be created
        else:
            return JsonResponse({"ok": False, "error": _("Game limit reached (%(limit)s). Delete or upgrade for more games.") % {"limit": max_games}}, status=403)

    # Recent partial game guard (10 min window)
    try:
        cutoff = timezone.now() - timezone.timedelta(minutes=10)
        partial_exists = (
            Game.objects
            .filter(user=prof, created_at__gte=cutoff)
            .annotate(num_chapters=Count("chapters"), num_chars=Count("characters"))
            .filter(Q(story__isnull=True) | Q(num_chapters=0) | Q(num_chars=0))
            .exists()
        )
        if partial_exists:
            return JsonResponse({"ok": False, "error": _("A game is already being generated. Please wait for it to finish.")}, status=409)
    except Exception as e:
        logger.warning(f"Failed to check for partial games: {e}")

    # Enqueue background creation and return a task id for polling
    logger.info(f"Enqueueing game creation task for user {prof.id}")
    async_result = create_game_task.apply_async(
        args=[str(prof.id)],
        kwargs={
            "genre": genre,
            "main_character": main_character,
            "visual_style": visual,
            "difficulty": difficulty,
            "story_length": story_length,
            "extra_requests": extra_requests,
        },
        soft_time_limit=soft_time_limit,
        time_limit=hard_time_limit,
    )
    logger.info(f"Task enqueued with ID: {async_result.id}")
    # store task id and timestamp in session to block re-submits and track age
    try:
        from datetime import datetime
        request.session["create_game_task_id"] = async_result.id
        request.session["create_game_task_time"] = datetime.now().timestamp()
    except Exception as e:
        logger.warning(f"Failed to store task ID in session: {e}")
    return JsonResponse({"ok": True, "task_id": async_result.id})

@require_post_or_delete_async
@api_login_required
async def api_delete_game(request: HttpRequest, game_id) -> JsonResponse:
    """Delete a game with per-window deletion cap, and unarchive if under active-game limit."""
    try:
        from user.security import validate_ownership
        
        logger.info(f"Delete game request: game_id={game_id}, user={request.user.id}")
        
        prof = await _a_ensure_profile_for_auth(request.user)
        
        try:
            logger.debug(f"api_delete_game: fetching game {game_id}")
            game = await asyncio.to_thread(lambda: Game.objects.select_related("user").get(pk=game_id))
            validate_ownership(prof, game, field='user')
            logger.debug(f"api_delete_game: fetched game {game_id} and validated ownership")
        except Game.DoesNotExist:
            logger.warning(f"Game not found: {game_id}")
            return JsonResponse({"ok": False, "error": _("Game not found")}, status=404)
        except PermissionDenied:
            logger.warning(f"Permission denied for user {request.user.id} on game {game_id}")
            return JsonResponse({"ok": False, "error": _("Permission denied")}, status=403)
        
        # Check if game is currently being created
        if game.status == 'creating':
            # Check if game creation has been stuck for more than 10 minutes
            time_since_creation = timezone.now() - game.created_at
            if time_since_creation.total_seconds() > 600:  # 10 minutes
                logger.warning(f"Game {game_id} has been creating for {time_since_creation.total_seconds()/60:.1f} minutes - considering it failed")
                
                # Clean up Redis locks and task state for stuck game
                try:
                    from redis import asyncio as aioredis
                    redis_client = aioredis.from_url(settings.REDIS_URL)
                    lock_keys = [
                        f"game_creation_lock:{game_id}",
                        f"game_creation_task:{game_id}",
                        f"game_creation_status:{game_id}",
                    ]
                    for key in lock_keys:
                        try:
                            await redis_client.delete(key)
                        except Exception:
                            pass
                    try:
                        task_keys = await redis_client.keys(f"celery-task-meta-*{game_id}*")
                        if task_keys:
                            await redis_client.delete(*task_keys)
                    except Exception:
                        pass
                    logger.info(f"Cleaned up Redis locks and task state for stuck game {game_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up Redis state for game {game_id}: {e}", exc_info=True)
                
                # Update game status to failed before deletion
                game.status = 'failed'
                await asyncio.to_thread(lambda: game.save(update_fields=['status']))
                
            else:
                logger.warning(f"Cannot delete game {game_id} - currently being created (started {time_since_creation.total_seconds():.0f}s ago)")
                return JsonResponse({"ok": False, "error": _("Game is currently being created. Please wait a moment and try again.")}, status=409)
        
        # Enforce deletion cap: 4x of plan game limit per window
        try:
            effective_plan = prof.get_effective_plan_for_limits()
        except Exception:
            effective_plan = prof.subscription_plan
        max_games = _plan_max_games(effective_plan)
        if max_games > 0:
            allowed = max_games * 4
            window_start = _get_deletion_window_start(prof)
            used = await asyncio.to_thread(lambda: GameDeletionLog.objects.filter(user=prof, created_at__gte=window_start).count())
            if used >= allowed:
                return JsonResponse({"ok": False, "error": _("Deletion limit reached (%(used)s/%(allowed)s).") % {"used": used, "allowed": allowed}}, status=429)

        # Delete the game (offload to thread; ORM is sync). Use primary key to avoid stale instance issues.
        logger.debug(f"api_delete_game: deleting game {game_id}")
        await asyncio.to_thread(lambda: Game.objects.filter(pk=game_id).delete())
        logger.info(f"Game {game_id} deleted successfully")

        # Log deletion for cap accounting
        try:
            await asyncio.to_thread(lambda: GameDeletionLog.objects.create(user=prof, deleted_game_id=game.id))
        except Exception:
            logger.warning("Failed to log game deletion", exc_info=True)
        
        # Check if we should unarchive a game
        try:
            effective_plan = prof.get_effective_plan_for_limits()
        except Exception:
            effective_plan = prof.subscription_plan
        max_games = _plan_max_games(effective_plan)
        
        # Count active games after deletion
        logger.debug(f"api_delete_game: counting active games for user {prof.id}")
        active_count = await asyncio.to_thread(lambda: Game.objects.filter(user_id=prof.id, is_archived=False).count())
        logger.debug(f"api_delete_game: active_count={active_count}, max_games={max_games}")
        
        # If below limit and have archived games, unarchive the most recent one
        if active_count < max_games:
            logger.debug("api_delete_game: looking for archived game to unarchive")
            archived_game = await asyncio.to_thread(lambda: Game.objects.filter(
                user_id=prof.id,
                is_archived=True
            ).order_by('-last_played_at').first())
            
            if archived_game:
                archived_game.is_archived = False
                await asyncio.to_thread(lambda: archived_game.save(update_fields=['is_archived']))
                logger.info(f"Unarchived game {archived_game.id} after deletion")
                return JsonResponse({
                    "ok": True, 
                    "message": _("Game deleted successfully"),
                    "unarchived_game": str(archived_game.id)
                })
        
        return JsonResponse({"ok": True, "message": _("Game deleted successfully")})
    except Exception as e:
        logger.error(f"Unexpected error in api_delete_game: {str(e)}", exc_info=True)
        if settings.DEBUG:
            return JsonResponse({"ok": False, "error": _("Internal error"), "debug": str(e)}, status=500)
        return JsonResponse({"ok": False, "error": _("An unexpected error occurred")}, status=500)

@require_GET
@api_login_required
def api_create_game_status(request: HttpRequest) -> JsonResponse:
    from celery.result import AsyncResult
    task_id = request.GET.get("task_id") or ""
    if not task_id:
        return JsonResponse({"ok": False, "error": _("task_id required")}, status=400)
    res = AsyncResult(task_id)
    meta = res.info if isinstance(res.info, dict) else {}
    state = res.state
    out = {"ok": True, "state": state, "progress": meta}
    
    # Log for debugging
    logger.info(f"Task {task_id} state: {state}, meta: {meta}")
    
    # Include generic error message for frontend
    if state == "FAILURE":
        # Never expose internal error details to frontend
        out["error"] = _("Game creation failed. Please try again.")
        # Log the actual error for debugging
        logger.error(f"Game creation task {task_id} failed: {res.info}")
    
    if state == "SUCCESS" and isinstance(meta, dict) and meta.get("ok"):
        # Gate success on game readiness: require story and chapters to exist
        gid = meta.get("game_id")
        try:
            game = Game.objects.get(pk=gid)
            has_story = GameStory.objects.filter(game=game).exists()
            has_chapters = GameChapter.objects.filter(game=game).exists()
            if has_story and has_chapters:
                out["game_id"] = gid
                out["redirect_url"] = redirect("in_game", game_id=gid).url
            else:
                # Not fully ready, present as PROGRESS to the client
                state = out["state"] = "PROGRESS"
                out["progress"] = {"step": "finalizing", "pct": 95}
        except Exception:
            # If any lookup fails, keep polling
            state = out["state"] = "PROGRESS"
            out["progress"] = {"step": "finalizing", "pct": 95}
        # clear session task id and time if matches
        try:
            if request.session.get("create_game_task_id") == task_id:
                request.session.pop("create_game_task_id", None)
                request.session.pop("create_game_task_time", None)
        except Exception as e:
            logger.debug(f"Failed to clear task from session: {e}")
    elif state in {"FAILURE", "REVOKED"}:
        try:
            if request.session.get("create_game_task_id") == task_id:
                request.session.pop("create_game_task_id", None)
                request.session.pop("create_game_task_time", None)
        except Exception as e:
            logger.debug(f"Failed to clear task from session: {e}")
    return JsonResponse(out)


@require_post_async
@api_login_required
async def api_generate_turn(request: HttpRequest, game_id) -> JsonResponse:
    try:
        from user.security import validate_ownership, validate_json_request, sanitize_user_input
        
        # Check rate limit: 15 turns per minute using proper sliding window
        from django.core.cache import cache
        import time
        
        cache_key = f"api_rate_limit:{request.user.id}:generate_turn"
        window_size = 60
        limit = 15
        current_time = int(time.time())
        window_start = current_time - window_size
        requests_data = cache.get(cache_key, [])
        requests_in_window = [req_time for req_time in requests_data if req_time > window_start]
        if len(requests_in_window) >= limit:
            oldest_request = min(requests_in_window)
            retry_after = oldest_request + window_size - current_time
            logger.warning(f"Rate limit exceeded for user {request.user.id} on turn generation")
            return JsonResponse({'error': _("Rate limit exceeded"),'retry_after': max(retry_after, 1)}, status=429)
        requests_in_window.append(current_time)
        cache.set(cache_key, requests_in_window, window_size)
        
        prof = await _a_ensure_profile_for_auth(request.user)
        try:
            game = await asyncio.to_thread(lambda: Game.objects.select_related("user").get(pk=game_id))
        except Game.DoesNotExist:
            return JsonResponse({"ok": False, "error": _("Game not found")}, status=404)
        
        try:
            await asyncio.to_thread(validate_ownership, prof, game, 'user')
        except PermissionDenied:
            from user.security import log_suspicious_activity
            log_suspicious_activity(request, f"Unauthorized turn generation attempt for game {game_id}")
            return JsonResponse({"ok": False, "error": _("Permission denied")}, status=403)
        
        try:
            if request.content_type == 'application/json':
                payload = validate_json_request(request)
            else:
                payload = request.POST
        except Exception as e:
            logger.warning(f"Failed to parse request payload for game {game_id}: {e}")
            return JsonResponse({"ok": False, "error": _("Invalid request format")}, status=400)
        
        if not payload or 'user_text' not in payload:
            logger.warning(f"Missing required fields in request for game {game_id}: {payload}")
            return JsonResponse({"ok": False, "error": _("Missing required fields")}, status=400)
        
        user_text = sanitize_user_input(payload.get("user_text", ""), max_length=1000)
        is_choice = str(payload.get("is_choice") or "0").lower() in {"1", "true", "yes", "on"}
        if not user_text.strip():
            logger.warning(f"Empty user_text received from user {request.user.id} for game {game_id}")
            return JsonResponse({"ok": False, "error": _("Cannot process empty message")}, status=400)
        logger.info(f"api_generate_turn received: user_text='{user_text}', is_choice={is_choice}")

        # Block playing archived games
        if getattr(game, "is_archived", False):
            return JsonResponse({"ok": False, "error": _("This game is archived and cannot be played.")}, status=403)

        # Enforce plan limits and audio time limit similar to api_reserve_turn
        try:
            effective_plan = prof.get_effective_plan_for_limits()
        except Exception:
            effective_plan = prof.subscription_plan
        limits = _plan_limits(effective_plan)
        now = timezone.now()
        try:
            window_start = prof.get_usage_window_start()
        except Exception:
            window_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        minute_ago = now - timezone.timedelta(seconds=60)
        user_game_ids = await asyncio.to_thread(lambda: list(Game.objects.filter(user=prof).values_list("id", flat=True)))

        # Monthly turns cap (only when limit > 0)
        monthly_used = await asyncio.to_thread(lambda: GameTurn.objects.filter(
            chapter__game_id__in=user_game_ids,
            reserved_at__gte=window_start,
        ).count())
        if (limits["monthly_turns"] or 0) > 0 and monthly_used >= limits["monthly_turns"]:
            return JsonResponse({"ok": False, "error": _("Monthly turn limit reached for your plan.")}, status=429)

        # Per-minute rate per plan (only when limit > 0)
        minute_used = await asyncio.to_thread(lambda: GameTurn.objects.filter(
            chapter__game_id__in=user_game_ids,
            reserved_at__gte=minute_ago,
        ).count())
        if (limits["per_minute_turns"] or 0) > 0 and minute_used >= limits["per_minute_turns"]:
            return JsonResponse({"ok": False, "error": _("Too many requests. Please wait a moment.")}, status=429)

        # Monthly audio seconds cap (time limit)
        from django.db.models import Sum
        monthly_audio = await asyncio.to_thread(lambda: (
            UsageTracking.objects.filter(user=prof, created_at__gte=window_start).aggregate(total=Sum('audio_duration_seconds'))['total'] or 0
        ))
        if float(monthly_audio) >= limits["monthly_audio_seconds"]:
            try:
                if prof.subscription_plan == ProfileUser.SubscriptionPlan.FREE:
                    request.session["show_pricing_cta"] = True
            except Exception:
                pass
            return JsonResponse({"ok": False, "error": _("Monthly audio generation limit reached for your plan.")}, status=429)

        sfx = await asyncio.to_thread(GameConstants.get_sound_effects)
        # Cooldown: enforce wait based on last media duration (paid) or 20s for FREE
        try:
            last_done = await asyncio.to_thread(lambda: (
                GameTurn.objects
                .filter(chapter__game_id=game.id, status="done")
                .only("audio_duration_seconds", "completed_at")
                .order_by("-completed_at")
                .first()
            ))
            if last_done and last_done.completed_at:
                plan = prof.subscription_plan
                required_secs = 20.0 if plan == ProfileUser.SubscriptionPlan.FREE else float(last_done.audio_duration_seconds or 0) * 0.75
                if required_secs and required_secs > 0:
                    elapsed = (timezone.now() - last_done.completed_at).total_seconds()
                    remaining = required_secs - elapsed
                    if remaining > 0:
                        import math
                        retry_after = int(math.ceil(remaining))
                        resp = JsonResponse({
                            "ok": False,
                            "error": _("Please wait %(sec)s seconds before the next action.") % {"sec": retry_after},
                            "retry_after": retry_after,
                        }, status=429)
                        try:
                            resp["Retry-After"] = str(retry_after)
                        except Exception:
                            pass
                        return resp
        except Exception:
            # Fail-open if cooldown check errors
            pass
        if is_choice:
            last_done = await asyncio.to_thread(lambda: (
                GameTurn.objects
                .filter(chapter__game_id=game.id, status="done")
                .order_by("-chapter__index", "-index")
                .first()
            ))
            if not last_done or not (last_done.choices or []):
                return JsonResponse({"ok": False, "error": _("No choices available at this time.")}, status=400)
            # Accept sanitized match as well to handle punctuation/whitespace normalization
            stored_choices = last_done.choices or []
            if user_text not in stored_choices:
                try:
                    norm_choices = [sanitize_user_input(c, max_length=1000) for c in stored_choices]
                except Exception:
                    norm_choices = stored_choices
                if user_text not in norm_choices:
                    return JsonResponse({"ok": False, "error": _("This choice is not available on the latest turn.")}, status=400)

        # Reserve turn and enqueue Celery processing; return pending
        t = await svc.a_reserve_turn(
            game_id=str(game.id),
            user_text=user_text,
            chapter_index=None,
        )
        try:
            process_turn_task.delay(str(t.id))
        except Exception as e:
            logger.error(f"Failed to enqueue turn processing task for turn {t.id}: {e}", exc_info=True)
        return JsonResponse({
            "ok": True,
            "turn_id": str(t.id),
            "index": t.index,
            "chapter_index": t.chapter.index,
            "status": t.status,
        })
    except Exception as e:
        logger.error(f"api_generate_turn failed for game {game_id}: {e}", exc_info=True)
        if settings.DEBUG:
            return JsonResponse({"ok": False, "error": _("Internal error"), "debug": str(e)}, status=500)
        return JsonResponse({"ok": False, "error": _("Internal error")}, status=500)


@require_post_async
@api_login_required
async def api_reserve_turn(request: HttpRequest, game_id) -> JsonResponse:
    from user.security import validate_ownership, validate_json_request, sanitize_user_input
    
    prof = _ensure_profile_for_auth(request.user)
    try:
        game = await asyncio.to_thread(lambda: Game.objects.get(pk=game_id))
    except Game.DoesNotExist:
        return JsonResponse({"ok": False, "error": "Game not found"}, status=404)
    
    # Validate ownership
    try:
        validate_ownership(prof, game, field='user')
    except PermissionDenied:
        from user.security import log_suspicious_activity
        log_suspicious_activity(request, f"Unauthorized turn generation attempt for game {game_id}")
        return JsonResponse({"ok": False, "error": "Permission denied"}, status=403)
    # Block playing archived games
    if getattr(game, "is_archived", False):
        return JsonResponse({"ok": False, "error": _("This game is archived and cannot be played.")}, status=403)
    
    try:
        # Try JSON first, fallback to form data
        if request.content_type == 'application/json':
            payload = validate_json_request(request)
        else:
            payload = request.POST
    except Exception as e:
        logger.warning(f"Failed to parse request payload for game {game_id}: {e}")
        return JsonResponse({"ok": False, "error": _("Invalid request format")}, status=400)
    
    # Validate that we have a valid payload
    if not payload or 'user_text' not in payload:
        logger.warning(f"Missing required fields in request for game {game_id}: {payload}")
        return JsonResponse({"ok": False, "error": _("Missing required fields")}, status=400)
    
    user_text = sanitize_user_input(payload.get("user_text", ""), max_length=1000)
    is_choice = str(payload.get("is_choice") or "0").lower() in {"1", "true", "yes", "on"}
    chapter_index = payload.get("chapter_index")
    chapter_index = int(chapter_index) if isinstance(chapter_index, (int, str)) and str(chapter_index).isdigit() else None
    # rate/plan limits
    try:
        effective_plan = prof.get_effective_plan_for_limits()
    except Exception:
        effective_plan = prof.subscription_plan
    limits = _plan_limits(effective_plan)
    now = timezone.now()
    # Determine cycle start (monthly for free, billing cycle for paid)
    try:
        month_start = prof.get_usage_window_start()
    except Exception:
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    minute_ago = now - timezone.timedelta(seconds=60)
    user_game_ids = Game.objects.filter(user=prof).values_list("id", flat=True)
    monthly_used = GameTurn.objects.filter(
        chapter__game_id__in=user_game_ids,
        reserved_at__gte=month_start,
    ).count()
    if monthly_used >= limits["monthly_turns"]:
        return JsonResponse({"ok": False, "error": _("Monthly turn limit reached for your plan.")}, status=429)
    minute_used = GameTurn.objects.filter(
        chapter__game_id__in=user_game_ids,
        reserved_at__gte=minute_ago,
    ).count()
    if (limits["per_minute_turns"] or 0) > 0 and minute_used >= limits["per_minute_turns"]:
        return JsonResponse({"ok": False, "error": _("Too many requests. Please wait a moment.")}, status=429)
    
    # Check audio quota
    from django.db.models import Sum
    monthly_audio = UsageTracking.objects.filter(
        user=prof,
        created_at__gte=month_start
    ).aggregate(total=Sum('audio_duration_seconds'))['total'] or 0
    
    if float(monthly_audio) >= limits["monthly_audio_seconds"]:
        # Signal UI to show pricing CTA across pages, only for Free plan
        try:
            if prof.subscription_plan == ProfileUser.SubscriptionPlan.FREE:
                request.session["show_pricing_cta"] = True
        except Exception:
            pass
        return JsonResponse({"ok": False, "error": _("Monthly audio generation limit reached for your plan.")}, status=429)

    # Cooldown: enforce wait based on last media duration (paid) or 20s for FREE
    try:
        last_done = (
            GameTurn.objects
            .filter(chapter__game=game, status="done")
            .only("audio_duration_seconds", "completed_at")
            .order_by("-completed_at")
            .first()
        )
        if last_done and last_done.completed_at:
            plan = prof.subscription_plan
            required_secs = 20.0 if plan == ProfileUser.SubscriptionPlan.FREE else float(last_done.audio_duration_seconds or 0) * 0.75
            if required_secs and required_secs > 0:
                elapsed = (timezone.now() - last_done.completed_at).total_seconds()
                remaining = required_secs - elapsed
                if remaining > 0:
                    import math
                    retry_after = int(math.ceil(remaining))
                    resp = JsonResponse({
                        "ok": False,
                        "error": _("Please wait %(sec)s seconds before the next action.") % {"sec": retry_after},
                        "retry_after": retry_after,
                    }, status=429)
                    try:
                        resp["Retry-After"] = str(retry_after)
                    except Exception:
                        pass
                    return resp
    except Exception:
        # Fail-open if cooldown check errors
        pass

    # Enforce: If this is a choice click, it must belong to the latest completed turn only
    if is_choice:
        last_done = (
            GameTurn.objects
            .filter(chapter__game=game, status="done")
            .order_by("-chapter__index", "-index")
            .first()
        )
        if not last_done or not (last_done.choices or []):
            return JsonResponse({"ok": False, "error": _("No choices available at this time.")}, status=400)
        if user_text not in (last_done.choices or []):
            return JsonResponse({"ok": False, "error": _("This choice is not available on the latest turn.")}, status=400)

    t = await svc.a_reserve_turn(
        game_id=str(game.id),
        user_text=user_text,
        chapter_index=chapter_index,
    )
    # enqueue background processing
    try:
        process_turn_task.delay(str(t.id))
    except Exception as e:
        logger.error(f"Failed to enqueue turn processing task for turn {t.id}: {e}", exc_info=True)
        # Continue anyway - the turn is reserved and can be processed later
    return JsonResponse({
        "ok": True,
        "turn_id": str(t.id),
        "index": t.index,
        "chapter_index": t.chapter.index,
        "status": t.status,
    })


@require_GET
@api_login_required
async def api_turn_detail(request: HttpRequest, turn_id) -> JsonResponse:
    from user.security import validate_ownership
    
    prof = await _a_ensure_profile_for_auth(request.user)
    try:
        t = await asyncio.to_thread(lambda: GameTurn.objects.select_related("chapter", "chapter__game").get(pk=turn_id))
    except GameTurn.DoesNotExist:
        return JsonResponse({"ok": False, "error": _("Turn not found")}, status=404)
    
    # Validate ownership through game
    try:
        validate_ownership(prof, t.chapter.game, field='user')
    except PermissionDenied:
        return JsonResponse({"ok": False, "error": _("Permission denied")}, status=403)
    
    return JsonResponse({
        "ok": True,
        "turn_id": str(t.id),
        "game_id": str(t.chapter.game_id),
        "chapter_id": str(t.chapter_id),
        "index": t.index,
        "chapter_index": t.chapter.index,
        "status": t.status,
        "scene": t.new_lines,  # Map database field to API field
        "choices": t.choices,
        "tts_url": t.generated_tts_url,
        "image_url": t.generated_visual_url,
        "selected_sound_effect": t.selected_sound_effect,
        "error_message": t.error_message,
        "audio_duration_seconds": float(t.audio_duration_seconds or 0),
        "completed_at": (t.completed_at.isoformat() if t.completed_at else None),
    })


@require_post_async
@api_login_required
async def api_process_turn(request: HttpRequest, turn_id) -> JsonResponse:
    from user.security import validate_ownership
    
    prof = _ensure_profile_for_auth(request.user)
    
    # Get turn and validate ownership
    try:
        turn = GameTurn.objects.select_related("chapter__game").get(pk=turn_id)
        validate_ownership(prof, turn.chapter.game, field='user')
    except GameTurn.DoesNotExist:
        return JsonResponse({"ok": False, "error": _("Turn not found")}, status=404)
    except PermissionDenied:
        return JsonResponse({"ok": False, "error": _("Permission denied")}, status=403)
    
    # Dev helper: process a reserved turn synchronously (in production this would be a worker job)
    sfx = GameConstants.get_sound_effects()
    t2 = await svc.a_process_turn_job(
        str(turn_id),
        llm_next=llm_next,
        tts_narration=svc.tts_narration,
        tts_character=svc.tts_character,
        tti_generate=svc.tti_generate,
        sound_effects=sfx,
    )
    return JsonResponse({
        "ok": True,
        "turn_id": str(t2.id),
        "status": t2.status,
        "scene": t2.new_lines,  # Map database field to API field
        "choices": t2.choices,
        "tts_url": t2.generated_tts_url,
        "image_url": t2.generated_visual_url,
        "selected_sound_effect": t2.selected_sound_effect,
        "error_message": t2.error_message,
    })


@require_post_async
@api_login_required  
async def api_retry_failed_turn(request: HttpRequest, turn_id) -> JsonResponse:
    """Retry a failed turn by re-enqueuing it for processing."""
    from user.security import validate_ownership
    
    prof = _ensure_profile_for_auth(request.user)
    
    # Get turn and validate ownership
    try:
        turn = GameTurn.objects.select_related("chapter__game").get(pk=turn_id)
        validate_ownership(prof, turn.chapter.game, field='user')
    except GameTurn.DoesNotExist:
        return JsonResponse({"ok": False, "error": "Turn not found"}, status=404)
    except PermissionDenied:
        return JsonResponse({"ok": False, "error": "Permission denied"}, status=403)
    
    # Only retry if the turn is in FAILED status
    if turn.status != TurnStatus.FAILED:
        return JsonResponse({"ok": False, "error": _("Turn is not in failed status (current: %(status)s)") % {"status": turn.status}}, status=400)
    
    # Check if max attempts reached (prevent infinite retries)
    max_attempts = 5
    if turn.attempts >= max_attempts:
        return JsonResponse({"ok": False, "error": _("Maximum retry attempts (%(max)s) reached") % {"max": max_attempts}}, status=429)
    
    # Reset turn to PENDING status for retry
    turn.status = TurnStatus.PENDING
    turn.error_message = ""
    turn.save(update_fields=["status", "error_message"])
    
    # Re-enqueue the turn for processing
    try:
        process_turn_task.delay(str(turn.id))
        logger.info(f"Retrying failed turn {turn.id} (attempt {turn.attempts + 1})")
    except Exception as e:
        logger.error(f"Failed to enqueue retry for turn {turn.id}: {e}", exc_info=True)
        # Revert status back to FAILED
        turn.status = TurnStatus.FAILED
        turn.error_message = "Failed to enqueue retry"
        turn.save(update_fields=["status", "error_message"])
        return JsonResponse({"ok": False, "error": "Failed to enqueue retry"}, status=500)
    
    return JsonResponse({
        "ok": True,
        "turn_id": str(turn.id),
        "status": turn.status,
        "attempt": turn.attempts + 1
    })


# ----------------------------
# Paddle webhook (skeleton)
# ----------------------------
from django.views.decorators.csrf import csrf_exempt
import hmac, hashlib, base64

@csrf_exempt
@require_GET
@api_login_required
def api_usage_stats(request: HttpRequest) -> JsonResponse:
    """Get current usage statistics for the authenticated user."""
    prof = _ensure_profile_for_auth(request.user)
    limits = _plan_limits(prof.subscription_plan)
    try:
        usage_stats = prof.current_cycle_usage()
    except Exception:
        usage_stats = prof.current_month_usage()
    
    audio_used = float(usage_stats.get('total_audio_seconds', 0) or 0)
    audio_limit = limits.get('monthly_audio_seconds', 0)
    turn_count = usage_stats.get('turn_count', 0) or 0
    monthly_turns_limit = limits.get('monthly_turns', 0)
    
    return JsonResponse({
        "ok": True,
        "usage": {
            "audio_seconds_used": audio_used,
            "audio_seconds_limit": audio_limit,
            "audio_percentage": min((audio_used / audio_limit * 100) if audio_limit > 0 else 0, 100),
            "turns_used": turn_count,
            "turns_limit": monthly_turns_limit,
            "turns_percentage": min((turn_count / monthly_turns_limit * 100) if monthly_turns_limit > 0 else 0, 100),
        },
        "plan": prof.subscription_plan,
    })


@login_required
@require_POST
def api_set_theme(request: HttpRequest) -> JsonResponse:
    """Update user's theme preference."""
    try:
        import json
        data = json.loads(request.body)
        theme = data.get('theme', 'system')
        
        if theme not in ['system', 'light', 'dark']:
            return JsonResponse({"ok": False, "error": _("Invalid theme")}, status=400)
        
        prof = _ensure_profile_for_auth(request.user)
        prof.theme = theme
        prof.save(update_fields=['theme'])
        
        return JsonResponse({"ok": True, "theme": theme})
    except Exception as e:
        logger.error(f"Error setting theme: {e}")
        return JsonResponse({"ok": False, "error": _("Failed to update theme")}, status=500)


@login_required
@require_POST
def api_set_language(request: HttpRequest) -> JsonResponse:
    """Update user's preferred language and activate for the session."""
    try:
        data = json.loads(request.body or '{}')
        lang = (data.get('language') or '').lower()
        if lang not in {'en','es','fr','pt-br'}:
            return JsonResponse({"ok": False, "error": _("Invalid language")}, status=400)
        prof = _ensure_profile_for_auth(request.user)
        prof.preferred_language = lang
        prof.save(update_fields=['preferred_language'])
        translation.activate(lang)
        request.session[translation.LANGUAGE_SESSION_KEY] = lang
        return JsonResponse({"ok": True, "language": lang})
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        return JsonResponse({"ok": False, "error": _("Failed to update language")}, status=500)


@require_POST
def paddle_webhook(request: HttpRequest) -> HttpResponse:
    """Verify Paddle Billing webhook and update user's plan.
    Minimal skeleton; align exact signature scheme with Paddle docs.
    """
    secret = (getattr(settings, 'PADDLE_WEBHOOK_SECRET', '') or '').encode('utf-8')
    sig = request.headers.get('Paddle-Signature') or request.headers.get('Paddle-Signature-V2') or ''
    raw = request.body
    try:
        payload = json.loads(raw.decode('utf-8'))
    except Exception:
        return HttpResponse(status=400)

    # Placeholder HMAC validation; replace with Paddle's spec if different
    if secret and sig:
        calc = base64.b64encode(hmac.new(secret, raw, hashlib.sha256).digest()).decode('utf-8')
        if not hmac.compare_digest(calc, sig):
            return HttpResponse(status=401)

    event_type = (payload.get('event_type') or payload.get('event') or '').lower()
    data = payload.get('data') or {}
    items = data.get('items') if isinstance(data.get('items'), list) else []
    price_id = (items[0].get('price', {}).get('id') if items else '')
    email = (data.get('customer', {}).get('email') or '').strip().lower()

    if email:
        prof = ProfileUser.objects.filter(email__iexact=email).first()
        if prof:
            desired = None
            if price_id and price_id == getattr(settings, 'PADDLE_PRICE_ID_PLUS', ''):
                desired = ProfileUser.SubscriptionPlan.PLUS
            elif price_id and price_id == getattr(settings, 'PADDLE_PRICE_ID_LITE', ''):
                desired = ProfileUser.SubscriptionPlan.LITE
            elif price_id and price_id == getattr(settings, 'PADDLE_PRICE_ID_PRO', ''):
                desired = ProfileUser.SubscriptionPlan.PRO
            # On activation/renewal/payment completion: set plan and reset cycle start, and clear cancel flag
            if event_type in { 'subscription.activated', 'subscription.updated', 'transaction.completed' } and desired:
                from django.utils import timezone as _tz
                prof.subscription_plan = desired
                prof.plan_cycle_started_at = _tz.now()
                prof.billing_is_canceled = False
                # Persist last paid plan metadata for effective limits when toggling FREE within month
                prof.last_paid_plan = desired
                prof.last_paid_cycle_started_at = prof.plan_cycle_started_at
                prof.save(update_fields=['subscription_plan', 'plan_cycle_started_at', 'billing_is_canceled', 'last_paid_plan', 'last_paid_cycle_started_at'])
            # On cancel/past_due -> keep access until end of current period; mark canceled
            if event_type in { 'subscription.canceled', 'subscription.past_due' }:
                prof.billing_is_canceled = True
                # Do not clear last_paid_*; they are needed for effective limits in the same calendar month
                prof.save(update_fields=['billing_is_canceled'])
    return HttpResponse(status=200)


# ----------------------------
# DRF viewsets (kept)
# ----------------------------

class GameViewSet(viewsets.ModelViewSet):
    queryset = Game.objects.all().select_related("user").prefetch_related("characters").order_by("-created_at")
    serializer_class = GameSerializer


class GameCharacterViewSet(viewsets.ModelViewSet):
    queryset = GameCharacter.objects.all().order_by("name")
    serializer_class = GameCharacterSerializer


class GameStoryViewSet(viewsets.ModelViewSet):
    queryset = GameStory.objects.select_related("game").all()
    serializer_class = GameStorySerializer


class GameChapterViewSet(viewsets.ModelViewSet):
    queryset = GameChapter.objects.select_related("game").all().order_by("game_id", "index")
    serializer_class = GameChapterSerializer

    # Basit filtre: /api/chapters/?game=<game_uuid>
    def get_queryset(self):
        qs = super().get_queryset()
        game_id = self.request.query_params.get("game")
        if game_id:
            qs = qs.filter(game_id=game_id)
        return qs


class GameTurnViewSet(viewsets.ModelViewSet):
    queryset = GameTurn.objects.select_related("chapter", "chapter__game").all().order_by("chapter_id", "index")
    serializer_class = GameTurnSerializer

    # Basit filtre: /api/turns/?chapter=<chapter_uuid>
    def get_queryset(self):
        qs = super().get_queryset()
        chapter_id = self.request.query_params.get("chapter")
        if chapter_id:
            qs = qs.filter(chapter_id=chapter_id)
        return qs

    @method_decorator(login_required)
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)