from django.conf import settings


def site_meta(request):
    # Lazy import to avoid potential circular imports at startup
    user_plan = None
    has_home = False
    show_pricing_cta = False
    try:
        if getattr(request, "user", None) and request.user.is_authenticated:
            from .models import User as ProfileUser  # noqa: WPS433
            email = (getattr(request.user, "email", "") or "").strip()
            if email:
                prof = ProfileUser.objects.filter(email__iexact=email).only("subscription_plan", "plan_cycle_started_at", "last_paid_plan", "last_paid_cycle_started_at").first()
                if prof:
                    user_plan = prof.subscription_plan
                    # Free users now have a home
                    has_home = True
                    # Determine if we should show pricing CTA for free plan
                    if prof.subscription_plan == ProfileUser.SubscriptionPlan.FREE:
                        try:
                            usage = prof.current_cycle_usage() or {}
                            audio_used = float(usage.get('total_audio_seconds') or 0)
                        except Exception:
                            audio_used = 0.0
                        # Audio limit mapping (seconds)
                        audio_limit = 14400.0  # 4 hours for free
                        if audio_used >= audio_limit:
                            show_pricing_cta = True
        # Session-triggered CTA (e.g., after a Free-plan limit error)
        if (
            getattr(request, 'session', None)
            and request.session.get('show_pricing_cta')
            and (user_plan == ProfileUser.SubscriptionPlan.FREE)
        ):
            show_pricing_cta = True
    except Exception:
        # keep defaults if anything goes wrong; do not raise in context processor
        user_plan = None
        has_home = False
        # keep show_pricing_cta default

    return {
        "GA_MEASUREMENT_ID": getattr(settings, "GA_MEASUREMENT_ID", ""),
        "SITE_URL": getattr(settings, "SITE_URL", ""),
        "DEBUG": getattr(settings, "DEBUG", False),
        "USER_SUBSCRIPTION_PLAN": user_plan,
        "HAS_HOME": has_home,
        "SHOW_PRICING_CTA": show_pricing_cta,
    }


