from django.urls import path
from django.views.generic import RedirectView, TemplateView
from . import views
from . import admin_views
from . import health
# dev helper: clear cache
from django.core.cache import cache
from django.conf import settings
if settings.DEBUG:
    cache.clear()

urlpatterns = [
    path("", views.landing_page, name="landing_page"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("home/", views.player_home, name="player_home"),
    path("account/", views.account_settings, name="account_settings"),
    path("games/", views.all_games, name="all_games"),
    path("games/new/", views.new_game, name="new_game"),
    path("games/<uuid:game_id>/", views.in_game, name="in_game"),
    path("pricing/", views.pricing_page, name="pricing"),
    path("privacy/", views.privacy_page, name="privacy"),
    path("terms/", views.terms_page, name="terms"),
    path("support/", views.support_page, name="support"),
    path("updates/", views.updates_page, name="updates"),
    path("upgrade/<str:plan>/", views.upgrade_plan, name="upgrade_plan"),
    # Billing (Paddle)
    path("billing/checkout/<str:plan>/", views.billing_checkout, name="billing_checkout"),
    path("billing/return/", views.billing_return, name="billing_return"),
    # App-specific API (avoid conflict with DRF router under /api/)
    path("uapi/games/create/", views.api_create_game, name="api_create_game"),
    path("uapi/games/create/status/", views.api_create_game_status, name="api_create_game_status"),
    path("uapi/games/<uuid:game_id>/delete/", views.api_delete_game, name="api_delete_game"),
    path("uapi/turns/<uuid:game_id>/generate/", views.api_generate_turn, name="api_generate_turn"),
    path("uapi/turns/<uuid:game_id>/reserve/", views.api_reserve_turn, name="api_reserve_turn"),
    path("uapi/turns/detail/<uuid:turn_id>/", views.api_turn_detail, name="api_turn_detail"),
    path("uapi/turns/<uuid:turn_id>/process/", views.api_process_turn, name="api_process_turn"),
    path("uapi/usage/stats/", views.api_usage_stats, name="api_usage_stats"),
    path("uapi/settings/theme/", views.api_set_theme, name="api_set_theme"),
    path("uapi/settings/language/", views.api_set_language, name="api_set_language"),
    # Billing webhooks
    path("webhooks/paddle/", views.paddle_webhook, name="paddle_webhook"),
    # Health checks
    path("health/", health.health_check, name="health_check"),
    path("health/ready/", health.readiness_check, name="readiness_check"),
    path("health/live/", health.liveness_check, name="liveness_check"),
    # Enhanced health checks for production monitoring
    path("health/liveness/", health.liveness_probe, name="liveness_probe"),
    path("health/readiness/", health.readiness_probe, name="readiness_probe"),
    path("health/detailed/", health.detailed_health, name="detailed_health"),
    # robots.txt
    path("robots.txt", TemplateView.as_view(template_name="robots.txt", content_type="text/plain")),
    # Legacy static redirects
    path("player-home/", RedirectView.as_view(pattern_name="player_home", permanent=True)),
    path("player-home/index.html", RedirectView.as_view(pattern_name="player_home", permanent=True)),
    path("player-home/new-game.html", RedirectView.as_view(pattern_name="new_game", permanent=True)),
    path("player-home/in-game/index.html", RedirectView.as_view(pattern_name="player_home", permanent=True)),
    path("pricing/index.html", RedirectView.as_view(pattern_name="pricing", permanent=True)),
    path("login/index.html", RedirectView.as_view(pattern_name="login", permanent=True)),
    path("register/index.html", RedirectView.as_view(pattern_name="register", permanent=True)),
]