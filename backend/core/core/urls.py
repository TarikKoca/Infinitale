from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns
from django.views.i18n import set_language
from django.contrib.sitemaps.views import sitemap
from django.contrib.sitemaps import GenericSitemap
from user.models import Game
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import DefaultRouter
from user.views import (
    GameViewSet, GameCharacterViewSet, GameStoryViewSet,
    GameChapterViewSet, GameTurnViewSet, custom_404
)
from user import admin_views

# Custom error handlers
handler404 = 'user.views.custom_404'

router = DefaultRouter()
router.register(r"games", GameViewSet, basename="game")
router.register(r"characters", GameCharacterViewSet, basename="character")
router.register(r"stories", GameStoryViewSet, basename="story")
router.register(r"chapters", GameChapterViewSet, basename="chapter")
router.register(r"turns", GameTurnViewSet, basename="turn")

urlpatterns = [
    # Admin monitoring endpoints (must come before Django admin to avoid conflicts)
    path("admin/status/", admin_views.admin_status_page, name="admin_status"),
    path("admin/api/health/", admin_views.api_health_check, name="api_health_check"),
    path("admin/api/system-metrics/", admin_views.api_system_metrics, name="api_system_metrics"),
    path("admin/api/error-metrics/", admin_views.api_error_metrics, name="api_error_metrics"),
    path("admin/api/game-metrics/", admin_views.api_game_metrics, name="api_game_metrics"),
    path("admin/api/run-test/", admin_views.api_run_test, name="api_run_test"),
    path("admin/api/run-all-tests/", admin_views.api_run_all_tests, name="api_run_all_tests"),
    path("admin/api/clear-cache/", admin_views.api_clear_cache, name="api_clear_cache"),
    path("admin/api/log-tail/", admin_views.api_log_tail, name="api_log_tail"),
    path("admin/api/service-details/", admin_views.api_service_details, name="api_service_details"),
    
    # Django admin
    path("admin/", admin.site.urls),

    # API endpoints (not localized)
    path("api/", include(router.urls)),
    path('accounts/', include('allauth.urls')),

    # i18n: language switching endpoint
    path('i18n/setlang/', set_language, name='set_language'),

    # Minimal sitemap with latest games (can be extended)
    path('sitemap.xml', sitemap, {
        'sitemaps': {
            'games': GenericSitemap({
                'queryset': Game.objects.all().order_by('-updated_at') if hasattr(Game, 'updated_at') else Game.objects.all(),
                'date_field': 'updated_at' if hasattr(Game, 'updated_at') else None,
            }, priority=0.3)
        }
    }, name='django.contrib.sitemaps.views.sitemap'),
]

# Localized user-facing URLs
urlpatterns += i18n_patterns(
    path("", include("user.urls")),
)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
