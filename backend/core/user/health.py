"""
Consolidated health check endpoints for production monitoring.
"""
import time
import psutil
import logging
from redis import asyncio as aioredis
from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from django.conf import settings
from django.views.decorators.http import require_GET
from django.views.decorators.cache import never_cache

logger = logging.getLogger(__name__)


@require_GET
@never_cache
async def health_check(request):
    """
    Basic health check endpoint for load balancers.
    Returns 200 if the service is running.
    """
    return JsonResponse({
        'status': 'healthy',
        'service': 'Eternalore',
        'timestamp': int(time.time())
    })


@require_GET
@never_cache
async def liveness_check(request):
    """
    Liveness check to detect if the application is in a deadlock.
    Returns 200 if the application can process requests.
    """
    try:
        # Simple check that the application can process a request
        return JsonResponse({
            'status': 'alive',
            'timestamp': int(time.time())
        })
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JsonResponse({
            'status': 'dead',
            'error': str(e),
            'timestamp': int(time.time())
        }, status=503)


@require_GET
@never_cache
async def liveness_probe(request):
    """
    Kubernetes liveness probe endpoint (alias for liveness_check).
    Returns 200 if the application is running.
    """
    return JsonResponse({"status": "alive"}, status=200)


@require_GET
@never_cache
async def readiness_check(request):
    """
    Readiness check that verifies all critical services are available.
    Returns 200 if all checks pass, 503 otherwise.
    """
    checks = {
        'database': False,
        'cache': False,
        'redis': False,
        'disk_space': False,
        'memory': False
    }
    errors = []
    start_time = time.time()
    
    # Check database
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        checks['database'] = True
    except Exception as e:
        errors.append(f"Database check failed: {str(e)}")
        logger.error(f"Database health check failed: {e}")
    
    # Check cache
    try:
        cache_key = '_health_check'
        cache.set(cache_key, 'ok', 1)
        if cache.get(cache_key) == 'ok':
            checks['cache'] = True
            cache.delete(cache_key)
    except Exception as e:
        errors.append(f"Cache check failed: {str(e)}")
        logger.error(f"Cache health check failed: {e}")
    
    # Check Redis (for Celery)
    try:
        redis_url = settings.CELERY_BROKER_URL
        if redis_url.startswith('redis://'):
            r = aioredis.from_url(redis_url)
            await r.ping()
            checks['redis'] = True
    except Exception as e:
        errors.append(f"Redis check failed: {str(e)}")
        logger.error(f"Redis health check failed: {e}")
    
    # Check disk space (require at least 100MB free)
    try:
        disk_usage = psutil.disk_usage('/')
        if disk_usage.free > 100 * 1024 * 1024:  # 100MB in bytes
            checks['disk_space'] = True
        else:
            errors.append(f"Low disk space: {disk_usage.free / 1024 / 1024:.2f}MB free")
    except Exception as e:
        errors.append(f"Disk check failed: {str(e)}")
    
    # Check memory (require at least 10% free)
    try:
        memory = psutil.virtual_memory()
        if memory.percent < 90:
            checks['memory'] = True
        else:
            errors.append(f"High memory usage: {memory.percent}%")
    except Exception as e:
        errors.append(f"Memory check failed: {str(e)}")
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Determine overall health
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    response_data = {
        'status': 'ready' if all_healthy else 'not_ready',
        'checks': checks,
        'response_time_ms': round(response_time * 1000, 2),
        'version': getattr(settings, 'VERSION', 'unknown'),
        'timestamp': int(time.time())
    }
    
    if errors:
        response_data['errors'] = errors
    
    return JsonResponse(response_data, status=status_code)


@require_GET  
@never_cache
async def readiness_probe(request):
    """
    Kubernetes readiness probe endpoint (alias for readiness_check).
    Checks if the application is ready to serve traffic.
    """
    return readiness_check(request)


@require_GET
@never_cache
async def detailed_health(request):
    """
    Detailed health check for monitoring systems.
    Requires authentication or specific header.
    """
    # Simple authentication via header or query param
    auth_token = request.META.get('HTTP_X_HEALTH_TOKEN') or request.GET.get('token')
    expected_token = getattr(settings, 'HEALTH_CHECK_TOKEN', None)
    
    if expected_token and auth_token != expected_token:
        return JsonResponse({"error": "Unauthorized"}, status=401)
    
    health_data = {
        "timestamp": time.time(),
        "environment": "production" if not settings.DEBUG else "development",
        "debug_mode": settings.DEBUG,
        "version": getattr(settings, 'VERSION', 'unknown'),
        "database": {},
        "cache": {},
        "system": {},
        "celery": {}
    }
    
    # Database metrics
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM user_game WHERE status = 'ongoing'")
            active_games = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM user_user")
            total_users = cursor.fetchone()[0]
            
            health_data["database"] = {
                "connected": True,
                "active_games": active_games,
                "total_users": total_users,
                "connection_count": len(connection.queries) if settings.DEBUG else "N/A"
            }
    except Exception as e:
        health_data["database"] = {"connected": False, "error": str(e)}
    
    # Cache metrics
    try:
        cache_test_key = "health_detailed_test"
        start = time.time()
        cache.set(cache_test_key, "test", 10)
        cache.get(cache_test_key)
        cache.delete(cache_test_key)
        cache_latency = (time.time() - start) * 1000
        
        health_data["cache"] = {
            "connected": True,
            "latency_ms": round(cache_latency, 2)
        }
    except Exception as e:
        health_data["cache"] = {"connected": False, "error": str(e)}
    
    # System metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_data["system"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": round(memory.available / 1024 / 1024, 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else "N/A"
        }
    except Exception as e:
        health_data["system"] = {"error": str(e)}
    
    # Celery metrics (if available)
    try:
        from celery import current_app
        inspect = current_app.control.inspect()
        stats = inspect.stats()
        if stats:
            worker_count = len(stats)
            health_data["celery"] = {
                "workers": worker_count,
                "status": "running"
            }
        else:
            health_data["celery"] = {"status": "no_workers"}
    except Exception as e:
        health_data["celery"] = {"status": "unavailable", "error": str(e)}
    
    return JsonResponse(health_data)