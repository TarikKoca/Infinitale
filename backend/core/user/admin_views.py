"""
Admin status monitoring views.
"""
import json
import logging
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.conf import settings

from user.monitoring import HealthMonitor, ServiceTester

logger = logging.getLogger(__name__)


@staff_member_required
def admin_status_page(request: HttpRequest) -> HttpResponse:
    """Main admin status monitoring page."""
    monitor = HealthMonitor()
    
    # Get initial data
    context = {
        'page_title': 'System Status Monitor',
        'initial_health': monitor.get_overall_health(),
        'game_metrics': monitor.get_game_metrics(),
        'error_metrics': monitor.get_error_metrics(hours=24)
    }
    
    return render(request, 'admin_status.html', context)


@staff_member_required
@require_http_methods(["GET"])
def api_health_check(request: HttpRequest) -> JsonResponse:
    """API endpoint for real-time health monitoring."""
    monitor = HealthMonitor()
    health_data = monitor.get_overall_health()
    return JsonResponse(health_data)


@staff_member_required
@require_http_methods(["GET"])
def api_system_metrics(request: HttpRequest) -> JsonResponse:
    """Get detailed system metrics."""
    monitor = HealthMonitor()
    metrics = monitor.get_system_metrics()
    return JsonResponse(metrics)


@staff_member_required
@require_http_methods(["GET"])
def api_error_metrics(request: HttpRequest) -> JsonResponse:
    """Get error metrics for specified time range."""
    hours = int(request.GET.get('hours', 24))
    monitor = HealthMonitor()
    metrics = monitor.get_error_metrics(hours=hours)
    return JsonResponse(metrics)


@staff_member_required
@require_http_methods(["GET"])
def api_game_metrics(request: HttpRequest) -> JsonResponse:
    """Get game-related metrics."""
    monitor = HealthMonitor()
    metrics = monitor.get_game_metrics()
    return JsonResponse(metrics)


@staff_member_required
@require_http_methods(["POST"])
def api_run_test(request: HttpRequest) -> JsonResponse:
    """Run a specific test."""
    from user.security import validate_json_request
    try:
        data = validate_json_request(request)
        test_name = data.get('test_name', '').strip()
        
        if not test_name:
            return JsonResponse({'error': 'test_name required'}, status=400)
        
        tester = ServiceTester()
        
        # Map test names to methods
        test_methods = {
            'database': tester.test_database,
            'redis': tester.test_redis,
            'celery': tester.test_celery,
            'openai_api': tester.test_openai_api,
            'replicate_api': tester.test_replicate_api,
            'user_creation': tester.test_user_creation,
            'game_creation': tester.test_game_creation,
            'turn_processing': tester.test_turn_processing,
            'media_storage': tester.test_media_storage,
        }
        
        if test_name not in test_methods:
            return JsonResponse({'error': f'Unknown test: {test_name}'}, status=400)
        
        # Run the test
        success, message, duration = test_methods[test_name]()
        
        return JsonResponse({
            'success': success,
            'message': message,
            'duration_ms': round(duration * 1000, 2),
            'test_name': test_name
        })
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Test execution error: {str(e)}',
            'duration_ms': 0
        }, status=500)


@staff_member_required
@require_http_methods(["POST"])
def api_run_all_tests(request: HttpRequest) -> JsonResponse:
    """Run all tests."""
    try:
        tester = ServiceTester()
        results = tester.run_all_tests()
        return JsonResponse(results)
    except Exception as e:
        logger.error(f"Failed to run all tests: {e}")
        return JsonResponse({
            'error': f'Failed to run tests: {str(e)}'
        }, status=500)


@staff_member_required
@require_http_methods(["POST"])
def api_clear_cache(request: HttpRequest) -> JsonResponse:
    """Clear application cache."""
    try:
        cache.clear()
        logger.info("Cache cleared by admin")
        return JsonResponse({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Failed to clear cache: {str(e)}'
        }, status=500)


@staff_member_required
@require_http_methods(["GET"])
def api_log_tail(request: HttpRequest) -> JsonResponse:
    """Get recent log entries."""
    try:
        # Validate and sanitize input to prevent command injection
        log_type = request.GET.get('type', 'main')
        if log_type not in ('main', 'error'):
            log_type = 'main'
        
        try:
            lines = int(request.GET.get('lines', 50))
            lines = max(1, min(lines, 1000))  # Limit to 1-1000 lines
        except (ValueError, TypeError):
            lines = 50
        
        if log_type == 'error':
            log_path = settings.BASE_DIR / 'var' / 'logs' / 'error.log'
        else:
            log_path = settings.BASE_DIR / 'var' / 'logs' / 'Eternalore.log'
        
        if not log_path.exists():
            return JsonResponse({
                'logs': [],
                'message': f'Log file not found: {log_path}'
            })
        
        # Read file directly instead of using subprocess to avoid command injection
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            log_lines = [line.rstrip() for line in log_lines]
        
        return JsonResponse({
            'logs': log_lines,
            'count': len(log_lines),
            'log_type': log_type
        })
        
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return JsonResponse({
            'error': f'Failed to read logs: {str(e)}'
        }, status=500)


@staff_member_required
@require_http_methods(["GET"])
def api_service_details(request: HttpRequest) -> JsonResponse:
    """Get detailed information about a specific service."""
    service = request.GET.get('service')
    
    if not service:
        return JsonResponse({'error': 'service parameter required'}, status=400)
    
    monitor = HealthMonitor()
    
    service_methods = {
        'database': monitor.get_database_health,
        'redis': monitor.get_redis_health,
        'celery': monitor.get_celery_health,
        'apis': monitor.get_api_health,
    }
    
    if service not in service_methods:
        return JsonResponse({'error': f'Unknown service: {service}'}, status=400)
    
    try:
        details = service_methods[service]()
        return JsonResponse({
            'service': service,
            'details': details
        })
    except Exception as e:
        logger.error(f"Failed to get service details: {e}")
        return JsonResponse({
            'error': f'Failed to get service details: {str(e)}'
        }, status=500)