import time
import logging
import uuid
from collections import defaultdict
from django.core.cache import cache
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
from django.shortcuts import redirect
from django.contrib import messages
from django.middleware.csrf import CsrfViewMiddleware
from django.http import HttpResponseForbidden

logger = logging.getLogger(__name__)


class CSRFFailureMiddleware(MiddlewareMixin):
    """
    Handle CSRF failures by redirecting to login or home page instead of showing error.
    """
    
    def process_exception(self, request, exception):
        """Check if the exception is a CSRF failure and redirect accordingly."""
        from django.core.exceptions import PermissionDenied
        
        # Check if this is a CSRF failure
        if isinstance(exception, PermissionDenied) and 'CSRF' in str(exception):
            logger.warning(f"CSRF failure for {request.path} from {self.get_client_ip(request)}")
            
            # Add a user-friendly message
            messages.warning(request, "Your session has expired. Please log in again.")
            
            # Redirect based on authentication status
            if request.user.is_authenticated:
                # If user is authenticated, redirect to home
                return redirect('player_home')
            else:
                # If not authenticated, redirect to login
                return redirect('login')
        
        return None
    
    def get_client_ip(self, request):
        """Get the client's IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', 'unknown')


class RateLimitMiddleware(MiddlewareMixin):
    """
    Simple rate limiting middleware using Django's cache backend.
    Limits requests per IP address.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)
        
        # Configure rate limits (requests per minute)
        self.rate_limits = {
            '/api/': 60,  # API endpoints
            '/game/': 30,  # Game endpoints
            '/': 120,  # General pages
        }
        self.default_limit = 120
        self.window_seconds = 60  # 1 minute window
    
    def get_client_ip(self, request):
        """Get the client's IP address, considering proxy headers."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    def get_rate_limit(self, path):
        """Get the rate limit for a given path."""
        for prefix, limit in self.rate_limits.items():
            if path.startswith(prefix):
                return limit
        return self.default_limit
    
    def process_request(self, request):
        """Check rate limit before processing request."""
        # Skip rate limiting in DEBUG mode
        if settings.DEBUG:
            return None
            
        # Skip rate limiting for authenticated staff users
        if request.user.is_authenticated and request.user.is_staff:
            return None
        
        # Skip rate limiting for HTML page navigations (user experience)
        if request.method == 'GET' and 'text/html' in request.headers.get('Accept', ''):
            return None

        ip = self.get_client_ip(request)
        path = request.path
        rate_limit = self.get_rate_limit(path)
        
        # Create cache key for this IP and path
        cache_key = f"rate_limit:{ip}:{path}"
        
        # Get current request count
        request_count = cache.get(cache_key, 0)
        
        if request_count >= rate_limit:
            logger.warning(f"Rate limit exceeded for IP {ip} on path {path}")

            # Build user-friendly response depending on request type
            retry_after_seconds = self.window_seconds
            accept_header = request.headers.get('Accept', '')

            response = JsonResponse({
                'error': 'Rate limit exceeded. Please try again later.',
                'retry_after': retry_after_seconds
            }, status=429)

            # Always include standard Retry-After header
            response['Retry-After'] = str(retry_after_seconds)
            return response
        
        # Increment request count
        cache.set(cache_key, request_count + 1, self.window_seconds)
        
        return None


class SecurityHeadersMiddleware(MiddlewareMixin):
    """
    Add security headers to all responses.
    """
    
    def process_response(self, request, response):
        """Add security headers to response."""
        # Content Security Policy
        if not settings.DEBUG:
            # Note: 'unsafe-inline' is required for Django admin and some third-party scripts
            # In a future iteration, consider implementing nonce-based CSP
            response['Content-Security-Policy'] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://www.google-analytics.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://www.google-analytics.com; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            )
        
        # Other security headers
        response['X-Content-Type-Options'] = 'nosniff'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        # Strict-Transport-Security is already handled by Django's SecurityMiddleware
        
        return response


class RequestLoggingMiddleware(MiddlewareMixin):
    """
    Log all requests for monitoring and debugging with request ID tracking.
    """
    
    def process_request(self, request):
        """Log incoming request and add request ID."""
        request.start_time = time.time()
        
        # Add request ID for tracking
        request.id = request.META.get('HTTP_X_REQUEST_ID') or str(uuid.uuid4())
        
        # Log request start
        if settings.DEBUG:
            logger.debug(
                f"[{request.id}] {request.method} {request.path} "
                f"from {self.get_client_ip(request)}"
            )
        
        return None
    
    def process_response(self, request, response):
        """Log response and timing."""
        # Add request ID to response headers
        if hasattr(request, 'id'):
            response['X-Request-ID'] = request.id
        
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Only log slow requests in production
            if duration > 1.0 or settings.DEBUG:
                logger.info(
                    f"[{getattr(request, 'id', 'unknown')}] "
                    f"{request.method} {request.path} "
                    f"status={response.status_code} "
                    f"duration={duration:.3f}s "
                    f"ip={self.get_client_ip(request)}"
                )
        
        return response
    
    def process_exception(self, request, exception):
        """Log exceptions with request ID."""
        logger.error(
            f"[{getattr(request, 'id', 'unknown')}] "
            f"Exception in {request.method} {request.path}: {exception}",
            exc_info=True
        )
        return None
    
    def get_client_ip(self, request):
        """Get the client's IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', 'unknown')


class ExceptionHandlingMiddleware(MiddlewareMixin):
    """
    Return JSON instead of HTML for unhandled exceptions when the client expects JSON.
    """
 
    def process_exception(self, request, exception):
        # Skip when DEBUG to let Django show detailed trace
        from django.http import JsonResponse
        from django.utils.log import log_response
        if settings.DEBUG:
            return None
 
        # Log using built-in helper to include request data
        log_response(
            "%s: %s" % (exception.__class__.__name__, exception),
            request,
            response=None,
            status_code=500,
            exc_info=True,
        )
 
        accept = request.headers.get("Accept", "")
        is_json = "application/json" in accept or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        if is_json:
            return JsonResponse({"error": "Internal server error."}, status=500)
        # Otherwise let default 500.html render (using template we added)
        return None