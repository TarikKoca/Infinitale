"""
Circuit breaker implementation for external service calls.
"""
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from enum import Enum
from django.core.cache import cache

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is in OPEN state."""
    def __init__(self, circuit_name: str, message: str = None):
        self.circuit_name = circuit_name
        if message is None:
            message = f"Circuit breaker {circuit_name} is OPEN"
        super().__init__(message)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external services.
    
    Usage:
        breaker = CircuitBreaker("openai", failure_threshold=5, timeout=60)
        
        @breaker
        def call_openai():
            # Your API call here
            pass
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique identifier for this circuit
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open before closing
            timeout: Seconds to wait before trying half-open state
            expected_exception: Exception type to catch (others will pass through)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        # Cache keys
        self.state_key = f"circuit_{name}_state"
        self.failures_key = f"circuit_{name}_failures"
        self.successes_key = f"circuit_{name}_successes"
        self.open_time_key = f"circuit_{name}_open_time"
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            state = self.get_state()
            
            if state == CircuitState.OPEN:
                # Check if we should transition to half-open
                open_time = cache.get(self.open_time_key, 0)
                if time.time() - open_time >= self.timeout:
                    self.set_state(CircuitState.HALF_OPEN)
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(self.name)
            
            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except self.expected_exception as e:
                self.on_failure()
                raise e
        
        return wrapper
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        state_value = cache.get(self.state_key, CircuitState.CLOSED.value)
        return CircuitState(state_value)
    
    def set_state(self, state: CircuitState) -> None:
        """Set circuit state."""
        cache.set(self.state_key, state.value, self.timeout * 2)
        if state == CircuitState.OPEN:
            cache.set(self.open_time_key, time.time(), self.timeout * 2)
            cache.delete(self.successes_key)
    
    def on_success(self) -> None:
        """Handle successful call."""
        state = self.get_state()
        
        if state == CircuitState.HALF_OPEN:
            successes = cache.get(self.successes_key, 0) + 1
            cache.set(self.successes_key, successes, self.timeout)
            
            if successes >= self.success_threshold:
                self.set_state(CircuitState.CLOSED)
                cache.delete(self.failures_key)
                cache.delete(self.successes_key)
                logger.info(f"Circuit {self.name} is now CLOSED (recovered)")
        
        elif state == CircuitState.CLOSED:
            # Reset failure count on success
            cache.delete(self.failures_key)
    
    def on_failure(self) -> None:
        """Handle failed call."""
        state = self.get_state()
        
        if state == CircuitState.HALF_OPEN:
            self.set_state(CircuitState.OPEN)
            logger.warning(f"Circuit {self.name} is now OPEN (half-open test failed)")
        
        elif state == CircuitState.CLOSED:
            failures = cache.get(self.failures_key, 0) + 1
            cache.set(self.failures_key, failures, self.timeout)
            
            if failures >= self.failure_threshold:
                self.set_state(CircuitState.OPEN)
                logger.warning(f"Circuit {self.name} is now OPEN (threshold reached)")
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        cache.delete(self.state_key)
        cache.delete(self.failures_key)
        cache.delete(self.successes_key)
        cache.delete(self.open_time_key)
        logger.info(f"Circuit {self.name} has been reset")


# Pre-configured circuit breakers for different services
openai_breaker = CircuitBreaker(
    "openai",
    failure_threshold=5,
    success_threshold=2,
    timeout=60
)

replicate_breaker = CircuitBreaker(
    "replicate",
    failure_threshold=3,
    success_threshold=2,
    timeout=120
)

fal_breaker = CircuitBreaker(
    "fal",
    failure_threshold=3,
    success_threshold=2,
    timeout=120
)


def with_circuit_breaker(
    service_name: str,
    fallback: Optional[Callable] = None,
    failure_threshold: int = 5,
    timeout: int = 60
) -> Callable:
    """
    Decorator factory for adding circuit breaker to functions.
    
    Usage:
        @with_circuit_breaker("openai", fallback=my_fallback_func)
        def call_openai_api():
            # Your API call
            pass
    """
    breaker = CircuitBreaker(
        service_name,
        failure_threshold=failure_threshold,
        timeout=timeout
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return breaker(func)(*args, **kwargs)
            except Exception as e:
                if fallback:
                    logger.info(f"Circuit breaker {service_name} triggered, using fallback")
                    return fallback(*args, **kwargs)
                raise e
        return wrapper
    return decorator