"""
Retry utilities with exponential backoff and circuit breaker patterns
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional, Type

from .exceptions import (
    CircuitBreakerOpenError,
    ExternalServiceError,
    NonRetryableError,
    RetryableError,
)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (
        RetryableError,
        ExternalServiceError,
        ConnectionError,
        TimeoutError,
    )
    non_retryable_exceptions: tuple = (NonRetryableError, ValueError, TypeError)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 3  # Number of successes needed to close circuit


class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""

    def __init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_changes: List[tuple] = []  # (timestamp, old_state, new_state)

    def record_success(self):
        self.success_count += 1
        self.total_calls += 1

    def record_failure(self):
        self.failure_count += 1
        self.total_calls += 1
        self.last_failure_time = datetime.now()

    def record_state_change(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState):
        self.state_changes.append((datetime.now(), old_state, new_state))

    def get_failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls


class CircuitBreaker:
    """Circuit breaker implementation for external service calls"""

    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"circuit_breaker.{name}")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True

        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout

    def _record_success(self):
        """Record a successful call"""
        self.stats.record_success()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _record_failure(self, exception: Exception):
        """Record a failed call"""
        self.stats.record_failure()
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0

        if (
            self.state == CircuitBreakerState.CLOSED
            and self.failure_count >= self.config.failure_threshold
        ):
            self._open_circuit()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.stats.record_state_change(old_state, self.state)
        self.logger.warning(
            f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
        )

    def _close_circuit(self):
        """Close the circuit breaker"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.stats.record_state_change(old_state, self.state)
        self.logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")

    def _half_open_circuit(self):
        """Set circuit breaker to half-open state"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.stats.record_state_change(old_state, self.state)
        self.logger.info(f"Circuit breaker '{self.name}' half-open, testing service")

    def call(self, func: Callable, *args, **kwargs):
        """Call a function through the circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._half_open_circuit()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    context={
                        "failure_count": self.failure_count,
                        "last_failure": self.last_failure_time,
                    },
                )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise

    async def async_call(self, func: Callable, *args, **kwargs):
        """Call an async function through the circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._half_open_circuit()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    context={
                        "failure_count": self.failure_count,
                        "last_failure": self.last_failure_time,
                    },
                )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt with exponential backoff and jitter"""
    delay = min(config.base_delay * (config.exponential_base ** (attempt - 1)), config.max_delay)

    if config.jitter:
        # Add jitter to prevent thundering herd
        jitter_range = delay * 0.1
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def retry(config: Optional[RetryConfig] = None):
    """Decorator for retrying function calls with exponential backoff"""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"retry.{func.__name__}")
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.non_retryable_exceptions as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(f"Final retry attempt failed for {func.__name__}: {e}")
                        break

                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise

            if last_exception:
                raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"retry.{func.__name__}")
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.non_retryable_exceptions as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt == config.max_attempts:
                        logger.error(f"Final retry attempt failed for {func.__name__}: {e}")
                        break

                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise

            if last_exception:
                raise last_exception

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def circuit_breaker(config: Optional[CircuitBreakerConfig] = None, name: str = "default"):
    """Decorator for adding circuit breaker pattern to function calls"""
    if config is None:
        config = CircuitBreakerConfig()

    # Global circuit breakers registry
    if not hasattr(circuit_breaker, "_breakers"):
        circuit_breaker._breakers = {}

    if name not in circuit_breaker._breakers:
        circuit_breaker._breakers[name] = CircuitBreaker(config, name)

    breaker = circuit_breaker._breakers[name]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.async_call(func, *args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_circuit_breaker_stats(name: str = "default") -> Optional[CircuitBreakerStats]:
    """Get statistics for a circuit breaker"""
    if hasattr(circuit_breaker, "_breakers") and name in circuit_breaker._breakers:
        return circuit_breaker._breakers[name].stats
    return None


# Combined decorator for retry + circuit breaker
def resilient_call(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    circuit_breaker_name: str = "default",
):
    """Decorator combining retry and circuit breaker patterns"""

    def decorator(func: Callable) -> Callable:
        # Apply circuit breaker first, then retry
        func = circuit_breaker(circuit_breaker_config, circuit_breaker_name)(func)
        func = retry(retry_config)(func)
        return func

    return decorator
