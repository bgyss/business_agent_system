"""
Test suite for utils.retry module
"""
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pytest

from utils.retry import (
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStats,
    CircuitBreaker,
    calculate_delay,
    retry,
    circuit_breaker,
    get_circuit_breaker_stats,
    resilient_call,
)
from utils.exceptions import (
    CircuitBreakerOpenError,
    RetryableError,
    NonRetryableError,
    ExternalServiceError,
)


class TestRetryConfig:
    """Test RetryConfig dataclass"""

    def test_default_config(self):
        """Test default retry configuration"""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert RetryableError in config.retryable_exceptions
        assert NonRetryableError in config.non_retryable_exceptions

    def test_custom_config(self):
        """Test custom retry configuration"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass"""

    def test_default_config(self):
        """Test default circuit breaker configuration"""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.expected_exception == Exception
        assert config.success_threshold == 3

    def test_custom_config(self):
        """Test custom circuit breaker configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ConnectionError,
            success_threshold=2
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.expected_exception == ConnectionError
        assert config.success_threshold == 2


class TestCircuitBreakerStats:
    """Test CircuitBreakerStats class"""

    def test_initial_stats(self):
        """Test initial statistics"""
        stats = CircuitBreakerStats()
        assert stats.failure_count == 0
        assert stats.success_count == 0
        assert stats.total_calls == 0
        assert stats.last_failure_time is None
        assert stats.state_changes == []
        assert stats.get_failure_rate() == 0.0

    def test_record_success(self):
        """Test recording successful calls"""
        stats = CircuitBreakerStats()
        stats.record_success()
        assert stats.success_count == 1
        assert stats.total_calls == 1
        assert stats.failure_count == 0

    def test_record_failure(self):
        """Test recording failed calls"""
        stats = CircuitBreakerStats()
        stats.record_failure()
        assert stats.failure_count == 1
        assert stats.total_calls == 1
        assert stats.success_count == 0
        assert stats.last_failure_time is not None

    def test_failure_rate_calculation(self):
        """Test failure rate calculation"""
        stats = CircuitBreakerStats()
        stats.record_success()
        stats.record_success()
        stats.record_failure()
        
        assert stats.total_calls == 3
        assert stats.get_failure_rate() == 1/3

    def test_state_change_recording(self):
        """Test recording state changes"""
        stats = CircuitBreakerStats()
        old_state = CircuitBreakerState.CLOSED
        new_state = CircuitBreakerState.OPEN
        
        stats.record_state_change(old_state, new_state)
        
        assert len(stats.state_changes) == 1
        timestamp, recorded_old, recorded_new = stats.state_changes[0]
        assert isinstance(timestamp, datetime)
        assert recorded_old == old_state
        assert recorded_new == new_state


class TestCircuitBreaker:
    """Test CircuitBreaker class"""

    def test_initial_state(self):
        """Test circuit breaker initial state"""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config, "test")
        
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker.name == "test"

    def test_successful_call(self):
        """Test successful function call through circuit breaker"""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config, "test")
        
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.stats.success_count == 1
        assert breaker.failure_count == 0

    def test_failed_call(self):
        """Test failed function call through circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config, "test")
        
        def failing_func():
            raise ConnectionError("Connection failed")
        
        # First failure
        with pytest.raises(ConnectionError):
            breaker.call(failing_func)
        
        assert breaker.failure_count == 1
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Second failure should open circuit
        with pytest.raises(ConnectionError):
            breaker.call(failing_func)
        
        assert breaker.failure_count == 2
        assert breaker.state == CircuitBreakerState.OPEN

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config, "test")
        
        def failing_func():
            raise Exception("Test failure")
        
        # Reach failure threshold
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(failing_func)

    def test_circuit_half_open_after_timeout(self):
        """Test circuit goes to half-open state after timeout"""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker(config, "test")
        
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        with pytest.raises(Exception):
            breaker.call(failing_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should try half-open state
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_closes_after_success_threshold(self):
        """Test circuit closes after success threshold in half-open state"""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker(config, "test")
        
        # Open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            breaker.call(failing_func)
        
        time.sleep(0.2)  # Wait for recovery timeout
        
        # Get to half-open state and reach success threshold
        def success_func():
            return "success"
        
        breaker.call(success_func)  # First success -> half-open
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        breaker.call(success_func)  # Second success -> closed
        assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_async_call(self):
        """Test async function call through circuit breaker"""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config, "test")
        
        async def async_success():
            return "async_success"
        
        result = await breaker.async_call(async_success)
        assert result == "async_success"
        assert breaker.stats.success_count == 1

    @pytest.mark.asyncio
    async def test_async_call_failure(self):
        """Test async function failure through circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config, "test")
        
        async def async_failing():
            raise Exception("Async failure")
        
        with pytest.raises(Exception):
            await breaker.async_call(async_failing)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.async_call(async_failing)


class TestCalculateDelay:
    """Test calculate_delay function"""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        delay1 = calculate_delay(1, config)
        delay2 = calculate_delay(2, config)
        delay3 = calculate_delay(3, config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_max_delay_cap(self):
        """Test maximum delay cap"""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=3.0, jitter=False)
        
        delay = calculate_delay(10, config)  # Would be 512 without cap
        assert delay == 3.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay"""
        config = RetryConfig(base_delay=10.0, exponential_base=2.0, jitter=True)
        
        delays = [calculate_delay(2, config) for _ in range(10)]
        
        # With jitter, delays should vary
        assert len(set(delays)) > 1
        
        # All delays should be around the base value (20.0 for attempt 2)
        for delay in delays:
            assert 18.0 <= delay <= 22.0  # 10% jitter range

    def test_no_negative_delays(self):
        """Test that delays are never negative"""
        config = RetryConfig(base_delay=0.1, jitter=True)
        
        for attempt in range(1, 6):
            delay = calculate_delay(attempt, config)
            assert delay >= 0


class TestRetryDecorator:
    """Test retry decorator"""

    def test_retry_success_on_first_attempt(self):
        """Test successful function on first attempt"""
        @retry()
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"

    def test_retry_success_after_failures(self):
        """Test success after some failures"""
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3, base_delay=0.01, jitter=False))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary failure")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry exhaustion"""
        @retry(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_failing():
            raise RetryableError("Always fails")
        
        with pytest.raises(RetryableError):
            always_failing()

    def test_non_retryable_error_not_retried(self):
        """Test non-retryable errors are not retried"""
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3))
        def non_retryable_func():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("Don't retry this")
        
        with pytest.raises(NonRetryableError):
            non_retryable_func()
        
        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async function retry success"""
        @retry(RetryConfig(max_attempts=2, base_delay=0.01))
        async def async_success():
            return "async_success"
        
        result = await async_success()
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_retry_after_failure(self):
        """Test async function retry after failure"""
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3, base_delay=0.01, jitter=False))
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ExternalServiceError("Temporary failure")
            return "success"
        
        result = await async_flaky()
        assert result == "success"
        assert call_count == 2

    def test_unexpected_error_not_retried(self):
        """Test unexpected errors are not retried"""
        call_count = 0
        
        @retry()
        def unexpected_error_func():
            nonlocal call_count
            call_count += 1
            raise KeyError("Unexpected error")
        
        with pytest.raises(KeyError):
            unexpected_error_func()
        
        assert call_count == 1


class TestCircuitBreakerDecorator:
    """Test circuit_breaker decorator"""

    def test_circuit_breaker_success(self):
        """Test successful calls through circuit breaker decorator"""
        @circuit_breaker(CircuitBreakerConfig(), "test_success")
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
        
        stats = get_circuit_breaker_stats("test_success")
        assert stats.success_count == 1

    def test_circuit_breaker_failure_and_open(self):
        """Test circuit breaker opens after failures"""
        @circuit_breaker(CircuitBreakerConfig(failure_threshold=2), "test_failure")
        def failing_func():
            raise Exception("Test failure")
        
        # First two failures
        for _ in range(2):
            with pytest.raises(Exception):
                failing_func()
        
        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            failing_func()

    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test async circuit breaker decorator"""
        @circuit_breaker(CircuitBreakerConfig(), "test_async")
        async def async_func():
            return "async_result"
        
        result = await async_func()
        assert result == "async_result"

    def test_multiple_circuit_breakers(self):
        """Test multiple named circuit breakers"""
        @circuit_breaker(CircuitBreakerConfig(), "breaker1")
        def func1():
            return "func1"
        
        @circuit_breaker(CircuitBreakerConfig(), "breaker2")
        def func2():
            return "func2"
        
        func1()
        func2()
        
        stats1 = get_circuit_breaker_stats("breaker1")
        stats2 = get_circuit_breaker_stats("breaker2")
        
        assert stats1.success_count == 1
        assert stats2.success_count == 1

    def test_get_circuit_breaker_stats_nonexistent(self):
        """Test getting stats for non-existent circuit breaker"""
        stats = get_circuit_breaker_stats("nonexistent")
        assert stats is None


class TestResilientCall:
    """Test resilient_call decorator (retry + circuit breaker)"""

    def test_resilient_call_success(self):
        """Test successful call with resilient decorator"""
        @resilient_call(
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker_config=CircuitBreakerConfig(),
            circuit_breaker_name="resilient_test"
        )
        def success_func():
            return "resilient_success"
        
        result = success_func()
        assert result == "resilient_success"

    def test_resilient_call_with_retry(self):
        """Test resilient call retries failures"""
        call_count = 0
        
        @resilient_call(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10),
            circuit_breaker_name="resilient_retry_test"
        )
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success_after_retry"
        
        result = flaky_func()
        assert result == "success_after_retry"
        assert call_count == 3

    def test_resilient_call_circuit_breaker_opens(self):
        """Test resilient call circuit breaker opens after failures"""
        @resilient_call(
            retry_config=RetryConfig(max_attempts=1, base_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
            circuit_breaker_name="resilient_cb_test"
        )
        def always_failing():
            raise Exception("Always fails")
        
        # First two calls will fail and exhaust retries
        for _ in range(2):
            with pytest.raises(Exception):
                always_failing()
        
        # Third call should hit open circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            always_failing()

    @pytest.mark.asyncio
    async def test_async_resilient_call(self):
        """Test async resilient call"""
        call_count = 0
        
        @resilient_call(
            retry_config=RetryConfig(max_attempts=2, base_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(),
            circuit_breaker_name="async_resilient_test"
        )
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Async temporary failure")
            return "async_resilient_success"
        
        result = await async_flaky()
        assert result == "async_resilient_success"
        assert call_count == 2


class TestIntegration:
    """Integration tests for retry and circuit breaker patterns"""

    def test_logging_during_retries(self):
        """Test that retries generate appropriate log messages"""
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            @retry(RetryConfig(max_attempts=2, base_delay=0.01))
            def failing_func():
                raise RetryableError("Test failure")
            
            with pytest.raises(RetryableError):
                failing_func()
            
            # Should have warning logs for retry attempts
            assert mock_log.warning.called

    def test_circuit_breaker_logging(self):
        """Test that circuit breaker state changes generate log messages"""
        with patch('logging.getLogger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1), "test_logging")
            
            def failing_func():
                raise Exception("Test failure")
            
            with pytest.raises(Exception):
                breaker.call(failing_func)
            
            # Should have warning log for circuit opening
            assert mock_log.warning.called

    def test_circuit_breaker_with_custom_exception(self):
        """Test circuit breaker with custom expected exception"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            expected_exception=ConnectionError
        )
        breaker = CircuitBreaker(config, "custom_exception_test")
        
        def connection_error_func():
            raise ConnectionError("Connection failed")
        
        def other_error_func():
            raise ValueError("Different error")
        
        # ConnectionError should trigger circuit breaker
        with pytest.raises(ConnectionError):
            breaker.call(connection_error_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # ValueError should not be caught by circuit breaker
        with pytest.raises(ValueError):
            breaker.call(other_error_func)


if __name__ == "__main__":
    pytest.main([__file__])