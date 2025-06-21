"""
Test suite for utils.error_handlers module
"""
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from utils.error_handlers import (
    ErrorHandler,
    register_recovery_strategy,
    error_handler_context,
    async_error_handler_context,
    graceful_degradation,
    safe_execute,
    async_safe_execute,
    recover_database_connection,
    recover_configuration_error,
    recover_external_service_error,
    ErrorContext,
    AsyncErrorContext,
    get_error_statistics,
    reset_error_statistics,
    _global_error_handler,
)
from utils.exceptions import (
    ConfigurationError,
    DatabaseConnectionError,
    ExternalServiceError,
    BusinessAgentError,
)


class TestErrorHandler:
    """Test ErrorHandler class"""

    def setup_method(self):
        """Setup for each test method"""
        self.error_handler = ErrorHandler("test_handler")

    def test_initialization(self):
        """Test ErrorHandler initialization"""
        assert self.error_handler.recovery_strategies == {}
        assert self.error_handler.error_counts == {}

    def test_register_recovery_strategy(self):
        """Test registering a recovery strategy"""
        def mock_strategy(error, context):
            return True

        self.error_handler.register_recovery_strategy(ValueError, mock_strategy)
        assert ValueError in self.error_handler.recovery_strategies
        assert self.error_handler.recovery_strategies[ValueError] == mock_strategy

    def test_handle_error_with_successful_recovery(self):
        """Test error handling with successful recovery"""
        def successful_recovery(error, context):
            return True

        self.error_handler.register_recovery_strategy(ValueError, successful_recovery)
        
        error = ValueError("Test error")
        result = self.error_handler.handle_error(error)
        
        assert result is True
        assert "ValueError:Test error" in self.error_handler.error_counts
        assert self.error_handler.error_counts["ValueError:Test error"] == 1

    def test_handle_error_with_failed_recovery(self):
        """Test error handling with failed recovery"""
        def failed_recovery(error, context):
            return False

        self.error_handler.register_recovery_strategy(ValueError, failed_recovery)
        
        error = ValueError("Test error")
        result = self.error_handler.handle_error(error)
        
        assert result is False

    def test_handle_error_no_recovery_strategy(self):
        """Test error handling with no recovery strategy"""
        error = ValueError("Test error")
        result = self.error_handler.handle_error(error)
        
        assert result is False

    def test_handle_error_with_context(self):
        """Test error handling with context"""
        def context_aware_recovery(error, context):
            assert context is not None
            assert context.get("test_key") == "test_value"
            return True

        self.error_handler.register_recovery_strategy(ValueError, context_aware_recovery)
        
        error = ValueError("Test error")
        context = {"test_key": "test_value"}
        result = self.error_handler.handle_error(error, context)
        
        assert result is True

    def test_handle_error_recovery_strategy_exception(self):
        """Test error handling when recovery strategy raises exception"""
        def failing_recovery(error, context):
            raise Exception("Recovery failed")

        self.error_handler.register_recovery_strategy(ValueError, failing_recovery)
        
        error = ValueError("Test error")
        result = self.error_handler.handle_error(error)
        
        assert result is False

    def test_error_count_tracking(self):
        """Test error count tracking"""
        error1 = ValueError("Test error")
        error2 = ValueError("Test error")  # Same error
        error3 = TypeError("Different error")

        self.error_handler.handle_error(error1)
        self.error_handler.handle_error(error2)
        self.error_handler.handle_error(error3)

        assert self.error_handler.error_counts["ValueError:Test error"] == 2
        assert self.error_handler.error_counts["TypeError:Different error"] == 1

    def test_get_error_statistics(self):
        """Test getting error statistics"""
        error1 = ValueError("Test error")
        error2 = TypeError("Different error")

        self.error_handler.handle_error(error1)
        self.error_handler.handle_error(error2)

        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 2
        assert stats["unique_errors"] == 2
        assert "ValueError:Test error" in stats["error_breakdown"]
        assert "TypeError:Different error" in stats["error_breakdown"]

    def test_inheritance_based_recovery(self):
        """Test recovery strategy matching based on inheritance"""
        def base_recovery(error, context):
            return True

        self.error_handler.register_recovery_strategy(Exception, base_recovery)
        
        # Subclass of Exception should match
        error = ValueError("Test error")
        result = self.error_handler.handle_error(error)
        
        assert result is True


class TestDecoratorAndContextManagers:
    """Test decorators and context managers"""

    def test_register_recovery_strategy_decorator(self):
        """Test register_recovery_strategy decorator"""
        @register_recovery_strategy(ValueError)
        def test_recovery(error, context):
            return True

        assert ValueError in _global_error_handler.recovery_strategies

    def test_error_handler_context_no_error(self):
        """Test error handler context with no error"""
        with error_handler_context():
            pass  # No error should occur

    def test_error_handler_context_with_error_reraise(self):
        """Test error handler context with error and reraise=True"""
        # Clear any recovery strategies that might interfere
        original_strategies = _global_error_handler.recovery_strategies.copy()
        _global_error_handler.recovery_strategies.clear()
        
        try:
            with pytest.raises(ValueError):
                with error_handler_context():
                    raise ValueError("Test error")
        finally:
            # Restore original strategies
            _global_error_handler.recovery_strategies = original_strategies

    def test_error_handler_context_with_error_no_reraise(self):
        """Test error handler context with error and reraise=False"""
        with error_handler_context(reraise=False):
            raise ValueError("Test error")
        # Should not raise

    @pytest.mark.asyncio
    async def test_async_error_handler_context_no_error(self):
        """Test async error handler context with no error"""
        async with async_error_handler_context():
            pass  # No error should occur

    @pytest.mark.asyncio
    async def test_async_error_handler_context_with_error(self):
        """Test async error handler context with error"""
        # Clear any recovery strategies that might interfere
        original_strategies = _global_error_handler.recovery_strategies.copy()
        _global_error_handler.recovery_strategies.clear()
        
        try:
            with pytest.raises(ValueError):
                async with async_error_handler_context():
                    raise ValueError("Test error")
        finally:
            # Restore original strategies
            _global_error_handler.recovery_strategies = original_strategies

    def test_graceful_degradation_success(self):
        """Test graceful degradation decorator with successful function"""
        @graceful_degradation(fallback_value="fallback")
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_graceful_degradation_failure(self):
        """Test graceful degradation decorator with failing function"""
        @graceful_degradation(fallback_value="fallback")
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "fallback"

    def test_graceful_degradation_no_logging(self):
        """Test graceful degradation with logging disabled"""
        @graceful_degradation(fallback_value="fallback", log_error=False)
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_async_graceful_degradation_success(self):
        """Test async graceful degradation with successful function"""
        @graceful_degradation(fallback_value="fallback")
        async def async_success():
            return "success"

        result = await async_success()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_graceful_degradation_failure(self):
        """Test async graceful degradation with failing function"""
        @graceful_degradation(fallback_value="fallback")
        async def async_failing():
            raise ValueError("Test error")

        result = await async_failing()
        assert result == "fallback"


class TestSafeExecute:
    """Test safe execution functions"""

    def test_safe_execute_success(self):
        """Test safe_execute with successful function"""
        def success_func(x, y):
            return x + y

        result, error = safe_execute(success_func, 1, 2)
        assert result == 3
        assert error is None

    def test_safe_execute_failure(self):
        """Test safe_execute with failing function"""
        def failing_func():
            raise ValueError("Test error")

        result, error = safe_execute(failing_func)
        assert result is None
        assert isinstance(error, ValueError)
        assert str(error) == "Test error"

    def test_safe_execute_with_kwargs(self):
        """Test safe_execute with keyword arguments"""
        def func_with_kwargs(x, y=10):
            return x * y

        result, error = safe_execute(func_with_kwargs, 5, y=3)
        assert result == 15
        assert error is None

    @pytest.mark.asyncio
    async def test_async_safe_execute_success(self):
        """Test async_safe_execute with successful function"""
        async def async_success(x, y):
            return x + y

        result, error = await async_safe_execute(async_success, 1, 2)
        assert result == 3
        assert error is None

    @pytest.mark.asyncio
    async def test_async_safe_execute_failure(self):
        """Test async_safe_execute with failing function"""
        async def async_failing():
            raise ValueError("Test error")

        result, error = await async_safe_execute(async_failing)
        assert result is None
        assert isinstance(error, ValueError)


class TestRecoveryStrategies:
    """Test built-in recovery strategies"""

    @patch('utils.error_handlers.get_logger')
    def test_recover_database_connection_success(self, mock_get_logger):
        """Test successful database connection recovery"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_session = Mock()
        mock_session_factory = Mock(return_value=mock_session)
        
        context = {"session_factory": mock_session_factory}
        error = DatabaseConnectionError("Connection lost")
        
        result = recover_database_connection(error, context)
        
        assert result is True
        mock_session_factory.assert_called_once()
        mock_session.close.assert_called_once()
        mock_logger.info.assert_called_with("Database connection recovered successfully")

    @patch('utils.error_handlers.get_logger')
    def test_recover_database_connection_failure(self, mock_get_logger):
        """Test failed database connection recovery"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_session_factory = Mock(side_effect=Exception("Connection failed"))
        
        context = {"session_factory": mock_session_factory}
        error = DatabaseConnectionError("Connection lost")
        
        result = recover_database_connection(error, context)
        
        assert result is False
        mock_logger.error.assert_called()

    @patch('utils.error_handlers.get_logger')
    def test_recover_database_connection_no_context(self, mock_get_logger):
        """Test database connection recovery with no context"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        error = DatabaseConnectionError("Connection lost")
        result = recover_database_connection(error, None)
        
        assert result is False

    @patch('utils.error_handlers.get_logger')
    def test_recover_configuration_error(self, mock_get_logger):
        """Test configuration error recovery"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        context = {"default_config_path": "/default/config.yaml"}
        error = ConfigurationError("Config invalid")
        
        result = recover_configuration_error(error, context)
        
        assert result is True
        mock_logger.info.assert_called_with("Attempting to load default configuration from /default/config.yaml")

    @patch('utils.error_handlers.get_logger')
    def test_recover_external_service_error(self, mock_get_logger):
        """Test external service error recovery"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        error = ExternalServiceError("Service unavailable")
        result = recover_external_service_error(error, None)
        
        assert result is False
        mock_logger.warning.assert_called_with("External service error detected, consider fallback mechanisms")


class TestErrorContext:
    """Test ErrorContext class"""

    @patch('utils.error_handlers.get_logger')
    def test_error_context_success(self, mock_get_logger):
        """Test ErrorContext with successful operation"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with ErrorContext("test_operation", param1="value1"):
            pass
        
        mock_logger.debug.assert_any_call("Starting operation: test_operation", extra={"context": {"param1": "value1"}})
        mock_logger.debug.assert_any_call("Operation completed successfully: test_operation")

    @patch('utils.error_handlers.get_logger')
    def test_error_context_with_error(self, mock_get_logger):
        """Test ErrorContext with error"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Setup a recovery strategy that returns False (no recovery)
        def no_recovery(error, context):
            return False
        
        _global_error_handler.register_recovery_strategy(ValueError, no_recovery)
        
        with pytest.raises(ValueError):
            with ErrorContext("test_operation"):
                raise ValueError("Test error")
        
        mock_logger.error.assert_called_with("Unrecoverable error in operation: test_operation")

    @patch('utils.error_handlers.get_logger')
    def test_error_context_with_recovery(self, mock_get_logger):
        """Test ErrorContext with successful error recovery"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Setup a recovery strategy that returns True (recovery successful)
        def successful_recovery(error, context):
            return True
        
        _global_error_handler.register_recovery_strategy(ValueError, successful_recovery)
        
        with ErrorContext("test_operation"):
            raise ValueError("Test error")
        # Should not raise because error was recovered
        
        mock_logger.info.assert_called_with("Error recovered in operation: test_operation")


class TestAsyncErrorContext:
    """Test AsyncErrorContext class"""

    @pytest.mark.asyncio
    @patch('utils.error_handlers.get_logger')
    async def test_async_error_context_success(self, mock_get_logger):
        """Test AsyncErrorContext with successful operation"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        async with AsyncErrorContext("async_test_operation", param1="value1"):
            pass
        
        mock_logger.debug.assert_any_call("Starting async operation: async_test_operation", extra={"context": {"param1": "value1"}})
        mock_logger.debug.assert_any_call("Async operation completed successfully: async_test_operation")

    @pytest.mark.asyncio
    @patch('utils.error_handlers.get_logger')
    async def test_async_error_context_with_error(self, mock_get_logger):
        """Test AsyncErrorContext with error"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Setup a recovery strategy that returns False
        def no_recovery(error, context):
            return False
        
        _global_error_handler.register_recovery_strategy(ValueError, no_recovery)
        
        with pytest.raises(ValueError):
            async with AsyncErrorContext("async_test_operation"):
                raise ValueError("Async test error")
        
        mock_logger.error.assert_called_with("Unrecoverable error in async operation: async_test_operation")

    @pytest.mark.asyncio
    @patch('utils.error_handlers.get_logger')
    async def test_async_error_context_with_recovery(self, mock_get_logger):
        """Test AsyncErrorContext with successful error recovery"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Setup a recovery strategy that returns True
        def successful_recovery(error, context):
            return True
        
        _global_error_handler.register_recovery_strategy(ValueError, successful_recovery)
        
        async with AsyncErrorContext("async_test_operation"):
            raise ValueError("Async test error")
        # Should not raise because error was recovered
        
        mock_logger.info.assert_called_with("Error recovered in async operation: async_test_operation")


class TestGlobalFunctions:
    """Test global utility functions"""

    def setup_method(self):
        """Setup for each test method"""
        reset_error_statistics()

    def test_get_error_statistics_empty(self):
        """Test getting error statistics when empty"""
        stats = get_error_statistics()
        assert stats["total_errors"] == 0
        assert stats["unique_errors"] == 0
        assert stats["error_breakdown"] == {}

    def test_get_error_statistics_with_errors(self):
        """Test getting error statistics with errors"""
        _global_error_handler.handle_error(ValueError("Test error 1"))
        _global_error_handler.handle_error(TypeError("Test error 2"))
        _global_error_handler.handle_error(ValueError("Test error 1"))  # Duplicate
        
        stats = get_error_statistics()
        assert stats["total_errors"] == 3
        assert stats["unique_errors"] == 2

    def test_reset_error_statistics(self):
        """Test resetting error statistics"""
        _global_error_handler.handle_error(ValueError("Test error"))
        
        # Verify error was recorded
        stats = get_error_statistics()
        assert stats["total_errors"] == 1
        
        # Reset and verify
        reset_error_statistics()
        stats = get_error_statistics()
        assert stats["total_errors"] == 0


class TestErrorHandlerIntegration:
    """Integration tests for error handler components"""

    def setup_method(self):
        """Setup for each test method"""
        reset_error_statistics()

    def test_full_error_handling_workflow(self):
        """Test complete error handling workflow"""
        # Register a recovery strategy
        recovery_called = False
        recovery_context = None
        
        def test_recovery(error, context):
            nonlocal recovery_called, recovery_context
            recovery_called = True
            recovery_context = context
            return True
        
        _global_error_handler.register_recovery_strategy(ValueError, test_recovery)
        
        # Use error context to handle an error
        with ErrorContext("integration_test", test_param="test_value"):
            raise ValueError("Integration test error")
        
        # Verify recovery was called
        assert recovery_called
        assert recovery_context["operation"] == "integration_test"
        assert recovery_context["test_param"] == "test_value"
        
        # Verify error was tracked
        stats = get_error_statistics()
        assert stats["total_errors"] == 1

    def test_nested_error_contexts(self):
        """Test nested error contexts"""
        context_calls = []
        
        def track_recovery(error, context):
            context_calls.append(context.get("operation", "unknown"))
            return True
        
        _global_error_handler.register_recovery_strategy(ValueError, track_recovery)
        
        with ErrorContext("outer_operation"):
            with ErrorContext("inner_operation"):
                raise ValueError("Nested error")
        
        # Should handle error in inner context
        assert len(context_calls) == 1
        assert context_calls[0] == "inner_operation"

    @pytest.mark.asyncio
    async def test_mixed_sync_async_error_handling(self):
        """Test mixing sync and async error handling"""
        recovery_count = 0
        
        def counting_recovery(error, context):
            nonlocal recovery_count
            recovery_count += 1
            return True
        
        _global_error_handler.register_recovery_strategy(ValueError, counting_recovery)
        
        # Sync error handling
        with ErrorContext("sync_operation"):
            raise ValueError("Sync error")
        
        # Async error handling
        async with AsyncErrorContext("async_operation"):
            raise ValueError("Async error")
        
        assert recovery_count == 2


if __name__ == "__main__":
    pytest.main([__file__])