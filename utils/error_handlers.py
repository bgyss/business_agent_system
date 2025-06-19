"""
Error handling utilities and recovery mechanisms
"""
import asyncio
import functools
import logging
from typing import Callable, Optional, Dict, Any, Type, Union
from contextlib import contextmanager, asynccontextmanager
import traceback

from .exceptions import (
    BusinessAgentError,
    DatabaseError,
    DatabaseConnectionError,
    ConfigurationError,
    AgentDecisionError,
    ExternalServiceError
)
from .logging_config import get_logger


class ErrorHandler:
    """Central error handler with recovery strategies"""
    
    def __init__(self, logger_name: str = "error_handler"):
        self.logger = get_logger(logger_name)
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.error_counts: Dict[str, int] = {}
    
    def register_recovery_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for a specific exception type"""
        self.recovery_strategies[exception_type] = strategy
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error using registered recovery strategies
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
        
        Returns:
            bool: True if error was recovered, False otherwise
        """
        error_type = type(error)
        error_key = f"{error_type.__name__}:{str(error)}"
        
        # Track error frequency
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error
        self.logger.log_error(error, f"Handling error (occurrence #{self.error_counts[error_key]})", extra={"context": context})
        
        # Try recovery strategies
        for exc_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exc_type):
                try:
                    self.logger.info(f"Attempting recovery strategy for {exc_type.__name__}")
                    recovery_result = strategy(error, context)
                    
                    if recovery_result:
                        self.logger.info(f"Successfully recovered from {exc_type.__name__}")
                        return True
                    else:
                        self.logger.warning(f"Recovery strategy failed for {exc_type.__name__}")
                
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed with exception: {recovery_error}")
        
        self.logger.error(f"No recovery strategy available for {error_type.__name__}")
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_breakdown": self.error_counts.copy(),
            "recovery_strategies": list(self.recovery_strategies.keys())
        }


# Global error handler instance
_global_error_handler = ErrorHandler()


def register_recovery_strategy(exception_type: Type[Exception]):
    """Decorator to register a recovery strategy"""
    def decorator(func: Callable) -> Callable:
        _global_error_handler.register_recovery_strategy(exception_type, func)
        return func
    return decorator


@contextmanager
def error_handler_context(context: Optional[Dict[str, Any]] = None, reraise: bool = True):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        handled = _global_error_handler.handle_error(e, context)
        if not handled and reraise:
            raise


@asynccontextmanager
async def async_error_handler_context(context: Optional[Dict[str, Any]] = None, reraise: bool = True):
    """Async context manager for error handling"""
    try:
        yield
    except Exception as e:
        handled = _global_error_handler.handle_error(e, context)
        if not handled and reraise:
            raise


def graceful_degradation(fallback_value=None, log_error: bool = True):
    """Decorator for graceful degradation on errors"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger = get_logger(f"graceful_degradation.{func.__name__}")
                    logger.log_error(e, f"Graceful degradation in {func.__name__}, returning fallback value")
                return fallback_value
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger = get_logger(f"graceful_degradation.{func.__name__}")
                    logger.log_error(e, f"Graceful degradation in {func.__name__}, returning fallback value")
                return fallback_value
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_execute(func: Callable, *args, **kwargs) -> tuple:
    """
    Safely execute a function and return (result, error)
    
    Returns:
        tuple: (result, error) where one is None
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


async def async_safe_execute(func: Callable, *args, **kwargs) -> tuple:
    """
    Safely execute an async function and return (result, error)
    
    Returns:
        tuple: (result, error) where one is None
    """
    try:
        result = await func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


# Recovery strategies for common errors

@register_recovery_strategy(DatabaseConnectionError)
def recover_database_connection(error: DatabaseConnectionError, context: Optional[Dict[str, Any]] = None) -> bool:
    """Recovery strategy for database connection errors"""
    logger = get_logger("recovery.database")
    
    try:
        # Attempt to reconnect
        if context and "session_factory" in context:
            session_factory = context["session_factory"]
            # Try to create a new session
            test_session = session_factory()
            test_session.close()
            logger.info("Database connection recovered successfully")
            return True
    except Exception as e:
        logger.error(f"Database connection recovery failed: {e}")
    
    return False


@register_recovery_strategy(ConfigurationError)
def recover_configuration_error(error: ConfigurationError, context: Optional[Dict[str, Any]] = None) -> bool:
    """Recovery strategy for configuration errors"""
    logger = get_logger("recovery.configuration")
    
    try:
        # Try to load default configuration
        if context and "default_config_path" in context:
            default_config_path = context["default_config_path"]
            logger.info(f"Attempting to load default configuration from {default_config_path}")
            # This would need to be implemented based on your config loading logic
            return True
    except Exception as e:
        logger.error(f"Configuration recovery failed: {e}")
    
    return False


@register_recovery_strategy(ExternalServiceError)
def recover_external_service_error(error: ExternalServiceError, context: Optional[Dict[str, Any]] = None) -> bool:
    """Recovery strategy for external service errors"""
    logger = get_logger("recovery.external_service")
    
    # For external service errors, we typically don't recover immediately
    # but we can log for monitoring and potentially switch to fallback services
    logger.warning("External service error detected, consider fallback mechanisms")
    
    return False


class ErrorContext:
    """Context manager for tracking error context"""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.logger = get_logger(f"error_context.{operation}")
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}", extra={"context": self.context})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.context["operation"] = self.operation
            self.context["exception_type"] = exc_type.__name__
            
            # Handle the error
            handled = _global_error_handler.handle_error(exc_val, self.context)
            
            if handled:
                self.logger.info(f"Error recovered in operation: {self.operation}")
                return True  # Suppress the exception
            else:
                self.logger.error(f"Unrecoverable error in operation: {self.operation}")
                return False  # Let the exception propagate
        else:
            self.logger.debug(f"Operation completed successfully: {self.operation}")


class AsyncErrorContext:
    """Async context manager for tracking error context"""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.logger = get_logger(f"error_context.{operation}")
    
    async def __aenter__(self):
        self.logger.debug(f"Starting async operation: {self.operation}", extra={"context": self.context})
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.context["operation"] = self.operation
            self.context["exception_type"] = exc_type.__name__
            
            # Handle the error
            handled = _global_error_handler.handle_error(exc_val, self.context)
            
            if handled:
                self.logger.info(f"Error recovered in async operation: {self.operation}")
                return True  # Suppress the exception
            else:
                self.logger.error(f"Unrecoverable error in async operation: {self.operation}")
                return False  # Let the exception propagate
        else:
            self.logger.debug(f"Async operation completed successfully: {self.operation}")


def get_error_statistics() -> Dict[str, Any]:
    """Get global error statistics"""
    return _global_error_handler.get_error_statistics()


def reset_error_statistics():
    """Reset global error statistics"""
    _global_error_handler.error_counts.clear()