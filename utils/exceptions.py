"""
Custom exceptions for the Business Agent Management System
"""

from typing import Any, Dict, Optional


class BusinessAgentError(Exception):
    """Base exception for all business agent system errors"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.message = message

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AgentInitializationError(BusinessAgentError):
    """Raised when an agent fails to initialize properly"""

    pass


class AgentDecisionError(BusinessAgentError):
    """Raised when an agent fails to make a decision"""

    pass


class AgentCommunicationError(BusinessAgentError):
    """Raised when agents fail to communicate"""

    pass


class DatabaseError(BusinessAgentError):
    """Base class for database-related errors"""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""

    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails"""

    pass


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails"""

    pass


class ConfigurationError(BusinessAgentError):
    """Base class for configuration-related errors"""

    pass


class ConfigurationFileError(ConfigurationError):
    """Raised when configuration file is invalid or missing"""

    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration values are invalid"""

    pass


class SimulationError(BusinessAgentError):
    """Base class for simulation-related errors"""

    pass


class DataGenerationError(SimulationError):
    """Raised when data generation fails"""

    pass


class BusinessProfileError(SimulationError):
    """Raised when business profile is invalid"""

    pass


class ExternalServiceError(BusinessAgentError):
    """Base class for external service errors"""

    pass


class AnthropicAPIError(ExternalServiceError):
    """Raised when Anthropic API calls fail"""

    pass


class DashboardError(BusinessAgentError):
    """Base class for dashboard-related errors"""

    pass


class DataVisualizationError(DashboardError):
    """Raised when data visualization fails"""

    pass


class ResourceExhaustionError(BusinessAgentError):
    """Raised when system resources are exhausted"""

    pass


class ValidationError(BusinessAgentError):
    """Raised when data validation fails"""

    pass


class RetryableError(BusinessAgentError):
    """Base class for errors that can be retried"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after  # Seconds to wait before retry


class NonRetryableError(BusinessAgentError):
    """Base class for errors that should not be retried"""

    pass


class CircuitBreakerOpenError(ExternalServiceError):
    """Raised when circuit breaker is open"""

    pass
