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


class AgentDecisionError(BusinessAgentError):
    """Raised when an agent fails to make a decision"""


class AgentCommunicationError(BusinessAgentError):
    """Raised when agents fail to communicate"""


class DatabaseError(BusinessAgentError):
    """Base class for database-related errors"""


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails"""


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails"""


class ConfigurationError(BusinessAgentError):
    """Base class for configuration-related errors"""


class ConfigurationFileError(ConfigurationError):
    """Raised when configuration file is invalid or missing"""


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration values are invalid"""


class SimulationError(BusinessAgentError):
    """Base class for simulation-related errors"""


class DataGenerationError(SimulationError):
    """Raised when data generation fails"""


class BusinessProfileError(SimulationError):
    """Raised when business profile is invalid"""


class ExternalServiceError(BusinessAgentError):
    """Base class for external service errors"""


class AnthropicAPIError(ExternalServiceError):
    """Raised when Anthropic API calls fail"""


class DashboardError(BusinessAgentError):
    """Base class for dashboard-related errors"""


class DataVisualizationError(DashboardError):
    """Raised when data visualization fails"""


class ResourceExhaustionError(BusinessAgentError):
    """Raised when system resources are exhausted"""


class ValidationError(BusinessAgentError):
    """Raised when data validation fails"""


class RetryableError(BusinessAgentError):
    """Base class for errors that can be retried"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after  # Seconds to wait before retry


class NonRetryableError(BusinessAgentError):
    """Base class for errors that should not be retried"""


class CircuitBreakerOpenError(ExternalServiceError):
    """Raised when circuit breaker is open"""
