"""
Custom exception classes for the Business Agent Management System.

This module defines specific exceptions for different types of errors that can occur
throughout the system, providing better error categorization and handling.
"""

from datetime import datetime
from typing import Any, Dict


class BusinessAgentException(Exception):
    """Base exception class for all Business Agent System errors."""

    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dict for logging and serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


# Agent-related exceptions
class AgentError(BusinessAgentException):
    """Base class for agent-related errors."""
    pass


class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize properly."""
    pass


class AgentDecisionError(AgentError):
    """Raised when an agent fails to make a decision."""
    pass


class AgentCommunicationError(AgentError):
    """Raised when agents fail to communicate with each other."""
    pass


class ClaudeAPIError(AgentError):
    """Raised when Claude API calls fail."""

    def __init__(self, message: str, api_response: Dict[str, Any] = None, retry_count: int = 0):
        super().__init__(message, "CLAUDE_API_ERROR")
        self.api_response = api_response
        self.retry_count = retry_count
        if api_response:
            self.context.update({"api_response": api_response})
        self.context.update({"retry_count": retry_count})


# Database-related exceptions
class DatabaseError(BusinessAgentException):
    """Base class for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseTransactionError(DatabaseError):
    """Raised when database transactions fail."""

    def __init__(self, message: str, operation: str = None, table: str = None):
        super().__init__(message, "DB_TRANSACTION_ERROR")
        self.operation = operation
        self.table = table
        if operation:
            self.context.update({"operation": operation})
        if table:
            self.context.update({"table": table})


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""
    pass


class DataValidationError(DatabaseError):
    """Raised when data validation fails before database operations."""

    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, "DATA_VALIDATION_ERROR")
        self.field = field
        self.value = value
        if field:
            self.context.update({"field": field})
        if value is not None:
            self.context.update({"value": str(value)})


# Configuration-related exceptions
class ConfigurationError(BusinessAgentException):
    """Base class for configuration-related errors."""
    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, config_section: str = None, missing_keys: list = None):
        super().__init__(message, "CONFIG_VALIDATION_ERROR")
        self.config_section = config_section
        self.missing_keys = missing_keys or []
        if config_section:
            self.context.update({"config_section": config_section})
        if missing_keys:
            self.context.update({"missing_keys": missing_keys})


class EnvironmentVariableError(ConfigurationError):
    """Raised when required environment variables are missing."""

    def __init__(self, message: str, variable_name: str = None):
        super().__init__(message, "ENV_VAR_ERROR")
        self.variable_name = variable_name
        if variable_name:
            self.context.update({"variable_name": variable_name})


# Simulation-related exceptions
class SimulationError(BusinessAgentException):
    """Base class for simulation-related errors."""
    pass


class SimulationInitializationError(SimulationError):
    """Raised when simulation fails to initialize."""
    pass


class DataGenerationError(SimulationError):
    """Raised when data generation fails during simulation."""

    def __init__(self, message: str, data_type: str = None, period: str = None):
        super().__init__(message, "DATA_GENERATION_ERROR")
        self.data_type = data_type
        self.period = period
        if data_type:
            self.context.update({"data_type": data_type})
        if period:
            self.context.update({"period": period})


class BusinessProfileError(SimulationError):
    """Raised when business profile configuration is invalid."""
    pass


# API and external service exceptions
class ExternalServiceError(BusinessAgentException):
    """Base class for external service errors."""
    pass


class APIRateLimitError(ExternalServiceError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, message: str, service: str = None, retry_after: int = None):
        super().__init__(message, "API_RATE_LIMIT_ERROR")
        self.service = service
        self.retry_after = retry_after
        if service:
            self.context.update({"service": service})
        if retry_after:
            self.context.update({"retry_after": retry_after})


class ServiceUnavailableError(ExternalServiceError):
    """Raised when external services are unavailable."""

    def __init__(self, message: str, service: str = None, status_code: int = None):
        super().__init__(message, "SERVICE_UNAVAILABLE_ERROR")
        self.service = service
        self.status_code = status_code
        if service:
            self.context.update({"service": service})
        if status_code:
            self.context.update({"status_code": status_code})


# Business logic exceptions
class BusinessLogicError(BusinessAgentException):
    """Base class for business logic errors."""
    pass


class InsufficientDataError(BusinessLogicError):
    """Raised when there's insufficient data to make a decision."""

    def __init__(self, message: str, data_type: str = None, required_count: int = None, actual_count: int = None):
        super().__init__(message, "INSUFFICIENT_DATA_ERROR")
        self.data_type = data_type
        self.required_count = required_count
        self.actual_count = actual_count
        if data_type:
            self.context.update({"data_type": data_type})
        if required_count is not None:
            self.context.update({"required_count": required_count})
        if actual_count is not None:
            self.context.update({"actual_count": actual_count})


class BusinessRuleViolationError(BusinessLogicError):
    """Raised when business rules are violated."""

    def __init__(self, message: str, rule_name: str = None, rule_value: Any = None):
        super().__init__(message, "BUSINESS_RULE_VIOLATION")
        self.rule_name = rule_name
        self.rule_value = rule_value
        if rule_name:
            self.context.update({"rule_name": rule_name})
        if rule_value is not None:
            self.context.update({"rule_value": str(rule_value)})


class InventoryError(BusinessLogicError):
    """Raised for inventory-related business logic errors."""

    def __init__(self, message: str, item_sku: str = None, current_stock: int = None):
        super().__init__(message, "INVENTORY_ERROR")
        self.item_sku = item_sku
        self.current_stock = current_stock
        if item_sku:
            self.context.update({"item_sku": item_sku})
        if current_stock is not None:
            self.context.update({"current_stock": current_stock})


# Financial exceptions
class FinancialError(BusinessLogicError):
    """Base class for financial-related errors."""
    pass


class InsufficientFundsError(FinancialError):
    """Raised when there are insufficient funds for an operation."""

    def __init__(self, message: str, available_amount: float = None, required_amount: float = None):
        super().__init__(message, "INSUFFICIENT_FUNDS_ERROR")
        self.available_amount = available_amount
        self.required_amount = required_amount
        if available_amount is not None:
            self.context.update({"available_amount": available_amount})
        if required_amount is not None:
            self.context.update({"required_amount": required_amount})


class CashFlowError(FinancialError):
    """Raised for cash flow related issues."""
    pass


class AccountingAnomalyError(FinancialError):
    """Raised when accounting anomalies are detected."""

    def __init__(self, message: str, transaction_id: str = None, anomaly_score: float = None):
        super().__init__(message, "ACCOUNTING_ANOMALY_ERROR")
        self.transaction_id = transaction_id
        self.anomaly_score = anomaly_score
        if transaction_id:
            self.context.update({"transaction_id": transaction_id})
        if anomaly_score is not None:
            self.context.update({"anomaly_score": anomaly_score})


# Dashboard and UI exceptions
class DashboardError(BusinessAgentException):
    """Base class for dashboard-related errors."""
    pass


class DataVisualizationError(DashboardError):
    """Raised when data visualization fails."""

    def __init__(self, message: str, chart_type: str = None, data_count: int = None):
        super().__init__(message, "DATA_VISUALIZATION_ERROR")
        self.chart_type = chart_type
        self.data_count = data_count
        if chart_type:
            self.context.update({"chart_type": chart_type})
        if data_count is not None:
            self.context.update({"data_count": data_count})


class ReportGenerationError(DashboardError):
    """Raised when report generation fails."""
    pass


# System-level exceptions
class SystemError(BusinessAgentException):
    """Base class for system-level errors."""
    pass


class ResourceExhaustionError(SystemError):
    """Raised when system resources are exhausted."""

    def __init__(self, message: str, resource_type: str = None, current_usage: float = None, limit: float = None):
        super().__init__(message, "RESOURCE_EXHAUSTION_ERROR")
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        if resource_type:
            self.context.update({"resource_type": resource_type})
        if current_usage is not None:
            self.context.update({"current_usage": current_usage})
        if limit is not None:
            self.context.update({"limit": limit})


class TimeoutError(SystemError):
    """Raised when operations timeout."""

    def __init__(self, message: str, operation: str = None, timeout_seconds: int = None):
        super().__init__(message, "TIMEOUT_ERROR")
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        if operation:
            self.context.update({"operation": operation})
        if timeout_seconds is not None:
            self.context.update({"timeout_seconds": timeout_seconds})


# Recovery and retry helpers
class RecoverableError(BusinessAgentException):
    """Base class for errors that can be recovered from with retry logic."""

    def __init__(self, message: str, max_retries: int = 3, backoff_seconds: int = 1):
        super().__init__(message)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.context.update({
            "max_retries": max_retries,
            "backoff_seconds": backoff_seconds
        })


class NonRecoverableError(BusinessAgentException):
    """Base class for errors that cannot be recovered from."""
    pass


# Exception mapping for common error types
ERROR_TYPE_MAPPING = {
    "database": DatabaseError,
    "agent": AgentError,
    "configuration": ConfigurationError,
    "simulation": SimulationError,
    "external_service": ExternalServiceError,
    "business_logic": BusinessLogicError,
    "dashboard": DashboardError,
    "system": SystemError
}


def create_exception(error_type: str, message: str, **kwargs) -> BusinessAgentException:
    """
    Factory function to create appropriate exception based on error type.
    
    Args:
        error_type: Type of error (database, agent, configuration, etc.)
        message: Error message
        **kwargs: Additional context for the exception
    
    Returns:
        Appropriate exception instance
    """
    exception_class = ERROR_TYPE_MAPPING.get(error_type, BusinessAgentException)
    return exception_class(message, **kwargs)
