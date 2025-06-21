"""
Test suite for utils.exceptions module
"""
import pytest
from typing import Dict, Any

from utils.exceptions import (
    BusinessAgentError,
    AgentInitializationError,
    AgentDecisionError,
    AgentCommunicationError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseTransactionError,
    ConfigurationError,
    ConfigurationFileError,
    ConfigurationValidationError,
    SimulationError,
    DataGenerationError,
    BusinessProfileError,
    ExternalServiceError,
    AnthropicAPIError,
    DashboardError,
    DataVisualizationError,
    ResourceExhaustionError,
    ValidationError,
    RetryableError,
    NonRetryableError,
    CircuitBreakerOpenError,
)


class TestBusinessAgentError:
    """Test the base BusinessAgentError class"""

    def test_basic_error_creation(self):
        """Test creating a basic error with just a message"""
        error = BusinessAgentError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}

    def test_error_with_code(self):
        """Test creating an error with an error code"""
        error = BusinessAgentError("Test error", error_code="E001")
        assert str(error) == "[E001] Test error"
        assert error.error_code == "E001"
        assert error.message == "Test error"

    def test_error_with_context(self):
        """Test creating an error with context data"""
        context = {"agent_id": "test_agent", "retry_count": 3}
        error = BusinessAgentError("Test error", context=context)
        assert error.context == context
        assert error.context["agent_id"] == "test_agent"
        assert error.context["retry_count"] == 3

    def test_error_with_code_and_context(self):
        """Test creating an error with both code and context"""
        context = {"operation": "data_processing"}
        error = BusinessAgentError("Processing failed", error_code="E002", context=context)
        assert str(error) == "[E002] Processing failed"
        assert error.error_code == "E002"
        assert error.context == context

    def test_error_inheritance(self):
        """Test that BusinessAgentError inherits from Exception"""
        error = BusinessAgentError("Test")
        assert isinstance(error, Exception)

    def test_empty_context_handling(self):
        """Test that None context is converted to empty dict"""
        error = BusinessAgentError("Test", context=None)
        assert error.context == {}


class TestAgentErrors:
    """Test agent-specific error classes"""

    def test_agent_initialization_error(self):
        """Test AgentInitializationError"""
        error = AgentInitializationError("Failed to initialize agent", error_code="INIT001")
        assert isinstance(error, BusinessAgentError)
        assert str(error) == "[INIT001] Failed to initialize agent"

    def test_agent_decision_error(self):
        """Test AgentDecisionError"""
        error = AgentDecisionError("Decision processing failed")
        assert isinstance(error, BusinessAgentError)
        assert str(error) == "Decision processing failed"

    def test_agent_communication_error(self):
        """Test AgentCommunicationError"""
        context = {"source_agent": "accounting", "target_agent": "inventory"}
        error = AgentCommunicationError("Message delivery failed", context=context)
        assert isinstance(error, BusinessAgentError)
        assert error.context["source_agent"] == "accounting"


class TestDatabaseErrors:
    """Test database-related error classes"""

    def test_database_error_base(self):
        """Test base DatabaseError"""
        error = DatabaseError("Database operation failed")
        assert isinstance(error, BusinessAgentError)

    def test_database_connection_error(self):
        """Test DatabaseConnectionError"""
        error = DatabaseConnectionError("Connection timeout", error_code="DB001")
        assert isinstance(error, DatabaseError)
        assert isinstance(error, BusinessAgentError)

    def test_database_query_error(self):
        """Test DatabaseQueryError"""
        context = {"query": "SELECT * FROM transactions", "table": "transactions"}
        error = DatabaseQueryError("Query execution failed", context=context)
        assert isinstance(error, DatabaseError)
        assert error.context["query"] == "SELECT * FROM transactions"

    def test_database_transaction_error(self):
        """Test DatabaseTransactionError"""
        error = DatabaseTransactionError("Transaction rollback failed")
        assert isinstance(error, DatabaseError)


class TestConfigurationErrors:
    """Test configuration-related error classes"""

    def test_configuration_error_base(self):
        """Test base ConfigurationError"""
        error = ConfigurationError("Configuration invalid")
        assert isinstance(error, BusinessAgentError)

    def test_configuration_file_error(self):
        """Test ConfigurationFileError"""
        context = {"file_path": "/config/restaurant.yaml"}
        error = ConfigurationFileError("Config file not found", context=context)
        assert isinstance(error, ConfigurationError)
        assert error.context["file_path"] == "/config/restaurant.yaml"

    def test_configuration_validation_error(self):
        """Test ConfigurationValidationError"""
        context = {"field": "api_key", "value": None}
        error = ConfigurationValidationError("Required field missing", context=context)
        assert isinstance(error, ConfigurationError)
        assert error.context["field"] == "api_key"


class TestSimulationErrors:
    """Test simulation-related error classes"""

    def test_simulation_error_base(self):
        """Test base SimulationError"""
        error = SimulationError("Simulation failed")
        assert isinstance(error, BusinessAgentError)

    def test_data_generation_error(self):
        """Test DataGenerationError"""
        context = {"generator": "financial", "period": "2023-01"}
        error = DataGenerationError("Failed to generate financial data", context=context)
        assert isinstance(error, SimulationError)
        assert error.context["generator"] == "financial"

    def test_business_profile_error(self):
        """Test BusinessProfileError"""
        error = BusinessProfileError("Invalid business profile parameters")
        assert isinstance(error, SimulationError)


class TestExternalServiceErrors:
    """Test external service error classes"""

    def test_external_service_error_base(self):
        """Test base ExternalServiceError"""
        error = ExternalServiceError("Service unavailable")
        assert isinstance(error, BusinessAgentError)

    def test_anthropic_api_error(self):
        """Test AnthropicAPIError"""
        context = {"status_code": 429, "request_id": "req_123"}
        error = AnthropicAPIError("Rate limit exceeded", context=context)
        assert isinstance(error, ExternalServiceError)
        assert error.context["status_code"] == 429


class TestDashboardErrors:
    """Test dashboard-related error classes"""

    def test_dashboard_error_base(self):
        """Test base DashboardError"""
        error = DashboardError("Dashboard initialization failed")
        assert isinstance(error, BusinessAgentError)

    def test_data_visualization_error(self):
        """Test DataVisualizationError"""
        context = {"chart_type": "line", "data_points": 1000}
        error = DataVisualizationError("Chart rendering failed", context=context)
        assert isinstance(error, DashboardError)
        assert error.context["chart_type"] == "line"


class TestMiscellaneousErrors:
    """Test other error classes"""

    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError"""
        context = {"resource": "memory", "current_usage": "95%"}
        error = ResourceExhaustionError("Memory exhausted", context=context)
        assert isinstance(error, BusinessAgentError)
        assert error.context["resource"] == "memory"

    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Data validation failed")
        assert isinstance(error, BusinessAgentError)


class TestRetryableErrors:
    """Test retry-related error classes"""

    def test_retryable_error(self):
        """Test RetryableError with retry_after"""
        error = RetryableError("Temporary failure", retry_after=30)
        assert isinstance(error, BusinessAgentError)
        assert error.retry_after == 30

    def test_retryable_error_without_retry_after(self):
        """Test RetryableError without retry_after"""
        error = RetryableError("Temporary failure")
        assert isinstance(error, BusinessAgentError)
        assert error.retry_after is None

    def test_retryable_error_with_context(self):
        """Test RetryableError with context and retry_after"""
        context = {"attempt": 2, "max_attempts": 5}
        error = RetryableError("Retry needed", retry_after=15, context=context, error_code="R001")
        assert error.retry_after == 15
        assert error.context["attempt"] == 2
        assert error.error_code == "R001"

    def test_non_retryable_error(self):
        """Test NonRetryableError"""
        error = NonRetryableError("Permanent failure")
        assert isinstance(error, BusinessAgentError)

    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError"""
        context = {"circuit": "anthropic_api", "failure_count": 5}
        error = CircuitBreakerOpenError("Circuit breaker is open", context=context)
        assert isinstance(error, ExternalServiceError)
        assert error.context["circuit"] == "anthropic_api"


class TestErrorHierarchy:
    """Test the error class hierarchy"""

    def test_all_errors_inherit_from_base(self):
        """Test that all error classes inherit from BusinessAgentError"""
        error_classes = [
            AgentInitializationError,
            AgentDecisionError,
            AgentCommunicationError,
            DatabaseError,
            DatabaseConnectionError,
            DatabaseQueryError,
            DatabaseTransactionError,
            ConfigurationError,
            ConfigurationFileError,
            ConfigurationValidationError,
            SimulationError,
            DataGenerationError,
            BusinessProfileError,
            ExternalServiceError,
            AnthropicAPIError,
            DashboardError,
            DataVisualizationError,
            ResourceExhaustionError,
            ValidationError,
            RetryableError,
            NonRetryableError,
            CircuitBreakerOpenError,
        ]

        for error_class in error_classes:
            error = error_class("Test message")
            assert isinstance(error, BusinessAgentError)
            assert isinstance(error, Exception)

    def test_database_error_hierarchy(self):
        """Test database error inheritance"""
        connection_error = DatabaseConnectionError("Connection failed")
        query_error = DatabaseQueryError("Query failed")
        transaction_error = DatabaseTransactionError("Transaction failed")

        assert isinstance(connection_error, DatabaseError)
        assert isinstance(query_error, DatabaseError)
        assert isinstance(transaction_error, DatabaseError)

    def test_configuration_error_hierarchy(self):
        """Test configuration error inheritance"""
        file_error = ConfigurationFileError("File not found")
        validation_error = ConfigurationValidationError("Invalid value")

        assert isinstance(file_error, ConfigurationError)
        assert isinstance(validation_error, ConfigurationError)

    def test_simulation_error_hierarchy(self):
        """Test simulation error inheritance"""
        data_error = DataGenerationError("Data generation failed")
        profile_error = BusinessProfileError("Profile invalid")

        assert isinstance(data_error, SimulationError)
        assert isinstance(profile_error, SimulationError)

    def test_external_service_error_hierarchy(self):
        """Test external service error inheritance"""
        anthropic_error = AnthropicAPIError("API error")
        circuit_error = CircuitBreakerOpenError("Circuit open")

        assert isinstance(anthropic_error, ExternalServiceError)
        assert isinstance(circuit_error, ExternalServiceError)

    def test_dashboard_error_hierarchy(self):
        """Test dashboard error inheritance"""
        viz_error = DataVisualizationError("Visualization failed")

        assert isinstance(viz_error, DashboardError)


if __name__ == "__main__":
    pytest.main([__file__])