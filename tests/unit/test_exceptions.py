"""
Test suite for exceptions.py module
"""

from datetime import datetime

from exceptions import (
    ERROR_TYPE_MAPPING,
    AccountingAnomalyError,
    AgentCommunicationError,
    AgentDecisionError,
    AgentError,
    AgentInitializationError,
    APIRateLimitError,
    BusinessAgentException,
    BusinessLogicError,
    BusinessProfileError,
    BusinessRuleViolationError,
    CashFlowError,
    ClaudeAPIError,
    ConfigFileNotFoundError,
    ConfigurationError,
    ConfigValidationError,
    DashboardError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    DatabaseTransactionError,
    DataGenerationError,
    DataValidationError,
    DataVisualizationError,
    EnvironmentVariableError,
    ExternalServiceError,
    FinancialError,
    InsufficientDataError,
    InsufficientFundsError,
    InventoryError,
    NonRecoverableError,
    RecoverableError,
    ReportGenerationError,
    ResourceExhaustionError,
    ServiceUnavailableError,
    SimulationError,
    SimulationInitializationError,
    SystemError,
    TimeoutError,
    create_exception,
)


class TestBusinessAgentException:
    """Test base BusinessAgentException class"""

    def test_basic_initialization(self):
        """Test basic exception initialization"""
        message = "Test error message"
        exc = BusinessAgentException(message)

        assert str(exc) == message
        assert exc.message == message
        assert exc.error_code == "BusinessAgentException"
        assert exc.context == {}
        assert isinstance(exc.timestamp, datetime)

    def test_initialization_with_error_code(self):
        """Test exception initialization with custom error code"""
        message = "Test error"
        error_code = "CUSTOM_ERROR"
        exc = BusinessAgentException(message, error_code=error_code)

        assert exc.error_code == error_code

    def test_initialization_with_context(self):
        """Test exception initialization with context"""
        message = "Test error"
        context = {"key": "value", "number": 42}
        exc = BusinessAgentException(message, context=context)

        assert exc.context == context

    def test_to_dict(self):
        """Test exception serialization to dictionary"""
        message = "Test error"
        error_code = "TEST_ERROR"
        context = {"test": True}
        exc = BusinessAgentException(message, error_code=error_code, context=context)

        result = exc.to_dict()

        assert result["error_type"] == "BusinessAgentException"
        assert result["error_code"] == error_code
        assert result["message"] == message
        assert result["context"] == context
        assert "timestamp" in result
        # Verify timestamp format
        datetime.fromisoformat(result["timestamp"])


class TestAgentExceptions:
    """Test agent-related exceptions"""

    def test_agent_error_inheritance(self):
        """Test AgentError inherits from BusinessAgentException"""
        exc = AgentError("test")
        assert isinstance(exc, BusinessAgentException)
        assert exc.error_code == "AgentError"

    def test_agent_initialization_error(self):
        """Test AgentInitializationError"""
        exc = AgentInitializationError("Failed to initialize")
        assert isinstance(exc, AgentError)
        assert "Failed to initialize" in str(exc)

    def test_agent_decision_error(self):
        """Test AgentDecisionError"""
        exc = AgentDecisionError("Decision failed")
        assert isinstance(exc, AgentError)
        assert "Decision failed" in str(exc)

    def test_agent_communication_error(self):
        """Test AgentCommunicationError"""
        exc = AgentCommunicationError("Communication failed")
        assert isinstance(exc, AgentError)
        assert "Communication failed" in str(exc)

    def test_claude_api_error(self):
        """Test ClaudeAPIError with API response and retry count"""
        api_response = {"error": "rate_limit", "retry_after": 30}
        retry_count = 3
        exc = ClaudeAPIError("API call failed", api_response=api_response, retry_count=retry_count)

        assert isinstance(exc, AgentError)
        assert exc.error_code == "CLAUDE_API_ERROR"
        assert exc.api_response == api_response
        assert exc.retry_count == retry_count
        assert exc.context["api_response"] == api_response
        assert exc.context["retry_count"] == retry_count


class TestDatabaseExceptions:
    """Test database-related exceptions"""

    def test_database_error_inheritance(self):
        """Test DatabaseError inherits from BusinessAgentException"""
        exc = DatabaseError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_database_connection_error(self):
        """Test DatabaseConnectionError"""
        exc = DatabaseConnectionError("Connection failed")
        assert isinstance(exc, DatabaseError)

    def test_database_transaction_error(self):
        """Test DatabaseTransactionError with operation and table"""
        operation = "INSERT"
        table = "transactions"
        exc = DatabaseTransactionError("Transaction failed", operation=operation, table=table)

        assert isinstance(exc, DatabaseError)
        assert exc.error_code == "DB_TRANSACTION_ERROR"
        assert exc.operation == operation
        assert exc.table == table
        assert exc.context["operation"] == operation
        assert exc.context["table"] == table

    def test_database_integrity_error(self):
        """Test DatabaseIntegrityError"""
        exc = DatabaseIntegrityError("Integrity constraint violated")
        assert isinstance(exc, DatabaseError)

    def test_data_validation_error(self):
        """Test DataValidationError with field and value"""
        field = "email"
        value = "invalid-email"
        exc = DataValidationError("Invalid email format", field=field, value=value)

        assert isinstance(exc, DatabaseError)
        assert exc.error_code == "DATA_VALIDATION_ERROR"
        assert exc.field == field
        assert exc.value == value
        assert exc.context["field"] == field
        assert exc.context["value"] == str(value)


class TestConfigurationExceptions:
    """Test configuration-related exceptions"""

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from BusinessAgentException"""
        exc = ConfigurationError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError"""
        exc = ConfigFileNotFoundError("Config file missing")
        assert isinstance(exc, ConfigurationError)

    def test_config_validation_error(self):
        """Test ConfigValidationError with section and missing keys"""
        section = "database"
        missing_keys = ["url", "password"]
        exc = ConfigValidationError(
            "Missing config", config_section=section, missing_keys=missing_keys
        )

        assert isinstance(exc, ConfigurationError)
        assert exc.error_code == "CONFIG_VALIDATION_ERROR"
        assert exc.config_section == section
        assert exc.missing_keys == missing_keys
        assert exc.context["config_section"] == section
        assert exc.context["missing_keys"] == missing_keys

    def test_environment_variable_error(self):
        """Test EnvironmentVariableError"""
        var_name = "ANTHROPIC_API_KEY"
        exc = EnvironmentVariableError("Missing env var", variable_name=var_name)

        assert isinstance(exc, ConfigurationError)
        assert exc.error_code == "ENV_VAR_ERROR"
        assert exc.variable_name == var_name
        assert exc.context["variable_name"] == var_name


class TestSimulationExceptions:
    """Test simulation-related exceptions"""

    def test_simulation_error_inheritance(self):
        """Test SimulationError inherits from BusinessAgentException"""
        exc = SimulationError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_simulation_initialization_error(self):
        """Test SimulationInitializationError"""
        exc = SimulationInitializationError("Simulation init failed")
        assert isinstance(exc, SimulationError)

    def test_data_generation_error(self):
        """Test DataGenerationError with data type and period"""
        data_type = "financial"
        period = "2023-Q1"
        exc = DataGenerationError("Data generation failed", data_type=data_type, period=period)

        assert isinstance(exc, SimulationError)
        assert exc.error_code == "DATA_GENERATION_ERROR"
        assert exc.data_type == data_type
        assert exc.period == period
        assert exc.context["data_type"] == data_type
        assert exc.context["period"] == period

    def test_business_profile_error(self):
        """Test BusinessProfileError"""
        exc = BusinessProfileError("Invalid business profile")
        assert isinstance(exc, SimulationError)


class TestExternalServiceExceptions:
    """Test external service exceptions"""

    def test_external_service_error_inheritance(self):
        """Test ExternalServiceError inherits from BusinessAgentException"""
        exc = ExternalServiceError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_api_rate_limit_error(self):
        """Test APIRateLimitError with service and retry_after"""
        service = "anthropic"
        retry_after = 60
        exc = APIRateLimitError("Rate limit exceeded", service=service, retry_after=retry_after)

        assert isinstance(exc, ExternalServiceError)
        assert exc.error_code == "API_RATE_LIMIT_ERROR"
        assert exc.service == service
        assert exc.retry_after == retry_after
        assert exc.context["service"] == service
        assert exc.context["retry_after"] == retry_after

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError with service and status code"""
        service = "database"
        status_code = 503
        exc = ServiceUnavailableError("Service down", service=service, status_code=status_code)

        assert isinstance(exc, ExternalServiceError)
        assert exc.error_code == "SERVICE_UNAVAILABLE_ERROR"
        assert exc.service == service
        assert exc.status_code == status_code
        assert exc.context["service"] == service
        assert exc.context["status_code"] == status_code


class TestBusinessLogicExceptions:
    """Test business logic exceptions"""

    def test_business_logic_error_inheritance(self):
        """Test BusinessLogicError inherits from BusinessAgentException"""
        exc = BusinessLogicError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_insufficient_data_error(self):
        """Test InsufficientDataError with data type and counts"""
        data_type = "transactions"
        required_count = 10
        actual_count = 3
        exc = InsufficientDataError(
            "Not enough data",
            data_type=data_type,
            required_count=required_count,
            actual_count=actual_count,
        )

        assert isinstance(exc, BusinessLogicError)
        assert exc.error_code == "INSUFFICIENT_DATA_ERROR"
        assert exc.data_type == data_type
        assert exc.required_count == required_count
        assert exc.actual_count == actual_count
        assert exc.context["data_type"] == data_type
        assert exc.context["required_count"] == required_count
        assert exc.context["actual_count"] == actual_count

    def test_business_rule_violation_error(self):
        """Test BusinessRuleViolationError with rule name and value"""
        rule_name = "max_transaction_amount"
        rule_value = 10000
        exc = BusinessRuleViolationError(
            "Rule violated", rule_name=rule_name, rule_value=rule_value
        )

        assert isinstance(exc, BusinessLogicError)
        assert exc.error_code == "BUSINESS_RULE_VIOLATION"
        assert exc.rule_name == rule_name
        assert exc.rule_value == rule_value
        assert exc.context["rule_name"] == rule_name
        assert exc.context["rule_value"] == str(rule_value)

    def test_inventory_error(self):
        """Test InventoryError with SKU and stock"""
        item_sku = "SKU-001"
        current_stock = 5
        exc = InventoryError("Low stock", item_sku=item_sku, current_stock=current_stock)

        assert isinstance(exc, BusinessLogicError)
        assert exc.error_code == "INVENTORY_ERROR"
        assert exc.item_sku == item_sku
        assert exc.current_stock == current_stock
        assert exc.context["item_sku"] == item_sku
        assert exc.context["current_stock"] == current_stock


class TestFinancialExceptions:
    """Test financial exceptions"""

    def test_financial_error_inheritance(self):
        """Test FinancialError inherits from BusinessLogicError"""
        exc = FinancialError("test")
        assert isinstance(exc, BusinessLogicError)

    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError with amounts"""
        available_amount = 100.50
        required_amount = 200.75
        exc = InsufficientFundsError(
            "Not enough funds", available_amount=available_amount, required_amount=required_amount
        )

        assert isinstance(exc, FinancialError)
        assert exc.error_code == "INSUFFICIENT_FUNDS_ERROR"
        assert exc.available_amount == available_amount
        assert exc.required_amount == required_amount
        assert exc.context["available_amount"] == available_amount
        assert exc.context["required_amount"] == required_amount

    def test_cash_flow_error(self):
        """Test CashFlowError"""
        exc = CashFlowError("Cash flow issue")
        assert isinstance(exc, FinancialError)

    def test_accounting_anomaly_error(self):
        """Test AccountingAnomalyError with transaction ID and score"""
        transaction_id = "TXN-12345"
        anomaly_score = 0.95
        exc = AccountingAnomalyError(
            "Anomaly detected", transaction_id=transaction_id, anomaly_score=anomaly_score
        )

        assert isinstance(exc, FinancialError)
        assert exc.error_code == "ACCOUNTING_ANOMALY_ERROR"
        assert exc.transaction_id == transaction_id
        assert exc.anomaly_score == anomaly_score
        assert exc.context["transaction_id"] == transaction_id
        assert exc.context["anomaly_score"] == anomaly_score


class TestDashboardExceptions:
    """Test dashboard exceptions"""

    def test_dashboard_error_inheritance(self):
        """Test DashboardError inherits from BusinessAgentException"""
        exc = DashboardError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_data_visualization_error(self):
        """Test DataVisualizationError with chart type and data count"""
        chart_type = "line_chart"
        data_count = 100
        exc = DataVisualizationError("Viz failed", chart_type=chart_type, data_count=data_count)

        assert isinstance(exc, DashboardError)
        assert exc.error_code == "DATA_VISUALIZATION_ERROR"
        assert exc.chart_type == chart_type
        assert exc.data_count == data_count
        assert exc.context["chart_type"] == chart_type
        assert exc.context["data_count"] == data_count

    def test_report_generation_error(self):
        """Test ReportGenerationError"""
        exc = ReportGenerationError("Report failed")
        assert isinstance(exc, DashboardError)


class TestSystemExceptions:
    """Test system-level exceptions"""

    def test_system_error_inheritance(self):
        """Test SystemError inherits from BusinessAgentException"""
        exc = SystemError("test")
        assert isinstance(exc, BusinessAgentException)

    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError with resource info"""
        resource_type = "memory"
        current_usage = 95.5
        limit = 100.0
        exc = ResourceExhaustionError(
            "Resource exhausted",
            resource_type=resource_type,
            current_usage=current_usage,
            limit=limit,
        )

        assert isinstance(exc, SystemError)
        assert exc.error_code == "RESOURCE_EXHAUSTION_ERROR"
        assert exc.resource_type == resource_type
        assert exc.current_usage == current_usage
        assert exc.limit == limit
        assert exc.context["resource_type"] == resource_type
        assert exc.context["current_usage"] == current_usage
        assert exc.context["limit"] == limit

    def test_timeout_error(self):
        """Test TimeoutError with operation and timeout"""
        operation = "database_query"
        timeout_seconds = 30
        exc = TimeoutError(
            "Operation timed out", operation=operation, timeout_seconds=timeout_seconds
        )

        assert isinstance(exc, SystemError)
        assert exc.error_code == "TIMEOUT_ERROR"
        assert exc.operation == operation
        assert exc.timeout_seconds == timeout_seconds
        assert exc.context["operation"] == operation
        assert exc.context["timeout_seconds"] == timeout_seconds


class TestRecoveryExceptions:
    """Test recovery and retry related exceptions"""

    def test_recoverable_error(self):
        """Test RecoverableError with retry parameters"""
        max_retries = 5
        backoff_seconds = 2
        exc = RecoverableError(
            "Recoverable error", max_retries=max_retries, backoff_seconds=backoff_seconds
        )

        assert isinstance(exc, BusinessAgentException)
        assert exc.max_retries == max_retries
        assert exc.backoff_seconds == backoff_seconds
        assert exc.context["max_retries"] == max_retries
        assert exc.context["backoff_seconds"] == backoff_seconds

    def test_non_recoverable_error(self):
        """Test NonRecoverableError"""
        exc = NonRecoverableError("Fatal error")
        assert isinstance(exc, BusinessAgentException)


class TestExceptionMapping:
    """Test exception mapping and factory functions"""

    def test_error_type_mapping_contents(self):
        """Test ERROR_TYPE_MAPPING contains expected mappings"""
        expected_mappings = {
            "database": DatabaseError,
            "agent": AgentError,
            "configuration": ConfigurationError,
            "simulation": SimulationError,
            "external_service": ExternalServiceError,
            "business_logic": BusinessLogicError,
            "dashboard": DashboardError,
            "system": SystemError,
        }

        assert ERROR_TYPE_MAPPING == expected_mappings

    def test_create_exception_known_type(self):
        """Test create_exception with known error type"""
        message = "Database connection failed"
        exc = create_exception("database", message)

        assert isinstance(exc, DatabaseError)
        assert str(exc) == message

    def test_create_exception_unknown_type(self):
        """Test create_exception with unknown error type falls back to base"""
        message = "Unknown error"
        exc = create_exception("unknown_type", message)

        assert isinstance(exc, BusinessAgentException)
        assert str(exc) == message

    def test_create_exception_with_kwargs(self):
        """Test create_exception passes through kwargs"""
        message = "Database error"
        error_code = "CUSTOM_DB_ERROR"
        context = {"table": "users"}
        exc = create_exception("database", message, error_code=error_code, context=context)

        assert isinstance(exc, DatabaseError)
        assert exc.error_code == error_code
        assert exc.context == context


class TestExceptionEdgeCases:
    """Test edge cases and error conditions"""

    def test_exception_with_none_values(self):
        """Test exceptions handle None values gracefully"""
        exc = DataValidationError("Error", field=None, value=None)
        assert exc.field is None
        assert exc.value is None
        # Should not add None values to context
        assert "field" not in exc.context
        assert "value" not in exc.context

    def test_exception_with_zero_values(self):
        """Test exceptions handle zero values correctly"""
        exc = InsufficientDataError("Error", required_count=0, actual_count=0)
        assert exc.required_count == 0
        assert exc.actual_count == 0
        # Zero values should be added to context
        assert exc.context["required_count"] == 0
        assert exc.context["actual_count"] == 0

    def test_exception_context_update_preserves_existing(self):
        """Test that context updates preserve existing context"""
        initial_context = {"existing_key": "existing_value"}
        exc = BusinessAgentException("Error", context=initial_context)

        # Manually update context to simulate what happens in specialized exceptions
        exc.context.update({"new_key": "new_value"})

        assert exc.context["existing_key"] == "existing_value"
        assert exc.context["new_key"] == "new_value"

    def test_exception_str_representation(self):
        """Test string representation of exceptions"""
        message = "Test error message"
        exc = BusinessAgentException(message)
        assert str(exc) == message

    def test_exception_inheritance_chain(self):
        """Test inheritance chain is correct"""
        exc = InventoryError("Test")

        assert isinstance(exc, InventoryError)
        assert isinstance(exc, BusinessLogicError)
        assert isinstance(exc, BusinessAgentException)
        assert isinstance(exc, Exception)

    def test_timestamp_is_recent(self):
        """Test that exception timestamp is recent"""
        before = datetime.now()
        exc = BusinessAgentException("Test")
        after = datetime.now()

        assert before <= exc.timestamp <= after

    def test_to_dict_handles_complex_context(self):
        """Test to_dict handles complex context values"""
        complex_context = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
            "bool": True,
        }
        exc = BusinessAgentException("Test", context=complex_context)
        result = exc.to_dict()

        assert result["context"] == complex_context
