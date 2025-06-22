"""
Test suite for utils.logging_config module
"""
import json
import logging
import logging.handlers
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from utils.logging_config import (
    StructuredFormatter,
    BusinessAgentAdapter,
    setup_logging,
    get_logger,
    get_agent_logger,
    get_simulation_logger,
    get_dashboard_logger,
    LogContext,
    PerformanceContext,
)
from utils.exceptions import BusinessAgentError


class TestStructuredFormatter:
    """Test StructuredFormatter class"""

    def setup_method(self):
        """Setup for each test method"""
        self.formatter = StructuredFormatter()

    def test_basic_log_formatting(self):
        """Test basic log record formatting"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 123
        assert "timestamp" in log_data

    def test_agent_context_formatting(self):
        """Test formatting with agent context"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=123,
            msg="Agent message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        record.agent_id = "test_agent"
        record.business_type = "restaurant"

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["agent_id"] == "test_agent"
        assert log_data["business_type"] == "restaurant"

    def test_decision_context_formatting(self):
        """Test formatting with decision context"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=123,
            msg="Decision message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        record.decision_id = "decision_123"

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["decision_id"] == "decision_123"

    def test_performance_context_formatting(self):
        """Test formatting with performance metrics"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=123,
            msg="Performance message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        record.performance_metric = {"name": "operation_time", "value": 100.5, "unit": "ms"}

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["performance_metric"]["name"] == "operation_time"
        assert log_data["performance_metric"]["value"] == 100.5
        assert log_data["performance_metric"]["unit"] == "ms"

    def test_error_context_formatting(self):
        """Test formatting with error context"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=123,
            msg="Error message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        record.error_code = "E001"
        record.context = {"operation": "data_processing"}

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["error_code"] == "E001"
        assert log_data["context"]["operation"] == "data_processing"

    def test_exception_formatting(self):
        """Test formatting with exception information"""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            exc_info = (type(e), e, e.__traceback__)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=123,
            msg="Error with exception",
            args=(),
            exc_info=exc_info
        )
        record.funcName = "test_function"
        record.module = "test_module"

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert "traceback" in log_data["exception"]

    def test_stack_trace_on_error(self):
        """Test stack trace inclusion for error level logs"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=123,
            msg="Error without exception",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"

        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)

        assert "stack_trace" in log_data
        assert isinstance(log_data["stack_trace"], list)


class TestBusinessAgentAdapter:
    """Test BusinessAgentAdapter class"""

    def setup_method(self):
        """Setup for each test method"""
        self.base_logger = logging.getLogger("test_logger")
        self.extra = {"agent_id": "test_agent", "business_type": "restaurant"}
        self.adapter = BusinessAgentAdapter(self.base_logger, self.extra)

    def test_initialization(self):
        """Test adapter initialization"""
        assert self.adapter.logger == self.base_logger
        assert self.adapter.extra == self.extra

    def test_process_method_merges_extra(self):
        """Test that process method merges extra context"""
        msg = "Test message"
        kwargs = {"extra": {"decision_id": "dec_123"}}

        processed_msg, processed_kwargs = self.adapter.process(msg, kwargs)

        assert processed_msg == msg
        assert processed_kwargs["extra"]["agent_id"] == "test_agent"
        assert processed_kwargs["extra"]["business_type"] == "restaurant"
        assert processed_kwargs["extra"]["decision_id"] == "dec_123"

    def test_process_method_creates_extra(self):
        """Test that process method creates extra if not present"""
        msg = "Test message"
        kwargs = {}

        processed_msg, processed_kwargs = self.adapter.process(msg, kwargs)

        assert processed_kwargs["extra"] == self.extra

    def test_log_decision(self):
        """Test log_decision method"""
        with patch.object(self.adapter, 'log') as mock_log:
            self.adapter.log_decision(logging.INFO, "dec_123", "Decision made")

            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == logging.INFO
            assert args[1] == "Decision made"
            assert kwargs["extra"]["decision_id"] == "dec_123"

    def test_log_performance(self):
        """Test log_performance method"""
        with patch.object(self.adapter, 'info') as mock_info:
            self.adapter.log_performance("operation_time", 123.45, "ms")

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert "Performance metric: operation_time = 123.45ms" in args[0]
            assert kwargs["extra"]["performance_metric"]["name"] == "operation_time"
            assert kwargs["extra"]["performance_metric"]["value"] == 123.45
            assert kwargs["extra"]["performance_metric"]["unit"] == "ms"

    def test_log_performance_default_unit(self):
        """Test log_performance with default unit"""
        with patch.object(self.adapter, 'info') as mock_info:
            self.adapter.log_performance("operation_time", 123.45)

            args, kwargs = mock_info.call_args
            assert kwargs["extra"]["performance_metric"]["unit"] == "ms"

    def test_log_error_with_business_agent_error(self):
        """Test log_error with BusinessAgentError"""
        error = BusinessAgentError(
            "Test error",
            error_code="E001",
            context={"operation": "test_op"}
        )

        with patch.object(self.adapter, 'error') as mock_error:
            self.adapter.log_error(error, "Custom error message")

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert args[0] == "Custom error message"
            assert kwargs["exc_info"] == error
            assert kwargs["extra"]["error_code"] == "E001"
            assert kwargs["extra"]["context"]["operation"] == "test_op"

    def test_log_error_with_regular_exception(self):
        """Test log_error with regular exception"""
        error = ValueError("Regular error")

        with patch.object(self.adapter, 'error') as mock_error:
            self.adapter.log_error(error)

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert args[0] == "Error occurred: Regular error"
            assert kwargs["exc_info"] == error

    def test_log_error_custom_context(self):
        """Test log_error with custom context"""
        error = ValueError("Test error")

        with patch.object(self.adapter, 'error') as mock_error:
            self.adapter.log_error(error, extra={"custom": "value"})

            args, kwargs = mock_error.call_args
            assert kwargs["extra"]["custom"] == "value"


class TestLoggingSetup:
    """Test logging setup functions"""

    def test_setup_logging_defaults(self):
        """Test setup_logging with default parameters"""
        with patch('logging.config.dictConfig') as mock_dict_config:
            setup_logging()

            mock_dict_config.assert_called_once()
            config = mock_dict_config.call_args[0][0]

            assert config["version"] == 1
            assert config["disable_existing_loggers"] is False
            assert "structured" in config["formatters"]
            assert "console" in config["handlers"]

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            with patch('logging.config.dictConfig') as mock_dict_config:
                setup_logging(log_file=str(log_file))

                config = mock_dict_config.call_args[0][0]
                assert "file" in config["handlers"]
                assert "error_file" in config["handlers"]

    def test_setup_logging_no_console(self):
        """Test setup_logging without console output"""
        with patch('logging.config.dictConfig') as mock_dict_config:
            setup_logging(console_output=False)

            config = mock_dict_config.call_args[0][0]
            assert "console" not in config["handlers"]

    def test_setup_logging_unstructured(self):
        """Test setup_logging with unstructured logging"""
        with patch('logging.config.dictConfig') as mock_dict_config:
            setup_logging(structured=False)

            config = mock_dict_config.call_args[0][0]
            console_handler = config["handlers"]["console"]
            assert console_handler["formatter"] == "standard"

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level"""
        with patch('logging.config.dictConfig') as mock_dict_config:
            setup_logging(log_level="DEBUG")

            config = mock_dict_config.call_args[0][0]
            assert config["loggers"]["agents"]["level"] == "DEBUG"


class TestLoggerFactories:
    """Test logger factory functions"""

    def test_get_logger(self):
        """Test get_logger function"""
        logger = get_logger("test_logger", custom_context="value")

        assert isinstance(logger, BusinessAgentAdapter)
        assert logger.extra["custom_context"] == "value"

    def test_get_agent_logger(self):
        """Test get_agent_logger function"""
        logger = get_agent_logger("test_agent", "restaurant")

        assert isinstance(logger, BusinessAgentAdapter)
        assert logger.extra["agent_id"] == "test_agent"
        assert logger.extra["business_type"] == "restaurant"

    def test_get_agent_logger_no_business_type(self):
        """Test get_agent_logger without business type"""
        logger = get_agent_logger("test_agent")

        assert logger.extra["agent_id"] == "test_agent"
        assert "business_type" not in logger.extra

    def test_get_simulation_logger(self):
        """Test get_simulation_logger function"""
        logger = get_simulation_logger("restaurant")

        assert isinstance(logger, BusinessAgentAdapter)
        assert logger.extra["business_type"] == "restaurant"

    def test_get_dashboard_logger(self):
        """Test get_dashboard_logger function"""
        logger = get_dashboard_logger()

        assert isinstance(logger, BusinessAgentAdapter)


class TestLogContext:
    """Test LogContext context manager"""

    def test_log_context_adds_context(self):
        """Test LogContext adds context during execution"""
        logger = get_logger("test_logger", initial="value")
        
        assert logger.extra["initial"] == "value"
        assert "temp" not in logger.extra

        with LogContext(logger, temp="context"):
            assert logger.extra["initial"] == "value"
            assert logger.extra["temp"] == "context"

        # Context should be restored
        assert logger.extra["initial"] == "value"
        assert "temp" not in logger.extra

    def test_log_context_with_exception(self):
        """Test LogContext logs exceptions"""
        logger = get_logger("test_logger")

        with patch.object(logger, 'log_error') as mock_log_error:
            try:
                with LogContext(logger, operation="test"):
                    raise ValueError("Test exception")
            except ValueError:
                pass

            mock_log_error.assert_called_once()
            args = mock_log_error.call_args[0]
            assert isinstance(args[0], ValueError)
            assert args[1] == "Exception occurred in log context"

    def test_log_context_no_exception_logging_for_non_exceptions(self):
        """Test LogContext doesn't log non-Exception types"""
        logger = get_logger("test_logger")

        with patch.object(logger, 'log_error') as mock_log_error:
            try:
                with LogContext(logger, operation="test"):
                    raise SystemExit(0)  # Not a subclass of Exception
            except SystemExit:
                pass

            mock_log_error.assert_not_called()


class TestPerformanceContext:
    """Test PerformanceContext context manager"""

    def test_performance_context_success(self):
        """Test PerformanceContext for successful operation"""
        logger = get_logger("test_logger")

        with patch.object(logger, 'debug') as mock_debug, \
             patch.object(logger, 'log_performance') as mock_log_perf:

            with PerformanceContext(logger, "test_operation"):
                pass

            # Should log start and completion
            assert mock_debug.call_count == 2
            mock_debug.assert_any_call("Starting operation: test_operation")

            # Should log performance
            mock_log_perf.assert_called_once()
            args = mock_log_perf.call_args[0]
            assert args[0] == "test_operation_duration"
            assert isinstance(args[1], float)
            assert args[2] == "ms"

    def test_performance_context_with_exception(self):
        """Test PerformanceContext with exception"""
        logger = get_logger("test_logger")

        with patch.object(logger, 'debug') as mock_debug, \
             patch.object(logger, 'log_performance') as mock_log_perf, \
             patch.object(logger, 'error') as mock_error:

            try:
                with PerformanceContext(logger, "test_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Should still log performance
            mock_log_perf.assert_called_once()

            # Should log error
            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            assert "Operation failed: test_operation" in error_msg

    @patch('time.perf_counter')
    def test_performance_context_timing(self, mock_perf_counter):
        """Test PerformanceContext timing calculation"""
        mock_perf_counter.side_effect = [100.0, 100.123]  # 123ms duration

        logger = get_logger("test_logger")

        with patch.object(logger, 'log_performance') as mock_log_perf:
            with PerformanceContext(logger, "test_operation"):
                pass

            args = mock_log_perf.call_args[0]
            assert abs(args[1] - 123.0) < 0.1  # ~123ms


class TestIntegration:
    """Integration tests for logging components"""

    def test_full_logging_workflow(self):
        """Test complete logging workflow with structured output"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Setup logging
            setup_logging(
                log_level="DEBUG",
                log_file=str(log_file),
                structured=True,
                console_output=False
            )

            # Get logger and log various types of messages
            logger = get_agent_logger("test_agent", "restaurant")

            logger.info("Test info message")
            logger.log_decision(logging.INFO, "dec_123", "Decision logged")
            logger.log_performance("operation_time", 150.5)

            error = BusinessAgentError("Test error", error_code="E001")
            logger.log_error(error, "Test error occurred")

            # Verify file was created and contains structured data
            assert log_file.exists()

            with open(log_file, 'r') as f:
                lines = f.readlines()

            assert len(lines) >= 4  # At least 4 log entries

            # Parse and verify structured logs
            for line in lines:
                log_data = json.loads(line.strip())
                assert "timestamp" in log_data
                assert "level" in log_data
                assert log_data["agent_id"] == "test_agent"
                assert log_data["business_type"] == "restaurant"

    def test_context_managers_together(self):
        """Test using LogContext and PerformanceContext together"""
        logger = get_logger("test_logger")

        with patch.object(logger, 'log_performance') as mock_log_perf:
            with LogContext(logger, operation="complex_operation"):
                with PerformanceContext(logger, "complex_operation"):
                    # Simulate work
                    pass

            mock_log_perf.assert_called_once()
            # Context should be merged properly
            assert logger.extra.get("operation") is None  # Should be restored

    def test_nested_contexts(self):
        """Test nested context managers"""
        logger = get_logger("test_logger", base="value")

        with LogContext(logger, level1="outer"):
            assert logger.extra["level1"] == "outer"

            with LogContext(logger, level2="inner"):
                assert logger.extra["level1"] == "outer"
                assert logger.extra["level2"] == "inner"

            # Inner context should be removed
            assert logger.extra["level1"] == "outer"
            assert "level2" not in logger.extra

        # All contexts should be restored
        assert logger.extra["base"] == "value"
        assert "level1" not in logger.extra


if __name__ == "__main__":
    pytest.main([__file__])