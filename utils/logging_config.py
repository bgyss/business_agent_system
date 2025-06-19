"""
Structured logging configuration for the Business Agent Management System
"""
import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

from .exceptions import BusinessAgentError


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra context if available
        if hasattr(record, 'agent_id'):
            log_data['agent_id'] = record.agent_id
        if hasattr(record, 'business_type'):
            log_data['business_type'] = record.business_type
        if hasattr(record, 'decision_id'):
            log_data['decision_id'] = record.decision_id
        if hasattr(record, 'performance_metric'):
            log_data['performance_metric'] = record.performance_metric
        if hasattr(record, 'error_code'):
            log_data['error_code'] = record.error_code
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and not record.exc_info:
            log_data['stack_trace'] = traceback.format_stack()
        
        return json.dumps(log_data, default=str)


class BusinessAgentAdapter(logging.LoggerAdapter):
    """Logger adapter for adding business context to log messages"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Merge extra context with any context in kwargs
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        
        return msg, kwargs
    
    def log_decision(self, level: int, decision_id: str, message: str, **kwargs):
        """Log an agent decision with structured context"""
        extra = kwargs.get('extra', {})
        extra['decision_id'] = decision_id
        kwargs['extra'] = extra
        self.log(level, message, **kwargs)
    
    def log_performance(self, metric_name: str, value: float, unit: str = "ms", **kwargs):
        """Log a performance metric"""
        extra = kwargs.get('extra', {})
        extra['performance_metric'] = {
            'name': metric_name,
            'value': value,
            'unit': unit
        }
        kwargs['extra'] = extra
        self.info(f"Performance metric: {metric_name} = {value}{unit}", **kwargs)
    
    def log_error(self, error: Exception, message: Optional[str] = None, **kwargs):
        """Log an error with structured context"""
        extra = kwargs.get('extra', {})
        
        if isinstance(error, BusinessAgentError):
            extra['error_code'] = error.error_code
            extra['context'] = error.context
        
        error_message = message or f"Error occurred: {str(error)}"
        kwargs['extra'] = extra
        self.error(error_message, exc_info=error, **kwargs)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration for the business agent system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Whether to use structured JSON logging
        console_output: Whether to output to console
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(funcName)s(): %(message)s"
            }
        },
        "handlers": {},
        "loggers": {
            "business_agent_system": {
                "level": log_level,
                "handlers": [],
                "propagate": False
            },
            "agents": {
                "level": log_level,
                "handlers": [],
                "propagate": False
            },
            "simulation": {
                "level": log_level,
                "handlers": [],
                "propagate": False
            },
            "dashboard": {
                "level": log_level,
                "handlers": [],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": []
        }
    }
    
    handler_names = []
    
    # Console handler
    if console_output:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "structured" if structured else "standard",
            "stream": "ext://sys.stdout"
        }
        handler_names.append("console")
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "structured" if structured else "detailed",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8"
        }
        handler_names.append("file")
    
    # Error file handler (always structured for easier parsing)
    if log_file:
        error_log_file = log_path.parent / f"{log_path.stem}_errors{log_path.suffix}"
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "structured",
            "filename": str(error_log_file),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8"
        }
        handler_names.append("error_file")
    
    # Assign handlers to loggers
    for logger_name in ["business_agent_system", "agents", "simulation", "dashboard", "root"]:
        if logger_name in config["loggers"]:
            config["loggers"][logger_name]["handlers"] = handler_names
        else:
            config[logger_name]["handlers"] = handler_names
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Set up some third-party library logging levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str, **context) -> BusinessAgentAdapter:
    """
    Get a logger with business context
    
    Args:
        name: Logger name
        **context: Additional context to include in all log messages
    
    Returns:
        BusinessAgentAdapter instance
    """
    logger = logging.getLogger(name)
    return BusinessAgentAdapter(logger, context)


def get_agent_logger(agent_id: str, business_type: Optional[str] = None) -> BusinessAgentAdapter:
    """
    Get a logger specifically for an agent
    
    Args:
        agent_id: Agent identifier
        business_type: Type of business (restaurant, retail, etc.)
    
    Returns:
        BusinessAgentAdapter instance with agent context
    """
    context = {"agent_id": agent_id}
    if business_type:
        context["business_type"] = business_type
    
    return get_logger(f"agents.{agent_id}", **context)


def get_simulation_logger(business_type: str) -> BusinessAgentAdapter:
    """
    Get a logger for business simulation
    
    Args:
        business_type: Type of business being simulated
    
    Returns:
        BusinessAgentAdapter instance with simulation context
    """
    return get_logger("simulation", business_type=business_type)


def get_dashboard_logger() -> BusinessAgentAdapter:
    """Get a logger for dashboard operations"""
    return get_logger("dashboard")


# Context managers for structured logging

class LogContext:
    """Context manager for adding structured context to logs"""
    
    def __init__(self, logger: BusinessAgentAdapter, **context):
        self.logger = logger
        self.context = context
        self.original_extra = logger.extra.copy()
    
    def __enter__(self):
        self.logger.extra.update(self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.extra = self.original_extra
        
        # Log any exceptions that occurred
        if exc_type and issubclass(exc_type, Exception):
            self.logger.log_error(exc_val, "Exception occurred in log context")


class PerformanceContext:
    """Context manager for performance logging"""
    
    def __init__(self, logger: BusinessAgentAdapter, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000
        
        if exc_type:
            self.logger.log_performance(
                f"{self.operation_name}_duration",
                duration_ms,
                "ms"
            )
            self.logger.error(f"Operation failed: {self.operation_name} after {duration_ms:.2f}ms")
        else:
            self.logger.log_performance(
                f"{self.operation_name}_duration",
                duration_ms,
                "ms"
            )
            self.logger.debug(f"Operation completed: {self.operation_name} in {duration_ms:.2f}ms")