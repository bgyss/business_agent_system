"""
Pytest configuration and fixtures for unit tests
"""

import os
import sys
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import AgentDecision

# Remove custom event_loop fixture to use pytest-asyncio's default


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response"""

    def _create_response(text: str = "Test Claude response"):
        mock_response = Mock()
        mock_response.content = [Mock(text=text)]
        return mock_response

    return _create_response


@pytest.fixture
def sample_agent_decision():
    """Create a sample agent decision for testing"""

    def _create_decision(
        agent_id: str = "test_agent",
        decision_type: str = "test_decision",
        confidence: float = 0.8,
        **kwargs,
    ):
        return AgentDecision(
            agent_id=agent_id,
            decision_type=decision_type,
            context=kwargs.get("context", {"test": "data"}),
            reasoning=kwargs.get("reasoning", "Test reasoning"),
            action=kwargs.get("action", "Test action"),
            confidence=confidence,
            timestamp=kwargs.get("timestamp", datetime.utcnow()),
        )

    return _create_decision


@pytest.fixture
def mock_transaction():
    """Create a mock financial transaction"""

    def _create_transaction(
        amount: Decimal = Decimal("100.00"),
        transaction_type: str = "INCOME",
        category: str = "sales",
        **kwargs,
    ):
        return Mock(
            id=kwargs.get("id", 1),
            amount=amount,
            transaction_type=transaction_type,
            category=category,
            transaction_date=kwargs.get("transaction_date", datetime.now()),
            description=kwargs.get("description", "Test transaction"),
            reference_number=kwargs.get("reference_number", "REF-001"),
        )

    return _create_transaction


@pytest.fixture
def mock_inventory_item():
    """Create a mock inventory item"""

    def _create_item(current_stock: int = 50, reorder_point: int = 20, **kwargs):
        return Mock(
            id=kwargs.get("id", 1),
            name=kwargs.get("name", "Test Item"),
            sku=kwargs.get("sku", "TEST-001"),
            current_stock=current_stock,
            reorder_point=reorder_point,
            minimum_stock=kwargs.get("minimum_stock", 10),
            reorder_quantity=kwargs.get("reorder_quantity", 100),
            unit_cost=kwargs.get("unit_cost", Decimal("10.00")),
            status=kwargs.get("status", "ACTIVE"),
            expiry_days=kwargs.get("expiry_days", None),
        )

    return _create_item


@pytest.fixture
def mock_employee():
    """Create a mock employee"""

    def _create_employee(
        employee_id: int = 1, first_name: str = "John", last_name: str = "Doe", **kwargs
    ):
        return Mock(
            id=employee_id,
            first_name=first_name,
            last_name=last_name,
            position=kwargs.get("position", "Server"),
            department=kwargs.get("department", "Restaurant"),
            hourly_rate=kwargs.get("hourly_rate", Decimal("15.00")),
            status=kwargs.get("status", "ACTIVE"),
            hire_date=kwargs.get("hire_date", date.today()),
        )

    return _create_employee


@pytest.fixture
def mock_time_record():
    """Create a mock time record"""

    def _create_record(
        employee_id: int = 1, record_type: str = "CLOCK_IN", timestamp: datetime = None, **kwargs
    ):
        return Mock(
            id=kwargs.get("id", 1),
            employee_id=employee_id,
            record_type=record_type,
            timestamp=timestamp or datetime.now(),
            location=kwargs.get("location", "Main Location"),
            notes=kwargs.get("notes", ""),
        )

    return _create_record


@pytest.fixture
def mock_database_session():
    """Create a mock database session with common query patterns"""

    def _create_session():
        session = Mock()

        # Setup common query patterns
        query_mock = Mock()
        session.query.return_value = query_mock

        filter_mock = Mock()
        query_mock.filter.return_value = filter_mock

        # Add common query chain methods
        filter_mock.all.return_value = []
        filter_mock.first.return_value = None
        filter_mock.count.return_value = 0
        filter_mock.order_by.return_value = filter_mock

        # Session management
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()

        return session

    return _create_session


@pytest.fixture
def agent_base_config():
    """Base configuration for all agents"""
    return {"check_interval": 300, "api_timeout": 30, "max_retries": 3}


@pytest.fixture
def accounting_config(agent_base_config):
    """Configuration specific to accounting agent"""
    config = agent_base_config.copy()
    config.update(
        {
            "anomaly_threshold": 0.25,
            "alert_thresholds": {
                "cash_low": 1000,
                "receivables_overdue": 30,
                "payables_overdue": 7,
            },
        }
    )
    return config


@pytest.fixture
def inventory_config(agent_base_config):
    """Configuration specific to inventory agent"""
    config = agent_base_config.copy()
    config.update(
        {"low_stock_multiplier": 1.2, "reorder_lead_time": 7, "consumption_analysis_days": 30}
    )
    return config


@pytest.fixture
def hr_config(agent_base_config):
    """Configuration specific to HR agent"""
    config = agent_base_config.copy()
    config.update(
        {"overtime_threshold": 40, "max_labor_cost_percentage": 0.30, "scheduling_buffer_hours": 2}
    )
    return config


@pytest.fixture(autouse=True)
def mock_logging():
    """Mock logging to reduce test output noise"""
    with patch("agents.base_agent.logging"):
        yield


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client that can be configured for different responses"""
    with patch("agents.base_agent.Anthropic") as mock_anthropic:
        # Default response
        mock_response = Mock()
        mock_response.content = [Mock(text="Default Claude response")]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        yield mock_anthropic


@pytest.fixture
def mock_sqlalchemy():
    """Mock SQLAlchemy components"""
    with patch("agents.base_agent.create_engine") as mock_engine, patch(
        "agents.base_agent.sessionmaker"
    ) as mock_sessionmaker:

        # Setup session mock
        session_mock = Mock()
        mock_sessionmaker.return_value = session_mock

        yield {"engine": mock_engine, "sessionmaker": mock_sessionmaker, "session": session_mock}


class AgentTestHelper:
    """Helper class for common agent testing operations"""

    @staticmethod
    def create_decision_context(**kwargs):
        """Create a standardized decision context"""
        default_context = {"timestamp": datetime.now().isoformat(), "test_mode": True}
        default_context.update(kwargs)
        return default_context

    @staticmethod
    def assert_decision_structure(decision: AgentDecision, expected_type: str = None):
        """Assert that a decision has the proper structure"""
        assert isinstance(decision, AgentDecision)
        assert decision.agent_id is not None
        assert decision.decision_type is not None
        assert decision.action is not None
        assert decision.reasoning is not None
        assert 0.0 <= decision.confidence <= 1.0
        assert isinstance(decision.timestamp, datetime)

        if expected_type:
            assert decision.decision_type == expected_type

    @staticmethod
    def create_mock_query_chain(return_value=None, count_value=0):
        """Create a mock query chain for database testing"""
        query_mock = Mock()
        filter_mock = Mock()

        query_mock.filter.return_value = filter_mock
        filter_mock.filter.return_value = filter_mock  # Allow chaining
        filter_mock.all.return_value = return_value or []
        filter_mock.first.return_value = return_value[0] if return_value else None
        filter_mock.count.return_value = count_value
        filter_mock.order_by.return_value = filter_mock

        return query_mock


@pytest.fixture
def agent_test_helper():
    """Provide the AgentTestHelper class"""
    return AgentTestHelper
