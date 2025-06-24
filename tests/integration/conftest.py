"""
Integration test fixtures and configuration.
"""

import asyncio
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import create_engine, event

# Import system components
from main import BusinessAgentSystem
from models.agent_decisions import Base as AgentBase

# Import all models to ensure tables are created
from models.employee import Base as EmployeeBase
from models.financial import Base as FinancialBase
from models.inventory import Base as InventoryBase
from simulation.business_simulator import BusinessSimulator


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    db_url = f"sqlite:///{db_path}"

    # Create engine and tables with foreign key support for SQLite
    engine = create_engine(db_url, echo=False)

    # Enable foreign key support for SQLite
    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    # Create all tables from all model bases
    FinancialBase.metadata.create_all(bind=engine)
    InventoryBase.metadata.create_all(bind=engine)
    EmployeeBase.metadata.create_all(bind=engine)
    AgentBase.metadata.create_all(bind=engine)

    yield db_url

    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def test_config(temp_db):
    """Create a test configuration dictionary."""
    return {
        "business": {
            "name": "Test Restaurant",
            "type": "restaurant",
            "description": "Test restaurant for integration testing",
        },
        "database": {"url": temp_db, "echo": False},
        "agents": {
            "accounting": {
                "enabled": True,
                "check_interval": 1,  # Fast for testing
                "anomaly_threshold": 0.25,
                "alert_thresholds": {
                    "cash_low": 1000,
                    "receivables_overdue": 30,
                    "payables_overdue": 7,
                },
            },
            "inventory": {
                "enabled": True,
                "check_interval": 1,
                "low_stock_multiplier": 1.3,
                "reorder_lead_time": 3,
                "consumption_analysis_days": 30,
            },
            "hr": {
                "enabled": True,
                "check_interval": 1,
                "overtime_threshold": 8,
                "max_labor_cost_percentage": 0.32,
                "scheduling_buffer_hours": 2,
            },
        },
        "simulation": {
            "enabled": True,
            "mode": "real_time",
            "simulation_interval": 1,  # Fast for testing
            "duration_minutes": 0,  # No auto-stop for testing
            "speed_multiplier": 10.0,  # Fast simulation
            "historical_days": 5,  # Small dataset for testing
            "business_profile": {
                "avg_daily_revenue": 1000,
                "revenue_variance": 0.25,
                "avg_transaction_size": 25,
                "seasonal_factors": dict.fromkeys(range(1, 13), 1.0),
                "customer_patterns": {
                    "monday": 0.8,
                    "tuesday": 0.9,
                    "wednesday": 1.0,
                    "thursday": 1.1,
                    "friday": 1.3,
                    "saturday": 1.4,
                    "sunday": 1.0,
                },
            },
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "anthropic": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "temperature": 0.1,
        },
    }


@pytest.fixture
def temp_config_file(test_config):
    """Create a temporary config file."""
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(config_fd, "w") as f:
        yaml.dump(test_config, f)

    yield config_path

    try:
        os.unlink(config_path)
    except OSError:
        pass


@pytest.fixture
def mock_anthropic_client():
    """Mock the Anthropic client to avoid API calls during testing."""

    def create_mock_response(prompt_text=""):
        """Create appropriate mock response based on prompt content."""
        mock_response = Mock()
        mock_response.content = [Mock()]

        # Provide context-aware responses based on prompt keywords
        if "anomaly" in prompt_text.lower() or "financial" in prompt_text.lower():
            mock_response.content[0].text = (
                "DECISION: No anomalies detected in financial data.\n"
                "CONFIDENCE: 0.85\n"
                "REASONING: Transaction patterns appear normal for this time period."
            )
        elif "inventory" in prompt_text.lower() or "stock" in prompt_text.lower():
            mock_response.content[0].text = (
                "DECISION: Stock levels are adequate. No immediate reorder needed.\n"
                "CONFIDENCE: 0.90\n"
                "REASONING: Current inventory levels above reorder thresholds."
            )
        elif "hr" in prompt_text.lower() or "employee" in prompt_text.lower():
            mock_response.content[0].text = (
                "DECISION: Staffing levels appropriate. No schedule adjustments needed.\n"
                "CONFIDENCE: 0.88\n"
                "REASONING: Labor costs within target parameters."
            )
        else:
            mock_response.content[0].text = (
                "DECISION: Analysis complete. No immediate action required.\n"
                "CONFIDENCE: 0.85\n"
                "REASONING: System operating within normal parameters."
            )

        return mock_response

    mock_client = Mock()
    mock_client.messages.create.side_effect = lambda **kwargs: create_mock_response(
        str(kwargs.get("messages", [{}])[-1].get("content", ""))
    )

    with patch("agents.base_agent.Anthropic") as mock_anthropic:
        mock_anthropic.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    test_env = {"ANTHROPIC_API_KEY": "test-api-key-for-testing"}

    with patch.dict(os.environ, test_env, clear=False):
        yield test_env


@pytest.fixture
def business_system(temp_config_file, mock_anthropic_client, mock_env_vars):
    """Create a BusinessAgentSystem instance for testing."""
    system = BusinessAgentSystem(temp_config_file)
    system.initialize_agents()

    yield system

    # Cleanup - for sync tests, we don't need to await
    # The running_system fixture handles async cleanup


@pytest_asyncio.fixture
async def running_system(business_system):
    """Start the business system and yield it running."""
    # Set system as running (like run() method does)
    business_system.is_running = True

    # Start agents but not the full run loop (which would run indefinitely)
    try:
        await business_system.start_agents()

        # Start simulator if enabled
        business_system.initialize_simulator()

        yield business_system

    finally:
        # Cleanup - ensure proper shutdown even if test fails
        business_system.is_running = False
        if hasattr(business_system, "shutdown"):
            await business_system.shutdown()
        else:
            # Fallback cleanup for agents
            for agent in business_system.agents.values():
                if hasattr(agent, "is_running"):
                    agent.is_running = False


@pytest.fixture
def simulator(test_config, temp_db):
    """Create a BusinessSimulator instance for testing."""
    sim_config = test_config["simulation"]
    simulator = BusinessSimulator(sim_config, temp_db)
    return simulator


# Removed event_loop fixture to avoid pytest-asyncio conflicts


class IntegrationTestHelper:
    """Helper class for integration testing."""

    @staticmethod
    async def wait_for_decisions(agent, expected_count=1, timeout=5):
        """Wait for agent to make decisions with better timeout handling."""
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.1
        max_checks = int(timeout / check_interval)

        for _ in range(max_checks):
            current_decisions = getattr(agent, "decisions_log", [])
            if len(current_decisions) >= expected_count:
                return current_decisions[-expected_count:]
            await asyncio.sleep(check_interval)

        # Final check and informative error
        current_count = len(getattr(agent, "decisions_log", []))
        raise TimeoutError(
            f"Agent {agent.agent_id} made {current_count} decisions, expected {expected_count} within {timeout}s"
        )

    @staticmethod
    async def send_test_message(system, message_content):
        """Send a test message through the system."""
        await system.message_queue.put(message_content)
        # Give time for message processing
        await asyncio.sleep(0.5)

    @staticmethod
    def verify_database_tables(db_url):
        """Verify that all expected database tables exist."""
        from sqlalchemy import create_engine, inspect

        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        expected_tables = [
            "accounts",
            "transactions",
            "accounts_receivable",
            "accounts_payable",
            "items",
            "stock_movements",
            "suppliers",
            "purchase_orders",
            "purchase_order_items",
            "employees",
            "time_records",
            "schedules",
            "agent_decisions",
        ]

        for table in expected_tables:
            assert table in tables, f"Expected table {table} not found in database"

        return tables


@pytest.fixture
def integration_helper():
    """Provide integration test helper."""
    return IntegrationTestHelper()
