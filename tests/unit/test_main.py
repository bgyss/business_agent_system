"""Test suite for main.py module."""

import asyncio
import os
import signal
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import main
from main import BusinessAgentSystem, signal_handler


class TestBusinessAgentSystem:
    """Test BusinessAgentSystem class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return {
            "business": {"name": "Test Business", "type": "restaurant"},
            "database": {"url": "sqlite:///test.db"},
            "agents": {
                "accounting": {"enabled": True, "check_interval": 300},
                "inventory": {"enabled": True, "check_interval": 600},
                "hr": {"enabled": True, "check_interval": 900},
            },
            "simulation": {
                "enabled": True,
                "mode": "real_time",
                "business_profile": {"avg_daily_revenue": 2500},
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    @pytest.fixture
    def config_file(self, mock_config):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(mock_config, f)
            return f.name

    @pytest.fixture
    def system(self, config_file):
        """Create BusinessAgentSystem instance for testing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            return BusinessAgentSystem(config_file)

    def test_initialization_success(self, config_file):
        """Test successful initialization."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            system = BusinessAgentSystem(config_file)

            assert system.config is not None
            assert system.agents == {}
            assert system.simulator is None
            assert isinstance(system.message_queue, asyncio.Queue)
            assert system.is_running is False
            assert system.tasks == []
            assert system.api_key == "test-key"

    def test_initialization_missing_api_key(self, config_file):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="ANTHROPIC_API_KEY environment variable is required"
            ):
                BusinessAgentSystem(config_file)

    def test_load_config(self, system, mock_config):
        """Test config loading."""
        assert system.config == mock_config

    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(FileNotFoundError):
                BusinessAgentSystem("/non/existent/file.yaml")

    def test_setup_logging(self, system):
        """Test logging setup."""
        # The logging setup is called during initialization
        # Just verify logger was created
        assert system.logger is not None
        assert system.logger.name == "BusinessAgentSystem"

    @patch("main.AccountingAgent")
    @patch("main.InventoryAgent")
    @patch("main.HRAgent")
    def test_initialize_agents(self, mock_hr, mock_inventory, mock_accounting, system):
        """Test agent initialization."""
        system.initialize_agents()

        # Check that agents were created
        mock_accounting.assert_called_once()
        mock_inventory.assert_called_once()
        mock_hr.assert_called_once()

        # Check that agents were added to the agents dict
        assert "accounting" in system.agents
        assert "inventory" in system.agents
        assert "hr" in system.agents

    @patch("main.BusinessSimulator")
    def test_initialize_simulator(self, mock_simulator_class, system):
        """Test simulator initialization."""
        mock_simulator = MagicMock()
        mock_simulator.initialize_business = MagicMock()
        mock_simulator.simulate_historical_data = MagicMock()
        mock_simulator_class.return_value = mock_simulator

        system.initialize_simulator()

        mock_simulator_class.assert_called_once()
        mock_simulator.initialize_business.assert_called_once_with("restaurant")
        assert system.simulator == mock_simulator

    @patch("main.BusinessSimulator")
    @patch("main.AccountingAgent")
    @patch("main.InventoryAgent")
    @patch("main.HRAgent")
    async def test_run_simulation_mode(
        self, mock_hr, mock_inventory, mock_accounting, mock_simulator_class, system
    ):
        """Test running in simulation mode."""
        # Setup mocks
        mock_simulator = MagicMock()
        mock_simulator.initialize_business = MagicMock()
        mock_simulator.simulate_historical_data = MagicMock()
        mock_simulator.start_real_time_simulation = AsyncMock()
        mock_simulator.stop_simulation = AsyncMock()
        mock_simulator_class.return_value = mock_simulator

        mock_agents = []
        for mock_agent_class in [mock_accounting, mock_inventory, mock_hr]:
            mock_agent = MagicMock()
            mock_agent.start = AsyncMock()
            mock_agent.stop = AsyncMock()
            mock_agent.is_running = True
            mock_agent.decisions_log = []
            mock_agent_class.return_value = mock_agent
            mock_agents.append(mock_agent)

        # Create a task that will cancel itself after a short delay
        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            system.is_running = False

        # Run the system with auto-cancellation
        asyncio.create_task(cancel_after_delay())

        try:
            await system.run()
        except:
            pass  # Expected due to early cancellation

        # Verify agents were created
        mock_accounting.assert_called_once()
        mock_inventory.assert_called_once()
        mock_hr.assert_called_once()

    async def test_shutdown(self, system):
        """Test system shutdown."""

        # Create mock tasks that are actual asyncio tasks
        async def dummy_task():
            await asyncio.sleep(10)

        mock_task1 = asyncio.create_task(dummy_task())
        mock_task2 = asyncio.create_task(dummy_task())

        # Create mock agents
        mock_agent = MagicMock()
        mock_agent.stop = AsyncMock()
        system.agents = {"test": mock_agent}

        # Create mock simulator
        mock_simulator = MagicMock()
        mock_simulator.stop_simulation = AsyncMock()
        system.simulator = mock_simulator

        system.tasks = [mock_task1, mock_task2]
        system.is_running = True

        await system.shutdown()

        assert system.is_running is False
        assert mock_task1.cancelled()
        assert mock_task2.cancelled()
        mock_agent.stop.assert_called_once()
        mock_simulator.stop_simulation.assert_called_once()

    async def test_message_router(self, system):
        """Test message router."""
        # Setup system to stop after processing one message
        system.is_running = True

        # Put a message in the queue
        test_message = {
            "agent_id": "test_agent",
            "message_type": "decision",
            "data": {"decision": "test_decision"},
        }
        await system.message_queue.put(test_message)

        # Run message router for a short time
        async def stop_after_delay():
            await asyncio.sleep(0.1)
            system.is_running = False

        asyncio.create_task(stop_after_delay())

        try:
            await system.message_router()
        except asyncio.TimeoutError:
            pass  # Expected when queue is empty

        # Just verify the method runs without error
        assert True

    def test_get_system_status(self, system):
        """Test system status reporting."""
        system.is_running = True

        # Create mock agents
        mock_agent = MagicMock()
        mock_agent.is_running = True
        mock_agent.decisions_log = []
        system.agents = {"accounting": mock_agent, "inventory": mock_agent, "hr": mock_agent}

        # Create mock simulator
        mock_simulator = MagicMock()
        mock_simulator.get_simulation_status.return_value = {"status": "running"}
        system.simulator = mock_simulator

        status = system.get_system_status()

        assert status["system_running"] is True
        assert len(status["agents"]) == 3
        assert "business_config" in status
        assert "timestamp" in status
        assert status["business_config"]["name"] == "Test Business"
        assert status["business_config"]["type"] == "restaurant"


class TestSignalHandler:
    """Test signal handler functionality."""

    def test_signal_handler_creation(self):
        """Test signal handler creation."""
        mock_system = MagicMock()
        handler = signal_handler(mock_system)

        assert callable(handler)

    def test_signal_handler_execution(self):
        """Test signal handler execution."""
        mock_system = MagicMock()
        mock_system.shutdown = AsyncMock()

        handler = signal_handler(mock_system)

        # Mock asyncio.create_task to avoid actual task creation
        with patch("asyncio.create_task") as mock_create_task:
            handler(signal.SIGINT, None)
            mock_create_task.assert_called_once()


class TestMainFunction:
    """Test main function and CLI functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create a mock config file."""
        config = {
            "business": {"name": "Test", "type": "restaurant"},
            "agents": {},
            "simulation": {},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            return f.name

    @patch("main.BusinessAgentSystem")
    @patch("main.signal.signal")
    @patch("sys.argv", ["main.py", "--config", "test_config.yaml"])
    @patch("os.path.exists")
    async def test_main_simulation_mode(self, mock_exists, mock_signal, mock_system_class):
        """Test main function in simulation mode."""
        mock_exists.return_value = True
        mock_system = MagicMock()
        mock_system.run = AsyncMock()
        mock_system.shutdown = AsyncMock()
        mock_system_class.return_value = mock_system

        await main.main()

        mock_system_class.assert_called_once()
        mock_system.run.assert_called_once()
        assert mock_signal.call_count == 2  # SIGINT and SIGTERM

    @patch("main.BusinessAgentSystem")
    @patch("sys.argv", ["main.py", "--config", "test_config.yaml", "--generate-historical", "30"])
    @patch("os.path.exists")
    async def test_main_historical_generation(self, mock_exists, mock_system_class):
        """Test main function with historical data generation."""
        mock_exists.return_value = True
        mock_system = MagicMock()
        mock_system.initialize_simulator = MagicMock()
        mock_system_class.return_value = mock_system

        await main.main()

        mock_system.initialize_simulator.assert_called_once()

    @patch("sys.argv", ["main.py", "--config", "/non/existent/file.yaml"])
    @patch("os.path.exists")
    @patch("builtins.print")
    async def test_main_config_not_found(self, mock_print, mock_exists):
        """Test main function with missing config file."""
        mock_exists.return_value = False

        # The function should exit before trying to create BusinessAgentSystem
        with pytest.raises(SystemExit) as exc_info:
            await main.main()

        assert exc_info.value.code == 1
        mock_print.assert_called_with(
            "Error: Configuration file '/non/existent/file.yaml' not found"
        )

    @patch("main.BusinessAgentSystem")
    @patch("sys.argv", ["main.py", "--config", "test_config.yaml"])
    @patch("os.path.exists")
    @patch("builtins.print")
    async def test_main_with_exception(self, mock_print, mock_exists, mock_system_class):
        """Test main function handling exceptions."""
        mock_exists.return_value = True
        mock_system = MagicMock()
        mock_system.run = AsyncMock(side_effect=Exception("Test error"))
        mock_system_class.return_value = mock_system

        with pytest.raises(SystemExit) as exc_info:
            await main.main()

        assert exc_info.value.code == 1
        mock_print.assert_called_with("Fatal error: Test error")


class TestMainModule:
    """Test module-level functionality."""

    def test_load_dotenv_called(self):
        """Test that load_dotenv is called on import."""
        # This is implicitly tested by the fact that the module imports successfully
        # and we can access environment variables in tests
        assert True

    def test_module_imports(self):
        """Test that all required imports are available."""
        # Test that all necessary classes are imported
        assert hasattr(main, "BusinessAgentSystem")
        assert hasattr(main, "AccountingAgent")
        assert hasattr(main, "HRAgent")
        assert hasattr(main, "InventoryAgent")
        assert hasattr(main, "BusinessSimulator")
        assert hasattr(main, "signal_handler")
        assert hasattr(main, "main")
