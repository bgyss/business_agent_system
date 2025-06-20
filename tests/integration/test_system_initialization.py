"""
Integration tests for system initialization and configuration.
"""
import os
import tempfile

import pytest
import yaml

from main import BusinessAgentSystem


class TestSystemInitialization:
    """Test system initialization and configuration loading."""

    def test_config_loading_valid_file(self, temp_config_file, mock_env_vars):
        """Test loading a valid configuration file."""
        system = BusinessAgentSystem(temp_config_file)

        assert system.config is not None
        assert system.config["business"]["name"] == "Test Restaurant"
        assert system.config["business"]["type"] == "restaurant"
        assert system.api_key == "test-api-key-for-testing"

    def test_config_loading_missing_file(self, mock_env_vars):
        """Test error handling for missing configuration file."""
        with pytest.raises(FileNotFoundError):
            BusinessAgentSystem("/nonexistent/config.yaml")

    def test_config_loading_invalid_yaml(self, mock_env_vars):
        """Test error handling for invalid YAML configuration."""
        # Create invalid YAML file
        config_fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(config_fd, 'w') as f:
            f.write("invalid: yaml: content: [")

        try:
            with pytest.raises(yaml.YAMLError):
                BusinessAgentSystem(config_path)
        finally:
            os.unlink(config_path)

    def test_missing_api_key(self, temp_config_file):
        """Test error handling for missing API key."""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
            BusinessAgentSystem(temp_config_file)

    def test_agent_initialization_all_enabled(self, business_system):
        """Test initialization of all agents when enabled."""
        assert len(business_system.agents) == 3
        assert "accounting" in business_system.agents
        assert "inventory" in business_system.agents
        assert "hr" in business_system.agents

        # Verify agent configuration
        accounting_agent = business_system.agents["accounting"]
        assert accounting_agent.agent_id == "accounting_agent"
        assert accounting_agent.config["check_interval"] == 1

    def test_agent_initialization_selective_enable(self, temp_config_file, mock_env_vars):
        """Test initialization with only some agents enabled."""
        # Modify config to disable some agents
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        config["agents"]["inventory"]["enabled"] = False
        config["agents"]["hr"]["enabled"] = False

        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f)

        system = BusinessAgentSystem(temp_config_file)
        system.initialize_agents()

        assert len(system.agents) == 1
        assert "accounting" in system.agents
        assert "inventory" not in system.agents
        assert "hr" not in system.agents

    def test_simulator_initialization_enabled(self, business_system):
        """Test simulator initialization when enabled."""
        business_system.initialize_simulator()

        assert business_system.simulator is not None
        assert business_system.simulator.financial_generator is not None
        assert business_system.simulator.inventory_simulator is not None

    def test_simulator_initialization_disabled(self, temp_config_file, mock_env_vars):
        """Test simulator initialization when disabled."""
        # Modify config to disable simulation
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        config["simulation"]["enabled"] = False

        with open(temp_config_file, 'w') as f:
            yaml.dump(config, f)

        system = BusinessAgentSystem(temp_config_file)
        system.initialize_agents()
        system.initialize_simulator()

        assert system.simulator is None

    def test_database_initialization(self, integration_helper, temp_db):
        """Test database table creation."""
        # Verify all expected tables are created
        tables = integration_helper.verify_database_tables(temp_db)
        assert len(tables) >= 13  # Minimum expected tables

    def test_logging_configuration(self, business_system):
        """Test logging configuration."""
        import logging

        # Get the system logger
        logger = logging.getLogger("BusinessAgentSystem")

        # Verify logger is configured
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_message_queue_initialization(self, business_system):
        """Test message queue initialization."""
        import asyncio

        assert business_system.message_queue is not None
        assert isinstance(business_system.message_queue, asyncio.Queue)

    def test_system_status_initial(self, business_system):
        """Test system status when initially created."""
        status = business_system.get_system_status()

        assert status["system_running"] is False
        assert len(status["agents"]) == 3
        assert status["business_config"]["name"] == "Test Restaurant"
        assert status["business_config"]["type"] == "restaurant"

        # All agents should be not running initially
        for agent_name, agent_status in status["agents"].items():
            assert agent_status["running"] is False
            assert agent_status["decisions"] == 0


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""

    def test_minimal_valid_config(self, temp_db, mock_env_vars):
        """Test system with minimal valid configuration."""
        minimal_config = {
            "business": {"name": "Test", "type": "restaurant"},
            "database": {"url": temp_db},
            "agents": {
                "accounting": {"enabled": True, "check_interval": 300}
            },
            "simulation": {"enabled": False}
        }

        config_fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(config_fd, 'w') as f:
            yaml.dump(minimal_config, f)

        try:
            system = BusinessAgentSystem(config_path)
            system.initialize_agents()

            assert len(system.agents) == 1
            assert "accounting" in system.agents
        finally:
            os.unlink(config_path)

    def test_config_with_missing_sections(self, temp_db, mock_env_vars):
        """Test configuration with missing optional sections."""
        incomplete_config = {
            "business": {"name": "Test", "type": "restaurant"},
            "database": {"url": temp_db},
            "agents": {}  # Empty agents section
        }

        config_fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(config_fd, 'w') as f:
            yaml.dump(incomplete_config, f)

        try:
            system = BusinessAgentSystem(config_path)
            system.initialize_agents()

            # Should handle missing agent configs gracefully
            assert len(system.agents) == 0
        finally:
            os.unlink(config_path)

    def test_config_with_invalid_business_type(self, temp_db, mock_env_vars):
        """Test configuration with invalid business type."""
        invalid_config = {
            "business": {"name": "Test", "type": "invalid_type"},
            "database": {"url": temp_db},
            "agents": {"accounting": {"enabled": True}},
            "simulation": {"enabled": True}
        }

        config_fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(config_fd, 'w') as f:
            yaml.dump(invalid_config, f)

        try:
            system = BusinessAgentSystem(config_path)
            system.initialize_agents()

            # Should still initialize but with default business profile
            system.initialize_simulator()
            assert system.simulator is not None
        finally:
            os.unlink(config_path)

    def test_database_url_validation(self, mock_env_vars):
        """Test database URL validation."""
        config_with_invalid_db = {
            "business": {"name": "Test", "type": "restaurant"},
            "database": {"url": "invalid://database/url"},
            "agents": {"accounting": {"enabled": True}},
            "simulation": {"enabled": False}
        }

        config_fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(config_fd, 'w') as f:
            yaml.dump(config_with_invalid_db, f)

        try:
            system = BusinessAgentSystem(config_path)
            # Should raise error when trying to initialize agents with invalid DB URL
            with pytest.raises(Exception):
                system.initialize_agents()
        finally:
            os.unlink(config_path)
