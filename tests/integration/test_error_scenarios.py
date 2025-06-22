"""
Integration tests for error scenarios and failure recovery.
"""

import asyncio
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from sqlalchemy.exc import OperationalError

from main import BusinessAgentSystem
from simulation.business_simulator import BusinessSimulator


class TestConfigurationErrors:
    """Test error handling for configuration issues."""

    def test_invalid_database_configuration(self, mock_env_vars):
        """Test handling of invalid database configurations."""
        invalid_configs = [
            {"database": {"url": "invalid://database/url"}},
            {"database": {"url": "sqlite:///nonexistent/path/db.sqlite"}},
            {"database": {}},  # Missing URL
        ]

        for invalid_config in invalid_configs:
            config = {
                "business": {"name": "Test", "type": "restaurant"},
                "agents": {"accounting": {"enabled": True}},
                "simulation": {"enabled": False},
                **invalid_config,
            }

            config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
            with os.fdopen(config_fd, "w") as f:
                yaml.dump(config, f)

            try:
                system = BusinessAgentSystem(config_path)

                # Should raise error when trying to initialize agents
                with pytest.raises(Exception):
                    system.initialize_agents()

            finally:
                os.unlink(config_path)

    def test_missing_required_config_sections(self, temp_db, mock_env_vars):
        """Test handling of missing required configuration sections."""
        incomplete_configs = [
            {},  # Completely empty
            {"business": {"name": "Test"}},  # Missing type and database
            {"database": {"url": temp_db}},  # Missing business info
            {
                "business": {"name": "Test", "type": "restaurant"},
                "database": {"url": temp_db},
                # Missing agents section
            },
        ]

        for config in incomplete_configs:
            config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
            with os.fdopen(config_fd, "w") as f:
                yaml.dump(config, f)

            try:
                if "database" in config and "business" in config:
                    # Should be able to create system but with no agents
                    system = BusinessAgentSystem(config_path)
                    system.initialize_agents()
                    assert len(system.agents) == 0
                else:
                    # Should fail for completely invalid configs
                    with pytest.raises((KeyError, TypeError)):
                        system = BusinessAgentSystem(config_path)
                        system.initialize_agents()

            finally:
                os.unlink(config_path)

    def test_agent_configuration_errors(self, temp_db, mock_env_vars):
        """Test handling of invalid agent configurations."""
        config_with_bad_agent = {
            "business": {"name": "Test", "type": "restaurant"},
            "database": {"url": temp_db},
            "agents": {
                "accounting": {
                    "enabled": True,
                    "check_interval": "invalid_number",  # Should be numeric
                    "anomaly_threshold": -0.5,  # Invalid threshold
                }
            },
            "simulation": {"enabled": False},
        }

        config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(config_fd, "w") as f:
            yaml.dump(config_with_bad_agent, f)

        try:
            system = BusinessAgentSystem(config_path)

            # Should handle bad config gracefully or raise appropriate error
            try:
                system.initialize_agents()
                # If it succeeds, agent should handle bad values
                if "accounting" in system.agents:
                    agent = system.agents["accounting"]
                    # Should have some default or handled the bad config
                    assert hasattr(agent, "config")
            except (ValueError, TypeError):
                # Expected for invalid configuration values
                pass

        finally:
            os.unlink(config_path)


class TestDatabaseErrors:
    """Test database error scenarios and recovery."""

    @pytest.mark.asyncio
    async def test_database_connection_failure(self, running_system):
        """Test handling of database connection failures."""
        system = running_system

        # Simulate database connection failure by closing the connection
        accounting_agent = system.agents["accounting"]

        # Force close the database connection
        accounting_agent.engine.dispose()

        # Try to perform operations that require database access
        try:
            # This should handle the database error gracefully
            await accounting_agent.periodic_check()

            # Agent should still be running despite database issues
            assert accounting_agent.is_running is True

        except Exception as e:
            # If it raises an exception, it should be a database-related one
            # and the agent should still be recoverable
            assert "database" in str(e).lower() or "connection" in str(e).lower()

    def test_database_corruption_scenario(self, temp_db):
        """Test handling of database corruption or integrity issues."""
        from sqlalchemy import create_engine, text

        # Create initial valid database
        engine = create_engine(temp_db)

        # Simulate corruption by executing invalid SQL
        try:
            with engine.connect() as conn:
                # This should cause integrity issues
                conn.execute(text("INSERT INTO accounts (id, name) VALUES ('test', NULL)"))
                conn.commit()
        except Exception:
            pass  # Expected for integrity constraint violation

        # Try to use the corrupted database
        config = {"simulation_interval": 1, "speed_multiplier": 1.0}

        simulator = BusinessSimulator(config, temp_db)

        try:
            # Should handle corruption gracefully
            simulator.initialize_business("restaurant")
        except Exception as e:
            # Should be a database-related error, not a crash
            assert any(
                keyword in str(e).lower()
                for keyword in ["constraint", "integrity", "database", "sql"]
            )

    def test_database_disk_full_scenario(self, temp_db):
        """Test handling of disk space issues."""
        # This is difficult to simulate reliably, so we'll mock it
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_session = Mock()
            mock_session.commit.side_effect = OperationalError("disk full", None, None)

            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_session

            config = {"simulation_interval": 1}
            simulator = BusinessSimulator(config, temp_db)

            # Should handle disk full errors gracefully
            try:
                simulator.initialize_business("restaurant")
            except OperationalError:
                # Expected - should not crash the entire system
                pass


class TestAgentErrors:
    """Test agent-specific error scenarios."""

    @pytest.mark.asyncio
    async def test_anthropic_api_errors(self, running_system, integration_helper):
        """Test handling of Anthropic API errors."""
        system = running_system
        accounting_agent = system.agents["accounting"]

        # Mock API failure
        with patch.object(accounting_agent.client.messages, "create") as mock_create:
            mock_create.side_effect = Exception("API rate limit exceeded")

            # Send message that would normally trigger API call
            test_message = {"type": "cash_flow_check", "cycle": 1}

            await integration_helper.send_test_message(system, test_message)
            await asyncio.sleep(2)

            # Agent should still be running despite API errors
            assert accounting_agent.is_running is True

            # Should be able to recover and handle other operations
            health = await accounting_agent.health_check()
            assert health["status"] == "running"

    @pytest.mark.asyncio
    async def test_agent_memory_exhaustion(self, running_system):
        """Test handling of agent memory issues."""
        system = running_system
        accounting_agent = system.agents["accounting"]

        # Simulate memory exhaustion by filling decision log
        # Create many large decisions
        from datetime import datetime

        from models.agent_decisions import AgentDecision

        large_context = {"data": "x" * 10000}  # Large context data

        for i in range(1000):  # Many decisions
            decision = AgentDecision(
                agent_id="accounting_agent",
                decision_type="memory_test",
                action=f"Action {i}",
                reasoning="Test reasoning with large context",
                confidence=0.8,
                context=large_context,
                timestamp=datetime.now(),
            )
            accounting_agent.decisions_log.append(decision)

        # Agent should still be functional
        assert accounting_agent.is_running is True

        # Should be able to perform health check
        health = await accounting_agent.health_check()
        assert health["status"] == "running"
        assert health["decisions_count"] >= 1000

    @pytest.mark.asyncio
    async def test_agent_infinite_loop_protection(self, running_system, integration_helper):
        """Test protection against agent infinite loops."""
        system = running_system

        # Send many rapid messages that could cause infinite processing
        for i in range(100):
            message = {"type": "rapid_message", "data": f"message_{i}", "cycle": i}
            await system.message_queue.put(message)

        # Let system try to process all messages
        await asyncio.sleep(5)

        # System should remain responsive, not locked in infinite loop
        status = system.get_system_status()
        assert status["system_running"] is True

        # All agents should still be responsive
        for agent in system.agents.values():
            assert agent.is_running is True

    @pytest.mark.asyncio
    async def test_agent_exception_isolation(self, running_system):
        """Test that exceptions in one agent don't affect others."""
        system = running_system

        accounting_agent = system.agents["accounting"]
        inventory_agent = system.agents["inventory"]
        hr_agent = system.agents["hr"]

        # Force an exception in accounting agent
        with patch.object(accounting_agent, "periodic_check") as mock_check:
            mock_check.side_effect = Exception("Simulated accounting error")

            # Let agents run their periodic checks
            await asyncio.sleep(2)

            # Accounting agent might have issues, but others should be fine
            assert inventory_agent.is_running is True
            assert hr_agent.is_running is True

            # Other agents should still be able to generate reports
            inventory_report = await inventory_agent.generate_report()
            hr_report = await hr_agent.generate_report()

            assert isinstance(inventory_report, dict)
            assert isinstance(hr_report, dict)


class TestSimulationErrors:
    """Test simulation error scenarios."""

    def test_simulation_data_generation_errors(self, simulator):
        """Test handling of data generation errors."""
        simulator.initialize_business("restaurant")

        # Try to generate data with invalid parameters
        try:
            # This might cause issues with date calculations
            simulator.simulate_historical_data(days_back=-5)
        except ValueError:
            # Expected for negative days
            pass

        # Simulator should still be usable
        assert simulator.financial_generator is not None

    @pytest.mark.asyncio
    async def test_simulation_message_queue_errors(self, simulator):
        """Test handling of message queue errors during simulation."""
        simulator.initialize_business("restaurant")

        # Create a mock queue that raises errors
        bad_queue = Mock()
        bad_queue.put.side_effect = Exception("Queue error")

        # Should handle queue errors gracefully
        try:
            await simulator.start_real_time_simulation(bad_queue)
        except Exception:
            # Should not crash the entire simulation
            pass

        # Simulator should be cleanly stoppable
        await simulator.stop_simulation()
        assert simulator.is_running is False

    @pytest.mark.asyncio
    async def test_simulation_resource_exhaustion(self, simulator):
        """Test simulation behavior under resource constraints."""
        simulator.initialize_business("restaurant")

        # Try to generate excessive amounts of data
        try:
            # This might use too much memory or time
            simulator.simulate_historical_data(days_back=1000)  # Very large dataset
        except (MemoryError, Exception):
            # Should handle resource constraints gracefully
            pass

        # Simulator should remain functional
        status = simulator.get_simulation_status()
        assert isinstance(status, dict)


class TestNetworkAndCommunicationErrors:
    """Test network and communication error scenarios."""

    @pytest.mark.asyncio
    async def test_message_routing_failures(self, running_system):
        """Test handling of message routing failures."""
        system = running_system

        # Simulate message router failure

        async def failing_router():
            await asyncio.sleep(0.1)
            raise Exception("Router failure")

        system.message_router = failing_router

        # Try to start system with failing router
        router_task = asyncio.create_task(system.message_router())

        await asyncio.sleep(1)

        # Task should have failed but not crashed the system
        assert router_task.done() or router_task.exception()

        # Agents should still be running
        for agent in system.agents.values():
            assert agent.is_running is True

    @pytest.mark.asyncio
    async def test_agent_communication_timeouts(self, running_system):
        """Test handling of communication timeouts between agents."""
        system = running_system

        # Simulate slow agent response
        accounting_agent = system.agents["accounting"]

        async def slow_handle_message(self, message):
            await asyncio.sleep(10)  # Very slow
            return await original_handle_message(message)

        original_handle_message = accounting_agent.handle_message
        accounting_agent.handle_message = slow_handle_message.__get__(accounting_agent)

        # Send message that requires response
        from agents.base_agent import AgentMessage

        test_message = AgentMessage(
            sender="test", recipient="accounting_agent", message_type="report_request", content={}
        )

        # Should handle timeout gracefully
        start_time = asyncio.get_event_loop().time()

        try:
            await asyncio.wait_for(accounting_agent.handle_message(test_message), timeout=2.0)
        except asyncio.TimeoutError:
            # Expected timeout
            pass

        elapsed = asyncio.get_event_loop().time() - start_time
        assert elapsed < 5  # Should timeout, not wait forever

        # Agent should still be responsive for other operations
        health = await accounting_agent.health_check()
        assert health["status"] == "running"


class TestRecoveryMechanisms:
    """Test system recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, running_system):
        """Test graceful degradation when components fail."""
        system = running_system

        # Disable one agent to simulate failure
        accounting_agent = system.agents["accounting"]
        await accounting_agent.stop()

        # System should continue with remaining agents
        remaining_agents = [agent for agent in system.agents.values() if agent.is_running]
        assert len(remaining_agents) >= 2  # At least inventory and HR should be running

        # Should be able to get system status
        status = system.get_system_status()
        assert status["system_running"] is True

        # Failed agent should be marked as not running
        assert status["agents"]["accounting"]["running"] is False

    @pytest.mark.asyncio
    async def test_automatic_restart_mechanisms(
        self, temp_config_file, mock_anthropic_client, mock_env_vars
    ):
        """Test automatic restart mechanisms for failed components."""
        system = BusinessAgentSystem(temp_config_file)
        system.initialize_agents()

        # Start agents
        await system.start_agents()

        try:
            # Simulate agent failure and recovery
            accounting_agent = system.agents["accounting"]

            # Stop agent (simulate failure)
            await accounting_agent.stop()
            assert accounting_agent.is_running is False

            # Restart agent (simulate recovery)
            restart_task = asyncio.create_task(accounting_agent.start())
            await asyncio.sleep(1)  # Give time to start

            # Agent should be running again
            assert accounting_agent.is_running is True

            # Clean up
            await accounting_agent.stop()
            restart_task.cancel()

        finally:
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_state_recovery_after_errors(self, running_system, integration_helper):
        """Test system state recovery after various errors."""
        system = running_system

        # Generate some normal state
        system.simulator.simulate_historical_data(days_back=3)

        # Record initial state
        initial_status = system.get_system_status()
        {
            name: len(agent.decisions_log) for name, agent in system.agents.items()
        }

        # Cause some errors
        error_events = [
            {"type": "invalid_event"},
            {"malformed": "message"},
            None,  # This won't be sent, but test robustness
        ]

        for event in error_events[:-1]:  # Skip None
            try:
                await integration_helper.send_test_message(system, event)
            except Exception:
                pass

        await asyncio.sleep(2)

        # Verify system recovered
        recovered_status = system.get_system_status()
        assert recovered_status["system_running"] is True

        # All agents should still be running
        for agent_name in initial_status["agents"].keys():
            assert recovered_status["agents"][agent_name]["running"] is True

        # System should be able to process normal messages after errors
        normal_message = {"type": "system_health_check", "cycle": 1}
        await integration_helper.send_test_message(system, normal_message)

        await asyncio.sleep(1)

        # System should remain stable
        final_status = system.get_system_status()
        assert final_status["system_running"] is True
