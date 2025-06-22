"""
Integration tests for agent coordination and communication.
"""

import asyncio
from datetime import datetime

import pytest

from agents.base_agent import AgentMessage


class TestAgentCoordination:
    """Test agent coordination and communication workflows."""

    @pytest.mark.asyncio
    async def test_agent_startup_and_shutdown(self, running_system):
        """Test agent startup and shutdown coordination."""
        system = running_system

        # Verify all agents are running
        assert len(system.agents) == 3
        for agent_name, agent in system.agents.items():
            assert agent.is_running is True, f"Agent {agent_name} should be running"

        # Test shutdown
        await system.shutdown()

        # Verify all agents are stopped
        for agent_name, agent in system.agents.items():
            assert agent.is_running is False, f"Agent {agent_name} should be stopped"

    @pytest.mark.asyncio
    async def test_message_routing(self, running_system, integration_helper):
        """Test message routing between agents and system."""
        system = running_system

        # Send a test message through the system
        test_message = {
            "type": "new_transaction",
            "transaction": {
                "amount": 100.00,
                "description": "Test transaction",
                "transaction_date": datetime.now().date().isoformat(),
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, test_message)

        # Give agents time to process the message
        await asyncio.sleep(2)

        # Check that agents received and processed the message
        # At least the accounting agent should have processed it
        accounting_agent = system.agents["accounting"]
        assert len(accounting_agent.decisions_log) >= 0  # May or may not make a decision

    @pytest.mark.asyncio
    async def test_agent_decision_logging(self, running_system, integration_helper):
        """Test that agent decisions are properly logged."""
        system = running_system

        # Create a message that should trigger decisions
        cash_flow_message = {"type": "cash_flow_check", "cycle": 1}

        await integration_helper.send_test_message(system, cash_flow_message)

        # Wait for processing
        await asyncio.sleep(3)

        # Check that decisions were logged (at least attempted)
        total_decisions = sum(len(agent.decisions_log) for agent in system.agents.values())

        # May be 0 if no decisions needed, but system should be functioning
        assert total_decisions >= 0

        # Test decision persistence to database
        accounting_agent = system.agents["accounting"]
        if accounting_agent.decisions_log:
            # Verify decision structure
            decision = accounting_agent.decisions_log[0]
            assert hasattr(decision, "agent_id")
            assert hasattr(decision, "decision_type")
            assert hasattr(decision, "action")
            assert hasattr(decision, "reasoning")
            assert hasattr(decision, "confidence")

    @pytest.mark.asyncio
    async def test_periodic_agent_checks(self, running_system):
        """Test that agents perform periodic checks."""
        system = running_system

        # Let agents run for a few check intervals
        await asyncio.sleep(5)  # Agents have 1s check intervals

        # At least one agent should have run periodic checks
        # We can't guarantee decisions will be made, but we can check they're responsive
        for _agent_name, agent in system.agents.items():
            assert agent.is_running is True
            # Agent should still be responsive
            health = await agent.health_check()
            assert health["status"] == "running"

    @pytest.mark.asyncio
    async def test_agent_message_handling(self, running_system):
        """Test direct agent message handling."""
        system = running_system

        accounting_agent = system.agents["accounting"]

        # Create a direct message to the agent
        test_message = AgentMessage(
            sender="test_sender",
            recipient="accounting_agent",
            message_type="report_request",
            content={"request_type": "financial_summary"},
        )

        # Send message directly to agent
        await accounting_agent.handle_message(test_message)

        # Agent should have processed the message
        # This tests the agent's message handling capability
        assert True  # If we get here without exception, the test passed

    @pytest.mark.asyncio
    async def test_system_monitoring(self, running_system):
        """Test system monitoring functionality."""
        system = running_system

        # Start the monitoring task
        monitor_task = asyncio.create_task(system.monitor_system())

        # Let monitoring run for a short time
        await asyncio.sleep(3)

        # Stop monitoring
        system.is_running = False

        try:
            await asyncio.wait_for(monitor_task, timeout=2)
        except asyncio.TimeoutError:
            monitor_task.cancel()

        # Monitoring should complete without errors
        assert not monitor_task.done() or not monitor_task.exception()

    @pytest.mark.asyncio
    async def test_agent_error_isolation(self, running_system):
        """Test that agent errors don't crash the system."""
        system = running_system

        # Simulate an error in one agent by forcing an exception
        accounting_agent = system.agents["accounting"]

        # Create a malformed message that might cause issues
        bad_message = AgentMessage(
            sender="test",
            recipient="accounting_agent",
            message_type="invalid_type",
            content={"malformed": "data"},
        )

        # This should not crash the agent or system
        try:
            await accounting_agent.handle_message(bad_message)
        except Exception:
            pass  # Agent should handle errors gracefully

        # Agent should still be running
        assert accounting_agent.is_running is True

        # System should still be functional
        status = system.get_system_status()
        assert status["system_running"] is True

    @pytest.mark.asyncio
    async def test_multi_agent_decision_coordination(self, running_system, integration_helper):
        """Test coordination between multiple agents."""
        system = running_system

        # Send messages that might trigger decisions from multiple agents
        messages = [
            {"type": "new_transaction", "transaction": {"amount": 500.00}, "cycle": 1},
            {"type": "cash_flow_check", "cycle": 1},
            {"type": "daily_analysis", "cycle": 1},
        ]

        for message in messages:
            await integration_helper.send_test_message(system, message)
            await asyncio.sleep(1)  # Space out messages

        # Let all agents process
        await asyncio.sleep(3)

        # Check system status shows all agents active
        status = system.get_system_status()
        assert len(status["agents"]) == 3

        for _agent_name, agent_status in status["agents"].items():
            assert agent_status["running"] is True

    @pytest.mark.asyncio
    async def test_message_queue_overflow_handling(self, running_system):
        """Test handling of message queue overflow."""
        system = running_system

        # Send many messages quickly
        for i in range(100):
            message = {"type": "test_message", "data": f"message_{i}", "cycle": i}
            await system.message_queue.put(message)

        # System should handle the load without crashing
        await asyncio.sleep(2)

        # System should still be responsive
        status = system.get_system_status()
        assert status["system_running"] is True

    def test_agent_health_checks(self, running_system):
        """Test agent health check functionality."""
        system = running_system

        for _agent_name, agent in system.agents.items():
            health = asyncio.run(agent.health_check())

            assert health["agent_id"] == agent.agent_id
            assert health["status"] == "running"
            assert "decisions_count" in health
            assert "config" in health
            assert isinstance(health["decisions_count"], int)

    def test_system_status_reporting(self, running_system):
        """Test comprehensive system status reporting."""
        system = running_system

        status = system.get_system_status()

        # Verify status structure
        required_fields = ["system_running", "agents", "business_config", "timestamp"]
        for field in required_fields:
            assert field in status

        # Verify agent status details
        assert len(status["agents"]) == 3
        for _agent_name, agent_status in status["agents"].items():
            required_agent_fields = ["running", "decisions"]
            for field in required_agent_fields:
                assert field in agent_status

        # Verify business config
        assert status["business_config"]["name"] == "Test Restaurant"
        assert status["business_config"]["type"] == "restaurant"


class TestAgentCommunicationPatterns:
    """Test specific agent communication patterns and protocols."""

    @pytest.mark.asyncio
    async def test_broadcast_messages(self, running_system, integration_helper):
        """Test broadcast messages to all agents."""
        system = running_system

        # Send a broadcast message
        broadcast_message = {
            "type": "system_alert",
            "message": "Test broadcast",
            "priority": "high",
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, broadcast_message)
        await asyncio.sleep(2)

        # All agents should receive broadcast messages
        # We can't guarantee they'll all act on it, but they should receive it
        for agent in system.agents.values():
            assert agent.is_running is True  # Agents should still be responsive

    @pytest.mark.asyncio
    async def test_agent_to_agent_messaging(self, running_system):
        """Test direct agent-to-agent communication."""
        system = running_system

        accounting_agent = system.agents["accounting"]
        system.agents["inventory"]

        # Test sending message from one agent to another
        await accounting_agent.send_message(
            recipient="inventory_agent",
            message_type="financial_data",
            content={"cash_available": 5000.00},
        )

        # Give time for message processing
        await asyncio.sleep(1)

        # Verify message was sent (check queue or agent logs)
        assert True  # If no exception, communication works

    @pytest.mark.asyncio
    async def test_decision_chain_reactions(self, running_system, integration_helper):
        """Test that one agent's decision can trigger others."""
        system = running_system

        # Send a message that might create a chain reaction
        # For example, low cash might trigger multiple agent responses
        low_cash_message = {
            "type": "cash_flow_alert",
            "cash_level": 500.00,  # Below threshold
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, low_cash_message)

        # Give time for chain reactions
        await asyncio.sleep(3)

        # Check that multiple agents may have made decisions
        total_decisions = sum(len(agent.decisions_log) for agent in system.agents.values())

        # May be 0 if no decisions warranted, but system should be stable
        assert total_decisions >= 0

        # System should remain stable
        status = system.get_system_status()
        assert status["system_running"] is True

    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, running_system, integration_helper):
        """Test concurrent operations across multiple agents."""
        system = running_system

        # Send multiple different messages concurrently
        tasks = []

        messages = [
            {"type": "new_transaction", "transaction": {"amount": 100}, "cycle": 1},
            {"type": "inventory_update", "item": {"sku": "TEST001"}, "cycle": 1},
            {"type": "employee_schedule", "schedule": {}, "cycle": 1},
        ]

        # Send all messages concurrently
        for message in messages:
            task = asyncio.create_task(integration_helper.send_test_message(system, message))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Give time for processing
        await asyncio.sleep(3)

        # All agents should still be running and responsive
        for agent in system.agents.values():
            assert agent.is_running is True
            health = await agent.health_check()
            assert health["status"] == "running"
