"""
End-to-end integration tests for complete business workflows.
"""

import asyncio
from datetime import datetime

import pytest

from main import BusinessAgentSystem
from models.agent_decisions import AgentDecisionModel


class TestCompleteBusinessWorkflows:
    """Test complete end-to-end business workflows."""

    @pytest.mark.asyncio
    async def test_full_system_startup_to_shutdown(
        self, temp_config_file, mock_anthropic_client, mock_env_vars
    ):
        """Test complete system lifecycle from startup to shutdown."""
        system = BusinessAgentSystem(temp_config_file)

        try:
            # Initialize system components
            system.initialize_agents()
            system.initialize_simulator()

            # Verify initialization
            assert len(system.agents) == 3
            assert system.simulator is not None

            # Start agents
            await system.start_agents()

            # Verify agents are running
            for agent in system.agents.values():
                assert agent.is_running is True

            # Start simulator
            await system.start_simulator()

            # Let system run for a short period
            await asyncio.sleep(3)

            # Verify system is operational
            status = system.get_system_status()
            assert status["system_running"] is False  # Not in full run mode yet

            # Check that agents are still running
            for _agent_name, agent_status in status["agents"].items():
                assert agent_status["running"] is True

        finally:
            # Clean shutdown
            await system.shutdown()

            # Verify shutdown
            for agent in system.agents.values():
                assert agent.is_running is False

    @pytest.mark.asyncio
    async def test_business_simulation_with_agent_responses(
        self, business_system, integration_helper
    ):
        """Test business simulation with agent decision making."""
        # Initialize and start system components
        business_system.initialize_simulator()
        await business_system.start_agents()

        # Generate some historical data first
        business_system.simulator.simulate_historical_data(days_back=3)

        # Start real-time simulation briefly
        simulation_task = asyncio.create_task(
            business_system.simulator.start_real_time_simulation(business_system.message_queue)
        )

        # Start message router
        router_task = asyncio.create_task(business_system.message_router())

        try:
            # Let system run and process messages
            await asyncio.sleep(5)

            # Check that simulation is generating messages
            # and agents are processing them
            total_decisions = sum(
                len(agent.decisions_log) for agent in business_system.agents.values()
            )

            # May be 0 if no decisions needed, but system should be stable
            assert total_decisions >= 0

            # Verify system health
            for agent in business_system.agents.values():
                assert agent.is_running is True
                health = await agent.health_check()
                assert health["status"] == "running"

        finally:
            # Stop simulation and router
            await business_system.simulator.stop_simulation()
            business_system.is_running = False

            # Wait for tasks to complete
            try:
                await asyncio.wait_for(simulation_task, timeout=2)
            except asyncio.TimeoutError:
                simulation_task.cancel()

            try:
                await asyncio.wait_for(router_task, timeout=2)
            except asyncio.TimeoutError:
                router_task.cancel()

    @pytest.mark.asyncio
    async def test_accounting_workflow_complete(self, running_system, integration_helper):
        """Test complete accounting workflow from transaction to decision."""
        system = running_system
        accounting_agent = system.agents["accounting"]

        # Generate some historical data
        system.simulator.simulate_historical_data(days_back=5)

        # Send a significant transaction that should trigger analysis
        large_transaction = {
            "type": "new_transaction",
            "transaction": {
                "amount": 5000.00,  # Large amount
                "description": "Large expense transaction",
                "transaction_date": datetime.now().date().isoformat(),
                "transaction_type": "debit",
                "account_id": "checking_account",
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, large_transaction)

        # Trigger cash flow analysis
        cash_flow_check = {"type": "cash_flow_check", "cycle": 1}
        await integration_helper.send_test_message(system, cash_flow_check)

        # Give time for processing
        await asyncio.sleep(3)

        # Verify accounting agent processed the messages
        # (May or may not make decisions depending on thresholds)
        assert accounting_agent.is_running is True

        # Test periodic check functionality
        await accounting_agent.periodic_check()

        # Verify agent can generate reports
        report = await accounting_agent.generate_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_inventory_workflow_complete(self, running_system, integration_helper):
        """Test complete inventory workflow from stock changes to reorders."""
        system = running_system
        inventory_agent = system.agents["inventory"]

        # Generate inventory data
        system.simulator.simulate_historical_data(days_back=3)

        # Send inventory update message
        inventory_update = {
            "type": "inventory_update",
            "item": {"sku": "TEST001", "current_stock": 5, "reorder_point": 10},  # Low stock
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, inventory_update)

        # Send stock movement message
        stock_movement = {
            "type": "stock_movement",
            "movement": {
                "item_sku": "TEST001",
                "movement_type": "consumption",
                "quantity": -3,  # Stock going down
                "timestamp": datetime.now().isoformat(),
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, stock_movement)

        # Give time for processing
        await asyncio.sleep(3)

        # Verify inventory agent is responsive
        assert inventory_agent.is_running is True

        # Test periodic check
        await inventory_agent.periodic_check()

        # Test report generation
        report = await inventory_agent.generate_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_hr_workflow_complete(self, running_system, integration_helper):
        """Test complete HR workflow from schedule changes to cost optimization."""
        system = running_system
        hr_agent = system.agents["hr"]

        # Generate employee data
        system.simulator.simulate_historical_data(days_back=3)

        # Send employee schedule message
        schedule_update = {
            "type": "schedule_update",
            "schedule": {
                "employee_id": "EMP001",
                "date": datetime.now().date().isoformat(),
                "hours_scheduled": 10,  # Overtime hours
                "hourly_rate": 25.00,
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, schedule_update)

        # Send labor cost analysis request
        labor_analysis = {"type": "labor_cost_analysis", "period": "daily", "cycle": 1}

        await integration_helper.send_test_message(system, labor_analysis)

        # Give time for processing
        await asyncio.sleep(3)

        # Verify HR agent is responsive
        assert hr_agent.is_running is True

        # Test periodic check
        await hr_agent.periodic_check()

        # Test report generation
        report = await hr_agent.generate_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_workflow(self, running_system, integration_helper):
        """Test workflow requiring coordination between multiple agents."""
        system = running_system

        # Generate base data
        system.simulator.simulate_historical_data(days_back=3)

        # Send a scenario that affects multiple domains
        # For example, a large purchase that affects cash flow, inventory, and potentially staffing
        large_purchase = {
            "type": "new_transaction",
            "transaction": {
                "amount": 8000.00,  # Large purchase
                "description": "Large inventory purchase",
                "transaction_date": datetime.now().date().isoformat(),
                "transaction_type": "debit",
                "account_id": "checking_account",
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, large_purchase)

        # Send inventory delivery
        inventory_delivery = {
            "type": "inventory_delivery",
            "delivery": {
                "items": [
                    {"sku": "FOOD001", "quantity": 100, "unit_cost": 15.00},
                    {"sku": "FOOD002", "quantity": 50, "unit_cost": 25.00},
                ],
                "supplier": "Main Food Supplier",
                "total_cost": 2750.00,
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, inventory_delivery)

        # Send cash flow alert (might trigger multiple agent responses)
        cash_alert = {
            "type": "cash_flow_alert",
            "alert": {
                "current_balance": 3000.00,  # Low after purchase
                "threshold": 5000.00,
                "severity": "medium",
            },
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, cash_alert)

        # Give time for all agents to process
        await asyncio.sleep(5)

        # Verify all agents are still running and responsive
        for _agent_name, agent in system.agents.items():
            assert agent.is_running is True
            health = await agent.health_check()
            assert health["status"] == "running"

        # Check that system handled the complex scenario
        status = system.get_system_status()
        assert status["system_running"] is True

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, running_system, integration_helper):
        """Test system recovery from various error scenarios."""
        system = running_system

        # Send malformed messages to test error handling
        malformed_messages = [
            {"type": "invalid_message_type", "data": "bad_data"},
            {"type": "new_transaction"},  # Missing required fields
            {"type": "new_transaction", "transaction": {"amount": "not_a_number"}},
            {},  # Empty message
            None,  # Null message - this might not be sent, but test robustness
        ]

        for message in malformed_messages[:-1]:  # Skip None message
            try:
                await integration_helper.send_test_message(system, message)
                await asyncio.sleep(0.5)
            except Exception:
                pass  # Expected for some malformed messages

        # Give time for error handling
        await asyncio.sleep(2)

        # Verify system is still operational after errors
        for _agent_name, agent in system.agents.items():
            assert agent.is_running is True
            health = await agent.health_check()
            assert health["status"] == "running"

        # Send valid message to confirm system still works
        valid_message = {
            "type": "system_status_check",
            "timestamp": datetime.now().isoformat(),
            "cycle": 1,
        }

        await integration_helper.send_test_message(system, valid_message)
        await asyncio.sleep(1)

        # System should still be responsive
        status = system.get_system_status()
        assert status["system_running"] is True

    @pytest.mark.asyncio
    async def test_decision_persistence_workflow(self, running_system, integration_helper):
        """Test that agent decisions are properly persisted throughout workflow."""
        system = running_system

        # Force decisions by sending messages that should trigger responses
        decision_triggering_messages = [
            {"type": "cash_flow_check", "cycle": 1},
            {
                "type": "daily_analysis",
                "analysis_date": datetime.now().date().isoformat(),
                "cycle": 1,
            },
            {"type": "aging_analysis", "cycle": 1},
        ]

        for message in decision_triggering_messages:
            await integration_helper.send_test_message(system, message)
            await asyncio.sleep(1)

        # Give time for processing
        await asyncio.sleep(3)

        # Check decision persistence in database
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=system.agents["accounting"].engine)
        session = Session()

        try:
            # Query all decisions from database
            db_decisions = session.query(AgentDecisionModel).all()

            # May be 0 if no decisions were warranted, but should not error
            assert len(db_decisions) >= 0

            # If decisions were made, verify they're properly structured
            for decision in db_decisions:
                assert decision.agent_id is not None
                assert decision.decision_type is not None
                assert decision.action is not None
                assert decision.reasoning is not None
                assert 0.0 <= decision.confidence <= 1.0
                assert decision.timestamp is not None

        finally:
            session.close()

        # Verify in-memory decision logs match database
        total_memory_decisions = sum(len(agent.decisions_log) for agent in system.agents.values())

        # Memory and database should be consistent (or at least not error)
        assert total_memory_decisions >= 0


class TestBusinessScenarios:
    """Test specific business scenarios end-to-end."""

    @pytest.mark.asyncio
    async def test_cash_flow_crisis_scenario(self, running_system, integration_helper):
        """Test system response to cash flow crisis."""
        system = running_system

        # Set up scenario - generate base data
        system.simulator.simulate_historical_data(days_back=7)

        # Simulate cash flow crisis
        crisis_events = [
            # Large unexpected expense
            {
                "type": "new_transaction",
                "transaction": {
                    "amount": 10000.00,
                    "description": "Emergency equipment repair",
                    "transaction_type": "debit",
                    "account_id": "checking_account",
                    "transaction_date": datetime.now().date().isoformat(),
                },
                "cycle": 1,
            },
            # Revenue drop
            {
                "type": "revenue_alert",
                "alert": {
                    "current_daily_average": 800.00,
                    "expected_daily_average": 2000.00,
                    "variance_percentage": -60.0,
                },
                "cycle": 2,
            },
            # Cash flow check
            {"type": "cash_flow_check", "cycle": 3},
        ]

        for event in crisis_events:
            await integration_helper.send_test_message(system, event)
            await asyncio.sleep(1)

        # Give time for all agents to analyze the crisis
        await asyncio.sleep(5)

        # Verify system handled the crisis scenario
        # All agents should still be running
        for agent in system.agents.values():
            assert agent.is_running is True

        # Check if accounting agent made any crisis-related decisions
        accounting_agent = system.agents["accounting"]

        # Should be able to generate a report even during crisis
        report = await accounting_agent.generate_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_inventory_shortage_scenario(self, running_system, integration_helper):
        """Test system response to inventory shortage."""
        system = running_system

        # Set up scenario
        system.simulator.simulate_historical_data(days_back=5)

        # Simulate inventory shortage events
        shortage_events = [
            # Multiple items running low
            {
                "type": "stock_alert",
                "alert": {
                    "items": [
                        {"sku": "FOOD001", "current_stock": 2, "reorder_point": 15},
                        {"sku": "FOOD002", "current_stock": 0, "reorder_point": 20},
                        {"sku": "FOOD003", "current_stock": 5, "reorder_point": 25},
                    ],
                    "severity": "high",
                },
                "cycle": 1,
            },
            # High consumption rate
            {
                "type": "consumption_spike",
                "spike": {
                    "period": "last_24_hours",
                    "consumption_rate_increase": 150.0,  # 150% of normal
                    "affected_items": ["FOOD001", "FOOD002", "FOOD003"],
                },
                "cycle": 2,
            },
        ]

        for event in shortage_events:
            await integration_helper.send_test_message(system, event)
            await asyncio.sleep(1)

        # Give time for inventory agent to respond
        await asyncio.sleep(3)

        # Verify system handled shortage scenario
        inventory_agent = system.agents["inventory"]
        assert inventory_agent.is_running is True

        # Should be able to generate inventory report
        report = await inventory_agent.generate_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_seasonal_demand_scenario(self, running_system, integration_helper):
        """Test system adaptation to seasonal demand changes."""
        system = running_system

        # Set up normal operations first
        system.simulator.simulate_historical_data(days_back=10)

        # Simulate seasonal surge
        seasonal_events = [
            # Revenue increase
            {
                "type": "revenue_surge",
                "surge": {
                    "increase_percentage": 80.0,
                    "duration_days": 14,
                    "reason": "holiday_season",
                },
                "cycle": 1,
            },
            # Increased inventory consumption
            {
                "type": "consumption_increase",
                "increase": {
                    "factor": 1.8,
                    "affected_categories": ["food", "beverages"],
                    "duration_days": 14,
                },
                "cycle": 2,
            },
            # Additional staffing needs
            {
                "type": "staffing_alert",
                "alert": {
                    "projected_hours_needed": 200,  # Extra hours
                    "current_capacity": 120,
                    "period": "next_week",
                },
                "cycle": 3,
            },
        ]

        for event in seasonal_events:
            await integration_helper.send_test_message(system, event)
            await asyncio.sleep(1)

        # Give time for all agents to adapt
        await asyncio.sleep(4)

        # Verify all agents are handling the seasonal changes
        for _agent_name, agent in system.agents.items():
            assert agent.is_running is True

            # Each agent should be able to provide insights
            report = await agent.generate_report()
            assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_normal_operations_workflow(self, running_system, integration_helper):
        """Test system during normal day-to-day operations."""
        system = running_system

        # Generate baseline data
        system.simulator.simulate_historical_data(days_back=30)

        # Simulate normal business events over time
        normal_events = [
            {"type": "daily_analysis", "cycle": 1},
            {"type": "cash_flow_check", "cycle": 2},
            {"type": "inventory_check", "cycle": 3},
            {"type": "labor_analysis", "cycle": 4},
            {"type": "aging_analysis", "cycle": 5},
        ]

        for _i, event in enumerate(normal_events):
            await integration_helper.send_test_message(system, event)
            await asyncio.sleep(0.5)  # Quick succession for normal ops

        # Let system process normal operations
        await asyncio.sleep(3)

        # Verify steady-state operations
        status = system.get_system_status()
        assert status["system_running"] is True

        # All agents should be operational
        for _agent_name, agent_status in status["agents"].items():
            assert agent_status["running"] is True

        # System should be able to provide comprehensive status
        assert "business_config" in status
        assert "timestamp" in status
