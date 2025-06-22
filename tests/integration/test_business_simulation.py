"""
Integration tests for business simulation workflows.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from models.employee import Employee
from models.financial import Account, AccountsPayable, AccountsReceivable, Transaction
from models.inventory import Item, StockMovement, Supplier
from simulation.business_simulator import BusinessSimulator


class TestBusinessSimulation:
    """Test complete business simulation workflows."""

    def test_simulator_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.config is not None
        assert simulator.db_url is not None
        assert simulator.engine is not None
        assert simulator.SessionLocal is not None

        # Should be initialized but not running
        assert simulator.is_running is False
        assert simulator.financial_generator is None
        assert simulator.inventory_simulator is None

    def test_business_initialization_restaurant(self, simulator):
        """Test business initialization for restaurant type."""
        simulator.initialize_business("restaurant")

        assert simulator.financial_generator is not None
        assert simulator.inventory_simulator is not None

        # Check that initial data was created
        session = simulator.SessionLocal()
        try:
            account_count = session.query(Account).count()
            supplier_count = session.query(Supplier).count()
            employee_count = session.query(Employee).count()

            assert account_count >= 4  # At least the basic accounts
            assert supplier_count > 0  # Should have created suppliers
            assert employee_count >= 3  # Should have created employees
        finally:
            session.close()

    def test_business_initialization_retail(self, simulator):
        """Test business initialization for retail type."""
        simulator.initialize_business("retail")

        assert simulator.financial_generator is not None
        assert simulator.inventory_simulator is not None

        session = simulator.SessionLocal()
        try:
            # Verify initial accounts
            accounts = session.query(Account).all()
            account_names = [acc.name for acc in accounts]

            assert "Business Checking" in account_names
            assert "Business Savings" in account_names
            assert "Revenue" in account_names
            assert "General Expenses" in account_names
        finally:
            session.close()

    def test_historical_data_generation(self, simulator):
        """Test historical data generation."""
        simulator.initialize_business("restaurant")
        simulator.simulate_historical_data(days_back=5)

        session = simulator.SessionLocal()
        try:
            # Verify data was generated
            transaction_count = session.query(Transaction).count()
            receivable_count = session.query(AccountsReceivable).count()
            payable_count = session.query(AccountsPayable).count()
            item_count = session.query(Item).count()
            movement_count = session.query(StockMovement).count()

            assert transaction_count > 0
            assert receivable_count >= 0  # May be 0 if no credit transactions
            assert payable_count >= 0  # May be 0 if no payables
            assert item_count > 0
            assert movement_count > 0

            # Verify transactions have reasonable dates (within reasonable range allowing for timezone differences)
            latest_transaction = (
                session.query(Transaction).order_by(Transaction.transaction_date.desc()).first()
            )
            # Allow up to 12 hours future for timezone differences
            assert latest_transaction.transaction_date <= datetime.now() + timedelta(hours=12)

            earliest_transaction = (
                session.query(Transaction).order_by(Transaction.transaction_date.asc()).first()
            )
            assert earliest_transaction.transaction_date >= (
                datetime.now() - timedelta(days=7)  # Allow one extra day buffer
            )

        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_real_time_simulation_basic(self, simulator):
        """Test basic real-time simulation functionality."""
        simulator.initialize_business("restaurant")

        # Create a message queue for testing
        message_queue = asyncio.Queue()

        # Start simulation for a short time
        simulation_task = asyncio.create_task(simulator.start_real_time_simulation(message_queue))

        # Let it run for a few cycles
        await asyncio.sleep(3)

        # Stop simulation
        await simulator.stop_simulation()

        # Wait for simulation task to complete
        try:
            await asyncio.wait_for(simulation_task, timeout=2)
        except asyncio.TimeoutError:
            simulation_task.cancel()

        # Check that messages were generated
        messages = []
        while not message_queue.empty():
            messages.append(await message_queue.get())

        assert len(messages) > 0

        # Verify message structure
        for message in messages:
            assert isinstance(message, dict)
            assert "type" in message
            assert message["type"] in [
                "new_transaction",
                "cash_flow_check",
                "daily_analysis",
                "aging_analysis",
            ]

    @pytest.mark.asyncio
    async def test_real_time_simulation_with_duration(self, test_config, temp_db):
        """Test real-time simulation with time limit."""
        # Create simulator with short duration
        config = test_config["simulation"].copy()
        config["duration_minutes"] = 0.1  # 6 seconds
        config["speed_multiplier"] = 10.0  # Fast simulation

        simulator = BusinessSimulator(config, temp_db)
        simulator.initialize_business("restaurant")

        message_queue = asyncio.Queue()

        start_time = datetime.now()

        # Start simulation
        await simulator.start_real_time_simulation(message_queue)

        end_time = datetime.now()
        elapsed_seconds = (end_time - start_time).total_seconds()

        # Should have stopped automatically after duration
        assert simulator.is_running is False
        assert elapsed_seconds >= 6  # At least the minimum duration
        assert elapsed_seconds <= 15  # But not too much longer

    def test_simulation_status(self, simulator):
        """Test simulation status reporting."""
        simulator.initialize_business("restaurant")
        simulator.simulate_historical_data(days_back=3)

        status = simulator.get_simulation_status()

        assert status is not None
        assert "is_running" in status
        assert "transaction_count" in status
        assert "receivable_count" in status
        assert "payable_count" in status
        assert "business_type" in status

        assert status["is_running"] is False
        assert status["transaction_count"] > 0
        assert status["business_type"] == "restaurant"

    def test_anomaly_generation(self, simulator):
        """Test that anomalies are generated in historical data."""
        simulator.initialize_business("restaurant")
        simulator.simulate_historical_data(days_back=10)

        session = simulator.SessionLocal()
        try:
            transactions = session.query(Transaction).all()

            # With 10 days of data and 1% anomaly rate, should have some anomalies
            # Check for transactions with unusual amounts
            amounts = [float(t.amount) for t in transactions]

            if len(amounts) > 10:  # Only check if we have enough data
                avg_amount = sum(amounts) / len(amounts)
                max_amount = max(amounts)
                min_amount = min(amounts)

                # Should have some variation indicating anomalies were added
                assert max_amount > avg_amount * 1.5 or min_amount < avg_amount * 0.5
        finally:
            session.close()

    def test_inventory_simulation_integration(self, simulator):
        """Test inventory simulation integration."""
        simulator.initialize_business("restaurant")
        simulator.simulate_historical_data(days_back=7)

        session = simulator.SessionLocal()
        try:
            items = session.query(Item).all()
            movements = session.query(StockMovement).all()

            assert len(items) > 0
            assert len(movements) > 0

            # Verify movement types
            movement_types = {m.movement_type for m in movements}
            expected_types = {"consumption", "delivery", "adjustment"}

            # Should have at least some of these movement types
            assert len(movement_types.intersection(expected_types)) > 0

            # Verify items have reasonable data
            for item in items[:5]:  # Check first 5 items
                assert item.name is not None
                assert item.sku is not None
                assert item.unit_cost >= 0
                assert item.current_stock >= 0
        finally:
            session.close()

    def test_employee_data_generation(self, simulator):
        """Test employee data generation."""
        simulator.initialize_business("restaurant")

        session = simulator.SessionLocal()
        try:
            employees = session.query(Employee).all()

            assert len(employees) >= 3

            # Verify employee data
            for employee in employees:
                assert employee.employee_id is not None
                assert employee.first_name is not None
                assert employee.last_name is not None
                assert employee.email is not None
                assert employee.position is not None
                assert employee.hourly_rate > 0
                assert employee.hire_date is not None
        finally:
            session.close()

    def test_multiple_business_types(self, temp_db):
        """Test that different business types generate different data profiles."""
        # Test restaurant
        config = {"simulation_interval": 1, "speed_multiplier": 1.0}
        restaurant_sim = BusinessSimulator(config, temp_db)
        restaurant_sim.initialize_business("restaurant")
        restaurant_sim.simulate_historical_data(days_back=3)

        session = restaurant_sim.SessionLocal()
        try:
            restaurant_transactions = session.query(Transaction).count()
            restaurant_items = session.query(Item).count()
        finally:
            session.close()

        # Clear data for retail test
        session = restaurant_sim.SessionLocal()
        try:
            session.query(Transaction).delete()
            session.query(Item).delete()
            session.query(StockMovement).delete()
            session.commit()
        finally:
            session.close()

        # Test retail
        retail_sim = BusinessSimulator(config, temp_db)
        retail_sim.initialize_business("retail")
        retail_sim.simulate_historical_data(days_back=3)

        session = retail_sim.SessionLocal()
        try:
            retail_transactions = session.query(Transaction).count()
            retail_items = session.query(Item).count()
        finally:
            session.close()

        # Both should generate data
        assert restaurant_transactions > 0
        assert restaurant_items > 0
        assert retail_transactions > 0
        assert retail_items > 0

    def test_scenario_generation(self, simulator):
        """Test scenario generation functionality."""
        scenarios = simulator.generate_sample_scenarios()

        assert len(scenarios) > 0

        # Verify scenario structure
        for scenario in scenarios:
            assert "name" in scenario
            assert "description" in scenario
            assert "actions" in scenario
            assert isinstance(scenario["actions"], list)
            assert len(scenario["actions"]) > 0

        # Check for specific scenarios
        scenario_names = [s["name"] for s in scenarios]
        expected_scenarios = [
            "Cash Flow Crisis",
            "Seasonal Rush",
            "Equipment Failure",
            "New Competitor",
        ]

        for expected in expected_scenarios:
            assert expected in scenario_names


class TestSimulationErrorHandling:
    """Test error handling in simulation workflows."""

    def test_simulation_without_initialization(self, simulator):
        """Test simulation fails gracefully without initialization."""
        with pytest.raises(ValueError, match="Business must be initialized first"):
            simulator.simulate_historical_data(days_back=5)

    def test_invalid_business_type(self, simulator):
        """Test handling of invalid business type."""
        # Should not raise an error, but fall back to default
        simulator.initialize_business("invalid_type")

        # Should still create the components
        assert simulator.financial_generator is not None
        assert simulator.inventory_simulator is not None

    @pytest.mark.asyncio
    async def test_simulation_error_recovery(self, simulator):
        """Test simulation continues after errors."""
        simulator.initialize_business("restaurant")

        message_queue = asyncio.Queue()

        # Start simulation
        simulation_task = asyncio.create_task(simulator.start_real_time_simulation(message_queue))

        # Let it run briefly
        await asyncio.sleep(1)

        # Stop simulation
        await simulator.stop_simulation()

        # Should not raise exceptions
        try:
            await asyncio.wait_for(simulation_task, timeout=3)
        except asyncio.TimeoutError:
            simulation_task.cancel()

        # Should have processed some cycles without crashing
        assert not simulation_task.done() or not simulation_task.exception()

    def test_database_constraint_handling(self, simulator):
        """Test handling of database constraint violations."""
        simulator.initialize_business("restaurant")

        # Try to initialize again (should handle duplicate data gracefully)
        simulator.initialize_business("restaurant")

        session = simulator.SessionLocal()
        try:
            # Should not have duplicate accounts
            accounts = session.query(Account).all()
            account_ids = [acc.id for acc in accounts]

            # Should not have duplicates
            assert len(account_ids) == len(set(account_ids))
        finally:
            session.close()
