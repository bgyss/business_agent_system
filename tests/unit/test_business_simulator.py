"""
Unit tests for BusinessSimulator class
"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.business_simulator import BusinessSimulator


class TestBusinessSimulator:
    """Test cases for BusinessSimulator"""

    @pytest.fixture
    def mock_database(self):
        """Mock database components"""
        with patch("simulation.business_simulator.create_engine") as mock_engine, patch(
            "simulation.business_simulator.sessionmaker"
        ) as mock_sessionmaker:

            # Setup mock session
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session

            # Setup mock engine
            mock_engine.return_value = Mock()

            yield {
                "engine": mock_engine,
                "sessionmaker": mock_sessionmaker,
                "session": mock_session,
            }

    @pytest.fixture
    def business_config(self):
        """Basic business configuration"""
        return {"duration_minutes": 10, "speed_multiplier": 2.0, "simulation_interval": 5}

    @pytest.fixture
    def business_simulator(self, mock_database, business_config):
        """Create BusinessSimulator instance for testing"""
        with patch("simulation.business_simulator.FinancialBase"), patch(
            "simulation.business_simulator.InventoryBase"
        ), patch("simulation.business_simulator.EmployeeBase"):

            simulator = BusinessSimulator(config=business_config, db_url="sqlite:///:memory:")
            return simulator

    def test_initialization(self, business_config, mock_database):
        """Test BusinessSimulator initialization"""
        with patch("simulation.business_simulator.FinancialBase"), patch(
            "simulation.business_simulator.InventoryBase"
        ), patch("simulation.business_simulator.EmployeeBase"):

            simulator = BusinessSimulator(config=business_config, db_url="test_db_url")

            assert simulator.config == business_config
            assert simulator.db_url == "test_db_url"
            assert simulator.financial_generator is None
            assert simulator.inventory_simulator is None
            assert simulator.is_running is False

            # Verify database setup
            mock_database["engine"].assert_called_once_with("test_db_url")
            mock_database["sessionmaker"].assert_called_once()

    def test_initialization_default_db_url(self, business_config, mock_database):
        """Test BusinessSimulator initialization with default database URL"""
        with patch("simulation.business_simulator.FinancialBase"), patch(
            "simulation.business_simulator.InventoryBase"
        ), patch("simulation.business_simulator.EmployeeBase"):

            simulator = BusinessSimulator(config=business_config)

            assert simulator.db_url == "sqlite:///business_simulation.db"

    def test_initialize_business_restaurant(self, business_simulator):
        """Test business initialization for restaurant type"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock empty database
        mock_session.query.return_value.count.return_value = 0

        with patch("simulation.business_simulator.get_restaurant_profile") as mock_profile, patch(
            "simulation.business_simulator.get_restaurant_inventory_profile"
        ) as mock_inv_profile, patch(
            "simulation.business_simulator.FinancialDataGenerator"
        ) as mock_fin_gen, patch(
            "simulation.business_simulator.InventorySimulator"
        ) as mock_inv_sim:

            mock_profile.return_value = Mock()
            mock_inv_profile.return_value = Mock()

            business_simulator.initialize_business("restaurant")

            # Verify generators were created
            mock_profile.assert_called_once()
            mock_inv_profile.assert_called_once()
            mock_fin_gen.assert_called_once()
            mock_inv_sim.assert_called_once()

            # Verify initial data creation
            assert mock_session.add.call_count >= 4  # At least accounts
            mock_session.commit.assert_called()
            mock_session.close.assert_called()

    def test_initialize_business_retail(self, business_simulator):
        """Test business initialization for retail type"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock empty database
        mock_session.query.return_value.count.return_value = 0

        with patch("simulation.business_simulator.get_retail_profile") as mock_profile, patch(
            "simulation.business_simulator.get_retail_inventory_profile"
        ) as mock_inv_profile, patch("simulation.business_simulator.FinancialDataGenerator"), patch(
            "simulation.business_simulator.InventorySimulator"
        ):

            mock_profile.return_value = Mock()
            mock_inv_profile.return_value = Mock()

            business_simulator.initialize_business("retail")

            # Verify correct profiles were used
            mock_profile.assert_called_once()
            mock_inv_profile.assert_called_once()

    def test_initialize_business_existing_data(self, business_simulator):
        """Test business initialization when data already exists"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock existing data
        mock_session.query.return_value.count.return_value = 5

        with patch("simulation.business_simulator.get_restaurant_profile") as mock_profile, patch(
            "simulation.business_simulator.get_restaurant_inventory_profile"
        ) as mock_inv_profile:

            mock_profile.return_value = Mock()
            mock_inv_profile.return_value = Mock()

            business_simulator.initialize_business("restaurant")

            # Should not add new accounts/suppliers/employees
            mock_session.add.assert_not_called()
            # Should only commit for setting up generators
            mock_session.close.assert_called()

    def test_simulate_historical_data_not_initialized(self, business_simulator):
        """Test simulate_historical_data raises error when not initialized"""
        with pytest.raises(ValueError, match="Business must be initialized first"):
            business_simulator.simulate_historical_data()

    def test_simulate_historical_data_success(self, business_simulator):
        """Test successful historical data simulation"""
        # Setup mocks
        mock_financial_gen = Mock()
        mock_inventory_sim = Mock()
        business_simulator.financial_generator = mock_financial_gen
        business_simulator.inventory_simulator = mock_inventory_sim

        # Mock session
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock data generation
        financial_data = {
            "transactions": [{"id": 1}, {"id": 2}],
            "accounts_receivable": [{"id": 1}],
            "accounts_payable": [{"id": 1}],
        }
        mock_financial_gen.generate_period_data.return_value = financial_data
        mock_financial_gen.generate_anomalies.return_value = [{"id": 3}]

        inventory_items = [{"sku": "ITEM001", "id": 1}]
        mock_inventory_sim.generate_initial_inventory.return_value = inventory_items

        # Proper stock movement data structure
        from datetime import datetime

        from models.inventory import StockMovementType

        mock_movement_date = datetime.now()

        mock_inventory_sim.simulate_daily_consumption.return_value = [
            {
                "item_sku": "ITEM001",
                "movement_type": StockMovementType.OUT,
                "quantity": 5,
                "unit_cost": 2.50,
                "reference_number": "CONSUMPTION-20240101-ITEM001",
                "notes": "Daily consumption",
                "movement_date": mock_movement_date,
            }
        ]

        mock_inventory_sim.simulate_deliveries.return_value = [
            {
                "item_sku": "ITEM001",
                "movement_type": StockMovementType.IN,
                "quantity": 10,
                "unit_cost": 2.00,
                "reference_number": "DELIVERY-20240101-ITEM001",
                "notes": "Supplier delivery",
                "movement_date": mock_movement_date,
            }
        ]

        mock_inventory_sim.simulate_stock_adjustments.return_value = [
            {
                "item_sku": "ITEM001",
                "movement_type": StockMovementType.ADJUSTMENT,
                "quantity": -1,
                "unit_cost": 2.50,
                "reference_number": "ADJ-20240101-ITEM001",
                "notes": "Inventory adjustment",
                "movement_date": mock_movement_date,
            }
        ]

        # Mock existing item check
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_session.flush.return_value = None

        business_simulator.simulate_historical_data(days_back=5)

        # Verify data generation was called
        mock_financial_gen.generate_period_data.assert_called_once()
        mock_financial_gen.generate_anomalies.assert_called_once()
        mock_inventory_sim.generate_initial_inventory.assert_called_once()

        # Verify database operations
        assert mock_session.add.call_count > 0
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_simulate_historical_data_existing_items(self, business_simulator):
        """Test historical data simulation with existing inventory items"""
        # Setup mocks
        mock_financial_gen = Mock()
        mock_inventory_sim = Mock()
        business_simulator.financial_generator = mock_financial_gen
        business_simulator.inventory_simulator = mock_inventory_sim

        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock existing item
        existing_item = Mock()
        existing_item.id = "existing_id"
        mock_session.query.return_value.filter_by.return_value.first.return_value = existing_item

        # Mock data
        financial_data = {"transactions": [], "accounts_receivable": [], "accounts_payable": []}
        mock_financial_gen.generate_period_data.return_value = financial_data
        mock_financial_gen.generate_anomalies.return_value = []

        inventory_items = [{"sku": "EXISTING_ITEM", "id": 1}]
        mock_inventory_sim.generate_initial_inventory.return_value = inventory_items
        mock_inventory_sim.simulate_daily_consumption.return_value = []
        mock_inventory_sim.simulate_deliveries.return_value = []
        mock_inventory_sim.simulate_stock_adjustments.return_value = []

        business_simulator.simulate_historical_data(days_back=1)

        # Should not add the existing item again
        mock_session.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_real_time_simulation_basic(self, business_simulator):
        """Test basic real-time simulation startup"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        mock_financial_gen = Mock()
        business_simulator.financial_generator = mock_financial_gen
        mock_financial_gen.generate_daily_transactions.return_value = [{"id": 1, "amount": 100}]

        message_queue = asyncio.Queue()

        # Start simulation in background and stop it quickly
        task = asyncio.create_task(business_simulator.start_real_time_simulation(message_queue))

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Stop simulation
        await business_simulator.stop_simulation()

        # Wait for task to complete
        await task

        # Verify simulation started and stopped
        assert business_simulator.is_running is False

        # Should have generated some data
        mock_financial_gen.generate_daily_transactions.assert_called()
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_start_real_time_simulation_with_duration(self, business_simulator):
        """Test real-time simulation with duration limit"""
        # Set very short duration for testing
        business_simulator.config["duration_minutes"] = 0.01  # 0.6 seconds

        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        mock_financial_gen = Mock()
        business_simulator.financial_generator = mock_financial_gen
        mock_financial_gen.generate_daily_transactions.return_value = []

        message_queue = asyncio.Queue()

        # This should complete automatically due to duration limit
        await business_simulator.start_real_time_simulation(message_queue)

        assert business_simulator.is_running is False

    @pytest.mark.asyncio
    async def test_start_real_time_simulation_message_generation(self, business_simulator):
        """Test that real-time simulation generates appropriate messages"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        mock_financial_gen = Mock()
        business_simulator.financial_generator = mock_financial_gen
        mock_financial_gen.generate_daily_transactions.return_value = [{"id": 1, "amount": 100}]

        message_queue = asyncio.Queue()

        # Start simulation
        task = asyncio.create_task(business_simulator.start_real_time_simulation(message_queue))

        # Let it run for a short time
        await asyncio.sleep(0.2)

        # Stop simulation
        await business_simulator.stop_simulation()
        await task

        # Check that messages were generated
        assert not message_queue.empty()

        # Get first message
        message = await message_queue.get()
        assert "type" in message
        assert message["type"] in [
            "new_transaction",
            "cash_flow_check",
            "daily_analysis",
            "aging_analysis",
        ]

    @pytest.mark.asyncio
    async def test_start_real_time_simulation_error_handling(self, business_simulator):
        """Test error handling in real-time simulation"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Make financial generator raise an exception
        mock_financial_gen = Mock()
        business_simulator.financial_generator = mock_financial_gen
        mock_financial_gen.generate_daily_transactions.side_effect = Exception("Test error")

        message_queue = asyncio.Queue()

        # Start simulation
        task = asyncio.create_task(business_simulator.start_real_time_simulation(message_queue))

        # Let it try to run
        await asyncio.sleep(0.1)

        # Stop simulation
        await business_simulator.stop_simulation()
        await task

        # Should have handled the error gracefully
        assert business_simulator.is_running is False

    @pytest.mark.asyncio
    async def test_stop_simulation(self, business_simulator):
        """Test stopping simulation"""
        business_simulator.is_running = True

        await business_simulator.stop_simulation()

        assert business_simulator.is_running is False

    def test_get_simulation_status_with_data(self, business_simulator):
        """Test getting simulation status with data"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock query results
        mock_session.query.return_value.count.side_effect = [
            10,
            5,
            3,
        ]  # transactions, receivables, payables

        mock_transaction = Mock()
        mock_transaction.transaction_date = datetime.now()
        mock_session.query.return_value.order_by.return_value.first.return_value = mock_transaction

        # Mock financial generator
        mock_profile = Mock()
        mock_profile.business_type = "restaurant"
        mock_financial_gen = Mock()
        mock_financial_gen.profile = mock_profile
        business_simulator.financial_generator = mock_financial_gen

        business_simulator.is_running = True

        status = business_simulator.get_simulation_status()

        assert status["is_running"] is True
        assert status["transaction_count"] == 10
        assert status["receivable_count"] == 5
        assert status["payable_count"] == 3
        assert status["latest_transaction_date"] == mock_transaction.transaction_date
        assert status["business_type"] == "restaurant"

        mock_session.close.assert_called_once()

    def test_get_simulation_status_no_data(self, business_simulator):
        """Test getting simulation status with no data"""
        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock empty results
        mock_session.query.return_value.count.return_value = 0
        mock_session.query.return_value.order_by.return_value.first.return_value = None

        status = business_simulator.get_simulation_status()

        assert status["is_running"] is False
        assert status["transaction_count"] == 0
        assert status["receivable_count"] == 0
        assert status["payable_count"] == 0
        assert status["latest_transaction_date"] is None
        assert status["business_type"] is None

    def test_generate_sample_scenarios(self, business_simulator):
        """Test generating sample scenarios"""
        scenarios = business_simulator.generate_sample_scenarios()

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

        # Check first scenario structure
        scenario = scenarios[0]
        assert "name" in scenario
        assert "description" in scenario
        assert "actions" in scenario
        assert isinstance(scenario["actions"], list)

        # Verify all expected scenarios are present
        scenario_names = [s["name"] for s in scenarios]
        expected_scenarios = [
            "Cash Flow Crisis",
            "Seasonal Rush",
            "Equipment Failure",
            "New Competitor",
        ]
        for expected in expected_scenarios:
            assert expected in scenario_names

    def test_apply_scenario(self, business_simulator):
        """Test applying a scenario (basic implementation)"""
        # This is a stub method in the current implementation
        # Just verify it doesn't raise an error
        business_simulator.apply_scenario("Cash Flow Crisis")
        # No assertions needed as it's a placeholder implementation

    def test_database_session_error_handling(self, business_simulator):
        """Test that database session errors are handled properly"""
        mock_session = Mock()
        mock_session.commit.side_effect = Exception("Database error")
        business_simulator.SessionLocal.return_value = mock_session

        # Mock generators
        mock_financial_gen = Mock()
        mock_inventory_sim = Mock()
        business_simulator.financial_generator = mock_financial_gen
        business_simulator.inventory_simulator = mock_inventory_sim

        # Mock data generation
        financial_data = {
            "transactions": [{"id": 1}],
            "accounts_receivable": [],
            "accounts_payable": [],
        }
        mock_financial_gen.generate_period_data.return_value = financial_data
        mock_financial_gen.generate_anomalies.return_value = []

        inventory_items = []
        mock_inventory_sim.generate_initial_inventory.return_value = inventory_items
        mock_inventory_sim.simulate_daily_consumption.return_value = []
        mock_inventory_sim.simulate_deliveries.return_value = []
        mock_inventory_sim.simulate_stock_adjustments.return_value = []

        # This should not raise an exception despite database error
        with pytest.raises(Exception):
            business_simulator.simulate_historical_data(days_back=1)

        # Session should still be closed
        mock_session.close.assert_called_once()

    @patch("simulation.business_simulator.random.shuffle")
    def test_simulate_historical_data_transaction_shuffling(self, mock_shuffle, business_simulator):
        """Test that transactions are shuffled during historical data generation"""
        # Setup mocks
        mock_financial_gen = Mock()
        mock_inventory_sim = Mock()
        business_simulator.financial_generator = mock_financial_gen
        business_simulator.inventory_simulator = mock_inventory_sim

        mock_session = Mock()
        business_simulator.SessionLocal.return_value = mock_session

        # Mock data
        financial_data = {
            "transactions": [{"id": 1}, {"id": 2}],
            "accounts_receivable": [],
            "accounts_payable": [],
        }
        mock_financial_gen.generate_period_data.return_value = financial_data
        mock_financial_gen.generate_anomalies.return_value = [{"id": 3}]

        mock_inventory_sim.generate_initial_inventory.return_value = []
        mock_inventory_sim.simulate_daily_consumption.return_value = []
        mock_inventory_sim.simulate_deliveries.return_value = []
        mock_inventory_sim.simulate_stock_adjustments.return_value = []

        business_simulator.simulate_historical_data(days_back=1)

        # Verify shuffle was called on the combined transaction list
        mock_shuffle.assert_called_once()

    def test_config_access(self, business_simulator, business_config):
        """Test that configuration is properly accessible"""
        assert business_simulator.config == business_config
        assert business_simulator.config.get("duration_minutes") == 10
        assert business_simulator.config.get("speed_multiplier") == 2.0
        assert business_simulator.config.get("simulation_interval") == 5
        assert business_simulator.config.get("nonexistent_key", "default") == "default"
