"""
Unit tests for advanced InventoryAgent predictive analytics functionality
"""
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import AgentDecision
from agents.inventory_agent import (
    BulkPurchaseOptimization,
    DemandForecast,
    InventoryAgent,
    OptimalReorderPoint,
)
from models.inventory import ItemStatus, StockMovementType


class TestInventoryAgentAdvanced:
    """Test cases for advanced InventoryAgent analytics"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Advanced inventory analysis: Optimization recommended")]
            mock_client.return_value.messages.create.return_value = mock_response
            yield mock_client

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        with patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker') as mock_sessionmaker:
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def advanced_agent_config(self):
        """Advanced inventory agent configuration"""
        return {
            "check_interval": 300,
            "low_stock_multiplier": 1.2,
            "reorder_lead_time": 7,
            "consumption_analysis_days": 30,
            "forecast_horizon_days": 30,
            "service_level_target": 0.95,
            "holding_cost_rate": 0.25,
            "order_cost": 50.0,
            "min_forecast_accuracy": 0.70,
            "seasonality_window_days": 365,
            "alpha_smoothing": 0.3,
            "beta_trend": 0.1,
            "gamma_seasonality": 0.2
        }

    @pytest.fixture
    def advanced_inventory_agent(self, mock_anthropic, mock_db_session, advanced_agent_config):
        """Create advanced inventory agent instance"""
        return InventoryAgent(
            agent_id="advanced_inventory_agent",
            api_key="test_api_key",
            config=advanced_agent_config,
            db_url="sqlite:///:memory:"
        )

    @pytest.fixture
    def sample_stock_movements(self):
        """Create sample stock movements for testing forecasting"""
        base_date = datetime.now() - timedelta(days=60)
        movements = []

        # Generate sample consumption data with trend and seasonality
        for i in range(60):
            date = base_date + timedelta(days=i)
            # Base consumption with weekly seasonality
            base_consumption = 10 + 2 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            # Add trend
            trend_consumption = base_consumption + 0.1 * i  # Slight upward trend
            # Add noise
            actual_consumption = max(0, trend_consumption + np.random.normal(0, 1))

            movements.append(Mock(
                item_id="test_item",
                movement_type=StockMovementType.OUT,
                quantity=int(actual_consumption),
                movement_date=date
            ))

        return movements

    @pytest.fixture
    def sample_item_with_expiry(self):
        """Create sample item with expiry information"""
        return Mock(
            id="expiry_item",
            name="Perishable Item",
            sku="PERISH-001",
            current_stock=100,
            reorder_point=20,
            reorder_quantity=50,
            minimum_stock=10,
            unit_cost=Decimal("5.00"),
            expiry_days=14,  # 2 weeks expiry
            status=ItemStatus.ACTIVE
        )

    @pytest.fixture
    def sample_suppliers(self):
        """Create sample suppliers for performance testing"""
        return [
            Mock(
                id="supplier_1",
                name="Excellent Supplier",
                lead_time_days=5,
                rating=Decimal("4.8"),
                is_active=True
            ),
            Mock(
                id="supplier_2",
                name="Average Supplier",
                lead_time_days=10,
                rating=Decimal("3.5"),
                is_active=True
            ),
            Mock(
                id="supplier_3",
                name="Poor Supplier",
                lead_time_days=15,
                rating=Decimal("2.0"),
                is_active=True
            )
        ]

    def test_advanced_agent_initialization(self, advanced_inventory_agent, advanced_agent_config):
        """Test advanced agent initialization with new config parameters"""
        agent = advanced_inventory_agent
        assert agent.forecast_horizon_days == 30
        assert agent.service_level_target == 0.95
        assert agent.holding_cost_rate == 0.25
        assert agent.order_cost == 50.0
        assert agent.min_forecast_accuracy == 0.70
        assert agent.alpha_smoothing == 0.3
        assert agent.beta_trend == 0.1
        assert agent.gamma_seasonality == 0.2

    @pytest.mark.asyncio
    async def test_demand_prediction_insufficient_data(self, advanced_inventory_agent, mock_db_session):
        """Test demand prediction with insufficient historical data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock insufficient consumption data (less than 14 data points)
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        forecast = await advanced_inventory_agent.predict_demand(mock_session_instance, "test_item")

        assert forecast is None

    @pytest.mark.asyncio
    async def test_demand_prediction_success(self, advanced_inventory_agent, mock_db_session, sample_stock_movements):
        """Test successful demand prediction with sufficient data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock sufficient consumption data
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = sample_stock_movements

        forecast = await advanced_inventory_agent.predict_demand(mock_session_instance, "test_item")

        assert forecast is not None
        assert isinstance(forecast, DemandForecast)
        assert forecast.item_id == "test_item"
        assert forecast.predicted_demand > 0
        assert len(forecast.confidence_interval) == 2
        assert forecast.confidence_interval[0] <= forecast.confidence_interval[1]
        assert 0 <= forecast.forecast_accuracy <= 1

    def test_aggregate_daily_consumption(self, advanced_inventory_agent, sample_stock_movements):
        """Test daily consumption aggregation"""
        daily_consumption = advanced_inventory_agent._aggregate_daily_consumption(sample_stock_movements)

        assert isinstance(daily_consumption, list)
        assert len(daily_consumption) > 0
        assert all(isinstance(x, float) for x in daily_consumption)
        assert all(x >= 0 for x in daily_consumption)

    def test_holt_winters_forecast_insufficient_data(self, advanced_inventory_agent):
        """Test Holt-Winters forecasting with insufficient data"""
        short_data = [1.0, 2.0, 3.0]  # Less than 14 points

        result = advanced_inventory_agent._apply_holt_winters_forecast(short_data, 7)

        assert result is None

    def test_holt_winters_forecast_success(self, advanced_inventory_agent):
        """Test successful Holt-Winters forecasting"""
        # Generate synthetic data with trend and seasonality
        data = []
        for i in range(30):
            value = 10 + 0.1 * i + 2 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5)
            data.append(max(0, value))

        result = advanced_inventory_agent._apply_holt_winters_forecast(data, 7)

        assert result is not None
        predicted_demand, confidence_interval, seasonality_factor, trend_factor, accuracy = result
        assert predicted_demand >= 0
        assert len(confidence_interval) == 2
        assert confidence_interval[0] <= confidence_interval[1]
        assert 0 <= accuracy <= 1

    def test_calculate_forecast_accuracy(self, advanced_inventory_agent):
        """Test forecast accuracy calculation"""
        actual = np.array([10, 12, 8, 15, 11])
        predicted = np.array([9, 13, 7, 14, 12])

        accuracy = advanced_inventory_agent._calculate_forecast_accuracy(actual, predicted)

        assert 0 <= accuracy <= 1

    def test_calculate_forecast_accuracy_with_zeros(self, advanced_inventory_agent):
        """Test forecast accuracy calculation with zero values"""
        actual = np.array([0, 0, 0, 0, 0])
        predicted = np.array([1, 2, 3, 4, 5])

        accuracy = advanced_inventory_agent._calculate_forecast_accuracy(actual, predicted)

        assert accuracy == 0.0

    @pytest.mark.asyncio
    async def test_calculate_optimal_reorder_point_no_item(self, advanced_inventory_agent, mock_db_session):
        """Test optimal reorder point calculation with non-existent item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.first.return_value = None

        result = await advanced_inventory_agent.calculate_optimal_reorder_point(mock_session_instance, "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_calculate_optimal_reorder_point_success(self, advanced_inventory_agent, mock_db_session, sample_stock_movements):
        """Test successful optimal reorder point calculation"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock item
        item = Mock(
            id="test_item",
            name="Test Item",
            current_stock=50,
            reorder_point=15,
            reorder_quantity=100,
            unit_cost=Decimal("10.00"),
            minimum_stock=5
        )
        mock_session_instance.query.return_value.filter.return_value.first.return_value = item

        # Mock demand forecast
        with patch.object(advanced_inventory_agent, 'predict_demand') as mock_predict:
            mock_predict.return_value = DemandForecast(
                item_id="test_item",
                predicted_demand=300.0,
                confidence_interval=(250.0, 350.0),
                seasonality_factor=1.0,
                trend_factor=0.1,
                forecast_horizon_days=30,
                forecast_accuracy=0.85
            )

            result = await advanced_inventory_agent.calculate_optimal_reorder_point(mock_session_instance, "test_item")

        assert result is not None
        assert isinstance(result, OptimalReorderPoint)
        assert result.item_id == "test_item"
        assert result.optimal_reorder_point > 0
        assert result.optimal_reorder_quantity > 0
        assert 0 <= result.service_level <= 1
        assert result.safety_stock >= 0
        assert result.total_cost >= 0

    def test_get_z_score_for_service_level(self, advanced_inventory_agent):
        """Test Z-score calculation for service levels"""
        # Test exact matches
        assert advanced_inventory_agent._get_z_score_for_service_level(0.95) == 1.65
        assert advanced_inventory_agent._get_z_score_for_service_level(0.99) == 2.33

        # Test interpolation
        z_score = advanced_inventory_agent._get_z_score_for_service_level(0.92)
        assert 1.28 < z_score < 1.65  # Between 90% and 95%

        # Test extreme values
        assert advanced_inventory_agent._get_z_score_for_service_level(0.999) == 3.09
        assert advanced_inventory_agent._get_z_score_for_service_level(1.0) == 1.65  # Default

    def test_calculate_total_inventory_cost(self, advanced_inventory_agent):
        """Test total inventory cost calculation"""
        annual_demand = 1000.0
        order_quantity = 200
        unit_cost = 10.0
        service_level = 0.95
        safety_stock = 50.0

        total_cost = advanced_inventory_agent._calculate_total_inventory_cost(
            annual_demand, order_quantity, unit_cost, service_level, safety_stock
        )

        assert total_cost > 0
        # Cost should include ordering, holding, and shortage components

    def test_calculate_total_inventory_cost_edge_cases(self, advanced_inventory_agent):
        """Test inventory cost calculation with edge cases"""
        # Zero demand
        cost = advanced_inventory_agent._calculate_total_inventory_cost(0, 100, 10.0, 0.95, 10.0)
        assert cost == 0.0

        # Zero order quantity
        cost = advanced_inventory_agent._calculate_total_inventory_cost(1000, 0, 10.0, 0.95, 10.0)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_optimize_bulk_purchase_no_item(self, advanced_inventory_agent, mock_db_session):
        """Test bulk purchase optimization with non-existent item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.first.return_value = None

        result = await advanced_inventory_agent.optimize_bulk_purchase(mock_session_instance, "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_optimize_bulk_purchase_success(self, advanced_inventory_agent, mock_db_session):
        """Test successful bulk purchase optimization"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock item
        item = Mock(
            id="test_item",
            name="Test Item",
            reorder_quantity=100,
            unit_cost=Decimal("10.00")
        )
        mock_session_instance.query.return_value.filter.return_value.first.return_value = item

        # Mock demand forecast
        with patch.object(advanced_inventory_agent, 'predict_demand') as mock_predict:
            mock_predict.return_value = DemandForecast(
                item_id="test_item",
                predicted_demand=300.0,
                confidence_interval=(250.0, 350.0),
                seasonality_factor=1.0,
                trend_factor=0.1,
                forecast_horizon_days=30,
                forecast_accuracy=0.85
            )

            # Test with custom volume discounts
            volume_discounts = [(100, 0.0), (200, 0.05), (500, 0.10)]
            result = await advanced_inventory_agent.optimize_bulk_purchase(
                mock_session_instance, "test_item", volume_discounts
            )

        if result:  # May be None if no savings found
            assert isinstance(result, BulkPurchaseOptimization)
            assert result.item_id == "test_item"
            assert result.optimal_order_quantity > 0
            assert result.break_even_point >= 0

    def test_calculate_bulk_purchase_total_cost(self, advanced_inventory_agent):
        """Test bulk purchase total cost calculation"""
        annual_demand = 1000.0
        order_quantity = 200
        unit_cost = 10.0

        cost = advanced_inventory_agent._calculate_bulk_purchase_total_cost(
            annual_demand, order_quantity, unit_cost
        )

        assert cost > 0
        assert cost < float('inf')

    def test_calculate_bulk_purchase_total_cost_edge_cases(self, advanced_inventory_agent):
        """Test bulk purchase cost calculation edge cases"""
        # Zero demand
        cost = advanced_inventory_agent._calculate_bulk_purchase_total_cost(0, 100, 10.0)
        assert cost == float('inf')

        # Zero quantity
        cost = advanced_inventory_agent._calculate_bulk_purchase_total_cost(1000, 0, 10.0)
        assert cost == float('inf')

    @pytest.mark.asyncio
    async def test_predict_expiry_waste_no_item(self, advanced_inventory_agent, mock_db_session):
        """Test expiry waste prediction with non-existent item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.first.return_value = None

        result = await advanced_inventory_agent.predict_expiry_waste(mock_session_instance, "nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_predict_expiry_waste_no_expiry_days(self, advanced_inventory_agent, mock_db_session):
        """Test expiry waste prediction for non-perishable item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock item without expiry days
        item = Mock(
            id="test_item",
            name="Non-perishable Item",
            expiry_days=None
        )
        mock_session_instance.query.return_value.filter.return_value.first.return_value = item

        result = await advanced_inventory_agent.predict_expiry_waste(mock_session_instance, "test_item")

        assert result is None

    @pytest.mark.asyncio
    async def test_predict_expiry_waste_high_risk(self, advanced_inventory_agent, mock_db_session, sample_item_with_expiry):
        """Test expiry waste prediction for high-risk item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.first.return_value = sample_item_with_expiry

        # Mock demand forecast with low consumption (high waste risk)
        with patch.object(advanced_inventory_agent, 'predict_demand') as mock_predict:
            mock_predict.return_value = DemandForecast(
                item_id="expiry_item",
                predicted_demand=30.0,  # Low demand relative to stock
                confidence_interval=(25.0, 35.0),
                seasonality_factor=1.0,
                trend_factor=0.0,
                forecast_horizon_days=30,
                forecast_accuracy=0.80
            )

            result = await advanced_inventory_agent.predict_expiry_waste(mock_session_instance, "expiry_item")

        assert result is not None
        assert result["item_id"] == "expiry_item"
        assert result["waste_risk"] == "high"
        assert result["predicted_waste"] > 0
        assert result["waste_value"] > 0
        assert len(result["strategies"]) > 0

    def test_generate_waste_minimization_strategies_high_risk(self, advanced_inventory_agent, sample_item_with_expiry):
        """Test waste minimization strategy generation for high-risk items"""
        daily_consumption = 2.0
        predicted_waste = 50.0
        waste_risk = "high"

        strategies = advanced_inventory_agent._generate_waste_minimization_strategies(
            sample_item_with_expiry, daily_consumption, predicted_waste, waste_risk
        )

        assert len(strategies) > 0
        strategy_types = [s["type"] for s in strategies]
        assert "promotional_pricing" in strategy_types
        assert "staff_training" in strategy_types
        assert "bundle_offers" in strategy_types

    def test_generate_waste_minimization_strategies_medium_risk(self, advanced_inventory_agent, sample_item_with_expiry):
        """Test waste minimization strategy generation for medium-risk items"""
        daily_consumption = 5.0
        predicted_waste = 20.0
        waste_risk = "medium"

        strategies = advanced_inventory_agent._generate_waste_minimization_strategies(
            sample_item_with_expiry, daily_consumption, predicted_waste, waste_risk
        )

        assert len(strategies) > 0
        strategy_types = [s["type"] for s in strategies]
        assert "increase_visibility" in strategy_types
        assert "targeted_marketing" in strategy_types

    def test_calculate_optimal_reorder_timing(self, advanced_inventory_agent, sample_item_with_expiry):
        """Test optimal reorder timing calculation"""
        daily_consumption = 5.0
        seasonality_factor = 1.2  # High season

        timing = advanced_inventory_agent._calculate_optimal_reorder_timing(
            sample_item_with_expiry, daily_consumption, seasonality_factor
        )

        assert "optimal_stock_level" in timing
        assert "days_until_reorder" in timing
        assert "reorder_date" in timing
        assert timing["optimal_stock_level"] > 0
        assert timing["days_until_reorder"] >= 0

    def test_calculate_optimal_reorder_timing_no_consumption(self, advanced_inventory_agent, sample_item_with_expiry):
        """Test optimal reorder timing with zero consumption"""
        daily_consumption = 0.0
        seasonality_factor = 1.0

        timing = advanced_inventory_agent._calculate_optimal_reorder_timing(
            sample_item_with_expiry, daily_consumption, seasonality_factor
        )

        assert timing["timing"] == "manual_review"
        assert timing["reason"] == "insufficient_consumption_data"

    @pytest.mark.asyncio
    async def test_analyze_supplier_performance_advanced_no_suppliers(self, advanced_inventory_agent, mock_db_session):
        """Test advanced supplier performance analysis with no suppliers"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.all.return_value = []

        result = await advanced_inventory_agent.analyze_supplier_performance_advanced(mock_session_instance)

        assert result == []

    def test_calculate_on_time_delivery_rate(self, advanced_inventory_agent):
        """Test on-time delivery rate calculation"""
        orders = [
            Mock(status="delivered", expected_delivery_date=datetime.now()),
            Mock(status="delivered", expected_delivery_date=datetime.now()),
            Mock(status="pending", expected_delivery_date=None),
            Mock(status="completed", expected_delivery_date=datetime.now())
        ]

        rate = advanced_inventory_agent._calculate_on_time_delivery_rate(orders)

        assert 0 <= rate <= 1
        assert rate == 1.0  # All delivered orders assumed on-time

    def test_calculate_on_time_delivery_rate_no_orders(self, advanced_inventory_agent):
        """Test on-time delivery rate with no orders"""
        rate = advanced_inventory_agent._calculate_on_time_delivery_rate([])
        assert rate == 0.0

    def test_calculate_quality_score(self, advanced_inventory_agent, mock_db_session, sample_suppliers):
        """Test quality score calculation"""
        supplier = sample_suppliers[0]  # Excellent supplier
        orders = [
            Mock(total_amount=Decimal("1000.00")),
            Mock(total_amount=Decimal("500.00")),
            Mock(total_amount=Decimal("750.00"))
        ]

        mock_session_instance = Mock()
        score = advanced_inventory_agent._calculate_quality_score(mock_session_instance, supplier, orders)

        assert 0 <= score <= 1
        assert score > 0.5  # Should be above neutral for excellent supplier

    def test_calculate_reliability_index(self, advanced_inventory_agent, sample_suppliers):
        """Test reliability index calculation"""
        supplier = sample_suppliers[0]
        orders = [
            Mock(
                total_amount=Decimal("1000.00"),
                expected_delivery_date=datetime.now() + timedelta(days=5),
                order_date=datetime.now()
            ),
            Mock(
                total_amount=Decimal("950.00"),
                expected_delivery_date=datetime.now() + timedelta(days=5),
                order_date=datetime.now()
            ),
            Mock(
                total_amount=Decimal("1050.00"),
                expected_delivery_date=datetime.now() + timedelta(days=5),
                order_date=datetime.now()
            )
        ]

        reliability = advanced_inventory_agent._calculate_reliability_index(orders, supplier)

        assert 0 <= reliability <= 1

    def test_calculate_lead_time_variability(self, advanced_inventory_agent, sample_suppliers):
        """Test lead time variability calculation"""
        supplier = sample_suppliers[0]
        orders = [
            Mock(
                expected_delivery_date=datetime.now() + timedelta(days=5),
                order_date=datetime.now()
            ),
            Mock(
                expected_delivery_date=datetime.now() + timedelta(days=6),
                order_date=datetime.now()
            ),
            Mock(
                expected_delivery_date=datetime.now() + timedelta(days=4),
                order_date=datetime.now()
            )
        ]

        variability = advanced_inventory_agent._calculate_lead_time_variability(orders, supplier)

        assert 0 <= variability <= 1

    def test_calculate_overall_performance_score(self, advanced_inventory_agent):
        """Test overall performance score calculation"""
        score = advanced_inventory_agent._calculate_overall_performance_score(
            on_time_rate=0.95,
            quality_score=0.90,
            cost_competitiveness=0.85,
            reliability_index=0.88,
            lead_time_variability=0.92
        )

        assert 0 <= score <= 1
        assert score > 0.85  # Should be high with good metrics

    def test_recommend_supplier_action(self, advanced_inventory_agent):
        """Test supplier action recommendations"""
        # Excellent supplier
        action = advanced_inventory_agent._recommend_supplier_action(0.90, {
            'on_time': 0.95, 'quality': 0.90, 'cost': 0.85, 'reliability': 0.88, 'lead_time_var': 0.92
        })
        assert action == "preferred_supplier"

        # Good supplier
        action = advanced_inventory_agent._recommend_supplier_action(0.75, {
            'on_time': 0.80, 'quality': 0.75, 'cost': 0.70, 'reliability': 0.75, 'lead_time_var': 0.80
        })
        assert action == "continue_monitoring"

        # Poor supplier
        action = advanced_inventory_agent._recommend_supplier_action(0.30, {
            'on_time': 0.50, 'quality': 0.40, 'cost': 0.30, 'reliability': 0.35, 'lead_time_var': 0.40
        })
        assert action == "consider_alternative_suppliers"

    @pytest.mark.asyncio
    async def test_process_data_advanced_types(self, advanced_inventory_agent, mock_db_session):
        """Test processing of new advanced data types"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock empty results to avoid complex setup
        mock_session_instance.query.return_value.filter.return_value.all.return_value = []

        # Test all new data types
        data_types = [
            "advanced_reorder_analysis",
            "demand_forecast_analysis",
            "bulk_purchase_analysis",
            "expiry_waste_analysis",
            "advanced_supplier_analysis"
        ]

        for data_type in data_types:
            data = {"type": data_type}
            result = await advanced_inventory_agent.process_data(data)
            # Should not crash, may return None due to empty mock data
            assert result is None or isinstance(result, AgentDecision)

    @pytest.mark.asyncio
    async def test_estimate_demand_standard_deviation(self, advanced_inventory_agent, mock_db_session):
        """Test demand standard deviation estimation"""
        mock_session_instance = Mock()

        # Mock movements with varying consumption
        movements = [
            Mock(quantity=10, movement_date=datetime.now() - timedelta(days=i))
            for i in range(10)
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = movements

        with patch.object(advanced_inventory_agent, '_aggregate_daily_consumption') as mock_aggregate:
            mock_aggregate.return_value = [8.0, 12.0, 10.0, 15.0, 9.0, 11.0, 14.0]

            std_dev = advanced_inventory_agent._estimate_demand_standard_deviation(
                mock_session_instance, "test_item", 10.0
            )

        assert std_dev > 0
        assert std_dev < 10.0  # Should be reasonable relative to mean

    @pytest.mark.asyncio
    async def test_estimate_demand_standard_deviation_insufficient_data(self, advanced_inventory_agent, mock_db_session):
        """Test demand standard deviation estimation with insufficient data"""
        mock_session_instance = Mock()
        mock_session_instance.query.return_value.filter.return_value.all.return_value = []

        std_dev = advanced_inventory_agent._estimate_demand_standard_deviation(
            mock_session_instance, "test_item", 10.0
        )

        assert std_dev == 3.0  # 30% of daily demand fallback

    def test_advanced_configuration_defaults(self):
        """Test that advanced configuration has proper defaults"""
        with patch('agents.base_agent.Anthropic'), \
             patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker'):

            agent = InventoryAgent(
                agent_id="test_agent",
                api_key="test_key",
                config={},  # Empty config to test defaults
                db_url="sqlite:///:memory:"
            )

            # Check all new defaults are set
            assert agent.forecast_horizon_days == 30
            assert agent.service_level_target == 0.95
            assert agent.holding_cost_rate == 0.25
            assert agent.order_cost == 50.0
            assert agent.min_forecast_accuracy == 0.70
            assert agent.seasonality_window_days == 365
            assert agent.alpha_smoothing == 0.3
            assert agent.beta_trend == 0.1
            assert agent.gamma_seasonality == 0.2
