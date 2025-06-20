"""
Unit tests for Enhanced Inventory Agent
"""
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.inventory_agent_enhanced import (
    DemandForecast,
    EnhancedInventoryAgent,
    ExpiryIntelligence,
    OptimalReorderPoint,
    SupplierPerformance,
)
from models.inventory import ItemStatus, StockMovementType


class TestEnhancedInventoryAgent:
    """Test cases for Enhanced Inventory Agent"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Enhanced inventory analysis with demand forecast: Optimize reorder points")]
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
    def enhanced_config(self):
        """Enhanced inventory agent configuration"""
        return {
            "check_interval": 300,
            "low_stock_multiplier": 1.2,
            "reorder_lead_time": 7,
            "consumption_analysis_days": 30,
            "service_level_target": 0.95,
            "holding_cost_rate": 0.25,
            "ordering_cost": 50.0,
            "stockout_cost_multiplier": 3.0,
            "forecast_horizon_days": 30,
            "bulk_discount_tiers": {
                "100": 0.02,
                "250": 0.05,
                "500": 0.08
            }
        }

    @pytest.fixture
    def enhanced_inventory_agent(self, mock_anthropic, mock_db_session, enhanced_config):
        """Create enhanced inventory agent instance"""
        return EnhancedInventoryAgent(
            agent_id="enhanced_inventory_agent",
            api_key="test_api_key",
            config=enhanced_config,
            db_url="sqlite:///:memory:"
        )

    @pytest.fixture
    def sample_item(self):
        """Create sample inventory item"""
        return Mock(
            id="ITEM001",
            name="Test Item",
            sku="TEST-001",
            unit_cost=Decimal("10.00"),
            current_stock=50,
            minimum_stock=5,
            maximum_stock=100,
            reorder_point=15,
            reorder_quantity=40,
            status=ItemStatus.ACTIVE
        )

    @pytest.fixture
    def sample_movements(self):
        """Create sample stock movements"""
        movements = []
        base_date = datetime.now().date() - timedelta(days=30)

        for i in range(30):
            movement_date = base_date + timedelta(days=i)
            movements.append(Mock(
                item_id="ITEM001",
                movement_type=StockMovementType.OUT,
                quantity=5 + (i % 3),  # Varying consumption
                movement_date=movement_date,
                unit_cost=Decimal("10.00")
            ))

        return movements

    def test_initialization(self, enhanced_inventory_agent, enhanced_config):
        """Test enhanced agent initialization"""
        assert enhanced_inventory_agent.agent_id == "enhanced_inventory_agent"
        assert enhanced_inventory_agent.service_level_target == 0.95
        assert enhanced_inventory_agent.holding_cost_rate == 0.25
        assert enhanced_inventory_agent.forecast_horizon_days == 30
        assert len(enhanced_inventory_agent.bulk_discount_tiers) == 3

    def test_system_prompt_enhanced(self, enhanced_inventory_agent):
        """Test enhanced system prompt"""
        prompt = enhanced_inventory_agent.system_prompt
        assert "advanced analytics capabilities" in prompt
        assert "demand forecasting" in prompt
        assert "Bulk purchase optimization" in prompt
        assert "supplier performance analytics" in prompt
        assert "service level optimization" in prompt

    @pytest.mark.asyncio
    async def test_demand_forecasting_single_item(self, enhanced_inventory_agent, mock_db_session, sample_movements):
        """Test demand forecasting for single item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = sample_movements

        data = {
            "type": "demand_forecast_request",
            "item_id": "ITEM001",
            "forecast_days": 14
        }

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand') as mock_forecast:
            mock_forecast.return_value = DemandForecast(
                item_id="ITEM001",
                predicted_demand=70.0,
                confidence_interval=(60.0, 80.0),
                seasonality_factor=1.1,
                trend_factor=0.05,
                forecast_horizon_days=14,
                forecast_accuracy=0.85,
                historical_patterns={"avg_daily": 5.0, "std_daily": 1.2},
                revenue_correlation=0.7,
                method_used="ensemble_with_seasonality"
            )

            decision = await enhanced_inventory_agent.process_data(data)

            assert decision is not None
            assert decision.decision_type == "demand_forecast_analysis"
            assert decision.confidence > 0.5
            assert "forecast" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_forecast_item_demand_comprehensive(self, enhanced_inventory_agent, sample_movements):
        """Test comprehensive item demand forecasting"""
        mock_session = Mock()

        # Mock the query chain for movements
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = sample_movements

        with patch.object(enhanced_inventory_agent, '_calculate_revenue_correlation', return_value=0.75):
            forecast = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)

            assert forecast is not None
            assert forecast.item_id == "ITEM001"
            assert forecast.predicted_demand > 0
            assert forecast.forecast_horizon_days == 14
            assert forecast.forecast_accuracy > 0
            assert forecast.method_used == "ensemble_with_seasonality"
            assert len(forecast.historical_patterns) > 0

    @pytest.mark.asyncio
    async def test_forecast_item_demand_insufficient_data(self, enhanced_inventory_agent):
        """Test demand forecasting with insufficient data"""
        mock_session = Mock()

        # Return very few movements
        few_movements = [Mock(
            item_id="ITEM001",
            movement_type=StockMovementType.OUT,
            quantity=5,
            movement_date=datetime.now().date()
        )]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = few_movements

        forecast = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)

        assert forecast is None

    def test_calculate_weekly_seasonality(self, enhanced_inventory_agent):
        """Test weekly seasonality calculation"""
        # Create consumption data with weekly pattern
        consumption_series = []
        for week in range(4):
            for day in range(7):
                # Higher consumption on weekends
                base_consumption = 10 if day in [5, 6] else 5
                consumption_series.append(base_consumption + np.random.random())

        seasonality = enhanced_inventory_agent._calculate_weekly_seasonality(consumption_series)

        assert len(seasonality) == 7
        assert all(0.5 <= factor <= 2.0 for factor in seasonality.values())
        # Weekend days should have higher factors
        assert seasonality[5] > seasonality[0]  # Saturday > Monday
        assert seasonality[6] > seasonality[1]  # Sunday > Tuesday

    def test_calculate_forecast_accuracy(self, enhanced_inventory_agent):
        """Test forecast accuracy calculation"""
        # Create realistic consumption data
        consumption_series = [5, 6, 4, 7, 5, 6, 5, 4, 6, 5, 7, 4, 6, 5, 8, 5, 6, 4, 7, 5, 6]

        accuracy = enhanced_inventory_agent._calculate_forecast_accuracy(consumption_series)

        assert 0.1 <= accuracy <= 0.95
        assert isinstance(accuracy, float)

    @pytest.mark.asyncio
    async def test_optimal_reorder_point_calculation(self, enhanced_inventory_agent, sample_item):
        """Test optimal reorder point calculation"""
        mock_session = Mock()

        # Mock forecast
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=35.0,  # 5 per day for 7 days
            confidence_interval=(30.0, 40.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.85,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.5},
            revenue_correlation=0.7,
            method_used="ensemble"
        )

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            optimization = await enhanced_inventory_agent._calculate_optimal_reorder_point(mock_session, sample_item)

            assert optimization is not None
            assert optimization.item_id == "ITEM001"
            assert optimization.optimal_reorder_point > 0
            assert optimization.optimal_reorder_quantity > 0
            assert optimization.service_level == 0.95
            assert optimization.total_cost > 0
            assert optimization.safety_stock >= 0

    def test_get_z_score_for_service_level(self, enhanced_inventory_agent):
        """Test Z-score calculation for service levels"""
        # Test common service levels
        assert enhanced_inventory_agent._get_z_score_for_service_level(0.95) == 1.65
        assert enhanced_inventory_agent._get_z_score_for_service_level(0.99) == 2.33
        assert enhanced_inventory_agent._get_z_score_for_service_level(0.90) == 1.28

    @pytest.mark.asyncio
    async def test_reorder_optimization_comprehensive(self, enhanced_inventory_agent, mock_db_session):
        """Test comprehensive reorder point optimization"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock items needing optimization
        items = [
            Mock(id="ITEM001", name="Item 1", reorder_point=15, unit_cost=Decimal("10.00"),
                 minimum_stock=5, reorder_quantity=40, status=ItemStatus.ACTIVE),
            Mock(id="ITEM002", name="Item 2", reorder_point=20, unit_cost=Decimal("15.00"),
                 minimum_stock=8, reorder_quantity=50, status=ItemStatus.ACTIVE)
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = items

        # Mock optimization results
        optimization1 = OptimalReorderPoint(
            item_id="ITEM001", optimal_reorder_point=18, optimal_reorder_quantity=45,
            service_level=0.95, safety_stock=8, lead_time_demand=35.0,
            demand_variability=5.0, total_cost=500.0, holding_cost=200.0,
            ordering_cost=150.0, stockout_cost=150.0
        )

        optimization2 = OptimalReorderPoint(
            item_id="ITEM002", optimal_reorder_point=22, optimal_reorder_quantity=55,
            service_level=0.95, safety_stock=10, lead_time_demand=42.0,
            demand_variability=6.0, total_cost=750.0, holding_cost=300.0,
            ordering_cost=225.0, stockout_cost=225.0
        )

        with patch.object(enhanced_inventory_agent, '_calculate_optimal_reorder_point',
                         side_effect=[optimization1, optimization2]):

            decision = await enhanced_inventory_agent._optimize_reorder_points(mock_session_instance)

            assert decision is not None
            assert decision.decision_type == "reorder_point_optimization"
            assert decision.context["total_items_analyzed"] == 2
            assert decision.confidence >= 0.75

    @pytest.mark.asyncio
    async def test_bulk_purchase_optimization(self, enhanced_inventory_agent, mock_db_session, sample_item):
        """Test bulk purchase optimization"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock items needing bulk analysis
        sample_item.current_stock = 10  # Below reorder point
        mock_session_instance.query.return_value.filter.return_value.all.return_value = [sample_item]

        # Mock forecast
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=150.0,  # 90-day demand
            confidence_interval=(140.0, 160.0),
            seasonality_factor=1.0,
            trend_factor=0.05,
            forecast_horizon_days=90,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 1.67},
            revenue_correlation=0.6,
            method_used="ensemble"
        )

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            decision = await enhanced_inventory_agent._analyze_bulk_purchase_opportunities(mock_session_instance)

            # Should find optimization opportunities
            assert decision is not None
            assert decision.decision_type == "bulk_purchase_optimization"

    @pytest.mark.asyncio
    async def test_bulk_purchase_calculation(self, enhanced_inventory_agent, sample_item):
        """Test bulk purchase calculation logic"""
        mock_session = Mock()

        # Mock forecast for 90 days
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=120.0,  # 90-day demand
            confidence_interval=(110.0, 130.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=90,
            forecast_accuracy=0.85,
            historical_patterns={"avg_daily": 1.33},
            revenue_correlation=0.7,
            method_used="ensemble"
        )

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            optimization = await enhanced_inventory_agent._calculate_bulk_purchase_optimization(mock_session, sample_item)

            assert optimization is not None
            assert optimization.item_id == "ITEM001"
            assert optimization.optimal_order_quantity >= 100  # Should hit discount tier
            assert optimization.total_cost_savings >= 0
            assert optimization.roi_months > 0

    @pytest.mark.asyncio
    async def test_expiry_intelligence_analysis(self, enhanced_inventory_agent, mock_db_session):
        """Test expiry intelligence analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock perishable items
        perishable_item = Mock(
            id="ITEM001",
            name="Perishable Item",
            current_stock=30,
            maximum_stock=50,
            expiry_days=7,
            status=ItemStatus.ACTIVE
        )
        mock_session_instance.query.return_value.filter.return_value.all.return_value = [perishable_item]

        # Mock expiry analysis
        expiry_analysis = ExpiryIntelligence(
            item_id="ITEM001",
            predicted_waste_amount=5.0,
            optimal_ordering_frequency=3,
            risk_score=0.8,
            recommended_discount_timing=2,
            shelf_life_optimization={"reduce_order_quantity": True},
            rotation_efficiency=0.7
        )

        with patch.object(enhanced_inventory_agent, '_analyze_expiry_intelligence', return_value=expiry_analysis):
            decision = await enhanced_inventory_agent._perform_expiry_intelligence(mock_session_instance)

            assert decision is not None
            assert decision.decision_type == "expiry_intelligence_analysis"
            assert decision.context["high_risk_items"] >= 0

    @pytest.mark.asyncio
    async def test_expiry_intelligence_calculation(self, enhanced_inventory_agent):
        """Test expiry intelligence calculation"""
        mock_session = Mock()

        # Mock perishable item
        perishable_item = Mock(
            id="ITEM001",
            name="Milk",
            current_stock=20,
            maximum_stock=30,
            expiry_days=5
        )

        # Mock forecast
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=15.0,  # 5-day demand
            confidence_interval=(12.0, 18.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=5,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 3.0, "std_daily": 0.5},
            revenue_correlation=0.6,
            method_used="ensemble"
        )

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            analysis = await enhanced_inventory_agent._analyze_expiry_intelligence(mock_session, perishable_item)

            assert analysis is not None
            assert analysis.item_id == "ITEM001"
            assert analysis.predicted_waste_amount >= 0
            assert 0 <= analysis.risk_score <= 1
            assert analysis.recommended_discount_timing > 0
            assert 0 <= analysis.rotation_efficiency <= 1

    @pytest.mark.asyncio
    async def test_supplier_performance_analysis(self, enhanced_inventory_agent, mock_db_session):
        """Test supplier performance analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock suppliers with recent activity
        suppliers = [
            Mock(id="SUP001", name="Supplier 1"),
            Mock(id="SUP002", name="Supplier 2")
        ]
        mock_session_instance.query.return_value.join.return_value.filter.return_value.distinct.return_value.all.return_value = suppliers

        # Mock supplier performance
        performance1 = SupplierPerformance(
            supplier_id="SUP001",
            overall_score=0.85,
            reliability_score=0.9,
            cost_competitiveness=0.8,
            quality_score=0.85,
            delivery_performance=0.9,
            risk_assessment={"single_source_risk": 0.3},
            recommendation="Preferred supplier"
        )

        performance2 = SupplierPerformance(
            supplier_id="SUP002",
            overall_score=0.55,
            reliability_score=0.6,
            cost_competitiveness=0.7,
            quality_score=0.5,
            delivery_performance=0.6,
            risk_assessment={"single_source_risk": 0.4},
            recommendation="Review relationship"
        )

        with patch.object(enhanced_inventory_agent, '_analyze_individual_supplier_performance',
                         side_effect=[performance1, performance2]):

            decision = await enhanced_inventory_agent._analyze_supplier_performance(mock_session_instance)

            assert decision is not None
            assert decision.decision_type == "supplier_performance_analysis"
            assert decision.context["total_suppliers_analyzed"] == 2
            assert decision.context["poor_performers"] == 1  # SUP002 has score < 0.6

    @pytest.mark.asyncio
    async def test_individual_supplier_performance(self, enhanced_inventory_agent):
        """Test individual supplier performance calculation"""
        mock_session = Mock()

        supplier = Mock(id="SUP001", name="Test Supplier")

        # Mock purchase orders
        po1 = Mock(
            supplier_id="SUP001",
            order_date=datetime.now().date() - timedelta(days=30),
            expected_delivery_date=datetime.now().date() - timedelta(days=25),
            total_amount=Decimal("1000.00")
        )
        po2 = Mock(
            supplier_id="SUP001",
            order_date=datetime.now().date() - timedelta(days=15),
            expected_delivery_date=datetime.now().date() - timedelta(days=10),
            total_amount=Decimal("1500.00")
        )

        mock_session.query.return_value.filter.return_value.all.return_value = [po1, po2]

        performance = await enhanced_inventory_agent._analyze_individual_supplier_performance(mock_session, supplier)

        assert performance is not None
        assert performance.supplier_id == "SUP001"
        assert 0 <= performance.overall_score <= 1
        assert 0 <= performance.reliability_score <= 1
        assert performance.recommendation is not None

    @pytest.mark.asyncio
    async def test_comprehensive_inventory_analysis(self, enhanced_inventory_agent, mock_db_session):
        """Test comprehensive inventory health analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock inventory metrics
        mock_session_instance.query.return_value.filter.return_value.count.side_effect = [100, 15, 8]  # total, low stock, overstocked
        mock_session_instance.query.return_value.filter.return_value.scalar.side_effect = [50000.0, 2500.0]  # total value, recent movements

        decision = await enhanced_inventory_agent._comprehensive_inventory_analysis(mock_session_instance)

        assert decision is not None
        assert decision.decision_type == "comprehensive_inventory_analysis"
        assert decision.context["total_active_items"] == 100
        assert decision.context["low_stock_items"] == 15
        assert decision.context["overstocked_items"] == 8
        assert decision.confidence > 0.5

    @pytest.mark.asyncio
    async def test_enhanced_stock_movement_analysis(self, enhanced_inventory_agent, mock_db_session, sample_item):
        """Test enhanced stock movement analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        mock_session_instance.query.return_value.filter.return_value.first.return_value = sample_item

        movement_data = {
            "item_id": "ITEM001",
            "movement_type": StockMovementType.OUT,
            "quantity": 15  # Higher than normal
        }

        data = {
            "type": "stock_movement",
            "movement": movement_data
        }

        # Mock forecast
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=70.0,
            confidence_interval=(60.0, 80.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=14,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.0},
            revenue_correlation=0.7,
            method_used="ensemble"
        )

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            decision = await enhanced_inventory_agent.process_data(data)

            # Should detect unusual consumption
            assert decision is not None
            assert decision.decision_type == "unusual_consumption_alert"
            assert "unusual consumption" in decision.action.lower()

    @pytest.mark.asyncio
    async def test_consumption_movement_analysis(self, enhanced_inventory_agent, sample_item):
        """Test consumption movement analysis"""
        mock_session = Mock()

        # Mock forecast showing normal consumption is 5 units/day
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=35.0,
            confidence_interval=(30.0, 40.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.0},
            revenue_correlation=0.7,
            method_used="ensemble"
        )

        # Test unusual consumption (3x normal)
        decision = await enhanced_inventory_agent._analyze_consumption_movement(mock_session, sample_item, 15, forecast)

        assert decision is not None
        assert decision.decision_type == "unusual_consumption_alert"
        assert decision.context["consumption_ratio"] == 3.0

    @pytest.mark.asyncio
    async def test_receipt_movement_analysis(self, enhanced_inventory_agent, sample_item):
        """Test receipt movement analysis"""
        mock_session = Mock()

        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=35.0,
            confidence_interval=(30.0, 40.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.0},
            revenue_correlation=0.7,
            method_used="ensemble"
        )

        # Test overstock situation
        decision = await enhanced_inventory_agent._analyze_receipt_movement(mock_session, sample_item, 60, forecast)

        assert decision is not None
        assert decision.decision_type == "overstock_alert"
        assert decision.context["excess_quantity"] == 10  # 50 + 60 - 100 = 10 excess

    @pytest.mark.asyncio
    async def test_adjustment_movement_analysis(self, enhanced_inventory_agent, sample_item):
        """Test adjustment movement analysis"""
        mock_session = Mock()

        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=35.0,
            confidence_interval=(30.0, 40.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.0},
            revenue_correlation=0.7,
            method_used="ensemble"
        )

        # Test significant adjustment (20% of current stock)
        decision = await enhanced_inventory_agent._analyze_adjustment_movement(mock_session, sample_item, -10, forecast)

        assert decision is not None
        assert decision.decision_type == "inventory_adjustment_alert"
        assert decision.context["adjustment_percentage"] == 20.0  # 10/50 * 100

    @pytest.mark.asyncio
    async def test_generate_enhanced_report(self, enhanced_inventory_agent, mock_db_session):
        """Test enhanced report generation"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock inventory counts
        mock_session_instance.query.return_value.filter.return_value.count.side_effect = [100, 12]  # total, low stock

        # Mock forecasts
        with patch.object(enhanced_inventory_agent, '_forecast_all_items_demand', return_value=[]):
            report = await enhanced_inventory_agent.generate_report()

            assert isinstance(report, dict)
            assert report["agent_id"] == "enhanced_inventory_agent"
            assert report["report_type"] == "enhanced_inventory_intelligence"
            assert "intelligence_capabilities" in report
            assert "demand_forecasting" in report["intelligence_capabilities"]
            assert "bulk_purchase_analysis" in report["intelligence_capabilities"]

    @pytest.mark.asyncio
    async def test_periodic_check_enhanced(self, enhanced_inventory_agent):
        """Test enhanced periodic check with intelligent scheduling"""
        with patch.object(enhanced_inventory_agent, '_queue_analysis_message') as mock_queue:
            # Mock current time for different scenarios

            # Test 6 AM (demand forecasting time)
            with patch('agents.inventory_agent_enhanced.datetime') as mock_datetime:
                mock_datetime.now.return_value = Mock(hour=6, minute=15, weekday=lambda: 1)

                await enhanced_inventory_agent.periodic_check()

                # Should queue demand forecast
                mock_queue.assert_called()

    @pytest.mark.asyncio
    async def test_queue_analysis_message(self, enhanced_inventory_agent):
        """Test analysis message queuing"""
        mock_queue = AsyncMock()
        enhanced_inventory_agent.message_queue = mock_queue

        await enhanced_inventory_agent._queue_analysis_message("demand_forecast_request")

        mock_queue.put.assert_called_once()
        call_args = mock_queue.put.call_args[0][0]
        assert call_args["type"] == "demand_forecast_request"
        assert call_args["agent_id"] == "enhanced_inventory_agent"

    @pytest.mark.asyncio
    async def test_error_handling_in_process_data(self, enhanced_inventory_agent, mock_db_session):
        """Test error handling in enhanced process_data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        mock_session_instance.close.side_effect = Exception("Session close error")

        data = {"type": "invalid_type"}

        # Should handle the error gracefully
        decision = await enhanced_inventory_agent.process_data(data)

        assert decision is None

    @pytest.mark.asyncio
    async def test_forecast_all_items_demand(self, enhanced_inventory_agent, mock_db_session):
        """Test forecasting demand for all active items"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock active items with recent movement
        mock_session_instance.query.return_value.filter.return_value.join.return_value.filter.return_value.distinct.return_value.all.return_value = [
            ("ITEM001",),
            ("ITEM002",)
        ]

        # Mock individual forecasts
        forecast1 = DemandForecast(
            item_id="ITEM001", predicted_demand=50.0, confidence_interval=(45.0, 55.0),
            seasonality_factor=1.0, trend_factor=0.05, forecast_horizon_days=30,
            forecast_accuracy=0.8, historical_patterns={"avg_daily": 1.67},
            revenue_correlation=0.7, method_used="ensemble"
        )

        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', side_effect=[forecast1, None]):
            forecasts = await enhanced_inventory_agent._forecast_all_items_demand(mock_session_instance, 30)

            assert len(forecasts) == 1
            assert forecasts[0].item_id == "ITEM001"

    def test_config_defaults(self):
        """Test enhanced configuration defaults"""
        with patch('agents.base_agent.Anthropic'), \
             patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker'):

            agent = EnhancedInventoryAgent(
                agent_id="test_agent",
                api_key="test_key",
                config={},  # Empty config
                db_url="sqlite:///:memory:"
            )

            assert agent.service_level_target == 0.95
            assert agent.holding_cost_rate == 0.25
            assert agent.ordering_cost == 50.0
            assert agent.forecast_horizon_days == 30
            assert len(agent.bulk_discount_tiers) > 0
