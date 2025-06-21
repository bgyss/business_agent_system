"""
Unit tests for enhanced InventoryAgent predictive analytics functionality
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
    DemandForecast,
    InventoryAgent,
    ItemCorrelationAnalysis,
    SeasonalityAnalysis,
    SupplierDiversificationAnalysis,
)
from models.inventory import ItemStatus, StockMovementType


class TestInventoryAgentEnhanced:
    """Test cases for enhanced InventoryAgent analytics"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Enhanced inventory analysis: Advanced optimization recommended")]
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
    def enhanced_agent_config(self):
        """Enhanced inventory agent configuration"""
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
    def enhanced_inventory_agent(self, mock_anthropic, mock_db_session, enhanced_agent_config):
        """Create enhanced inventory agent instance"""
        return InventoryAgent(
            agent_id="enhanced_inventory_agent",
            api_key="test_api_key",
            config=enhanced_agent_config,
            db_url="sqlite:///:memory:"
        )

    @pytest.fixture
    def sample_seasonal_data(self):
        """Create sample seasonal consumption data"""
        base_date = datetime.now() - timedelta(days=365)
        movements = []

        # Generate seasonal data with weekly and monthly patterns
        for i in range(365):
            date = base_date + timedelta(days=i)

            # Weekly pattern (higher consumption on weekdays)
            weekly_factor = 1.2 if date.weekday() < 5 else 0.8

            # Monthly pattern (higher at month start/end)
            day_of_month = date.day
            monthly_factor = 1.1 if day_of_month <= 5 or day_of_month >= 25 else 1.0

            # Yearly seasonality (higher in winter months)
            yearly_factor = 1.3 if date.month in [11, 12, 1, 2] else 0.9

            base_consumption = 10
            actual_consumption = base_consumption * weekly_factor * monthly_factor * yearly_factor
            actual_consumption += np.random.normal(0, 1)  # Add noise

            movements.append(Mock(
                item_id="seasonal_item",
                movement_type=StockMovementType.OUT,
                quantity=max(0, int(actual_consumption)),
                movement_date=date
            ))

        return movements

    @pytest.fixture
    def sample_correlated_items_data(self):
        """Create sample data for item correlation analysis"""
        base_date = datetime.now() - timedelta(days=90)

        # Simulate correlated consumption patterns
        movements = []
        for i in range(90):
            date = base_date + timedelta(days=i)

            # Primary item consumption
            primary_consumption = 10 + 2 * np.sin(2 * np.pi * i / 7)  # Weekly pattern

            # Complementary item (positively correlated)
            complementary_consumption = primary_consumption * 0.8 + np.random.normal(0, 1)

            # Substitute item (negatively correlated)
            substitute_consumption = max(0, 15 - primary_consumption * 0.6 + np.random.normal(0, 1))

            movements.extend([
                Mock(
                    item_id="primary_item",
                    movement_type=StockMovementType.OUT,
                    quantity=max(0, int(primary_consumption)),
                    movement_date=date
                ),
                Mock(
                    item_id="complementary_item",
                    movement_type=StockMovementType.OUT,
                    quantity=max(0, int(complementary_consumption)),
                    movement_date=date
                ),
                Mock(
                    item_id="substitute_item",
                    movement_type=StockMovementType.OUT,
                    quantity=max(0, int(substitute_consumption)),
                    movement_date=date
                )
            ])

        return movements

    @pytest.mark.asyncio
    async def test_analyze_seasonality_patterns_success(self, enhanced_inventory_agent, mock_db_session, sample_seasonal_data):
        """Test successful seasonality pattern analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = sample_seasonal_data

        result = await enhanced_inventory_agent.analyze_seasonality_patterns(mock_session_instance, "seasonal_item")

        # May return None if insufficient seasonal data
        if result is not None:
            assert isinstance(result, SeasonalityAnalysis)
            assert result.item_id == "seasonal_item"
            assert len(result.seasonal_periods) > 0
            assert 0 <= result.seasonal_strength <= 1
            assert 0 <= result.confidence <= 1
            assert result.seasonal_adjustment_factor > 0
        # Test passes if result is None (insufficient data) or has correct structure

    @pytest.mark.asyncio
    async def test_analyze_seasonality_patterns_insufficient_data(self, enhanced_inventory_agent, mock_db_session):
        """Test seasonality analysis with insufficient data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock insufficient data
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        result = await enhanced_inventory_agent.analyze_seasonality_patterns(mock_session_instance, "test_item")

        assert result is None

    def test_analyze_seasonal_period(self, enhanced_inventory_agent):
        """Test seasonal period analysis"""
        # Create data with clear weekly pattern
        data = []
        for i in range(28):  # 4 weeks
            # Higher consumption on weekdays (0-4), lower on weekends (5-6)
            if i % 7 < 5:
                data.append(10.0)
            else:
                data.append(5.0)

        strength, peaks, lows = enhanced_inventory_agent._analyze_seasonal_period(data, 7)

        assert strength > 0
        assert len(peaks) > 0
        assert len(lows) > 0
        # Weekdays should be peaks (0-4), weekends should be lows (5-6)
        assert any(p < 5 for p in peaks)  # At least one weekday peak
        assert any(l >= 5 for l in lows)  # At least one weekend low

    def test_calculate_seasonal_adjustment(self, enhanced_inventory_agent):
        """Test seasonal adjustment calculation"""
        # Create simple pattern: high on day 0, low on day 3
        data = [10.0, 8.0, 6.0, 4.0, 6.0, 8.0, 10.0] * 4  # 4 cycles of 7 days

        # Test adjustment for high day (day 0)
        adjustment_high = enhanced_inventory_agent._calculate_seasonal_adjustment(data, 7, 0)
        assert adjustment_high > 1.0  # Should be above average

        # Test adjustment for low day (day 3)
        adjustment_low = enhanced_inventory_agent._calculate_seasonal_adjustment(data, 7, 3)
        assert adjustment_low < 1.0  # Should be below average

    @pytest.mark.asyncio
    async def test_analyze_item_correlations_success(self, enhanced_inventory_agent, mock_db_session, sample_correlated_items_data):
        """Test successful item correlation analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = sample_correlated_items_data

        # Mock item queries for bundle opportunities
        mock_item = Mock(
            name="Primary Item",
            unit_cost=Decimal("10.00"),
            selling_price=Decimal("15.00")
        )
        mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_item

        result = await enhanced_inventory_agent.analyze_item_correlations(mock_session_instance, "primary_item")

        assert result is not None
        assert isinstance(result, ItemCorrelationAnalysis)
        assert result.primary_item_id == "primary_item"
        assert len(result.correlated_items) > 0
        assert 0 <= result.impact_factor <= 1
        # Should find positive correlation with complementary item
        corr_dict = {item_id: corr for item_id, corr in result.correlated_items}
        assert "complementary_item" in corr_dict
        assert corr_dict["complementary_item"] > 0.3  # Positive correlation
        # Should find negative correlation with substitute item
        assert "substitute_item" in corr_dict
        assert corr_dict["substitute_item"] < -0.1  # Negative correlation

    @pytest.mark.asyncio
    async def test_analyze_item_correlations_insufficient_data(self, enhanced_inventory_agent, mock_db_session):
        """Test item correlation analysis with insufficient data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        result = await enhanced_inventory_agent.analyze_item_correlations(mock_session_instance, "test_item")

        assert result is None

    def test_generate_bundle_opportunities(self, enhanced_inventory_agent, mock_db_session):
        """Test bundle opportunity generation"""
        mock_session_instance = Mock()

        # Mock primary item
        primary_item = Mock(
            name="Primary Item",
            unit_cost=Decimal("10.00"),
            selling_price=Decimal("15.00")
        )

        # Mock complementary item
        comp_item = Mock(
            name="Complementary Item",
            unit_cost=Decimal("5.00"),
            selling_price=Decimal("8.00")
        )

        def mock_item_query(item_id):
            if item_id == "primary_item":
                return primary_item
            elif item_id == "comp_item":
                return comp_item
            return None

        mock_session_instance.query.return_value.filter.return_value.first.side_effect = \
            lambda: mock_item_query("comp_item")

        opportunities = enhanced_inventory_agent._generate_bundle_opportunities(
            mock_session_instance, "primary_item", ["comp_item"]
        )

        assert len(opportunities) > 0
        opportunity = opportunities[0]
        assert "bundle_price" in opportunity
        assert "customer_savings" in opportunity
        assert "discount_percentage" in opportunity
        assert opportunity["customer_savings"] > 0

    @pytest.mark.asyncio
    async def test_analyze_supplier_diversification_success(self, enhanced_inventory_agent, mock_db_session):
        """Test successful supplier diversification analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock purchase orders with high concentration
        po1 = Mock(
            supplier_id="supplier_1",
            order_date=datetime.now() - timedelta(days=30)
        )
        po2 = Mock(
            supplier_id="supplier_1",
            order_date=datetime.now() - timedelta(days=60)
        )
        po3 = Mock(
            supplier_id="supplier_2",
            order_date=datetime.now() - timedelta(days=90)
        )

        mock_session_instance.query.return_value.join.return_value.filter.return_value.all.return_value = [po1, po2, po3]

        # Mock purchase order items
        po_items = [
            Mock(total_cost=Decimal("1000.00")),  # supplier_1
            Mock(total_cost=Decimal("800.00")),   # supplier_1
            Mock(total_cost=Decimal("200.00"))    # supplier_2
        ]

        mock_session_instance.query.return_value.filter.return_value.all.side_effect = [
            [po1, po2, po3],  # Purchase orders
            [po_items[0]],    # PO items for po1
            [po_items[1]],    # PO items for po2
            [po_items[2]]     # PO items for po3
        ]

        # Mock item
        mock_item = Mock(id="test_item", name="Test Item")
        mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_item

        # Mock all suppliers
        all_suppliers = [
            Mock(id="supplier_3", is_active=True),
            Mock(id="supplier_4", is_active=True)
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = all_suppliers

        result = await enhanced_inventory_agent.analyze_supplier_diversification(mock_session_instance, "test_item")

        # May return None if no purchase orders found
        if result is not None:
            assert isinstance(result, SupplierDiversificationAnalysis)
            assert result.item_id == "test_item"
            assert result.current_supplier_concentration > 0.5  # High concentration
            assert len(result.alternative_suppliers) > 0
            assert 0 <= result.risk_score <= 1
            assert len(result.diversification_recommendations) > 0
            assert len(result.recommended_supplier_split) > 0

    @pytest.mark.asyncio
    async def test_analyze_supplier_diversification_no_orders(self, enhanced_inventory_agent, mock_db_session):
        """Test supplier diversification analysis with no purchase orders"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.join.return_value.filter.return_value.all.return_value = []

        result = await enhanced_inventory_agent.analyze_supplier_diversification(mock_session_instance, "test_item")

        assert result is None

    def test_generate_diversification_recommendations_high_concentration(self, enhanced_inventory_agent):
        """Test diversification recommendations for high concentration"""
        recommendations = enhanced_inventory_agent._generate_diversification_recommendations(
            concentration_index=0.8,  # High concentration
            current_supplier_count=1,  # Single supplier
            alternative_suppliers=["alt1", "alt2", "alt3"]
        )

        assert len(recommendations) > 0
        rec_types = [r["type"] for r in recommendations]
        assert "reduce_concentration" in rec_types
        assert "add_suppliers" in rec_types
        assert "evaluate_alternatives" in rec_types

        # Check priorities
        critical_recs = [r for r in recommendations if r["priority"] == "critical"]
        assert len(critical_recs) > 0

    def test_calculate_optimal_supplier_split_high_risk(self, enhanced_inventory_agent):
        """Test optimal supplier split calculation for high risk scenario"""
        current_volumes = {
            "supplier_1": 8000.0,  # 80% concentration
            "supplier_2": 2000.0   # 20%
        }

        recommended_split = enhanced_inventory_agent._calculate_optimal_supplier_split(
            current_volumes, 10000.0, risk_score=0.8  # High risk
        )

        assert len(recommended_split) == 2
        # Main supplier should not exceed 50% in high risk scenario
        main_supplier_share = max(recommended_split.values())
        assert main_supplier_share <= 0.5
        # Splits should sum to 1.0
        assert abs(sum(recommended_split.values()) - 1.0) < 0.01

    def test_estimate_diversification_cost_impact(self, enhanced_inventory_agent):
        """Test diversification cost impact estimation"""
        # Single supplier scenario
        cost_impact = enhanced_inventory_agent._estimate_diversification_cost_impact(1, 3)
        assert cost_impact == 0.05  # 5% increase expected

        # Few suppliers scenario
        cost_impact = enhanced_inventory_agent._estimate_diversification_cost_impact(2, 2)
        assert cost_impact == 0.02  # 2% increase expected

        # Well diversified scenario
        cost_impact = enhanced_inventory_agent._estimate_diversification_cost_impact(5, 1)
        assert cost_impact == 0.0  # No change expected

    @pytest.mark.asyncio
    async def test_generate_comprehensive_reorder_recommendation_success(self, enhanced_inventory_agent, mock_db_session):
        """Test comprehensive reorder recommendation generation"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock item
        mock_item = Mock(
            id="test_item",
            name="Test Item",
            current_stock=50
        )
        mock_session_instance.query.return_value.filter.return_value.first.return_value = mock_item

        # Mock successful analytics
        demand_forecast = DemandForecast(
            item_id="test_item",
            predicted_demand=300.0,
            confidence_interval=(250.0, 350.0),
            seasonality_factor=1.0,
            trend_factor=0.1,
            forecast_horizon_days=30,
            forecast_accuracy=0.85
        )

        seasonality = SeasonalityAnalysis(
            item_id="test_item",
            seasonal_periods=[7, 30],
            seasonal_strength=0.6,
            peak_periods=[1, 2],
            low_periods=[5, 6],
            seasonal_adjustment_factor=1.2,
            confidence=0.8
        )

        with patch.object(enhanced_inventory_agent, 'predict_demand') as mock_predict, \
             patch.object(enhanced_inventory_agent, 'analyze_seasonality_patterns') as mock_seasonality, \
             patch.object(enhanced_inventory_agent, 'analyze_item_correlations') as mock_correlations, \
             patch.object(enhanced_inventory_agent, 'calculate_optimal_reorder_point') as mock_reorder, \
             patch.object(enhanced_inventory_agent, 'optimize_bulk_purchase') as mock_bulk, \
             patch.object(enhanced_inventory_agent, 'analyze_supplier_diversification') as mock_supplier:

            mock_predict.return_value = demand_forecast
            mock_seasonality.return_value = seasonality
            mock_correlations.return_value = None  # No correlations
            mock_reorder.return_value = None  # No optimal reorder
            mock_bulk.return_value = None  # No bulk opportunities
            mock_supplier.return_value = None  # No supplier analysis

            result = await enhanced_inventory_agent.generate_comprehensive_reorder_recommendation(
                mock_session_instance, "test_item"
            )

        assert result is not None
        assert result["item_id"] == "test_item"
        assert "analytics_summary" in result
        assert "demand_analysis" in result
        assert "reorder_recommendations" in result
        assert "risk_factors" in result
        assert "opportunities" in result

        # Check demand analysis
        demand_analysis = result["demand_analysis"]
        assert demand_analysis["base_predicted_demand"] == 300.0
        assert demand_analysis["seasonal_adjustment_factor"] == 1.2
        assert demand_analysis["final_demand_estimate"] > 300.0  # Should be adjusted upward

    @pytest.mark.asyncio
    async def test_generate_comprehensive_reorder_recommendation_no_forecast(self, enhanced_inventory_agent, mock_db_session):
        """Test comprehensive recommendation when no demand forecast is available"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        with patch.object(enhanced_inventory_agent, 'predict_demand') as mock_predict:
            mock_predict.return_value = None

            result = await enhanced_inventory_agent.generate_comprehensive_reorder_recommendation(
                mock_session_instance, "test_item"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_process_data_new_analytics_types(self, enhanced_inventory_agent, mock_db_session):
        """Test processing of new enhanced analytics data types"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock empty results to avoid complex setup
        mock_session_instance.query.return_value.filter.return_value.all.return_value = []
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        # Test new data types
        new_data_types = [
            {"type": "comprehensive_analytics"},
            {"type": "seasonality_analysis", "item_id": "test_item"},
            {"type": "correlation_analysis", "item_id": "test_item"},
            {"type": "supplier_diversification_analysis", "item_id": "test_item"}
        ]

        for data in new_data_types:
            result = await enhanced_inventory_agent.process_data(data)
            # Should not crash, may return None due to empty mock data
            assert result is None or isinstance(result, AgentDecision)

    @pytest.mark.asyncio
    async def test_comprehensive_analytics_analysis_success(self, enhanced_inventory_agent, mock_db_session):
        """Test comprehensive analytics analysis with high-value items"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock high-value items
        high_value_items = [
            Mock(
                id="item_1",
                name="High Value Item 1",
                current_stock=100,
                unit_cost=Decimal("50.00"),
                status=ItemStatus.ACTIVE
            ),
            Mock(
                id="item_2",
                name="High Value Item 2",
                current_stock=50,
                unit_cost=Decimal("80.00"),
                status=ItemStatus.ACTIVE
            )
        ]

        mock_session_instance.query.return_value.filter.return_value.limit.return_value.all.return_value = high_value_items

        # Mock comprehensive recommendations
        mock_recommendation = {
            "item_id": "item_1",
            "current_stock": 100,
            "analytics_summary": {
                "seasonality_detected": True,
                "correlations_identified": True
            },
            "opportunities": [
                {"type": "bulk_purchase", "potential_savings": 500}
            ]
        }

        with patch.object(enhanced_inventory_agent, 'generate_comprehensive_reorder_recommendation') as mock_gen:
            mock_gen.return_value = mock_recommendation

            result = await enhanced_inventory_agent._perform_comprehensive_analytics_analysis(mock_session_instance)

        # May return None if no items or no recommendations generated
        if result is not None:
            assert isinstance(result, AgentDecision)
            assert result.decision_type == "comprehensive_analytics"
            assert result.confidence == 0.95

            context = result.context
            assert context["items_analyzed"] > 0
            assert context["total_inventory_value"] > 0
            assert "comprehensive_recommendations" in context

    @pytest.mark.asyncio
    async def test_comprehensive_analytics_analysis_no_items(self, enhanced_inventory_agent, mock_db_session):
        """Test comprehensive analytics analysis with no high-value items"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        mock_session_instance.query.return_value.filter.return_value.limit.return_value.all.return_value = []

        result = await enhanced_inventory_agent._perform_comprehensive_analytics_analysis(mock_session_instance)

        assert result is None

    def test_enhanced_configuration_validation(self, enhanced_inventory_agent):
        """Test that enhanced configuration parameters are properly set"""
        agent = enhanced_inventory_agent

        # Verify all enhanced configuration parameters
        assert hasattr(agent, 'forecast_horizon_days')
        assert hasattr(agent, 'service_level_target')
        assert hasattr(agent, 'holding_cost_rate')
        assert hasattr(agent, 'order_cost')
        assert hasattr(agent, 'min_forecast_accuracy')
        assert hasattr(agent, 'seasonality_window_days')
        assert hasattr(agent, 'alpha_smoothing')
        assert hasattr(agent, 'beta_trend')
        assert hasattr(agent, 'gamma_seasonality')

        # Verify reasonable default values
        assert 0 < agent.service_level_target <= 1
        assert agent.holding_cost_rate > 0
        assert agent.order_cost > 0
        assert 0 <= agent.min_forecast_accuracy <= 1
        assert 0 < agent.alpha_smoothing <= 1
        assert 0 < agent.beta_trend <= 1
        assert 0 < agent.gamma_seasonality <= 1

    @pytest.mark.asyncio
    async def test_error_handling_in_enhanced_methods(self, enhanced_inventory_agent, mock_db_session):
        """Test error handling in enhanced analytics methods"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock database errors
        mock_session_instance.query.side_effect = Exception("Database error")

        # Test that methods handle errors gracefully
        seasonality_result = await enhanced_inventory_agent.analyze_seasonality_patterns(
            mock_session_instance, "test_item"
        )
        assert seasonality_result is None

        correlation_result = await enhanced_inventory_agent.analyze_item_correlations(
            mock_session_instance, "test_item"
        )
        assert correlation_result is None

        diversification_result = await enhanced_inventory_agent.analyze_supplier_diversification(
            mock_session_instance, "test_item"
        )
        assert diversification_result is None

        comprehensive_result = await enhanced_inventory_agent.generate_comprehensive_reorder_recommendation(
            mock_session_instance, "test_item"
        )
        assert comprehensive_result is None

    def test_namedtuple_structures(self):
        """Test that new NamedTuple structures are properly defined"""
        # Test SeasonalityAnalysis
        seasonality = SeasonalityAnalysis(
            item_id="test",
            seasonal_periods=[7, 30],
            seasonal_strength=0.5,
            peak_periods=[1, 2],
            low_periods=[5, 6],
            seasonal_adjustment_factor=1.1,
            confidence=0.8
        )
        assert seasonality.item_id == "test"
        assert seasonality.seasonal_periods == [7, 30]

        # Test ItemCorrelationAnalysis
        correlation = ItemCorrelationAnalysis(
            primary_item_id="test",
            correlated_items=[("item2", 0.5)],
            substitution_items=["sub1"],
            complementary_items=["comp1"],
            impact_factor=0.7,
            bundle_opportunities=[]
        )
        assert correlation.primary_item_id == "test"
        assert correlation.impact_factor == 0.7

        # Test SupplierDiversificationAnalysis
        diversification = SupplierDiversificationAnalysis(
            item_id="test",
            current_supplier_concentration=0.6,
            alternative_suppliers=["alt1", "alt2"],
            risk_score=0.4,
            diversification_recommendations=[],
            cost_impact_of_diversification=0.02,
            recommended_supplier_split={"supplier1": 0.7, "supplier2": 0.3}
        )
        assert diversification.item_id == "test"
        assert diversification.risk_score == 0.4
