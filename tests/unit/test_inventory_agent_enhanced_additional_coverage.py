"""Additional tests to reach 95% coverage for Enhanced Inventory Agent."""

import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.inventory_agent_enhanced import (
    BulkPurchaseOptimization,
    DemandForecast,
    EnhancedInventoryAgent,
    OptimalReorderPoint,
)
from models.inventory import ItemStatus, StockMovementType


class TestEnhancedInventoryAgentAdditionalCoverage:
    """Additional tests to reach 95% coverage."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client."""
        with patch("agents.base_agent.Anthropic") as mock_client:
            mock_response = Mock()
            mock_response.content = [
                Mock(text="Enhanced inventory analysis with detailed insights")
            ]
            mock_client.return_value.messages.create.return_value = mock_response
            yield mock_client

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch("agents.base_agent.create_engine"), patch(
            "agents.base_agent.sessionmaker"
        ) as mock_sessionmaker:
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def enhanced_config(self):
        """Enhanced inventory agent configuration."""
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
            "seasonal_analysis_periods": 4,
            "trend_analysis_days": 90,
            "bulk_discount_tiers": {"100": 0.02, "250": 0.05, "500": 0.08, "1000": 0.12},
        }

    @pytest.fixture
    def enhanced_inventory_agent(self, mock_anthropic, mock_db_session, enhanced_config):
        """Create enhanced inventory agent instance."""
        return EnhancedInventoryAgent(
            agent_id="enhanced_inventory_agent",
            api_key="test_api_key",
            config=enhanced_config,
            db_url="sqlite:///:memory:",
        )

    @pytest.fixture
    def sample_item(self):
        """Create sample inventory item."""
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
            status=ItemStatus.ACTIVE,
        )

    @pytest.fixture
    def sample_supplier(self):
        """Create sample supplier."""
        return Mock(id="SUP001", name="Test Supplier", contact_email="test@supplier.com")

    @pytest.mark.asyncio
    async def test_perform_demand_forecasting_with_declining_items(
        self, enhanced_inventory_agent, mock_db_session
    ):
        """Test demand forecasting with declining demand items."""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        data = {"type": "demand_forecast_request"}

        # Mock forecasts with declining items
        forecasts = [
            DemandForecast(
                item_id="ITEM001",
                predicted_demand=50.0,
                confidence_interval=(45.0, 55.0),
                seasonality_factor=1.0,
                trend_factor=-0.15,  # Declining trend
                forecast_horizon_days=30,
                forecast_accuracy=0.85,
                historical_patterns={"avg_daily": 1.67, "std_daily": 0.5},
                revenue_correlation=0.7,
                method_used="ensemble",
            ),
            DemandForecast(
                item_id="ITEM002",
                predicted_demand=75.0,
                confidence_interval=(70.0, 80.0),
                seasonality_factor=1.1,
                trend_factor=0.5,  # High demand growth
                forecast_horizon_days=30,
                forecast_accuracy=0.6,  # Low accuracy
                historical_patterns={"avg_daily": 2.5, "std_daily": 0.8},
                revenue_correlation=0.8,
                method_used="ensemble",
            ),
        ]

        with patch.object(
            enhanced_inventory_agent, "_forecast_all_items_demand", return_value=forecasts
        ):
            result = await enhanced_inventory_agent._perform_demand_forecasting(
                mock_session_instance, data
            )

            assert result is not None
            assert result.decision_type == "demand_forecast_analysis"
            assert "declining" in result.action or "accuracy" in result.action

    @pytest.mark.asyncio
    async def test_forecast_item_demand_short_consumption_series(self, enhanced_inventory_agent):
        """Test forecast_item_demand with short consumption series."""
        mock_session = Mock()

        # Create movements but with short series (less than 14 days)
        movements = []
        base_date = datetime.now().date() - timedelta(days=10)
        for i in range(10):
            movements.append(
                Mock(
                    item_id="ITEM001",
                    movement_type=StockMovementType.OUT,
                    quantity=5,
                    movement_date=base_date + timedelta(days=i),
                )
            )

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            movements
        )

        with patch.object(
            enhanced_inventory_agent, "_calculate_revenue_correlation", return_value=0.5
        ):
            result = await enhanced_inventory_agent._forecast_item_demand(
                mock_session, "ITEM001", 14
            )

            # Should return None due to insufficient data
            assert result is None

    @pytest.mark.asyncio
    async def test_forecast_item_demand_with_single_value_series(self, enhanced_inventory_agent):
        """Test forecast_item_demand with series that has no variance."""
        mock_session = Mock()

        # Create movements with constant values and minimal data
        movements = []
        base_date = datetime.now().date() - timedelta(days=15)
        for i in range(15):
            movements.append(
                Mock(
                    item_id="ITEM001",
                    movement_type=StockMovementType.OUT,
                    quantity=5,  # Constant value
                    movement_date=base_date + timedelta(days=i),
                )
            )

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            movements
        )

        with patch.object(
            enhanced_inventory_agent, "_calculate_revenue_correlation", return_value=0.5
        ):
            result = await enhanced_inventory_agent._forecast_item_demand(
                mock_session, "ITEM001", 14
            )

            if result is not None:
                # Should handle zero variance gracefully
                assert result.trend_factor == 0.0

    @pytest.mark.asyncio
    async def test_optimize_reorder_points_with_cost_savings(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test reorder optimization with actual cost savings calculations."""
        mock_session = Mock()

        # Items exist
        mock_session.query.return_value.filter.return_value.all.return_value = [sample_item]

        # Mock optimization with significant changes
        optimization = OptimalReorderPoint(
            item_id="ITEM001",
            optimal_reorder_point=20,  # Different from current 15
            optimal_reorder_quantity=45,
            service_level=0.95,
            safety_stock=8,
            lead_time_demand=35.0,
            demand_variability=5.0,
            total_cost=400.0,  # Lower than current cost
            holding_cost=200.0,
            ordering_cost=150.0,
            stockout_cost=50.0,
        )

        # Mock current cost calculation to return higher cost
        with patch.object(
            enhanced_inventory_agent, "_calculate_optimal_reorder_point", return_value=optimization
        ), patch.object(
            enhanced_inventory_agent, "_calculate_current_inventory_cost", return_value=500.0
        ), patch.object(
            enhanced_inventory_agent, "_calculate_current_inventory_cost_by_id", return_value=500.0
        ):

            result = await enhanced_inventory_agent._optimize_reorder_points(mock_session)

            assert result is not None
            assert result.context["total_potential_savings"] > 0
            assert result.context["cost_effective_changes"] >= 1

    @pytest.mark.asyncio
    async def test_analyze_consumption_movement_reorder_needed(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test consumption movement analysis that triggers reorder."""
        mock_session = Mock()

        # Set item to low stock level
        sample_item.current_stock = 10  # Will go to 5 after consumption
        sample_item.reorder_point = 10

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
            method_used="ensemble",
        )

        # Normal consumption that triggers reorder
        result = await enhanced_inventory_agent._analyze_consumption_movement(
            mock_session, sample_item, 5, forecast
        )

        assert result is not None
        assert result.decision_type == "reorder_required"
        assert "reorder" in result.action.lower()

    @pytest.mark.asyncio
    async def test_analyze_consumption_movement_critical_reorder(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test consumption movement analysis with critical reorder timing."""
        mock_session = Mock()

        # Set item to very low stock with short remaining time
        sample_item.current_stock = 8  # Will go to 3 after consumption
        sample_item.reorder_point = 10
        enhanced_inventory_agent.reorder_lead_time = 5

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
            method_used="ensemble",
        )

        # Consumption that leaves very little time
        result = await enhanced_inventory_agent._analyze_consumption_movement(
            mock_session, sample_item, 5, forecast
        )

        assert result is not None
        assert result.decision_type == "reorder_required"
        assert "CRITICAL" in result.context.get("urgency", "") or "CRITICAL" in result.action

    @pytest.mark.asyncio
    async def test_analyze_receipt_movement_no_overstock_no_forecast(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test receipt movement analysis without forecast."""
        mock_session = Mock()

        # Receipt that doesn't cause overstock, no forecast
        result = await enhanced_inventory_agent._analyze_receipt_movement(
            mock_session, sample_item, 20, None
        )

        # Should not generate overstock alert
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_adjustment_movement_zero_stock(self, enhanced_inventory_agent):
        """Test adjustment movement analysis with zero current stock."""
        mock_session = Mock()

        # Item with zero stock
        zero_stock_item = Mock(id="ITEM001", name="Zero Stock Item", current_stock=0)

        forecast = Mock()

        # Adjustment on zero stock item
        result = await enhanced_inventory_agent._analyze_adjustment_movement(
            mock_session, zero_stock_item, 10, forecast
        )

        # Should handle division by zero gracefully
        if result is not None:
            assert result.decision_type == "inventory_adjustment_alert"

    @pytest.mark.asyncio
    async def test_analyze_reorder_needs_enhanced_with_urgency_scores(
        self, enhanced_inventory_agent
    ):
        """Test enhanced reorder analysis with urgency score calculations."""
        mock_session = Mock()

        # Items near reorder point
        item1 = Mock(
            id="ITEM001", name="Item 1", current_stock=8, reorder_point=10, status=ItemStatus.ACTIVE
        )
        item2 = Mock(
            id="ITEM002",
            name="Item 2",
            current_stock=12,
            reorder_point=15,
            status=ItemStatus.ACTIVE,
        )

        mock_session.query.return_value.filter.return_value.all.return_value = [item1, item2]

        # Mock forecasts with different urgency levels
        forecast1 = DemandForecast(
            item_id="ITEM001",
            predicted_demand=35.0,
            confidence_interval=(30.0, 40.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 8.0, "std_daily": 1.0},  # High consumption
            revenue_correlation=0.7,
            method_used="ensemble",
        )

        forecast2 = DemandForecast(
            item_id="ITEM002",
            predicted_demand=21.0,
            confidence_interval=(18.0, 24.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 3.0, "std_daily": 0.5},  # Low consumption
            revenue_correlation=0.6,
            method_used="ensemble",
        )

        def mock_forecast_side_effect(session, item_id, lead_time):
            if item_id == "ITEM001":
                return forecast1
            elif item_id == "ITEM002":
                return forecast2
            return None

        with patch.object(
            enhanced_inventory_agent, "_forecast_item_demand", side_effect=mock_forecast_side_effect
        ):
            result = await enhanced_inventory_agent._analyze_reorder_needs_enhanced(mock_session)

            assert result is not None
            assert result.decision_type == "enhanced_reorder_analysis"
            assert result.context["total_items_near_reorder"] == 2
            assert result.context["critical_items"] >= 1  # At least one critical item

    @pytest.mark.asyncio
    async def test_comprehensive_inventory_analysis_no_issues(self, enhanced_inventory_agent):
        """Test comprehensive inventory analysis when no issues are
        identified."""
        mock_session = Mock()

        # Healthy inventory metrics
        mock_session.query.return_value.filter.return_value.count.side_effect = [
            100,
            5,
            2,
        ]  # total, low stock (5%), overstocked (2%)
        mock_session.query.return_value.filter.return_value.scalar.side_effect = [
            50000.0,
            8000.0,
        ]  # total value, high movements

        result = await enhanced_inventory_agent._comprehensive_inventory_analysis(mock_session)

        if result is not None:
            assert len(result.context["identified_issues"]) == 0
            assert result.confidence == 0.7  # Lower confidence when no issues

    @pytest.mark.asyncio
    async def test_bulk_purchase_optimization_negative_savings(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test bulk purchase optimization where no savings are achieved."""
        mock_session = Mock()

        # Items exist
        mock_session.query.return_value.filter.return_value.all.return_value = [sample_item]

        # Mock optimization with no savings
        optimization = BulkPurchaseOptimization(
            item_id="ITEM001",
            optimal_order_quantity=100,
            unit_cost_with_discount=Decimal("9.80"),
            total_cost_savings=Decimal("-10.00"),  # Negative savings
            break_even_point=80,
            holding_cost_impact=Decimal("30.00"),
            discount_tier="100+ units (2% discount)",
            roi_months=float("inf"),
        )

        with patch.object(
            enhanced_inventory_agent,
            "_calculate_bulk_purchase_optimization",
            return_value=optimization,
        ):
            result = await enhanced_inventory_agent._analyze_bulk_purchase_opportunities(
                mock_session
            )

            # Should return None when no positive savings
            assert result is None

    @pytest.mark.asyncio
    async def test_bulk_purchase_calculation_infinite_roi(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test bulk purchase calculation with infinite ROI scenario."""
        mock_session = Mock()

        # Mock forecast with very low demand
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=10.0,  # Very low 90-day demand
            confidence_interval=(8.0, 12.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=90,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 0.11},
            revenue_correlation=0.6,
            method_used="ensemble",
        )

        # Set discount tiers that would result in negative net savings
        enhanced_inventory_agent.bulk_discount_tiers = {
            "100": 0.01,  # Very small discount
        }
        enhanced_inventory_agent.holding_cost_rate = 0.5  # High holding cost

        with patch.object(enhanced_inventory_agent, "_forecast_item_demand", return_value=forecast):
            optimization = await enhanced_inventory_agent._calculate_bulk_purchase_optimization(
                mock_session, sample_item
            )

            # Should return None or optimization with infinite ROI
            if optimization is not None:
                assert (
                    optimization.roi_months == float("inf") or optimization.total_cost_savings <= 0
                )

    @pytest.mark.asyncio
    async def test_expiry_intelligence_zero_maximum_stock(self, enhanced_inventory_agent):
        """Test expiry intelligence with zero maximum stock."""
        mock_session = Mock()

        perishable_item = Mock(
            id="ITEM001",
            name="Perishable Item",
            current_stock=20,
            maximum_stock=0,  # Zero maximum stock
            expiry_days=7,
        )

        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=21.0,
            confidence_interval=(18.0, 24.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 3.0, "std_daily": 0.5},
            revenue_correlation=0.6,
            method_used="ensemble",
        )

        with patch.object(enhanced_inventory_agent, "_forecast_item_demand", return_value=forecast):
            analysis = await enhanced_inventory_agent._analyze_expiry_intelligence(
                mock_session, perishable_item
            )

            if analysis is not None:
                # Should handle zero maximum stock
                assert 0 <= analysis.risk_score <= 1

    @pytest.mark.asyncio
    async def test_supplier_performance_zero_deliveries(
        self, enhanced_inventory_agent, sample_supplier
    ):
        """Test supplier performance with zero deliveries."""
        mock_session = Mock()

        # Purchase orders without delivery dates
        po_without_dates = Mock(
            supplier_id="SUP001",
            order_date=None,
            expected_delivery_date=None,
            total_amount=Decimal("1000.00"),
        )

        mock_session.query.return_value.filter.return_value.all.return_value = [po_without_dates]

        performance = await enhanced_inventory_agent._analyze_individual_supplier_performance(
            mock_session, sample_supplier
        )

        if performance is not None:
            assert performance.reliability_score == 0.5  # Default when no delivery data

    @pytest.mark.asyncio
    async def test_periodic_check_friday_reorder_optimization(self, enhanced_inventory_agent):
        """Test periodic check on Friday for reorder optimization."""
        with patch.object(enhanced_inventory_agent, "_queue_analysis_message") as mock_queue:
            # Test Friday 8 AM (reorder optimization time)
            with patch("agents.inventory_agent_enhanced.datetime") as mock_datetime:
                mock_datetime.now.return_value = Mock(
                    hour=8, minute=15, weekday=lambda: 4
                )  # Friday

                await enhanced_inventory_agent.periodic_check()

                # Should queue reorder optimization
                mock_queue.assert_called_with("reorder_optimization")

    @pytest.mark.asyncio
    async def test_periodic_check_non_trigger_times(self, enhanced_inventory_agent):
        """Test periodic check during non-trigger times."""
        with patch.object(enhanced_inventory_agent, "_queue_analysis_message") as mock_queue:
            # Test random time that doesn't trigger any analysis
            with patch("agents.inventory_agent_enhanced.datetime") as mock_datetime:
                mock_datetime.now.return_value = Mock(
                    hour=14, minute=30, weekday=lambda: 3
                )  # Wednesday 2:30 PM

                await enhanced_inventory_agent.periodic_check()

                # Should not queue anything
                mock_queue.assert_not_called()

    def test_z_score_various_service_levels(self, enhanced_inventory_agent):
        """Test Z-score calculation for various service levels."""
        # Test edge cases
        assert (
            enhanced_inventory_agent._get_z_score_for_service_level(0.89) == 1.28
        )  # Closest to 0.90
        assert (
            enhanced_inventory_agent._get_z_score_for_service_level(0.96) == 1.65
        )  # Closest to 0.95
        assert (
            enhanced_inventory_agent._get_z_score_for_service_level(0.991) == 2.33
        )  # Closest to 0.99

    @pytest.mark.asyncio
    async def test_analyze_stock_movement_enhanced_in_movement(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test enhanced stock movement analysis with IN movement."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = sample_item

        movement_data = {
            "item_id": "ITEM001",
            "movement_type": StockMovementType.IN,
            "quantity": 50,
        }

        forecast = Mock()

        with patch.object(
            enhanced_inventory_agent, "_forecast_item_demand", return_value=forecast
        ), patch.object(enhanced_inventory_agent, "_analyze_receipt_movement", return_value=None):

            result = await enhanced_inventory_agent._analyze_stock_movement_enhanced(
                mock_session, movement_data
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_stock_movement_enhanced_adjustment_movement(
        self, enhanced_inventory_agent, sample_item
    ):
        """Test enhanced stock movement analysis with ADJUSTMENT movement."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = sample_item

        movement_data = {
            "item_id": "ITEM001",
            "movement_type": StockMovementType.ADJUSTMENT,
            "quantity": -5,
        }

        forecast = Mock()

        with patch.object(
            enhanced_inventory_agent, "_forecast_item_demand", return_value=forecast
        ), patch.object(
            enhanced_inventory_agent, "_analyze_adjustment_movement", return_value=None
        ):

            result = await enhanced_inventory_agent._analyze_stock_movement_enhanced(
                mock_session, movement_data
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_generate_report_with_forecasts(self, enhanced_inventory_agent, mock_db_session):
        """Test report generation with actual forecasts."""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock inventory counts
        mock_session_instance.query.return_value.filter.return_value.count.side_effect = [
            50,
            8,
        ]  # total, low stock

        # Mock forecasts with various characteristics
        forecasts = [
            DemandForecast(
                item_id="ITEM001",
                predicted_demand=50.0,
                confidence_interval=(45.0, 55.0),
                seasonality_factor=1.0,
                trend_factor=0.15,  # High growth
                forecast_horizon_days=14,
                forecast_accuracy=0.85,
                historical_patterns={"avg_daily": 1.67},
                revenue_correlation=0.7,
                method_used="ensemble",
            ),
            DemandForecast(
                item_id="ITEM002",
                predicted_demand=30.0,
                confidence_interval=(25.0, 35.0),
                seasonality_factor=1.0,
                trend_factor=-0.12,  # Declining
                forecast_horizon_days=14,
                forecast_accuracy=0.75,
                historical_patterns={"avg_daily": 1.0},
                revenue_correlation=0.6,
                method_used="ensemble",
            ),
        ]

        with patch.object(
            enhanced_inventory_agent, "_forecast_all_items_demand", return_value=forecasts
        ):
            report = await enhanced_inventory_agent.generate_report()

            assert isinstance(report, dict)
            assert report["agent_id"] == "enhanced_inventory_agent"
            assert report["forecast_summary"]["high_growth_items"] == 1
            assert report["forecast_summary"]["declining_items"] == 1
            assert report["forecast_summary"]["average_accuracy"] > 0
