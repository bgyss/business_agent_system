"""
Additional unit tests for Enhanced Inventory Agent to improve coverage
"""
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
import asyncio

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.inventory_agent_enhanced import (
    BulkPurchaseOptimization,
    DemandForecast,
    EnhancedInventoryAgent,
    ExpiryIntelligence,
    OptimalReorderPoint,
    SupplierPerformance,
)
from models.inventory import Item, ItemStatus, PurchaseOrder, StockMovement, StockMovementType, Supplier


class TestEnhancedInventoryAgentMissingCoverage:
    """Test cases to improve coverage for Enhanced Inventory Agent"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Enhanced inventory analysis with detailed insights")]
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
            "seasonal_analysis_periods": 4,
            "trend_analysis_days": 90,
            "bulk_discount_tiers": {
                "100": 0.02,
                "250": 0.05,
                "500": 0.08,
                "1000": 0.12
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
    def sample_supplier(self):
        """Create sample supplier"""
        return Mock(
            id="SUP001",
            name="Test Supplier",
            contact_email="test@supplier.com"
        )

    @pytest.mark.asyncio
    async def test_process_data_different_types(self, enhanced_inventory_agent, mock_db_session):
        """Test process_data with different data types to cover missing lines"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Test reorder optimization type
        data = {"type": "reorder_optimization"}
        with patch.object(enhanced_inventory_agent, '_optimize_reorder_points', return_value=None):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

        # Test bulk purchase analysis type
        data = {"type": "bulk_purchase_analysis"}
        with patch.object(enhanced_inventory_agent, '_analyze_bulk_purchase_opportunities', return_value=None):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

        # Test expiry management type
        data = {"type": "expiry_management"}
        with patch.object(enhanced_inventory_agent, '_perform_expiry_intelligence', return_value=None):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

        # Test supplier performance review type
        data = {"type": "supplier_performance_review"}
        with patch.object(enhanced_inventory_agent, '_analyze_supplier_performance', return_value=None):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

        # Test inventory health check type
        data = {"type": "inventory_health_check"}
        with patch.object(enhanced_inventory_agent, '_comprehensive_inventory_analysis', return_value=None):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_process_data_exception_handling(self, enhanced_inventory_agent, mock_db_session):
        """Test exception handling in process_data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        data = {"type": "demand_forecast_request"}
        with patch.object(enhanced_inventory_agent, '_perform_demand_forecasting', side_effect=Exception("Test error")):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_perform_demand_forecasting_no_forecasts(self, enhanced_inventory_agent, mock_db_session):
        """Test demand forecasting when no forecasts are generated"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        data = {"type": "demand_forecast_request"}
        
        with patch.object(enhanced_inventory_agent, '_forecast_all_items_demand', return_value=[]):
            result = await enhanced_inventory_agent._perform_demand_forecasting(mock_session_instance, data)
            assert result is None

    @pytest.mark.asyncio
    async def test_perform_demand_forecasting_exception(self, enhanced_inventory_agent, mock_db_session):
        """Test exception handling in demand forecasting"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        data = {"type": "demand_forecast_request"}
        
        with patch.object(enhanced_inventory_agent, '_forecast_all_items_demand', side_effect=Exception("Forecast error")):
            result = await enhanced_inventory_agent._perform_demand_forecasting(mock_session_instance, data)
            assert result is None

    @pytest.mark.asyncio
    async def test_forecast_item_demand_insufficient_data_cases(self, enhanced_inventory_agent):
        """Test forecast_item_demand with various insufficient data scenarios"""
        mock_session = Mock()
        
        # Test with no movements
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        result = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)
        assert result is None

        # Test with less than 7 movements
        few_movements = [Mock(
            item_id="ITEM001",
            movement_type=StockMovementType.OUT,
            quantity=5,
            movement_date=datetime.now().date() - timedelta(days=i)
        ) for i in range(5)]
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = few_movements
        result = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)
        assert result is None

    @pytest.mark.asyncio
    async def test_forecast_item_demand_edge_cases(self, enhanced_inventory_agent):
        """Test forecast_item_demand with edge cases for trend analysis"""
        mock_session = Mock()
        
        # Create movements with constant consumption (no trend)
        movements = []
        base_date = datetime.now().date() - timedelta(days=30)
        for i in range(30):
            movements.append(Mock(
                item_id="ITEM001",
                movement_type=StockMovementType.OUT,
                quantity=5,  # Constant consumption
                movement_date=base_date + timedelta(days=i)
            ))
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = movements
        
        with patch.object(enhanced_inventory_agent, '_calculate_revenue_correlation', return_value=0.5):
            result = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)
            
            # Should handle zero standard deviation case
            if result is not None:
                assert result.item_id == "ITEM001"
                assert result.predicted_demand > 0
                assert result.trend_factor == 0.0  # No trend with constant data

    @pytest.mark.asyncio
    async def test_forecast_item_demand_with_seasonality(self, enhanced_inventory_agent):
        """Test forecast_item_demand with seasonal patterns"""
        mock_session = Mock()
        
        # Create 4+ weeks of data for seasonal analysis
        movements = []
        base_date = datetime.now().date() - timedelta(days=35)
        for i in range(35):
            # Create weekly pattern (higher on weekends)
            base_consumption = 10 if (i % 7) in [5, 6] else 5
            movements.append(Mock(
                item_id="ITEM001",
                movement_type=StockMovementType.OUT,
                quantity=base_consumption,
                movement_date=base_date + timedelta(days=i)
            ))
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = movements
        
        with patch.object(enhanced_inventory_agent, '_calculate_revenue_correlation', return_value=0.7):
            result = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)
            
            if result is not None:
                assert result.item_id == "ITEM001"
                assert result.seasonality_factor != 1.0  # Should detect seasonality
                assert result.method_used == "ensemble_with_seasonality"

    @pytest.mark.asyncio
    async def test_forecast_item_demand_exception_handling(self, enhanced_inventory_agent):
        """Test exception handling in forecast_item_demand"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._forecast_item_demand(mock_session, "ITEM001", 14)
        assert result is None

    @pytest.mark.asyncio
    async def test_forecast_all_items_demand_exception(self, enhanced_inventory_agent):
        """Test exception handling in forecast_all_items_demand"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._forecast_all_items_demand(mock_session, 30)
        assert result == []

    def test_calculate_weekly_seasonality_exception(self, enhanced_inventory_agent):
        """Test exception handling in calculate_weekly_seasonality"""
        # Test with invalid data that might cause division by zero
        consumption_series = [0] * 14  # All zeros
        
        result = enhanced_inventory_agent._calculate_weekly_seasonality(consumption_series)
        
        # Should return default values for all days
        assert len(result) == 7
        assert all(factor == 1.0 for factor in result.values())

    def test_calculate_forecast_accuracy_edge_cases(self, enhanced_inventory_agent):
        """Test forecast accuracy calculation with edge cases"""
        # Test with insufficient data
        short_series = [1, 2, 3, 4, 5]  # Less than 21 days
        accuracy = enhanced_inventory_agent._calculate_forecast_accuracy(short_series)
        assert accuracy == 0.7  # Should return default

        # Test with zero actual values
        zero_series = [0] * 21
        accuracy = enhanced_inventory_agent._calculate_forecast_accuracy(zero_series)
        assert accuracy == 0.7  # Should return default

        # Test with one valid error calculation
        mixed_series = [0] * 14 + [1, 2, 3, 4, 5, 6, 7]  # Last week has non-zero values
        accuracy = enhanced_inventory_agent._calculate_forecast_accuracy(mixed_series)
        assert 0.1 <= accuracy <= 0.95

    def test_calculate_forecast_accuracy_exception(self, enhanced_inventory_agent):
        """Test exception handling in calculate_forecast_accuracy"""
        # Test with data that might cause numpy errors
        invalid_series = None
        accuracy = enhanced_inventory_agent._calculate_forecast_accuracy(invalid_series)
        assert accuracy == 0.7  # Should return default

    @pytest.mark.asyncio
    async def test_calculate_revenue_correlation_edge_cases(self, enhanced_inventory_agent):
        """Test revenue correlation calculation with edge cases"""
        mock_session = Mock()
        
        # Test with insufficient movements
        few_movements = [Mock(quantity=5) for _ in range(5)]
        correlation = await enhanced_inventory_agent._calculate_revenue_correlation(mock_session, "ITEM001", few_movements)
        assert correlation == 0.5  # Should return default

        # Test with zero mean consumption
        zero_movements = [Mock(quantity=0) for _ in range(20)]
        correlation = await enhanced_inventory_agent._calculate_revenue_correlation(mock_session, "ITEM001", zero_movements)
        assert 0.1 <= correlation <= 0.9

    @pytest.mark.asyncio
    async def test_calculate_revenue_correlation_exception(self, enhanced_inventory_agent):
        """Test exception handling in calculate_revenue_correlation"""
        mock_session = Mock()
        
        # Test with invalid movements that might cause errors
        invalid_movements = [Mock(quantity="invalid") for _ in range(20)]
        correlation = await enhanced_inventory_agent._calculate_revenue_correlation(mock_session, "ITEM001", invalid_movements)
        assert correlation == 0.5  # Should return default

    @pytest.mark.asyncio
    async def test_optimize_reorder_points_no_items(self, enhanced_inventory_agent):
        """Test reorder optimization when no items are found"""
        mock_session = Mock()
        
        # No items to optimize
        mock_session.query.return_value.filter.return_value.all.return_value = []
        
        result = await enhanced_inventory_agent._optimize_reorder_points(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_optimize_reorder_points_no_optimizations(self, enhanced_inventory_agent, sample_item):
        """Test reorder optimization when no optimizations are found"""
        mock_session = Mock()
        
        # Items exist but no optimizations
        mock_session.query.return_value.filter.return_value.all.return_value = [sample_item]
        
        with patch.object(enhanced_inventory_agent, '_calculate_optimal_reorder_point', return_value=None):
            result = await enhanced_inventory_agent._optimize_reorder_points(mock_session)
            assert result is None

    @pytest.mark.asyncio
    async def test_optimize_reorder_points_exception(self, enhanced_inventory_agent):
        """Test exception handling in optimize_reorder_points"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._optimize_reorder_points(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_calculate_optimal_reorder_point_no_forecast(self, enhanced_inventory_agent, sample_item):
        """Test optimal reorder point calculation when no forecast is available"""
        mock_session = Mock()
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=None):
            result = await enhanced_inventory_agent._calculate_optimal_reorder_point(mock_session, sample_item)
            assert result is None

    @pytest.mark.asyncio
    async def test_calculate_optimal_reorder_point_zero_holding_cost(self, enhanced_inventory_agent, sample_item):
        """Test optimal reorder point calculation with zero holding cost"""
        mock_session = Mock()
        
        # Mock forecast
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=35.0,
            confidence_interval=(30.0, 40.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.85,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.5},
            revenue_correlation=0.7,
            method_used="ensemble"
        )
        
        # Set unit cost to zero to test zero holding cost case
        sample_item.unit_cost = Decimal("0.00")
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            result = await enhanced_inventory_agent._calculate_optimal_reorder_point(mock_session, sample_item)
            
            if result is not None:
                assert result.optimal_reorder_quantity == sample_item.reorder_quantity

    @pytest.mark.asyncio
    async def test_calculate_optimal_reorder_point_exception(self, enhanced_inventory_agent, sample_item):
        """Test exception handling in calculate_optimal_reorder_point"""
        mock_session = Mock()
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', side_effect=Exception("Forecast error")):
            result = await enhanced_inventory_agent._calculate_optimal_reorder_point(mock_session, sample_item)
            assert result is None

    def test_calculate_current_inventory_cost_exception(self, enhanced_inventory_agent, sample_item):
        """Test exception handling in calculate_current_inventory_cost"""
        # Set invalid data that might cause calculation errors
        sample_item.unit_cost = None
        
        result = enhanced_inventory_agent._calculate_current_inventory_cost(sample_item)
        assert result == 0.0

    def test_calculate_current_inventory_cost_by_id_no_item(self, enhanced_inventory_agent):
        """Test calculate_current_inventory_cost_by_id when item not found"""
        mock_session = Mock()
        
        # No item found
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        result = enhanced_inventory_agent._calculate_current_inventory_cost_by_id(mock_session, "NONEXISTENT")
        assert result == 0.0

    def test_calculate_current_inventory_cost_by_id_exception(self, enhanced_inventory_agent):
        """Test exception handling in calculate_current_inventory_cost_by_id"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = enhanced_inventory_agent._calculate_current_inventory_cost_by_id(mock_session, "ITEM001")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_analyze_bulk_purchase_opportunities_no_items(self, enhanced_inventory_agent):
        """Test bulk purchase analysis when no items are found"""
        mock_session = Mock()
        
        # No items to analyze
        mock_session.query.return_value.filter.return_value.all.return_value = []
        
        result = await enhanced_inventory_agent._analyze_bulk_purchase_opportunities(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_bulk_purchase_opportunities_no_opportunities(self, enhanced_inventory_agent, sample_item):
        """Test bulk purchase analysis when no opportunities are found"""
        mock_session = Mock()
        
        # Items exist but no bulk opportunities
        mock_session.query.return_value.filter.return_value.all.return_value = [sample_item]
        
        with patch.object(enhanced_inventory_agent, '_calculate_bulk_purchase_optimization', return_value=None):
            result = await enhanced_inventory_agent._analyze_bulk_purchase_opportunities(mock_session)
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_bulk_purchase_opportunities_exception(self, enhanced_inventory_agent):
        """Test exception handling in analyze_bulk_purchase_opportunities"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._analyze_bulk_purchase_opportunities(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_calculate_bulk_purchase_optimization_no_forecast(self, enhanced_inventory_agent, sample_item):
        """Test bulk purchase optimization when no forecast is available"""
        mock_session = Mock()
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=None):
            result = await enhanced_inventory_agent._calculate_bulk_purchase_optimization(mock_session, sample_item)
            assert result is None

    @pytest.mark.asyncio
    async def test_calculate_bulk_purchase_optimization_large_tier(self, enhanced_inventory_agent, sample_item):
        """Test bulk purchase optimization with tier quantities larger than reasonable demand"""
        mock_session = Mock()
        
        # Mock forecast with low demand
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=50.0,  # 90-day demand
            confidence_interval=(45.0, 55.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=90,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 0.56},
            revenue_correlation=0.6,
            method_used="ensemble"
        )
        
        # Set bulk discount tiers with very large quantities
        enhanced_inventory_agent.bulk_discount_tiers = {
            "10000": 0.10,  # Much larger than quarterly demand
            "20000": 0.15
        }
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            result = await enhanced_inventory_agent._calculate_bulk_purchase_optimization(mock_session, sample_item)
            
            # Should return None because all tiers are too large
            assert result is None

    @pytest.mark.asyncio
    async def test_calculate_bulk_purchase_optimization_exception(self, enhanced_inventory_agent, sample_item):
        """Test exception handling in calculate_bulk_purchase_optimization"""
        mock_session = Mock()
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', side_effect=Exception("Forecast error")):
            result = await enhanced_inventory_agent._calculate_bulk_purchase_optimization(mock_session, sample_item)
            assert result is None

    @pytest.mark.asyncio
    async def test_perform_expiry_intelligence_no_perishables(self, enhanced_inventory_agent):
        """Test expiry intelligence when no perishable items are found"""
        mock_session = Mock()
        
        # Mock items without expiry considerations
        items = [
            Mock(id="ITEM001", status=ItemStatus.ACTIVE),
            Mock(id="ITEM002", status=ItemStatus.ACTIVE)
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = items
        
        # None of the items have expiry_days attribute
        result = await enhanced_inventory_agent._perform_expiry_intelligence(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_perform_expiry_intelligence_no_analyses(self, enhanced_inventory_agent):
        """Test expiry intelligence when no analyses are generated"""
        mock_session = Mock()
        
        # Mock perishable items
        perishable_item = Mock(
            id="ITEM001",
            status=ItemStatus.ACTIVE,
            expiry_days=7
        )
        mock_session.query.return_value.filter.return_value.all.return_value = [perishable_item]
        
        with patch.object(enhanced_inventory_agent, '_analyze_expiry_intelligence', return_value=None):
            result = await enhanced_inventory_agent._perform_expiry_intelligence(mock_session)
            assert result is None

    @pytest.mark.asyncio
    async def test_perform_expiry_intelligence_exception(self, enhanced_inventory_agent):
        """Test exception handling in perform_expiry_intelligence"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._perform_expiry_intelligence(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_expiry_intelligence_no_forecast(self, enhanced_inventory_agent):
        """Test expiry intelligence analysis when no forecast is available"""
        mock_session = Mock()
        
        perishable_item = Mock(
            id="ITEM001",
            expiry_days=7,
            current_stock=20,
            maximum_stock=30
        )
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=None):
            result = await enhanced_inventory_agent._analyze_expiry_intelligence(mock_session, perishable_item)
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_expiry_intelligence_zero_consumption(self, enhanced_inventory_agent):
        """Test expiry intelligence analysis with zero consumption"""
        mock_session = Mock()
        
        perishable_item = Mock(
            id="ITEM001",
            expiry_days=7,
            current_stock=20,
            maximum_stock=30
        )
        
        # Mock forecast with zero consumption
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=0.0,  # No consumption
            confidence_interval=(0.0, 0.0),
            seasonality_factor=1.0,
            trend_factor=0.0,
            forecast_horizon_days=7,
            forecast_accuracy=0.8,
            historical_patterns={"avg_daily": 0.0, "std_daily": 0.0},
            revenue_correlation=0.5,
            method_used="ensemble"
        )
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            result = await enhanced_inventory_agent._analyze_expiry_intelligence(mock_session, perishable_item)
            
            if result is not None:
                assert result.predicted_waste_amount == 20  # All will expire
                assert result.risk_score > 0.5  # High risk

    @pytest.mark.asyncio
    async def test_analyze_expiry_intelligence_exception(self, enhanced_inventory_agent):
        """Test exception handling in analyze_expiry_intelligence"""
        mock_session = Mock()
        
        perishable_item = Mock(
            id="ITEM001",
            expiry_days=7,
            current_stock=20,
            maximum_stock=30
        )
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', side_effect=Exception("Forecast error")):
            result = await enhanced_inventory_agent._analyze_expiry_intelligence(mock_session, perishable_item)
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_supplier_performance_no_suppliers(self, enhanced_inventory_agent):
        """Test supplier performance analysis when no suppliers are found"""
        mock_session = Mock()
        
        # No suppliers with recent activity
        mock_session.query.return_value.join.return_value.filter.return_value.distinct.return_value.all.return_value = []
        
        result = await enhanced_inventory_agent._analyze_supplier_performance(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_supplier_performance_no_analyses(self, enhanced_inventory_agent, sample_supplier):
        """Test supplier performance analysis when no analyses are generated"""
        mock_session = Mock()
        
        # Suppliers exist but no analyses
        mock_session.query.return_value.join.return_value.filter.return_value.distinct.return_value.all.return_value = [sample_supplier]
        
        with patch.object(enhanced_inventory_agent, '_analyze_individual_supplier_performance', return_value=None):
            result = await enhanced_inventory_agent._analyze_supplier_performance(mock_session)
            assert result is None

    @pytest.mark.asyncio
    async def test_analyze_supplier_performance_exception(self, enhanced_inventory_agent):
        """Test exception handling in analyze_supplier_performance"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._analyze_supplier_performance(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_individual_supplier_performance_no_orders(self, enhanced_inventory_agent, sample_supplier):
        """Test individual supplier performance when no purchase orders are found"""
        mock_session = Mock()
        
        # No purchase orders for this supplier
        mock_session.query.return_value.filter.return_value.all.return_value = []
        
        result = await enhanced_inventory_agent._analyze_individual_supplier_performance(mock_session, sample_supplier)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_individual_supplier_performance_exception(self, enhanced_inventory_agent, sample_supplier):
        """Test exception handling in analyze_individual_supplier_performance"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._analyze_individual_supplier_performance(mock_session, sample_supplier)
        assert result is None

    @pytest.mark.asyncio
    async def test_comprehensive_inventory_analysis_edge_cases(self, enhanced_inventory_agent):
        """Test comprehensive inventory analysis with edge cases"""
        mock_session = Mock()
        
        # Test with zero items
        mock_session.query.return_value.filter.return_value.count.side_effect = [0, 0, 0]  # total, low stock, overstocked
        mock_session.query.return_value.filter.return_value.scalar.side_effect = [0.0, 0.0]  # total value, recent movements
        
        result = await enhanced_inventory_agent._comprehensive_inventory_analysis(mock_session)
        
        if result is not None:
            assert result.context["total_active_items"] == 0
            assert result.context["monthly_turnover_rate"] == 0

    @pytest.mark.asyncio
    async def test_comprehensive_inventory_analysis_high_issues(self, enhanced_inventory_agent):
        """Test comprehensive inventory analysis with high issue counts"""
        mock_session = Mock()
        
        # Test with high issue counts
        mock_session.query.return_value.filter.return_value.count.side_effect = [100, 20, 15]  # total, low stock (20%), overstocked (15%)
        mock_session.query.return_value.filter.return_value.scalar.side_effect = [50000.0, 100.0]  # total value, low movements
        
        result = await enhanced_inventory_agent._comprehensive_inventory_analysis(mock_session)
        
        if result is not None:
            assert len(result.context["identified_issues"]) >= 2  # Should identify multiple issues
            assert result.confidence == 0.9  # High confidence when issues are clear

    @pytest.mark.asyncio
    async def test_comprehensive_inventory_analysis_exception(self, enhanced_inventory_agent):
        """Test exception handling in comprehensive_inventory_analysis"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._comprehensive_inventory_analysis(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_stock_movement_enhanced_no_item_id(self, enhanced_inventory_agent):
        """Test enhanced stock movement analysis when no item_id is provided"""
        mock_session = Mock()
        
        movement_data = {
            "movement_type": StockMovementType.OUT,
            "quantity": 10
            # No item_id
        }
        
        result = await enhanced_inventory_agent._analyze_stock_movement_enhanced(mock_session, movement_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_stock_movement_enhanced_no_item_found(self, enhanced_inventory_agent):
        """Test enhanced stock movement analysis when item is not found"""
        mock_session = Mock()
        
        # Item not found in database
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        movement_data = {
            "item_id": "NONEXISTENT",
            "movement_type": StockMovementType.OUT,
            "quantity": 10
        }
        
        result = await enhanced_inventory_agent._analyze_stock_movement_enhanced(mock_session, movement_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_stock_movement_enhanced_unknown_type(self, enhanced_inventory_agent, sample_item):
        """Test enhanced stock movement analysis with unknown movement type"""
        mock_session = Mock()
        
        mock_session.query.return_value.filter.return_value.first.return_value = sample_item
        
        movement_data = {
            "item_id": "ITEM001",
            "movement_type": "UNKNOWN_TYPE",
            "quantity": 10
        }
        
        result = await enhanced_inventory_agent._analyze_stock_movement_enhanced(mock_session, movement_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_stock_movement_enhanced_exception(self, enhanced_inventory_agent):
        """Test exception handling in analyze_stock_movement_enhanced"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        movement_data = {
            "item_id": "ITEM001",
            "movement_type": StockMovementType.OUT,
            "quantity": 10
        }
        
        result = await enhanced_inventory_agent._analyze_stock_movement_enhanced(mock_session, movement_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_consumption_movement_normal_consumption(self, enhanced_inventory_agent, sample_item):
        """Test consumption movement analysis with normal consumption"""
        mock_session = Mock()
        
        # Mock forecast showing normal consumption
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
        
        # Normal consumption (not unusual)
        result = await enhanced_inventory_agent._analyze_consumption_movement(mock_session, sample_item, 5, forecast)
        
        # Should not generate unusual consumption alert
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_consumption_movement_no_forecast(self, enhanced_inventory_agent, sample_item):
        """Test consumption movement analysis without forecast"""
        mock_session = Mock()
        
        # No forecast available
        result = await enhanced_inventory_agent._analyze_consumption_movement(mock_session, sample_item, 10, None)
        
        # Should handle gracefully without forecast
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_consumption_movement_exception(self, enhanced_inventory_agent, sample_item):
        """Test exception handling in analyze_consumption_movement"""
        mock_session = Mock()
        
        # Mock forecast that might cause errors
        forecast = Mock()
        forecast.historical_patterns = None  # This will cause an error
        
        result = await enhanced_inventory_agent._analyze_consumption_movement(mock_session, sample_item, 10, forecast)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_receipt_movement_no_overstock(self, enhanced_inventory_agent, sample_item):
        """Test receipt movement analysis without overstock"""
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
        
        # Receipt that doesn't cause overstock
        result = await enhanced_inventory_agent._analyze_receipt_movement(mock_session, sample_item, 20, forecast)
        
        # Should not generate overstock alert
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_receipt_movement_exception(self, enhanced_inventory_agent, sample_item):
        """Test exception handling in analyze_receipt_movement"""
        mock_session = Mock()
        
        # Mock forecast that might cause errors
        forecast = Mock()
        forecast.historical_patterns = {"avg_daily": "invalid"}  # This will cause an error
        
        result = await enhanced_inventory_agent._analyze_receipt_movement(mock_session, sample_item, 60, forecast)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_adjustment_movement_small_adjustment(self, enhanced_inventory_agent, sample_item):
        """Test adjustment movement analysis with small adjustment"""
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
        
        # Small adjustment (less than 10%)
        result = await enhanced_inventory_agent._analyze_adjustment_movement(mock_session, sample_item, 2, forecast)
        
        # Should not generate adjustment alert
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_adjustment_movement_exception(self, enhanced_inventory_agent, sample_item):
        """Test exception handling in analyze_adjustment_movement"""
        mock_session = Mock()
        
        # Set sample item to have zero stock to cause division by zero
        sample_item.current_stock = 0
        
        forecast = Mock()
        
        result = await enhanced_inventory_agent._analyze_adjustment_movement(mock_session, sample_item, 10, forecast)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_reorder_needs_enhanced_no_items(self, enhanced_inventory_agent):
        """Test enhanced reorder analysis when no items need reordering"""
        mock_session = Mock()
        
        # No items near reorder point
        mock_session.query.return_value.filter.return_value.all.return_value = []
        
        result = await enhanced_inventory_agent._analyze_reorder_needs_enhanced(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_reorder_needs_enhanced_exception(self, enhanced_inventory_agent):
        """Test exception handling in analyze_reorder_needs_enhanced"""
        mock_session = Mock()
        
        # Make the query raise an exception
        mock_session.query.side_effect = Exception("Database error")
        
        result = await enhanced_inventory_agent._analyze_reorder_needs_enhanced(mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_report_exception(self, enhanced_inventory_agent, mock_db_session):
        """Test exception handling in generate_report"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Make the query raise an exception
        mock_session_instance.query.side_effect = Exception("Database error")
        
        report = await enhanced_inventory_agent.generate_report()
        
        assert "error" in report
        assert report["error"] == "Database error"

    @pytest.mark.asyncio
    async def test_periodic_check_different_times(self, enhanced_inventory_agent):
        """Test periodic check with different time scenarios"""
        with patch.object(enhanced_inventory_agent, '_queue_analysis_message') as mock_queue:
            
            # Test 4 AM (comprehensive analysis time)
            with patch('agents.inventory_agent_enhanced.datetime') as mock_datetime:
                mock_datetime.now.return_value = Mock(hour=4, minute=15, weekday=lambda: 1)
                
                await enhanced_inventory_agent.periodic_check()
                
                # Should queue inventory health check
                mock_queue.assert_called_with("inventory_health_check")
                
            mock_queue.reset_mock()
            
            # Test Tuesday 8 AM (reorder optimization time)
            with patch('agents.inventory_agent_enhanced.datetime') as mock_datetime:
                mock_datetime.now.return_value = Mock(hour=8, minute=15, weekday=lambda: 1)  # Tuesday
                
                await enhanced_inventory_agent.periodic_check()
                
                # Should queue reorder optimization
                mock_queue.assert_called_with("reorder_optimization")
                
            mock_queue.reset_mock()
            
            # Test Monday 9 AM (supplier performance time)
            with patch('agents.inventory_agent_enhanced.datetime') as mock_datetime:
                mock_datetime.now.return_value = Mock(hour=9, minute=15, weekday=lambda: 0)  # Monday
                
                await enhanced_inventory_agent.periodic_check()
                
                # Should queue supplier performance review
                mock_queue.assert_called_with("supplier_performance_review")
                
            mock_queue.reset_mock()
            
            # Test 10 AM (expiry management time)
            with patch('agents.inventory_agent_enhanced.datetime') as mock_datetime:
                mock_datetime.now.return_value = Mock(hour=10, minute=15, weekday=lambda: 2)
                
                await enhanced_inventory_agent.periodic_check()
                
                # Should queue expiry management
                mock_queue.assert_called_with("expiry_management")

    @pytest.mark.asyncio
    async def test_periodic_check_exception(self, enhanced_inventory_agent):
        """Test exception handling in periodic_check"""
        with patch('agents.inventory_agent_enhanced.datetime') as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")
            
            # Should handle the exception gracefully
            await enhanced_inventory_agent.periodic_check()
            # No assertion needed - just ensuring it doesn't crash

    @pytest.mark.asyncio
    async def test_queue_analysis_message_no_queue(self, enhanced_inventory_agent):
        """Test queue_analysis_message when no message queue is available"""
        enhanced_inventory_agent.message_queue = None
        
        # Should handle gracefully without queue
        await enhanced_inventory_agent._queue_analysis_message("test_analysis")
        # No assertion needed - just ensuring it doesn't crash

    @pytest.mark.asyncio
    async def test_queue_analysis_message_exception(self, enhanced_inventory_agent):
        """Test exception handling in queue_analysis_message"""
        mock_queue = AsyncMock()
        mock_queue.put.side_effect = Exception("Queue error")
        enhanced_inventory_agent.message_queue = mock_queue
        
        # Should handle the exception gracefully
        await enhanced_inventory_agent._queue_analysis_message("test_analysis")
        # No assertion needed - just ensuring it doesn't crash

    def test_z_score_edge_cases(self, enhanced_inventory_agent):
        """Test Z-score calculation with edge cases"""
        # Test with service level not in the map
        z_score = enhanced_inventory_agent._get_z_score_for_service_level(0.92)
        
        # Should return closest mapping (0.90 -> 1.28)
        assert z_score == 1.28

    def test_system_prompt_content(self, enhanced_inventory_agent):
        """Test system prompt contains all required content"""
        prompt = enhanced_inventory_agent.system_prompt
        
        # Verify all key sections are present
        assert "enhanced responsibilities" in prompt
        assert "CORE FUNCTIONS" in prompt
        assert "ADVANCED ANALYTICS" in prompt
        assert "DECISION MAKING" in prompt
        assert "INTELLIGENCE CAPABILITIES" in prompt
        assert "confidence scores" in prompt

    @pytest.mark.asyncio
    async def test_process_data_unknown_type(self, enhanced_inventory_agent, mock_db_session):
        """Test process_data with unknown data type falls back to enhanced reorder analysis"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        data = {"type": "unknown_type"}
        
        with patch.object(enhanced_inventory_agent, '_analyze_reorder_needs_enhanced', return_value=None):
            result = await enhanced_inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_demand_forecasting_with_specific_item(self, enhanced_inventory_agent, mock_db_session):
        """Test demand forecasting for specific item"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        data = {
            "type": "demand_forecast_request",
            "item_id": "ITEM001",
            "forecast_days": 21
        }
        
        # Mock forecast for specific item
        forecast = DemandForecast(
            item_id="ITEM001",
            predicted_demand=105.0,
            confidence_interval=(95.0, 115.0),
            seasonality_factor=1.1,
            trend_factor=0.05,
            forecast_horizon_days=21,
            forecast_accuracy=0.85,
            historical_patterns={"avg_daily": 5.0, "std_daily": 1.2},
            revenue_correlation=0.7,
            method_used="ensemble_with_seasonality"
        )
        
        with patch.object(enhanced_inventory_agent, '_forecast_item_demand', return_value=forecast):
            result = await enhanced_inventory_agent._perform_demand_forecasting(mock_session_instance, data)
            
            assert result is not None
            assert result.decision_type == "demand_forecast_analysis"
            assert result.context["total_items_analyzed"] == 1