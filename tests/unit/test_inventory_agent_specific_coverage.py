"""
Specific coverage tests for InventoryAgent targeting actual methods
This file focuses on testing methods that actually exist in the inventory agent
"""
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import AgentDecision
from agents.inventory_agent import (
    InventoryAgent, 
    DemandForecast,
    OptimalReorderPoint,
    BulkPurchaseOptimization,
    SeasonalityAnalysis,
    ItemCorrelationAnalysis,
    SupplierDiversificationAnalysis,
    SupplierPerformanceMetrics
)
from models.inventory import ItemStatus, StockMovementType


class TestInventoryAgentSpecificCoverage:
    """Test specific methods that exist in InventoryAgent"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Analysis completed")]
            mock_client.return_value.messages.create.return_value = mock_response
            yield mock_client

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        query_mock = Mock()
        filter_mock = Mock()
        session.query.return_value = query_mock
        query_mock.filter.return_value = filter_mock
        filter_mock.all.return_value = []
        filter_mock.first.return_value = None
        filter_mock.count.return_value = 0
        filter_mock.order_by.return_value = filter_mock
        filter_mock.limit.return_value = filter_mock
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        return session

    @pytest.fixture
    def inventory_agent(self, mock_anthropic, mock_db_session):
        """Create inventory agent"""
        with patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker') as mock_sessionmaker:
            mock_sessionmaker.return_value = lambda: mock_db_session
            
            config = {
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
            agent = InventoryAgent("test_inventory", "test_key", config, "sqlite:///:memory:")
            agent.SessionLocal = lambda: mock_db_session
            return agent

    # Test the Z-score calculation method that actually exists
    def test_get_z_score_for_service_level(self, inventory_agent):
        """Test Z-score calculation for service levels"""
        # Test exact matches
        assert inventory_agent._get_z_score_for_service_level(0.50) == 0.00
        assert inventory_agent._get_z_score_for_service_level(0.95) == 1.65
        assert inventory_agent._get_z_score_for_service_level(0.99) == 2.33
        
        # Test interpolation between values
        z_score = inventory_agent._get_z_score_for_service_level(0.92)
        assert isinstance(z_score, float)
        assert 1.28 < z_score < 1.65  # Between 90% and 95%
        
        # Test extreme values
        assert inventory_agent._get_z_score_for_service_level(1.0) == 1.65  # Default for 100%
        assert inventory_agent._get_z_score_for_service_level(0.999) == 3.09
        assert inventory_agent._get_z_score_for_service_level(0.40) == 1.65  # Default for low values

    # Test process_data branches that actually exist
    @pytest.mark.asyncio
    async def test_process_data_stock_movement(self, inventory_agent, mock_db_session):
        """Test stock movement processing"""
        data = {
            "type": "stock_movement",
            "movement": {
                "item_id": "test-item",
                "quantity": 10,
                "movement_type": "in"
            }
        }
        
        # Mock the _analyze_stock_movement method
        with patch.object(inventory_agent, '_analyze_stock_movement', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_process_data_daily_inventory_check(self, inventory_agent, mock_db_session):
        """Test daily inventory check processing"""
        data = {"type": "daily_inventory_check"}
        
        # Mock the _perform_daily_inventory_check method
        with patch.object(inventory_agent, '_perform_daily_inventory_check', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_process_data_reorder_analysis(self, inventory_agent, mock_db_session):
        """Test reorder analysis processing"""
        data = {"type": "reorder_analysis"}
        
        # Mock the _analyze_reorder_needs method
        with patch.object(inventory_agent, '_analyze_reorder_needs', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_process_data_expiry_check(self, inventory_agent, mock_db_session):
        """Test expiry check processing"""
        data = {"type": "expiry_check"}
        
        # Mock the _check_expiring_items method
        with patch.object(inventory_agent, '_check_expiring_items', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_process_data_supplier_performance(self, inventory_agent, mock_db_session):
        """Test supplier performance processing"""
        data = {"type": "supplier_performance"}
        
        # Mock the _analyze_supplier_performance method
        with patch.object(inventory_agent, '_analyze_supplier_performance', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    # Test seasonality analysis branch with missing item_id
    @pytest.mark.asyncio
    async def test_process_data_seasonality_analysis_no_item_id(self, inventory_agent, mock_db_session):
        """Test seasonality analysis without item_id"""
        data = {"type": "seasonality_analysis"}
        
        result = await inventory_agent.process_data(data)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_data_seasonality_analysis_no_results(self, inventory_agent, mock_db_session):
        """Test seasonality analysis with no seasonality found"""
        data = {
            "type": "seasonality_analysis",
            "item_id": "test-item"
        }
        
        # Mock analyze_seasonality_patterns to return None
        with patch.object(inventory_agent, 'analyze_seasonality_patterns', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    @pytest.mark.asyncio
    async def test_process_data_seasonality_analysis_with_results(self, inventory_agent, mock_db_session):
        """Test seasonality analysis with successful results"""
        data = {
            "type": "seasonality_analysis",
            "item_id": "test-item"
        }
        
        mock_seasonality = SeasonalityAnalysis(
            item_id="test-item",
            seasonal_periods=[7, 30],
            seasonal_strength=0.75,
            peak_periods=[1, 15],
            low_periods=[7, 23],
            seasonal_adjustment_factor=1.2,
            confidence=0.85
        )
        
        with patch.object(inventory_agent, 'analyze_seasonality_patterns', return_value=mock_seasonality), \
             patch.object(inventory_agent, 'analyze_with_claude', return_value="Seasonality analysis"):
            result = await inventory_agent.process_data(data)
            assert result is not None
            assert result.decision_type == "seasonality_analysis"
            assert result.confidence == 0.85

    # Test correlation analysis branches
    @pytest.mark.asyncio
    async def test_process_data_correlation_analysis_no_item_id(self, inventory_agent, mock_db_session):
        """Test correlation analysis without item_id"""
        data = {"type": "correlation_analysis"}
        
        result = await inventory_agent.process_data(data)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_data_correlation_analysis_no_results(self, inventory_agent, mock_db_session):
        """Test correlation analysis with no correlations found"""
        data = {
            "type": "correlation_analysis",
            "item_id": "test-item"
        }
        
        # Mock analyze_item_correlations to return None
        with patch.object(inventory_agent, 'analyze_item_correlations', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    # Test supplier diversification analysis branches
    @pytest.mark.asyncio
    async def test_process_data_supplier_diversification_no_item_id(self, inventory_agent, mock_db_session):
        """Test supplier diversification analysis without item_id"""
        data = {"type": "supplier_diversification_analysis"}
        
        result = await inventory_agent.process_data(data)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_data_supplier_diversification_no_results(self, inventory_agent, mock_db_session):
        """Test supplier diversification analysis with no results"""
        data = {
            "type": "supplier_diversification_analysis",
            "item_id": "test-item"
        }
        
        # Mock analyze_supplier_diversification to return None
        with patch.object(inventory_agent, 'analyze_supplier_diversification', return_value=None):
            result = await inventory_agent.process_data(data)
            assert result is None

    # Test unknown data type
    @pytest.mark.asyncio
    async def test_process_data_unknown_type(self, inventory_agent, mock_db_session):
        """Test processing unknown data type"""
        data = {"type": "unknown_analysis_type"}
        
        result = await inventory_agent.process_data(data)
        assert result is None

    # Test empty data
    @pytest.mark.asyncio
    async def test_process_data_empty_data(self, inventory_agent, mock_db_session):
        """Test processing empty data"""
        result = await inventory_agent.process_data({})
        assert result is None

    # Test the system prompt property
    def test_system_prompt_property(self, inventory_agent):
        """Test system prompt property"""
        prompt = inventory_agent.system_prompt
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "Inventory Management Agent" in prompt
        assert "stock levels" in prompt
        assert "reorder" in prompt

    # Test configuration initialization with all parameters
    def test_full_configuration_initialization(self):
        """Test configuration initialization with all parameters"""
        with patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker'), \
             patch('agents.base_agent.Anthropic'):
            
            config = {
                "low_stock_multiplier": 1.5,
                "reorder_lead_time": 10,
                "consumption_analysis_days": 45,
                "forecast_horizon_days": 60,
                "service_level_target": 0.98,
                "holding_cost_rate": 0.30,
                "order_cost": 75.0,
                "min_forecast_accuracy": 0.80,
                "seasonality_window_days": 400,
                "alpha_smoothing": 0.4,
                "beta_trend": 0.2,
                "gamma_seasonality": 0.3,
            }
            
            agent = InventoryAgent("test", "key", config, "sqlite:///:memory:")
            
            # Verify all config values are set
            assert agent.low_stock_multiplier == 1.5
            assert agent.reorder_lead_time == 10
            assert agent.consumption_analysis_days == 45
            assert agent.forecast_horizon_days == 60
            assert agent.service_level_target == 0.98
            assert agent.holding_cost_rate == 0.30
            assert agent.order_cost == 75.0
            assert agent.min_forecast_accuracy == 0.80
            assert agent.seasonality_window_days == 400
            assert agent.alpha_smoothing == 0.4
            assert agent.beta_trend == 0.2
            assert agent.gamma_seasonality == 0.3

    # Test configuration defaults
    def test_configuration_defaults(self):
        """Test configuration defaults when config is empty"""
        with patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker'), \
             patch('agents.base_agent.Anthropic'):
            
            # Test with empty config
            agent = InventoryAgent("test", "key", {}, "sqlite:///:memory:")
            
            # Verify default values
            assert agent.low_stock_multiplier == 1.2
            assert agent.reorder_lead_time == 7
            assert agent.consumption_analysis_days == 30
            assert agent.forecast_horizon_days == 30
            assert agent.service_level_target == 0.95
            assert agent.holding_cost_rate == 0.25
            assert agent.order_cost == 50.0
            assert agent.min_forecast_accuracy == 0.70
            assert agent.seasonality_window_days == 365
            assert agent.alpha_smoothing == 0.3
            assert agent.beta_trend == 0.1
            assert agent.gamma_seasonality == 0.2

    # Test exception handling in process_data
    @pytest.mark.asyncio
    async def test_process_data_exception_handling(self, inventory_agent, mock_db_session):
        """Test exception handling in process_data"""
        mock_db_session.query.side_effect = Exception("Database error")
        
        # Should handle exceptions gracefully
        result = await inventory_agent.process_data({"type": "daily_inventory_check"})
        assert result is None

    # Test the NamedTuple classes to ensure they're instantiated correctly
    def test_demand_forecast_namedtuple(self):
        """Test DemandForecast NamedTuple creation"""
        forecast = DemandForecast(
            item_id="test-item",
            predicted_demand=25.0,
            confidence_interval=(20.0, 30.0),
            seasonality_factor=1.1,
            trend_factor=1.05,
            forecast_horizon_days=30,
            forecast_accuracy=0.85
        )
        
        assert forecast.item_id == "test-item"
        assert forecast.predicted_demand == 25.0
        assert forecast.forecast_accuracy == 0.85

    def test_optimal_reorder_point_namedtuple(self):
        """Test OptimalReorderPoint NamedTuple creation"""
        reorder = OptimalReorderPoint(
            item_id="test-item",
            optimal_reorder_point=30,
            optimal_reorder_quantity=100,
            service_level=0.95,
            safety_stock=10,
            lead_time_demand=25.0,
            demand_variability=5.0,
            total_cost=500.0
        )
        
        assert reorder.item_id == "test-item"
        assert reorder.optimal_reorder_point == 30
        assert reorder.service_level == 0.95

    def test_bulk_purchase_optimization_namedtuple(self):
        """Test BulkPurchaseOptimization NamedTuple creation"""
        bulk = BulkPurchaseOptimization(
            item_id="test-item",
            optimal_order_quantity=500,
            unit_cost_with_discount=Decimal("4.50"),
            total_cost_savings=Decimal("250.00"),
            break_even_point=100,
            holding_cost_impact=Decimal("50.00"),
            recommended_purchase_timing=datetime.now()
        )
        
        assert bulk.item_id == "test-item"
        assert bulk.optimal_order_quantity == 500
        assert bulk.total_cost_savings == Decimal("250.00")

    def test_seasonality_analysis_namedtuple(self):
        """Test SeasonalityAnalysis NamedTuple creation"""
        seasonality = SeasonalityAnalysis(
            item_id="test-item",
            seasonal_periods=[7, 30],
            seasonal_strength=0.75,
            peak_periods=[1, 15],
            low_periods=[7, 23],
            seasonal_adjustment_factor=1.2,
            confidence=0.85
        )
        
        assert seasonality.item_id == "test-item"
        assert seasonality.seasonal_periods == [7, 30]
        assert seasonality.confidence == 0.85

    def test_item_correlation_analysis_namedtuple(self):
        """Test ItemCorrelationAnalysis NamedTuple creation"""
        correlations = ItemCorrelationAnalysis(
            primary_item_id="test-item",
            correlated_items=[("item2", 0.85), ("item3", 0.72)],
            substitution_items=["item4"],
            complementary_items=["item5"],
            impact_factor=0.8,
            bundle_opportunities=[{"items": ["test-item", "item2"], "discount": 0.1}]
        )
        
        assert correlations.primary_item_id == "test-item"
        assert len(correlations.correlated_items) == 2
        assert correlations.impact_factor == 0.8

    def test_supplier_diversification_analysis_namedtuple(self):
        """Test SupplierDiversificationAnalysis NamedTuple creation"""
        diversification = SupplierDiversificationAnalysis(
            item_id="test-item",
            current_supplier_concentration=0.9,
            alternative_suppliers=["sup2", "sup3"],
            risk_score=0.7,
            diversification_recommendations=[{"action": "Add secondary supplier", "priority": "high"}],
            cost_impact_of_diversification=0.05,
            recommended_supplier_split={"sup1": 0.6, "sup2": 0.4}
        )
        
        assert diversification.item_id == "test-item"
        assert diversification.current_supplier_concentration == 0.9
        assert diversification.risk_score == 0.7

    def test_supplier_performance_metrics_namedtuple(self):
        """Test SupplierPerformanceMetrics NamedTuple creation"""
        metrics = SupplierPerformanceMetrics(
            supplier_id="sup1",
            on_time_delivery_rate=0.95,
            quality_score=0.88,
            cost_competitiveness=0.75,
            reliability_index=0.90,
            lead_time_variability=0.15,
            overall_performance_score=0.85,
            recommended_action="Continue partnership"
        )
        
        assert metrics.supplier_id == "sup1"
        assert metrics.on_time_delivery_rate == 0.95
        assert metrics.overall_performance_score == 0.85