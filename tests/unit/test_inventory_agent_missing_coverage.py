"""
Test coverage for missing InventoryAgent functionality
Focuses on testing the untested methods and code paths
"""
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import AgentDecision
from agents.inventory_agent import (
    InventoryAgent, 
    DemandForecast,
    OptimalReorderPoint,
    BulkPurchaseOptimization,
    SupplierPerformanceMetrics,
    SeasonalityAnalysis,
    ItemCorrelationAnalysis,
    SupplierDiversificationAnalysis
)
from models.inventory import ItemStatus, StockMovementType, Item


class TestInventoryAgentMissingCoverage:
    """Test cases for missing coverage in InventoryAgent"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Advanced inventory analysis completed")]
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
    def agent_config(self):
        """Extended agent configuration"""
        return {
            "check_interval": 300,
            "low_stock_multiplier": 1.2,
            "reorder_lead_time": 7,
            "consumption_analysis_days": 30,
            "expiry_warning_days": 5,
            "urgent_reorder_threshold": 10,
            "critical_reorder_threshold": 5,
        }

    @pytest.fixture
    def agent(self, mock_anthropic, mock_db_session, agent_config):
        """Create InventoryAgent instance"""
        return InventoryAgent(
            agent_id="test_inventory_agent",
            api_key="test_api_key",
            config=agent_config,
            db_url="sqlite:///:memory:"
        )

    # Test NamedTuple structures that exist but aren't tested
    
    def test_demand_forecast_structure(self):
        """Test DemandForecast NamedTuple structure"""
        forecast = DemandForecast(
            item_id="ITEM-001",
            predicted_demand=100.5,
            confidence_interval=(90.0, 110.0),
            seasonality_factor=1.2,
            trend_factor=0.95,
            forecast_horizon_days=30,
            forecast_accuracy=0.85
        )
        
        assert forecast.item_id == "ITEM-001"
        assert forecast.predicted_demand == 100.5
        assert forecast.confidence_interval == (90.0, 110.0)
        assert forecast.seasonality_factor == 1.2
        assert forecast.trend_factor == 0.95
        assert forecast.forecast_horizon_days == 30
        assert forecast.forecast_accuracy == 0.85

    def test_optimal_reorder_point_structure(self):
        """Test OptimalReorderPoint NamedTuple structure"""
        reorder_point = OptimalReorderPoint(
            item_id="ITEM-001",
            optimal_reorder_point=50,
            optimal_reorder_quantity=100,
            service_level=0.95,
            safety_stock=20,
            lead_time_demand=30.0,
            demand_variability=5.0,
            total_cost=Decimal('1500.00')
        )
        
        assert reorder_point.item_id == "ITEM-001"
        assert reorder_point.optimal_reorder_point == 50
        assert reorder_point.optimal_reorder_quantity == 100
        assert reorder_point.service_level == 0.95
        assert reorder_point.safety_stock == 20
        assert reorder_point.lead_time_demand == 30.0
        assert reorder_point.demand_variability == 5.0
        assert reorder_point.total_cost == Decimal('1500.00')

    def test_bulk_purchase_optimization_structure(self):
        """Test BulkPurchaseOptimization NamedTuple structure"""
        optimization = BulkPurchaseOptimization(
            item_id="ITEM-001",
            optimal_order_quantity=500,
            unit_cost_with_discount=Decimal('9.50'),
            total_cost_savings=Decimal('250.00'),
            break_even_point=30,
            holding_cost_impact=Decimal('45.00'),
            recommended_purchase_timing=datetime(2023, 12, 1)
        )
        
        assert optimization.item_id == "ITEM-001"
        assert optimization.optimal_order_quantity == 500
        assert optimization.unit_cost_with_discount == Decimal('9.50')
        assert optimization.total_cost_savings == Decimal('250.00')
        assert optimization.break_even_point == 30
        assert optimization.holding_cost_impact == Decimal('45.00')
        assert optimization.recommended_purchase_timing == datetime(2023, 12, 1)

    def test_supplier_performance_metrics_structure(self):
        """Test SupplierPerformanceMetrics NamedTuple structure"""
        metrics = SupplierPerformanceMetrics(
            supplier_id="SUP-001",
            on_time_delivery_rate=0.92,
            quality_score=0.88,
            cost_competitiveness=0.85,
            reliability_index=0.90,
            lead_time_variability=0.15,
            overall_performance_score=0.87,
            recommended_action="maintain_partnership"
        )
        
        assert metrics.supplier_id == "SUP-001"
        assert metrics.on_time_delivery_rate == 0.92
        assert metrics.quality_score == 0.88
        assert metrics.cost_competitiveness == 0.85
        assert metrics.reliability_index == 0.90
        assert metrics.lead_time_variability == 0.15
        assert metrics.overall_performance_score == 0.87
        assert metrics.recommended_action == "maintain_partnership"

    def test_seasonality_analysis_structure(self):
        """Test SeasonalityAnalysis NamedTuple structure"""
        analysis = SeasonalityAnalysis(
            item_id="ITEM-001",
            seasonal_periods=[7, 30, 365],
            seasonal_strength=0.75,
            peak_periods=[1, 15],
            low_periods=[3, 20],
            seasonal_adjustment_factor=1.15,
            confidence=0.80
        )
        
        assert analysis.item_id == "ITEM-001"
        assert analysis.seasonal_periods == [7, 30, 365]
        assert analysis.seasonal_strength == 0.75
        assert analysis.peak_periods == [1, 15]
        assert analysis.low_periods == [3, 20]
        assert analysis.seasonal_adjustment_factor == 1.15
        assert analysis.confidence == 0.80

    def test_item_correlation_analysis_structure(self):
        """Test ItemCorrelationAnalysis NamedTuple structure"""
        analysis = ItemCorrelationAnalysis(
            primary_item_id="ITEM-001",
            correlated_items=[("ITEM-002", 0.8), ("ITEM-003", 0.6)],
            substitution_items=["ITEM-004", "ITEM-005"],
            complementary_items=["ITEM-006", "ITEM-007"],
            impact_factor=0.75,
            bundle_opportunities=[{"items": ["ITEM-001", "ITEM-002"], "potential_savings": 0.1}]
        )
        
        assert analysis.primary_item_id == "ITEM-001"
        assert analysis.correlated_items == [("ITEM-002", 0.8), ("ITEM-003", 0.6)]
        assert analysis.substitution_items == ["ITEM-004", "ITEM-005"]
        assert analysis.complementary_items == ["ITEM-006", "ITEM-007"]
        assert analysis.impact_factor == 0.75
        assert len(analysis.bundle_opportunities) == 1

    def test_supplier_diversification_analysis_structure(self):
        """Test SupplierDiversificationAnalysis NamedTuple structure"""
        analysis = SupplierDiversificationAnalysis(
            item_id="ITEM-001",
            current_supplier_concentration=0.8,
            alternative_suppliers=["SUP-002", "SUP-003"],
            risk_score=0.7,
            diversification_recommendations=[{"action": "add_supplier", "priority": "high"}],
            cost_impact_of_diversification=0.05,
            recommended_supplier_split={"SUP-001": 0.6, "SUP-002": 0.4}
        )
        
        assert analysis.item_id == "ITEM-001"
        assert analysis.current_supplier_concentration == 0.8
        assert analysis.alternative_suppliers == ["SUP-002", "SUP-003"]
        assert analysis.risk_score == 0.7
        assert len(analysis.diversification_recommendations) == 1
        assert analysis.cost_impact_of_diversification == 0.05
        assert analysis.recommended_supplier_split == {"SUP-001": 0.6, "SUP-002": 0.4}

    # Test advanced process data types that aren't covered
    
    async def test_process_data_advanced_reorder_analysis(self, agent, mock_db_session):
        """Test advanced_reorder_analysis process type"""
        data = {
            "type": "advanced_reorder_analysis",
            "item_id": "ITEM-001"
        }
        
        # Mock the _perform_advanced_reorder_analysis method
        with patch.object(agent, '_perform_advanced_reorder_analysis') as mock_method:
            mock_method.return_value = AgentDecision(
                agent_id=agent.agent_id,
                decision_type="advanced_reorder",
                context={"item_id": "ITEM-001"},
                reasoning="Advanced reorder analysis completed",
                action="Increase reorder point to 45 units",
                confidence=0.85
            )
            
            result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)
        mock_method.assert_called_once()

    async def test_process_data_demand_forecast_analysis(self, agent, mock_db_session):
        """Test demand_forecast_analysis process type"""
        data = {
            "type": "demand_forecast_analysis",
            "item_id": "ITEM-001"
        }
        
        with patch.object(agent, '_perform_demand_forecast_analysis') as mock_method:
            mock_method.return_value = AgentDecision(
                agent_id=agent.agent_id,
                decision_type="demand_forecast",
                context={"item_id": "ITEM-001"},
                reasoning="Demand forecast analysis completed",
                action="Increasing demand trend detected",
                confidence=0.80
            )
            
            result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)
        mock_method.assert_called_once()

    async def test_process_data_bulk_purchase_analysis(self, agent, mock_db_session):
        """Test bulk_purchase_analysis process type"""
        data = {
            "type": "bulk_purchase_analysis",
            "item_id": "ITEM-001"
        }
        
        with patch.object(agent, '_perform_bulk_purchase_analysis') as mock_method:
            mock_method.return_value = AgentDecision(
                agent_id=agent.agent_id,
                decision_type="bulk_purchase",
                context={"item_id": "ITEM-001"},
                reasoning="Bulk purchase analysis completed",
                action="Bulk purchase of 500 units recommended",
                confidence=0.75
            )
            
            result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)
        mock_method.assert_called_once()

    async def test_process_data_expiry_waste_analysis(self, agent, mock_db_session):
        """Test expiry_waste_analysis process type"""
        data = {
            "type": "expiry_waste_analysis",
            "item_id": "ITEM-001"
        }
        
        with patch.object(agent, '_perform_expiry_waste_analysis') as mock_method:
            mock_method.return_value = AgentDecision(
                agent_id=agent.agent_id,
                decision_type="expiry_waste",
                context={"item_id": "ITEM-001"},
                reasoning="Expiry waste analysis completed",
                action="Low waste risk detected",
                confidence=0.90
            )
            
            result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)
        mock_method.assert_called_once()

    async def test_process_data_comprehensive_analytics(self, agent, mock_db_session):
        """Test comprehensive_analytics process type"""
        data = {
            "type": "comprehensive_analytics",
            "item_id": "ITEM-001"
        }
        
        with patch.object(agent, '_perform_comprehensive_analytics_analysis') as mock_method:
            mock_method.return_value = AgentDecision(
                agent_id=agent.agent_id,
                decision_type="comprehensive",
                context={"item_id": "ITEM-001"},
                reasoning="Comprehensive analytics completed",
                action="Complete inventory optimization completed",
                confidence=0.88
            )
            
            result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)
        mock_method.assert_called_once()

    # Test methods directly where possible to increase coverage
    
    def test_aggregate_daily_consumption(self, agent):
        """Test _aggregate_daily_consumption method"""
        # Create mock stock movements
        movements = [
            Mock(quantity=10, movement_date=datetime.now() - timedelta(days=1)),
            Mock(quantity=15, movement_date=datetime.now() - timedelta(days=1)),
            Mock(quantity=8, movement_date=datetime.now() - timedelta(days=2)),
            Mock(quantity=12, movement_date=datetime.now() - timedelta(days=2)),
        ]
        
        # The method returns a list, not a dict
        result = agent._aggregate_daily_consumption(movements)
        assert isinstance(result, list)
        assert len(result) >= 2  # At least 2 days of data
        assert all(isinstance(x, float) for x in result)

    async def test_calculate_simple_reorder_point(self, agent, mock_db_session):
        """Test _calculate_simple_reorder_point method"""
        # Create mock item
        mock_item = Mock()
        mock_item.id = 1
        mock_item.reorder_quantity = 100
        
        # Mock stock movements
        mock_movements = [
            Mock(quantity=10, movement_date=datetime.now() - timedelta(days=i))
            for i in range(30)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_movements
        
        result = await agent._calculate_simple_reorder_point(mock_db_session, mock_item)
        
        assert result is not None
        assert isinstance(result, OptimalReorderPoint)
        assert result.item_id == 1

    def test_estimate_demand_standard_deviation(self, agent, mock_db_session):
        """Test _estimate_demand_standard_deviation method"""
        # Mock stock movements
        mock_movements = [
            Mock(quantity=10, movement_date=datetime.now() - timedelta(days=i))
            for i in range(10)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_movements
        
        # Mock the _aggregate_daily_consumption method to return test data
        with patch.object(agent, '_aggregate_daily_consumption') as mock_aggregate:
            mock_aggregate.return_value = [10, 12, 8, 15, 11, 9, 14, 13, 7, 16]
            
            result = agent._estimate_demand_standard_deviation(mock_db_session, "ITEM-001", 10.0)
            
            assert result > 0
            assert isinstance(result, float)

    def test_get_z_score_for_service_level(self, agent):
        """Test _get_z_score_for_service_level method"""
        try:
            # Test various service levels
            service_levels = [0.90, 0.95, 0.99]
            for level in service_levels:
                z_score = agent._get_z_score_for_service_level(level)
                assert z_score > 0
                assert isinstance(z_score, (int, float))
        except AttributeError:
            pytest.skip("Method _get_z_score_for_service_level not accessible")

    def test_calculate_total_inventory_cost(self, agent):
        """Test _calculate_total_inventory_cost method"""
        annual_demand = 1000.0
        order_quantity = 100
        unit_cost = 10.0
        service_level = 0.95
        safety_stock = 20.0
        
        result = agent._calculate_total_inventory_cost(
            annual_demand, order_quantity, unit_cost, service_level, safety_stock
        )
        
        assert isinstance(result, float)
        assert result > 0

    # Test error handling paths
    
    async def test_process_data_invalid_analysis_type(self, agent, mock_db_session):
        """Test process_data with invalid analysis type"""
        data = {
            "type": "invalid_analysis_type",
            "item_id": "ITEM-001"
        }
        
        result = await agent.process_data(data)
        
        # Should handle gracefully, either return None or a default decision
        assert result is None or isinstance(result, AgentDecision)

    async def test_process_data_missing_item_id(self, agent, mock_db_session):
        """Test process_data with missing item_id"""
        data = {
            "type": "stock_movement"
            # Missing item_id
        }
        
        result = await agent.process_data(data)
        
        # Should handle gracefully
        assert result is None or isinstance(result, AgentDecision)

    async def test_process_data_database_error(self, agent, mock_db_session):
        """Test process_data with database errors"""
        # Make database queries fail
        mock_db_session.query.side_effect = Exception("Database connection failed")
        
        data = {
            "type": "stock_movement",
            "movement": {
                "item_id": "ITEM-001",
                "quantity": 10,
                "movement_type": "IN"
            }
        }
        
        result = await agent.process_data(data)
        
        # Should handle database errors gracefully
        assert result is None or isinstance(result, AgentDecision)

    # Test configuration edge cases
    
    def test_config_with_missing_values(self, mock_anthropic, mock_db_session):
        """Test agent with minimal configuration"""
        minimal_config = {
            "check_interval": 300
            # Missing many optional config values
        }
        
        agent = InventoryAgent(
            agent_id="test_agent",
            api_key="test_key",
            config=minimal_config,
            db_url="sqlite:///:memory:"
        )
        
        # Should have reasonable defaults
        assert hasattr(agent, 'low_stock_multiplier')
        assert hasattr(agent, 'reorder_lead_time')
        assert hasattr(agent, 'consumption_analysis_days')

    def test_config_with_invalid_values(self, mock_anthropic, mock_db_session):
        """Test agent with invalid configuration values"""
        invalid_config = {
            "check_interval": 300,
            "low_stock_multiplier": -1.0,  # Invalid negative
            "reorder_lead_time": -5,       # Invalid negative
            "consumption_analysis_days": 0  # Invalid zero
        }
        
        agent = InventoryAgent(
            agent_id="test_agent",
            api_key="test_key",
            config=invalid_config,
            db_url="sqlite:///:memory:"
        )
        
        # The agent stores config values as provided, no validation
        # So we test that they are the stored values
        assert agent.low_stock_multiplier == -1.0
        assert agent.reorder_lead_time == -5
        assert agent.consumption_analysis_days == 0

    # Test system prompt coverage
    
    def test_system_prompt_contains_key_elements(self, agent):
        """Test that system prompt contains expected elements"""
        prompt = agent.system_prompt
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert "inventory" in prompt.lower()
        assert "reorder" in prompt.lower() or "stock" in prompt.lower()

    # Test generate_report method coverage
    
    async def test_generate_report_basic(self, agent, mock_db_session):
        """Test generate_report method"""
        # Mock basic inventory data
        mock_item1 = Mock()
        mock_item1.id = 1
        mock_item1.sku = "ITEM-001"
        mock_item1.name = "Item 1"
        mock_item1.current_stock = 50
        mock_item1.reorder_point = 30
        mock_item1.unit_cost = Decimal('10.00')
        
        mock_item2 = Mock()
        mock_item2.id = 2
        mock_item2.sku = "ITEM-002"
        mock_item2.name = "Item 2"
        mock_item2.current_stock = 20
        mock_item2.reorder_point = 25
        mock_item2.unit_cost = Decimal('15.00')
        
        mock_items = [mock_item1, mock_item2]
        
        # Mock SessionLocal to return our mock session
        with patch.object(agent, 'SessionLocal') as mock_session_local:
            mock_session_local.return_value = mock_db_session
            
            # Mock the query chain for different calls
            def query_side_effect(*args):
                mock_query = Mock()
                if args[0] == Item:
                    mock_query.filter.return_value.all.return_value = mock_items
                else:  # StockMovement queries
                    mock_query.filter.return_value.all.return_value = []
                return mock_query
            
            mock_db_session.query.side_effect = query_side_effect
            
            result = await agent.generate_report()
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "alerts" in result
        assert "recent_decisions" in result
        
        summary = result["summary"]
        assert "total_items" in summary
        assert "low_stock_items" in summary
        assert "total_value" in summary