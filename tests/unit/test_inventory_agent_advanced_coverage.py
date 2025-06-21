"""
Additional test coverage for InventoryAgent advanced functionality
This file specifically targets the missing coverage gaps identified in the analysis
"""
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

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
from models.inventory import ItemStatus, StockMovementType


class TestInventoryAgentAdvancedCoverage:
    """Test cases targeting missing coverage in InventoryAgent"""

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
        """Extended agent configuration for advanced features"""
        return {
            "check_interval": 300,
            "low_stock_multiplier": 1.2,
            "reorder_lead_time": 7,
            "consumption_analysis_days": 30,
            "expiry_warning_days": 5,
            "urgent_reorder_threshold": 10,
            "critical_reorder_threshold": 5,
            # Advanced analytics config
            "advanced_analytics_enabled": True,
            "forecast_horizon_days": 30,
            "seasonality_window_days": 365,
            "min_forecast_accuracy": 0.7,
            "seasonal_adjustment_factor": 0.2,
            "correlation_threshold": 0.3,
            "supplier_risk_threshold": 0.7,
            "bulk_purchase_threshold": 100,
            "waste_prediction_days": 14
        }

    @pytest.fixture
    def agent(self, mock_anthropic, mock_db_session, agent_config):
        """Create InventoryAgent instance with advanced config"""
        return InventoryAgent(
            agent_id="test_inventory_agent",
            api_key="test_api_key",
            config=agent_config,
            db_url="sqlite:///:memory:"
        )

    # Test Advanced Process Data Types (Lines 160-237)
    
    async def test_process_data_advanced_reorder_analysis(self, agent, mock_db_session):
        """Test advanced_reorder_analysis process type"""
        # Mock item data
        mock_item = Mock()
        mock_item.id = 1
        mock_item.sku = "ITEM-001"
        mock_item.name = "Test Item"
        mock_item.current_stock = 50
        mock_item.reorder_point = 30
        mock_item.reorder_quantity = 100
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_item
        
        # Mock consumption data
        mock_movements = [
            Mock(quantity=10, movement_date=datetime.now() - timedelta(days=i))
            for i in range(30)
        ]
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_movements
        
        data = {
            "analysis_type": "advanced_reorder_analysis",
            "item_id": 1,
            "include_demand_forecast": True,
            "include_seasonality": True
        }
        
        result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)
        assert "advanced_reorder" in result.decision_type.lower()

    async def test_process_data_demand_forecast_analysis(self, agent, mock_db_session):
        """Test demand_forecast_analysis process type"""
        # Mock historical data for forecasting
        historical_data = []
        for i in range(90):  # 90 days of data
            historical_data.append(Mock(
                consumption=10 + np.sin(i/7) * 3,  # Weekly pattern
                date=datetime.now() - timedelta(days=i)
            ))
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = historical_data
        
        data = {
            "analysis_type": "demand_forecast_analysis",
            "item_id": 1,
            "forecast_days": 30,
            "include_seasonality": True
        }
        
        result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)

    async def test_process_data_bulk_purchase_analysis(self, agent, mock_db_session):
        """Test bulk_purchase_analysis process type"""
        # Mock supplier offers
        mock_offers = [
            Mock(supplier_id=1, quantity=100, unit_price=Decimal('10.00')),
            Mock(supplier_id=1, quantity=500, unit_price=Decimal('9.50')),
            Mock(supplier_id=1, quantity=1000, unit_price=Decimal('9.00'))
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_offers
        
        data = {
            "analysis_type": "bulk_purchase_analysis",
            "item_id": 1,
            "quantities_to_analyze": [100, 500, 1000],
            "include_storage_costs": True
        }
        
        result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)

    async def test_process_data_seasonality_analysis(self, agent, mock_db_session):
        """Test seasonality_analysis process type"""
        # Generate seasonal data
        seasonal_data = []
        for i in range(365):
            base_demand = 100
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Yearly cycle
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)     # Weekly cycle
            seasonal_data.append(Mock(
                consumption=base_demand * seasonal_factor * weekly_factor,
                date=datetime.now() - timedelta(days=i)
            ))
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = seasonal_data
        
        data = {
            "analysis_type": "seasonality_analysis",
            "item_id": 1,
            "analysis_periods": ["weekly", "monthly", "quarterly", "yearly"]
        }
        
        result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)

    async def test_process_data_correlation_analysis(self, agent, mock_db_session):
        """Test correlation_analysis process type"""
        # Mock correlated items data
        mock_items = [Mock(id=i, sku=f"ITEM-{i:03d}") for i in range(1, 6)]
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_items
        
        # Mock consumption correlation data
        correlation_data = {}
        for item in mock_items:
            correlation_data[item.id] = [
                Mock(consumption=10 + np.random.normal(0, 2), date=datetime.now() - timedelta(days=i))
                for i in range(30)
            ]
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.side_effect = \
            lambda: correlation_data.get(mock_db_session.query.call_args[0][0], [])
        
        data = {
            "analysis_type": "correlation_analysis",
            "target_item_id": 1,
            "correlation_threshold": 0.3
        }
        
        result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)

    async def test_process_data_supplier_diversification_analysis(self, agent, mock_db_session):
        """Test supplier_diversification_analysis process type"""
        # Mock supplier data
        mock_suppliers = [
            Mock(id=1, name="Supplier A", performance_score=0.9),
            Mock(id=2, name="Supplier B", performance_score=0.8),
            Mock(id=3, name="Supplier C", performance_score=0.7)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_suppliers
        
        data = {
            "analysis_type": "supplier_diversification_analysis",
            "item_id": 1,
            "risk_tolerance": 0.3,
            "include_cost_analysis": True
        }
        
        result = await agent.process_data(data)
        
        assert result is not None
        assert isinstance(result, AgentDecision)

    # Test Demand Forecasting Methods (Lines 697-906)
    
    async def test_predict_demand(self, agent, mock_db_session):
        """Test predict_demand method"""
        # Mock historical consumption data
        historical_data = []
        for i in range(90):
            historical_data.append({
                'date': datetime.now() - timedelta(days=i),
                'consumption': 10 + np.sin(i/7) * 2  # Weekly pattern
            })
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            Mock(**data) for data in historical_data
        ]
        
        result = await agent.predict_demand(mock_db_session, 1, 30)
        
        assert isinstance(result, DemandForecast)
        assert result.item_id == 1
        assert result.forecast_days == 30
        assert len(result.daily_forecast) == 30
        assert result.confidence > 0.0

    async def test_predict_demand_insufficient_data(self, agent, mock_db_session):
        """Test predict_demand with insufficient historical data"""
        # Only 5 days of data
        historical_data = [
            Mock(date=datetime.now() - timedelta(days=i), consumption=10)
            for i in range(5)
        ]
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = historical_data
        
        result = await agent.predict_demand(mock_db_session, 1, 30)
        
        assert isinstance(result, DemandForecast)
        assert result.confidence < 0.5  # Low confidence due to insufficient data

    async def test_apply_holt_winters_forecast(self, agent):
        """Test _apply_holt_winters_forecast method"""
        # Create seasonal data
        data = []
        for i in range(52):  # 52 weeks
            base = 100
            trend = i * 0.5
            seasonal = 20 * np.sin(2 * np.pi * i / 12)  # Monthly seasonality
            data.append(base + trend + seasonal + np.random.normal(0, 5))
        
        result = agent._apply_holt_winters_forecast(data, 4)  # Forecast 4 periods
        
        assert len(result) == 4
        assert all(isinstance(x, (int, float)) for x in result)

    async def test_calculate_forecast_accuracy(self, agent):
        """Test _calculate_forecast_accuracy method"""
        actual = [10, 12, 11, 13, 14]
        predicted = [10.5, 11.8, 11.2, 12.9, 13.7]
        
        accuracy = agent._calculate_forecast_accuracy(actual, predicted)
        
        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.8  # Should be high accuracy for close predictions

    # Test Optimal Reorder Point Calculations (Lines 927-1161)
    
    async def test_calculate_optimal_reorder_point(self, agent, mock_db_session):
        """Test calculate_optimal_reorder_point method"""
        # Mock item data
        mock_item = Mock()
        mock_item.id = 1
        mock_item.current_stock = 100
        mock_item.unit_cost = Decimal('10.00')
        mock_item.storage_cost_per_unit = Decimal('0.50')
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_item
        
        # Mock consumption data
        consumption_data = [Mock(consumption=15, date=datetime.now() - timedelta(days=i)) for i in range(30)]
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = consumption_data
        
        result = await agent.calculate_optimal_reorder_point(mock_db_session, 1, service_level=0.95)
        
        assert isinstance(result, OptimalReorderPoint)
        assert result.item_id == 1
        assert result.optimal_reorder_point > 0
        assert 0.0 <= result.service_level <= 1.0

    async def test_calculate_simple_reorder_point(self, agent):
        """Test _calculate_simple_reorder_point method"""
        avg_consumption = 10
        lead_time = 7
        safety_stock = 20
        
        result = agent._calculate_simple_reorder_point(avg_consumption, lead_time, safety_stock)
        
        expected = (avg_consumption * lead_time) + safety_stock
        assert result == expected

    async def test_estimate_demand_standard_deviation(self, agent):
        """Test _estimate_demand_standard_deviation method"""
        consumption_data = [10, 12, 8, 15, 11, 9, 14, 13, 7, 16]
        
        result = agent._estimate_demand_standard_deviation(consumption_data)
        
        assert result > 0
        assert isinstance(result, float)

    # Test Bulk Purchase Optimization (Lines 1162-1316)
    
    async def test_optimize_bulk_purchase(self, agent, mock_db_session):
        """Test optimize_bulk_purchase method"""
        # Mock item and pricing data
        mock_item = Mock()
        mock_item.id = 1
        mock_item.current_stock = 50
        mock_item.storage_cost_per_unit = Decimal('0.25')
        
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_item
        
        # Mock pricing tiers
        pricing_tiers = [
            {'quantity': 100, 'unit_price': Decimal('10.00')},
            {'quantity': 500, 'unit_price': Decimal('9.50')},
            {'quantity': 1000, 'unit_price': Decimal('9.00')}
        ]
        
        # Mock demand forecast
        with patch.object(agent, 'predict_demand') as mock_predict:
            mock_forecast = DemandForecast(
                item_id=1,
                forecast_days=90,
                daily_forecast=[10] * 90,
                total_forecast=900,
                confidence=0.8,
                method="test"
            )
            mock_predict.return_value = mock_forecast
            
            result = await agent.optimize_bulk_purchase(mock_db_session, 1, pricing_tiers)
        
        assert isinstance(result, BulkPurchaseOptimization)
        assert result.item_id == 1
        assert result.optimal_quantity > 0
        assert result.total_cost > 0

    async def test_calculate_bulk_purchase_total_cost(self, agent):
        """Test _calculate_bulk_purchase_total_cost method"""
        quantity = 500
        unit_price = Decimal('9.50')
        storage_cost_per_unit = Decimal('0.25')
        holding_period_days = 90
        
        result = agent._calculate_bulk_purchase_total_cost(
            quantity, unit_price, storage_cost_per_unit, holding_period_days
        )
        
        expected_purchase_cost = quantity * unit_price
        expected_storage_cost = quantity * storage_cost_per_unit * (holding_period_days / 365)
        expected_total = expected_purchase_cost + expected_storage_cost
        
        assert abs(result - expected_total) < Decimal('0.01')

    async def test_analyze_bulk_purchase_opportunities(self, agent, mock_db_session):
        """Test analyze_bulk_purchase_opportunities method"""
        # Mock multiple items
        mock_items = [
            Mock(id=1, sku="ITEM-001", name="Item 1"),
            Mock(id=2, sku="ITEM-002", name="Item 2")
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_items
        
        # Mock optimization results
        with patch.object(agent, 'optimize_bulk_purchase') as mock_optimize:
            mock_optimization = BulkPurchaseOptimization(
                item_id=1,
                optimal_quantity=500,
                total_cost=Decimal('5000'),
                savings_amount=Decimal('250'),
                savings_percentage=5.0,
                break_even_days=30
            )
            mock_optimize.return_value = mock_optimization
            
            result = await agent.analyze_bulk_purchase_opportunities(mock_db_session, minimum_savings=100)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(opt, BulkPurchaseOptimization) for opt in result)

    # Test Seasonality Analysis (Lines 1530-1672)
    
    async def test_analyze_seasonality_patterns(self, agent, mock_db_session):
        """Test analyze_seasonality_patterns method"""
        # Generate seasonal consumption data
        consumption_data = []
        for i in range(365):
            base = 100
            yearly_pattern = 20 * np.sin(2 * np.pi * i / 365)
            monthly_pattern = 10 * np.sin(2 * np.pi * i / 30)
            weekly_pattern = 5 * np.sin(2 * np.pi * i / 7)
            consumption = base + yearly_pattern + monthly_pattern + weekly_pattern
            
            consumption_data.append(Mock(
                consumption=consumption,
                date=datetime.now() - timedelta(days=365-i)
            ))
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = consumption_data
        
        result = await agent.analyze_seasonality_patterns(mock_db_session, 1)
        
        assert isinstance(result, SeasonalityAnalysis)
        assert result.item_id == 1
        assert 'weekly' in result.seasonal_patterns
        assert 'monthly' in result.seasonal_patterns
        assert 'yearly' in result.seasonal_patterns

    async def test_analyze_seasonal_period(self, agent):
        """Test _analyze_seasonal_period method"""
        # Create data with clear weekly pattern
        data = []
        for i in range(84):  # 12 weeks
            base = 100
            pattern = 20 * np.sin(2 * np.pi * i / 7)  # Weekly cycle
            data.append(base + pattern)
        
        strength, confidence = agent._analyze_seasonal_period(data, 7)
        
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert strength > 0.5  # Should detect the strong weekly pattern

    async def test_calculate_seasonal_adjustment(self, agent):
        """Test _calculate_seasonal_adjustment method"""
        seasonal_pattern = [1.2, 1.1, 0.9, 0.8, 1.0, 1.1, 1.2]  # Weekly pattern
        current_day_of_week = 0  # Monday
        
        adjustment = agent._calculate_seasonal_adjustment(seasonal_pattern, current_day_of_week)
        
        assert adjustment == 1.2  # Should return Monday's factor

    # Test Item Correlation Analysis (Lines 1673-1810)
    
    async def test_analyze_item_correlations(self, agent, mock_db_session):
        """Test analyze_item_correlations method"""
        # Mock target item
        target_item = Mock(id=1, sku="ITEM-001")
        mock_db_session.query.return_value.filter.return_value.first.return_value = target_item
        
        # Mock other items
        other_items = [Mock(id=i, sku=f"ITEM-{i:03d}") for i in range(2, 6)]
        mock_db_session.query.return_value.filter.return_value.all.return_value = other_items
        
        # Mock consumption data for correlation
        def mock_consumption_query(*args, **kwargs):
            # Return correlated data for item 2, uncorrelated for others
            item_id = args[0] if args else 1
            if item_id == 2:
                return [Mock(consumption=10 + i, date=datetime.now() - timedelta(days=i)) for i in range(30)]
            else:
                return [Mock(consumption=np.random.randint(5, 15), date=datetime.now() - timedelta(days=i)) for i in range(30)]
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.side_effect = mock_consumption_query
        
        result = await agent.analyze_item_correlations(mock_db_session, 1, correlation_threshold=0.3)
        
        assert isinstance(result, ItemCorrelationAnalysis)
        assert result.target_item_id == 1
        assert isinstance(result.correlations, dict)

    async def test_generate_bundle_opportunities(self, agent):
        """Test _generate_bundle_opportunities method"""
        correlations = {
            2: {'correlation': 0.8, 'type': 'complementary'},
            3: {'correlation': 0.7, 'type': 'substitute'},
            4: {'correlation': 0.9, 'type': 'complementary'}
        }
        
        bundles = agent._generate_bundle_opportunities(correlations, min_correlation=0.75)
        
        assert isinstance(bundles, list)
        assert len(bundles) > 0
        assert all('items' in bundle for bundle in bundles)

    # Test Supplier Diversification Analysis (Lines 1811-2010)
    
    async def test_analyze_supplier_diversification(self, agent, mock_db_session):
        """Test analyze_supplier_diversification method"""
        # Mock item
        mock_item = Mock(id=1, sku="ITEM-001")
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_item
        
        # Mock suppliers
        suppliers = [
            Mock(id=1, name="Supplier A", performance_score=0.9, market_share=0.6),
            Mock(id=2, name="Supplier B", performance_score=0.8, market_share=0.3),
            Mock(id=3, name="Supplier C", performance_score=0.7, market_share=0.1)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = suppliers
        
        result = await agent.analyze_supplier_diversification(mock_db_session, 1)
        
        assert isinstance(result, SupplierDiversificationAnalysis)
        assert result.item_id == 1
        assert result.risk_score >= 0.0
        assert isinstance(result.recommendations, list)

    async def test_calculate_optimal_supplier_split(self, agent):
        """Test _calculate_optimal_supplier_split method"""
        suppliers = [
            {'id': 1, 'performance_score': 0.9, 'cost_factor': 1.0},
            {'id': 2, 'performance_score': 0.8, 'cost_factor': 0.95},
            {'id': 3, 'performance_score': 0.7, 'cost_factor': 0.9}
        ]
        
        split = agent._calculate_optimal_supplier_split(suppliers, total_volume=1000)
        
        assert isinstance(split, dict)
        assert len(split) == len(suppliers)
        assert abs(sum(split.values()) - 1000) < 1  # Should sum to total volume

    # Test Advanced Supplier Performance (Lines 2011-2347)
    
    async def test_analyze_supplier_performance_advanced(self, agent, mock_db_session):
        """Test analyze_supplier_performance_advanced method"""
        supplier_id = 1
        
        # Mock purchase orders and deliveries
        mock_orders = [
            Mock(
                id=i,
                order_date=datetime.now() - timedelta(days=i*7),
                expected_delivery_date=datetime.now() - timedelta(days=i*7-5),
                actual_delivery_date=datetime.now() - timedelta(days=i*7-4),
                total_amount=Decimal('1000'),
                quality_score=0.9
            )
            for i in range(10)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_orders
        
        result = await agent.analyze_supplier_performance_advanced(mock_db_session, supplier_id)
        
        assert isinstance(result, SupplierPerformanceMetrics)
        assert result.supplier_id == supplier_id
        assert 0.0 <= result.on_time_delivery_rate <= 1.0
        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.overall_score <= 1.0

    async def test_calculate_on_time_delivery_rate(self, agent):
        """Test _calculate_on_time_delivery_rate method"""
        orders = [
            Mock(expected_delivery_date=datetime(2023, 1, 10), actual_delivery_date=datetime(2023, 1, 10)),  # On time
            Mock(expected_delivery_date=datetime(2023, 1, 15), actual_delivery_date=datetime(2023, 1, 16)),  # Late
            Mock(expected_delivery_date=datetime(2023, 1, 20), actual_delivery_date=datetime(2023, 1, 19)),  # Early
        ]
        
        rate = agent._calculate_on_time_delivery_rate(orders)
        
        assert 0.0 <= rate <= 1.0
        assert abs(rate - 2/3) < 0.01  # 2 out of 3 on time or early

    async def test_calculate_quality_score(self, agent):
        """Test _calculate_quality_score method"""
        orders = [
            Mock(quality_score=0.9),
            Mock(quality_score=0.8),
            Mock(quality_score=0.95),
            Mock(quality_score=0.85)
        ]
        
        score = agent._calculate_quality_score(orders)
        
        expected = (0.9 + 0.8 + 0.95 + 0.85) / 4
        assert abs(score - expected) < 0.01

    # Test Expiry Waste Prediction (Lines 1317-1502)
    
    async def test_predict_expiry_waste(self, agent, mock_db_session):
        """Test predict_expiry_waste method"""
        # Mock items with expiry dates
        mock_items = [
            Mock(
                id=1,
                sku="ITEM-001",
                current_stock=100,
                expiry_date=datetime.now() + timedelta(days=5),
                unit_cost=Decimal('10.00')
            ),
            Mock(
                id=2,
                sku="ITEM-002", 
                current_stock=50,
                expiry_date=datetime.now() + timedelta(days=15),
                unit_cost=Decimal('20.00')
            )
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_items
        
        # Mock consumption forecast
        with patch.object(agent, 'predict_demand') as mock_predict:
            mock_forecast = DemandForecast(
                item_id=1,
                forecast_days=14,
                daily_forecast=[5] * 14,
                total_forecast=70,
                confidence=0.8,
                method="test"
            )
            mock_predict.return_value = mock_forecast
            
            result = await agent.predict_expiry_waste(mock_db_session, days_ahead=14)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('item_id' in item for item in result)
        assert all('predicted_waste' in item for item in result)

    # Test Comprehensive Analytics (Lines 2637-2842)
    
    async def test_generate_comprehensive_reorder_recommendation(self, agent, mock_db_session):
        """Test generate_comprehensive_reorder_recommendation method"""
        item_id = 1
        
        # Mock item data
        mock_item = Mock(
            id=1,
            sku="ITEM-001",
            name="Test Item",
            current_stock=50,
            reorder_point=30,
            reorder_quantity=100
        )
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_item
        
        # Mock all the analytics methods
        with patch.object(agent, 'predict_demand') as mock_demand, \
             patch.object(agent, 'calculate_optimal_reorder_point') as mock_reorder, \
             patch.object(agent, 'analyze_seasonality_patterns') as mock_seasonal, \
             patch.object(agent, 'optimize_bulk_purchase') as mock_bulk:
            
            # Setup mock returns
            mock_demand.return_value = DemandForecast(1, 30, [10]*30, 300, 0.8, "test")
            mock_reorder.return_value = OptimalReorderPoint(1, 45, 0.95, "statistical")
            mock_seasonal.return_value = SeasonalityAnalysis(1, {}, 1.0, 0.8)
            mock_bulk.return_value = BulkPurchaseOptimization(1, 500, Decimal('5000'), Decimal('250'), 5.0, 30)
            
            result = await agent.generate_comprehensive_reorder_recommendation(mock_db_session, item_id)
        
        assert isinstance(result, dict)
        assert 'item_id' in result
        assert 'recommendation' in result
        assert 'analysis_components' in result

    # Test NamedTuple Data Structures (Lines 24-100)
    
    def test_demand_forecast_namedtuple(self):
        """Test DemandForecast NamedTuple"""
        forecast = DemandForecast(
            item_id=1,
            forecast_days=30,
            daily_forecast=[10, 12, 11],
            total_forecast=33,
            confidence=0.85,
            method="holt_winters"
        )
        
        assert forecast.item_id == 1
        assert forecast.forecast_days == 30
        assert forecast.total_forecast == 33
        assert forecast.confidence == 0.85

    def test_optimal_reorder_point_namedtuple(self):
        """Test OptimalReorderPoint NamedTuple"""
        reorder_point = OptimalReorderPoint(
            item_id=1,
            optimal_reorder_point=45,
            service_level=0.95,
            method="statistical"
        )
        
        assert reorder_point.item_id == 1
        assert reorder_point.optimal_reorder_point == 45
        assert reorder_point.service_level == 0.95

    def test_bulk_purchase_optimization_namedtuple(self):
        """Test BulkPurchaseOptimization NamedTuple"""
        optimization = BulkPurchaseOptimization(
            item_id=1,
            optimal_quantity=500,
            total_cost=Decimal('5000.00'),
            savings_amount=Decimal('250.00'),
            savings_percentage=5.0,
            break_even_days=30
        )
        
        assert optimization.item_id == 1
        assert optimization.optimal_quantity == 500
        assert optimization.total_cost == Decimal('5000.00')

    # Test Error Handling in Advanced Methods
    
    async def test_predict_demand_database_error(self, agent, mock_db_session):
        """Test predict_demand error handling"""
        mock_db_session.query.side_effect = Exception("Database error")
        
        result = await agent.predict_demand(mock_db_session, 1, 30)
        
        # Should return a fallback forecast
        assert isinstance(result, DemandForecast)
        assert result.confidence < 0.5  # Low confidence due to error

    async def test_analyze_seasonality_no_data(self, agent, mock_db_session):
        """Test seasonality analysis with no data"""
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        
        result = await agent.analyze_seasonality_patterns(mock_db_session, 1)
        
        assert isinstance(result, SeasonalityAnalysis)
        assert result.overall_seasonality_strength == 0.0

    async def test_bulk_purchase_optimization_invalid_pricing(self, agent, mock_db_session):
        """Test bulk purchase optimization with invalid pricing data"""
        mock_item = Mock(id=1, current_stock=50, storage_cost_per_unit=Decimal('0.25'))
        mock_db_session.query.return_value.filter.return_value.first.return_value = mock_item
        
        # Empty pricing tiers
        pricing_tiers = []
        
        with patch.object(agent, 'predict_demand') as mock_predict:
            mock_predict.return_value = DemandForecast(1, 90, [10]*90, 900, 0.8, "test")
            
            result = await agent.optimize_bulk_purchase(mock_db_session, 1, pricing_tiers)
        
        # Should handle gracefully
        assert result is None or isinstance(result, BulkPurchaseOptimization)

    # Test Configuration Edge Cases
    
    def test_advanced_config_defaults(self, mock_anthropic, mock_db_session):
        """Test advanced configuration defaults"""
        minimal_config = {"check_interval": 300}
        
        agent = InventoryAgent(
            agent_id="test_agent",
            api_key="test_key",
            config=minimal_config,
            db_url="sqlite:///:memory:"
        )
        
        # Should have default values for advanced analytics
        assert hasattr(agent, 'forecast_horizon_days')
        assert hasattr(agent, 'seasonality_window_days')
        assert hasattr(agent, 'correlation_threshold')

    def test_invalid_config_handling(self, mock_anthropic, mock_db_session):
        """Test handling of invalid configuration values"""
        invalid_config = {
            "check_interval": 300,
            "forecast_horizon_days": -10,  # Invalid negative value
            "correlation_threshold": 1.5,  # Invalid > 1.0
            "min_forecast_accuracy": 2.0   # Invalid > 1.0
        }
        
        agent = InventoryAgent(
            agent_id="test_agent",
            api_key="test_key", 
            config=invalid_config,
            db_url="sqlite:///:memory:"
        )
        
        # Should sanitize invalid values
        assert agent.forecast_horizon_days > 0
        assert 0.0 <= agent.correlation_threshold <= 1.0
        assert 0.0 <= agent.min_forecast_accuracy <= 1.0