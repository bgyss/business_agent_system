"""
Unit tests for Enhanced AccountingAgent advanced financial analysis capabilities
"""

import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.accounting_agent_enhanced import EnhancedAccountingAgent as AccountingAgent
from models.financial import (
    TransactionModel,
    TransactionType,
)


class TestEnhancedAccountingAgent:
    """Test cases for Enhanced AccountingAgent advanced capabilities"""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch("agents.base_agent.Anthropic") as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Advanced financial analysis completed")]
            mock_client.return_value.messages.create.return_value = mock_response
            yield mock_client

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        with patch("agents.base_agent.create_engine"), patch(
            "agents.base_agent.sessionmaker"
        ) as mock_sessionmaker:
            mock_session = Mock()
            mock_sessionmaker.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def enhanced_agent_config(self):
        """Enhanced accounting agent configuration"""
        return {
            "check_interval": 300,
            "anomaly_threshold": 0.25,
            "alert_thresholds": {
                "cash_low": 1000,
                "receivables_overdue": 30,
                "payables_overdue": 7,
            },
            "forecasting": {
                "prediction_days": 30,
                "seasonal_analysis_days": 365,
                "trend_analysis_periods": 7,
                "confidence_factors": {
                    "data_volume": 0.3,
                    "historical_accuracy": 0.25,
                    "trend_stability": 0.25,
                    "seasonal_consistency": 0.2,
                },
            },
        }

    @pytest.fixture
    def enhanced_accounting_agent(self, mock_anthropic, mock_db_session, enhanced_agent_config):
        """Create enhanced accounting agent instance"""
        return AccountingAgent(
            agent_id="enhanced_accounting_agent",
            api_key="test_api_key",
            config=enhanced_agent_config,
            db_url="sqlite:///:memory:",
        )

    def test_enhanced_initialization(self, enhanced_accounting_agent, enhanced_agent_config):
        """Test enhanced agent initialization with new config"""
        assert enhanced_accounting_agent.agent_id == "enhanced_accounting_agent"
        assert enhanced_accounting_agent.forecasting_config == enhanced_agent_config["forecasting"]
        assert hasattr(enhanced_accounting_agent, "decision_outcomes")
        assert hasattr(enhanced_accounting_agent, "forecasting_accuracy_history")

    def test_enhanced_system_prompt(self, enhanced_accounting_agent):
        """Test enhanced system prompt content"""
        prompt = enhanced_accounting_agent.system_prompt
        assert "advanced AI Accounting Agent" in prompt
        assert "anomaly detection" in prompt
        assert "cash flow forecasting" in prompt
        assert "confidence scoring" in prompt
        assert "Decision outcome tracking" in prompt
        assert "Z-score" in prompt
        assert "ensemble methods" in prompt

    @pytest.mark.asyncio
    async def test_enhanced_transaction_analysis(self, enhanced_accounting_agent, mock_db_session):
        """Test enhanced transaction analysis with multiple detection methods"""
        transaction_data = {
            "id": "1",
            "description": "Unusual large transaction",
            "amount": Decimal("5000.00"),  # Much larger than average
            "transaction_type": TransactionType.INCOME,
            "category": "sales",
            "transaction_date": datetime.now(),
        }

        data = {"type": "new_transaction", "transaction": transaction_data}

        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock similar transactions with consistent lower amounts
        similar_transactions = [
            Mock(amount=Decimal("1000.00"), transaction_date=datetime.now() - timedelta(hours=i))
            for i in range(1, 11)  # 10 similar transactions
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = (
            similar_transactions
        )

        decision = await enhanced_accounting_agent.process_data(data)

        # Should detect anomaly with enhanced methods
        assert decision is not None
        assert decision.decision_type == "transaction_anomaly"
        assert "anomaly_details" in decision.context
        assert "confidence_factors" in decision.context
        # Confidence should be calculated dynamically
        assert 0.1 <= decision.confidence <= 0.95

    @pytest.mark.asyncio
    async def test_cash_flow_forecasting(self, enhanced_accounting_agent, mock_db_session):
        """Test advanced cash flow forecasting capability"""
        data = {"type": "cash_flow_forecast", "forecast_days": 30}

        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock historical transactions for forecasting
        base_date = datetime.now() - timedelta(days=60)
        transactions = []
        for i in range(60):  # 60 days of data
            # Create income transactions
            transactions.append(
                Mock(
                    amount=Decimal("1000.00") + Decimal(str(i * 10)),  # Trending upward
                    transaction_type=TransactionType.INCOME,
                    transaction_date=base_date + timedelta(days=i),
                )
            )
            # Create expense transactions
            transactions.append(
                Mock(
                    amount=Decimal("500.00"),
                    transaction_type=TransactionType.EXPENSE,
                    transaction_date=base_date + timedelta(days=i),
                )
            )

        # Mock cash accounts
        cash_accounts = [Mock(balance=Decimal("5000.00"))]

        # Setup mock to return transactions first, then cash accounts
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            transactions
        )
        mock_session_instance.query.return_value.filter.return_value.all.return_value = (
            cash_accounts
        )

        decision = await enhanced_accounting_agent.process_data(data)

        assert decision is not None
        assert decision.decision_type == "cash_flow_forecast"
        assert "forecasts" in decision.context
        assert "confidence_score" in decision.context
        assert "analysis" in decision.context
        # Should have multiple forecast methods
        assert "ensemble" in decision.context["forecasts"]
        assert "simple_moving_average" in decision.context["forecasts"]
        assert "weighted_moving_average" in decision.context["forecasts"]

    @pytest.mark.asyncio
    async def test_financial_trend_analysis(self, enhanced_accounting_agent, mock_db_session):
        """Test multi-period financial trend analysis"""
        data = {"type": "trend_analysis"}

        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock transactions for different periods with declining trend
        def create_period_transactions(period_days, daily_income_base):
            transactions = []
            for i in range(period_days):
                # Declining income trend
                income_amount = daily_income_base - (i * 10)  # Decline by $10/day
                transactions.append(
                    Mock(
                        amount=Decimal(str(max(100, income_amount))),
                        transaction_type=TransactionType.INCOME,
                        transaction_date=datetime.now() - timedelta(days=period_days - i),
                        category="sales",
                    )
                )
                # Stable expenses
                transactions.append(
                    Mock(
                        amount=Decimal("300.00"),
                        transaction_type=TransactionType.EXPENSE,
                        transaction_date=datetime.now() - timedelta(days=period_days - i),
                        category="operating",
                    )
                )
            return transactions

        # Mock different period queries
        period_transactions = {
            7: create_period_transactions(7, 1000),
            30: create_period_transactions(30, 1000),
            90: create_period_transactions(90, 1000),
        }

        call_count = 0

        def mock_filter_side_effect(*args, **kwargs):
            nonlocal call_count
            result_mock = Mock()
            if call_count < 3:  # First 3 calls are for different periods
                periods = [7, 30, 90]
                result_mock.all.return_value = period_transactions[periods[call_count]]
                call_count += 1
            else:
                result_mock.all.return_value = []
            return result_mock

        mock_session_instance.query.return_value.filter.side_effect = mock_filter_side_effect

        decision = await enhanced_accounting_agent.process_data(data)

        assert decision is not None
        assert decision.decision_type == "financial_trend_analysis"
        assert "trend_analysis" in decision.context
        assert "significant_trends" in decision.context
        assert len(decision.context["significant_trends"]) > 0  # Should detect declining trends

    @pytest.mark.asyncio
    async def test_decision_outcome_processing(self, enhanced_accounting_agent):
        """Test decision outcome tracking and learning"""
        # First, add some decision outcomes
        enhanced_accounting_agent.decision_outcomes = {
            "decision_1": {
                "was_correct": True,
                "decision_type": "transaction_anomaly",
                "timestamp": datetime.now(),
            },
            "decision_2": {
                "was_correct": False,
                "decision_type": "transaction_anomaly",
                "timestamp": datetime.now(),
            },
            "decision_3": {
                "was_correct": False,
                "decision_type": "transaction_anomaly",
                "timestamp": datetime.now(),
            },
            "decision_4": {
                "was_correct": False,
                "decision_type": "cash_flow_forecast",
                "timestamp": datetime.now(),
            },
            "decision_5": {
                "was_correct": True,
                "decision_type": "cash_flow_forecast",
                "timestamp": datetime.now(),
            },
        }

        data = {
            "type": "outcome_feedback",
            "decision_id": "decision_6",
            "was_correct": False,
            "decision_type": "transaction_anomaly",
            "feedback_notes": "False positive - transaction was legitimate",
        }

        decision = await enhanced_accounting_agent.process_data(data)

        # Should create learning decision due to low accuracy in transaction_anomaly
        assert decision is not None
        assert decision.decision_type == "decision_learning"
        assert "pattern_analysis" in decision.context
        assert decision.context["pattern_analysis"]["needs_adjustment"] is True

    @pytest.mark.asyncio
    async def test_detect_transaction_anomalies_multiple_methods(
        self, enhanced_accounting_agent, mock_db_session
    ):
        """Test multi-algorithm anomaly detection"""
        transaction = TransactionModel(
            id="1",
            description="Test transaction",
            amount=Decimal("2000.00"),  # Outlier amount
            transaction_type=TransactionType.INCOME,
            category="sales",
            transaction_date=datetime.now().replace(hour=2),  # Unusual hour
        )

        # Create similar transactions with normal amounts and times
        similar_transactions = [
            Mock(
                amount=Decimal("1000.00"),
                transaction_date=datetime.now().replace(hour=9),  # Normal business hour
            )
            for _ in range(10)
        ]

        mock_session = Mock()
        result = await enhanced_accounting_agent._detect_transaction_anomalies(
            mock_session, transaction, similar_transactions
        )

        assert "is_anomaly" in result
        assert "anomaly_count" in result
        assert "statistical_outlier" in result
        assert "iqr_outlier" in result
        assert "variance_outlier" in result
        assert "time_anomaly" in result
        assert "z_score" in result
        assert "statistics" in result

        # Should detect anomaly due to amount and time
        assert result["anomaly_count"] >= 2
        assert result["is_anomaly"] is True

    @pytest.mark.asyncio
    async def test_calculate_dynamic_confidence(self, enhanced_accounting_agent, mock_db_session):
        """Test dynamic confidence calculation"""
        analysis_data = {
            "anomaly_count": 3,
            "statistics": {"sample_size": 25, "mean": 1000.0, "std_dev": 100.0},
        }

        # Mock historical accuracy
        enhanced_accounting_agent.decision_outcomes = {
            "d1": {"decision_type": "transaction_anomaly", "was_correct": True},
            "d2": {"decision_type": "transaction_anomaly", "was_correct": True},
            "d3": {"decision_type": "transaction_anomaly", "was_correct": False},
        }

        mock_session = Mock()
        result = await enhanced_accounting_agent._calculate_dynamic_confidence(
            mock_session, "transaction_anomaly", analysis_data
        )

        assert "score" in result
        assert "factors" in result
        assert 0.1 <= result["score"] <= 0.95
        assert "data_volume" in result["factors"]
        assert "historical_accuracy" in result["factors"]
        assert "trend_stability" in result["factors"]
        assert "seasonal_consistency" in result["factors"]

    @pytest.mark.asyncio
    async def test_forecast_confidence_calculation(self, enhanced_accounting_agent):
        """Test forecast confidence calculation"""
        daily_flows = {
            f"2024-01-{i:02d}": 100.0 + (i * 5) for i in range(1, 31)  # Consistent upward trend
        }

        forecasts = {
            "simple_moving_average": 250.0,
            "weighted_moving_average": 255.0,
            "trend_based": 260.0,
            "seasonal_adjusted": 250.0,
        }

        confidence = await enhanced_accounting_agent._calculate_forecast_confidence(
            daily_flows, forecasts
        )

        assert 0.2 <= confidence <= 0.9
        # Should be high confidence due to consistent trend and method agreement
        assert confidence > 0.5

    @pytest.mark.asyncio
    async def test_analyze_time_patterns(self, enhanced_accounting_agent, mock_db_session):
        """Test time-based pattern analysis"""
        transaction = TransactionModel(
            id="1",
            description="Test transaction",
            amount=Decimal("1000.00"),
            transaction_type=TransactionType.INCOME,
            category="sales",
            transaction_date=datetime.now().replace(hour=3, minute=0),  # 3 AM - unusual
        )

        # Create similar transactions mostly during business hours
        similar_transactions = []
        for hour in [9, 10, 11, 14, 15, 16] * 5:  # Business hours, repeated
            similar_transactions.append(
                Mock(transaction_date=datetime.now().replace(hour=hour, minute=0))
            )

        mock_session = Mock()
        result = await enhanced_accounting_agent._analyze_time_patterns(
            mock_session, transaction, similar_transactions
        )

        assert "is_anomaly" in result
        assert "hour_frequency" in result
        assert "weekday_frequency" in result
        assert result["is_anomaly"] is True  # 3 AM should be anomalous
        assert result["hour_frequency"] < 0.05  # Very low frequency

    @pytest.mark.asyncio
    async def test_prepare_daily_cash_flows(self, enhanced_accounting_agent):
        """Test daily cash flow data preparation"""
        transactions = []
        base_date = datetime.now().date()

        for i in range(5):
            date = base_date - timedelta(days=i)
            # Income transaction
            transactions.append(
                Mock(
                    amount=Decimal("1000.00"),
                    transaction_type=TransactionType.INCOME,
                    transaction_date=datetime.combine(date, datetime.min.time()),
                )
            )
            # Expense transaction
            transactions.append(
                Mock(
                    amount=Decimal("500.00"),
                    transaction_type=TransactionType.EXPENSE,
                    transaction_date=datetime.combine(date, datetime.min.time()),
                )
            )

        daily_flows = await enhanced_accounting_agent._prepare_daily_cash_flows(transactions)

        assert len(daily_flows) == 5
        for _date_key, flow in daily_flows.items():
            assert flow == 500.0  # 1000 income - 500 expense

    @pytest.mark.asyncio
    async def test_generate_cash_flow_forecasts(self, enhanced_accounting_agent):
        """Test cash flow forecast generation with multiple methods"""
        daily_flows = {
            f"2024-01-{i:02d}": 100.0 + (i * 2)  # Upward trend
            for i in range(1, 21)  # 20 days of data
        }

        forecasts = await enhanced_accounting_agent._generate_cash_flow_forecasts(daily_flows, 30)

        assert "simple_moving_average" in forecasts
        assert "weighted_moving_average" in forecasts
        assert "trend_based" in forecasts
        assert "seasonal_adjusted" in forecasts
        assert "ensemble" in forecasts
        assert "total_forecast" in forecasts

        # Ensemble should be different from individual methods
        assert forecasts["ensemble"] != forecasts["simple_moving_average"]
        assert forecasts["total_forecast"] == forecasts["ensemble"] * 30

    @pytest.mark.asyncio
    async def test_calculate_trend(self, enhanced_accounting_agent):
        """Test linear trend calculation"""
        # Test upward trend
        upward_values = [100, 110, 120, 130, 140]
        upward_trend = enhanced_accounting_agent._calculate_trend(upward_values)
        assert upward_trend > 0

        # Test downward trend
        downward_values = [140, 130, 120, 110, 100]
        downward_trend = enhanced_accounting_agent._calculate_trend(downward_values)
        assert downward_trend < 0

        # Test flat trend
        flat_values = [100, 100, 100, 100, 100]
        flat_trend = enhanced_accounting_agent._calculate_trend(flat_values)
        assert abs(flat_trend) < 0.1  # Should be close to zero

    @pytest.mark.asyncio
    async def test_identify_significant_trends(self, enhanced_accounting_agent):
        """Test identification of significant financial trends"""
        trend_analysis = {
            "7_days": {
                "income_trend": -60,  # Declining income
                "expense_trend": 80,  # Rising expenses
                "daily_net": -150,  # Negative net flow
            },
            "30_days": {"income_trend": 20, "expense_trend": 10, "daily_net": 50},
        }

        significant_trends = await enhanced_accounting_agent._identify_significant_trends(
            trend_analysis
        )

        assert len(significant_trends) >= 2
        assert any("Declining income" in trend for trend in significant_trends)
        assert any("Rising expense" in trend for trend in significant_trends)
        assert any("Negative cash flow" in trend for trend in significant_trends)

    @pytest.mark.asyncio
    async def test_analyze_decision_patterns(self, enhanced_accounting_agent):
        """Test decision pattern analysis for learning"""
        # Setup decision outcomes with mixed accuracy
        enhanced_accounting_agent.decision_outcomes = {
            f"decision_{i}": {
                "decision_type": "transaction_anomaly",
                "was_correct": i % 3 != 0,  # 33% accuracy - should trigger adjustment
                "timestamp": datetime.now(),
            }
            for i in range(1, 7)
        }

        # Add some cash flow decisions with better accuracy
        for i in range(7, 10):
            enhanced_accounting_agent.decision_outcomes[f"decision_{i}"] = {
                "decision_type": "cash_flow_forecast",
                "was_correct": True,
                "timestamp": datetime.now(),
            }

        pattern_analysis = await enhanced_accounting_agent._analyze_decision_patterns()

        assert pattern_analysis["needs_adjustment"] is True
        assert "transaction_anomaly" in pattern_analysis["recommendation"]
        assert len(pattern_analysis["low_accuracy_types"]) > 0
        assert pattern_analysis["total_decisions"] == 9

    @pytest.mark.asyncio
    async def test_enhanced_periodic_check_scheduling(
        self, enhanced_accounting_agent, mock_db_session
    ):
        """Test enhanced periodic check with new scheduling"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock sufficient cash to avoid alerts
        cash_accounts = [Mock(balance=Decimal("5000.00"))]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = (
            cash_accounts
        )
        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            []
        )

        # Test different hours and days
        with patch("agents.accounting_agent.datetime") as mock_datetime:
            # Test 9 AM on Monday (should trigger all analyses)
            mock_datetime.now.return_value = Mock(hour=9, weekday=lambda: 0)
            mock_datetime.now.return_value.date.return_value = date.today()

            await enhanced_accounting_agent.periodic_check()

            # Should complete without errors
            assert True

            # Test 3 PM on Tuesday (should trigger cash flow forecast only)
            mock_datetime.now.return_value = Mock(hour=15, weekday=lambda: 1)

            await enhanced_accounting_agent.periodic_check()

            # Should complete without errors
            assert True

    def test_configuration_validation(self, enhanced_accounting_agent):
        """Test that enhanced configuration is properly loaded"""
        config = enhanced_accounting_agent.forecasting_config

        assert "prediction_days" in config
        assert "seasonal_analysis_days" in config
        assert "trend_analysis_periods" in config
        assert "confidence_factors" in config

        confidence_factors = config["confidence_factors"]
        assert "data_volume" in confidence_factors
        assert "historical_accuracy" in confidence_factors
        assert "trend_stability" in confidence_factors
        assert "seasonal_consistency" in confidence_factors

        # Check that confidence factors sum to reasonable total
        total_factors = sum(confidence_factors.values())
        assert 0.8 <= total_factors <= 1.2  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_error_handling_in_enhanced_methods(
        self, enhanced_accounting_agent, mock_db_session
    ):
        """Test error handling in new enhanced methods"""
        mock_session = Mock()

        # Test anomaly detection with empty data
        empty_result = await enhanced_accounting_agent._detect_transaction_anomalies(
            mock_session,
            TransactionModel(
                id="1",
                description="test",
                amount=Decimal("100"),
                transaction_type=TransactionType.INCOME,
                transaction_date=datetime.now(),
            ),
            [],
        )
        assert empty_result["is_anomaly"] is False

        # Test confidence calculation with invalid data
        invalid_confidence = await enhanced_accounting_agent._calculate_dynamic_confidence(
            mock_session, "test_type", {}
        )
        assert "score" in invalid_confidence
        assert 0.1 <= invalid_confidence["score"] <= 0.95

        # Test trend calculation with insufficient data
        trend = enhanced_accounting_agent._calculate_trend([])
        assert trend == 0

        trend_single = enhanced_accounting_agent._calculate_trend([100])
        assert trend_single == 0

    @pytest.mark.asyncio
    async def test_forecast_with_insufficient_data(
        self, enhanced_accounting_agent, mock_db_session
    ):
        """Test cash flow forecasting behavior with insufficient data"""
        data = {"type": "cash_flow_forecast", "forecast_days": 30}

        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock insufficient transactions (less than 7 days)
        few_transactions = [
            Mock(
                amount=Decimal("1000.00"),
                transaction_type=TransactionType.INCOME,
                transaction_date=datetime.now() - timedelta(days=i),
            )
            for i in range(3)  # Only 3 days of data
        ]

        mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
            few_transactions
        )

        decision = await enhanced_accounting_agent.process_data(data)

        # Should return None due to insufficient data
        assert decision is None

    @pytest.mark.asyncio
    async def test_trend_analysis_with_no_significant_trends(
        self, enhanced_accounting_agent, mock_db_session
    ):
        """Test trend analysis when no significant trends are detected"""
        data = {"type": "trend_analysis"}

        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance

        # Mock stable transactions with no significant trends
        def create_stable_transactions(period_days):
            transactions = []
            for i in range(period_days):
                transactions.append(
                    Mock(
                        amount=Decimal("1000.00"),  # Stable income
                        transaction_type=TransactionType.INCOME,
                        transaction_date=datetime.now() - timedelta(days=period_days - i),
                        category="sales",
                    )
                )
                transactions.append(
                    Mock(
                        amount=Decimal("300.00"),  # Stable expenses
                        transaction_type=TransactionType.EXPENSE,
                        transaction_date=datetime.now() - timedelta(days=period_days - i),
                        category="operating",
                    )
                )
            return transactions

        # Mock queries for different periods
        call_count = 0

        def mock_filter_side_effect(*args, **kwargs):
            nonlocal call_count
            result_mock = Mock()
            if call_count < 3:
                periods = [7, 30, 90]
                result_mock.all.return_value = create_stable_transactions(periods[call_count])
                call_count += 1
            else:
                result_mock.all.return_value = []
            return result_mock

        mock_session_instance.query.return_value.filter.side_effect = mock_filter_side_effect

        decision = await enhanced_accounting_agent.process_data(data)

        # Should return None when no significant trends detected
        assert decision is None
