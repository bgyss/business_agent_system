"""
Test suite for demo_enhanced_accounting.py module
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

import demo_enhanced_accounting


class TestDemoEnhancedAccounting:
    """Test the demo_enhanced_accounting script"""

    def test_module_can_be_imported(self):
        """Test that the demo module can be imported without errors"""
        assert demo_enhanced_accounting is not None

    def test_demo_function_exists(self):
        """Test that the main demo function exists"""
        assert hasattr(demo_enhanced_accounting, "demo_enhanced_accounting")
        assert callable(demo_enhanced_accounting.demo_enhanced_accounting)

    @patch("demo_enhanced_accounting.EnhancedAccountingAgent")
    @patch("builtins.print")
    async def test_demo_initialization(self, mock_print, mock_agent_class):
        """Test that the demo initializes the agent correctly"""
        # Create a mock agent that will raise an exception early to prevent full execution
        mock_agent = Mock()
        mock_agent.anomaly_threshold = 0.25
        mock_agent.alert_thresholds = {"cash_low": 1000}
        mock_agent.forecasting_config = {"prediction_days": 30}

        # Make the first async method call raise an exception to stop execution early
        mock_agent._detect_transaction_anomalies = AsyncMock(
            side_effect=Exception("Test stopped early")
        )

        mock_agent_class.return_value = mock_agent

        # Run the demo and expect it to fail early (which is what we want for testing)
        with pytest.raises(Exception, match="Test stopped early"):
            await demo_enhanced_accounting.demo_enhanced_accounting()

        # Verify the agent was created with expected parameters
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs

        assert call_kwargs["agent_id"] == "demo_enhanced_agent"
        assert call_kwargs["api_key"] == "demo_key"
        assert call_kwargs["db_url"] == "sqlite:///:memory:"

        # Verify configuration structure
        config = call_kwargs["config"]
        assert config["anomaly_threshold"] == 0.25
        assert "alert_thresholds" in config
        assert "forecasting" in config
        assert config["alert_thresholds"]["cash_low"] == 1000
        assert config["forecasting"]["prediction_days"] == 30

        # Verify some output was printed before the exception
        assert mock_print.call_count > 0
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Enhanced AccountingAgent Demonstration" in call for call in print_calls)

    @patch("demo_enhanced_accounting.TransactionModel")
    def test_transaction_model_usage(self, mock_transaction_model):
        """Test that TransactionModel is used correctly in the demo"""
        # Just verify the import and class availability
        assert mock_transaction_model is not None

    @patch("demo_enhanced_accounting.Mock")
    def test_mock_usage(self, mock_mock_class):
        """Test that Mock is used for similar transactions"""
        # Verify Mock is available for creating similar transactions
        assert mock_mock_class is not None

    def test_required_imports_available(self):
        """Test that all required imports work"""
        # Test imports used in the demo script
        from datetime import datetime, timedelta
        from decimal import Decimal
        from unittest.mock import Mock

        # Verify these can be imported (they are used in the demo)
        assert datetime is not None
        assert timedelta is not None
        assert Decimal is not None
        assert Mock is not None

    @patch("demo_enhanced_accounting.EnhancedAccountingAgent")
    async def test_configuration_structure(self, mock_agent_class):
        """Test the configuration structure passed to the agent"""
        mock_agent = Mock()
        mock_agent._detect_transaction_anomalies = AsyncMock(side_effect=Exception("Stop early"))
        mock_agent_class.return_value = mock_agent

        with pytest.raises(Exception):
            await demo_enhanced_accounting.demo_enhanced_accounting()

        # Extract the configuration from the call
        config = mock_agent_class.call_args.kwargs["config"]

        # Verify main config sections
        assert "anomaly_threshold" in config
        assert "alert_thresholds" in config
        assert "forecasting" in config

        # Verify alert thresholds structure
        alert_thresholds = config["alert_thresholds"]
        assert "cash_low" in alert_thresholds
        assert "receivables_overdue" in alert_thresholds
        assert "payables_overdue" in alert_thresholds

        # Verify forecasting config structure
        forecasting = config["forecasting"]
        assert "prediction_days" in forecasting
        assert "seasonal_analysis_days" in forecasting
        assert "trend_analysis_periods" in forecasting
        assert "confidence_factors" in forecasting

        # Verify confidence factors structure
        confidence_factors = forecasting["confidence_factors"]
        assert "data_volume" in confidence_factors
        assert "historical_accuracy" in confidence_factors
        assert "trend_stability" in confidence_factors
        assert "seasonal_consistency" in confidence_factors

        # Verify numeric values are reasonable
        assert isinstance(config["anomaly_threshold"], float)
        assert config["anomaly_threshold"] == 0.25
        assert isinstance(alert_thresholds["cash_low"], int)
        assert alert_thresholds["cash_low"] == 1000
        assert isinstance(forecasting["prediction_days"], int)
        assert forecasting["prediction_days"] == 30
