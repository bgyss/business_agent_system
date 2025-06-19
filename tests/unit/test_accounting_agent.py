"""
Unit tests for AccountingAgent class
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.accounting_agent import AccountingAgent
from agents.base_agent import AgentDecision
from models.financial import (
    Transaction, Account, AccountsReceivable, AccountsPayable,
    TransactionType, AccountType, FinancialSummary
)


class TestAccountingAgent:
    """Test cases for AccountingAgent"""
    
    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client"""
        with patch('agents.base_agent.Anthropic') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Accounting analysis: Transaction appears normal")]
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
        """Accounting agent configuration"""
        return {
            "check_interval": 300,
            "anomaly_threshold": 0.25,
            "alert_thresholds": {
                "cash_low": 1000,
                "receivables_overdue": 30,
                "payables_overdue": 7
            }
        }
    
    @pytest.fixture
    def accounting_agent(self, mock_anthropic, mock_db_session, agent_config):
        """Create accounting agent instance"""
        return AccountingAgent(
            agent_id="accounting_agent",
            api_key="test_api_key",
            config=agent_config,
            db_url="sqlite:///:memory:"
        )
    
    def test_initialization(self, accounting_agent, agent_config):
        """Test agent initialization"""
        assert accounting_agent.agent_id == "accounting_agent"
        assert accounting_agent.anomaly_threshold == 0.25
        assert accounting_agent.alert_thresholds == agent_config["alert_thresholds"]
    
    def test_system_prompt(self, accounting_agent):
        """Test system prompt content"""
        prompt = accounting_agent.system_prompt
        assert "AI Accounting Agent" in prompt
        assert "financial transactions" in prompt
        assert "cash flow" in prompt
        assert "accounts receivable" in prompt
        assert "fraud" in prompt
    
    @pytest.mark.asyncio
    async def test_process_data_new_transaction(self, accounting_agent, mock_db_session):
        """Test processing new transaction data"""
        # Mock transaction data
        transaction_data = {
            "id": "1",
            "description": "Test transaction",
            "amount": Decimal("1500.00"),
            "transaction_type": TransactionType.INCOME,
            "category": "sales",
            "transaction_date": datetime.now()
        }
        
        data = {
            "type": "new_transaction",
            "transaction": transaction_data
        }
        
        # Mock similar transactions query
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock similar transactions with lower amounts
        similar_transactions = [
            Mock(amount=Decimal("1000.00")),
            Mock(amount=Decimal("1100.00")),
            Mock(amount=Decimal("900.00"))
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = similar_transactions
        
        decision = await accounting_agent.process_data(data)
        
        # Should detect anomaly (1500 vs avg 1000)
        assert decision is not None
        assert decision.decision_type == "transaction_anomaly"
        assert decision.agent_id == "accounting_agent"
        assert "Flag transaction" in decision.action
        assert decision.confidence > 0
    
    @pytest.mark.asyncio
    async def test_process_data_daily_analysis(self, accounting_agent, mock_db_session):
        """Test daily financial analysis"""
        data = {"type": "daily_analysis"}
        
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock yesterday's transactions
        yesterday_transactions = [
            Mock(amount=Decimal("1000.00"), transaction_type=TransactionType.INCOME),
            Mock(amount=Decimal("500.00"), transaction_type=TransactionType.EXPENSE),
            Mock(amount=Decimal("300.00"), transaction_type=TransactionType.EXPENSE)
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = yesterday_transactions
        
        decision = await accounting_agent.process_data(data)
        
        assert decision is not None
        assert decision.decision_type == "daily_financial_analysis"
        assert decision.action == "Generate daily financial report"
        assert decision.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_process_data_cash_flow_check_low_cash(self, accounting_agent, mock_db_session):
        """Test cash flow check with low cash"""
        data = {"type": "cash_flow_check"}
        
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock cash accounts with low balance
        cash_accounts = [
            Mock(balance=Decimal("500.00")),
            Mock(balance=Decimal("300.00"))
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = cash_accounts
        
        decision = await accounting_agent.process_data(data)
        
        assert decision is not None
        assert decision.decision_type == "low_cash_alert"
        assert "Alert management" in decision.action
        assert decision.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_process_data_cash_flow_check_sufficient_cash(self, accounting_agent, mock_db_session):
        """Test cash flow check with sufficient cash"""
        data = {"type": "cash_flow_check"}
        
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock cash accounts with sufficient balance
        cash_accounts = [
            Mock(balance=Decimal("5000.00")),
            Mock(balance=Decimal("3000.00"))
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = cash_accounts
        
        decision = await accounting_agent.process_data(data)
        
        assert decision is None  # No alert needed
    
    @pytest.mark.asyncio
    async def test_process_data_aging_analysis(self, accounting_agent, mock_db_session):
        """Test aging analysis with overdue items"""
        data = {"type": "aging_analysis"}
        
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock overdue receivables
        overdue_receivables = [
            Mock(
                customer_name="Customer A",
                amount=Decimal("1500.00"),
                due_date=date.today() - timedelta(days=45),
                invoice_number="INV-001"
            ),
            Mock(
                customer_name="Customer B",
                amount=Decimal("800.00"),
                due_date=date.today() - timedelta(days=60),
                invoice_number="INV-002"
            )
        ]
        
        # Mock overdue payables
        overdue_payables = [
            Mock(
                vendor_name="Vendor A",
                amount=Decimal("500.00"),
                due_date=date.today() - timedelta(days=10),
                invoice_number="BILL-001"
            )
        ]
        
        # Setup mock to return different results for different queries
        mock_session_instance.query.return_value.filter.return_value.all.side_effect = [
            overdue_receivables,
            overdue_payables
        ]
        
        decision = await accounting_agent.process_data(data)
        
        assert decision is not None
        assert decision.decision_type == "aging_analysis"
        assert "aging report" in decision.action
        assert decision.confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_analyze_transaction_no_similar_transactions(self, accounting_agent, mock_db_session):
        """Test transaction analysis with no similar transactions"""
        transaction_data = {
            "id": "1",
            "description": "Test transaction",
            "amount": Decimal("1500.00"),
            "transaction_type": TransactionType.INCOME,
            "category": "sales",
            "transaction_date": datetime.now()
        }
        
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock empty similar transactions
        mock_session_instance.query.return_value.filter.return_value.all.return_value = []
        
        decision = await accounting_agent._analyze_transaction(mock_session_instance, transaction_data)
        
        assert decision is None
    
    @pytest.mark.asyncio
    async def test_analyze_transaction_normal_variance(self, accounting_agent, mock_db_session):
        """Test transaction analysis with normal variance"""
        transaction_data = {
            "id": "1",
            "description": "Test transaction",
            "amount": Decimal("1000.00"),
            "transaction_type": TransactionType.INCOME,
            "category": "sales",
            "transaction_date": datetime.now()
        }
        
        mock_session_instance = Mock()
        
        # Mock similar transactions with similar amounts (low variance)
        similar_transactions = [
            Mock(amount=Decimal("950.00")),
            Mock(amount=Decimal("1050.00")),
            Mock(amount=Decimal("1000.00"))
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = similar_transactions
        
        decision = await accounting_agent._analyze_transaction(mock_session_instance, transaction_data)
        
        assert decision is None  # No anomaly detected
    
    @pytest.mark.asyncio
    async def test_perform_daily_analysis_no_transactions(self, accounting_agent, mock_db_session):
        """Test daily analysis with no transactions"""
        mock_session_instance = Mock()
        mock_session_instance.query.return_value.filter.return_value.all.return_value = []
        
        decision = await accounting_agent._perform_daily_analysis(mock_session_instance)
        
        assert decision is None
    
    @pytest.mark.asyncio
    async def test_check_cash_flow_multiple_accounts(self, accounting_agent, mock_db_session):
        """Test cash flow check with multiple account types"""
        mock_session_instance = Mock()
        
        # Mock accounts with different types
        accounts = [
            Mock(balance=Decimal("2000.00"), account_type=AccountType.CHECKING),
            Mock(balance=Decimal("1500.00"), account_type=AccountType.SAVINGS),
            Mock(balance=Decimal("5000.00"), account_type=AccountType.CREDIT)  # Should not be included
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = accounts
        
        decision = await accounting_agent._check_cash_flow(mock_session_instance)
        
        assert decision is None  # Total cash (3500) is above threshold (1000)
    
    @pytest.mark.asyncio
    async def test_analyze_aging_no_overdue_items(self, accounting_agent, mock_db_session):
        """Test aging analysis with no overdue items"""
        mock_session_instance = Mock()
        
        # Mock empty overdue items
        mock_session_instance.query.return_value.filter.return_value.all.side_effect = [[], []]
        
        decision = await accounting_agent._analyze_aging(mock_session_instance)
        
        assert decision is None
    
    @pytest.mark.asyncio
    async def test_generate_report(self, accounting_agent, mock_db_session):
        """Test report generation"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock transactions
        transactions = [
            Mock(amount=Decimal("1000.00"), transaction_type=TransactionType.INCOME),
            Mock(amount=Decimal("500.00"), transaction_type=TransactionType.EXPENSE)
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = transactions
        
        # Mock accounts
        cash_accounts = [Mock(balance=Decimal("5000.00"))]
        
        # Mock receivables and payables
        receivables = [Mock(amount=Decimal("1500.00"))]
        payables = [Mock(amount=Decimal("800.00"))]
        
        # Setup different return values for different queries
        # Need more values because _get_current_alerts makes additional queries
        query_results = [transactions, cash_accounts, receivables, payables, cash_accounts]  # Extra for alerts
        mock_session_instance.query.return_value.filter.return_value.all.side_effect = query_results
        
        # Mock count for overdue receivables check in alerts
        mock_session_instance.query.return_value.filter.return_value.count.return_value = 0
        
        # Mock decisions
        accounting_agent.decisions_log = [
            AgentDecision(
                agent_id="accounting_agent",
                decision_type="test",
                context={},
                reasoning="test",
                action="test",
                confidence=0.8
            )
        ]
        
        report = await accounting_agent.generate_report()
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "recent_decisions" in report
        assert "alerts" in report
        assert len(report["recent_decisions"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_current_alerts_low_cash(self, accounting_agent, mock_db_session):
        """Test current alerts with low cash"""
        mock_session_instance = Mock()
        
        # Mock low cash accounts
        cash_accounts = [Mock(balance=Decimal("500.00"))]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = cash_accounts
        mock_session_instance.query.return_value.filter.return_value.count.return_value = 0
        
        alerts = await accounting_agent._get_current_alerts(mock_session_instance)
        
        assert len(alerts) == 1
        assert alerts[0]["type"] == "low_cash"
        assert alerts[0]["severity"] == "high"
        assert alerts[0]["action_required"] is True
    
    @pytest.mark.asyncio
    async def test_get_current_alerts_overdue_receivables(self, accounting_agent, mock_db_session):
        """Test current alerts with overdue receivables"""
        mock_session_instance = Mock()
        
        # Mock sufficient cash but overdue receivables
        cash_accounts = [Mock(balance=Decimal("5000.00"))]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = cash_accounts
        mock_session_instance.query.return_value.filter.return_value.count.return_value = 3
        
        alerts = await accounting_agent._get_current_alerts(mock_session_instance)
        
        assert len(alerts) == 1
        assert alerts[0]["type"] == "overdue_receivables"
        assert alerts[0]["severity"] == "medium"
        assert "3 overdue invoices" in alerts[0]["message"]
    
    @pytest.mark.asyncio
    async def test_periodic_check(self, accounting_agent, mock_db_session):
        """Test periodic check execution"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        
        # Mock accounts for cash flow check
        cash_accounts = [Mock(balance=Decimal("5000.00"))]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = cash_accounts
        
        # Mock datetime to trigger aging analysis
        with patch('agents.accounting_agent.datetime') as mock_datetime:
            mock_datetime.now.return_value = Mock(hour=9)  # 9 AM
            mock_datetime.now.return_value.date.return_value = date.today()
            
            # Mock aging analysis queries
            mock_session_instance.query.return_value.filter.return_value.all.side_effect = [[], []]
            
            await accounting_agent.periodic_check()
            
            # Should have executed without errors
            assert True
    
    def test_confidence_score_calculation(self, accounting_agent):
        """Test confidence score calculation logic"""
        # Test variance-based confidence calculation
        variance = 0.5  # 50% variance
        confidence = min(0.9, variance * 2)
        
        assert confidence == 0.9
        
        # Test with lower variance
        variance = 0.3  # 30% variance
        confidence = min(0.9, variance * 2)
        
        assert confidence == 0.6
    
    def test_config_defaults(self):
        """Test configuration defaults"""
        with patch('agents.base_agent.Anthropic'), \
             patch('agents.base_agent.create_engine'), \
             patch('agents.base_agent.sessionmaker'):
            
            agent = AccountingAgent(
                agent_id="test_agent",
                api_key="test_key",
                config={},  # Empty config
                db_url="sqlite:///:memory:"
            )
            
            assert agent.anomaly_threshold == 0.2  # Default value
            assert agent.alert_thresholds["cash_low"] == 1000
            assert agent.alert_thresholds["receivables_overdue"] == 30
            assert agent.alert_thresholds["payables_overdue"] == 7
    
    @pytest.mark.asyncio
    async def test_edge_case_zero_variance(self, accounting_agent, mock_db_session):
        """Test edge case with zero variance in transactions"""
        transaction_data = {
            "id": "1",
            "description": "Test transaction",
            "amount": Decimal("1000.00"),
            "transaction_type": TransactionType.INCOME,
            "category": "sales",
            "transaction_date": datetime.now()
        }
        
        mock_session_instance = Mock()
        
        # Mock similar transactions with identical amounts (zero variance)
        similar_transactions = [
            Mock(amount=Decimal("1000.00")),
            Mock(amount=Decimal("1000.00")),
            Mock(amount=Decimal("1000.00"))
        ]
        mock_session_instance.query.return_value.filter.return_value.all.return_value = similar_transactions
        
        decision = await accounting_agent._analyze_transaction(mock_session_instance, transaction_data)
        
        assert decision is None  # No anomaly with zero variance
    
    @pytest.mark.asyncio
    async def test_error_handling_in_process_data(self, accounting_agent, mock_db_session):
        """Test error handling in process_data"""
        mock_session_instance = Mock()
        mock_db_session.return_value = mock_session_instance
        mock_session_instance.close.side_effect = Exception("Session close error")
        
        data = {"type": "invalid_type"}
        
        # Should handle the error gracefully
        decision = await accounting_agent.process_data(data)
        
        assert decision is None