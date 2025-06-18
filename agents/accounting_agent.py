import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from agents.base_agent import BaseAgent, AgentDecision
from models.financial import (
    Account, Transaction, AccountsReceivable, AccountsPayable,
    TransactionModel, FinancialSummary, CashFlowStatement,
    TransactionType, AccountType
)


class AccountingAgent(BaseAgent):
    def __init__(self, agent_id: str, api_key: str, config: Dict[str, Any], db_url: str):
        super().__init__(agent_id, api_key, config, db_url)
        # Remove duplicate engine and session creation since BaseAgent now handles this
        self.anomaly_threshold = config.get("anomaly_threshold", 0.2)  # 20% variance
        self.alert_thresholds = config.get("alert_thresholds", {
            "cash_low": 1000,
            "receivables_overdue": 30,  # days
            "payables_overdue": 7  # days
        })
    
    @property
    def system_prompt(self) -> str:
        return """You are an AI Accounting Agent responsible for monitoring financial transactions and providing insights.
        
        Your responsibilities include:
        1. Analyzing financial transactions for unusual patterns or anomalies
        2. Monitoring cash flow and providing alerts for low cash situations
        3. Tracking accounts receivable and payable aging
        4. Identifying cost-saving opportunities
        5. Generating financial summaries and reports
        6. Flagging potential fraud or errors in transactions
        
        You should provide clear, actionable recommendations based on financial data analysis.
        Always consider the business context and provide reasoning for your decisions.
        
        When analyzing transactions, look for:
        - Unusual transaction amounts (significantly higher or lower than typical)
        - Transactions outside normal business hours
        - Duplicate transactions
        - Missing documentation or reference numbers
        - Cash flow issues
        - Aging receivables and payables
        """
    
    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        session = self.SessionLocal()
        try:
            if data.get("type") == "new_transaction":
                return await self._analyze_transaction(session, data["transaction"])
            elif data.get("type") == "daily_analysis":
                return await self._perform_daily_analysis(session)
            elif data.get("type") == "cash_flow_check":
                return await self._check_cash_flow(session)
            elif data.get("type") == "aging_analysis":
                return await self._analyze_aging(session)
        finally:
            session.close()
        
        return None
    
    async def _analyze_transaction(self, session, transaction_data: Dict[str, Any]) -> Optional[AgentDecision]:
        transaction = TransactionModel(**transaction_data)
        
        # Get recent similar transactions for comparison
        similar_transactions = session.query(Transaction).filter(
            and_(
                Transaction.transaction_type == transaction.transaction_type,
                Transaction.category == transaction.category,
                Transaction.transaction_date >= datetime.now() - timedelta(days=30)
            )
        ).all()
        
        if not similar_transactions:
            return None
        
        # Calculate average amount for similar transactions
        avg_amount = sum(t.amount for t in similar_transactions) / len(similar_transactions)
        variance = abs(float(transaction.amount) - float(avg_amount)) / float(avg_amount)
        
        context = {
            "transaction": transaction.dict(),
            "similar_count": len(similar_transactions),
            "average_amount": float(avg_amount),
            "variance": variance,
            "threshold": self.anomaly_threshold
        }
        
        if variance > self.anomaly_threshold:
            reasoning = await self.analyze_with_claude(
                f"Analyze this transaction anomaly. Transaction amount: ${transaction.amount}, "
                f"Average for similar transactions: ${avg_amount:.2f}, "
                f"Variance: {variance:.2%}. Should this be flagged?",
                context
            )
            
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="transaction_anomaly",
                context=context,
                reasoning=reasoning,
                action=f"Flag transaction {transaction.id} for review",
                confidence=min(0.9, variance * 2)
            )
        
        return None
    
    async def _perform_daily_analysis(self, session) -> Optional[AgentDecision]:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # Get yesterday's transactions
        daily_transactions = session.query(Transaction).filter(
            func.date(Transaction.transaction_date) == yesterday
        ).all()
        
        if not daily_transactions:
            return None
        
        total_income = sum(t.amount for t in daily_transactions if t.transaction_type == TransactionType.INCOME)
        total_expenses = sum(t.amount for t in daily_transactions if t.transaction_type == TransactionType.EXPENSE)
        net_flow = total_income - total_expenses
        
        # Get last 30 days average for comparison
        thirty_days_ago = today - timedelta(days=30)
        historical_transactions = session.query(Transaction).filter(
            and_(
                Transaction.transaction_date >= thirty_days_ago,
                Transaction.transaction_date < yesterday
            )
        ).all()
        
        context = {
            "date": str(yesterday),
            "transaction_count": len(daily_transactions),
            "total_income": float(total_income),
            "total_expenses": float(total_expenses),
            "net_flow": float(net_flow),
            "historical_avg_income": 0,
            "historical_avg_expenses": 0
        }
        
        if historical_transactions:
            hist_income = sum(t.amount for t in historical_transactions if t.transaction_type == TransactionType.INCOME)
            hist_expenses = sum(t.amount for t in historical_transactions if t.transaction_type == TransactionType.EXPENSE)
            hist_days = len(set(t.transaction_date.date() for t in historical_transactions))
            
            if hist_days > 0:
                context["historical_avg_income"] = float(hist_income / hist_days)
                context["historical_avg_expenses"] = float(hist_expenses / hist_days)
        
        analysis = await self.analyze_with_claude(
            f"Analyze yesterday's financial performance. "
            f"Income: ${total_income}, Expenses: ${total_expenses}, Net: ${net_flow}. "
            f"Provide insights and recommendations.",
            context
        )
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="daily_financial_analysis",
            context=context,
            reasoning=analysis,
            action="Generate daily financial report",
            confidence=0.8
        )
    
    async def _check_cash_flow(self, session) -> Optional[AgentDecision]:
        # Get current cash balances
        cash_accounts = session.query(Account).filter(
            Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS])
        ).all()
        
        total_cash = sum(account.balance for account in cash_accounts)
        low_cash_threshold = self.alert_thresholds["cash_low"]
        
        context = {
            "total_cash": float(total_cash),
            "threshold": low_cash_threshold,
            "accounts": [{"name": acc.name, "balance": float(acc.balance)} for acc in cash_accounts]
        }
        
        if total_cash < low_cash_threshold:
            reasoning = await self.analyze_with_claude(
                f"Cash balance is low: ${total_cash}. Threshold: ${low_cash_threshold}. "
                f"What actions should be taken?",
                context
            )
            
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="low_cash_alert",
                context=context,
                reasoning=reasoning,
                action="Alert management of low cash situation",
                confidence=0.9
            )
        
        return None
    
    async def _analyze_aging(self, session) -> Optional[AgentDecision]:
        today = datetime.now().date()
        overdue_receivables_threshold = self.alert_thresholds["receivables_overdue"]
        overdue_payables_threshold = self.alert_thresholds["payables_overdue"]
        
        # Check overdue receivables
        overdue_receivables = session.query(AccountsReceivable).filter(
            and_(
                AccountsReceivable.due_date < today - timedelta(days=overdue_receivables_threshold),
                AccountsReceivable.status == "unpaid"
            )
        ).all()
        
        # Check overdue payables
        overdue_payables = session.query(AccountsPayable).filter(
            and_(
                AccountsPayable.due_date < today - timedelta(days=overdue_payables_threshold),
                AccountsPayable.status == "unpaid"
            )
        ).all()
        
        if not overdue_receivables and not overdue_payables:
            return None
        
        context = {
            "overdue_receivables": [
                {
                    "customer": ar.customer_name,
                    "amount": float(ar.amount),
                    "days_overdue": (today - ar.due_date).days,
                    "invoice_number": ar.invoice_number
                }
                for ar in overdue_receivables
            ],
            "overdue_payables": [
                {
                    "vendor": ap.vendor_name,
                    "amount": float(ap.amount),
                    "days_overdue": (today - ap.due_date).days,
                    "invoice_number": ap.invoice_number
                }
                for ap in overdue_payables
            ],
            "total_overdue_receivables": float(sum(ar.amount for ar in overdue_receivables)),
            "total_overdue_payables": float(sum(ap.amount for ap in overdue_payables))
        }
        
        analysis = await self.analyze_with_claude(
            f"Aging analysis shows {len(overdue_receivables)} overdue receivables "
            f"totaling ${context['total_overdue_receivables']} and "
            f"{len(overdue_payables)} overdue payables totaling ${context['total_overdue_payables']}. "
            f"Provide collection and payment priority recommendations.",
            context
        )
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="aging_analysis",
            context=context,
            reasoning=analysis,
            action="Generate aging report and collection recommendations",
            confidence=0.85
        )
    
    async def generate_report(self) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            # Generate comprehensive financial summary
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            transactions = session.query(Transaction).filter(
                Transaction.transaction_date >= start_date
            ).all()
            
            total_revenue = sum(t.amount for t in transactions if t.transaction_type == TransactionType.INCOME)
            total_expenses = sum(t.amount for t in transactions if t.transaction_type == TransactionType.EXPENSE)
            
            cash_accounts = session.query(Account).filter(
                Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS])
            ).all()
            cash_balance = sum(acc.balance for acc in cash_accounts)
            
            receivables = session.query(AccountsReceivable).filter(
                AccountsReceivable.status == "unpaid"
            ).all()
            total_receivables = sum(ar.amount for ar in receivables)
            
            payables = session.query(AccountsPayable).filter(
                AccountsPayable.status == "unpaid"
            ).all()
            total_payables = sum(ap.amount for ap in payables)
            
            summary = FinancialSummary(
                total_revenue=total_revenue,
                total_expenses=total_expenses,
                net_income=total_revenue - total_expenses,
                cash_balance=cash_balance,
                accounts_receivable=total_receivables,
                accounts_payable=total_payables,
                period_start=start_date,
                period_end=end_date,
                transaction_count=len(transactions)
            )
            
            return {
                "summary": summary.dict(),
                "recent_decisions": [d.dict() for d in self.get_decision_history(10)],
                "alerts": await self._get_current_alerts(session)
            }
        finally:
            session.close()
    
    async def _get_current_alerts(self, session) -> List[Dict[str, Any]]:
        alerts = []
        
        # Check cash levels
        cash_accounts = session.query(Account).filter(
            Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS])
        ).all()
        total_cash = sum(acc.balance for acc in cash_accounts)
        
        if total_cash < self.alert_thresholds["cash_low"]:
            alerts.append({
                "type": "low_cash",
                "severity": "high",
                "message": f"Cash balance is low: ${total_cash}",
                "action_required": True
            })
        
        # Check overdue items
        today = datetime.now().date()
        overdue_receivables = session.query(AccountsReceivable).filter(
            and_(
                AccountsReceivable.due_date < today,
                AccountsReceivable.status == "unpaid"
            )
        ).count()
        
        if overdue_receivables > 0:
            alerts.append({
                "type": "overdue_receivables",
                "severity": "medium",
                "message": f"{overdue_receivables} overdue invoices need collection",
                "action_required": True
            })
        
        return alerts
    
    async def periodic_check(self):
        """Perform periodic accounting analysis"""
        session = self.SessionLocal()
        try:
            # Perform cash flow check
            decision = await self._check_cash_flow(session)
            if decision:
                self.log_decision(decision)
            
            # Perform aging analysis once per day
            current_hour = datetime.now().hour
            if current_hour == 9:  # 9 AM daily aging analysis
                aging_decision = await self._analyze_aging(session)
                if aging_decision:
                    self.log_decision(aging_decision)
            
            self.logger.debug("Periodic accounting check completed")
            
        except Exception as e:
            self.logger.error(f"Error in periodic check: {e}")
        finally:
            session.close()