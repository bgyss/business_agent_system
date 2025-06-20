import asyncio
import json
import math
import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Union

from sqlalchemy import create_engine, and_, func, desc, asc
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
        
        # Enhanced configuration for advanced analytics
        self.forecasting_config = config.get("forecasting", {
            "prediction_days": 30,
            "seasonal_analysis_days": 365,
            "trend_analysis_periods": 7,
            "confidence_factors": {
                "data_volume": 0.3,
                "historical_accuracy": 0.25,
                "trend_stability": 0.25,
                "seasonal_consistency": 0.2
            }
        })
        
        # Decision outcome tracking for learning
        self.decision_outcomes = {}  # Track decision outcomes for learning
        self.forecasting_accuracy_history = []  # Track forecasting accuracy over time
    
    @property
    def system_prompt(self) -> str:
        return """You are an advanced AI Accounting Agent with sophisticated financial analysis capabilities.
        
        Your enhanced responsibilities include:
        1. Advanced anomaly detection using multiple statistical methods (Z-score, IQR, variance analysis, time patterns)
        2. Predictive cash flow forecasting using ensemble methods and trend analysis
        3. Multi-period financial trend analysis with confidence scoring
        4. Dynamic confidence calculation based on data quality and historical accuracy
        5. Decision outcome tracking and continuous learning from feedback
        6. Traditional accounting monitoring (receivables, payables, cash flow alerts)
        
        Advanced capabilities:
        - Multi-algorithm anomaly detection combining statistical, time-based, and pattern analysis
        - Cash flow forecasting using moving averages, trend projection, and seasonal adjustments
        - Trend analysis across weekly, monthly, and quarterly periods
        - Confidence scoring based on data volume, historical accuracy, and analysis consistency
        - Learning from decision outcomes to improve future accuracy
        
        You should provide clear, actionable recommendations with confidence levels based on:
        - Quality and volume of available data
        - Historical accuracy of similar decisions
        - Consistency across multiple analysis methods
        - Seasonal and temporal patterns
        
        When analyzing financial data, consider:
        - Statistical significance of patterns and anomalies
        - Predictive indicators for cash flow challenges
        - Trending patterns that may indicate systemic issues
        - Time-based patterns that suggest operational changes
        - Historical context and seasonal variations
        - Confidence levels to guide decision urgency
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
            elif data.get("type") == "cash_flow_forecast":
                return await self._forecast_cash_flow(session, data.get("forecast_days", 30))
            elif data.get("type") == "trend_analysis":
                return await self._analyze_financial_trends(session)
            elif data.get("type") == "outcome_feedback":
                return await self._process_decision_outcome(data)
        except Exception as e:
            self.logger.error(f"Error in process_data: {e}")
            return None
        finally:
            try:
                session.close()
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")
        
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
        
        # Enhanced anomaly detection with multiple algorithms
        anomaly_results = await self._detect_transaction_anomalies(
            session, transaction, similar_transactions
        )
        
        if anomaly_results["is_anomaly"]:
            # Calculate dynamic confidence score
            confidence = await self._calculate_dynamic_confidence(
                session, "transaction_anomaly", anomaly_results
            )
            
            context = {
                "transaction": transaction.model_dump(),
                "similar_count": len(similar_transactions),
                "anomaly_details": anomaly_results,
                "confidence_factors": confidence["factors"]
            }
            
            reasoning = await self.analyze_with_claude(
                f"Advanced anomaly analysis: {anomaly_results['description']}. "
                f"Multiple detection algorithms flagged this transaction. "
                f"Confidence: {confidence['score']:.2%}. Should this be flagged?",
                context
            )
            
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="transaction_anomaly",
                context=context,
                reasoning=reasoning,
                action=f"Flag transaction {transaction.id} for review",
                confidence=confidence["score"]
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
                "summary": summary.model_dump(),
                "recent_decisions": [d.to_dict() for d in self.get_decision_history(10)],
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
    
    # =====================================
    # ADVANCED FINANCIAL ANALYSIS METHODS
    # =====================================
    
    async def _detect_transaction_anomalies(
        self, 
        session, 
        transaction: TransactionModel, 
        similar_transactions: List[Transaction]
    ) -> Dict[str, Any]:
        """
        Enhanced anomaly detection using multiple algorithms.
        
        Args:
            session: Database session
            transaction: Transaction to analyze
            similar_transactions: List of similar historical transactions
            
        Returns:
            Dict with anomaly detection results and details
        """
        try:
            amounts = [float(t.amount) for t in similar_transactions]
            transaction_amount = float(transaction.amount)
            
            # 1. Statistical outlier detection (Z-score)
            if len(amounts) >= 3:
                mean_amount = statistics.mean(amounts)
                std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
                z_score = abs(transaction_amount - mean_amount) / std_amount if std_amount > 0 else 0
                is_statistical_outlier = z_score > 2.5  # 99.4% confidence
            else:
                is_statistical_outlier = False
                z_score = 0
                mean_amount = sum(amounts) / len(amounts) if amounts else 0
                std_amount = 0
            
            # 2. Interquartile Range (IQR) method
            if len(amounts) >= 5:
                q1 = statistics.quantiles(amounts, n=4)[0]
                q3 = statistics.quantiles(amounts, n=4)[2]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                is_iqr_outlier = transaction_amount < lower_bound or transaction_amount > upper_bound
            else:
                is_iqr_outlier = False
                lower_bound = upper_bound = 0
            
            # 3. Percentage variance from median
            median_amount = statistics.median(amounts) if amounts else 0
            median_variance = abs(transaction_amount - median_amount) / median_amount if median_amount > 0 else 0
            is_variance_outlier = median_variance > self.anomaly_threshold
            
            # 4. Time-based anomaly (unusual hour/day patterns)
            time_patterns = await self._analyze_time_patterns(session, transaction, similar_transactions)
            is_time_anomaly = time_patterns["is_anomaly"]
            
            # Combine detection methods
            anomaly_count = sum([
                is_statistical_outlier,
                is_iqr_outlier, 
                is_variance_outlier,
                is_time_anomaly
            ])
            
            is_anomaly = anomaly_count >= 2  # Require at least 2 methods to flag
            
            return {
                "is_anomaly": is_anomaly,
                "anomaly_count": anomaly_count,
                "statistical_outlier": is_statistical_outlier,
                "iqr_outlier": is_iqr_outlier,
                "variance_outlier": is_variance_outlier,
                "time_anomaly": is_time_anomaly,
                "z_score": z_score,
                "median_variance": median_variance,
                "time_patterns": time_patterns,
                "statistics": {
                    "mean": mean_amount,
                    "median": median_amount,
                    "std_dev": std_amount,
                    "sample_size": len(amounts)
                },
                "description": f"Transaction amount ${transaction_amount:.2f} analyzed against {len(amounts)} similar transactions"
            }
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return {"is_anomaly": False, "error": str(e)}
    
    async def _analyze_time_patterns(self, session, transaction: TransactionModel, similar_transactions: List[Transaction]) -> Dict[str, Any]:
        """Analyze time-based patterns for anomaly detection."""
        try:
            transaction_hour = transaction.transaction_date.hour
            transaction_weekday = transaction.transaction_date.weekday()
            
            # Analyze hour patterns
            hour_counts = {}
            weekday_counts = {}
            
            for t in similar_transactions:
                hour = t.transaction_date.hour
                weekday = t.transaction_date.weekday()
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
                weekday_counts[weekday] = weekday_counts.get(weekday, 0) + 1
            
            # Check if transaction time is unusual
            total_transactions = len(similar_transactions)
            hour_frequency = hour_counts.get(transaction_hour, 0) / total_transactions if total_transactions > 0 else 0
            weekday_frequency = weekday_counts.get(transaction_weekday, 0) / total_transactions if total_transactions > 0 else 0
            
            # Flag as anomaly if time patterns are very unusual (< 5% of transactions)
            is_time_anomaly = hour_frequency < 0.05 or weekday_frequency < 0.05
            
            return {
                "is_anomaly": is_time_anomaly,
                "hour_frequency": hour_frequency,
                "weekday_frequency": weekday_frequency,
                "transaction_hour": transaction_hour,
                "transaction_weekday": transaction_weekday
            }
            
        except Exception as e:
            self.logger.error(f"Error in time pattern analysis: {e}")
            return {"is_anomaly": False, "error": str(e)}
    
    async def _calculate_dynamic_confidence(
        self, 
        session, 
        decision_type: str, 
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate dynamic confidence score based on multiple factors."""
        try:
            factors = self.forecasting_config["confidence_factors"]
            confidence_score = 0.0
            factor_details = {}
            
            # Factor 1: Data volume (more data = higher confidence)
            if "statistics" in analysis_data and "sample_size" in analysis_data["statistics"]:
                sample_size = analysis_data["statistics"]["sample_size"]
                volume_factor = min(1.0, sample_size / 30)  # Max confidence at 30+ samples
                confidence_score += factors["data_volume"] * volume_factor
                factor_details["data_volume"] = volume_factor
            
            # Factor 2: Historical accuracy of similar decisions
            historical_accuracy = await self._get_historical_accuracy(session, decision_type)
            confidence_score += factors["historical_accuracy"] * historical_accuracy
            factor_details["historical_accuracy"] = historical_accuracy
            
            # Factor 3: Analysis consistency (how many methods agree)
            if "anomaly_count" in analysis_data:
                consistency_factor = analysis_data["anomaly_count"] / 4  # 4 total methods
                confidence_score += factors["trend_stability"] * consistency_factor
                factor_details["trend_stability"] = consistency_factor
            
            # Factor 4: Seasonal consistency (if applicable)
            seasonal_factor = await self._calculate_seasonal_consistency(session, analysis_data)
            confidence_score += factors["seasonal_consistency"] * seasonal_factor
            factor_details["seasonal_consistency"] = seasonal_factor
            
            return {
                "score": min(0.95, max(0.1, confidence_score)),  # Cap between 10% and 95%
                "factors": factor_details
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic confidence: {e}")
            return {"score": 0.5, "factors": {}, "error": str(e)}\n    \n    async def _get_historical_accuracy(self, session, decision_type: str) -> float:\n        \"\"\"Get historical accuracy for decisions of this type.\"\"\"\n        try:\n            # Get recent decisions of this type with outcomes\n            type_outcomes = [\n                outcome for decision_id, outcome in self.decision_outcomes.items()\n                if outcome.get(\"decision_type\") == decision_type\n            ]\n            \n            if not type_outcomes:\n                return 0.7  # Default moderate confidence\n            \n            # Calculate accuracy rate\n            correct_decisions = sum(1 for outcome in type_outcomes if outcome.get(\"was_correct\", False))\n            accuracy = correct_decisions / len(type_outcomes)\n            \n            return accuracy\n            \n        except Exception as e:\n            self.logger.error(f\"Error getting historical accuracy: {e}\")\n            return 0.5\n    \n    async def _calculate_seasonal_consistency(self, session, analysis_data: Dict[str, Any]) -> float:\n        \"\"\"Calculate seasonal consistency factor.\"\"\"\n        try:\n            # For now, return a moderate factor\n            # In a full implementation, this would analyze seasonal patterns\n            return 0.6\n            \n        except Exception as e:\n            self.logger.error(f\"Error calculating seasonal consistency: {e}\")\n            return 0.5\n    \n    async def _forecast_cash_flow(\n        self, \n        session, \n        forecast_days: int = 30\n    ) -> Optional[AgentDecision]:\n        \"\"\"Advanced cash flow forecasting with predictive analytics.\"\"\"\n        try:\n            # Get historical transaction data\n            end_date = datetime.now()\n            start_date = end_date - timedelta(days=self.forecasting_config[\"seasonal_analysis_days\"])\n            \n            transactions = session.query(Transaction).filter(\n                Transaction.transaction_date >= start_date\n            ).order_by(Transaction.transaction_date).all()\n            \n            if len(transactions) < 7:  # Need at least a week of data\n                return None\n            \n            # Prepare data for forecasting\n            daily_flows = await self._prepare_daily_cash_flows(transactions)\n            \n            # Generate forecasts using multiple methods\n            forecasts = await self._generate_cash_flow_forecasts(daily_flows, forecast_days)\n            \n            # Identify potential issues\n            forecast_analysis = await self._analyze_forecast_results(session, forecasts, forecast_days)\n            \n            # Calculate confidence for forecast\n            confidence = await self._calculate_forecast_confidence(daily_flows, forecasts)\n            \n            context = {\n                \"forecast_days\": forecast_days,\n                \"historical_data_days\": len(daily_flows),\n                \"forecasts\": forecasts,\n                \"analysis\": forecast_analysis,\n                \"confidence_score\": confidence\n            }\n            \n            # Generate detailed analysis with Claude\n            reasoning = await self.analyze_with_claude(\n                f\"Cash flow forecast analysis for next {forecast_days} days. \"\n                f\"Forecast shows: {forecast_analysis['summary']}. \"\n                f\"Confidence: {confidence:.2%}. Provide strategic recommendations.\",\n                context\n            )\n            \n            return AgentDecision(\n                agent_id=self.agent_id,\n                decision_type=\"cash_flow_forecast\",\n                context=context,\n                reasoning=reasoning,\n                action=forecast_analysis[\"recommended_action\"],\n                confidence=confidence\n            )\n            \n        except Exception as e:\n            self.logger.error(f\"Error in cash flow forecasting: {e}\")\n            return None\n    \n    async def _prepare_daily_cash_flows(self, transactions: List[Transaction]) -> Dict[str, float]:\n        \"\"\"Prepare daily cash flow data for forecasting.\"\"\"\n        daily_flows = {}\n        \n        for transaction in transactions:\n            date_key = transaction.transaction_date.date().isoformat()\n            amount = float(transaction.amount)\n            \n            if transaction.transaction_type == TransactionType.INCOME:\n                daily_flows[date_key] = daily_flows.get(date_key, 0) + amount\n            elif transaction.transaction_type == TransactionType.EXPENSE:\n                daily_flows[date_key] = daily_flows.get(date_key, 0) - amount\n        \n        return daily_flows\n    \n    async def _generate_cash_flow_forecasts(\n        self, \n        daily_flows: Dict[str, float], \n        forecast_days: int\n    ) -> Dict[str, Any]:\n        \"\"\"Generate cash flow forecasts using multiple methods.\"\"\"\n        try:\n            flow_values = list(daily_flows.values())\n            \n            # Method 1: Simple moving average\n            window_size = min(7, len(flow_values))\n            recent_flows = flow_values[-window_size:]\n            simple_avg = sum(recent_flows) / len(recent_flows)\n            \n            # Method 2: Weighted moving average (more weight to recent data)\n            weights = [i + 1 for i in range(len(recent_flows))]\n            weighted_avg = sum(flow * weight for flow, weight in zip(recent_flows, weights)) / sum(weights)\n            \n            # Method 3: Trend-based projection\n            if len(flow_values) >= 14:\n                recent_trend = self._calculate_trend(flow_values[-14:])  # 2-week trend\n                trend_forecast = recent_flows[-1] + (recent_trend * forecast_days / 2)\n            else:\n                trend_forecast = simple_avg\n            \n            # Method 4: Seasonal adjustment (if enough data)\n            seasonal_forecast = await self._apply_seasonal_adjustment(\n                daily_flows, simple_avg, forecast_days\n            )\n            \n            # Combine forecasts with weights\n            ensemble_forecast = (\n                simple_avg * 0.25 +\n                weighted_avg * 0.35 +\n                trend_forecast * 0.25 +\n                seasonal_forecast * 0.15\n            )\n            \n            return {\n                \"simple_moving_average\": simple_avg,\n                \"weighted_moving_average\": weighted_avg,\n                \"trend_based\": trend_forecast,\n                \"seasonal_adjusted\": seasonal_forecast,\n                \"ensemble\": ensemble_forecast,\n                \"daily_forecast\": ensemble_forecast,\n                \"total_forecast\": ensemble_forecast * forecast_days\n            }\n            \n        except Exception as e:\n            self.logger.error(f\"Error generating forecasts: {e}\")\n            return {\"ensemble\": 0, \"total_forecast\": 0}\n    \n    def _calculate_trend(self, values: List[float]) -> float:\n        \"\"\"Calculate trend using simple linear regression.\"\"\"\n        try:\n            n = len(values)\n            if n < 2:\n                return 0\n            \n            x_sum = sum(range(n))\n            y_sum = sum(values)\n            xy_sum = sum(i * values[i] for i in range(n))\n            x_squared_sum = sum(i * i for i in range(n))\n            \n            slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)\n            return slope\n            \n        except Exception as e:\n            self.logger.error(f\"Error calculating trend: {e}\")\n            return 0\n    \n    async def _apply_seasonal_adjustment(\n        self, \n        daily_flows: Dict[str, float], \n        base_forecast: float, \n        forecast_days: int\n    ) -> float:\n        \"\"\"Apply seasonal adjustments to forecasts.\"\"\"\n        try:\n            # For now, return base forecast\n            # In full implementation, this would analyze seasonal patterns\n            return base_forecast\n            \n        except Exception as e:\n            self.logger.error(f\"Error in seasonal adjustment: {e}\")\n            return base_forecast\n    \n    async def _analyze_forecast_results(\n        self, \n        session, \n        forecasts: Dict[str, Any], \n        forecast_days: int\n    ) -> Dict[str, Any]:\n        \"\"\"Analyze forecast results and identify potential issues.\"\"\"\n        try:\n            total_forecast = forecasts[\"total_forecast\"]\n            daily_forecast = forecasts[\"daily_forecast\"]\n            \n            # Get current cash position\n            current_cash = await self._get_current_cash_balance(session)\n            projected_cash = current_cash + total_forecast\n            \n            # Analyze results\n            is_shortage_predicted = projected_cash < self.alert_thresholds[\"cash_low\"]\n            shortage_severity = \"high\" if projected_cash < 0 else \"medium\" if is_shortage_predicted else \"low\"\n            \n            # Generate summary and recommendations\n            if is_shortage_predicted:\n                summary = f\"Cash shortage predicted: ${projected_cash:.2f} in {forecast_days} days\"\n                recommended_action = \"Urgent: Implement cash flow improvement measures\"\n            elif daily_forecast < 0:\n                summary = f\"Negative daily cash flow predicted: ${daily_forecast:.2f}/day\"\n                recommended_action = \"Monitor cash flow closely and optimize expenses\"\n            else:\n                summary = f\"Positive cash flow predicted: ${total_forecast:.2f} over {forecast_days} days\"\n                recommended_action = \"Continue current financial management approach\"\n            \n            return {\n                \"summary\": summary,\n                \"current_cash\": current_cash,\n                \"projected_cash\": projected_cash,\n                \"is_shortage_predicted\": is_shortage_predicted,\n                \"shortage_severity\": shortage_severity,\n                \"recommended_action\": recommended_action,\n                \"daily_forecast\": daily_forecast\n            }\n            \n        except Exception as e:\n            self.logger.error(f\"Error analyzing forecast results: {e}\")\n            return {\"summary\": \"Error in forecast analysis\", \"recommended_action\": \"Review forecast data\"}\n    \n    async def _get_current_cash_balance(self, session) -> float:\n        \"\"\"Get current total cash balance.\"\"\"\n        try:\n            cash_accounts = session.query(Account).filter(\n                Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS])\n            ).all()\n            \n            return float(sum(account.balance for account in cash_accounts))\n            \n        except Exception as e:\n            self.logger.error(f\"Error getting cash balance: {e}\")\n            return 0.0\n    \n    async def _calculate_forecast_confidence(\n        self, \n        daily_flows: Dict[str, float], \n        forecasts: Dict[str, Any]\n    ) -> float:\n        \"\"\"Calculate confidence in the forecast based on data quality and consistency.\"\"\"\n        try:\n            # Factor 1: Data consistency (lower variance = higher confidence)\n            flow_values = list(daily_flows.values())\n            if len(flow_values) > 1:\n                variance = statistics.variance(flow_values)\n                mean_abs_flow = statistics.mean([abs(f) for f in flow_values])\n                consistency_factor = max(0.1, 1 - (variance / (mean_abs_flow ** 2)) if mean_abs_flow > 0 else 0.1)\n            else:\n                consistency_factor = 0.3\n            \n            # Factor 2: Data volume (more data = higher confidence)\n            volume_factor = min(1.0, len(flow_values) / 30)\n            \n            # Factor 3: Forecast method agreement\n            forecast_values = [\n                forecasts[\"simple_moving_average\"],\n                forecasts[\"weighted_moving_average\"],\n                forecasts[\"trend_based\"],\n                forecasts[\"seasonal_adjusted\"]\n            ]\n            \n            if len(forecast_values) > 1:\n                forecast_variance = statistics.variance(forecast_values)\n                forecast_mean = statistics.mean([abs(f) for f in forecast_values])\n                agreement_factor = max(0.1, 1 - (forecast_variance / (forecast_mean ** 2)) if forecast_mean > 0 else 0.1)\n            else:\n                agreement_factor = 0.5\n            \n            # Combine factors\n            confidence = (\n                consistency_factor * 0.4 +\n                volume_factor * 0.3 +\n                agreement_factor * 0.3\n            )\n            \n            return min(0.9, max(0.2, confidence))\n            \n        except Exception as e:\n            self.logger.error(f\"Error calculating forecast confidence: {e}\")\n            return 0.5\n    \n    async def _analyze_financial_trends(self, session) -> Optional[AgentDecision]:\n        \"\"\"Analyze financial trends across multiple time periods.\"\"\"\n        try:\n            periods = self.forecasting_config[\"trend_analysis_periods\"]\n            end_date = datetime.now()\n            \n            # Analyze trends over different periods\n            trend_analysis = {}\n            \n            for period in [7, 30, 90]:  # Weekly, monthly, quarterly trends\n                start_date = end_date - timedelta(days=period)\n                \n                transactions = session.query(Transaction).filter(\n                    Transaction.transaction_date >= start_date\n                ).all()\n                \n                if transactions:\n                    trend_data = await self._calculate_period_trends(transactions, period)\n                    trend_analysis[f\"{period}_days\"] = trend_data\n            \n            if not trend_analysis:\n                return None\n            \n            # Identify significant trends\n            significant_trends = await self._identify_significant_trends(trend_analysis)\n            \n            if not significant_trends:\n                return None\n            \n            # Calculate confidence for trend analysis\n            confidence = await self._calculate_trend_confidence(trend_analysis)\n            \n            context = {\n                \"trend_analysis\": trend_analysis,\n                \"significant_trends\": significant_trends,\n                \"analysis_periods\": list(trend_analysis.keys())\n            }\n            \n            reasoning = await self.analyze_with_claude(\n                f\"Financial trend analysis reveals: {len(significant_trends)} significant trends. \"\n                f\"Key findings: {', '.join(significant_trends)}. \"\n                f\"Provide strategic recommendations based on these trends.\",\n                context\n            )\n            \n            return AgentDecision(\n                agent_id=self.agent_id,\n                decision_type=\"financial_trend_analysis\",\n                context=context,\n                reasoning=reasoning,\n                action=\"Generate comprehensive trend analysis report\",\n                confidence=confidence\n            )\n            \n        except Exception as e:\n            self.logger.error(f\"Error in trend analysis: {e}\")\n            return None\n    \n    async def _calculate_period_trends(\n        self, \n        transactions: List[Transaction], \n        period_days: int\n    ) -> Dict[str, Any]:\n        \"\"\"Calculate trends for a specific period.\"\"\"\n        try:\n            # Separate income and expenses\n            income_transactions = [t for t in transactions if t.transaction_type == TransactionType.INCOME]\n            expense_transactions = [t for t in transactions if t.transaction_type == TransactionType.EXPENSE]\n            \n            total_income = sum(float(t.amount) for t in income_transactions)\n            total_expenses = sum(float(t.amount) for t in expense_transactions)\n            net_flow = total_income - total_expenses\n            \n            # Calculate daily averages\n            daily_income = total_income / period_days\n            daily_expenses = total_expenses / period_days\n            daily_net = net_flow / period_days\n            \n            # Calculate trends within the period\n            income_trend = self._calculate_trend([float(t.amount) for t in income_transactions[-14:]])\n            expense_trend = self._calculate_trend([float(t.amount) for t in expense_transactions[-14:]])\n            \n            # Analyze category distributions\n            expense_categories = {}\n            for t in expense_transactions:\n                category = t.category or \"uncategorized\"\n                expense_categories[category] = expense_categories.get(category, 0) + float(t.amount)\n            \n            return {\n                \"period_days\": period_days,\n                \"total_income\": total_income,\n                \"total_expenses\": total_expenses,\n                \"net_flow\": net_flow,\n                \"daily_income\": daily_income,\n                \"daily_expenses\": daily_expenses,\n                \"daily_net\": daily_net,\n                \"income_trend\": income_trend,\n                \"expense_trend\": expense_trend,\n                \"transaction_count\": len(transactions),\n                \"expense_categories\": expense_categories\n            }\n            \n        except Exception as e:\n            self.logger.error(f\"Error calculating period trends: {e}\")\n            return {}\n    \n    async def _identify_significant_trends(self, trend_analysis: Dict[str, Any]) -> List[str]:\n        \"\"\"Identify significant trends that require attention.\"\"\"\n        significant_trends = []\n        \n        try:\n            # Compare different periods\n            periods = sorted(trend_analysis.keys())\n            \n            for period in periods:\n                data = trend_analysis[period]\n                \n                # Check for negative trends\n                if data.get(\"income_trend\", 0) < -50:  # Declining income\n                    significant_trends.append(f\"Declining income trend in {period}\")\n                \n                if data.get(\"expense_trend\", 0) > 50:  # Rising expenses\n                    significant_trends.append(f\"Rising expense trend in {period}\")\n                \n                if data.get(\"daily_net\", 0) < -100:  # Negative daily net flow\n                    significant_trends.append(f\"Negative cash flow in {period}\")\n            \n            # Compare short-term vs long-term trends\n            if \"7_days\" in trend_analysis and \"30_days\" in trend_analysis:\n                short_term = trend_analysis[\"7_days\"]\n                long_term = trend_analysis[\"30_days\"]\n                \n                if short_term.get(\"daily_net\", 0) < long_term.get(\"daily_net\", 0) * 0.7:\n                    significant_trends.append(\"Short-term performance decline\")\n            \n            return significant_trends\n            \n        except Exception as e:\n            self.logger.error(f\"Error identifying significant trends: {e}\")\n            return []\n    \n    async def _calculate_trend_confidence(self, trend_analysis: Dict[str, Any]) -> float:\n        \"\"\"Calculate confidence in trend analysis.\"\"\"\n        try:\n            # Base confidence on data volume and consistency\n            total_transactions = sum(\n                data.get(\"transaction_count\", 0) for data in trend_analysis.values()\n            )\n            \n            volume_factor = min(1.0, total_transactions / 50)  # Max confidence at 50+ transactions\n            \n            # Check consistency across periods\n            if len(trend_analysis) >= 2:\n                consistency_checks = 0\n                total_checks = 0\n                \n                for period1, period2 in zip(list(trend_analysis.keys())[:-1], list(trend_analysis.keys())[1:]):\n                    data1 = trend_analysis[period1]\n                    data2 = trend_analysis[period2]\n                    \n                    # Check if trends are consistent\n                    if (data1.get(\"income_trend\", 0) * data2.get(\"income_trend\", 0)) >= 0:\n                        consistency_checks += 1\n                    if (data1.get(\"expense_trend\", 0) * data2.get(\"expense_trend\", 0)) >= 0:\n                        consistency_checks += 1\n                    \n                    total_checks += 2\n                \n                consistency_factor = consistency_checks / total_checks if total_checks > 0 else 0.5\n            else:\n                consistency_factor = 0.5\n            \n            confidence = volume_factor * 0.6 + consistency_factor * 0.4\n            return min(0.9, max(0.3, confidence))\n            \n        except Exception as e:\n            self.logger.error(f\"Error calculating trend confidence: {e}\")\n            return 0.5\n    \n    async def _process_decision_outcome(self, data: Dict[str, Any]) -> Optional[AgentDecision]:\n        \"\"\"Process feedback on decision outcomes for learning.\"\"\"\n        try:\n            decision_id = data.get(\"decision_id\")\n            was_correct = data.get(\"was_correct\", False)\n            feedback_notes = data.get(\"feedback_notes\", \"\")\n            decision_type = data.get(\"decision_type\")\n            \n            if not decision_id:\n                return None\n            \n            # Store outcome for future confidence calculations\n            self.decision_outcomes[decision_id] = {\n                \"was_correct\": was_correct,\n                \"feedback_notes\": feedback_notes,\n                \"decision_type\": decision_type,\n                \"timestamp\": datetime.now()\n            }\n            \n            # Analyze patterns in decision outcomes\n            outcome_analysis = await self._analyze_decision_patterns()\n            \n            if outcome_analysis[\"needs_adjustment\"]:\n                context = {\n                    \"decision_id\": decision_id,\n                    \"outcome\": {\"was_correct\": was_correct, \"feedback\": feedback_notes},\n                    \"pattern_analysis\": outcome_analysis\n                }\n                \n                reasoning = await self.analyze_with_claude(\n                    f\"Decision outcome received: {'Correct' if was_correct else 'Incorrect'}. \"\n                    f\"Pattern analysis suggests: {outcome_analysis['recommendation']}. \"\n                    f\"How should the decision-making process be adjusted?\",\n                    context\n                )\n                \n                return AgentDecision(\n                    agent_id=self.agent_id,\n                    decision_type=\"decision_learning\",\n                    context=context,\n                    reasoning=reasoning,\n                    action=\"Adjust decision-making parameters based on feedback\",\n                    confidence=0.8\n                )\n            \n            return None\n            \n        except Exception as e:\n            self.logger.error(f\"Error processing decision outcome: {e}\")\n            return None\n    \n    async def _analyze_decision_patterns(self) -> Dict[str, Any]:\n        \"\"\"Analyze patterns in decision outcomes to identify improvement opportunities.\"\"\"\n        try:\n            if len(self.decision_outcomes) < 5:\n                return {\"needs_adjustment\": False, \"recommendation\": \"Insufficient data\"}\n            \n            # Analyze accuracy by decision type\n            type_accuracy = {}\n            for outcome in self.decision_outcomes.values():\n                decision_type = outcome.get(\"decision_type\", \"unknown\")\n                if decision_type not in type_accuracy:\n                    type_accuracy[decision_type] = {\"correct\": 0, \"total\": 0}\n                \n                type_accuracy[decision_type][\"total\"] += 1\n                if outcome.get(\"was_correct\", False):\n                    type_accuracy[decision_type][\"correct\"] += 1\n            \n            # Identify types with low accuracy\n            low_accuracy_types = []\n            for decision_type, stats in type_accuracy.items():\n                accuracy = stats[\"correct\"] / stats[\"total\"]\n                if accuracy < 0.7 and stats[\"total\"] >= 3:  # At least 3 decisions to be significant\n                    low_accuracy_types.append((decision_type, accuracy))\n            \n            needs_adjustment = len(low_accuracy_types) > 0\n            \n            if needs_adjustment:\n                recommendation = f\"Low accuracy in: {', '.join([f'{t[0]} ({t[1]:.1%})' for t in low_accuracy_types])}\"\n            else:\n                recommendation = \"Decision accuracy is acceptable across all types\"\n            \n            return {\n                \"needs_adjustment\": needs_adjustment,\n                \"recommendation\": recommendation,\n                \"type_accuracy\": type_accuracy,\n                \"low_accuracy_types\": low_accuracy_types,\n                \"total_decisions\": len(self.decision_outcomes)\n            }\n            \n        except Exception as e:\n            self.logger.error(f\"Error analyzing decision patterns: {e}\")\n            return {\"needs_adjustment\": False, \"recommendation\": \"Error in pattern analysis\"}\n    \n    async def periodic_check(self):
        """Perform periodic accounting analysis with enhanced capabilities"""
        session = self.SessionLocal()
        try:
            current_hour = datetime.now().hour
            
            # Perform cash flow check
            decision = await self._check_cash_flow(session)
            if decision:
                self.log_decision(decision)
            
            # Perform aging analysis once per day
            if current_hour == 9:  # 9 AM daily aging analysis
                aging_decision = await self._analyze_aging(session)
                if aging_decision:
                    self.log_decision(aging_decision)
            
            # Perform cash flow forecasting twice per day (9 AM and 3 PM)
            if current_hour in [9, 15]:
                forecast_decision = await self._forecast_cash_flow(session, 30)
                if forecast_decision:
                    self.log_decision(forecast_decision)
            
            # Perform trend analysis once per week (Monday at 10 AM)
            if current_hour == 10 and datetime.now().weekday() == 0:
                trend_decision = await self._analyze_financial_trends(session)
                if trend_decision:
                    self.log_decision(trend_decision)
            
            self.logger.debug("Enhanced periodic accounting check completed")
            
        except Exception as e:
            self.logger.error(f"Error in periodic check: {e}")
        finally:
            session.close()