import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func

from agents.base_agent import AgentDecision, BaseAgent
from models.financial import (
    Account,
    AccountsPayable,
    AccountsReceivable,
    AccountType,
    FinancialSummary,
    Transaction,
    TransactionModel,
    TransactionType,
)


class AccountingAgent(BaseAgent):
    def __init__(self, agent_id: str, api_key: str, config: Dict[str, Any], db_url: str):
        super().__init__(agent_id, api_key, config, db_url)
        # Remove duplicate engine and session creation since BaseAgent now handles this
        self.anomaly_threshold = config.get("anomaly_threshold", 0.2)  # 20% variance
        self.alert_thresholds = config.get(
            "alert_thresholds",
            {"cash_low": 1000, "receivables_overdue": 30, "payables_overdue": 7},  # days  # days
        )

        # Enhanced configuration for advanced analytics
        self.forecasting_config = config.get(
            "forecasting",
            {
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
        )

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

    async def _analyze_transaction(
        self, session, transaction_data: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        transaction = TransactionModel(**transaction_data)

        # Get recent similar transactions for comparison
        similar_transactions = (
            session.query(Transaction)
            .filter(
                and_(
                    Transaction.transaction_type == transaction.transaction_type,
                    Transaction.category == transaction.category,
                    Transaction.transaction_date >= datetime.now() - timedelta(days=30),
                )
            )
            .all()
        )

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
                "confidence_factors": confidence["factors"],
            }

            reasoning = await self.analyze_with_claude(
                f"Advanced anomaly analysis: {anomaly_results['description']}. "
                f"Multiple detection algorithms flagged this transaction. "
                f"Confidence: {confidence['score']:.2%}. Should this be flagged?",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="transaction_anomaly",
                context=context,
                reasoning=reasoning,
                action=f"Flag transaction {transaction.id} for review",
                confidence=confidence["score"],
            )

        return None

    async def _perform_daily_analysis(self, session) -> Optional[AgentDecision]:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Get yesterday's transactions
        daily_transactions = (
            session.query(Transaction)
            .filter(func.date(Transaction.transaction_date) == yesterday)
            .all()
        )

        if not daily_transactions:
            return None

        total_income = sum(
            t.amount for t in daily_transactions if t.transaction_type == TransactionType.INCOME
        )
        total_expenses = sum(
            t.amount for t in daily_transactions if t.transaction_type == TransactionType.EXPENSE
        )
        net_flow = total_income - total_expenses

        # Get last 30 days average for comparison
        thirty_days_ago = today - timedelta(days=30)
        historical_transactions = (
            session.query(Transaction)
            .filter(
                and_(
                    Transaction.transaction_date >= thirty_days_ago,
                    Transaction.transaction_date < yesterday,
                )
            )
            .all()
        )

        context = {
            "date": str(yesterday),
            "transaction_count": len(daily_transactions),
            "total_income": float(total_income),
            "total_expenses": float(total_expenses),
            "net_flow": float(net_flow),
            "historical_avg_income": 0,
            "historical_avg_expenses": 0,
        }

        if historical_transactions:
            hist_income = sum(
                t.amount
                for t in historical_transactions
                if t.transaction_type == TransactionType.INCOME
            )
            hist_expenses = sum(
                t.amount
                for t in historical_transactions
                if t.transaction_type == TransactionType.EXPENSE
            )
            hist_days = len({t.transaction_date.date() for t in historical_transactions})

            if hist_days > 0:
                context["historical_avg_income"] = float(hist_income / hist_days)
                context["historical_avg_expenses"] = float(hist_expenses / hist_days)

        analysis = await self.analyze_with_claude(
            f"Analyze yesterday's financial performance. "
            f"Income: ${total_income}, Expenses: ${total_expenses}, Net: ${net_flow}. "
            f"Provide insights and recommendations.",
            context,
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="daily_financial_analysis",
            context=context,
            reasoning=analysis,
            action="Generate daily financial report",
            confidence=0.8,
        )

    async def _check_cash_flow(self, session) -> Optional[AgentDecision]:
        # Get current cash balances
        cash_accounts = (
            session.query(Account)
            .filter(Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS]))
            .all()
        )

        total_cash = sum(account.balance for account in cash_accounts)
        low_cash_threshold = self.alert_thresholds["cash_low"]

        context = {
            "total_cash": float(total_cash),
            "threshold": low_cash_threshold,
            "accounts": [
                {"name": acc.name, "balance": float(acc.balance)} for acc in cash_accounts
            ],
        }

        if total_cash < low_cash_threshold:
            reasoning = await self.analyze_with_claude(
                f"Cash balance is low: ${total_cash}. Threshold: ${low_cash_threshold}. "
                f"What actions should be taken?",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="low_cash_alert",
                context=context,
                reasoning=reasoning,
                action="Alert management of low cash situation",
                confidence=0.9,
            )

        return None

    async def _analyze_aging(self, session) -> Optional[AgentDecision]:
        today = datetime.now().date()
        overdue_receivables_threshold = self.alert_thresholds["receivables_overdue"]
        overdue_payables_threshold = self.alert_thresholds["payables_overdue"]

        # Check overdue receivables
        overdue_receivables = (
            session.query(AccountsReceivable)
            .filter(
                and_(
                    AccountsReceivable.due_date
                    < today - timedelta(days=overdue_receivables_threshold),
                    AccountsReceivable.status == "unpaid",
                )
            )
            .all()
        )

        # Check overdue payables
        overdue_payables = (
            session.query(AccountsPayable)
            .filter(
                and_(
                    AccountsPayable.due_date < today - timedelta(days=overdue_payables_threshold),
                    AccountsPayable.status == "unpaid",
                )
            )
            .all()
        )

        if not overdue_receivables and not overdue_payables:
            return None

        context = {
            "overdue_receivables": [
                {
                    "customer": ar.customer_name,
                    "amount": float(ar.amount),
                    "days_overdue": (today - ar.due_date).days,
                    "invoice_number": ar.invoice_number,
                }
                for ar in overdue_receivables
            ],
            "overdue_payables": [
                {
                    "vendor": ap.vendor_name,
                    "amount": float(ap.amount),
                    "days_overdue": (today - ap.due_date).days,
                    "invoice_number": ap.invoice_number,
                }
                for ap in overdue_payables
            ],
            "total_overdue_receivables": float(sum(ar.amount for ar in overdue_receivables)),
            "total_overdue_payables": float(sum(ap.amount for ap in overdue_payables)),
        }

        analysis = await self.analyze_with_claude(
            f"Aging analysis shows {len(overdue_receivables)} overdue receivables "
            f"totaling ${context['total_overdue_receivables']} and "
            f"{len(overdue_payables)} overdue payables totaling ${context['total_overdue_payables']}. "
            f"Provide collection and payment priority recommendations.",
            context,
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="aging_analysis",
            context=context,
            reasoning=analysis,
            action="Generate aging report and collection recommendations",
            confidence=0.85,
        )

    async def generate_report(self) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            # Generate comprehensive financial summary
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            transactions = (
                session.query(Transaction).filter(Transaction.transaction_date >= start_date).all()
            )

            total_revenue = sum(
                t.amount for t in transactions if t.transaction_type == TransactionType.INCOME
            )
            total_expenses = sum(
                t.amount for t in transactions if t.transaction_type == TransactionType.EXPENSE
            )

            cash_accounts = (
                session.query(Account)
                .filter(Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS]))
                .all()
            )
            cash_balance = sum(acc.balance for acc in cash_accounts)

            receivables = (
                session.query(AccountsReceivable)
                .filter(AccountsReceivable.status == "unpaid")
                .all()
            )
            total_receivables = sum(ar.amount for ar in receivables)

            payables = (
                session.query(AccountsPayable).filter(AccountsPayable.status == "unpaid").all()
            )
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
                transaction_count=len(transactions),
            )

            return {
                "summary": summary.model_dump(),
                "recent_decisions": [d.to_dict() for d in self.get_decision_history(10)],
                "alerts": await self._get_current_alerts(session),
            }
        finally:
            session.close()

    async def _get_current_alerts(self, session) -> List[Dict[str, Any]]:
        alerts = []

        # Check cash levels
        cash_accounts = (
            session.query(Account)
            .filter(Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS]))
            .all()
        )
        total_cash = sum(acc.balance for acc in cash_accounts)

        if total_cash < self.alert_thresholds["cash_low"]:
            alerts.append(
                {
                    "type": "low_cash",
                    "severity": "high",
                    "message": f"Cash balance is low: ${total_cash}",
                    "action_required": True,
                }
            )

        # Check overdue items
        today = datetime.now().date()
        overdue_receivables = (
            session.query(AccountsReceivable)
            .filter(
                and_(AccountsReceivable.due_date < today, AccountsReceivable.status == "unpaid")
            )
            .count()
        )

        if overdue_receivables > 0:
            alerts.append(
                {
                    "type": "overdue_receivables",
                    "severity": "medium",
                    "message": f"{overdue_receivables} overdue invoices need collection",
                    "action_required": True,
                }
            )

        return alerts

    # =====================================
    # ADVANCED FINANCIAL ANALYSIS METHODS
    # =====================================

    async def _detect_transaction_anomalies(
        self, session, transaction: TransactionModel, similar_transactions: List[Transaction]
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
                z_score = (
                    abs(transaction_amount - mean_amount) / std_amount if std_amount > 0 else 0
                )
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
                is_iqr_outlier = (
                    transaction_amount < lower_bound or transaction_amount > upper_bound
                )
            else:
                is_iqr_outlier = False
                lower_bound = upper_bound = 0

            # 3. Percentage variance from median
            median_amount = statistics.median(amounts) if amounts else 0
            median_variance = (
                abs(transaction_amount - median_amount) / median_amount if median_amount > 0 else 0
            )
            is_variance_outlier = median_variance > self.anomaly_threshold

            # 4. Time-based anomaly (unusual hour/day patterns)
            time_patterns = await self._analyze_time_patterns(
                session, transaction, similar_transactions
            )
            is_time_anomaly = time_patterns["is_anomaly"]

            # Combine detection methods
            anomaly_count = sum(
                [is_statistical_outlier, is_iqr_outlier, is_variance_outlier, is_time_anomaly]
            )

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
                    "sample_size": len(amounts),
                },
                "description": f"Transaction amount ${transaction_amount:.2f} analyzed against {len(amounts)} similar transactions",
            }

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return {"is_anomaly": False, "error": str(e)}

    async def _analyze_time_patterns(
        self, session, transaction: TransactionModel, similar_transactions: List[Transaction]
    ) -> Dict[str, Any]:
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
            hour_frequency = (
                hour_counts.get(transaction_hour, 0) / total_transactions
                if total_transactions > 0
                else 0
            )
            weekday_frequency = (
                weekday_counts.get(transaction_weekday, 0) / total_transactions
                if total_transactions > 0
                else 0
            )

            # Flag as anomaly if time patterns are very unusual (< 5% of transactions)
            is_time_anomaly = hour_frequency < 0.05 or weekday_frequency < 0.05

            return {
                "is_anomaly": is_time_anomaly,
                "hour_frequency": hour_frequency,
                "weekday_frequency": weekday_frequency,
                "transaction_hour": transaction_hour,
                "transaction_weekday": transaction_weekday,
            }

        except Exception as e:
            self.logger.error(f"Error in time pattern analysis: {e}")
            return {"is_anomaly": False, "error": str(e)}

    async def _calculate_dynamic_confidence(
        self, session, decision_type: str, analysis_data: Dict[str, Any]
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
                "factors": factor_details,
            }

        except Exception as e:
            self.logger.error(f"Error calculating dynamic confidence: {e}")
            return {"score": 0.5, "factors": {}, "error": str(e)}

    async def _get_historical_accuracy(self, session, decision_type: str) -> float:
        """Get historical accuracy for decisions of this type."""
        try:
            # Get recent decisions of this type with outcomes
            type_outcomes = [
                outcome
                for decision_id, outcome in self.decision_outcomes.items()
                if outcome.get("decision_type") == decision_type
            ]

            if not type_outcomes:
                return 0.7  # Default moderate confidence

            # Calculate accuracy rate
            correct_decisions = sum(
                1 for outcome in type_outcomes if outcome.get("was_correct", False)
            )
            accuracy = correct_decisions / len(type_outcomes)

            return accuracy

        except Exception as e:
            self.logger.error(f"Error getting historical accuracy: {e}")
            return 0.5

    async def _calculate_seasonal_consistency(
        self, session, analysis_data: Dict[str, Any]
    ) -> float:
        """Calculate seasonal consistency factor."""
        try:
            # For now, return a moderate factor
            # In a full implementation, this would analyze seasonal patterns
            return 0.6

        except Exception as e:
            self.logger.error(f"Error calculating seasonal consistency: {e}")
            return 0.5

    async def _forecast_cash_flow(
        self, session, forecast_days: int = 30
    ) -> Optional[AgentDecision]:
        """Advanced cash flow forecasting with predictive analytics."""
        try:
            # Get historical transaction data
            end_date = datetime.now()
            start_date = end_date - timedelta(
                days=self.forecasting_config["seasonal_analysis_days"]
            )

            transactions = (
                session.query(Transaction)
                .filter(Transaction.transaction_date >= start_date)
                .order_by(Transaction.transaction_date)
                .all()
            )

            if len(transactions) < 7:  # Need at least a week of data
                return None

            # Prepare data for forecasting
            daily_flows = await self._prepare_daily_cash_flows(transactions)

            # Generate forecasts using multiple methods
            forecasts = await self._generate_cash_flow_forecasts(daily_flows, forecast_days)

            # Identify potential issues
            forecast_analysis = await self._analyze_forecast_results(
                session, forecasts, forecast_days
            )

            # Calculate confidence for forecast
            confidence = await self._calculate_forecast_confidence(daily_flows, forecasts)

            context = {
                "forecast_days": forecast_days,
                "historical_data_days": len(daily_flows),
                "forecasts": forecasts,
                "analysis": forecast_analysis,
                "confidence_score": confidence,
            }

            # Generate detailed analysis with Claude
            reasoning = await self.analyze_with_claude(
                f"Cash flow forecast analysis for next {forecast_days} days. "
                f"Forecast shows: {forecast_analysis['summary']}. "
                f"Confidence: {confidence:.2%}. Provide strategic recommendations.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="cash_flow_forecast",
                context=context,
                reasoning=reasoning,
                action=forecast_analysis["recommended_action"],
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Error in cash flow forecasting: {e}")
            return None

    async def _prepare_daily_cash_flows(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Prepare daily cash flow data for forecasting."""
        daily_flows = {}

        for transaction in transactions:
            date_key = transaction.transaction_date.date().isoformat()
            amount = float(transaction.amount)

            if transaction.transaction_type == TransactionType.INCOME:
                daily_flows[date_key] = daily_flows.get(date_key, 0) + amount
            elif transaction.transaction_type == TransactionType.EXPENSE:
                daily_flows[date_key] = daily_flows.get(date_key, 0) - amount

        return daily_flows

    async def _generate_cash_flow_forecasts(
        self, daily_flows: Dict[str, float], forecast_days: int
    ) -> Dict[str, Any]:
        """Generate cash flow forecasts using multiple methods."""
        try:
            flow_values = list(daily_flows.values())

            # Method 1: Simple moving average
            window_size = min(7, len(flow_values))
            recent_flows = flow_values[-window_size:]
            simple_avg = sum(recent_flows) / len(recent_flows)

            # Method 2: Weighted moving average (more weight to recent data)
            weights = [i + 1 for i in range(len(recent_flows))]
            weighted_avg = sum(flow * weight for flow, weight in zip(recent_flows, weights)) / sum(
                weights
            )

            # Method 3: Trend-based projection
            if len(flow_values) >= 14:
                recent_trend = self._calculate_trend(flow_values[-14:])  # 2-week trend
                trend_forecast = recent_flows[-1] + (recent_trend * forecast_days / 2)
            else:
                trend_forecast = simple_avg

            # Method 4: Seasonal adjustment (if enough data)
            seasonal_forecast = await self._apply_seasonal_adjustment(
                daily_flows, simple_avg, forecast_days
            )

            # Combine forecasts with weights
            ensemble_forecast = (
                simple_avg * 0.25
                + weighted_avg * 0.35
                + trend_forecast * 0.25
                + seasonal_forecast * 0.15
            )

            return {
                "simple_moving_average": simple_avg,
                "weighted_moving_average": weighted_avg,
                "trend_based": trend_forecast,
                "seasonal_adjusted": seasonal_forecast,
                "ensemble": ensemble_forecast,
                "daily_forecast": ensemble_forecast,
                "total_forecast": ensemble_forecast * forecast_days,
            }

        except Exception as e:
            self.logger.error(f"Error generating forecasts: {e}")
            return {"ensemble": 0, "total_forecast": 0}

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression."""
        try:
            n = len(values)
            if n < 2:
                return 0

            x_sum = sum(range(n))
            y_sum = sum(values)
            xy_sum = sum(i * values[i] for i in range(n))
            x_squared_sum = sum(i * i for i in range(n))

            slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)
            return slope

        except Exception as e:
            self.logger.error(f"Error calculating trend: {e}")
            return 0

    async def _apply_seasonal_adjustment(
        self, daily_flows: Dict[str, float], base_forecast: float, forecast_days: int
    ) -> float:
        """Apply seasonal adjustments to forecasts."""
        try:
            # For now, return base forecast
            # In full implementation, this would analyze seasonal patterns
            return base_forecast

        except Exception as e:
            self.logger.error(f"Error in seasonal adjustment: {e}")
            return base_forecast

    async def _analyze_forecast_results(
        self, session, forecasts: Dict[str, Any], forecast_days: int
    ) -> Dict[str, Any]:
        """Analyze forecast results and identify potential issues."""
        try:
            total_forecast = forecasts["total_forecast"]
            daily_forecast = forecasts["daily_forecast"]

            # Get current cash position
            current_cash = await self._get_current_cash_balance(session)
            projected_cash = current_cash + total_forecast

            # Analyze results
            is_shortage_predicted = projected_cash < self.alert_thresholds["cash_low"]
            shortage_severity = (
                "high" if projected_cash < 0 else "medium" if is_shortage_predicted else "low"
            )

            # Generate summary and recommendations
            if is_shortage_predicted:
                summary = f"Cash shortage predicted: ${projected_cash:.2f} in {forecast_days} days"
                recommended_action = "Urgent: Implement cash flow improvement measures"
            elif daily_forecast < 0:
                summary = f"Negative daily cash flow predicted: ${daily_forecast:.2f}/day"
                recommended_action = "Monitor cash flow closely and optimize expenses"
            else:
                summary = (
                    f"Positive cash flow predicted: ${total_forecast:.2f} over {forecast_days} days"
                )
                recommended_action = "Continue current financial management approach"

            return {
                "summary": summary,
                "current_cash": current_cash,
                "projected_cash": projected_cash,
                "is_shortage_predicted": is_shortage_predicted,
                "shortage_severity": shortage_severity,
                "recommended_action": recommended_action,
                "daily_forecast": daily_forecast,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing forecast results: {e}")
            return {
                "summary": "Error in forecast analysis",
                "recommended_action": "Review forecast data",
            }

    async def _get_current_cash_balance(self, session) -> float:
        """Get current total cash balance."""
        try:
            cash_accounts = (
                session.query(Account)
                .filter(Account.account_type.in_([AccountType.CHECKING, AccountType.SAVINGS]))
                .all()
            )

            return float(sum(account.balance for account in cash_accounts))

        except Exception as e:
            self.logger.error(f"Error getting cash balance: {e}")
            return 0.0

    async def _calculate_forecast_confidence(
        self, daily_flows: Dict[str, float], forecasts: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the forecast based on data quality and consistency."""
        try:
            # Factor 1: Data consistency (lower variance = higher confidence)
            flow_values = list(daily_flows.values())
            if len(flow_values) > 1:
                variance = statistics.variance(flow_values)
                mean_abs_flow = statistics.mean([abs(f) for f in flow_values])
                consistency_factor = max(
                    0.1, 1 - (variance / (mean_abs_flow**2)) if mean_abs_flow > 0 else 0.1
                )
            else:
                consistency_factor = 0.3

            # Factor 2: Data volume (more data = higher confidence)
            volume_factor = min(1.0, len(flow_values) / 30)

            # Factor 3: Forecast method agreement
            forecast_values = [
                forecasts["simple_moving_average"],
                forecasts["weighted_moving_average"],
                forecasts["trend_based"],
                forecasts["seasonal_adjusted"],
            ]

            if len(forecast_values) > 1:
                forecast_variance = statistics.variance(forecast_values)
                forecast_mean = statistics.mean([abs(f) for f in forecast_values])
                agreement_factor = max(
                    0.1, 1 - (forecast_variance / (forecast_mean**2)) if forecast_mean > 0 else 0.1
                )
            else:
                agreement_factor = 0.5

            # Combine factors
            confidence = consistency_factor * 0.4 + volume_factor * 0.3 + agreement_factor * 0.3

            return min(0.9, max(0.2, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating forecast confidence: {e}")
            return 0.5

    async def _analyze_financial_trends(self, session) -> Optional[AgentDecision]:
        """Analyze financial trends across multiple time periods."""
        try:
            self.forecasting_config["trend_analysis_periods"]
            end_date = datetime.now()

            # Analyze trends over different periods
            trend_analysis = {}

            for period in [7, 30, 90]:  # Weekly, monthly, quarterly trends
                start_date = end_date - timedelta(days=period)

                transactions = (
                    session.query(Transaction)
                    .filter(Transaction.transaction_date >= start_date)
                    .all()
                )

                if transactions:
                    trend_data = await self._calculate_period_trends(transactions, period)
                    trend_analysis[f"{period}_days"] = trend_data

            if not trend_analysis:
                return None

            # Identify significant trends
            significant_trends = await self._identify_significant_trends(trend_analysis)

            if not significant_trends:
                return None

            # Calculate confidence for trend analysis
            confidence = await self._calculate_trend_confidence(trend_analysis)

            context = {
                "trend_analysis": trend_analysis,
                "significant_trends": significant_trends,
                "analysis_periods": list(trend_analysis.keys()),
            }

            reasoning = await self.analyze_with_claude(
                f"Financial trend analysis reveals: {len(significant_trends)} significant trends. "
                f"Key findings: {', '.join(significant_trends)}. "
                f"Provide strategic recommendations based on these trends.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="financial_trend_analysis",
                context=context,
                reasoning=reasoning,
                action="Generate comprehensive trend analysis report",
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {e}")
            return None

    async def _calculate_period_trends(
        self, transactions: List[Transaction], period_days: int
    ) -> Dict[str, Any]:
        """Calculate trends for a specific period."""
        try:
            # Separate income and expenses
            income_transactions = [
                t for t in transactions if t.transaction_type == TransactionType.INCOME
            ]
            expense_transactions = [
                t for t in transactions if t.transaction_type == TransactionType.EXPENSE
            ]

            total_income = sum(float(t.amount) for t in income_transactions)
            total_expenses = sum(float(t.amount) for t in expense_transactions)
            net_flow = total_income - total_expenses

            # Calculate daily averages
            daily_income = total_income / period_days
            daily_expenses = total_expenses / period_days
            daily_net = net_flow / period_days

            # Calculate trends within the period
            income_trend = self._calculate_trend(
                [float(t.amount) for t in income_transactions[-14:]]
            )
            expense_trend = self._calculate_trend(
                [float(t.amount) for t in expense_transactions[-14:]]
            )

            # Analyze category distributions
            expense_categories = {}
            for t in expense_transactions:
                category = t.category or "uncategorized"
                expense_categories[category] = expense_categories.get(category, 0) + float(t.amount)

            return {
                "period_days": period_days,
                "total_income": total_income,
                "total_expenses": total_expenses,
                "net_flow": net_flow,
                "daily_income": daily_income,
                "daily_expenses": daily_expenses,
                "daily_net": daily_net,
                "income_trend": income_trend,
                "expense_trend": expense_trend,
                "transaction_count": len(transactions),
                "expense_categories": expense_categories,
            }

        except Exception as e:
            self.logger.error(f"Error calculating period trends: {e}")
            return {}

    async def _identify_significant_trends(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """Identify significant trends that require attention."""
        significant_trends = []

        try:
            # Compare different periods
            periods = sorted(trend_analysis.keys())

            for period in periods:
                data = trend_analysis[period]

                # Check for negative trends
                if data.get("income_trend", 0) < -50:  # Declining income
                    significant_trends.append(f"Declining income trend in {period}")

                if data.get("expense_trend", 0) > 50:  # Rising expenses
                    significant_trends.append(f"Rising expense trend in {period}")

                if data.get("daily_net", 0) < -100:  # Negative daily net flow
                    significant_trends.append(f"Negative cash flow in {period}")

            # Compare short-term vs long-term trends
            if "7_days" in trend_analysis and "30_days" in trend_analysis:
                short_term = trend_analysis["7_days"]
                long_term = trend_analysis["30_days"]

                if short_term.get("daily_net", 0) < long_term.get("daily_net", 0) * 0.7:
                    significant_trends.append("Short-term performance decline")

            return significant_trends

        except Exception as e:
            self.logger.error(f"Error identifying significant trends: {e}")
            return []

    async def _calculate_trend_confidence(self, trend_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in trend analysis."""
        try:
            # Base confidence on data volume and consistency
            total_transactions = sum(
                data.get("transaction_count", 0) for data in trend_analysis.values()
            )

            volume_factor = min(1.0, total_transactions / 50)  # Max confidence at 50+ transactions

            # Check consistency across periods
            if len(trend_analysis) >= 2:
                consistency_checks = 0
                total_checks = 0

                for period1, period2 in zip(
                    list(trend_analysis.keys())[:-1], list(trend_analysis.keys())[1:]
                ):
                    data1 = trend_analysis[period1]
                    data2 = trend_analysis[period2]

                    # Check if trends are consistent
                    if (data1.get("income_trend", 0) * data2.get("income_trend", 0)) >= 0:
                        consistency_checks += 1
                    if (data1.get("expense_trend", 0) * data2.get("expense_trend", 0)) >= 0:
                        consistency_checks += 1

                    total_checks += 2

                consistency_factor = consistency_checks / total_checks if total_checks > 0 else 0.5
            else:
                consistency_factor = 0.5

            confidence = volume_factor * 0.6 + consistency_factor * 0.4
            return min(0.9, max(0.3, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating trend confidence: {e}")
            return 0.5

    async def _process_decision_outcome(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """Process feedback on decision outcomes for learning."""
        try:
            decision_id = data.get("decision_id")
            was_correct = data.get("was_correct", False)
            feedback_notes = data.get("feedback_notes", "")
            decision_type = data.get("decision_type")

            if not decision_id:
                return None

            # Store outcome for future confidence calculations
            self.decision_outcomes[decision_id] = {
                "was_correct": was_correct,
                "feedback_notes": feedback_notes,
                "decision_type": decision_type,
                "timestamp": datetime.now(),
            }

            # Analyze patterns in decision outcomes
            outcome_analysis = await self._analyze_decision_patterns()

            if outcome_analysis["needs_adjustment"]:
                context = {
                    "decision_id": decision_id,
                    "outcome": {"was_correct": was_correct, "feedback": feedback_notes},
                    "pattern_analysis": outcome_analysis,
                }

                reasoning = await self.analyze_with_claude(
                    f"Decision outcome received: {'Correct' if was_correct else 'Incorrect'}. "
                    f"Pattern analysis suggests: {outcome_analysis['recommendation']}. "
                    f"How should the decision-making process be adjusted?",
                    context,
                )

                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="decision_learning",
                    context=context,
                    reasoning=reasoning,
                    action="Adjust decision-making parameters based on feedback",
                    confidence=0.8,
                )

            return None

        except Exception as e:
            self.logger.error(f"Error processing decision outcome: {e}")
            return None

    async def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in decision outcomes to identify improvement opportunities."""
        try:
            if len(self.decision_outcomes) < 5:
                return {"needs_adjustment": False, "recommendation": "Insufficient data"}

            # Analyze accuracy by decision type
            type_accuracy = {}
            for outcome in self.decision_outcomes.values():
                decision_type = outcome.get("decision_type", "unknown")
                if decision_type not in type_accuracy:
                    type_accuracy[decision_type] = {"correct": 0, "total": 0}

                type_accuracy[decision_type]["total"] += 1
                if outcome.get("was_correct", False):
                    type_accuracy[decision_type]["correct"] += 1

            # Identify types with low accuracy
            low_accuracy_types = []
            for decision_type, stats in type_accuracy.items():
                accuracy = stats["correct"] / stats["total"]
                if accuracy < 0.7 and stats["total"] >= 3:  # At least 3 decisions to be significant
                    low_accuracy_types.append((decision_type, accuracy))

            needs_adjustment = len(low_accuracy_types) > 0

            if needs_adjustment:
                recommendation = f"Low accuracy in: {', '.join([f'{t[0]} ({t[1]:.1%})' for t in low_accuracy_types])}"
            else:
                recommendation = "Decision accuracy is acceptable across all types"

            return {
                "needs_adjustment": needs_adjustment,
                "recommendation": recommendation,
                "type_accuracy": type_accuracy,
                "low_accuracy_types": low_accuracy_types,
                "total_decisions": len(self.decision_outcomes),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing decision patterns: {e}")
            return {"needs_adjustment": False, "recommendation": "Error in pattern analysis"}

    async def periodic_check(self):
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
