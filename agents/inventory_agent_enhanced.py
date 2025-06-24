import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from agents.base_agent import AgentDecision, BaseAgent
from models.inventory import (
    Item,
    ItemStatus,
    PurchaseOrder,
    StockMovement,
    StockMovementType,
    Supplier,
)

logger = logging.getLogger(__name__)


@dataclass
class DemandForecast:
    """Demand forecast result with comprehensive analytics."""

    item_id: str
    predicted_demand: float
    confidence_interval: Tuple[float, float]
    seasonality_factor: float
    trend_factor: float
    forecast_horizon_days: int
    forecast_accuracy: float
    historical_patterns: Dict[str, float]
    revenue_correlation: float
    method_used: str


@dataclass
class OptimalReorderPoint:
    """Optimal reorder point calculation with service level optimization."""

    item_id: str
    optimal_reorder_point: int
    optimal_reorder_quantity: int
    service_level: float
    safety_stock: int
    lead_time_demand: float
    demand_variability: float
    total_cost: float
    holding_cost: float
    ordering_cost: float
    stockout_cost: float


@dataclass
class BulkPurchaseOptimization:
    """Bulk purchase optimization with discount analysis."""

    item_id: str
    optimal_order_quantity: int
    unit_cost_with_discount: Decimal
    total_cost_savings: Decimal
    break_even_point: int
    holding_cost_impact: Decimal
    discount_tier: str
    roi_months: float


@dataclass
class ExpiryIntelligence:
    """Advanced expiry management analytics."""

    item_id: str
    predicted_waste_amount: float
    optimal_ordering_frequency: int
    risk_score: float
    recommended_discount_timing: int
    shelf_life_optimization: Dict[str, Any]
    rotation_efficiency: float


@dataclass
class SupplierPerformance:
    """Comprehensive supplier analytics."""

    supplier_id: str
    overall_score: float
    reliability_score: float
    cost_competitiveness: float
    quality_score: float
    delivery_performance: float
    risk_assessment: Dict[str, float]
    recommendation: str


class EnhancedInventoryAgent(BaseAgent):
    """Enhanced Inventory Management Agent with advanced analytics and
    optimization."""

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        config: Dict[str, Any],
        db_url: str = "sqlite:///business_simulation.db",
        message_queue: Optional[asyncio.Queue] = None,
    ):
        super().__init__(agent_id, api_key, config, db_url, message_queue)

        # Enhanced configuration
        self.low_stock_multiplier = config.get("low_stock_multiplier", 1.2)
        self.reorder_lead_time = config.get("reorder_lead_time", 7)
        self.consumption_analysis_days = config.get("consumption_analysis_days", 30)
        self.service_level_target = config.get("service_level_target", 0.95)
        self.holding_cost_rate = config.get("holding_cost_rate", 0.25)  # 25% annual
        self.ordering_cost = config.get("ordering_cost", 50.0)  # $50 per order
        self.stockout_cost_multiplier = config.get("stockout_cost_multiplier", 3.0)

        # Forecasting parameters
        self.forecast_horizon_days = config.get("forecast_horizon_days", 30)
        self.seasonal_analysis_periods = config.get("seasonal_analysis_periods", 4)
        self.trend_analysis_days = config.get("trend_analysis_days", 90)

        # Bulk purchase parameters
        self.bulk_discount_tiers = config.get(
            "bulk_discount_tiers",
            {
                "100": 0.02,  # 2% discount for 100+ units
                "250": 0.05,  # 5% discount for 250+ units
                "500": 0.08,  # 8% discount for 500+ units
                "1000": 0.12,  # 12% discount for 1000+ units
            },
        )

        logger.info(f"Enhanced InventoryAgent {agent_id} initialized with advanced analytics")

    @property
    def system_prompt(self) -> str:
        """Enhanced system prompt describing advanced capabilities."""
        return f"""You are an AI Inventory Management Agent with advanced analytics capabilities for {self.agent_id}.

Your enhanced responsibilities include:

CORE FUNCTIONS:
- Advanced demand forecasting using multiple statistical methods
- Optimal reorder point calculation with service level optimization
- Bulk purchase optimization with discount analysis
- Intelligent expiry management and waste reduction
- Comprehensive supplier performance analytics

ADVANCED ANALYTICS:
- Multi-method demand forecasting (moving average, trend analysis, seasonal patterns)
- Dynamic safety stock calculation based on demand variability
- EOQ optimization with holding and ordering cost analysis
- Predictive expiry management with rotation optimization
- Supplier scoring using reliability, cost, quality, and delivery metrics

DECISION MAKING:
- Prioritize inventory optimization to minimize total cost while maintaining service levels
- Consider seasonal patterns, trends, and business growth in forecasting
- Optimize bulk purchasing decisions based on discount tiers and cash flow
- Proactively manage expiry risks through intelligent ordering and pricing
- Recommend supplier selection based on comprehensive performance analysis

INTELLIGENCE CAPABILITIES:
- Learn from historical demand patterns and forecast accuracy
- Adapt reorder points based on actual service level performance
- Optimize inventory turnover while minimizing stockouts
- Predict and prevent waste through advanced expiry analytics
- Continuous supplier performance monitoring and optimization

Provide detailed analysis with confidence scores, reasoning, and specific recommendations.
Focus on data-driven decisions that optimize total inventory cost and business performance."""

    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        """Enhanced data processing with advanced analytics."""
        session = None
        try:
            session = self.SessionLocal()

            data_type = data.get("type")

            if data_type == "stock_movement":
                return await self._analyze_stock_movement_enhanced(
                    session, data.get("movement", {})
                )
            elif data_type == "demand_forecast_request":
                return await self._perform_demand_forecasting(session, data)
            elif data_type == "reorder_optimization":
                return await self._optimize_reorder_points(session)
            elif data_type == "bulk_purchase_analysis":
                return await self._analyze_bulk_purchase_opportunities(session)
            elif data_type == "expiry_management":
                return await self._perform_expiry_intelligence(session)
            elif data_type == "supplier_performance_review":
                return await self._analyze_supplier_performance(session)
            elif data_type == "inventory_health_check":
                return await self._comprehensive_inventory_analysis(session)
            else:
                # Fall back to enhanced reorder analysis
                return await self._analyze_reorder_needs_enhanced(session)

        except Exception as e:
            logger.error(f"Error processing inventory data: {e}")
            return None
        finally:
            if session:
                session.close()

    async def _perform_demand_forecasting(
        self, session: Session, data: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        """Perform comprehensive demand forecasting for inventory items."""
        try:
            item_id = data.get("item_id")
            forecast_days = data.get("forecast_days", self.forecast_horizon_days)

            if item_id:
                # Forecast for specific item
                forecast = await self._forecast_item_demand(session, item_id, forecast_days)
                forecasts = [forecast] if forecast else []
            else:
                # Forecast for all active items
                forecasts = await self._forecast_all_items_demand(session, forecast_days)

            if not forecasts:
                return None

            # Analyze forecasts and generate recommendations
            high_demand_items = [
                f
                for f in forecasts
                if f.predicted_demand > f.historical_patterns.get("avg_daily", 0) * 1.5
            ]
            declining_items = [f for f in forecasts if f.trend_factor < -0.1]
            low_accuracy_items = [f for f in forecasts if f.forecast_accuracy < 0.7]

            context = {
                "total_items_analyzed": len(forecasts),
                "high_demand_items": len(high_demand_items),
                "declining_demand_items": len(declining_items),
                "forecast_period_days": forecast_days,
                "average_accuracy": sum(f.forecast_accuracy for f in forecasts) / len(forecasts),
                "forecasts": [f.__dict__ for f in forecasts[:10]],  # Top 10 for context
            }

            # Generate Claude analysis
            analysis_prompt = f"""
            Demand forecasting analysis for {len(forecasts)} inventory items over {forecast_days} days:

            Key Findings:
            - {len(high_demand_items)} items showing high demand growth (>50% increase)
            - {len(declining_items)} items with declining demand trends
            - Average forecast accuracy: {context['average_accuracy']:.1%}
            - {len(low_accuracy_items)} items with low forecast accuracy (<70%)

            Recommend inventory adjustments and purchasing strategies.
            """

            reasoning = await self.analyze_with_claude(analysis_prompt, context)

            # Determine confidence based on forecast accuracy and data quality
            confidence = min(0.9, max(0.3, context["average_accuracy"] * 0.8 + 0.2))

            action_items = []
            if high_demand_items:
                action_items.append(
                    f"Increase stock levels for {len(high_demand_items)} high-demand items"
                )
            if declining_items:
                action_items.append(f"Review {len(declining_items)} items with declining demand")
            if low_accuracy_items:
                action_items.append(
                    f"Improve data collection for {len(low_accuracy_items)} items with low forecast accuracy"
                )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="demand_forecast_analysis",
                context=context,
                reasoning=reasoning,
                action=(
                    "; ".join(action_items) if action_items else "Monitor current inventory levels"
                ),
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error in demand forecasting: {e}")
            return None

    async def _forecast_item_demand(
        self, session: Session, item_id: str, forecast_days: int
    ) -> Optional[DemandForecast]:
        """Forecast demand for a specific item using multiple methods."""
        try:
            # Get historical consumption data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.trend_analysis_days)

            movements = (
                session.query(StockMovement)
                .filter(
                    StockMovement.item_id == item_id,
                    StockMovement.movement_type == StockMovementType.OUT,
                    StockMovement.movement_date >= start_date,
                )
                .order_by(StockMovement.movement_date)
                .all()
            )

            if len(movements) < 7:  # Need at least a week of data
                return None

            # Prepare daily consumption data
            daily_consumption = defaultdict(float)
            for movement in movements:
                day = movement.movement_date.date()
                daily_consumption[day] += float(movement.quantity)

            # Fill missing days with 0
            current_date = start_date
            consumption_series = []
            while current_date <= end_date:
                consumption_series.append(daily_consumption.get(current_date, 0.0))
                current_date += timedelta(days=1)

            if len(consumption_series) < 14:  # Need at least 2 weeks
                return None

            # Method 1: Simple moving average
            window_size = min(14, len(consumption_series) // 2)
            sma_forecast = np.mean(consumption_series[-window_size:]) * forecast_days

            # Method 2: Weighted moving average (more weight to recent data)
            weights = np.exp(np.linspace(-1, 0, window_size))
            weights = weights / weights.sum()
            wma_forecast = (
                np.average(consumption_series[-window_size:], weights=weights) * forecast_days
            )

            # Method 3: Trend-based forecast
            x = np.arange(len(consumption_series))
            y = np.array(consumption_series)

            if len(x) > 1 and np.std(y) > 0:
                trend_coef = np.polyfit(x, y, 1)
                trend_forecast = (
                    trend_coef[0] * (len(x) + forecast_days / 2) + trend_coef[1]
                ) * forecast_days
                trend_factor = trend_coef[0] / (np.mean(y) + 1e-6)  # Avoid division by zero
            else:
                trend_forecast = sma_forecast
                trend_factor = 0.0

            # Method 4: Seasonal adjustment
            if len(consumption_series) >= 28:  # Need at least 4 weeks for weekly seasonality
                weekly_pattern = self._calculate_weekly_seasonality(consumption_series)
                seasonal_factor = weekly_pattern.get(end_date.weekday(), 1.0)
            else:
                seasonal_factor = 1.0

            # Ensemble forecast (weighted average of methods)
            forecasts = [sma_forecast, wma_forecast, trend_forecast]
            weights = [0.3, 0.4, 0.3]  # Favor weighted moving average
            ensemble_forecast = np.average(forecasts, weights=weights)

            # Apply seasonal adjustment
            adjusted_forecast = ensemble_forecast * seasonal_factor

            # Calculate confidence interval (±1 standard deviation)
            std_dev = np.std(consumption_series[-window_size:]) * np.sqrt(forecast_days)
            confidence_interval = (max(0, adjusted_forecast - std_dev), adjusted_forecast + std_dev)

            # Calculate forecast accuracy based on recent predictions vs actuals
            forecast_accuracy = self._calculate_forecast_accuracy(consumption_series)

            # Calculate revenue correlation if revenue data available
            revenue_correlation = await self._calculate_revenue_correlation(
                session, item_id, movements
            )

            # Historical patterns
            historical_patterns = {
                "avg_daily": np.mean(consumption_series),
                "std_daily": np.std(consumption_series),
                "min_daily": np.min(consumption_series),
                "max_daily": np.max(consumption_series),
                "trend_slope": trend_factor,
            }

            return DemandForecast(
                item_id=item_id,
                predicted_demand=adjusted_forecast,
                confidence_interval=confidence_interval,
                seasonality_factor=seasonal_factor,
                trend_factor=trend_factor,
                forecast_horizon_days=forecast_days,
                forecast_accuracy=forecast_accuracy,
                historical_patterns=historical_patterns,
                revenue_correlation=revenue_correlation,
                method_used="ensemble_with_seasonality",
            )

        except Exception as e:
            logger.error(f"Error forecasting demand for item {item_id}: {e}")
            return None

    async def _forecast_all_items_demand(
        self, session: Session, forecast_days: int
    ) -> List[DemandForecast]:
        """Forecast demand for all active items."""
        try:
            # Get all active items with recent movement
            recent_date = datetime.now().date() - timedelta(days=30)

            active_items = (
                session.query(Item.id)
                .filter(Item.status == ItemStatus.ACTIVE)
                .join(StockMovement, Item.id == StockMovement.item_id)
                .filter(StockMovement.movement_date >= recent_date)
                .distinct()
                .all()
            )

            forecasts = []
            for (item_id,) in active_items:
                forecast = await self._forecast_item_demand(session, item_id, forecast_days)
                if forecast:
                    forecasts.append(forecast)

            return forecasts

        except Exception as e:
            logger.error(f"Error forecasting demand for all items: {e}")
            return []

    def _calculate_weekly_seasonality(self, consumption_series: List[float]) -> Dict[int, float]:
        """Calculate weekly seasonality factors (0=Monday, 6=Sunday)"""
        try:
            # Group by day of week
            daily_totals = defaultdict(list)

            for i, consumption in enumerate(consumption_series):
                day_of_week = i % 7  # Assuming data starts on Monday
                daily_totals[day_of_week].append(consumption)

            # Calculate average for each day
            weekly_averages = {}
            overall_average = np.mean(consumption_series)

            for day, values in daily_totals.items():
                if values and overall_average > 0:
                    weekly_averages[day] = np.mean(values) / overall_average
                else:
                    weekly_averages[day] = 1.0

            return weekly_averages

        except Exception as e:
            logger.error(f"Error calculating weekly seasonality: {e}")
            return dict.fromkeys(range(7), 1.0)

    def _calculate_forecast_accuracy(self, consumption_series: List[float]) -> float:
        """Calculate forecast accuracy based on recent prediction
        performance."""
        try:
            if len(consumption_series) < 21:  # Need at least 3 weeks
                return 0.7  # Default moderate accuracy

            # Use last week as test data
            train_data = consumption_series[:-7]
            test_data = consumption_series[-7:]

            # Simple forecast using training data
            predicted = np.mean(train_data[-14:])  # Average of last 2 weeks

            # Calculate MAPE (Mean Absolute Percentage Error)
            errors = []
            for actual in test_data:
                if actual > 0:
                    error = abs(actual - predicted) / actual
                    errors.append(error)

            if errors:
                mape = np.mean(errors)
                accuracy = max(0.1, 1.0 - mape)  # Convert MAPE to accuracy
            else:
                accuracy = 0.7  # Default

            return min(0.95, accuracy)

        except Exception as e:
            logger.error(f"Error calculating forecast accuracy: {e}")
            return 0.7

    async def _calculate_revenue_correlation(
        self, session: Session, item_id: str, movements: List[StockMovement]
    ) -> float:
        """Calculate correlation between item consumption and overall
        revenue."""
        try:
            # This is a simplified calculation - in a real system, you'd have revenue data
            # For now, assume correlation based on movement frequency and volume

            if len(movements) < 14:
                return 0.5  # Default moderate correlation

            # Calculate consumption volatility as proxy for revenue correlation
            daily_consumption = [float(m.quantity) for m in movements]
            cv = np.std(daily_consumption) / (np.mean(daily_consumption) + 1e-6)

            # Higher coefficient of variation suggests lower correlation with steady revenue
            correlation = max(0.1, 1.0 - cv)
            return min(0.9, correlation)

        except Exception as e:
            logger.error(f"Error calculating revenue correlation: {e}")
            return 0.5

    async def _optimize_reorder_points(self, session: Session) -> Optional[AgentDecision]:
        """Optimize reorder points for all items using advanced analytics."""
        try:
            # Get all active items that need optimization
            items = (
                session.query(Item)
                .filter(Item.status == ItemStatus.ACTIVE, Item.reorder_point.isnot(None))
                .all()
            )

            if not items:
                return None

            optimizations = []
            total_cost_savings = 0.0

            for item in items:
                optimization = await self._calculate_optimal_reorder_point(session, item)
                if optimization:
                    optimizations.append(optimization)

                    # Calculate potential cost savings
                    current_total_cost = self._calculate_current_inventory_cost(item)
                    savings = current_total_cost - optimization.total_cost
                    total_cost_savings += max(0, savings)

            if not optimizations:
                return None

            # Analyze optimization results
            significant_changes = [
                opt
                for opt in optimizations
                if abs(opt.optimal_reorder_point - item.reorder_point) > item.reorder_point * 0.1
            ]

            high_service_level = [opt for opt in optimizations if opt.service_level > 0.98]
            cost_effective = [
                opt
                for opt in optimizations
                if opt.total_cost
                < self._calculate_current_inventory_cost_by_id(session, opt.item_id)
            ]

            context = {
                "total_items_analyzed": len(optimizations),
                "significant_changes_needed": len(significant_changes),
                "high_service_level_items": len(high_service_level),
                "cost_effective_changes": len(cost_effective),
                "total_potential_savings": total_cost_savings,
                "optimization_details": [opt.__dict__ for opt in optimizations[:10]],
            }

            analysis_prompt = f"""
            Reorder point optimization analysis for {len(optimizations)} inventory items:

            Key Findings:
            - {len(significant_changes)} items need significant reorder point changes (>10%)
            - {len(cost_effective)} items show potential cost savings
            - Total potential annual savings: ${total_cost_savings:,.2f}
            - {len(high_service_level)} items achieving >98% service level

            Recommend implementation priorities and change management approach.
            """

            reasoning = await self.analyze_with_claude(analysis_prompt, context)

            confidence = 0.85 if len(optimizations) > 10 else 0.75

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="reorder_point_optimization",
                context=context,
                reasoning=reasoning,
                action=f"Optimize reorder points for {len(significant_changes)} items with potential savings of ${total_cost_savings:,.2f}",
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error optimizing reorder points: {e}")
            return None

    async def _calculate_optimal_reorder_point(
        self, session: Session, item: Item
    ) -> Optional[OptimalReorderPoint]:
        """Calculate optimal reorder point using EOQ and service level
        optimization."""
        try:
            # Get demand forecast
            forecast = await self._forecast_item_demand(session, item.id, self.reorder_lead_time)
            if not forecast:
                return None

            # Extract demand parameters
            daily_demand = forecast.predicted_demand / self.reorder_lead_time
            demand_std = forecast.historical_patterns["std_daily"]
            lead_time_demand = daily_demand * self.reorder_lead_time

            # Calculate demand variability during lead time
            lead_time_std = demand_std * np.sqrt(self.reorder_lead_time)

            # Calculate safety stock for target service level
            z_score = self._get_z_score_for_service_level(self.service_level_target)
            safety_stock = z_score * lead_time_std

            # Optimal reorder point
            optimal_reorder_point = int(lead_time_demand + safety_stock)

            # EOQ calculation
            annual_demand = daily_demand * 365
            holding_cost_per_unit = float(item.unit_cost) * self.holding_cost_rate

            if holding_cost_per_unit > 0:
                eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / holding_cost_per_unit)
                optimal_reorder_quantity = max(item.minimum_stock or 1, int(eoq))
            else:
                optimal_reorder_quantity = item.reorder_quantity

            # Calculate total costs
            holding_cost = (optimal_reorder_quantity / 2 + safety_stock) * holding_cost_per_unit
            ordering_cost = (annual_demand / optimal_reorder_quantity) * self.ordering_cost

            # Estimate stockout cost
            stockout_probability = 1 - self.service_level_target
            stockout_cost = (
                stockout_probability
                * annual_demand
                * float(item.unit_cost)
                * self.stockout_cost_multiplier
            )

            total_cost = holding_cost + ordering_cost + stockout_cost

            return OptimalReorderPoint(
                item_id=item.id,
                optimal_reorder_point=optimal_reorder_point,
                optimal_reorder_quantity=optimal_reorder_quantity,
                service_level=self.service_level_target,
                safety_stock=int(safety_stock),
                lead_time_demand=lead_time_demand,
                demand_variability=lead_time_std,
                total_cost=total_cost,
                holding_cost=holding_cost,
                ordering_cost=ordering_cost,
                stockout_cost=stockout_cost,
            )

        except Exception as e:
            logger.error(f"Error calculating optimal reorder point for item {item.id}: {e}")
            return None

    def _get_z_score_for_service_level(self, service_level: float) -> float:
        """Get Z-score for given service level (assuming normal
        distribution)"""
        # Common service levels and their Z-scores
        service_level_map = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.99: 2.33, 0.995: 2.58}

        # Find closest service level
        closest_level = min(service_level_map.keys(), key=lambda x: abs(x - service_level))
        return service_level_map[closest_level]

    def _calculate_current_inventory_cost(self, item: Item) -> float:
        """Calculate current inventory cost for comparison."""
        try:
            annual_demand = 365 * 10  # Estimate based on reorder quantity
            holding_cost_per_unit = float(item.unit_cost) * self.holding_cost_rate

            current_holding_cost = (item.reorder_quantity / 2) * holding_cost_per_unit
            current_ordering_cost = (annual_demand / item.reorder_quantity) * self.ordering_cost

            return current_holding_cost + current_ordering_cost

        except Exception as e:
            logger.error(f"Error calculating current inventory cost: {e}")
            return 0.0

    def _calculate_current_inventory_cost_by_id(self, session: Session, item_id: str) -> float:
        """Calculate current inventory cost by item ID."""
        try:
            item = session.query(Item).filter(Item.id == item_id).first()
            if item:
                return self._calculate_current_inventory_cost(item)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating current inventory cost for {item_id}: {e}")
            return 0.0

    async def _analyze_bulk_purchase_opportunities(
        self, session: Session
    ) -> Optional[AgentDecision]:
        """Analyze bulk purchase opportunities with discount optimization."""
        try:
            # Get items that might benefit from bulk purchasing
            items = (
                session.query(Item)
                .filter(
                    Item.status == ItemStatus.ACTIVE, Item.current_stock <= Item.reorder_point * 1.5
                )
                .all()
            )

            if not items:
                return None

            bulk_opportunities = []
            total_savings = 0.0

            for item in items:
                optimization = await self._calculate_bulk_purchase_optimization(session, item)
                if optimization and optimization.total_cost_savings > 0:
                    bulk_opportunities.append(optimization)
                    total_savings += float(optimization.total_cost_savings)

            if not bulk_opportunities:
                return None

            # Sort by savings potential
            bulk_opportunities.sort(key=lambda x: x.total_cost_savings, reverse=True)

            high_value_opportunities = [
                opt for opt in bulk_opportunities if opt.total_cost_savings > 100
            ]
            quick_roi_opportunities = [opt for opt in bulk_opportunities if opt.roi_months < 6]

            context = {
                "total_opportunities": len(bulk_opportunities),
                "high_value_opportunities": len(high_value_opportunities),
                "quick_roi_opportunities": len(quick_roi_opportunities),
                "total_potential_savings": total_savings,
                "top_opportunities": [opt.__dict__ for opt in bulk_opportunities[:5]],
            }

            analysis_prompt = f"""
            Bulk purchase optimization analysis for {len(bulk_opportunities)} items:

            Key Findings:
            - Total potential savings: ${total_savings:,.2f}
            - {len(high_value_opportunities)} high-value opportunities (>$100 savings)
            - {len(quick_roi_opportunities)} quick ROI opportunities (<6 months)

            Recommend bulk purchase strategy and cash flow considerations.
            """

            reasoning = await self.analyze_with_claude(analysis_prompt, context)

            confidence = 0.8 if len(bulk_opportunities) > 5 else 0.7

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="bulk_purchase_optimization",
                context=context,
                reasoning=reasoning,
                action=f"Implement bulk purchasing for {len(high_value_opportunities)} high-value items with potential savings of ${total_savings:,.2f}",
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error analyzing bulk purchase opportunities: {e}")
            return None

    async def _calculate_bulk_purchase_optimization(
        self, session: Session, item: Item
    ) -> Optional[BulkPurchaseOptimization]:
        """Calculate optimal bulk purchase quantity with discount analysis."""
        try:
            # Get demand forecast
            forecast = await self._forecast_item_demand(session, item.id, 90)  # 3-month forecast
            if not forecast:
                return None

            quarterly_demand = forecast.predicted_demand
            base_unit_cost = float(item.unit_cost)

            best_savings = 0.0
            best_optimization = None

            # Analyze each discount tier
            for tier_quantity, discount_rate in self.bulk_discount_tiers.items():
                tier_qty = int(tier_quantity)

                # Skip if tier quantity is much larger than reasonable demand
                if tier_qty > quarterly_demand * 2:
                    continue

                discounted_cost = base_unit_cost * (1 - discount_rate)
                immediate_savings = (base_unit_cost - discounted_cost) * tier_qty

                # Calculate holding cost impact for excess inventory
                excess_inventory = max(0, tier_qty - quarterly_demand)
                holding_cost_impact = (
                    excess_inventory * discounted_cost * self.holding_cost_rate * 0.25
                )  # 3 months

                net_savings = immediate_savings - holding_cost_impact

                # Calculate ROI in months
                if net_savings > 0:
                    roi_months = (tier_qty * discounted_cost) / (
                        net_savings * 4
                    )  # Quarterly to monthly
                else:
                    roi_months = float("inf")

                if net_savings > best_savings:
                    best_savings = net_savings
                    best_optimization = BulkPurchaseOptimization(
                        item_id=item.id,
                        optimal_order_quantity=tier_qty,
                        unit_cost_with_discount=Decimal(str(discounted_cost)),
                        total_cost_savings=Decimal(str(net_savings)),
                        break_even_point=int(tier_qty * 0.8),  # Estimate
                        holding_cost_impact=Decimal(str(holding_cost_impact)),
                        discount_tier=f"{tier_quantity}+ units ({discount_rate:.1%} discount)",
                        roi_months=roi_months,
                    )

            return best_optimization

        except Exception as e:
            logger.error(f"Error calculating bulk purchase optimization for item {item.id}: {e}")
            return None

    async def _perform_expiry_intelligence(self, session: Session) -> Optional[AgentDecision]:
        """Perform advanced expiry management and waste minimization."""
        try:
            # Get items with expiry dates
            items_with_expiry = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()

            # Filter items that have expiry considerations (perishables)
            perishable_items = [
                item
                for item in items_with_expiry
                if hasattr(item, "expiry_days") and getattr(item, "expiry_days", None)
            ]

            if not perishable_items:
                return None

            expiry_analyses = []
            total_waste_risk = 0.0

            for item in perishable_items:
                analysis = await self._analyze_expiry_intelligence(session, item)
                if analysis:
                    expiry_analyses.append(analysis)
                    total_waste_risk += analysis.predicted_waste_amount

            if not expiry_analyses:
                return None

            # Sort by risk score
            expiry_analyses.sort(key=lambda x: x.risk_score, reverse=True)

            high_risk_items = [
                analysis for analysis in expiry_analyses if analysis.risk_score > 0.7
            ]
            discount_candidates = [
                analysis
                for analysis in expiry_analyses
                if analysis.recommended_discount_timing <= 3
            ]

            context = {
                "total_perishable_items": len(expiry_analyses),
                "high_risk_items": len(high_risk_items),
                "discount_candidates": len(discount_candidates),
                "total_predicted_waste": total_waste_risk,
                "expiry_analyses": [analysis.__dict__ for analysis in expiry_analyses[:10]],
            }

            analysis_prompt = f"""
            Expiry management analysis for {len(expiry_analyses)} perishable items:

            Key Findings:
            - {len(high_risk_items)} items at high risk of expiry (>70% risk score)
            - {len(discount_candidates)} items should be discounted within 3 days
            - Total predicted waste amount: {total_waste_risk:.1f} units

            Recommend pricing strategies and ordering adjustments to minimize waste.
            """

            reasoning = await self.analyze_with_claude(analysis_prompt, context)

            confidence = 0.75

            action_items = []
            if high_risk_items:
                action_items.append(
                    f"Implement immediate discount strategy for {len(high_risk_items)} high-risk items"
                )
            if discount_candidates:
                action_items.append(
                    f"Schedule price reductions for {len(discount_candidates)} items"
                )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="expiry_intelligence_analysis",
                context=context,
                reasoning=reasoning,
                action="; ".join(action_items) if action_items else "Monitor expiry schedules",
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error performing expiry intelligence: {e}")
            return None

    async def _analyze_expiry_intelligence(
        self, session: Session, item: Item
    ) -> Optional[ExpiryIntelligence]:
        """Analyze expiry intelligence for a specific item."""
        try:
            expiry_days = getattr(item, "expiry_days", 30)  # Default 30 days if not specified

            # Get demand forecast
            forecast = await self._forecast_item_demand(session, item.id, expiry_days)
            if not forecast:
                return None

            daily_consumption = forecast.predicted_demand / expiry_days
            current_stock = item.current_stock

            # Calculate predicted waste
            if daily_consumption > 0:
                days_to_consume = current_stock / daily_consumption
                if days_to_consume > expiry_days:
                    predicted_waste = current_stock - (daily_consumption * expiry_days)
                else:
                    predicted_waste = 0.0
            else:
                predicted_waste = current_stock  # All will expire if no consumption

            # Calculate risk score (0-1, where 1 is highest risk)
            risk_factors = []

            # Factor 1: Time to expiry vs consumption rate
            if daily_consumption > 0:
                consumption_risk = min(1.0, days_to_consume / expiry_days)
            else:
                consumption_risk = 1.0
            risk_factors.append(consumption_risk * 0.4)

            # Factor 2: Current stock level relative to normal
            if item.maximum_stock > 0:
                stock_level_risk = current_stock / item.maximum_stock
            else:
                stock_level_risk = 0.5
            risk_factors.append(stock_level_risk * 0.3)

            # Factor 3: Demand volatility
            demand_volatility = forecast.historical_patterns["std_daily"] / (
                forecast.historical_patterns["avg_daily"] + 1e-6
            )
            volatility_risk = min(1.0, demand_volatility)
            risk_factors.append(volatility_risk * 0.3)

            risk_score = sum(risk_factors)

            # Calculate optimal ordering frequency
            if daily_consumption > 0:
                optimal_frequency = int(
                    expiry_days * 0.7 / daily_consumption
                )  # Order for 70% of shelf life
            else:
                optimal_frequency = expiry_days

            # Recommend discount timing (days before expiry)
            discount_timing = max(
                1, int(expiry_days * 0.3)
            )  # Start discounting at 30% of shelf life

            # Shelf life optimization recommendations
            shelf_life_optimization = {
                "reduce_order_quantity": predicted_waste > current_stock * 0.1,
                "increase_order_frequency": optimal_frequency < 7,
                "implement_fifo": True,
                "discount_threshold_days": discount_timing,
            }

            # Calculate rotation efficiency
            rotation_efficiency = min(1.0, daily_consumption * expiry_days / (current_stock + 1e-6))

            return ExpiryIntelligence(
                item_id=item.id,
                predicted_waste_amount=predicted_waste,
                optimal_ordering_frequency=optimal_frequency,
                risk_score=risk_score,
                recommended_discount_timing=discount_timing,
                shelf_life_optimization=shelf_life_optimization,
                rotation_efficiency=rotation_efficiency,
            )

        except Exception as e:
            logger.error(f"Error analyzing expiry intelligence for item {item.id}: {e}")
            return None

    async def _analyze_supplier_performance(self, session: Session) -> Optional[AgentDecision]:
        """Analyze comprehensive supplier performance."""
        try:
            # Get all suppliers with recent activity
            recent_date = datetime.now().date() - timedelta(days=90)

            suppliers = (
                session.query(Supplier)
                .join(PurchaseOrder)
                .filter(PurchaseOrder.order_date >= recent_date)
                .distinct()
                .all()
            )

            if not suppliers:
                return None

            supplier_analyses = []

            for supplier in suppliers:
                analysis = await self._analyze_individual_supplier_performance(session, supplier)
                if analysis:
                    supplier_analyses.append(analysis)

            if not supplier_analyses:
                return None

            # Sort by overall score
            supplier_analyses.sort(key=lambda x: x.overall_score, reverse=True)

            excellent_suppliers = [s for s in supplier_analyses if s.overall_score > 0.8]
            poor_performers = [s for s in supplier_analyses if s.overall_score < 0.6]
            sorted(supplier_analyses, key=lambda x: x.cost_competitiveness, reverse=True)[:3]

            context = {
                "total_suppliers_analyzed": len(supplier_analyses),
                "excellent_suppliers": len(excellent_suppliers),
                "poor_performers": len(poor_performers),
                "supplier_performance_details": [s.__dict__ for s in supplier_analyses],
            }

            analysis_prompt = f"""
            Supplier performance analysis for {len(supplier_analyses)} suppliers:

            Key Findings:
            - {len(excellent_suppliers)} suppliers with excellent performance (>80% score)
            - {len(poor_performers)} suppliers underperforming (<60% score)
            - Top cost-competitive suppliers identified

            Recommend supplier relationship management and sourcing strategy adjustments.
            """

            reasoning = await self.analyze_with_claude(analysis_prompt, context)

            confidence = 0.85

            action_items = []
            if poor_performers:
                action_items.append(
                    f"Review contracts with {len(poor_performers)} underperforming suppliers"
                )
            if excellent_suppliers:
                action_items.append(
                    f"Strengthen relationships with {len(excellent_suppliers)} top suppliers"
                )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="supplier_performance_analysis",
                context=context,
                reasoning=reasoning,
                action=(
                    "; ".join(action_items)
                    if action_items
                    else "Continue monitoring supplier performance"
                ),
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error analyzing supplier performance: {e}")
            return None

    async def _analyze_individual_supplier_performance(
        self, session: Session, supplier: Supplier
    ) -> Optional[SupplierPerformance]:
        """Analyze performance metrics for an individual supplier."""
        try:
            recent_date = datetime.now().date() - timedelta(days=90)

            # Get recent purchase orders from this supplier
            purchase_orders = (
                session.query(PurchaseOrder)
                .filter(
                    PurchaseOrder.supplier_id == supplier.id,
                    PurchaseOrder.order_date >= recent_date,
                )
                .all()
            )

            if not purchase_orders:
                return None

            # Calculate reliability score (on-time delivery)
            on_time_deliveries = 0
            total_deliveries = 0

            for po in purchase_orders:
                if po.expected_delivery_date and po.order_date:
                    total_deliveries += 1
                    # Assume delivered on expected date for now (would need actual delivery data)
                    on_time_deliveries += 1  # Simplified

            reliability_score = (
                on_time_deliveries / total_deliveries if total_deliveries > 0 else 0.5
            )

            # Calculate cost competitiveness
            total_order_value = sum(float(po.total_amount) for po in purchase_orders)
            total_order_value / len(purchase_orders) if purchase_orders else 0

            # Compare to market average (simplified calculation)
            # In real implementation, you'd compare to other suppliers
            cost_competitiveness = 0.75  # Placeholder score

            # Quality score (based on return/complaint data - simplified)
            quality_score = 0.85  # Placeholder score

            # Delivery performance score
            delivery_performance = reliability_score

            # Calculate overall score (weighted average)
            weights = {"reliability": 0.3, "cost": 0.25, "quality": 0.25, "delivery": 0.2}

            overall_score = (
                reliability_score * weights["reliability"]
                + cost_competitiveness * weights["cost"]
                + quality_score * weights["quality"]
                + delivery_performance * weights["delivery"]
            )

            # Risk assessment
            risk_assessment = {
                "single_source_risk": 0.3,  # Risk of depending on single supplier
                "financial_stability": 0.2,  # Supplier financial health risk
                "geographic_risk": 0.1,  # Location-based risks
                "capacity_risk": 0.15,  # Supplier capacity constraints
            }

            # Generate recommendation
            if overall_score > 0.8:
                recommendation = "Preferred supplier - increase business volume"
            elif overall_score > 0.6:
                recommendation = "Satisfactory supplier - maintain current relationship"
            else:
                recommendation = "Review supplier relationship - consider alternatives"

            return SupplierPerformance(
                supplier_id=supplier.id,
                overall_score=overall_score,
                reliability_score=reliability_score,
                cost_competitiveness=cost_competitiveness,
                quality_score=quality_score,
                delivery_performance=delivery_performance,
                risk_assessment=risk_assessment,
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Error analyzing performance for supplier {supplier.id}: {e}")
            return None

    async def _comprehensive_inventory_analysis(self, session: Session) -> Optional[AgentDecision]:
        """Perform comprehensive inventory health analysis."""
        try:
            # Get overall inventory metrics
            total_items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).count()

            # Items below reorder point
            low_stock_items = (
                session.query(Item)
                .filter(Item.status == ItemStatus.ACTIVE, Item.current_stock <= Item.reorder_point)
                .count()
            )

            # Overstocked items
            overstocked_items = (
                session.query(Item)
                .filter(Item.status == ItemStatus.ACTIVE, Item.current_stock >= Item.maximum_stock)
                .count()
            )

            # Calculate total inventory value
            total_value = (
                session.query(func.sum(Item.current_stock * Item.unit_cost))
                .filter(Item.status == ItemStatus.ACTIVE)
                .scalar()
                or 0
            )

            # Get recent stock movements for turnover analysis
            recent_date = datetime.now().date() - timedelta(days=30)
            recent_movements = (
                session.query(func.sum(StockMovement.quantity))
                .filter(
                    StockMovement.movement_type == StockMovementType.OUT,
                    StockMovement.movement_date >= recent_date,
                )
                .scalar()
                or 0
            )

            # Calculate inventory turnover (simplified)
            monthly_turnover = (
                float(recent_movements) / (float(total_value) + 1e-6) if total_value > 0 else 0
            )

            # Identify key issues
            issues = []
            if low_stock_items > total_items * 0.15:  # More than 15% low stock
                issues.append("High number of low-stock items")
            if overstocked_items > total_items * 0.1:  # More than 10% overstocked
                issues.append("Significant overstock situation")
            if monthly_turnover < 0.1:  # Low turnover
                issues.append("Low inventory turnover rate")

            context = {
                "total_active_items": total_items,
                "low_stock_items": low_stock_items,
                "overstocked_items": overstocked_items,
                "total_inventory_value": float(total_value),
                "monthly_turnover_rate": monthly_turnover,
                "identified_issues": issues,
                "low_stock_percentage": (
                    low_stock_items / total_items * 100 if total_items > 0 else 0
                ),
                "overstock_percentage": (
                    overstocked_items / total_items * 100 if total_items > 0 else 0
                ),
            }

            analysis_prompt = f"""
            Comprehensive inventory health analysis:

            Current Status:
            - Total active items: {total_items}
            - Low stock items: {low_stock_items} ({context['low_stock_percentage']:.1f}%)
            - Overstocked items: {overstocked_items} ({context['overstock_percentage']:.1f}%)
            - Total inventory value: ${total_value:,.2f}
            - Monthly turnover rate: {monthly_turnover:.1%}

            Key Issues: {', '.join(issues) if issues else 'No major issues identified'}

            Provide strategic recommendations for inventory optimization.
            """

            reasoning = await self.analyze_with_claude(analysis_prompt, context)

            # Determine confidence based on data quality and issue severity
            confidence = 0.8
            if len(issues) > 2:
                confidence = 0.9  # High confidence when clear issues identified
            elif len(issues) == 0:
                confidence = 0.7  # Lower confidence when no issues (might be missing something)

            action_items = []
            if low_stock_items > 0:
                action_items.append(f"Reorder {low_stock_items} low-stock items")
            if overstocked_items > 0:
                action_items.append(
                    f"Implement clearance strategy for {overstocked_items} overstocked items"
                )
            if monthly_turnover < 0.1:
                action_items.append(
                    "Improve inventory turnover through demand forecasting and optimization"
                )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="comprehensive_inventory_analysis",
                context=context,
                reasoning=reasoning,
                action=(
                    "; ".join(action_items)
                    if action_items
                    else "Continue monitoring inventory health"
                ),
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error in comprehensive inventory analysis: {e}")
            return None

    async def _analyze_stock_movement_enhanced(
        self, session: Session, movement_data: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        """Enhanced stock movement analysis with predictive insights."""
        try:
            item_id = movement_data.get("item_id")
            movement_type = movement_data.get("movement_type")
            quantity = movement_data.get("quantity", 0)

            if not item_id:
                return None

            item = session.query(Item).filter(Item.id == item_id).first()
            if not item:
                return None

            # Get demand forecast for context
            forecast = await self._forecast_item_demand(session, item_id, 14)

            # Enhanced analysis based on movement type
            if movement_type == StockMovementType.OUT:
                return await self._analyze_consumption_movement(session, item, quantity, forecast)
            elif movement_type == StockMovementType.IN:
                return await self._analyze_receipt_movement(session, item, quantity, forecast)
            elif movement_type == StockMovementType.ADJUSTMENT:
                return await self._analyze_adjustment_movement(session, item, quantity, forecast)
            else:
                return None

        except Exception as e:
            logger.error(f"Error in enhanced stock movement analysis: {e}")
            return None

    async def _analyze_consumption_movement(
        self, session: Session, item: Item, quantity: float, forecast: Optional[DemandForecast]
    ) -> Optional[AgentDecision]:
        """Analyze consumption movement with predictive context."""
        try:
            new_stock_level = item.current_stock - quantity

            # Check if this consumption is unusual
            if forecast:
                expected_daily = forecast.historical_patterns["avg_daily"]
                if quantity > expected_daily * 2:  # More than 2x normal consumption
                    context = {
                        "item_id": item.id,
                        "item_name": item.name,
                        "unusual_consumption": quantity,
                        "normal_daily_consumption": expected_daily,
                        "consumption_ratio": quantity / expected_daily if expected_daily > 0 else 0,
                        "remaining_stock": new_stock_level,
                        "reorder_point": item.reorder_point,
                    }

                    reasoning = await self.analyze_with_claude(
                        f"Unusual consumption detected for {item.name}: {quantity} units consumed "
                        f"(normal daily: {expected_daily:.1f}). Current stock: {new_stock_level}.",
                        context,
                    )

                    return AgentDecision(
                        agent_id=self.agent_id,
                        decision_type="unusual_consumption_alert",
                        context=context,
                        reasoning=reasoning,
                        action=f"Investigate unusual consumption pattern for {item.name}",
                        confidence=0.8,
                    )

            # Check if approaching reorder point
            if new_stock_level <= item.reorder_point:
                days_remaining = (
                    new_stock_level / forecast.historical_patterns["avg_daily"]
                    if forecast and forecast.historical_patterns["avg_daily"] > 0
                    else 0
                )

                context = {
                    "item_id": item.id,
                    "item_name": item.name,
                    "current_stock": new_stock_level,
                    "reorder_point": item.reorder_point,
                    "estimated_days_remaining": days_remaining,
                    "lead_time": self.reorder_lead_time,
                }

                urgency = "CRITICAL" if days_remaining < self.reorder_lead_time else "HIGH"

                reasoning = await self.analyze_with_claude(
                    f"Stock level for {item.name} has reached reorder point. "
                    f"Estimated {days_remaining:.1f} days remaining. Urgency: {urgency}",
                    context,
                )

                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="reorder_required",
                    context=context,
                    reasoning=reasoning,
                    action=f"Reorder {item.name} - {urgency} priority",
                    confidence=0.9,
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing consumption movement: {e}")
            return None

    async def _analyze_receipt_movement(
        self, session: Session, item: Item, quantity: float, forecast: Optional[DemandForecast]
    ) -> Optional[AgentDecision]:
        """Analyze stock receipt with optimization insights."""
        try:
            new_stock_level = item.current_stock + quantity

            # Check for overstock situation
            if new_stock_level > item.maximum_stock:
                excess_quantity = new_stock_level - item.maximum_stock

                if forecast:
                    days_of_supply = (
                        new_stock_level / forecast.historical_patterns["avg_daily"]
                        if forecast.historical_patterns["avg_daily"] > 0
                        else 0
                    )
                else:
                    days_of_supply = 0

                context = {
                    "item_id": item.id,
                    "item_name": item.name,
                    "new_stock_level": new_stock_level,
                    "maximum_stock": item.maximum_stock,
                    "excess_quantity": excess_quantity,
                    "days_of_supply": days_of_supply,
                }

                reasoning = await self.analyze_with_claude(
                    f"Overstock situation for {item.name}: {new_stock_level} units "
                    f"(maximum: {item.maximum_stock}). {days_of_supply:.1f} days of supply.",
                    context,
                )

                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="overstock_alert",
                    context=context,
                    reasoning=reasoning,
                    action=f"Review ordering strategy for {item.name} - excess inventory detected",
                    confidence=0.85,
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing receipt movement: {e}")
            return None

    async def _analyze_adjustment_movement(
        self, session: Session, item: Item, quantity: float, forecast: Optional[DemandForecast]
    ) -> Optional[AgentDecision]:
        """Analyze inventory adjustment with accuracy insights."""
        try:
            adjustment_percentage = (
                abs(quantity) / item.current_stock if item.current_stock > 0 else 0
            )

            # Significant adjustments warrant investigation
            if adjustment_percentage > 0.1:  # More than 10% adjustment
                context = {
                    "item_id": item.id,
                    "item_name": item.name,
                    "adjustment_quantity": quantity,
                    "adjustment_percentage": adjustment_percentage * 100,
                    "current_stock": item.current_stock,
                    "adjustment_type": "increase" if quantity > 0 else "decrease",
                }

                reasoning = await self.analyze_with_claude(
                    f"Significant inventory adjustment for {item.name}: "
                    f"{quantity:+.0f} units ({adjustment_percentage:.1%} change).",
                    context,
                )

                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="inventory_adjustment_alert",
                    context=context,
                    reasoning=reasoning,
                    action=f"Investigate inventory accuracy for {item.name}",
                    confidence=0.75,
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing adjustment movement: {e}")
            return None

    async def _analyze_reorder_needs_enhanced(self, session: Session) -> Optional[AgentDecision]:
        """Enhanced reorder analysis with predictive insights."""
        try:
            # Get items approaching reorder point
            items_near_reorder = (
                session.query(Item)
                .filter(
                    Item.status == ItemStatus.ACTIVE,
                    Item.current_stock <= Item.reorder_point * self.low_stock_multiplier,
                )
                .all()
            )

            if not items_near_reorder:
                return None

            priority_items = []
            for item in items_near_reorder:
                forecast = await self._forecast_item_demand(
                    session, item.id, self.reorder_lead_time
                )

                if forecast:
                    days_remaining = (
                        item.current_stock / forecast.historical_patterns["avg_daily"]
                        if forecast.historical_patterns["avg_daily"] > 0
                        else 0
                    )
                    urgency_score = max(0, 1 - days_remaining / self.reorder_lead_time)
                else:
                    urgency_score = 0.5

                priority_items.append(
                    {"item": item, "urgency_score": urgency_score, "forecast": forecast}
                )

            # Sort by urgency
            priority_items.sort(key=lambda x: x["urgency_score"], reverse=True)

            critical_items = [item for item in priority_items if item["urgency_score"] > 0.8]

            context = {
                "total_items_near_reorder": len(items_near_reorder),
                "critical_items": len(critical_items),
                "reorder_analysis": [
                    {
                        "item_id": item["item"].id,
                        "item_name": item["item"].name,
                        "urgency_score": item["urgency_score"],
                        "current_stock": item["item"].current_stock,
                        "reorder_point": item["item"].reorder_point,
                    }
                    for item in priority_items[:10]
                ],
            }

            reasoning = await self.analyze_with_claude(
                f"Reorder analysis for {len(items_near_reorder)} items approaching reorder point. "
                f"{len(critical_items)} items require immediate attention.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="enhanced_reorder_analysis",
                context=context,
                reasoning=reasoning,
                action=f"Prioritize reordering for {len(critical_items)} critical items",
                confidence=0.85,
            )

        except Exception as e:
            logger.error(f"Error in enhanced reorder analysis: {e}")
            return None

    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive inventory intelligence report."""
        session = None
        try:
            session = self.SessionLocal()

            # Get current inventory status
            total_items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).count()
            low_stock_count = (
                session.query(Item)
                .filter(Item.status == ItemStatus.ACTIVE, Item.current_stock <= Item.reorder_point)
                .count()
            )

            # Generate sample forecasts for top items
            forecasts = await self._forecast_all_items_demand(session, 14)

            return {
                "agent_id": self.agent_id,
                "report_type": "enhanced_inventory_intelligence",
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_active_items": total_items,
                    "items_requiring_reorder": low_stock_count,
                    "forecasted_items": len(forecasts),
                    "reorder_percentage": (
                        (low_stock_count / total_items * 100) if total_items > 0 else 0
                    ),
                },
                "intelligence_capabilities": {
                    "demand_forecasting": "Multi-method ensemble with seasonality",
                    "reorder_optimization": "EOQ with service level targets",
                    "bulk_purchase_analysis": "Discount tier optimization",
                    "expiry_management": "Predictive waste minimization",
                    "supplier_analytics": "Performance scoring and selection",
                },
                "recent_decisions": [decision.__dict__ for decision in self.decisions_log[-5:]],
                "forecast_summary": {
                    "average_accuracy": (
                        sum(f.forecast_accuracy for f in forecasts) / len(forecasts)
                        if forecasts
                        else 0
                    ),
                    "high_growth_items": len([f for f in forecasts if f.trend_factor > 0.1]),
                    "declining_items": len([f for f in forecasts if f.trend_factor < -0.1]),
                },
            }

        except Exception as e:
            logger.error(f"Error generating enhanced inventory report: {e}")
            return {"error": str(e)}
        finally:
            if session:
                session.close()

    async def periodic_check(self) -> None:
        """Enhanced periodic check with intelligent scheduling."""
        try:
            current_time = datetime.now()

            # Run comprehensive analysis every 4 hours
            if current_time.hour % 4 == 0 and current_time.minute < 30:
                await self._queue_analysis_message("inventory_health_check")

            # Run demand forecasting daily at 6 AM
            if current_time.hour == 6 and current_time.minute < 30:
                await self._queue_analysis_message("demand_forecast_request")

            # Run reorder optimization twice per week
            if current_time.weekday() in [1, 4] and current_time.hour == 8:  # Tuesday and Friday
                await self._queue_analysis_message("reorder_optimization")

            # Run supplier performance analysis weekly
            if current_time.weekday() == 0 and current_time.hour == 9:  # Monday morning
                await self._queue_analysis_message("supplier_performance_review")

            # Run expiry management for perishables daily
            if current_time.hour == 10 and current_time.minute < 30:
                await self._queue_analysis_message("expiry_management")

        except Exception as e:
            logger.error(f"Error in enhanced periodic check: {e}")

    async def _queue_analysis_message(self, analysis_type: str) -> None:
        """Queue analysis message for processing."""
        try:
            if self.message_queue:
                message = {
                    "type": analysis_type,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id,
                }
                await self.message_queue.put(message)
        except Exception as e:
            logger.error(f"Error queuing analysis message: {e}")
