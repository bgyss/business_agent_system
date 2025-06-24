import math
import statistics
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from sqlalchemy import and_
from sqlalchemy.orm import Session

from agents.base_agent import AgentDecision, BaseAgent
from models.inventory import (
    InventorySummary,
    Item,
    ItemStatus,
    PurchaseOrder,
    PurchaseOrderItem,
    StockMovement,
    StockMovementType,
    Supplier,
)


class DemandForecast(NamedTuple):
    """Demand forecast result."""

    item_id: str
    predicted_demand: float
    confidence_interval: Tuple[float, float]
    seasonality_factor: float
    trend_factor: float
    forecast_horizon_days: int
    forecast_accuracy: float


class OptimalReorderPoint(NamedTuple):
    """Optimal reorder point calculation result."""

    item_id: str
    optimal_reorder_point: int
    optimal_reorder_quantity: int
    service_level: float
    safety_stock: int
    lead_time_demand: float
    demand_variability: float
    total_cost: float


class BulkPurchaseOptimization(NamedTuple):
    """Bulk purchase optimization result."""

    item_id: str
    optimal_order_quantity: int
    unit_cost_with_discount: Decimal
    total_cost_savings: Decimal
    break_even_point: int
    holding_cost_impact: Decimal
    recommended_purchase_timing: datetime


class SupplierPerformanceMetrics(NamedTuple):
    """Comprehensive supplier performance metrics."""

    supplier_id: str
    on_time_delivery_rate: float
    quality_score: float
    cost_competitiveness: float
    reliability_index: float
    lead_time_variability: float
    overall_performance_score: float
    recommended_action: str


class SeasonalityAnalysis(NamedTuple):
    """Seasonal pattern analysis results."""

    item_id: str
    seasonal_periods: List[int]  # Detected periods (e.g., [7, 30, 365] for weekly, monthly, yearly)
    seasonal_strength: float  # 0-1, how strong the seasonality is
    peak_periods: List[int]  # Day numbers within period when demand peaks
    low_periods: List[int]  # Day numbers within period when demand is lowest
    seasonal_adjustment_factor: float  # Current period adjustment
    confidence: float  # Confidence in seasonal detection


class ItemCorrelationAnalysis(NamedTuple):
    """Cross-item correlation analysis."""

    primary_item_id: str
    correlated_items: List[Tuple[str, float]]  # (item_id, correlation_coefficient)
    substitution_items: List[str]  # Items that can substitute for this one
    complementary_items: List[str]  # Items often bought together
    impact_factor: float  # How much this item's stock affects others
    bundle_opportunities: List[Dict[str, Any]]  # Bundling recommendations


class SupplierDiversificationAnalysis(NamedTuple):
    """Supplier risk and diversification analysis."""

    item_id: str
    current_supplier_concentration: float  # 0-1, higher = more concentrated risk
    alternative_suppliers: List[str]  # Available alternative supplier IDs
    risk_score: float  # 0-1, supply chain risk level
    diversification_recommendations: List[Dict[str, Any]]  # Specific actions
    cost_impact_of_diversification: float  # Expected cost change
    recommended_supplier_split: Dict[str, float]  # supplier_id -> percentage


class InventoryAgent(BaseAgent):
    def __init__(self, agent_id: str, api_key: str, config: Dict[str, Any], db_url: str):
        super().__init__(agent_id, api_key, config, db_url)
        self.low_stock_multiplier = config.get("low_stock_multiplier", 1.2)
        self.reorder_lead_time = config.get("reorder_lead_time", 7)
        self.consumption_analysis_days = config.get("consumption_analysis_days", 30)

        # Advanced analytics configuration
        self.forecast_horizon_days = config.get("forecast_horizon_days", 30)
        self.service_level_target = config.get("service_level_target", 0.95)
        self.holding_cost_rate = config.get("holding_cost_rate", 0.25)  # 25% annual holding cost
        self.order_cost = config.get("order_cost", 50.0)  # Fixed cost per order
        self.min_forecast_accuracy = config.get("min_forecast_accuracy", 0.70)
        self.seasonality_window_days = config.get("seasonality_window_days", 365)

        # Demand forecasting parameters
        self.alpha_smoothing = config.get("alpha_smoothing", 0.3)  # Exponential smoothing
        self.beta_trend = config.get("beta_trend", 0.1)  # Trend smoothing
        self.gamma_seasonality = config.get("gamma_seasonality", 0.2)  # Seasonality smoothing

    @property
    def system_prompt(self) -> str:
        return """You are an AI Inventory Management Agent responsible for monitoring stock levels and optimizing inventory.

        Your responsibilities include:
        1. Monitoring current stock levels and identifying low stock situations
        2. Analyzing consumption patterns to predict future needs
        3. Generating automatic reorder suggestions based on consumption and lead times
        4. Tracking supplier performance and delivery reliability
        5. Identifying slow-moving or obsolete inventory
        6. Monitoring for expired or expiring products
        7. Optimizing inventory costs while maintaining service levels

        When making reorder decisions, consider:
        - Current stock levels vs. reorder points
        - Recent consumption patterns and trends
        - Seasonal variations in demand
        - Supplier lead times and reliability
        - Storage capacity limitations
        - Cash flow implications of large orders
        - Bulk purchase discounts vs. carrying costs

        Provide clear, actionable recommendations with cost-benefit analysis.
        """

    async def process_data(self, data: Dict[str, Any]) -> Optional[AgentDecision]:
        session = self.SessionLocal()
        try:
            if data.get("type") == "stock_movement":
                return await self._analyze_stock_movement(session, data["movement"])
            elif data.get("type") == "daily_inventory_check":
                return await self._perform_daily_inventory_check(session)
            elif data.get("type") == "reorder_analysis":
                return await self._analyze_reorder_needs(session)
            elif data.get("type") == "expiry_check":
                return await self._check_expiring_items(session)
            elif data.get("type") == "supplier_performance":
                return await self._analyze_supplier_performance(session)
            elif data.get("type") == "advanced_reorder_analysis":
                return await self._perform_advanced_reorder_analysis(session)
            elif data.get("type") == "demand_forecast_analysis":
                return await self._perform_demand_forecast_analysis(session)
            elif data.get("type") == "bulk_purchase_analysis":
                return await self._perform_bulk_purchase_analysis(session)
            elif data.get("type") == "expiry_waste_analysis":
                return await self._perform_expiry_waste_analysis(session)
            elif data.get("type") == "advanced_supplier_analysis":
                return await self._perform_advanced_supplier_analysis(session)
            elif data.get("type") == "comprehensive_analytics":
                return await self._perform_comprehensive_analytics_analysis(session)
            elif data.get("type") == "seasonality_analysis":
                item_id = data.get("item_id")
                if item_id:
                    seasonality = await self.analyze_seasonality_patterns(session, item_id)
                    if seasonality:
                        context = {"item_id": item_id, "seasonality": seasonality._asdict()}
                        reasoning = await self.analyze_with_claude(
                            f"Seasonality analysis for item {item_id} detected patterns with "
                            f"{seasonality.confidence:.1%} confidence. Seasonal strength: {seasonality.seasonal_strength:.2f}",
                            context,
                        )
                        return AgentDecision(
                            agent_id=self.agent_id,
                            decision_type="seasonality_analysis",
                            context=context,
                            reasoning=reasoning,
                            action="Apply seasonal adjustments to inventory planning",
                            confidence=seasonality.confidence,
                        )
            elif data.get("type") == "correlation_analysis":
                item_id = data.get("item_id")
                if item_id:
                    correlations = await self.analyze_item_correlations(session, item_id)
                    if correlations:
                        context = {"item_id": item_id, "correlations": correlations._asdict()}
                        reasoning = await self.analyze_with_claude(
                            f"Item correlation analysis for {item_id} found {len(correlations.correlated_items)} "
                            f"correlated items with impact factor {correlations.impact_factor:.2f}",
                            context,
                        )
                        return AgentDecision(
                            agent_id=self.agent_id,
                            decision_type="correlation_analysis",
                            context=context,
                            reasoning=reasoning,
                            action="Optimize inventory based on item correlations",
                            confidence=0.85,
                        )
            elif data.get("type") == "supplier_diversification_analysis":
                item_id = data.get("item_id")
                if item_id:
                    diversification = await self.analyze_supplier_diversification(session, item_id)
                    if diversification:
                        context = {"item_id": item_id, "diversification": diversification._asdict()}
                        reasoning = await self.analyze_with_claude(
                            f"Supplier diversification analysis for {item_id} shows risk score {diversification.risk_score:.2f} "
                            f"with concentration index {diversification.current_supplier_concentration:.2f}",
                            context,
                        )
                        return AgentDecision(
                            agent_id=self.agent_id,
                            decision_type="supplier_diversification_analysis",
                            context=context,
                            reasoning=reasoning,
                            action="Implement supplier diversification strategies",
                            confidence=0.88,
                        )
        except Exception as e:
            self.logger.error(f"Error in process_data: {e}")
            return None
        finally:
            try:
                session.close()
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")

        return None

    async def _analyze_stock_movement(
        self, session, movement_data: Dict[str, Any]
    ) -> Optional[AgentDecision]:
        item_id = movement_data.get("item_id")
        movement_type = movement_data.get("movement_type")
        quantity = movement_data.get("quantity", 0)

        item = session.query(Item).filter(Item.id == item_id).first()
        if not item:
            return None

        # Calculate new stock level
        if movement_type == StockMovementType.IN:
            new_stock = item.current_stock + quantity
        elif movement_type == StockMovementType.OUT:
            new_stock = item.current_stock - quantity
        else:
            new_stock = item.current_stock

        context = {
            "item": {
                "id": item.id,
                "name": item.name,
                "sku": item.sku,
                "current_stock": item.current_stock,
                "new_stock": new_stock,
                "reorder_point": item.reorder_point,
                "minimum_stock": item.minimum_stock,
            },
            "movement": movement_data,
        }

        # Check if this movement triggers a low stock alert
        if (
            item.reorder_point is not None
            and new_stock <= item.reorder_point
            and item.current_stock > item.reorder_point
        ):
            reasoning = await self.analyze_with_claude(
                f"Item {item.name} (SKU: {item.sku}) has dropped to {new_stock} units, "
                f"which is at or below the reorder point of {item.reorder_point}. "
                f"Recent movement: {movement_type} of {quantity} units. "
                f"Should we trigger a reorder?",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="low_stock_alert",
                context=context,
                reasoning=reasoning,
                action=f"Generate reorder recommendation for {item.name}",
                confidence=0.85,
            )

        # Check for unusual consumption patterns
        if (
            movement_type == StockMovementType.OUT
            and item.current_stock > 0
            and quantity > item.current_stock * 0.5
        ):
            reasoning = await self.analyze_with_claude(
                f"Large stock movement detected for {item.name}: {quantity} units out "
                f"(represents {quantity/item.current_stock:.1%} of current stock). "
                f"Is this a normal usage pattern or should it be investigated?",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="unusual_consumption",
                context=context,
                reasoning=reasoning,
                action=f"Investigate large consumption of {item.name}",
                confidence=0.7,
            )

        return None

    async def _perform_daily_inventory_check(self, session) -> Optional[AgentDecision]:
        # Get all active items
        items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()

        low_stock_items = []
        out_of_stock_items = []

        for item in items:
            if item.current_stock <= 0:
                out_of_stock_items.append(item)
            elif item.reorder_point is not None and item.current_stock <= item.reorder_point:
                low_stock_items.append(item)

        if not low_stock_items and not out_of_stock_items:
            return None

        context = {
            "low_stock_count": len(low_stock_items),
            "out_of_stock_count": len(out_of_stock_items),
            "low_stock_items": [
                {
                    "name": item.name,
                    "sku": item.sku,
                    "current_stock": item.current_stock,
                    "reorder_point": item.reorder_point,
                    "estimated_value": float(item.current_stock * item.unit_cost),
                }
                for item in low_stock_items[:10]  # Limit to top 10
            ],
            "out_of_stock_items": [
                {
                    "name": item.name,
                    "sku": item.sku,
                    "reorder_quantity": item.reorder_quantity,
                    "estimated_cost": float(item.reorder_quantity * item.unit_cost),
                }
                for item in out_of_stock_items[:10]  # Limit to top 10
            ],
        }

        analysis = await self.analyze_with_claude(
            f"Daily inventory check shows {len(low_stock_items)} items with low stock "
            f"and {len(out_of_stock_items)} items out of stock. "
            f"Provide prioritized action plan for restocking.",
            context,
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="daily_inventory_check",
            context=context,
            reasoning=analysis,
            action="Generate comprehensive reorder recommendations",
            confidence=0.9,
        )

    async def _analyze_reorder_needs(self, session) -> Optional[AgentDecision]:
        # Get consumption data for the last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=self.consumption_analysis_days)

        # Get all items that need analysis
        items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()

        reorder_suggestions = []

        for item in items:
            # Calculate consumption rate
            consumption_movements = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.item_id == item.id,
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= thirty_days_ago,
                    )
                )
                .all()
            )

            if not consumption_movements:
                continue

            total_consumed = sum(movement.quantity for movement in consumption_movements)
            daily_consumption = total_consumed / self.consumption_analysis_days

            # Calculate days of stock remaining
            days_remaining = (
                item.current_stock / daily_consumption if daily_consumption > 0 else 999
            )

            # Check if reorder is needed considering lead time
            if days_remaining <= self.reorder_lead_time * self.low_stock_multiplier:
                # Calculate suggested order quantity
                lead_time_consumption = daily_consumption * self.reorder_lead_time
                safety_stock = daily_consumption * 7  # 1 week safety stock
                suggested_quantity = max(
                    item.reorder_quantity,
                    int(lead_time_consumption + safety_stock - item.current_stock),
                )

                # Determine urgency
                if days_remaining <= self.reorder_lead_time * 0.5:
                    urgency = "critical"
                elif days_remaining <= self.reorder_lead_time * 0.8:
                    urgency = "high"
                else:
                    urgency = "medium"

                reorder_suggestions.append(
                    {
                        "item_id": item.id,
                        "item_name": item.name,
                        "sku": item.sku,
                        "current_stock": item.current_stock,
                        "daily_consumption": round(daily_consumption, 2),
                        "days_remaining": round(days_remaining, 1),
                        "suggested_quantity": suggested_quantity,
                        "estimated_cost": float(suggested_quantity * item.unit_cost),
                        "urgency": urgency,
                    }
                )

        if not reorder_suggestions:
            return None

        # Sort by urgency and days remaining
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        reorder_suggestions.sort(key=lambda x: (urgency_order[x["urgency"]], x["days_remaining"]))

        context = {
            "reorder_count": len(reorder_suggestions),
            "total_estimated_cost": sum(item["estimated_cost"] for item in reorder_suggestions),
            "critical_items": [
                item for item in reorder_suggestions if item["urgency"] == "critical"
            ],
            "high_priority_items": [
                item for item in reorder_suggestions if item["urgency"] == "high"
            ],
            "suggestions": reorder_suggestions[:15],  # Limit to top 15
        }

        analysis = await self.analyze_with_claude(
            f"Reorder analysis identifies {len(reorder_suggestions)} items needing restock. "
            f"Total estimated cost: ${context['total_estimated_cost']:,.2f}. "
            f"Critical items: {len(context['critical_items'])}, "
            f"High priority: {len(context['high_priority_items'])}. "
            f"Provide prioritized purchasing recommendations considering cash flow.",
            context,
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="reorder_analysis",
            context=context,
            reasoning=analysis,
            action="Create purchase orders for critical and high-priority items",
            confidence=0.88,
        )

    async def _check_expiring_items(self, session) -> Optional[AgentDecision]:
        # Find items with expiry dates
        items_with_expiry = (
            session.query(Item)
            .filter(
                and_(
                    Item.expiry_days.isnot(None),
                    Item.current_stock > 0,
                    Item.status == ItemStatus.ACTIVE,
                )
            )
            .all()
        )

        if not items_with_expiry:
            return None

        expiring_soon = []
        datetime.now().date()

        # This is a simplified check - in a real system, you'd track batch expiry dates
        for item in items_with_expiry:
            # Assume items expire based on days since last received
            # In practice, you'd track actual expiry dates per batch
            if item.expiry_days <= 7:  # Items that expire within a week
                expiring_soon.append(
                    {
                        "name": item.name,
                        "sku": item.sku,
                        "current_stock": item.current_stock,
                        "expiry_days": item.expiry_days,
                        "estimated_loss": float(item.current_stock * item.unit_cost),
                    }
                )

        if not expiring_soon:
            return None

        context = {
            "expiring_items_count": len(expiring_soon),
            "total_potential_loss": sum(item["estimated_loss"] for item in expiring_soon),
            "expiring_items": expiring_soon,
        }

        analysis = await self.analyze_with_claude(
            f"Expiry check shows {len(expiring_soon)} items expiring soon with "
            f"potential loss of ${context['total_potential_loss']:,.2f}. "
            f"Recommend actions to minimize waste.",
            context,
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="expiry_alert",
            context=context,
            reasoning=analysis,
            action="Implement waste reduction measures for expiring items",
            confidence=0.85,
        )

    async def _analyze_supplier_performance(self, session) -> Optional[AgentDecision]:
        # Get recent purchase orders and their delivery performance
        thirty_days_ago = datetime.now() - timedelta(days=30)

        recent_orders = (
            session.query(PurchaseOrder).filter(PurchaseOrder.order_date >= thirty_days_ago).all()
        )

        if not recent_orders:
            return None

        supplier_performance = {}

        for order in recent_orders:
            supplier_id = order.supplier_id
            if supplier_id not in supplier_performance:
                supplier = session.query(Supplier).filter(Supplier.id == supplier_id).first()
                supplier_performance[supplier_id] = {
                    "name": supplier.name if supplier else "Unknown",
                    "orders": 0,
                    "on_time_deliveries": 0,
                    "total_value": 0,
                    "avg_delay": 0,
                }

            perf = supplier_performance[supplier_id]
            perf["orders"] += 1
            perf["total_value"] += float(order.total_amount)

            # Check if delivered on time (simplified check)
            if order.expected_delivery_date and order.status == "delivered":
                # In a real system, you'd track actual delivery date
                # For now, assume delivered on time if status is delivered
                perf["on_time_deliveries"] += 1

        # Calculate performance metrics
        poor_performers = []
        for supplier_id, perf in supplier_performance.items():
            if perf["orders"] >= 2:  # Only analyze suppliers with multiple orders
                on_time_rate = perf["on_time_deliveries"] / perf["orders"]
                if on_time_rate < 0.8:  # Less than 80% on-time delivery
                    poor_performers.append(
                        {
                            "supplier_id": supplier_id,
                            "name": perf["name"],
                            "on_time_rate": on_time_rate,
                            "order_count": perf["orders"],
                            "total_value": perf["total_value"],
                        }
                    )

        if not poor_performers:
            return None

        context = {
            "analysis_period_days": 30,
            "total_suppliers_analyzed": len(
                [s for s in supplier_performance.values() if s["orders"] >= 2]
            ),
            "poor_performers": poor_performers,
            "total_affected_value": sum(p["total_value"] for p in poor_performers),
        }

        analysis = await self.analyze_with_claude(
            f"Supplier performance analysis shows {len(poor_performers)} suppliers "
            f"with poor delivery performance affecting ${context['total_affected_value']:,.2f} "
            f"in orders. Recommend supplier management actions.",
            context,
        )

        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="supplier_performance",
            context=context,
            reasoning=analysis,
            action="Review and potentially change underperforming suppliers",
            confidence=0.75,
        )

    async def generate_report(self) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            # Generate comprehensive inventory summary
            items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()

            total_value = sum(float(item.current_stock * item.unit_cost) for item in items)
            low_stock_items = [
                item
                for item in items
                if item.reorder_point is not None and item.current_stock <= item.reorder_point
            ]
            out_of_stock_items = [item for item in items if item.current_stock <= 0]

            # Get top moving items (simplified - based on recent movements)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_movements = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= thirty_days_ago,
                    )
                )
                .all()
            )

            item_consumption = {}
            for movement in recent_movements:
                if movement.item_id not in item_consumption:
                    item_consumption[movement.item_id] = 0
                item_consumption[movement.item_id] += movement.quantity

            top_moving = sorted(item_consumption.items(), key=lambda x: x[1], reverse=True)[:5]
            top_moving_items = []
            for item_id, quantity in top_moving:
                item = session.query(Item).filter(Item.id == item_id).first()
                if item:
                    top_moving_items.append(f"{item.name} ({quantity} units)")

            summary = InventorySummary(
                total_items=len(items),
                total_value=Decimal(str(total_value)),
                low_stock_items=len(low_stock_items),
                out_of_stock_items=len(out_of_stock_items),
                items_to_reorder=[item.name for item in low_stock_items[:10]],
                expiring_soon=[],  # Would implement proper expiry tracking
                top_moving_items=top_moving_items,
                slow_moving_items=[],  # Would implement slow-moving analysis
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

        # Check for out of stock items
        out_of_stock = (
            session.query(Item)
            .filter(and_(Item.current_stock <= 0, Item.status == ItemStatus.ACTIVE))
            .count()
        )

        # Handle mock objects in tests
        try:
            out_of_stock_count = int(out_of_stock)
        except (TypeError, ValueError):
            out_of_stock_count = 0

        if out_of_stock_count > 0:
            alerts.append(
                {
                    "type": "out_of_stock",
                    "severity": "high",
                    "message": f"{out_of_stock_count} items are out of stock",
                    "action_required": True,
                }
            )

        # Check for low stock items
        low_stock = (
            session.query(Item)
            .filter(
                and_(
                    Item.current_stock <= Item.reorder_point,
                    Item.current_stock > 0,
                    Item.status == ItemStatus.ACTIVE,
                )
            )
            .count()
        )

        # Handle mock objects in tests
        try:
            low_stock_count = int(low_stock)
        except (TypeError, ValueError):
            low_stock_count = 0

        if low_stock_count > 0:
            alerts.append(
                {
                    "type": "low_stock",
                    "severity": "medium",
                    "message": f"{low_stock_count} items need reordering",
                    "action_required": True,
                }
            )

        return alerts

    # Advanced Predictive Analytics Methods

    async def predict_demand(
        self, session: Session, item_id: str, forecast_days: int = None
    ) -> Optional[DemandForecast]:
        """Predict demand using Triple Exponential Smoothing (Holt-Winters)
        with seasonal patterns.

        Args:
            session: Database session
            item_id: Item ID to forecast
            forecast_days: Number of days to forecast (defaults to config)

        Returns:
            DemandForecast object with prediction and confidence intervals
        """
        if forecast_days is None:
            forecast_days = self.forecast_horizon_days

        try:
            # Get historical consumption data
            cutoff_date = datetime.now() - timedelta(days=self.seasonality_window_days)

            consumption_data = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.item_id == item_id,
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= cutoff_date,
                    )
                )
                .order_by(StockMovement.movement_date)
                .all()
            )

            if len(consumption_data) < 14:  # Need minimum data points
                return None

            # Aggregate daily consumption
            daily_consumption = self._aggregate_daily_consumption(consumption_data)

            if len(daily_consumption) < 7:
                return None

            # Apply Triple Exponential Smoothing
            forecast_result = self._apply_holt_winters_forecast(daily_consumption, forecast_days)

            if forecast_result is None:
                return None

            predicted_demand, confidence_interval, seasonality_factor, trend_factor, accuracy = (
                forecast_result
            )

            return DemandForecast(
                item_id=item_id,
                predicted_demand=predicted_demand,
                confidence_interval=confidence_interval,
                seasonality_factor=seasonality_factor,
                trend_factor=trend_factor,
                forecast_horizon_days=forecast_days,
                forecast_accuracy=accuracy,
            )

        except Exception as e:
            self.logger.error(f"Error in demand prediction for item {item_id}: {e}")
            return None

    def _aggregate_daily_consumption(self, movements: List[StockMovement]) -> List[float]:
        """Aggregate stock movements into daily consumption data."""
        daily_data = {}

        for movement in movements:
            date_key = movement.movement_date.date()
            if date_key not in daily_data:
                daily_data[date_key] = 0.0
            daily_data[date_key] += float(movement.quantity)

        # Fill gaps with zeros and return chronological list
        if not daily_data:
            return []

        start_date = min(daily_data.keys())
        end_date = max(daily_data.keys())

        result = []
        current_date = start_date
        while current_date <= end_date:
            result.append(daily_data.get(current_date, 0.0))
            current_date += timedelta(days=1)

        return result

    def _apply_holt_winters_forecast(
        self, data: List[float], forecast_periods: int
    ) -> Optional[Tuple[float, Tuple[float, float], float, float, float]]:
        """Apply Holt-Winters Triple Exponential Smoothing for demand
        forecasting.

        Returns:
            Tuple of (predicted_demand, confidence_interval, seasonality_factor, trend_factor, accuracy)
        """
        if len(data) < 14:
            return None

        try:
            # Convert to numpy array for calculations
            y = np.array(data, dtype=float)
            n = len(y)

            # Detect seasonality period (weekly pattern)
            seasonal_period = min(7, n // 2)

            # Initialize components
            # Level (initial average)
            level = np.mean(y[:seasonal_period])

            # Trend (initial trend)
            if n >= 2 * seasonal_period:
                trend = (
                    np.mean(y[seasonal_period : 2 * seasonal_period]) - np.mean(y[:seasonal_period])
                ) / seasonal_period
            else:
                trend = 0.0

            # Seasonal factors
            seasonal = np.zeros(seasonal_period)
            for i in range(seasonal_period):
                seasonal_values = []
                for j in range(i, n, seasonal_period):
                    if j < n:
                        seasonal_values.append(y[j])
                if seasonal_values:
                    seasonal[i] = np.mean(seasonal_values) / level if level > 0 else 1.0

            # Normalize seasonal factors
            seasonal_sum = np.sum(seasonal)
            if seasonal_sum > 0:
                seasonal = seasonal * seasonal_period / seasonal_sum
            else:
                seasonal = np.ones(seasonal_period)

            # Apply smoothing
            levels = [level]
            trends = [trend]
            seasonals = [seasonal.copy()]
            fitted = []

            for t in range(n):
                # Current seasonal index
                s_idx = t % seasonal_period

                if t == 0:
                    fitted.append(level + trend + seasonal[s_idx])
                    continue

                # Update level
                new_level = self.alpha_smoothing * (y[t] / seasonal[s_idx]) + (
                    1 - self.alpha_smoothing
                ) * (level + trend)

                # Update trend
                new_trend = self.beta_trend * (new_level - level) + (1 - self.beta_trend) * trend

                # Update seasonal
                new_seasonal = seasonal.copy()
                new_seasonal[s_idx] = (
                    self.gamma_seasonality * (y[t] / new_level)
                    + (1 - self.gamma_seasonality) * seasonal[s_idx]
                )

                # Store values
                level = new_level
                trend = new_trend
                seasonal = new_seasonal

                levels.append(level)
                trends.append(trend)
                seasonals.append(seasonal.copy())

                # Calculate fitted value
                fitted_value = level + trend + seasonal[s_idx]
                fitted.append(fitted_value)

            # Generate forecast
            forecast_values = []
            for h in range(1, forecast_periods + 1):
                s_idx = (n + h - 1) % seasonal_period
                forecast_value = level + h * trend + seasonal[s_idx]
                forecast_values.append(max(0, forecast_value))  # Ensure non-negative

            # Calculate forecast accuracy (MAPE)
            accuracy = self._calculate_forecast_accuracy(y, fitted)

            # Calculate prediction confidence interval
            residuals = y - np.array(fitted[:n])
            mse = np.mean(residuals**2)
            std_error = math.sqrt(mse)

            # 95% confidence interval
            z_score = 1.96
            predicted_demand = np.mean(forecast_values)
            margin_of_error = z_score * std_error
            confidence_interval = (
                max(0, predicted_demand - margin_of_error),
                predicted_demand + margin_of_error,
            )

            # Calculate factors
            avg_seasonal = np.mean(seasonal)
            seasonality_factor = avg_seasonal
            trend_factor = trend

            return (
                predicted_demand,
                confidence_interval,
                seasonality_factor,
                trend_factor,
                accuracy,
            )

        except Exception as e:
            self.logger.error(f"Error in Holt-Winters forecasting: {e}")
            return None

    def _calculate_forecast_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE) for forecast
        accuracy."""
        try:
            # Remove zeros to avoid division by zero
            non_zero_mask = actual != 0
            if not np.any(non_zero_mask):
                return 0.0

            actual_nz = actual[non_zero_mask]
            predicted_nz = predicted[non_zero_mask]

            # Calculate MAPE
            mape = np.mean(np.abs((actual_nz - predicted_nz) / actual_nz)) * 100

            # Convert to accuracy (100 - MAPE)
            accuracy = max(0, 100 - mape) / 100
            return min(1.0, accuracy)

        except Exception:
            return 0.0

    async def calculate_optimal_reorder_point(
        self, session: Session, item_id: str
    ) -> Optional[OptimalReorderPoint]:
        """Calculate optimal reorder point using service level targets and
        demand variability.

        Uses statistical inventory theory to optimize reorder points considering:
        - Demand variability and lead time uncertainty
        - Target service level
        - Lead time demand distribution
        - Safety stock requirements

        Args:
            session: Database session
            item_id: Item ID to analyze

        Returns:
            OptimalReorderPoint with recommendations
        """
        try:
            item = session.query(Item).filter(Item.id == item_id).first()
            if not item:
                return None

            # Get demand forecast
            demand_forecast = await self.predict_demand(session, item_id)
            if (
                not demand_forecast
                or demand_forecast.forecast_accuracy < self.min_forecast_accuracy
            ):
                # Fallback to simple consumption analysis
                return await self._calculate_simple_reorder_point(session, item)

            # Get lead time information
            # Note: Fixed supplier query to properly join with items table
            # For now, use default lead time - in production would need supplier relationship
            lead_time_days = self.reorder_lead_time

            # Calculate demand statistics
            daily_demand = demand_forecast.predicted_demand / demand_forecast.forecast_horizon_days
            demand_std = self._estimate_demand_standard_deviation(session, item_id, daily_demand)

            # Lead time demand
            lead_time_demand = daily_demand * lead_time_days

            # Calculate safety stock using normal distribution
            # For service level (e.g., 95%), find z-score
            z_score = self._get_z_score_for_service_level(self.service_level_target)

            # Safety stock = z * sqrt(lead_time) * demand_std
            safety_stock = z_score * math.sqrt(lead_time_days) * demand_std

            # Optimal reorder point
            optimal_reorder_point = int(lead_time_demand + safety_stock)

            # Calculate optimal order quantity using Economic Order Quantity (EOQ)
            annual_demand = daily_demand * 365
            if annual_demand > 0:
                eoq = math.sqrt(
                    (2 * annual_demand * self.order_cost)
                    / (float(item.unit_cost) * self.holding_cost_rate)
                )
                optimal_order_quantity = max(int(eoq), item.reorder_quantity)
            else:
                optimal_order_quantity = item.reorder_quantity

            # Calculate total cost (ordering + holding + shortage)
            total_cost = self._calculate_total_inventory_cost(
                annual_demand,
                optimal_order_quantity,
                float(item.unit_cost),
                self.service_level_target,
                safety_stock,
            )

            return OptimalReorderPoint(
                item_id=item_id,
                optimal_reorder_point=optimal_reorder_point,
                optimal_reorder_quantity=optimal_order_quantity,
                service_level=self.service_level_target,
                safety_stock=int(safety_stock),
                lead_time_demand=lead_time_demand,
                demand_variability=demand_std,
                total_cost=total_cost,
            )

        except Exception as e:
            self.logger.error(f"Error calculating optimal reorder point for item {item_id}: {e}")
            return None

    async def _calculate_simple_reorder_point(
        self, session: Session, item: Item
    ) -> Optional[OptimalReorderPoint]:
        """Fallback calculation when advanced forecasting is not available."""
        try:
            # Get recent consumption data
            thirty_days_ago = datetime.now() - timedelta(days=30)

            consumption_movements = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.item_id == item.id,
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= thirty_days_ago,
                    )
                )
                .all()
            )

            if not consumption_movements:
                return None

            total_consumed = sum(movement.quantity for movement in consumption_movements)
            daily_consumption = total_consumed / 30

            # Simple lead time demand + safety stock
            lead_time_demand = daily_consumption * self.reorder_lead_time
            safety_stock = daily_consumption * 7  # 1 week safety stock

            optimal_reorder_point = int(lead_time_demand + safety_stock)

            return OptimalReorderPoint(
                item_id=item.id,
                optimal_reorder_point=optimal_reorder_point,
                optimal_reorder_quantity=item.reorder_quantity,
                service_level=0.90,  # Assumed service level
                safety_stock=int(safety_stock),
                lead_time_demand=lead_time_demand,
                demand_variability=daily_consumption * 0.3,  # Assumed 30% variability
                total_cost=0.0,
            )

        except Exception as e:
            self.logger.error(f"Error in simple reorder point calculation: {e}")
            return None

    def _estimate_demand_standard_deviation(
        self, session: Session, item_id: str, daily_demand: float
    ) -> float:
        """Estimate demand standard deviation from historical data."""
        try:
            # Get last 60 days of consumption data
            sixty_days_ago = datetime.now() - timedelta(days=60)

            movements = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.item_id == item_id,
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= sixty_days_ago,
                    )
                )
                .all()
            )

            if len(movements) < 5:
                # Fallback: assume 30% coefficient of variation
                return daily_demand * 0.3

            # Aggregate by day
            daily_consumption = self._aggregate_daily_consumption(movements)

            if len(daily_consumption) < 5:
                return daily_demand * 0.3

            # Calculate standard deviation
            std_dev = (
                statistics.stdev(daily_consumption)
                if len(daily_consumption) > 1
                else daily_demand * 0.3
            )
            return std_dev

        except Exception:
            # Fallback
            return daily_demand * 0.3

    def _get_z_score_for_service_level(self, service_level: float) -> float:
        """Get Z-score for given service level (normal distribution)."""
        # Common service levels and their z-scores
        service_levels = {
            0.50: 0.00,
            0.80: 0.84,
            0.85: 1.04,
            0.90: 1.28,
            0.95: 1.65,
            0.97: 1.88,
            0.99: 2.33,
            0.995: 2.58,
            0.999: 3.09,
        }

        # Find closest match or interpolate
        if service_level in service_levels:
            return service_levels[service_level]

        # Linear interpolation
        service_keys = sorted(service_levels.keys())
        for i in range(len(service_keys) - 1):
            if service_keys[i] <= service_level <= service_keys[i + 1]:
                # Linear interpolation
                x1, y1 = service_keys[i], service_levels[service_keys[i]]
                x2, y2 = service_keys[i + 1], service_levels[service_keys[i + 1]]
                return y1 + (service_level - x1) * (y2 - y1) / (x2 - x1)

        # Default for extreme values
        if service_level >= 1.0:
            return 1.65  # Default to 95% for 100% service level (impossible in practice)
        elif service_level >= 0.999:
            return 3.09
        else:
            return 1.65  # Default to 95%

    def _calculate_total_inventory_cost(
        self,
        annual_demand: float,
        order_quantity: int,
        unit_cost: float,
        service_level: float,
        safety_stock: float,
    ) -> float:
        """Calculate total inventory cost including ordering, holding, and
        shortage costs."""
        try:
            if annual_demand <= 0 or order_quantity <= 0:
                return 0.0

            # Ordering cost
            ordering_cost = (annual_demand / order_quantity) * self.order_cost

            # Holding cost (average inventory * holding rate * unit cost)
            average_inventory = (order_quantity / 2) + safety_stock
            holding_cost = average_inventory * self.holding_cost_rate * unit_cost

            # Shortage cost (simplified - based on stockout probability)
            shortage_probability = 1 - service_level
            shortage_cost = (
                shortage_probability * annual_demand * unit_cost * 0.1
            )  # 10% shortage cost rate

            return ordering_cost + holding_cost + shortage_cost

        except Exception:
            return 0.0

    async def optimize_bulk_purchase(
        self, session: Session, item_id: str, volume_discounts: List[Tuple[int, float]] = None
    ) -> Optional[BulkPurchaseOptimization]:
        """Optimize bulk purchase decisions considering volume discounts and
        holding costs.

        Args:
            session: Database session
            item_id: Item ID to analyze
            volume_discounts: List of (quantity, discount_rate) tuples

        Returns:
            BulkPurchaseOptimization with cost analysis and recommendations
        """
        try:
            item = session.query(Item).filter(Item.id == item_id).first()
            if not item:
                return None

            # Get demand forecast
            demand_forecast = await self.predict_demand(session, item_id)
            if not demand_forecast:
                return None

            # Default volume discounts if not provided
            if volume_discounts is None:
                base_unit_cost = float(item.unit_cost)
                volume_discounts = [
                    (item.reorder_quantity, 0.0),  # No discount for normal quantity
                    (item.reorder_quantity * 2, 0.02),  # 2% discount for 2x
                    (item.reorder_quantity * 5, 0.05),  # 5% discount for 5x
                    (item.reorder_quantity * 10, 0.08),  # 8% discount for 10x
                ]

            # Calculate optimal quantity for each discount tier
            best_option = None
            base_unit_cost = float(item.unit_cost)
            annual_demand = (
                demand_forecast.predicted_demand / demand_forecast.forecast_horizon_days
            ) * 365

            for quantity, discount_rate in volume_discounts:
                if quantity <= 0:
                    continue

                # Calculate unit cost with discount
                discounted_unit_cost = base_unit_cost * (1 - discount_rate)

                # Calculate total costs
                total_cost = self._calculate_bulk_purchase_total_cost(
                    annual_demand, quantity, discounted_unit_cost
                )

                # Calculate savings compared to base option
                base_cost = self._calculate_bulk_purchase_total_cost(
                    annual_demand, item.reorder_quantity, base_unit_cost
                )
                cost_savings = base_cost - total_cost

                # Calculate holding cost impact
                extra_inventory = max(0, quantity - item.reorder_quantity)
                holding_cost_impact = (
                    extra_inventory * discounted_unit_cost * self.holding_cost_rate
                )

                # Calculate break-even point (days to consume extra inventory)
                daily_demand = annual_demand / 365
                break_even_days = extra_inventory / daily_demand if daily_demand > 0 else 999

                # Determine optimal purchase timing
                if break_even_days <= 90:  # Within 3 months
                    timing = datetime.now() + timedelta(days=7)  # Buy soon
                elif break_even_days <= 180:  # Within 6 months
                    timing = datetime.now() + timedelta(days=30)  # Buy within a month
                else:
                    timing = datetime.now() + timedelta(days=60)  # Buy within 2 months

                option = {
                    "quantity": quantity,
                    "unit_cost": discounted_unit_cost,
                    "total_cost": total_cost,
                    "cost_savings": cost_savings,
                    "holding_cost_impact": holding_cost_impact,
                    "break_even_days": break_even_days,
                    "timing": timing,
                }

                # Select best option (highest positive savings considering holding costs)
                net_savings = cost_savings - holding_cost_impact
                if best_option is None or net_savings > best_option.get(
                    "net_savings", -float("inf")
                ):
                    option["net_savings"] = net_savings
                    best_option = option

            if not best_option or best_option["cost_savings"] <= 0:
                return None

            return BulkPurchaseOptimization(
                item_id=item_id,
                optimal_order_quantity=best_option["quantity"],
                unit_cost_with_discount=Decimal(str(best_option["unit_cost"])),
                total_cost_savings=Decimal(str(best_option["cost_savings"])),
                break_even_point=int(best_option["break_even_days"]),
                holding_cost_impact=Decimal(str(best_option["holding_cost_impact"])),
                recommended_purchase_timing=best_option["timing"],
            )

        except Exception as e:
            self.logger.error(f"Error optimizing bulk purchase for item {item_id}: {e}")
            return None

    def _calculate_bulk_purchase_total_cost(
        self, annual_demand: float, order_quantity: int, unit_cost: float
    ) -> float:
        """Calculate total cost for bulk purchase option."""
        try:
            if annual_demand <= 0 or order_quantity <= 0:
                return float("inf")

            # Annual ordering cost
            ordering_cost = (annual_demand / order_quantity) * self.order_cost

            # Annual holding cost
            average_inventory = order_quantity / 2
            holding_cost = average_inventory * self.holding_cost_rate * unit_cost

            # Purchase cost
            purchase_cost = annual_demand * unit_cost

            return ordering_cost + holding_cost + purchase_cost

        except Exception:
            return float("inf")

    async def analyze_bulk_purchase_opportunities(
        self, session: Session
    ) -> List[BulkPurchaseOptimization]:
        """Analyze bulk purchase opportunities for all active items."""
        opportunities = []

        try:
            items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()

            for item in items:
                opportunity = await self.optimize_bulk_purchase(session, item.id)
                if opportunity and opportunity.total_cost_savings > 100:  # Minimum $100 savings
                    opportunities.append(opportunity)

            # Sort by cost savings (descending)
            opportunities.sort(key=lambda x: x.total_cost_savings, reverse=True)

        except Exception as e:
            self.logger.error(f"Error analyzing bulk purchase opportunities: {e}")

        return opportunities

    async def predict_expiry_waste(
        self, session: Session, item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Predict and minimize waste from expiring inventory using advanced
        analytics.

        Args:
            session: Database session
            item_id: Item ID to analyze

        Returns:
            Dictionary with expiry predictions and waste minimization strategies
        """
        try:
            item = session.query(Item).filter(Item.id == item_id).first()
            if not item or not item.expiry_days:
                return None

            # Get current stock and consumption patterns
            demand_forecast = await self.predict_demand(session, item_id)
            if not demand_forecast:
                return None

            daily_consumption = (
                demand_forecast.predicted_demand / demand_forecast.forecast_horizon_days
            )

            # Calculate expiry risk based on current stock and consumption rate
            current_stock = item.current_stock
            days_to_consume = current_stock / daily_consumption if daily_consumption > 0 else 999

            # Risk assessment
            if days_to_consume > item.expiry_days:
                waste_risk = "high"
                predicted_waste = current_stock - (daily_consumption * item.expiry_days)
                waste_value = predicted_waste * float(item.unit_cost)
            elif days_to_consume > item.expiry_days * 0.8:
                waste_risk = "medium"
                predicted_waste = max(
                    0, current_stock - (daily_consumption * item.expiry_days * 0.9)
                )
                waste_value = predicted_waste * float(item.unit_cost)
            else:
                waste_risk = "low"
                predicted_waste = 0
                waste_value = 0

            # Generate waste minimization strategies
            strategies = self._generate_waste_minimization_strategies(
                item, daily_consumption, predicted_waste, waste_risk
            )

            # Calculate optimal order timing to minimize expiry risk
            optimal_reorder_timing = self._calculate_optimal_reorder_timing(
                item, daily_consumption, demand_forecast.seasonality_factor
            )

            return {
                "item_id": item_id,
                "item_name": item.name,
                "current_stock": current_stock,
                "expiry_days": item.expiry_days,
                "daily_consumption": daily_consumption,
                "days_to_consume": days_to_consume,
                "waste_risk": waste_risk,
                "predicted_waste": predicted_waste,
                "waste_value": waste_value,
                "strategies": strategies,
                "optimal_reorder_timing": optimal_reorder_timing,
            }

        except Exception as e:
            self.logger.error(f"Error predicting expiry waste for item {item_id}: {e}")
            return None

    def _generate_waste_minimization_strategies(
        self, item: Item, daily_consumption: float, predicted_waste: float, waste_risk: str
    ) -> List[Dict[str, Any]]:
        """Generate actionable waste minimization strategies."""
        strategies = []

        if waste_risk == "high" and predicted_waste > 0:
            # Aggressive strategies for high waste risk
            strategies.extend(
                [
                    {
                        "type": "promotional_pricing",
                        "description": "Offer 15-25% discount to accelerate sales",
                        "expected_impact": predicted_waste * 0.7,
                        "urgency": "immediate",
                    },
                    {
                        "type": "staff_training",
                        "description": "Train staff to prioritize FIFO (First In, First Out)",
                        "expected_impact": predicted_waste * 0.3,
                        "urgency": "immediate",
                    },
                    {
                        "type": "bundle_offers",
                        "description": "Create bundles with slower-moving items",
                        "expected_impact": predicted_waste * 0.5,
                        "urgency": "within_24h",
                    },
                ]
            )

        elif waste_risk == "medium":
            strategies.extend(
                [
                    {
                        "type": "increase_visibility",
                        "description": "Move items to prominent display locations",
                        "expected_impact": predicted_waste * 0.4,
                        "urgency": "within_48h",
                    },
                    {
                        "type": "targeted_marketing",
                        "description": "Send targeted promotions to frequent customers",
                        "expected_impact": predicted_waste * 0.6,
                        "urgency": "within_48h",
                    },
                ]
            )

        # Universal strategies
        strategies.append(
            {
                "type": "donation_partnership",
                "description": "Partner with local food banks or charities",
                "expected_impact": min(predicted_waste, daily_consumption * 2),
                "urgency": "ongoing",
            }
        )

        # Adjust reorder strategy
        if daily_consumption > 0:
            reduced_order_qty = max(
                int(daily_consumption * item.expiry_days * 0.8), item.minimum_stock
            )
            if reduced_order_qty < item.reorder_quantity:
                strategies.append(
                    {
                        "type": "reduce_order_quantity",
                        "description": f"Reduce next order to {reduced_order_qty} units",
                        "expected_impact": "Prevent future waste",
                        "urgency": "next_order",
                    }
                )

        return strategies

    def _calculate_optimal_reorder_timing(
        self, item: Item, daily_consumption: float, seasonality_factor: float
    ) -> Dict[str, Any]:
        """Calculate optimal reorder timing to minimize expiry risk."""
        try:
            if daily_consumption <= 0:
                return {"timing": "manual_review", "reason": "insufficient_consumption_data"}

            # Calculate optimal stock level (percentage of expiry period)
            if item.expiry_days <= 7:  # Very perishable
                optimal_stock_days = item.expiry_days * 0.6
            elif item.expiry_days <= 30:  # Moderately perishable
                optimal_stock_days = item.expiry_days * 0.7
            else:  # Less perishable
                optimal_stock_days = item.expiry_days * 0.8

            # Adjust for seasonality
            if seasonality_factor > 1.2:  # High season
                optimal_stock_days *= 1.1
            elif seasonality_factor < 0.8:  # Low season
                optimal_stock_days *= 0.9

            optimal_stock_level = int(daily_consumption * optimal_stock_days)

            # Calculate when to reorder
            consumption_until_reorder = max(item.current_stock - optimal_stock_level, 0)
            days_until_reorder = consumption_until_reorder / daily_consumption

            return {
                "optimal_stock_level": optimal_stock_level,
                "days_until_reorder": days_until_reorder,
                "reorder_date": datetime.now() + timedelta(days=days_until_reorder),
                "reason": "expiry_optimized",
            }

        except Exception:
            return {"timing": "manual_review", "reason": "calculation_error"}

    async def analyze_all_expiry_risks(self, session: Session) -> List[Dict[str, Any]]:
        """Analyze expiry risks for all perishable items."""
        expiry_risks = []

        try:
            # Get all items with expiry dates
            perishable_items = (
                session.query(Item)
                .filter(
                    and_(
                        Item.expiry_days.isnot(None),
                        Item.current_stock > 0,
                        Item.status == ItemStatus.ACTIVE,
                    )
                )
                .all()
            )

            for item in perishable_items:
                risk_analysis = await self.predict_expiry_waste(session, item.id)
                if risk_analysis:
                    expiry_risks.append(risk_analysis)

            # Sort by waste value (highest first)
            expiry_risks.sort(key=lambda x: x["waste_value"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error analyzing expiry risks: {e}")

        return expiry_risks

    async def analyze_seasonality_patterns(
        self, session: Session, item_id: str, analysis_periods: List[int] = None
    ) -> Optional[SeasonalityAnalysis]:
        """Advanced seasonality detection using multiple period analysis.

        Args:
            session: Database session
            item_id: Item ID to analyze
            analysis_periods: Periods to test for seasonality (defaults to [7, 30, 90, 365])

        Returns:
            SeasonalityAnalysis with detected patterns
        """
        if analysis_periods is None:
            analysis_periods = [7, 30, 90, 365]  # Weekly, monthly, quarterly, yearly

        try:
            # Get longer historical data for seasonality analysis
            cutoff_date = datetime.now() - timedelta(days=max(730, max(analysis_periods) * 2))

            consumption_data = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.item_id == item_id,
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= cutoff_date,
                    )
                )
                .order_by(StockMovement.movement_date)
                .all()
            )

            if len(consumption_data) < max(analysis_periods) * 2:
                return None

            daily_consumption = self._aggregate_daily_consumption(consumption_data)
            if len(daily_consumption) < max(analysis_periods):
                return None

            # Detect seasonal patterns for each period
            seasonal_results = {}
            for period in analysis_periods:
                if len(daily_consumption) >= period * 2:
                    strength, peaks, lows = self._analyze_seasonal_period(daily_consumption, period)
                    seasonal_results[period] = {"strength": strength, "peaks": peaks, "lows": lows}

            if not seasonal_results:
                return None

            # Find strongest seasonal pattern
            strongest_period = max(
                seasonal_results.keys(), key=lambda p: seasonal_results[p]["strength"]
            )
            strongest_strength = seasonal_results[strongest_period]["strength"]

            # Calculate current period adjustment
            current_day_of_period = (datetime.now() - cutoff_date).days % strongest_period
            current_adjustment = self._calculate_seasonal_adjustment(
                daily_consumption, strongest_period, current_day_of_period
            )

            # Determine confidence based on data length and pattern strength
            data_quality = min(1.0, len(daily_consumption) / (strongest_period * 4))
            confidence = strongest_strength * data_quality

            return SeasonalityAnalysis(
                item_id=item_id,
                seasonal_periods=list(seasonal_results.keys()),
                seasonal_strength=strongest_strength,
                peak_periods=seasonal_results[strongest_period]["peaks"],
                low_periods=seasonal_results[strongest_period]["lows"],
                seasonal_adjustment_factor=current_adjustment,
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Error in seasonality analysis for item {item_id}: {e}")
            return None

    def _analyze_seasonal_period(
        self, data: List[float], period: int
    ) -> Tuple[float, List[int], List[int]]:
        """Analyze seasonality for a specific period."""
        try:
            data_array = np.array(data)
            n_cycles = len(data) // period

            if n_cycles < 2:
                return 0.0, [], []

            # Reshape data into cycles
            cycles_data = data_array[: n_cycles * period].reshape(n_cycles, period)

            # Calculate average pattern
            avg_pattern = np.mean(cycles_data, axis=0)
            overall_mean = np.mean(avg_pattern)

            if overall_mean == 0:
                return 0.0, [], []

            # Calculate seasonal strength (coefficient of variation of the pattern)
            pattern_std = np.std(avg_pattern)
            seasonal_strength = min(1.0, pattern_std / overall_mean)

            # Find peaks and lows (top/bottom 20% of pattern)
            threshold_high = np.percentile(avg_pattern, 80)
            threshold_low = np.percentile(avg_pattern, 20)

            peaks = [i for i, val in enumerate(avg_pattern) if val >= threshold_high]
            lows = [i for i, val in enumerate(avg_pattern) if val <= threshold_low]

            return seasonal_strength, peaks, lows

        except Exception:
            return 0.0, [], []

    def _calculate_seasonal_adjustment(
        self, data: List[float], period: int, current_day: int
    ) -> float:
        """Calculate seasonal adjustment factor for current day."""
        try:
            data_array = np.array(data)
            n_cycles = len(data) // period

            if n_cycles < 1:
                return 1.0

            cycles_data = data_array[: n_cycles * period].reshape(n_cycles, period)
            avg_pattern = np.mean(cycles_data, axis=0)
            overall_mean = np.mean(avg_pattern)

            if overall_mean == 0:
                return 1.0

            # Adjustment factor for current day in the period
            current_day_avg = avg_pattern[current_day % period]
            adjustment = current_day_avg / overall_mean

            # Clamp adjustment to reasonable range
            return max(0.5, min(2.0, adjustment))

        except Exception:
            return 1.0

    async def analyze_item_correlations(
        self, session: Session, item_id: str, correlation_window_days: int = 90
    ) -> Optional[ItemCorrelationAnalysis]:
        """Analyze correlations between items for demand planning.

        Args:
            session: Database session
            item_id: Primary item to analyze
            correlation_window_days: Time window for correlation analysis

        Returns:
            ItemCorrelationAnalysis with cross-item insights
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=correlation_window_days)

            # Get consumption data for all items in the time window
            all_movements = (
                session.query(StockMovement)
                .filter(
                    and_(
                        StockMovement.movement_type == StockMovementType.OUT,
                        StockMovement.movement_date >= cutoff_date,
                    )
                )
                .order_by(StockMovement.movement_date)
                .all()
            )

            # Group by item and create daily consumption series
            item_consumption = {}
            for movement in all_movements:
                if movement.item_id not in item_consumption:
                    item_consumption[movement.item_id] = {}

                date_key = movement.movement_date.date()
                if date_key not in item_consumption[movement.item_id]:
                    item_consumption[movement.item_id][date_key] = 0.0
                item_consumption[movement.item_id][date_key] += float(movement.quantity)

            if item_id not in item_consumption or len(item_consumption) < 3:
                return None

            # Convert to aligned time series
            all_dates = sorted(set().union(*[dates.keys() for dates in item_consumption.values()]))
            if len(all_dates) < 30:  # Need sufficient data points
                return None

            # Create consumption matrix
            consumption_matrix = {}
            for item, dates_consumption in item_consumption.items():
                consumption_matrix[item] = [dates_consumption.get(date, 0.0) for date in all_dates]

            # Calculate correlations with primary item
            primary_consumption = np.array(consumption_matrix[item_id])
            correlations = []

            for other_item, other_consumption in consumption_matrix.items():
                if other_item != item_id:
                    other_array = np.array(other_consumption)

                    # Calculate Pearson correlation
                    if np.std(primary_consumption) > 0 and np.std(other_array) > 0:
                        correlation = np.corrcoef(primary_consumption, other_array)[0, 1]
                        if not np.isnan(correlation):
                            correlations.append((other_item, correlation))

            # Sort by absolute correlation strength
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            # Categorize relationships
            substitution_items = [
                item for item, corr in correlations if corr < -0.3
            ]  # Negative correlation
            complementary_items = [
                item for item, corr in correlations if corr > 0.5
            ]  # Strong positive correlation

            # Calculate impact factor (how much this item affects others)
            impact_scores = [abs(corr) for _, corr in correlations[:10]]  # Top 10 correlations
            impact_factor = np.mean(impact_scores) if impact_scores else 0.0

            # Generate bundle opportunities
            bundle_opportunities = self._generate_bundle_opportunities(
                session, item_id, complementary_items[:5]
            )

            return ItemCorrelationAnalysis(
                primary_item_id=item_id,
                correlated_items=correlations[:15],  # Top 15 correlations
                substitution_items=substitution_items[:5],
                complementary_items=complementary_items[:5],
                impact_factor=impact_factor,
                bundle_opportunities=bundle_opportunities,
            )

        except Exception as e:
            self.logger.error(f"Error in item correlation analysis for {item_id}: {e}")
            return None

    def _generate_bundle_opportunities(
        self, session: Session, primary_item_id: str, complementary_items: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate bundling opportunities based on item correlations."""
        opportunities = []

        try:
            # Get item details for bundle analysis
            primary_item = session.query(Item).filter(Item.id == primary_item_id).first()
            if not primary_item:
                return opportunities

            for comp_item_id in complementary_items:
                comp_item = session.query(Item).filter(Item.id == comp_item_id).first()
                if comp_item:
                    # Calculate potential bundle pricing
                    combined_cost = float(primary_item.unit_cost) + float(comp_item.unit_cost)
                    suggested_bundle_price = combined_cost * 1.15  # 15% markup
                    individual_price = float(primary_item.selling_price or 0) + float(
                        comp_item.selling_price or 0
                    )

                    if individual_price > 0:
                        discount_percentage = (
                            individual_price - suggested_bundle_price
                        ) / individual_price

                        opportunities.append(
                            {
                                "primary_item": primary_item.name,
                                "complementary_item": comp_item.name,
                                "bundle_price": suggested_bundle_price,
                                "individual_price": individual_price,
                                "customer_savings": individual_price - suggested_bundle_price,
                                "discount_percentage": discount_percentage,
                                "estimated_margin_improvement": 0.05,  # Estimated 5% margin improvement
                            }
                        )

        except Exception as e:
            self.logger.error(f"Error generating bundle opportunities: {e}")

        return opportunities

    async def analyze_supplier_diversification(
        self, session: Session, item_id: str
    ) -> Optional[SupplierDiversificationAnalysis]:
        """Analyze supplier concentration risk and diversification
        opportunities.

        Args:
            session: Database session
            item_id: Item to analyze for supplier diversification

        Returns:
            SupplierDiversificationAnalysis with risk assessment and recommendations
        """
        try:
            # Get recent purchase orders for this item
            analysis_period = datetime.now() - timedelta(days=180)  # 6 months

            purchase_orders = (
                session.query(PurchaseOrder)
                .join(PurchaseOrderItem)
                .filter(
                    and_(
                        PurchaseOrderItem.item_id == item_id,
                        PurchaseOrder.order_date >= analysis_period,
                    )
                )
                .all()
            )

            if not purchase_orders:
                return None

            # Calculate supplier concentration
            supplier_volumes = {}
            total_volume = 0

            for order in purchase_orders:
                supplier_id = order.supplier_id
                order_items = (
                    session.query(PurchaseOrderItem)
                    .filter(
                        and_(
                            PurchaseOrderItem.purchase_order_id == order.id,
                            PurchaseOrderItem.item_id == item_id,
                        )
                    )
                    .all()
                )

                for order_item in order_items:
                    volume = float(order_item.total_cost)
                    supplier_volumes[supplier_id] = supplier_volumes.get(supplier_id, 0) + volume
                    total_volume += volume

            if total_volume == 0:
                return None

            # Calculate concentration ratio (Herfindahl-Hirschman Index)
            concentration_scores = [
                (volume / total_volume) ** 2 for volume in supplier_volumes.values()
            ]
            concentration_index = sum(concentration_scores)

            # Get all potential suppliers for this item category
            item = session.query(Item).filter(Item.id == item_id).first()
            if not item:
                return None

            # Find alternative suppliers (simplified - in practice would use category matching)
            all_suppliers = session.query(Supplier).filter(Supplier.is_active).all()
            current_suppliers = set(supplier_volumes.keys())
            alternative_suppliers = [s.id for s in all_suppliers if s.id not in current_suppliers]

            # Calculate risk score
            risk_factors = {
                "concentration": concentration_index,  # Higher concentration = higher risk
                "supplier_count": min(
                    1.0, len(supplier_volumes) / 3
                ),  # Fewer suppliers = higher risk
                "geographic_risk": 0.3,  # Placeholder - would analyze supplier locations
                "supplier_stability": 0.2,  # Placeholder - would analyze supplier financial health
            }

            risk_score = (
                risk_factors["concentration"] * 0.4
                + (1 - risk_factors["supplier_count"]) * 0.3
                + risk_factors["geographic_risk"] * 0.2
                + risk_factors["supplier_stability"] * 0.1
            )

            # Generate diversification recommendations
            recommendations = self._generate_diversification_recommendations(
                concentration_index, len(supplier_volumes), alternative_suppliers
            )

            # Recommend optimal supplier split
            recommended_split = self._calculate_optimal_supplier_split(
                supplier_volumes, total_volume, risk_score
            )

            # Estimate cost impact of diversification
            cost_impact = self._estimate_diversification_cost_impact(
                len(supplier_volumes), len(alternative_suppliers)
            )

            return SupplierDiversificationAnalysis(
                item_id=item_id,
                current_supplier_concentration=concentration_index,
                alternative_suppliers=alternative_suppliers[:5],  # Top 5 alternatives
                risk_score=risk_score,
                diversification_recommendations=recommendations,
                cost_impact_of_diversification=cost_impact,
                recommended_supplier_split=recommended_split,
            )

        except Exception as e:
            self.logger.error(f"Error in supplier diversification analysis for {item_id}: {e}")
            return None

    def _generate_diversification_recommendations(
        self,
        concentration_index: float,
        current_supplier_count: int,
        alternative_suppliers: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate specific diversification recommendations."""
        recommendations = []

        if concentration_index > 0.5:  # High concentration
            recommendations.append(
                {
                    "type": "reduce_concentration",
                    "priority": "high",
                    "description": "Supplier concentration is high. Distribute orders among more suppliers.",
                    "target_concentration": 0.3,
                    "estimated_timeline": "3-6 months",
                }
            )

        if current_supplier_count < 2:
            recommendations.append(
                {
                    "type": "add_suppliers",
                    "priority": "critical",
                    "description": "Single supplier dependency creates critical risk. Add backup suppliers.",
                    "target_supplier_count": 3,
                    "estimated_timeline": "1-3 months",
                }
            )

        if len(alternative_suppliers) > 0:
            recommendations.append(
                {
                    "type": "evaluate_alternatives",
                    "priority": "medium",
                    "description": f"Evaluate {min(3, len(alternative_suppliers))} alternative suppliers for cost and quality.",
                    "suppliers_to_evaluate": alternative_suppliers[:3],
                    "estimated_timeline": "2-4 months",
                }
            )

        return recommendations

    def _calculate_optimal_supplier_split(
        self, current_volumes: Dict[str, float], total_volume: float, risk_score: float
    ) -> Dict[str, float]:
        """Calculate optimal split percentages among suppliers."""
        supplier_count = len(current_volumes)

        if risk_score > 0.7:  # High risk - more even distribution
            # Target more even distribution
            target_main_supplier = 0.5  # Maximum 50% to any single supplier
            remaining_split = (1.0 - target_main_supplier) / max(1, supplier_count - 1)
        elif risk_score > 0.4:  # Medium risk
            target_main_supplier = 0.7  # Maximum 70% to main supplier
            remaining_split = (1.0 - target_main_supplier) / max(1, supplier_count - 1)
        else:  # Low risk - current distribution likely OK
            # Maintain roughly current proportions but cap at 80%
            proportions = {k: v / total_volume for k, v in current_volumes.items()}
            main_supplier = max(proportions.values())
            if main_supplier > 0.8:
                target_main_supplier = 0.8
                remaining_split = (1.0 - target_main_supplier) / max(1, supplier_count - 1)
            else:
                return proportions

        # Build recommended split
        suppliers = list(current_volumes.keys())
        recommended_split = {}

        # Assign main supplier
        main_supplier_id = max(current_volumes.keys(), key=lambda k: current_volumes[k])
        recommended_split[main_supplier_id] = target_main_supplier

        # Distribute remainder
        for supplier_id in suppliers:
            if supplier_id != main_supplier_id:
                recommended_split[supplier_id] = remaining_split

        return recommended_split

    def _estimate_diversification_cost_impact(
        self, current_suppliers: int, alternatives: int
    ) -> float:
        """Estimate cost impact of supplier diversification."""
        # Simplified cost impact estimation
        if current_suppliers == 1:
            # Moving from single to multiple suppliers
            return 0.05  # 5% cost increase due to smaller order volumes
        elif current_suppliers < 3 and alternatives > 0:
            # Adding suppliers for better diversification
            return 0.02  # 2% cost increase
        else:
            # Already well diversified
            return 0.0

    async def analyze_supplier_performance_advanced(
        self, session: Session, supplier_id: str = None
    ) -> List[SupplierPerformanceMetrics]:
        """Advanced supplier performance analysis with multi-factor evaluation.

        Args:
            session: Database session
            supplier_id: Optional specific supplier ID, or None for all suppliers

        Returns:
            List of SupplierPerformanceMetrics with comprehensive analysis
        """
        try:
            # Get suppliers to analyze
            if supplier_id:
                suppliers = (
                    session.query(Supplier)
                    .filter(and_(Supplier.id == supplier_id, Supplier.is_active))
                    .all()
                )
            else:
                suppliers = session.query(Supplier).filter(Supplier.is_active).all()

            if not suppliers:
                return []

            performance_metrics = []
            analysis_period = datetime.now() - timedelta(days=180)  # 6 months

            for supplier in suppliers:
                metrics = await self._calculate_comprehensive_supplier_metrics(
                    session, supplier, analysis_period
                )
                if metrics:
                    performance_metrics.append(metrics)

            # Sort by overall performance score (descending)
            performance_metrics.sort(key=lambda x: x.overall_performance_score, reverse=True)

            return performance_metrics

        except Exception as e:
            self.logger.error(f"Error in advanced supplier performance analysis: {e}")
            return []

    async def _calculate_comprehensive_supplier_metrics(
        self, session: Session, supplier: Supplier, analysis_period: datetime
    ) -> Optional[SupplierPerformanceMetrics]:
        """Calculate comprehensive performance metrics for a supplier."""
        try:
            # Get purchase orders for this supplier in the analysis period
            orders = (
                session.query(PurchaseOrder)
                .filter(
                    and_(
                        PurchaseOrder.supplier_id == supplier.id,
                        PurchaseOrder.order_date >= analysis_period,
                    )
                )
                .all()
            )

            if not orders:
                return None

            # 1. On-time delivery rate
            on_time_delivery_rate = self._calculate_on_time_delivery_rate(orders)

            # 2. Quality score (based on returns, discrepancies)
            quality_score = self._calculate_quality_score(session, supplier, orders)

            # 3. Cost competitiveness
            cost_competitiveness = self._calculate_cost_competitiveness(session, supplier, orders)

            # 4. Reliability index (consistency of performance)
            reliability_index = self._calculate_reliability_index(orders, supplier)

            # 5. Lead time variability
            lead_time_variability = self._calculate_lead_time_variability(orders, supplier)

            # 6. Overall performance score (weighted combination)
            overall_score = self._calculate_overall_performance_score(
                on_time_delivery_rate,
                quality_score,
                cost_competitiveness,
                reliability_index,
                lead_time_variability,
            )

            # 7. Recommend action based on performance
            recommended_action = self._recommend_supplier_action(
                overall_score,
                {
                    "on_time": on_time_delivery_rate,
                    "quality": quality_score,
                    "cost": cost_competitiveness,
                    "reliability": reliability_index,
                    "lead_time_var": lead_time_variability,
                },
            )

            return SupplierPerformanceMetrics(
                supplier_id=supplier.id,
                on_time_delivery_rate=on_time_delivery_rate,
                quality_score=quality_score,
                cost_competitiveness=cost_competitiveness,
                reliability_index=reliability_index,
                lead_time_variability=lead_time_variability,
                overall_performance_score=overall_score,
                recommended_action=recommended_action,
            )

        except Exception as e:
            self.logger.error(f"Error calculating supplier metrics for {supplier.id}: {e}")
            return None

    def _calculate_on_time_delivery_rate(self, orders: List[PurchaseOrder]) -> float:
        """Calculate on-time delivery rate."""
        if not orders:
            return 0.0

        on_time_count = 0
        delivered_orders = 0

        for order in orders:
            if order.status in ["delivered", "completed"]:
                delivered_orders += 1
                # Simplified: assume delivered orders with status 'delivered' are on time
                # In real implementation, you'd compare actual vs expected delivery dates
                if order.expected_delivery_date:
                    # Assume on-time if order is marked as delivered
                    on_time_count += 1

        return on_time_count / delivered_orders if delivered_orders > 0 else 0.0

    def _calculate_quality_score(
        self, session: Session, supplier: Supplier, orders: List[PurchaseOrder]
    ) -> float:
        """Calculate quality score based on returns and discrepancies."""
        # Simplified quality calculation
        # In a real system, you'd track returns, damage, quality issues

        total_value = sum(float(order.total_amount) for order in orders)
        if total_value == 0:
            return 0.5  # Neutral score

        # Base quality score from supplier rating
        base_score = float(supplier.rating) / 5.0 if supplier.rating else 0.5

        # Adjust based on order consistency
        order_count = len(orders)
        if order_count >= 10:
            consistency_bonus = 0.1
        elif order_count >= 5:
            consistency_bonus = 0.05
        else:
            consistency_bonus = 0.0

        quality_score = min(1.0, base_score + consistency_bonus)
        return quality_score

    def _calculate_cost_competitiveness(
        self, session: Session, supplier: Supplier, orders: List[PurchaseOrder]
    ) -> float:
        """Calculate cost competitiveness compared to market average."""
        if not orders:
            return 0.5

        # Get items supplied by this supplier
        supplier_items = set()
        for order in orders:
            # Get order items
            order_items = (
                session.query(PurchaseOrderItem)
                .filter(PurchaseOrderItem.purchase_order_id == order.id)
                .all()
            )

            for order_item in order_items:
                supplier_items.add(order_item.item_id)

        if not supplier_items:
            return 0.5

        # Compare unit costs with other suppliers for same items
        competitive_scores = []

        for item_id in supplier_items:
            item = session.query(Item).filter(Item.id == item_id).first()
            if item:
                # Get recent orders for this item from other suppliers
                recent_orders = (
                    session.query(PurchaseOrder)
                    .join(PurchaseOrderItem)
                    .filter(
                        and_(
                            PurchaseOrderItem.item_id == item_id,
                            PurchaseOrder.supplier_id != supplier.id,
                            PurchaseOrder.order_date >= datetime.now() - timedelta(days=90),
                        )
                    )
                    .all()
                )

                if recent_orders:
                    # Compare average costs
                    supplier_cost = float(item.unit_cost)
                    market_costs = []

                    for market_order in recent_orders:
                        market_items = (
                            session.query(PurchaseOrderItem)
                            .filter(
                                and_(
                                    PurchaseOrderItem.purchase_order_id == market_order.id,
                                    PurchaseOrderItem.item_id == item_id,
                                )
                            )
                            .all()
                        )

                        for market_item in market_items:
                            market_costs.append(float(market_item.unit_cost))

                    if market_costs:
                        avg_market_cost = statistics.mean(market_costs)
                        if avg_market_cost > 0:
                            # Lower cost = higher competitiveness
                            competitiveness = min(1.0, avg_market_cost / supplier_cost)
                            competitive_scores.append(competitiveness)

        return statistics.mean(competitive_scores) if competitive_scores else 0.5

    def _calculate_reliability_index(
        self, orders: List[PurchaseOrder], supplier: Supplier
    ) -> float:
        """Calculate reliability based on consistency of performance."""
        if len(orders) < 3:
            return 0.5  # Insufficient data

        # Calculate variance in delivery performance
        order_values = [float(order.total_amount) for order in orders]
        lead_times = []

        for order in orders:
            if order.expected_delivery_date and order.order_date:
                expected_lead_time = (order.expected_delivery_date - order.order_date).days
                actual_lead_time = supplier.lead_time_days  # Simplified
                lead_times.append(abs(expected_lead_time - actual_lead_time))

        # Lower variance = higher reliability
        if lead_times:
            lead_time_variance = statistics.variance(lead_times) if len(lead_times) > 1 else 0
            # Normalize variance to 0-1 scale (inverse relationship)
            reliability = max(0.0, 1.0 - (lead_time_variance / 10.0))
        else:
            reliability = 0.5

        # Adjust for order value consistency
        if order_values:
            value_variance = statistics.variance(order_values) if len(order_values) > 1 else 0
            avg_value = statistics.mean(order_values)
            cv = (math.sqrt(value_variance) / avg_value) if avg_value > 0 else 0
            # Lower coefficient of variation = higher reliability
            value_reliability = max(0.0, 1.0 - cv)
            reliability = (reliability + value_reliability) / 2

        return min(1.0, reliability)

    def _calculate_lead_time_variability(
        self, orders: List[PurchaseOrder], supplier: Supplier
    ) -> float:
        """Calculate lead time variability (lower is better)."""
        if not orders:
            return 1.0  # High variability (bad)

        actual_lead_times = []

        for order in orders:
            if order.expected_delivery_date and order.order_date:
                actual_lead_time = (order.expected_delivery_date - order.order_date).days
                actual_lead_times.append(actual_lead_time)

        if not actual_lead_times:
            return 0.5  # Neutral

        # Calculate coefficient of variation
        if len(actual_lead_times) > 1:
            std_dev = statistics.stdev(actual_lead_times)
            mean_lead_time = statistics.mean(actual_lead_times)
            cv = std_dev / mean_lead_time if mean_lead_time > 0 else 1.0
        else:
            cv = 0.0

        # Convert to 0-1 scale where 0 = high variability, 1 = low variability
        variability_score = max(0.0, 1.0 - cv)
        return variability_score

    def _calculate_overall_performance_score(
        self,
        on_time_rate: float,
        quality_score: float,
        cost_competitiveness: float,
        reliability_index: float,
        lead_time_variability: float,
    ) -> float:
        """Calculate weighted overall performance score."""
        # Weights for different factors
        weights = {
            "on_time": 0.25,  # 25% - On-time delivery
            "quality": 0.25,  # 25% - Quality
            "cost": 0.20,  # 20% - Cost competitiveness
            "reliability": 0.15,  # 15% - Reliability
            "lead_time": 0.15,  # 15% - Lead time consistency
        }

        overall_score = (
            on_time_rate * weights["on_time"]
            + quality_score * weights["quality"]
            + cost_competitiveness * weights["cost"]
            + reliability_index * weights["reliability"]
            + lead_time_variability * weights["lead_time"]
        )

        return min(1.0, overall_score)

    def _recommend_supplier_action(self, overall_score: float, metrics: Dict[str, float]) -> str:
        """Recommend action based on supplier performance."""
        if overall_score >= 0.85:
            return "preferred_supplier"
        elif overall_score >= 0.70:
            return "continue_monitoring"
        elif overall_score >= 0.50:
            # Identify specific issues
            issues = []
            if metrics["on_time"] < 0.80:
                issues.append("delivery_delays")
            if metrics["quality"] < 0.70:
                issues.append("quality_concerns")
            if metrics["cost"] < 0.60:
                issues.append("cost_optimization_needed")

            if issues:
                return f"improvement_plan_required: {', '.join(issues)}"
            else:
                return "performance_discussion_needed"
        else:
            return "consider_alternative_suppliers"

    # Enhanced Analysis Methods for New Process Data Types

    async def _perform_advanced_reorder_analysis(self, session: Session) -> Optional[AgentDecision]:
        """Perform advanced reorder analysis using optimal reorder points."""
        try:
            items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()
            advanced_suggestions = []

            for item in items[:20]:  # Limit to prevent overwhelming analysis
                optimal_reorder = await self.calculate_optimal_reorder_point(session, item.id)
                if optimal_reorder:
                    # Compare with current settings
                    current_gap = abs(
                        (item.reorder_point or 0) - optimal_reorder.optimal_reorder_point
                    )
                    if current_gap > 5:  # Significant difference
                        advanced_suggestions.append(
                            {
                                "item_id": item.id,
                                "item_name": item.name,
                                "current_reorder_point": item.reorder_point,
                                "optimal_reorder_point": optimal_reorder.optimal_reorder_point,
                                "current_reorder_quantity": item.reorder_quantity,
                                "optimal_reorder_quantity": optimal_reorder.optimal_reorder_quantity,
                                "service_level": optimal_reorder.service_level,
                                "cost_impact": optimal_reorder.total_cost,
                                "improvement_potential": current_gap,
                            }
                        )

            if not advanced_suggestions:
                return None

            # Sort by improvement potential
            advanced_suggestions.sort(key=lambda x: x["improvement_potential"], reverse=True)

            context = {
                "analysis_type": "advanced_reorder_optimization",
                "items_analyzed": len(items),
                "optimization_opportunities": len(advanced_suggestions),
                "top_suggestions": advanced_suggestions[:10],
            }

            reasoning = await self.analyze_with_claude(
                f"Advanced reorder analysis identified {len(advanced_suggestions)} optimization opportunities. "
                f"These recommendations use statistical demand forecasting and service level optimization. "
                f"Implementing these changes could improve inventory efficiency and reduce costs.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="advanced_reorder_optimization",
                context=context,
                reasoning=reasoning,
                action="Update reorder points and quantities based on statistical analysis",
                confidence=0.92,
            )

        except Exception as e:
            self.logger.error(f"Error in advanced reorder analysis: {e}")
            return None

    async def _perform_demand_forecast_analysis(self, session: Session) -> Optional[AgentDecision]:
        """Perform demand forecasting analysis for high-value items."""
        try:
            # Get high-value items for forecasting
            high_value_items = (
                session.query(Item)
                .filter(
                    and_(
                        Item.status == ItemStatus.ACTIVE,
                        Item.unit_cost * Item.current_stock > 500,  # Focus on high-value inventory
                    )
                )
                .limit(15)
                .all()
            )

            forecast_results = []

            for item in high_value_items:
                forecast = await self.predict_demand(session, item.id)
                if forecast and forecast.forecast_accuracy > self.min_forecast_accuracy:
                    forecast_results.append(
                        {
                            "item_id": item.id,
                            "item_name": item.name,
                            "predicted_demand": forecast.predicted_demand,
                            "confidence_interval": forecast.confidence_interval,
                            "forecast_accuracy": forecast.forecast_accuracy,
                            "seasonality_factor": forecast.seasonality_factor,
                            "trend_factor": forecast.trend_factor,
                            "current_stock": item.current_stock,
                            "stock_value": float(item.current_stock * item.unit_cost),
                        }
                    )

            if not forecast_results:
                return None

            # Calculate total forecasted demand and inventory insights
            total_predicted_demand = sum(f["predicted_demand"] for f in forecast_results)
            total_stock_value = sum(f["stock_value"] for f in forecast_results)
            avg_accuracy = sum(f["forecast_accuracy"] for f in forecast_results) / len(
                forecast_results
            )

            context = {
                "analysis_type": "demand_forecasting",
                "items_forecasted": len(forecast_results),
                "total_predicted_demand": total_predicted_demand,
                "total_stock_value": total_stock_value,
                "average_accuracy": avg_accuracy,
                "forecasts": forecast_results,
            }

            reasoning = await self.analyze_with_claude(
                f"Demand forecasting analysis completed for {len(forecast_results)} high-value items "
                f"with average accuracy of {avg_accuracy:.1%}. Total predicted demand: {total_predicted_demand:.1f} units. "
                f"Use these forecasts to optimize purchasing decisions and inventory levels.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="demand_forecast_analysis",
                context=context,
                reasoning=reasoning,
                action="Use demand forecasts to optimize inventory planning",
                confidence=avg_accuracy,
            )

        except Exception as e:
            self.logger.error(f"Error in demand forecast analysis: {e}")
            return None

    async def _perform_bulk_purchase_analysis(self, session: Session) -> Optional[AgentDecision]:
        """Analyze bulk purchase opportunities across all items."""
        try:
            opportunities = await self.analyze_bulk_purchase_opportunities(session)

            if not opportunities:
                return None

            total_savings = sum(float(opp.total_cost_savings) for opp in opportunities)

            # Categorize opportunities by savings potential
            high_impact = [opp for opp in opportunities if float(opp.total_cost_savings) > 1000]
            medium_impact = [
                opp for opp in opportunities if 500 <= float(opp.total_cost_savings) <= 1000
            ]

            context = {
                "analysis_type": "bulk_purchase_optimization",
                "total_opportunities": len(opportunities),
                "total_potential_savings": total_savings,
                "high_impact_opportunities": len(high_impact),
                "medium_impact_opportunities": len(medium_impact),
                "top_opportunities": [
                    {
                        "item_id": opp.item_id,
                        "optimal_quantity": opp.optimal_order_quantity,
                        "unit_cost_with_discount": float(opp.unit_cost_with_discount),
                        "total_savings": float(opp.total_cost_savings),
                        "break_even_days": opp.break_even_point,
                        "recommended_timing": opp.recommended_purchase_timing.isoformat(),
                    }
                    for opp in opportunities[:10]
                ],
            }

            reasoning = await self.analyze_with_claude(
                f"Bulk purchase analysis identified {len(opportunities)} cost optimization opportunities "
                f"with total potential savings of ${total_savings:,.2f}. "
                f"{len(high_impact)} opportunities offer savings over $1,000 each. "
                f"Consider implementing bulk purchases for high-impact items first.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="bulk_purchase_optimization",
                context=context,
                reasoning=reasoning,
                action="Implement bulk purchase strategies for high-savings opportunities",
                confidence=0.88,
            )

        except Exception as e:
            self.logger.error(f"Error in bulk purchase analysis: {e}")
            return None

    async def _perform_expiry_waste_analysis(self, session: Session) -> Optional[AgentDecision]:
        """Analyze expiry risks and waste minimization opportunities."""
        try:
            expiry_risks = await self.analyze_all_expiry_risks(session)

            if not expiry_risks:
                return None

            # Categorize risks
            high_risk = [risk for risk in expiry_risks if risk["waste_risk"] == "high"]
            medium_risk = [risk for risk in expiry_risks if risk["waste_risk"] == "medium"]

            total_waste_value = sum(risk["waste_value"] for risk in expiry_risks)

            # Collect all strategies
            all_strategies = []
            for risk in expiry_risks:
                for strategy in risk["strategies"]:
                    strategy["item_name"] = risk["item_name"]
                    all_strategies.append(strategy)

            context = {
                "analysis_type": "expiry_waste_minimization",
                "total_items_at_risk": len(expiry_risks),
                "high_risk_items": len(high_risk),
                "medium_risk_items": len(medium_risk),
                "total_potential_waste_value": total_waste_value,
                "expiry_risks": expiry_risks[:10],  # Top 10 by waste value
                "recommended_strategies": all_strategies[:15],  # Top 15 strategies
            }

            reasoning = await self.analyze_with_claude(
                f"Expiry waste analysis identified {len(expiry_risks)} items at risk "
                f"with potential waste value of ${total_waste_value:,.2f}. "
                f"{len(high_risk)} items have high waste risk requiring immediate action. "
                f"Implementing waste minimization strategies could save significant costs.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="expiry_waste_minimization",
                context=context,
                reasoning=reasoning,
                action="Implement waste minimization strategies for high-risk items",
                confidence=0.90,
            )

        except Exception as e:
            self.logger.error(f"Error in expiry waste analysis: {e}")
            return None

    async def _perform_advanced_supplier_analysis(
        self, session: Session
    ) -> Optional[AgentDecision]:
        """Perform comprehensive supplier performance analysis."""
        try:
            supplier_metrics = await self.analyze_supplier_performance_advanced(session)

            if not supplier_metrics:
                return None

            # Categorize suppliers by performance
            preferred = [
                s for s in supplier_metrics if s.recommended_action == "preferred_supplier"
            ]
            needs_improvement = [
                s for s in supplier_metrics if "improvement" in s.recommended_action
            ]
            consider_alternatives = [
                s for s in supplier_metrics if "alternative" in s.recommended_action
            ]

            avg_performance = sum(s.overall_performance_score for s in supplier_metrics) / len(
                supplier_metrics
            )

            context = {
                "analysis_type": "advanced_supplier_performance",
                "total_suppliers_analyzed": len(supplier_metrics),
                "preferred_suppliers": len(preferred),
                "suppliers_needing_improvement": len(needs_improvement),
                "suppliers_to_replace": len(consider_alternatives),
                "average_performance_score": avg_performance,
                "supplier_rankings": [
                    {
                        "supplier_id": s.supplier_id,
                        "overall_score": s.overall_performance_score,
                        "on_time_delivery": s.on_time_delivery_rate,
                        "quality_score": s.quality_score,
                        "cost_competitiveness": s.cost_competitiveness,
                        "reliability": s.reliability_index,
                        "recommended_action": s.recommended_action,
                    }
                    for s in supplier_metrics[:10]
                ],
            }

            reasoning = await self.analyze_with_claude(
                f"Advanced supplier analysis evaluated {len(supplier_metrics)} suppliers. "
                f"Average performance score: {avg_performance:.2f}. "
                f"{len(preferred)} suppliers are performing excellently, "
                f"{len(needs_improvement)} need improvement plans, "
                f"and {len(consider_alternatives)} should be replaced. "
                f"Focus on supplier development and diversification strategies.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="advanced_supplier_performance",
                context=context,
                reasoning=reasoning,
                action="Implement supplier improvement and diversification strategies",
                confidence=0.87,
            )

        except Exception as e:
            self.logger.error(f"Error in advanced supplier analysis: {e}")
            return None

    # Enhanced integration methods that combine multiple analytics

    async def generate_comprehensive_reorder_recommendation(
        self, session: Session, item_id: str
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive reorder recommendation using all available
        analytics.

        Combines demand forecasting, seasonality, correlations, and
        supplier analysis for optimal reorder decisions.
        """
        try:
            # Gather all analytics
            demand_forecast = await self.predict_demand(session, item_id)
            seasonality = await self.analyze_seasonality_patterns(session, item_id)
            correlations = await self.analyze_item_correlations(session, item_id)
            optimal_reorder = await self.calculate_optimal_reorder_point(session, item_id)
            bulk_optimization = await self.optimize_bulk_purchase(session, item_id)
            supplier_diversification = await self.analyze_supplier_diversification(session, item_id)

            if not demand_forecast:
                return None

            # Get item details
            item = session.query(Item).filter(Item.id == item_id).first()
            if not item:
                return None

            # Calculate enhanced recommendations
            base_demand = demand_forecast.predicted_demand

            # Apply seasonal adjustment
            seasonal_adjustment = 1.0
            if seasonality and seasonality.confidence > 0.7:
                seasonal_adjustment = seasonality.seasonal_adjustment_factor

            adjusted_demand = base_demand * seasonal_adjustment

            # Consider correlation impacts
            correlation_impact = 1.0
            if correlations and correlations.impact_factor > 0.3:
                # High-impact items might need buffer stock
                correlation_impact = 1.1

            final_demand_estimate = adjusted_demand * correlation_impact

            # Build comprehensive recommendation
            recommendation = {
                "item_id": item_id,
                "item_name": item.name,
                "current_stock": item.current_stock,
                "analytics_summary": {
                    "demand_forecast_available": demand_forecast is not None,
                    "seasonality_detected": seasonality is not None
                    and seasonality.confidence > 0.5,
                    "correlations_identified": correlations is not None
                    and len(correlations.correlated_items) > 0,
                    "optimal_reorder_calculated": optimal_reorder is not None,
                    "bulk_opportunities": bulk_optimization is not None,
                    "supplier_risk_assessed": supplier_diversification is not None,
                },
                "demand_analysis": {
                    "base_predicted_demand": base_demand,
                    "seasonal_adjustment_factor": seasonal_adjustment,
                    "correlation_impact_factor": correlation_impact,
                    "final_demand_estimate": final_demand_estimate,
                    "forecast_accuracy": demand_forecast.forecast_accuracy,
                    "confidence_interval": demand_forecast.confidence_interval,
                },
                "reorder_recommendations": {},
                "risk_factors": [],
                "opportunities": [],
            }

            # Add optimal reorder recommendations
            if optimal_reorder:
                recommendation["reorder_recommendations"] = {
                    "optimal_reorder_point": optimal_reorder.optimal_reorder_point,
                    "optimal_reorder_quantity": optimal_reorder.optimal_reorder_quantity,
                    "service_level": optimal_reorder.service_level,
                    "safety_stock": optimal_reorder.safety_stock,
                    "total_cost_estimate": optimal_reorder.total_cost,
                }

            # Add seasonality insights
            if seasonality:
                recommendation["seasonality_insights"] = {
                    "seasonal_strength": seasonality.seasonal_strength,
                    "current_period_type": (
                        "peak"
                        if seasonality.seasonal_adjustment_factor > 1.1
                        else "low" if seasonality.seasonal_adjustment_factor < 0.9 else "normal"
                    ),
                    "confidence": seasonality.confidence,
                }

            # Add correlation insights
            if correlations:
                recommendation["correlation_insights"] = {
                    "has_substitutes": len(correlations.substitution_items) > 0,
                    "substitute_items": correlations.substitution_items,
                    "complementary_items": correlations.complementary_items,
                    "bundle_opportunities": len(correlations.bundle_opportunities),
                }

            # Add risk factors
            if supplier_diversification and supplier_diversification.risk_score > 0.5:
                recommendation["risk_factors"].append(
                    {
                        "type": "supplier_concentration",
                        "severity": (
                            "high" if supplier_diversification.risk_score > 0.7 else "medium"
                        ),
                        "description": "High supplier concentration risk detected",
                        "concentration_index": supplier_diversification.current_supplier_concentration,
                    }
                )

            # Add opportunities
            if bulk_optimization and bulk_optimization.total_cost_savings > 100:
                recommendation["opportunities"].append(
                    {
                        "type": "bulk_purchase",
                        "potential_savings": float(bulk_optimization.total_cost_savings),
                        "optimal_quantity": bulk_optimization.optimal_order_quantity,
                        "break_even_days": bulk_optimization.break_even_point,
                    }
                )

            if correlations and correlations.bundle_opportunities:
                recommendation["opportunities"].append(
                    {
                        "type": "product_bundling",
                        "bundle_count": len(correlations.bundle_opportunities),
                        "estimated_margin_improvement": 0.05,
                    }
                )

            return recommendation

        except Exception as e:
            self.logger.error(f"Error generating comprehensive recommendation for {item_id}: {e}")
            return None

    async def _perform_comprehensive_analytics_analysis(
        self, session: Session
    ) -> Optional[AgentDecision]:
        """Perform comprehensive analytics analysis combining all advanced
        methods."""
        try:
            # Get high-value items for comprehensive analysis
            high_value_items = (
                session.query(Item)
                .filter(
                    and_(
                        Item.status == ItemStatus.ACTIVE,
                        Item.unit_cost * Item.current_stock > 1000,  # Focus on high-value inventory
                    )
                )
                .limit(10)
                .all()
            )

            comprehensive_results = []

            for item in high_value_items:
                recommendation = await self.generate_comprehensive_reorder_recommendation(
                    session, item.id
                )
                if recommendation:
                    comprehensive_results.append(recommendation)

            if not comprehensive_results:
                return None

            # Analyze overall insights
            total_value_analyzed = sum(
                r["current_stock"]
                * float(session.query(Item).filter(Item.id == r["item_id"]).first().unit_cost)
                for r in comprehensive_results
            )

            items_with_seasonality = sum(
                1 for r in comprehensive_results if r["analytics_summary"]["seasonality_detected"]
            )

            items_with_correlations = sum(
                1
                for r in comprehensive_results
                if r["analytics_summary"]["correlations_identified"]
            )

            total_opportunities = sum(len(r["opportunities"]) for r in comprehensive_results)

            context = {
                "analysis_type": "comprehensive_analytics",
                "items_analyzed": len(comprehensive_results),
                "total_inventory_value": total_value_analyzed,
                "items_with_seasonality": items_with_seasonality,
                "items_with_correlations": items_with_correlations,
                "total_opportunities": total_opportunities,
                "comprehensive_recommendations": comprehensive_results[:5],  # Top 5 for context
            }

            reasoning = await self.analyze_with_claude(
                f"Comprehensive analytics analysis completed for {len(comprehensive_results)} high-value items "
                f"worth ${total_value_analyzed:,.2f}. "
                f"Detected seasonality in {items_with_seasonality} items, correlations in {items_with_correlations} items. "
                f"Identified {total_opportunities} optimization opportunities. "
                f"Provide strategic inventory management recommendations.",
                context,
            )

            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="comprehensive_analytics",
                context=context,
                reasoning=reasoning,
                action="Implement comprehensive inventory optimization strategies",
                confidence=0.95,
            )

        except Exception as e:
            self.logger.error(f"Error in comprehensive analytics analysis: {e}")
            return None
