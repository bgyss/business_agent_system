import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
from agents.base_agent import BaseAgent, AgentDecision
from models.inventory import (
    Item, StockMovement, Supplier, PurchaseOrder, PurchaseOrderItem,
    ItemModel, InventorySummary, ReorderSuggestion,
    StockMovementType, ItemStatus
)


class InventoryAgent(BaseAgent):
    def __init__(self, agent_id: str, api_key: str, config: Dict[str, Any], db_url: str):
        super().__init__(agent_id, api_key, config)
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.low_stock_multiplier = config.get("low_stock_multiplier", 1.2)
        self.reorder_lead_time = config.get("reorder_lead_time", 7)
        self.consumption_analysis_days = config.get("consumption_analysis_days", 30)
    
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
        finally:
            session.close()
        
        return None
    
    async def _analyze_stock_movement(self, session, movement_data: Dict[str, Any]) -> Optional[AgentDecision]:
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
                "minimum_stock": item.minimum_stock
            },
            "movement": movement_data
        }
        
        # Check if this movement triggers a low stock alert
        if new_stock <= item.reorder_point and item.current_stock > item.reorder_point:
            reasoning = await self.analyze_with_claude(
                f"Item {item.name} (SKU: {item.sku}) has dropped to {new_stock} units, "
                f"which is at or below the reorder point of {item.reorder_point}. "
                f"Recent movement: {movement_type} of {quantity} units. "
                f"Should we trigger a reorder?",
                context
            )
            
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="low_stock_alert",
                context=context,
                reasoning=reasoning,
                action=f"Generate reorder recommendation for {item.name}",
                confidence=0.85
            )
        
        # Check for unusual consumption patterns
        if movement_type == StockMovementType.OUT and quantity > item.current_stock * 0.5:
            reasoning = await self.analyze_with_claude(
                f"Large stock movement detected for {item.name}: {quantity} units out "
                f"(represents {quantity/item.current_stock:.1%} of current stock). "
                f"Is this a normal usage pattern or should it be investigated?",
                context
            )
            
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="unusual_consumption",
                context=context,
                reasoning=reasoning,
                action=f"Investigate large consumption of {item.name}",
                confidence=0.7
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
            elif item.current_stock <= item.reorder_point:
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
                    "estimated_value": float(item.current_stock * item.unit_cost)
                }
                for item in low_stock_items[:10]  # Limit to top 10
            ],
            "out_of_stock_items": [
                {
                    "name": item.name,
                    "sku": item.sku,
                    "reorder_quantity": item.reorder_quantity,
                    "estimated_cost": float(item.reorder_quantity * item.unit_cost)
                }
                for item in out_of_stock_items[:10]  # Limit to top 10
            ]
        }
        
        analysis = await self.analyze_with_claude(
            f"Daily inventory check shows {len(low_stock_items)} items with low stock "
            f"and {len(out_of_stock_items)} items out of stock. "
            f"Provide prioritized action plan for restocking.",
            context
        )
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="daily_inventory_check",
            context=context,
            reasoning=analysis,
            action="Generate comprehensive reorder recommendations",
            confidence=0.9
        )
    
    async def _analyze_reorder_needs(self, session) -> Optional[AgentDecision]:
        # Get consumption data for the last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=self.consumption_analysis_days)
        
        # Get all items that need analysis
        items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()
        
        reorder_suggestions = []
        
        for item in items:
            # Calculate consumption rate
            consumption_movements = session.query(StockMovement).filter(
                and_(
                    StockMovement.item_id == item.id,
                    StockMovement.movement_type == StockMovementType.OUT,
                    StockMovement.movement_date >= thirty_days_ago
                )
            ).all()
            
            if not consumption_movements:
                continue
            
            total_consumed = sum(movement.quantity for movement in consumption_movements)
            daily_consumption = total_consumed / self.consumption_analysis_days
            
            # Calculate days of stock remaining
            days_remaining = item.current_stock / daily_consumption if daily_consumption > 0 else 999
            
            # Check if reorder is needed considering lead time
            if days_remaining <= self.reorder_lead_time * self.low_stock_multiplier:
                # Calculate suggested order quantity
                lead_time_consumption = daily_consumption * self.reorder_lead_time
                safety_stock = daily_consumption * 7  # 1 week safety stock
                suggested_quantity = max(
                    item.reorder_quantity,
                    int(lead_time_consumption + safety_stock - item.current_stock)
                )
                
                # Determine urgency
                if days_remaining <= self.reorder_lead_time * 0.5:
                    urgency = "critical"
                elif days_remaining <= self.reorder_lead_time * 0.8:
                    urgency = "high"
                else:
                    urgency = "medium"
                
                reorder_suggestions.append({
                    "item_id": item.id,
                    "item_name": item.name,
                    "sku": item.sku,
                    "current_stock": item.current_stock,
                    "daily_consumption": round(daily_consumption, 2),
                    "days_remaining": round(days_remaining, 1),
                    "suggested_quantity": suggested_quantity,
                    "estimated_cost": float(suggested_quantity * item.unit_cost),
                    "urgency": urgency
                })
        
        if not reorder_suggestions:
            return None
        
        # Sort by urgency and days remaining
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        reorder_suggestions.sort(key=lambda x: (urgency_order[x["urgency"]], x["days_remaining"]))
        
        context = {
            "reorder_count": len(reorder_suggestions),
            "total_estimated_cost": sum(item["estimated_cost"] for item in reorder_suggestions),
            "critical_items": [item for item in reorder_suggestions if item["urgency"] == "critical"],
            "high_priority_items": [item for item in reorder_suggestions if item["urgency"] == "high"],
            "suggestions": reorder_suggestions[:15]  # Limit to top 15
        }
        
        analysis = await self.analyze_with_claude(
            f"Reorder analysis identifies {len(reorder_suggestions)} items needing restock. "
            f"Total estimated cost: ${context['total_estimated_cost']:,.2f}. "
            f"Critical items: {len(context['critical_items'])}, "
            f"High priority: {len(context['high_priority_items'])}. "
            f"Provide prioritized purchasing recommendations considering cash flow.",
            context
        )
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="reorder_analysis",
            context=context,
            reasoning=analysis,
            action="Create purchase orders for critical and high-priority items",
            confidence=0.88
        )
    
    async def _check_expiring_items(self, session) -> Optional[AgentDecision]:
        # Find items with expiry dates
        items_with_expiry = session.query(Item).filter(
            and_(
                Item.expiry_days.isnot(None),
                Item.current_stock > 0,
                Item.status == ItemStatus.ACTIVE
            )
        ).all()
        
        if not items_with_expiry:
            return None
        
        expiring_soon = []
        today = datetime.now().date()
        
        # This is a simplified check - in a real system, you'd track batch expiry dates
        for item in items_with_expiry:
            # Assume items expire based on days since last received
            # In practice, you'd track actual expiry dates per batch
            if item.expiry_days <= 7:  # Items that expire within a week
                expiring_soon.append({
                    "name": item.name,
                    "sku": item.sku,
                    "current_stock": item.current_stock,
                    "expiry_days": item.expiry_days,
                    "estimated_loss": float(item.current_stock * item.unit_cost)
                })
        
        if not expiring_soon:
            return None
        
        context = {
            "expiring_items_count": len(expiring_soon),
            "total_potential_loss": sum(item["estimated_loss"] for item in expiring_soon),
            "expiring_items": expiring_soon
        }
        
        analysis = await self.analyze_with_claude(
            f"Expiry check shows {len(expiring_soon)} items expiring soon with "
            f"potential loss of ${context['total_potential_loss']:,.2f}. "
            f"Recommend actions to minimize waste.",
            context
        )
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="expiry_alert",
            context=context,
            reasoning=analysis,
            action="Implement waste reduction measures for expiring items",
            confidence=0.85
        )
    
    async def _analyze_supplier_performance(self, session) -> Optional[AgentDecision]:
        # Get recent purchase orders and their delivery performance
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        recent_orders = session.query(PurchaseOrder).filter(
            PurchaseOrder.order_date >= thirty_days_ago
        ).all()
        
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
                    "avg_delay": 0
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
                    poor_performers.append({
                        "supplier_id": supplier_id,
                        "name": perf["name"],
                        "on_time_rate": on_time_rate,
                        "order_count": perf["orders"],
                        "total_value": perf["total_value"]
                    })
        
        if not poor_performers:
            return None
        
        context = {
            "analysis_period_days": 30,
            "total_suppliers_analyzed": len([s for s in supplier_performance.values() if s["orders"] >= 2]),
            "poor_performers": poor_performers,
            "total_affected_value": sum(p["total_value"] for p in poor_performers)
        }
        
        analysis = await self.analyze_with_claude(
            f"Supplier performance analysis shows {len(poor_performers)} suppliers "
            f"with poor delivery performance affecting ${context['total_affected_value']:,.2f} "
            f"in orders. Recommend supplier management actions.",
            context
        )
        
        return AgentDecision(
            agent_id=self.agent_id,
            decision_type="supplier_performance",
            context=context,
            reasoning=analysis,
            action="Review and potentially change underperforming suppliers",
            confidence=0.75
        )
    
    async def generate_report(self) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            # Generate comprehensive inventory summary
            items = session.query(Item).filter(Item.status == ItemStatus.ACTIVE).all()
            
            total_value = sum(float(item.current_stock * item.unit_cost) for item in items)
            low_stock_items = [item for item in items if item.current_stock <= item.reorder_point]
            out_of_stock_items = [item for item in items if item.current_stock <= 0]
            
            # Get top moving items (simplified - based on recent movements)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_movements = session.query(StockMovement).filter(
                and_(
                    StockMovement.movement_type == StockMovementType.OUT,
                    StockMovement.movement_date >= thirty_days_ago
                )
            ).all()
            
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
                slow_moving_items=[]  # Would implement slow-moving analysis
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
        
        # Check for out of stock items
        out_of_stock = session.query(Item).filter(
            and_(
                Item.current_stock <= 0,
                Item.status == ItemStatus.ACTIVE
            )
        ).count()
        
        if out_of_stock > 0:
            alerts.append({
                "type": "out_of_stock",
                "severity": "high",
                "message": f"{out_of_stock} items are out of stock",
                "action_required": True
            })
        
        # Check for low stock items
        low_stock = session.query(Item).filter(
            and_(
                Item.current_stock <= Item.reorder_point,
                Item.current_stock > 0,
                Item.status == ItemStatus.ACTIVE
            )
        ).count()
        
        if low_stock > 0:
            alerts.append({
                "type": "low_stock",
                "severity": "medium",
                "message": f"{low_stock} items need reordering",
                "action_required": True
            })
        
        return alerts