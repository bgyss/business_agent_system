import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from models.inventory import ItemStatus, StockMovementType


@dataclass
class InventoryProfile:
    business_type: str
    items: List[Dict[str, Any]]
    consumption_patterns: Dict[str, float]  # day_of_week -> multiplier
    seasonal_factors: Dict[int, float]  # month -> multiplier
    waste_rate: float  # percentage of items that get wasted
    delivery_variance_days: int  # variance in delivery times


class InventorySimulator:
    def __init__(self, profile: InventoryProfile):
        self.profile = profile
        self.suppliers = self._generate_suppliers()

    def _generate_suppliers(self) -> List[Dict[str, Any]]:
        if self.profile.business_type == "restaurant":
            return [
                {
                    "name": "Fresh Foods Distributor",
                    "items": ["vegetables", "meat", "dairy", "bread"],
                    "lead_time": 2,
                    "reliability": 0.9,
                    "cost_multiplier": 1.0
                },
                {
                    "name": "Beverage Wholesale",
                    "items": ["beverages", "alcohol"],
                    "lead_time": 3,
                    "reliability": 0.95,
                    "cost_multiplier": 0.9
                },
                {
                    "name": "Dry Goods Supply",
                    "items": ["dry_goods", "spices", "cleaning_supplies"],
                    "lead_time": 5,
                    "reliability": 0.85,
                    "cost_multiplier": 0.8
                }
            ]
        else:  # retail
            return [
                {
                    "name": "Main Wholesale Supplier",
                    "items": ["electronics", "clothing", "accessories"],
                    "lead_time": 7,
                    "reliability": 0.88,
                    "cost_multiplier": 1.0
                },
                {
                    "name": "Local Distributor",
                    "items": ["home_goods", "books", "toys"],
                    "lead_time": 3,
                    "reliability": 0.92,
                    "cost_multiplier": 1.1
                },
                {
                    "name": "Direct Manufacturer",
                    "items": ["specialty_items"],
                    "lead_time": 14,
                    "reliability": 0.75,
                    "cost_multiplier": 0.7
                }
            ]

    def generate_initial_inventory(self) -> List[Dict[str, Any]]:
        items = []

        for item_template in self.profile.items:
            # Generate variations of base items
            for i in range(item_template.get("variations", 1)):
                base_name = item_template["name"]
                if item_template.get("variations", 1) > 1:
                    name = f"{base_name} - Variant {i+1}"
                    sku = f"{item_template['sku']}-{i+1:02d}"
                else:
                    name = base_name
                    sku = item_template["sku"]

                # Add some variance to costs and quantities
                cost_variance = random.uniform(0.8, 1.2)
                quantity_variance = random.uniform(0.7, 1.3)

                item = {
                    "sku": sku,
                    "name": name,
                    "description": item_template.get("description", f"Description for {name}"),
                    "category": item_template["category"],
                    "unit_cost": round(item_template["unit_cost"] * cost_variance, 2),
                    "selling_price": round(item_template.get("selling_price", item_template["unit_cost"] * 2.5) * cost_variance, 2),
                    "current_stock": int(item_template["initial_stock"] * quantity_variance),
                    "minimum_stock": item_template["minimum_stock"],
                    "maximum_stock": item_template["maximum_stock"],
                    "reorder_point": item_template["reorder_point"],
                    "reorder_quantity": item_template["reorder_quantity"],
                    "unit_of_measure": item_template.get("unit_of_measure", "each"),
                    "status": ItemStatus.ACTIVE,
                    "expiry_days": item_template.get("expiry_days")
                }
                items.append(item)

        return items

    def simulate_daily_consumption(self, items: List[Dict[str, Any]], date: datetime) -> List[Dict[str, Any]]:
        movements = []

        # Apply day-of-week and seasonal factors
        day_factor = self.profile.consumption_patterns.get(date.strftime('%A').lower(), 1.0)
        month_factor = self.profile.seasonal_factors.get(date.month, 1.0)
        combined_factor = day_factor * month_factor

        for item in items:
            if item["current_stock"] <= 0:
                continue

            # Calculate base consumption for this item
            category = item["category"]
            base_consumption = self._get_base_consumption(category, item["current_stock"])

            # Apply factors and randomness
            daily_consumption = max(0, int(base_consumption * combined_factor * random.uniform(0.5, 1.5)))

            # Don't consume more than available stock
            actual_consumption = min(daily_consumption, item["current_stock"])

            if actual_consumption > 0:
                movements.append({
                    "item_id": item.get("id"),  # Would be set after item creation
                    "item_sku": item["sku"],
                    "movement_type": StockMovementType.OUT,
                    "quantity": actual_consumption,
                    "unit_cost": item["unit_cost"],
                    "reference_number": f"CONSUMPTION-{date.strftime('%Y%m%d')}-{item['sku']}",
                    "notes": f"Daily consumption - {category}",
                    "movement_date": date + timedelta(
                        hours=random.randint(8, 20),
                        minutes=random.randint(0, 59)
                    )
                })

                # Update item stock for next calculations
                item["current_stock"] -= actual_consumption

        # Simulate waste (spoilage, breakage, etc.)
        waste_movements = self._simulate_waste(items, date)
        movements.extend(waste_movements)

        return movements

    def _get_base_consumption(self, category: str, current_stock: int) -> float:
        """Calculate base daily consumption based on category and stock levels"""
        consumption_rates = {
            # Restaurant categories
            "vegetables": 0.15,  # 15% of stock per day
            "meat": 0.12,
            "dairy": 0.18,
            "bread": 0.25,
            "beverages": 0.10,
            "alcohol": 0.05,
            "dry_goods": 0.03,
            "spices": 0.02,
            "cleaning_supplies": 0.05,

            # Retail categories
            "electronics": 0.02,
            "clothing": 0.04,
            "accessories": 0.03,
            "home_goods": 0.03,
            "books": 0.02,
            "toys": 0.05,
            "specialty_items": 0.01
        }

        rate = consumption_rates.get(category, 0.05)  # Default 5% per day
        return current_stock * rate

    def _simulate_waste(self, items: List[Dict[str, Any]], date: datetime) -> List[Dict[str, Any]]:
        waste_movements = []

        for item in items:
            if item["current_stock"] <= 0:
                continue

            # Higher waste rates for perishable items
            if item.get("expiry_days") and item["expiry_days"] <= 7:
                waste_chance = 0.02  # 2% chance per day for very perishable items
            elif item.get("expiry_days") and item["expiry_days"] <= 30:
                waste_chance = 0.005  # 0.5% chance for moderately perishable
            else:
                waste_chance = 0.001  # 0.1% chance for non-perishable items

            if random.random() < waste_chance:
                waste_quantity = max(1, int(item["current_stock"] * random.uniform(0.01, 0.05)))
                waste_quantity = min(waste_quantity, item["current_stock"])

                waste_movements.append({
                    "item_id": item.get("id"),
                    "item_sku": item["sku"],
                    "movement_type": StockMovementType.WASTE,
                    "quantity": waste_quantity,
                    "unit_cost": item["unit_cost"],
                    "reference_number": f"WASTE-{date.strftime('%Y%m%d')}-{item['sku']}",
                    "notes": f"Waste/spoilage - {item['category']}",
                    "movement_date": date + timedelta(
                        hours=random.randint(6, 22),
                        minutes=random.randint(0, 59)
                    )
                })

                item["current_stock"] -= waste_quantity

        return waste_movements

    def simulate_deliveries(self, items: List[Dict[str, Any]], date: datetime) -> List[Dict[str, Any]]:
        deliveries = []

        for item in items:
            # Check if item needs reordering
            if item["current_stock"] <= item["reorder_point"]:
                # Find appropriate supplier
                supplier = self._find_supplier_for_item(item)
                if not supplier:
                    continue

                # Check if delivery should arrive today (based on order lead time)
                # In a real system, you'd track actual orders and their expected delivery dates
                if random.random() < 0.3:  # 30% chance of delivery on any given day for low stock items
                    # Check if supplier is reliable
                    if random.random() < supplier["reliability"]:
                        delivery_quantity = item["reorder_quantity"]

                        # Add some variance to delivery quantity
                        variance = random.uniform(0.9, 1.1)
                        actual_quantity = int(delivery_quantity * variance)

                        # Calculate cost with supplier multiplier
                        cost_per_unit = item["unit_cost"] * supplier["cost_multiplier"]

                        deliveries.append({
                            "item_id": item.get("id"),
                            "item_sku": item["sku"],
                            "movement_type": StockMovementType.IN,
                            "quantity": actual_quantity,
                            "unit_cost": round(cost_per_unit, 2),
                            "reference_number": f"DELIVERY-{date.strftime('%Y%m%d')}-{supplier['name'][:3].upper()}-{item['sku']}",
                            "notes": f"Delivery from {supplier['name']}",
                            "movement_date": date + timedelta(
                                hours=random.randint(8, 16),
                                minutes=random.randint(0, 59)
                            )
                        })

                        # Update item stock
                        item["current_stock"] += actual_quantity

        return deliveries

    def _find_supplier_for_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best supplier for a given item"""
        category = item["category"]

        suitable_suppliers = [
            supplier for supplier in self.suppliers
            if category in supplier["items"] or "all" in supplier["items"]
        ]

        if not suitable_suppliers:
            return None

        # Choose supplier based on reliability and cost
        return max(suitable_suppliers, key=lambda s: s["reliability"] * (2 - s["cost_multiplier"]))

    def generate_purchase_orders(self, items: List[Dict[str, Any]], date: datetime) -> List[Dict[str, Any]]:
        purchase_orders = []

        # Check for items that need reordering
        items_to_order = [item for item in items if item["current_stock"] <= item["reorder_point"]]

        if not items_to_order:
            return purchase_orders

        # Group items by supplier
        supplier_orders = {}

        for item in items_to_order:
            supplier = self._find_supplier_for_item(item)
            if not supplier:
                continue

            supplier_name = supplier["name"]
            if supplier_name not in supplier_orders:
                supplier_orders[supplier_name] = {
                    "supplier": supplier,
                    "items": []
                }

            supplier_orders[supplier_name]["items"].append(item)

        # Create purchase orders
        for supplier_name, order_data in supplier_orders.items():
            supplier = order_data["supplier"]
            items_list = order_data["items"]

            # Calculate total amount
            total_amount = 0
            po_items = []

            for item in items_list:
                quantity = item["reorder_quantity"]
                unit_cost = item["unit_cost"] * supplier["cost_multiplier"]
                line_total = quantity * unit_cost
                total_amount += line_total

                po_items.append({
                    "item_id": item.get("id"),
                    "item_sku": item["sku"],
                    "quantity_ordered": quantity,
                    "unit_cost": round(unit_cost, 2),
                    "total_cost": round(line_total, 2)
                })

            # Create the purchase order
            po_number = f"PO-{date.strftime('%Y%m%d')}-{supplier_name[:3].upper()}-{random.randint(1000, 9999)}"
            expected_delivery = date + timedelta(days=supplier["lead_time"])

            purchase_order = {
                "po_number": po_number,
                "supplier_name": supplier_name,
                "status": "pending",
                "order_date": date,
                "expected_delivery_date": expected_delivery,
                "total_amount": round(total_amount, 2),
                "notes": f"Auto-generated reorder for {len(items_list)} items",
                "items": po_items
            }

            purchase_orders.append(purchase_order)

        return purchase_orders

    def simulate_stock_adjustments(self, items: List[Dict[str, Any]], date: datetime) -> List[Dict[str, Any]]:
        adjustments = []

        # Occasionally perform stock adjustments (cycle counts, corrections, etc.)
        if random.random() < 0.05:  # 5% chance per day
            # Select random items for adjustment
            num_adjustments = random.randint(1, min(5, len(items)))
            selected_items = random.sample(items, num_adjustments)

            for item in selected_items:
                # Simulate finding discrepancies
                if random.random() < 0.3:  # 30% chance of finding a discrepancy
                    # Small random adjustment (usually small discrepancies)
                    adjustment = random.randint(-5, 5)
                    if adjustment != 0:
                        adjustments.append({
                            "item_id": item.get("id"),
                            "item_sku": item["sku"],
                            "movement_type": StockMovementType.ADJUSTMENT,
                            "quantity": adjustment,
                            "unit_cost": item["unit_cost"],
                            "reference_number": f"ADJ-{date.strftime('%Y%m%d')}-{item['sku']}",
                            "notes": "Inventory adjustment - cycle count correction",
                            "movement_date": date + timedelta(
                                hours=random.randint(9, 17),
                                minutes=random.randint(0, 59)
                            )
                        })

                        item["current_stock"] += adjustment

        return adjustments


def get_restaurant_inventory_profile() -> InventoryProfile:
    return InventoryProfile(
        business_type="restaurant",
        items=[
            {
                "name": "Fresh Lettuce", "sku": "VEG001", "category": "vegetables",
                "unit_cost": 2.50, "selling_price": 0.0,  # Ingredient, not sold directly
                "initial_stock": 50, "minimum_stock": 5, "maximum_stock": 100,
                "reorder_point": 10, "reorder_quantity": 40, "expiry_days": 5,
                "variations": 1
            },
            {
                "name": "Ground Beef", "sku": "MEAT001", "category": "meat",
                "unit_cost": 8.00, "selling_price": 0.0,
                "initial_stock": 30, "minimum_stock": 3, "maximum_stock": 60,
                "reorder_point": 8, "reorder_quantity": 25, "expiry_days": 3,
                "variations": 1
            },
            {
                "name": "Milk", "sku": "DAIRY001", "category": "dairy",
                "unit_cost": 3.50, "selling_price": 0.0,
                "initial_stock": 20, "minimum_stock": 2, "maximum_stock": 40,
                "reorder_point": 5, "reorder_quantity": 15, "expiry_days": 7,
                "variations": 1
            },
            {
                "name": "Soft Drinks", "sku": "BEV001", "category": "beverages",
                "unit_cost": 1.25, "selling_price": 3.50,
                "initial_stock": 100, "minimum_stock": 10, "maximum_stock": 200,
                "reorder_point": 25, "reorder_quantity": 75, "expiry_days": 180,
                "variations": 3
            },
            {
                "name": "Flour", "sku": "DRY001", "category": "dry_goods",
                "unit_cost": 4.00, "selling_price": 0.0,
                "initial_stock": 15, "minimum_stock": 2, "maximum_stock": 30,
                "reorder_point": 5, "reorder_quantity": 12, "expiry_days": 365,
                "variations": 1
            }
        ],
        consumption_patterns={
            "monday": 0.7, "tuesday": 0.8, "wednesday": 0.9,
            "thursday": 1.0, "friday": 1.3, "saturday": 1.4, "sunday": 1.1
        },
        seasonal_factors={
            1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.3, 8: 1.2, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.4
        },
        waste_rate=0.05,  # 5% waste rate
        delivery_variance_days=2
    )


def get_retail_inventory_profile() -> InventoryProfile:
    return InventoryProfile(
        business_type="retail",
        items=[
            {
                "name": "Wireless Headphones", "sku": "ELEC001", "category": "electronics",
                "unit_cost": 45.00, "selling_price": 89.99,
                "initial_stock": 25, "minimum_stock": 3, "maximum_stock": 50,
                "reorder_point": 8, "reorder_quantity": 20, "expiry_days": None,
                "variations": 2
            },
            {
                "name": "T-Shirt", "sku": "CLOTH001", "category": "clothing",
                "unit_cost": 12.00, "selling_price": 29.99,
                "initial_stock": 40, "minimum_stock": 5, "maximum_stock": 80,
                "reorder_point": 12, "reorder_quantity": 30, "expiry_days": None,
                "variations": 4
            },
            {
                "name": "Coffee Mug", "sku": "HOME001", "category": "home_goods",
                "unit_cost": 8.50, "selling_price": 19.99,
                "initial_stock": 30, "minimum_stock": 4, "maximum_stock": 60,
                "reorder_point": 10, "reorder_quantity": 25, "expiry_days": None,
                "variations": 3
            },
            {
                "name": "Popular Novel", "sku": "BOOK001", "category": "books",
                "unit_cost": 6.00, "selling_price": 14.99,
                "initial_stock": 20, "minimum_stock": 2, "maximum_stock": 40,
                "reorder_point": 6, "reorder_quantity": 15, "expiry_days": None,
                "variations": 1
            }
        ],
        consumption_patterns={
            "monday": 0.8, "tuesday": 0.9, "wednesday": 0.9,
            "thursday": 1.0, "friday": 1.2, "saturday": 1.4, "sunday": 0.6
        },
        seasonal_factors={
            1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.0, 6: 0.9,
            7: 0.8, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.6
        },
        waste_rate=0.01,  # 1% waste rate (damage, theft, etc.)
        delivery_variance_days=3
    )
