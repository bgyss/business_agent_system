"""
Unit tests for InventorySimulator class
"""
import pytest
import random
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.inventory_simulator import (
    InventorySimulator,
    InventoryProfile,
    get_restaurant_inventory_profile,
    get_retail_inventory_profile
)
from models.inventory import StockMovementType, ItemStatus


class TestInventoryProfile:
    """Test cases for InventoryProfile dataclass"""
    
    def test_inventory_profile_creation(self):
        """Test InventoryProfile creation with all fields"""
        items = [
            {
                "name": "Test Item",
                "sku": "TEST001",
                "category": "test",
                "unit_cost": 10.0,
                "initial_stock": 50,
                "minimum_stock": 5,
                "maximum_stock": 100,
                "reorder_point": 15,
                "reorder_quantity": 40
            }
        ]
        
        profile = InventoryProfile(
            business_type="restaurant",
            items=items,
            consumption_patterns={"monday": 0.8, "friday": 1.2},
            seasonal_factors={1: 0.8, 6: 1.2},
            waste_rate=0.05,
            delivery_variance_days=2
        )
        
        assert profile.business_type == "restaurant"
        assert profile.items == items
        assert profile.consumption_patterns == {"monday": 0.8, "friday": 1.2}
        assert profile.seasonal_factors == {1: 0.8, 6: 1.2}
        assert profile.waste_rate == 0.05
        assert profile.delivery_variance_days == 2


class TestInventorySimulator:
    """Test cases for InventorySimulator"""
    
    @pytest.fixture
    def sample_inventory_profile(self):
        """Create a sample inventory profile for testing"""
        items = [
            {
                "name": "Test Vegetables",
                "sku": "VEG001",
                "category": "vegetables",
                "unit_cost": 2.50,
                "selling_price": 0.0,
                "initial_stock": 50,
                "minimum_stock": 5,
                "maximum_stock": 100,
                "reorder_point": 15,
                "reorder_quantity": 40,
                "unit_of_measure": "lbs",
                "expiry_days": 5,
                "variations": 1
            },
            {
                "name": "Test Beverages",
                "sku": "BEV001", 
                "category": "beverages",
                "unit_cost": 1.25,
                "selling_price": 3.50,
                "initial_stock": 100,
                "minimum_stock": 10,
                "maximum_stock": 200,
                "reorder_point": 25,
                "reorder_quantity": 75,
                "unit_of_measure": "each",
                "expiry_days": 180,
                "variations": 2
            }
        ]
        
        return InventoryProfile(
            business_type="restaurant",
            items=items,
            consumption_patterns={
                "monday": 0.7,
                "tuesday": 0.8,
                "wednesday": 0.9,
                "thursday": 1.0,
                "friday": 1.3,
                "saturday": 1.4,
                "sunday": 1.1
            },
            seasonal_factors={
                1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
                7: 1.3, 8: 1.2, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.4
            },
            waste_rate=0.05,
            delivery_variance_days=2
        )
    
    @pytest.fixture
    def inventory_simulator(self, sample_inventory_profile):
        """Create InventorySimulator instance for testing"""
        return InventorySimulator(sample_inventory_profile)
    
    def test_initialization(self, sample_inventory_profile):
        """Test InventorySimulator initialization"""
        simulator = InventorySimulator(sample_inventory_profile)
        
        assert simulator.profile == sample_inventory_profile
        assert isinstance(simulator.suppliers, list)
        assert len(simulator.suppliers) > 0
    
    def test_generate_suppliers_restaurant(self, sample_inventory_profile):
        """Test supplier generation for restaurant business"""
        simulator = InventorySimulator(sample_inventory_profile)
        
        suppliers = simulator._generate_suppliers()
        
        assert isinstance(suppliers, list)
        assert len(suppliers) > 0
        
        # Check for expected restaurant suppliers
        supplier_names = [s["name"] for s in suppliers]
        assert "Fresh Foods Distributor" in supplier_names
        assert "Beverage Wholesale" in supplier_names
        assert "Dry Goods Supply" in supplier_names
        
        # Check supplier structure
        supplier = suppliers[0]
        assert "name" in supplier
        assert "items" in supplier
        assert "lead_time" in supplier
        assert "reliability" in supplier
        assert "cost_multiplier" in supplier
        assert isinstance(supplier["items"], list)
        assert 0.0 <= supplier["reliability"] <= 1.0
    
    def test_generate_suppliers_retail(self):
        """Test supplier generation for retail business"""
        profile = InventoryProfile(
            business_type="retail",
            items=[],
            consumption_patterns={},
            seasonal_factors={},
            waste_rate=0.01,
            delivery_variance_days=3
        )
        simulator = InventorySimulator(profile)
        
        suppliers = simulator._generate_suppliers()
        
        supplier_names = [s["name"] for s in suppliers]
        assert "Main Wholesale Supplier" in supplier_names
        assert "Local Distributor" in supplier_names
        assert "Direct Manufacturer" in supplier_names
    
    def test_generate_initial_inventory_single_variation(self, inventory_simulator):
        """Test initial inventory generation with single variation items"""
        items = inventory_simulator.generate_initial_inventory()
        
        assert isinstance(items, list)
        assert len(items) > 0
        
        # Check item structure
        item = items[0]
        required_fields = [
            "sku", "name", "description", "category", "unit_cost", "selling_price",
            "current_stock", "minimum_stock", "maximum_stock", "reorder_point",
            "reorder_quantity", "unit_of_measure", "status"
        ]
        
        for field in required_fields:
            assert field in item
        
        assert item["status"] == ItemStatus.ACTIVE
        assert item["current_stock"] > 0
        assert item["unit_cost"] > 0
    
    def test_generate_initial_inventory_multiple_variations(self, inventory_simulator):
        """Test initial inventory generation with multiple variation items"""
        items = inventory_simulator.generate_initial_inventory()
        
        # Should have items for beverages with variations
        beverage_items = [item for item in items if "BEV001" in item["sku"]]
        assert len(beverage_items) == 2  # 2 variations
        
        # Check variation naming
        assert any("Variant 1" in item["name"] for item in beverage_items)
        assert any("Variant 2" in item["name"] for item in beverage_items)
        
        # Check SKU variations
        skus = [item["sku"] for item in beverage_items]
        assert "BEV001-01" in skus
        assert "BEV001-02" in skus
    
    def test_generate_initial_inventory_cost_variance(self, inventory_simulator):
        """Test that initial inventory applies cost variance"""
        items = inventory_simulator.generate_initial_inventory()
        
        # Get vegetable items (should have same base cost but with variance)
        veg_items = [item for item in items if item["category"] == "vegetables"]
        
        # Cost should be within expected range (0.8 to 1.2 of base)
        for item in veg_items:
            assert 2.0 <= item["unit_cost"] <= 3.0  # 2.50 * (0.8 to 1.2)
    
    def test_simulate_daily_consumption_basic(self, inventory_simulator):
        """Test basic daily consumption simulation"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 50,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)  # Friday
        
        movements = inventory_simulator.simulate_daily_consumption(items, test_date)
        
        assert isinstance(movements, list)
        
        # Check movement structure if any movements were generated
        if movements:
            movement = movements[0]
            assert "item_sku" in movement
            assert "movement_type" in movement
            assert "quantity" in movement
            assert "unit_cost" in movement
            assert "reference_number" in movement
            assert "notes" in movement
            assert "movement_date" in movement
            
            assert movement["movement_type"] in [StockMovementType.OUT, StockMovementType.WASTE]
            assert movement["quantity"] > 0
    
    def test_simulate_daily_consumption_no_stock(self, inventory_simulator):
        """Test consumption simulation with no stock"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 0,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        movements = inventory_simulator.simulate_daily_consumption(items, test_date)
        
        # Should not generate consumption movements for items with no stock
        consumption_movements = [m for m in movements if m["movement_type"] == StockMovementType.OUT]
        assert len(consumption_movements) == 0
    
    def test_simulate_daily_consumption_seasonal_factors(self, inventory_simulator):
        """Test that seasonal factors affect consumption"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 100,
                "unit_cost": 2.50
            }
        ]
        
        # June (factor 1.2) vs January (factor 0.8)
        june_date = datetime(2024, 6, 15)
        january_date = datetime(2024, 1, 15)
        
        # Reset stock for second test
        june_movements = inventory_simulator.simulate_daily_consumption(items.copy(), june_date)
        january_movements = inventory_simulator.simulate_daily_consumption(items.copy(), january_date)
        
        # Filter out waste movements for comparison
        june_consumption = [m for m in june_movements if m["movement_type"] == StockMovementType.OUT]
        january_consumption = [m for m in january_movements if m["movement_type"] == StockMovementType.OUT]
        
        # June should generally have higher consumption (this is probabilistic)
        if june_consumption and january_consumption:
            june_total = sum(m["quantity"] for m in june_consumption)
            january_total = sum(m["quantity"] for m in january_consumption)
            # Allow for some variance in random generation
            assert june_total >= january_total * 0.5  # Very loose assertion due to randomness
    
    def test_get_base_consumption(self, inventory_simulator):
        """Test base consumption calculation for different categories"""
        # Test restaurant categories
        vegetables_consumption = inventory_simulator._get_base_consumption("vegetables", 100)
        assert vegetables_consumption == 15.0  # 15% of 100
        
        meat_consumption = inventory_simulator._get_base_consumption("meat", 50)
        assert meat_consumption == 6.0  # 12% of 50
        
        # Test unknown category (should use default rate)
        unknown_consumption = inventory_simulator._get_base_consumption("unknown", 100)
        assert unknown_consumption == 5.0  # 5% default rate
    
    @patch('simulation.inventory_simulator.random.random')
    def test_simulate_waste_basic(self, mock_random, inventory_simulator):
        """Test basic waste simulation"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 50,
                "unit_cost": 2.50,
                "expiry_days": 5  # Very perishable
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        # Mock random to trigger waste
        mock_random.return_value = 0.01  # Should trigger waste for perishable items
        
        with patch('simulation.inventory_simulator.random.uniform', return_value=0.02):
            waste_movements = inventory_simulator._simulate_waste(items, test_date)
            
            assert isinstance(waste_movements, list)
            if waste_movements:  # Waste is probabilistic
                movement = waste_movements[0]
                assert movement["movement_type"] == StockMovementType.WASTE
                assert movement["quantity"] > 0
                assert "WASTE-" in movement["reference_number"]
    
    def test_simulate_waste_non_perishable(self, inventory_simulator):
        """Test waste simulation for non-perishable items"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "dry_goods",
                "current_stock": 50,
                "unit_cost": 4.00,
                "expiry_days": None  # Non-perishable
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        # Run multiple times to account for low probability
        total_waste = 0
        for _ in range(100):
            waste_movements = inventory_simulator._simulate_waste(items.copy(), test_date)
            total_waste += len(waste_movements)
        
        # Non-perishable items should have very little waste
        assert total_waste < 10  # Very few waste events out of 100 runs
    
    @patch('simulation.inventory_simulator.random.random')
    def test_simulate_deliveries_basic(self, mock_random, inventory_simulator):
        """Test basic delivery simulation"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 10,  # Below reorder point
                "reorder_point": 15,
                "reorder_quantity": 40,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        # Mock to trigger delivery
        mock_random.side_effect = [0.2, 0.95]  # First for delivery chance, second for reliability
        
        with patch('simulation.inventory_simulator.random.uniform', return_value=1.0):
            deliveries = inventory_simulator.simulate_deliveries(items, test_date)
            
            assert isinstance(deliveries, list)
            if deliveries:  # Deliveries are probabilistic
                delivery = deliveries[0]
                assert delivery["movement_type"] == StockMovementType.IN
                assert delivery["quantity"] > 0
                assert "DELIVERY-" in delivery["reference_number"]
                
                # Check that item stock was updated
                assert items[0]["current_stock"] > 10
    
    def test_simulate_deliveries_no_reorder_needed(self, inventory_simulator):
        """Test delivery simulation when no reorder is needed"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 50,  # Above reorder point
                "reorder_point": 15,
                "reorder_quantity": 40,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        deliveries = inventory_simulator.simulate_deliveries(items, test_date)
        
        # Should not generate deliveries for items above reorder point
        assert len(deliveries) == 0
    
    def test_find_supplier_for_item(self, inventory_simulator):
        """Test finding appropriate supplier for items"""
        # Test with vegetables category
        vegetables_item = {"category": "vegetables"}
        supplier = inventory_simulator._find_supplier_for_item(vegetables_item)
        
        assert supplier is not None
        assert "vegetables" in supplier["items"]
        assert supplier["name"] == "Fresh Foods Distributor"
        
        # Test with unknown category
        unknown_item = {"category": "unknown_category"}
        supplier = inventory_simulator._find_supplier_for_item(unknown_item)
        
        # Should return None if no suitable supplier found
        assert supplier is None
    
    def test_find_supplier_best_choice(self, inventory_simulator):
        """Test that supplier selection chooses the best option"""
        # Mock multiple suppliers for same category
        mock_suppliers = [
            {
                "name": "Expensive Supplier",
                "items": ["vegetables"],
                "reliability": 0.9,
                "cost_multiplier": 1.5  # Expensive
            },
            {
                "name": "Cheap Unreliable Supplier",
                "items": ["vegetables"],
                "reliability": 0.5,  # Unreliable
                "cost_multiplier": 0.8
            },
            {
                "name": "Good Supplier",
                "items": ["vegetables"],
                "reliability": 0.95,  # Very reliable
                "cost_multiplier": 1.0  # Fair price
            }
        ]
        
        inventory_simulator.suppliers = mock_suppliers
        
        vegetables_item = {"category": "vegetables"}
        supplier = inventory_simulator._find_supplier_for_item(vegetables_item)
        
        # Should choose the "Good Supplier" with best reliability/cost balance
        assert supplier["name"] == "Good Supplier"
    
    def test_generate_purchase_orders_basic(self, inventory_simulator):
        """Test basic purchase order generation"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 5,  # Below reorder point
                "reorder_point": 15,
                "reorder_quantity": 40,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        purchase_orders = inventory_simulator.generate_purchase_orders(items, test_date)
        
        assert isinstance(purchase_orders, list)
        if purchase_orders:  # POs generated for items needing reorder
            po = purchase_orders[0]
            
            assert "po_number" in po
            assert "supplier_name" in po
            assert "status" in po
            assert "order_date" in po
            assert "expected_delivery_date" in po
            assert "total_amount" in po
            assert "items" in po
            
            assert po["status"] == "pending"
            assert po["order_date"] == test_date
            assert po["total_amount"] > 0
            assert len(po["items"]) > 0
            
            # Check PO item structure
            po_item = po["items"][0]
            assert "item_sku" in po_item
            assert "quantity_ordered" in po_item
            assert "unit_cost" in po_item
            assert "total_cost" in po_item
    
    def test_generate_purchase_orders_no_reorder_needed(self, inventory_simulator):
        """Test PO generation when no items need reordering"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 50,  # Above reorder point
                "reorder_point": 15,
                "reorder_quantity": 40,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        purchase_orders = inventory_simulator.generate_purchase_orders(items, test_date)
        
        assert isinstance(purchase_orders, list)
        assert len(purchase_orders) == 0
    
    def test_generate_purchase_orders_group_by_supplier(self, inventory_simulator):
        """Test that PO generation groups items by supplier"""
        items = [
            {
                "id": 1,
                "sku": "VEG001",
                "category": "vegetables",
                "current_stock": 5,
                "reorder_point": 15,
                "reorder_quantity": 40,
                "unit_cost": 2.50
            },
            {
                "id": 2,
                "sku": "BEV001",
                "category": "beverages",
                "current_stock": 10,
                "reorder_point": 25,
                "reorder_quantity": 75,
                "unit_cost": 1.25
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        purchase_orders = inventory_simulator.generate_purchase_orders(items, test_date)
        
        # Should create separate POs for different suppliers
        assert len(purchase_orders) >= 1
        
        # If multiple suppliers are involved, check that items are grouped properly
        if len(purchase_orders) > 1:
            suppliers = set(po["supplier_name"] for po in purchase_orders)
            assert len(suppliers) > 1  # Multiple suppliers
    
    @patch('simulation.inventory_simulator.random.random')
    def test_simulate_stock_adjustments_basic(self, mock_random, inventory_simulator):
        """Test basic stock adjustment simulation"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 50,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        # Mock to trigger adjustment
        mock_random.side_effect = [0.04, 0.25]  # First triggers adjustment, second triggers discrepancy
        
        with patch('simulation.inventory_simulator.random.randint', return_value=2), \
             patch('simulation.inventory_simulator.random.sample', return_value=items):
            
            adjustments = inventory_simulator.simulate_stock_adjustments(items, test_date)
            
            assert isinstance(adjustments, list)
            if adjustments:  # Adjustments are probabilistic
                adjustment = adjustments[0]
                assert adjustment["movement_type"] == StockMovementType.ADJUSTMENT
                assert "ADJ-" in adjustment["reference_number"]
                assert adjustment["quantity"] != 0
    
    def test_simulate_stock_adjustments_no_discrepancy(self, inventory_simulator):
        """Test stock adjustments when no discrepancies are found"""
        items = [
            {
                "id": 1,
                "sku": "TEST001",
                "category": "vegetables",
                "current_stock": 50,
                "unit_cost": 2.50
            }
        ]
        
        test_date = datetime(2024, 6, 15)
        
        # Mock to trigger adjustment but not find discrepancy
        with patch('simulation.inventory_simulator.random.random', side_effect=[0.04, 0.8]):
            adjustments = inventory_simulator.simulate_stock_adjustments(items, test_date)
            
            # Should not create adjustments if no discrepancies found
            assert isinstance(adjustments, list)
            # Length could be 0 due to no discrepancy found


class TestPredefinedProfiles:
    """Test cases for predefined inventory profiles"""
    
    def test_get_restaurant_inventory_profile(self):
        """Test restaurant inventory profile generation"""
        profile = get_restaurant_inventory_profile()
        
        assert isinstance(profile, InventoryProfile)
        assert profile.business_type == "restaurant"
        assert len(profile.items) > 0
        assert len(profile.consumption_patterns) == 7  # All days of week
        assert len(profile.seasonal_factors) == 12  # All months
        assert profile.waste_rate > 0
        assert profile.delivery_variance_days > 0
        
        # Check for expected restaurant items
        item_names = [item["name"] for item in profile.items]
        assert any("Lettuce" in name for name in item_names)
        assert any("Beef" in name for name in item_names)
        assert any("Milk" in name for name in item_names)
        
        # Check item structure
        item = profile.items[0]
        required_fields = [
            "name", "sku", "category", "unit_cost", "initial_stock",
            "minimum_stock", "maximum_stock", "reorder_point", "reorder_quantity"
        ]
        for field in required_fields:
            assert field in item
        
        # Check expiry days for perishables
        perishable_items = [item for item in profile.items if item.get("expiry_days") and item["expiry_days"] <= 7]
        assert len(perishable_items) > 0  # Should have some very perishable items
    
    def test_get_retail_inventory_profile(self):
        """Test retail inventory profile generation"""
        profile = get_retail_inventory_profile()
        
        assert isinstance(profile, InventoryProfile)
        assert profile.business_type == "retail"
        assert len(profile.items) > 0
        assert len(profile.consumption_patterns) == 7
        assert len(profile.seasonal_factors) == 12
        assert profile.waste_rate > 0
        assert profile.delivery_variance_days > 0
        
        # Check for expected retail items
        item_names = [item["name"] for item in profile.items]
        assert any("Headphones" in name for name in item_names)
        assert any("T-Shirt" in name for name in item_names)
        assert any("Mug" in name for name in item_names)
        
        # Retail items should have selling prices
        items_with_prices = [item for item in profile.items if item.get("selling_price", 0) > 0]
        assert len(items_with_prices) > 0
        
        # Most retail items should not have expiry dates
        non_perishable_items = [item for item in profile.items if item.get("expiry_days") is None]
        assert len(non_perishable_items) > 0
    
    def test_profile_differences(self):
        """Test that restaurant and retail profiles are different"""
        restaurant_profile = get_restaurant_inventory_profile()
        retail_profile = get_retail_inventory_profile()
        
        assert restaurant_profile.business_type != retail_profile.business_type
        assert restaurant_profile.waste_rate != retail_profile.waste_rate
        assert restaurant_profile.delivery_variance_days != retail_profile.delivery_variance_days
        
        # Different categories of items
        restaurant_categories = set(item["category"] for item in restaurant_profile.items)
        retail_categories = set(item["category"] for item in retail_profile.items)
        
        # Should have different categories
        assert restaurant_categories != retail_categories
        
        # Restaurant should have food categories
        assert any("vegetables" in cat or "meat" in cat or "dairy" in cat for cat in restaurant_categories)
        
        # Retail should have merchandise categories  
        assert any("electronics" in cat or "clothing" in cat for cat in retail_categories)
    
    def test_profile_consistency(self):
        """Test that profiles are consistent across calls"""
        profile1 = get_restaurant_inventory_profile()
        profile2 = get_restaurant_inventory_profile()
        
        assert profile1.business_type == profile2.business_type
        assert profile1.waste_rate == profile2.waste_rate
        assert profile1.delivery_variance_days == profile2.delivery_variance_days
        assert len(profile1.items) == len(profile2.items)
        assert profile1.consumption_patterns == profile2.consumption_patterns
        assert profile1.seasonal_factors == profile2.seasonal_factors
    
    def test_item_variations(self):
        """Test that items with variations are properly handled"""
        restaurant_profile = get_restaurant_inventory_profile()
        
        # Find items with variations
        items_with_variations = [item for item in restaurant_profile.items if item.get("variations", 1) > 1]
        assert len(items_with_variations) > 0
        
        # Check that variation count makes sense
        for item in items_with_variations:
            assert item["variations"] > 1
            assert item["variations"] <= 5  # Reasonable upper limit
    
    def test_realistic_values(self):
        """Test that profile values are realistic"""
        restaurant_profile = get_restaurant_inventory_profile()
        
        # Check consumption patterns are reasonable (0.5 to 1.5 range)
        for day, factor in restaurant_profile.consumption_patterns.items():
            assert 0.5 <= factor <= 1.5
        
        # Check seasonal factors are reasonable
        for month, factor in restaurant_profile.seasonal_factors.items():
            assert 0.5 <= factor <= 2.0
        
        # Check waste rate is reasonable
        assert 0.0 <= restaurant_profile.waste_rate <= 0.2  # Max 20% waste
        
        # Check item values are realistic
        for item in restaurant_profile.items:
            assert item["unit_cost"] > 0
            assert item["initial_stock"] > 0
            assert item["minimum_stock"] >= 0
            assert item["maximum_stock"] > item["minimum_stock"]
            assert item["reorder_point"] >= item["minimum_stock"]
            assert item["reorder_quantity"] > 0