# simulation/CLAUDE.md - Business Simulation and Configuration Guide

This document provides comprehensive guidance for business simulation, data generation, and configuration management in the Business Agent Management System.

## Business Configuration

### Adding New Business Types

1. **Create Configuration File**:
   - Copy existing config from `config/` directory
   - Adjust business-specific parameters
   - Update agent thresholds and intervals
   - Configure simulation profiles

2. **Key Configuration Sections**:
   ```yaml
   business:
     name: "Business Name"
     type: "business_type"
   
   agents:
     accounting:
       anomaly_threshold: 0.25
       alert_thresholds:
         cash_low: 1000
   
   simulation:
     business_profile:
       avg_daily_revenue: 2500
       seasonal_factors: {...}
   ```

3. **Business Profile Requirements**:
   - Revenue patterns (daily, seasonal, weekly)
   - Expense categories and typical amounts
   - Customer behavior patterns
   - Inventory consumption rates (if applicable)
   - Staffing requirements and labor costs

### Simulation Customization

**Financial Data Generation**:
- Realistic transaction patterns
- Seasonal revenue variations
- Expense category distributions
- Accounts receivable/payable timing
- Anomaly injection for testing

**Inventory Simulation**:
- Consumption based on sales volume
- Supplier delivery patterns
- Waste and spoilage modeling
- Reorder point calculations
- Purchase order generation

**HR Data Simulation**:
- Employee scheduling patterns
- Time tracking and attendance
- Labor cost calculations
- Overtime management
- Leave request handling

## Simulation Engine Architecture

### Business Simulator Core
```python
class BusinessSimulator:
    """Core business simulation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.business_profile = config["simulation"]["business_profile"]
        self.generators = {
            "financial": FinancialGenerator(config),
            "inventory": InventorySimulator(config),
            "employee": EmployeeSimulator(config)
        }
        self.anomaly_injector = AnomalyInjector(config)
    
    async def simulate_business_day(self, date: datetime) -> Dict[str, Any]:
        """Simulate a full business day"""
        daily_data = {}
        
        # Generate financial transactions
        daily_data["transactions"] = await self.generators["financial"].generate_daily_transactions(date)
        
        # Simulate inventory movements
        daily_data["inventory_movements"] = await self.generators["inventory"].simulate_daily_consumption(date)
        
        # Generate employee time entries
        daily_data["time_entries"] = await self.generators["employee"].generate_daily_time_entries(date)
        
        # Inject anomalies if configured
        if self.config.get("inject_anomalies", False):
            daily_data = self.anomaly_injector.inject_anomalies(daily_data, date)
        
        return daily_data
```

### Financial Data Generation

#### Transaction Generator
```python
class FinancialGenerator:
    """Generate realistic financial transactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile = config["simulation"]["business_profile"]
        self.seasonal_factors = self.profile.get("seasonal_factors", {})
        self.revenue_patterns = self.profile.get("revenue_patterns", {})
    
    async def generate_daily_transactions(self, date: datetime) -> List[Dict[str, Any]]:
        """Generate daily financial transactions"""
        transactions = []
        
        # Calculate daily revenue based on business profile
        base_revenue = self.profile["avg_daily_revenue"]
        seasonal_multiplier = self._get_seasonal_multiplier(date)
        day_of_week_multiplier = self._get_day_of_week_multiplier(date)
        
        daily_revenue = base_revenue * seasonal_multiplier * day_of_week_multiplier
        
        # Generate revenue transactions
        revenue_transactions = self._generate_revenue_transactions(date, daily_revenue)
        transactions.extend(revenue_transactions)
        
        # Generate expense transactions
        expense_transactions = self._generate_expense_transactions(date, daily_revenue)
        transactions.extend(expense_transactions)
        
        # Generate accounts receivable/payable
        ar_ap_transactions = self._generate_ar_ap_transactions(date)
        transactions.extend(ar_ap_transactions)
        
        return transactions
    
    def _get_seasonal_multiplier(self, date: datetime) -> float:
        """Calculate seasonal revenue multiplier"""
        month = date.month
        return self.seasonal_factors.get(str(month), 1.0)
    
    def _get_day_of_week_multiplier(self, date: datetime) -> float:
        """Calculate day-of-week revenue multiplier"""
        day_name = date.strftime('%A').lower()
        return self.revenue_patterns.get(day_name, 1.0)
    
    def _generate_revenue_transactions(self, date: datetime, total_revenue: float) -> List[Dict]:
        """Generate individual revenue transactions"""
        transactions = []
        remaining_revenue = total_revenue
        
        # Generate multiple transactions throughout the day
        num_transactions = max(1, int(total_revenue / 100))  # Roughly $100 per transaction
        
        for i in range(num_transactions):
            if remaining_revenue <= 0:
                break
            
            # Random transaction amount (weighted toward smaller amounts)
            amount = min(
                remaining_revenue,
                np.random.exponential(total_revenue / num_transactions)
            )
            
            transaction_time = date + timedelta(
                hours=random.randint(8, 20),  # Business hours
                minutes=random.randint(0, 59)
            )
            
            transactions.append({
                "amount": round(amount, 2),
                "description": f"Sales revenue - Transaction {i+1}",
                "account_id": 4001,  # Revenue account
                "transaction_type": "credit",
                "transaction_date": transaction_time,
                "reference_number": f"REV-{date.strftime('%Y%m%d')}-{i+1:03d}"
            })
            
            remaining_revenue -= amount
        
        return transactions
```

#### Expense Pattern Generation
```python
def _generate_expense_transactions(self, date: datetime, daily_revenue: float) -> List[Dict]:
    """Generate realistic expense transactions"""
    transactions = []
    expense_categories = self.profile.get("expense_categories", {})
    
    for category, config in expense_categories.items():
        # Calculate category expense based on revenue percentage
        category_amount = daily_revenue * config["percentage_of_revenue"]
        
        # Add some randomness
        variance = config.get("variance", 0.2)
        actual_amount = category_amount * (1 + random.uniform(-variance, variance))
        
        if actual_amount > config.get("min_amount", 10):
            transactions.append({
                "amount": round(actual_amount, 2),
                "description": f"{category} - {date.strftime('%Y-%m-%d')}",
                "account_id": config["account_id"],
                "transaction_type": "debit",
                "transaction_date": date + timedelta(hours=random.randint(9, 17)),
                "reference_number": f"EXP-{category[:3].upper()}-{date.strftime('%Y%m%d')}"
            })
    
    return transactions
```

### Inventory Simulation

#### Stock Movement Generator
```python
class InventorySimulator:
    """Simulate realistic inventory movements"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile = config["simulation"]["business_profile"]
        self.consumption_patterns = self.profile.get("inventory_patterns", {})
    
    async def simulate_daily_consumption(self, date: datetime) -> List[Dict[str, Any]]:
        """Simulate daily inventory consumption"""
        movements = []
        
        # Get all active items from database
        items = await self._get_active_items()
        
        for item in items:
            # Calculate consumption based on item category and business activity
            consumption = self._calculate_item_consumption(item, date)
            
            if consumption > 0:
                movements.append({
                    "item_id": item["id"],
                    "movement_type": "OUT",
                    "quantity": consumption,
                    "movement_date": date + timedelta(
                        hours=random.randint(8, 20),
                        minutes=random.randint(0, 59)
                    ),
                    "reference_number": f"USAGE-{date.strftime('%Y%m%d')}-{item['id']}",
                    "notes": f"Daily consumption - {item['name']}"
                })
        
        # Simulate supplier deliveries
        delivery_movements = await self._simulate_supplier_deliveries(date)
        movements.extend(delivery_movements)
        
        return movements
    
    def _calculate_item_consumption(self, item: Dict, date: datetime) -> int:
        """Calculate realistic item consumption for the day"""
        base_consumption = item.get("avg_daily_consumption", 5)
        
        # Apply seasonal factors
        seasonal_multiplier = self._get_seasonal_multiplier(date, item["category"])
        
        # Apply day-of-week factors
        dow_multiplier = self._get_day_of_week_multiplier(date)
        
        # Add randomness (Poisson distribution for discrete consumption)
        expected_consumption = base_consumption * seasonal_multiplier * dow_multiplier
        actual_consumption = np.random.poisson(max(0.1, expected_consumption))
        
        # Don't consume more than available stock
        return min(actual_consumption, item["current_stock"])
    
    async def _simulate_supplier_deliveries(self, date: datetime) -> List[Dict]:
        """Simulate supplier deliveries based on purchase orders"""
        deliveries = []
        
        # Check for pending purchase orders that should be delivered
        pending_orders = await self._get_pending_purchase_orders(date)
        
        for order in pending_orders:
            # Random chance of delivery (based on supplier reliability)
            delivery_probability = order["supplier"]["reliability_score"]
            
            if random.random() < delivery_probability:
                for item in order["items"]:
                    deliveries.append({
                        "item_id": item["item_id"],
                        "movement_type": "IN",
                        "quantity": item["quantity"],
                        "movement_date": date + timedelta(
                            hours=random.randint(9, 16)
                        ),
                        "reference_number": f"PO-{order['id']}-DELIVERY",
                        "notes": f"Delivery from {order['supplier']['name']}",
                        "unit_cost": item["unit_cost"]
                    })
        
        return deliveries
```

### Employee Data Simulation

#### Time Entry Generation
```python
class EmployeeSimulator:
    """Simulate employee time tracking and scheduling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile = config["simulation"]["business_profile"]
        self.staffing_patterns = self.profile.get("staffing_patterns", {})
    
    async def generate_daily_time_entries(self, date: datetime) -> List[Dict[str, Any]]:
        """Generate realistic employee time entries"""
        time_entries = []
        
        # Get active employees
        employees = await self._get_active_employees()
        
        for employee in employees:
            # Check if employee is scheduled to work
            if self._is_employee_scheduled(employee, date):
                entry = self._generate_employee_time_entry(employee, date)
                if entry:
                    time_entries.append(entry)
        
        return time_entries
    
    def _generate_employee_time_entry(self, employee: Dict, date: datetime) -> Optional[Dict]:
        """Generate time entry for specific employee"""
        # Get employee's typical schedule
        schedule = employee.get("typical_schedule", {})
        day_name = date.strftime('%A').lower()
        
        if day_name not in schedule:
            return None  # Employee doesn't work this day
        
        day_schedule = schedule[day_name]
        
        # Add some variation to start/end times
        start_variation = random.randint(-15, 15)  # ±15 minutes
        end_variation = random.randint(-30, 30)    # ±30 minutes
        
        clock_in = datetime.combine(
            date,
            time(hour=day_schedule["start_hour"], minute=day_schedule["start_minute"])
        ) + timedelta(minutes=start_variation)
        
        clock_out = datetime.combine(
            date,
            time(hour=day_schedule["end_hour"], minute=day_schedule["end_minute"])
        ) + timedelta(minutes=end_variation)
        
        # Simulate break time
        break_minutes = random.randint(15, 60)  # 15-60 minute break
        
        # Occasional late arrival or early departure
        if random.random() < 0.1:  # 10% chance
            if random.random() < 0.5:
                clock_in += timedelta(minutes=random.randint(5, 30))  # Late arrival
            else:
                clock_out -= timedelta(minutes=random.randint(5, 30))  # Early departure
        
        return {
            "employee_id": employee["id"],
            "clock_in": clock_in,
            "clock_out": clock_out,
            "break_minutes": break_minutes,
            "notes": f"Regular shift - {date.strftime('%A')}"
        }
```

### Anomaly Injection System

#### Anomaly Generator
```python
class AnomalyInjector:
    """Inject realistic business anomalies for agent testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anomaly_types = config.get("anomaly_types", {})
        self.injection_probability = config.get("anomaly_probability", 0.05)  # 5% chance
    
    def inject_anomalies(self, daily_data: Dict, date: datetime) -> Dict:
        """Inject anomalies into daily business data"""
        if random.random() > self.injection_probability:
            return daily_data  # No anomaly today
        
        # Choose random anomaly type
        anomaly_type = random.choice(list(self.anomaly_types.keys()))
        
        if anomaly_type == "financial_anomaly":
            daily_data = self._inject_financial_anomaly(daily_data, date)
        elif anomaly_type == "inventory_anomaly":
            daily_data = self._inject_inventory_anomaly(daily_data, date)
        elif anomaly_type == "employee_anomaly":
            daily_data = self._inject_employee_anomaly(daily_data, date)
        
        # Log anomaly for testing verification
        self._log_injected_anomaly(anomaly_type, date)
        
        return daily_data
    
    def _inject_financial_anomaly(self, daily_data: Dict, date: datetime) -> Dict:
        """Inject financial anomalies"""
        anomaly_config = self.anomaly_types["financial_anomaly"]
        
        if "large_transaction" in anomaly_config:
            # Add unusually large transaction
            large_amount = random.uniform(5000, 20000)
            anomaly_transaction = {
                "amount": large_amount,
                "description": "Large unusual transaction - potential fraud",
                "account_id": 4001,
                "transaction_type": "credit",
                "transaction_date": date + timedelta(hours=random.randint(10, 18)),
                "reference_number": f"ANOM-{date.strftime('%Y%m%d')}-LARGE"
            }
            daily_data["transactions"].append(anomaly_transaction)
        
        if "duplicate_transactions" in anomaly_config:
            # Add duplicate transactions
            if daily_data["transactions"]:
                original = random.choice(daily_data["transactions"])
                duplicate = original.copy()
                duplicate["reference_number"] += "-DUP"
                duplicate["transaction_date"] += timedelta(minutes=5)
                daily_data["transactions"].append(duplicate)
        
        return daily_data
    
    def _inject_inventory_anomaly(self, daily_data: Dict, date: datetime) -> Dict:
        """Inject inventory anomalies"""
        if "excessive_consumption" in self.anomaly_types.get("inventory_anomaly", {}):
            # Simulate theft or excessive waste
            if daily_data["inventory_movements"]:
                # Pick random item and increase consumption dramatically
                movement = random.choice(daily_data["inventory_movements"])
                if movement["movement_type"] == "OUT":
                    movement["quantity"] *= random.randint(3, 8)  # 3-8x normal consumption
                    movement["notes"] += " - ANOMALY: Excessive consumption detected"
        
        return daily_data
    
    def _log_injected_anomaly(self, anomaly_type: str, date: datetime):
        """Log injected anomaly for testing verification"""
        log_entry = {
            "date": date.isoformat(),
            "anomaly_type": anomaly_type,
            "injected_at": datetime.utcnow().isoformat()
        }
        
        # Write to anomaly log file
        log_file = "simulation/anomaly_log.json"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Failed to log anomaly: {e}")
```

### Configuration Management

#### Business Profile Templates
```yaml
# config/restaurant_profile.yaml
business:
  name: "Sample Restaurant"
  type: "restaurant"
  
simulation:
  business_profile:
    avg_daily_revenue: 2500
    
    seasonal_factors:
      "1": 0.8   # January - slow
      "2": 0.9   # February
      "3": 1.0   # March
      "4": 1.1   # April
      "5": 1.2   # May - busy
      "6": 1.3   # June - peak
      "7": 1.4   # July - peak
      "8": 1.3   # August
      "9": 1.1   # September
      "10": 1.0  # October
      "11": 1.2  # November - holidays
      "12": 1.4  # December - holidays
    
    revenue_patterns:
      monday: 0.8
      tuesday: 0.9
      wednesday: 0.95
      thursday: 1.0
      friday: 1.3
      saturday: 1.4
      sunday: 1.1
    
    expense_categories:
      food_costs:
        percentage_of_revenue: 0.30
        variance: 0.15
        account_id: 5001
        min_amount: 100
      
      labor_costs:
        percentage_of_revenue: 0.25
        variance: 0.10
        account_id: 5002
        min_amount: 200
      
      utilities:
        percentage_of_revenue: 0.05
        variance: 0.20
        account_id: 5003
        min_amount: 50
    
    inventory_patterns:
      food_items:
        base_consumption_per_100_revenue: 5
        seasonal_variance: 0.2
        spoilage_rate: 0.02
      
      beverages:
        base_consumption_per_100_revenue: 8
        seasonal_variance: 0.3
        spoilage_rate: 0.01
    
    staffing_patterns:
      min_staff_weekday: 3
      min_staff_weekend: 5
      peak_hours: [12, 13, 18, 19, 20]  # Lunch and dinner
      
anomaly_types:
  financial_anomaly:
    - large_transaction
    - duplicate_transactions
    - negative_revenue
  
  inventory_anomaly:
    - excessive_consumption
    - missing_deliveries
    - spoilage_spike
  
  employee_anomaly:
    - excessive_overtime
    - no_show
    - early_departure

anomaly_probability: 0.05  # 5% chance per day
```

### Performance Optimization

#### Batch Data Generation
```python
class OptimizedSimulator:
    """Performance-optimized simulation for large datasets"""
    
    async def generate_batch_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate data for multiple days efficiently"""
        days = (end_date - start_date).days
        batch_size = 10  # Process 10 days at a time
        
        all_data = {
            "transactions": [],
            "inventory_movements": [],
            "time_entries": []
        }
        
        for i in range(0, days, batch_size):
            batch_start = start_date + timedelta(days=i)
            batch_end = min(start_date + timedelta(days=i + batch_size), end_date)
            
            batch_data = await self._generate_batch_chunk(batch_start, batch_end)
            
            # Merge batch data
            for key in all_data:
                all_data[key].extend(batch_data.get(key, []))
            
            # Yield control to allow other operations
            await asyncio.sleep(0.01)
        
        return all_data
    
    async def _generate_batch_chunk(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate data for a chunk of days"""
        chunk_data = {
            "transactions": [],
            "inventory_movements": [],
            "time_entries": []
        }
        
        current_date = start_date
        while current_date < end_date:
            daily_data = await self.simulate_business_day(current_date)
            
            for key in chunk_data:
                chunk_data[key].extend(daily_data.get(key, []))
            
            current_date += timedelta(days=1)
        
        return chunk_data
```

---

*This document should be updated when new simulation patterns are added or when business profile requirements change.*