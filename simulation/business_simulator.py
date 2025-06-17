import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from simulation.financial_generator import FinancialDataGenerator, get_restaurant_profile, get_retail_profile
from models.financial import Account, Transaction, AccountsReceivable, AccountsPayable, Base as FinancialBase
from models.inventory import Item, StockMovement, Supplier, Base as InventoryBase
from models.employee import Employee, TimeRecord, Schedule, Base as EmployeeBase


class BusinessSimulator:
    def __init__(self, config: Dict[str, Any], db_url: str = "sqlite:///business_simulation.db"):
        self.config = config
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        FinancialBase.metadata.create_all(bind=self.engine)
        InventoryBase.metadata.create_all(bind=self.engine)
        EmployeeBase.metadata.create_all(bind=self.engine)
        
        self.financial_generator = None
        self.is_running = False
        
    def initialize_business(self, business_type: str = "restaurant"):
        session = self.SessionLocal()
        try:
            # Set up financial generator
            if business_type == "restaurant":
                profile = get_restaurant_profile()
            else:
                profile = get_retail_profile()
            
            self.financial_generator = FinancialDataGenerator(profile)
            
            # Create initial accounts if they don't exist
            existing_accounts = session.query(Account).count()
            if existing_accounts == 0:
                initial_accounts = [
                    Account(
                        id="checking_account",
                        name="Business Checking",
                        account_type="checking",
                        balance=25000.00,
                        description="Main business checking account"
                    ),
                    Account(
                        id="savings_account", 
                        name="Business Savings",
                        account_type="savings",
                        balance=50000.00,
                        description="Business savings account"
                    ),
                    Account(
                        id="revenue_account",
                        name="Revenue",
                        account_type="revenue", 
                        balance=0.00,
                        description="Revenue tracking account"
                    ),
                    Account(
                        id="expense_account",
                        name="General Expenses",
                        account_type="expense",
                        balance=0.00,
                        description="General expense tracking account"
                    )
                ]
                
                for account in initial_accounts:
                    session.add(account)
                
                session.commit()
                print(f"Created {len(initial_accounts)} initial accounts")
            
            # Create sample suppliers if they don't exist
            existing_suppliers = session.query(Supplier).count()
            if existing_suppliers == 0:
                suppliers = []
                for vendor_name, vendor_info in self.financial_generator.vendors.items():
                    supplier = Supplier(
                        name=vendor_name,
                        contact_person=f"Contact for {vendor_name}",
                        email=f"contact@{vendor_name.lower().replace(' ', '')}.com",
                        phone=f"555-{len(suppliers):04d}",
                        lead_time_days=vendor_info["payment_terms"] // 2,
                        payment_terms=f"Net {vendor_info['payment_terms']} days"
                    )
                    suppliers.append(supplier)
                    session.add(supplier)
                
                session.commit()
                print(f"Created {len(suppliers)} suppliers")
            
            # Create sample employees if they don't exist
            existing_employees = session.query(Employee).count()
            if existing_employees == 0:
                employees = [
                    Employee(
                        employee_id="EMP001",
                        first_name="John",
                        last_name="Manager",
                        email="john.manager@business.com",
                        hire_date=datetime.now().date() - timedelta(days=365),
                        position="General Manager",
                        department="Management",
                        hourly_rate=25.00,
                        is_full_time=True
                    ),
                    Employee(
                        employee_id="EMP002", 
                        first_name="Sarah",
                        last_name="Server",
                        email="sarah.server@business.com",
                        hire_date=datetime.now().date() - timedelta(days=180),
                        position="Server" if business_type == "restaurant" else "Sales Associate",
                        department="Operations",
                        hourly_rate=15.00,
                        is_full_time=False
                    ),
                    Employee(
                        employee_id="EMP003",
                        first_name="Mike",
                        last_name="Kitchen" if business_type == "restaurant" else "Stock",
                        email="mike.kitchen@business.com",
                        hire_date=datetime.now().date() - timedelta(days=90),
                        position="Cook" if business_type == "restaurant" else "Stock Clerk", 
                        department="Operations",
                        hourly_rate=18.00,
                        is_full_time=True
                    )
                ]
                
                for employee in employees:
                    session.add(employee)
                
                session.commit()
                print(f"Created {len(employees)} employees")
                
        finally:
            session.close()
    
    def simulate_historical_data(self, days_back: int = 90):
        if not self.financial_generator:
            raise ValueError("Business must be initialized first")
            
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Generating {days_back} days of historical data...")
        
        # Generate financial data
        financial_data = self.financial_generator.generate_period_data(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time())
        )
        
        # Add some anomalies for testing
        anomalous_transactions = self.financial_generator.generate_anomalies(
            financial_data["transactions"], 
            anomaly_rate=0.01
        )
        financial_data["transactions"].extend(anomalous_transactions)
        
        session = self.SessionLocal()
        try:
            # Insert transactions
            for tx_data in financial_data["transactions"]:
                transaction = Transaction(**tx_data)
                session.add(transaction)
            
            # Insert accounts receivable  
            for ar_data in financial_data["accounts_receivable"]:
                receivable = AccountsReceivable(**ar_data)
                session.add(receivable)
            
            # Insert accounts payable
            for ap_data in financial_data["accounts_payable"]:
                payable = AccountsPayable(**ap_data)
                session.add(payable)
            
            session.commit()
            
            print(f"Generated:")
            print(f"  - {len(financial_data['transactions'])} transactions")
            print(f"  - {len(financial_data['accounts_receivable'])} receivables")
            print(f"  - {len(financial_data['accounts_payable'])} payables")
            
        finally:
            session.close()
    
    async def start_real_time_simulation(self, message_queue: asyncio.Queue):
        self.is_running = True
        print("Starting real-time business simulation...")
        
        while self.is_running:
            try:
                # Generate today's transactions
                today = datetime.now()
                daily_transactions = self.financial_generator.generate_daily_transactions(today)
                
                # Add transactions to database
                session = self.SessionLocal()
                try:
                    for tx_data in daily_transactions[-3:]:  # Only add a few per cycle
                        transaction = Transaction(**tx_data)
                        session.add(transaction)
                    session.commit()
                    
                    # Notify agents of new data
                    if daily_transactions:
                        await message_queue.put({
                            "type": "new_transaction",
                            "transaction": daily_transactions[-1]  # Send latest transaction
                        })
                
                finally:
                    session.close()
                
                # Generate other events occasionally
                if datetime.now().minute == 0:  # Every hour
                    await message_queue.put({"type": "cash_flow_check"})
                
                if datetime.now().hour == 9 and datetime.now().minute == 0:  # Daily at 9 AM
                    await message_queue.put({"type": "daily_analysis"})
                
                if datetime.now().day == 1 and datetime.now().hour == 8:  # Monthly at 8 AM on 1st
                    await message_queue.put({"type": "aging_analysis"})
                
                # Wait before next cycle
                await asyncio.sleep(self.config.get("simulation_interval", 60))  # Default 1 minute
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                await asyncio.sleep(5)
    
    async def stop_simulation(self):
        self.is_running = False
        print("Stopping business simulation...")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        session = self.SessionLocal()
        try:
            transaction_count = session.query(Transaction).count()
            receivable_count = session.query(AccountsReceivable).count()
            payable_count = session.query(AccountsPayable).count()
            
            latest_transaction = session.query(Transaction).order_by(
                Transaction.created_at.desc()
            ).first()
            
            return {
                "is_running": self.is_running,
                "transaction_count": transaction_count,
                "receivable_count": receivable_count,
                "payable_count": payable_count,
                "latest_transaction_date": latest_transaction.transaction_date if latest_transaction else None,
                "business_type": self.financial_generator.profile.business_type if self.financial_generator else None
            }
            
        finally:
            session.close()
    
    def generate_sample_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = [
            {
                "name": "Cash Flow Crisis",
                "description": "Simulate a period of low cash flow",
                "actions": [
                    "Reduce daily revenue by 40% for 2 weeks",
                    "Add unexpected large expense ($5000)",
                    "Delay customer payments by 15 days"
                ]
            },
            {
                "name": "Seasonal Rush",
                "description": "Simulate holiday season increase",
                "actions": [
                    "Increase daily revenue by 60% for 1 month",
                    "Add overtime labor costs",
                    "Increase supply costs by 20%"
                ]
            },
            {
                "name": "Equipment Failure",
                "description": "Major equipment breakdown",
                "actions": [
                    "Add emergency repair cost ($3000)",
                    "Reduce revenue by 25% for 3 days",
                    "Add equipment rental costs"
                ]
            },
            {
                "name": "New Competitor",
                "description": "Competition affects business",
                "actions": [
                    "Reduce daily revenue by 15% ongoing",
                    "Increase marketing expenses",
                    "Add promotional discounts"
                ]
            }
        ]
        
        return scenarios
    
    def apply_scenario(self, scenario_name: str):
        print(f"Applying scenario: {scenario_name}")
        # This would modify the generator parameters to simulate the scenario
        # Implementation would depend on specific scenario requirements
        pass