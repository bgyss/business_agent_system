import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from models.financial import TransactionType


@dataclass
class BusinessProfile:
    name: str
    business_type: str
    avg_daily_revenue: float
    revenue_variance: float
    avg_transaction_size: float
    expense_categories: Dict[str, float]  # category -> daily average
    seasonal_factors: Dict[int, float]  # month -> multiplier
    customer_patterns: Dict[str, float]  # day_of_week -> multiplier


class FinancialDataGenerator:
    def __init__(self, business_profile: BusinessProfile, start_date: datetime = None):
        self.profile = business_profile
        self.start_date = start_date or datetime.now() - timedelta(days=90)
        self.customers = self._generate_customers()
        self.vendors = self._generate_vendors()

    def _generate_customers(self) -> List[str]:
        if self.profile.business_type == "restaurant":
            return [
                "Walk-in Customer", "DoorDash", "Uber Eats", "Grubhub",
                "Catering Order", "Private Party", "Regular Customer #1",
                "Regular Customer #2", "Corporate Account", "Special Event"
            ]
        elif self.profile.business_type == "retail":
            return [
                "Cash Sale", "Credit Card Sale", "Online Order", "Wholesale Customer",
                "Regular Customer #1", "Regular Customer #2", "Seasonal Customer",
                "Bulk Order", "Corporate Account", "Return Customer"
            ]
        else:
            return ["Customer #1", "Customer #2", "Customer #3", "Online Sale", "Repeat Customer"]

    def _generate_vendors(self) -> Dict[str, Dict[str, Any]]:
        if self.profile.business_type == "restaurant":
            return {
                "Fresh Foods Distributor": {"category": "food_supplies", "payment_terms": 15},
                "ABC Beverage Co": {"category": "beverages", "payment_terms": 30},
                "Kitchen Equipment Rental": {"category": "equipment", "payment_terms": 30},
                "City Utilities": {"category": "utilities", "payment_terms": 30},
                "Restaurant Supply Co": {"category": "supplies", "payment_terms": 15},
                "Cleaning Services Inc": {"category": "services", "payment_terms": 15},
                "Marketing Agency": {"category": "marketing", "payment_terms": 30},
                "Insurance Company": {"category": "insurance", "payment_terms": 30}
            }
        else:
            return {
                "Wholesale Supplier": {"category": "inventory", "payment_terms": 30},
                "Shipping Company": {"category": "shipping", "payment_terms": 15},
                "Utility Company": {"category": "utilities", "payment_terms": 30},
                "Marketing Services": {"category": "marketing", "payment_terms": 30},
                "Equipment Lease": {"category": "equipment", "payment_terms": 30},
                "Insurance Provider": {"category": "insurance", "payment_terms": 30},
                "Office Supplies": {"category": "supplies", "payment_terms": 15},
                "Software Subscription": {"category": "software", "payment_terms": 30}
            }

    def generate_daily_transactions(self, date: datetime) -> List[Dict[str, Any]]:
        transactions = []

        # Apply seasonal and day-of-week factors
        month_factor = self.profile.seasonal_factors.get(date.month, 1.0)
        day_factor = self.profile.customer_patterns.get(date.strftime('%A').lower(), 1.0)

        # Generate revenue transactions
        daily_revenue_target = self.profile.avg_daily_revenue * month_factor * day_factor
        daily_revenue_actual = random.normalvariate(daily_revenue_target,
                                                   daily_revenue_target * self.profile.revenue_variance)

        # Generate individual sales transactions
        revenue_generated = 0
        while revenue_generated < daily_revenue_actual:
            transaction_amount = max(1, random.normalvariate(
                self.profile.avg_transaction_size,
                self.profile.avg_transaction_size * 0.3
            ))

            if revenue_generated + transaction_amount > daily_revenue_actual * 1.2:
                transaction_amount = daily_revenue_actual - revenue_generated

            if transaction_amount > 0:
                transactions.append({
                    "description": f"Sale - {random.choice(self.customers)}",
                    "amount": round(transaction_amount, 2),
                    "transaction_type": TransactionType.INCOME,
                    "category": "sales",
                    "transaction_date": date + timedelta(
                        hours=random.randint(9, 21),
                        minutes=random.randint(0, 59)
                    ),
                    "from_account_id": None,
                    "to_account_id": "revenue_account",
                    "reference_number": f"SALE-{date.strftime('%Y%m%d')}-{len(transactions)+1:04d}"
                })
                revenue_generated += transaction_amount

        # Generate expense transactions
        for category, daily_avg in self.profile.expense_categories.items():
            if random.random() < 0.7:  # 70% chance of expense on any given day
                variance = 0.3 if category not in ["rent", "insurance"] else 0.05
                amount = max(1, random.normalvariate(daily_avg, daily_avg * variance))

                transactions.append({
                    "description": f"{category.replace('_', ' ').title()} - {self._get_vendor_for_category(category)}",
                    "amount": round(amount, 2),
                    "transaction_type": TransactionType.EXPENSE,
                    "category": category,
                    "transaction_date": date + timedelta(
                        hours=random.randint(8, 18),
                        minutes=random.randint(0, 59)
                    ),
                    "from_account_id": "checking_account",
                    "to_account_id": None,
                    "reference_number": f"EXP-{date.strftime('%Y%m%d')}-{category.upper()}-{random.randint(1000, 9999)}"
                })

        # Occasionally generate larger one-time expenses
        if random.random() < 0.05:  # 5% chance per day
            large_expenses = [
                ("Equipment Repair", "maintenance", random.randint(200, 1500)),
                ("Marketing Campaign", "marketing", random.randint(500, 2000)),
                ("Equipment Purchase", "equipment", random.randint(1000, 5000)),
                ("Legal Fees", "professional_services", random.randint(300, 1200)),
                ("Bulk Supply Purchase", "supplies", random.randint(400, 1800))
            ]

            expense = random.choice(large_expenses)
            transactions.append({
                "description": expense[0],
                "amount": expense[2],
                "transaction_type": TransactionType.EXPENSE,
                "category": expense[1],
                "transaction_date": date + timedelta(
                    hours=random.randint(9, 17),
                    minutes=random.randint(0, 59)
                ),
                "from_account_id": "checking_account",
                "to_account_id": None,
                "reference_number": f"EXP-{date.strftime('%Y%m%d')}-SPECIAL-{random.randint(1000, 9999)}"
            })

        return transactions

    def _get_vendor_for_category(self, category: str) -> str:
        matching_vendors = [name for name, info in self.vendors.items()
                          if info["category"] == category]
        if matching_vendors:
            return random.choice(matching_vendors)
        return "General Vendor"

    def generate_accounts_receivable(self, date: datetime) -> List[Dict[str, Any]]:
        receivables = []

        # Generate B2B invoices occasionally
        if random.random() < 0.1:  # 10% chance per day
            amount = random.randint(500, 5000)
            due_date = date + timedelta(days=random.choice([15, 30, 45]))

            receivables.append({
                "customer_name": random.choice(["Corporate Client A", "Corporate Client B",
                                             "Catering Contract", "Bulk Order Customer"]),
                "invoice_number": f"INV-{date.strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                "amount": amount,
                "due_date": due_date,
                "invoice_date": date,
                "status": "unpaid"
            })

        return receivables

    def generate_accounts_payable(self, date: datetime) -> List[Dict[str, Any]]:
        payables = []

        # Generate vendor invoices
        for vendor_name, vendor_info in self.vendors.items():
            if random.random() < 0.05:  # 5% chance per vendor per day
                category = vendor_info["category"]

                if category in ["rent", "insurance"]:
                    if date.day == 1:  # Monthly bills on 1st of month
                        amount = random.randint(800, 3000)
                    else:
                        continue
                else:
                    amount = random.randint(100, 2000)

                due_date = date + timedelta(days=vendor_info["payment_terms"])

                payables.append({
                    "vendor_name": vendor_name,
                    "invoice_number": f"VINV-{date.strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                    "amount": amount,
                    "due_date": due_date,
                    "invoice_date": date,
                    "status": "unpaid"
                })

        return payables

    def generate_period_data(self, start_date: datetime, end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        all_transactions = []
        all_receivables = []
        all_payables = []

        current_date = start_date
        while current_date <= end_date:
            # Skip Sundays for most business types
            if current_date.weekday() != 6 or self.profile.business_type == "restaurant":
                daily_transactions = self.generate_daily_transactions(current_date)
                all_transactions.extend(daily_transactions)

                daily_receivables = self.generate_accounts_receivable(current_date)
                all_receivables.extend(daily_receivables)

                daily_payables = self.generate_accounts_payable(current_date)
                all_payables.extend(daily_payables)

            current_date += timedelta(days=1)

        return {
            "transactions": all_transactions,
            "accounts_receivable": all_receivables,
            "accounts_payable": all_payables
        }

    def generate_anomalies(self, transactions: List[Dict[str, Any]], anomaly_rate: float = 0.02) -> List[Dict[str, Any]]:
        anomalous_transactions = []
        
        # Handle empty input
        if not transactions:
            return anomalous_transactions
            
        num_anomalies = max(1, int(len(transactions) * anomaly_rate))
        selected_transactions = random.sample(transactions, num_anomalies)

        for transaction in selected_transactions:
            anomaly_type = random.choice([
                "unusual_amount", "unusual_time", "duplicate", "missing_reference"
            ])

            anomalous_transaction = transaction.copy()

            if anomaly_type == "unusual_amount":
                # Make amount 3-10x larger than normal
                multiplier = random.uniform(3, 10)
                anomalous_transaction["amount"] = round(transaction["amount"] * multiplier, 2)
                anomalous_transaction["description"] += " [ANOMALY: Unusual Amount]"

            elif anomaly_type == "unusual_time":
                # Transaction at 2-4 AM
                anomalous_transaction["transaction_date"] = transaction["transaction_date"].replace(
                    hour=random.randint(2, 4),
                    minute=random.randint(0, 59)
                )
                anomalous_transaction["description"] += " [ANOMALY: Unusual Time]"

            elif anomaly_type == "duplicate":
                # Create near-duplicate transaction
                anomalous_transaction["transaction_date"] = transaction["transaction_date"] + timedelta(minutes=random.randint(1, 30))
                anomalous_transaction["description"] += " [ANOMALY: Potential Duplicate]"

            elif anomaly_type == "missing_reference":
                anomalous_transaction["reference_number"] = None
                anomalous_transaction["description"] += " [ANOMALY: Missing Reference]"

            anomalous_transactions.append(anomalous_transaction)

        return anomalous_transactions


def get_restaurant_profile() -> BusinessProfile:
    return BusinessProfile(
        name="Sample Restaurant",
        business_type="restaurant",
        avg_daily_revenue=2500,
        revenue_variance=0.25,
        avg_transaction_size=45,
        expense_categories={
            "food_supplies": 600,
            "beverages": 200,
            "labor": 800,
            "rent": 150,  # Daily portion of monthly rent
            "utilities": 50,
            "supplies": 80,
            "marketing": 30,
            "insurance": 25,
            "maintenance": 40
        },
        seasonal_factors={
            1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.3, 8: 1.2, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.4
        },
        customer_patterns={
            "monday": 0.7, "tuesday": 0.8, "wednesday": 0.9,
            "thursday": 1.0, "friday": 1.3, "saturday": 1.4, "sunday": 1.1
        }
    )


def get_retail_profile() -> BusinessProfile:
    return BusinessProfile(
        name="Sample Retail Store",
        business_type="retail",
        avg_daily_revenue=1800,
        revenue_variance=0.3,
        avg_transaction_size=65,
        expense_categories={
            "inventory": 500,
            "labor": 400,
            "rent": 100,
            "utilities": 35,
            "supplies": 25,
            "marketing": 50,
            "insurance": 20,
            "shipping": 60,
            "software": 15
        },
        seasonal_factors={
            1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.0, 6: 0.9,
            7: 0.8, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.6
        },
        customer_patterns={
            "monday": 0.8, "tuesday": 0.9, "wednesday": 0.9,
            "thursday": 1.0, "friday": 1.2, "saturday": 1.4, "sunday": 0.6
        }
    )
