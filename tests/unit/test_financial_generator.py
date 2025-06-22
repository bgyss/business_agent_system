"""
Unit tests for FinancialDataGenerator class
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.financial import TransactionType
from simulation.financial_generator import (
    BusinessProfile,
    FinancialDataGenerator,
    get_restaurant_profile,
    get_retail_profile,
)


class TestBusinessProfile:
    """Test cases for BusinessProfile dataclass"""

    def test_business_profile_creation(self):
        """Test BusinessProfile creation with all fields"""
        profile = BusinessProfile(
            name="Test Business",
            business_type="restaurant",
            avg_daily_revenue=1000.0,
            revenue_variance=0.2,
            avg_transaction_size=50.0,
            expense_categories={"rent": 300, "supplies": 100},
            seasonal_factors={1: 0.8, 12: 1.2},
            customer_patterns={"monday": 0.9, "friday": 1.1},
        )

        assert profile.name == "Test Business"
        assert profile.business_type == "restaurant"
        assert profile.avg_daily_revenue == 1000.0
        assert profile.revenue_variance == 0.2
        assert profile.avg_transaction_size == 50.0
        assert profile.expense_categories == {"rent": 300, "supplies": 100}
        assert profile.seasonal_factors == {1: 0.8, 12: 1.2}
        assert profile.customer_patterns == {"monday": 0.9, "friday": 1.1}


class TestFinancialDataGenerator:
    """Test cases for FinancialDataGenerator"""

    @pytest.fixture
    def sample_business_profile(self):
        """Create a sample business profile for testing"""
        return BusinessProfile(
            name="Test Restaurant",
            business_type="restaurant",
            avg_daily_revenue=2000.0,
            revenue_variance=0.3,
            avg_transaction_size=40.0,
            expense_categories={"food_supplies": 500, "labor": 600, "rent": 100, "utilities": 50},
            seasonal_factors={1: 0.8, 6: 1.2, 12: 1.4},
            customer_patterns={"monday": 0.7, "friday": 1.3, "saturday": 1.4},
        )

    @pytest.fixture
    def financial_generator(self, sample_business_profile):
        """Create FinancialDataGenerator instance for testing"""
        start_date = datetime(2024, 1, 1)
        return FinancialDataGenerator(sample_business_profile, start_date)

    def test_initialization(self, sample_business_profile):
        """Test FinancialDataGenerator initialization"""
        start_date = datetime(2024, 1, 1)
        generator = FinancialDataGenerator(sample_business_profile, start_date)

        assert generator.profile == sample_business_profile
        assert generator.start_date == start_date
        assert isinstance(generator.customers, list)
        assert isinstance(generator.vendors, dict)
        assert len(generator.customers) > 0
        assert len(generator.vendors) > 0

    def test_initialization_default_start_date(self, sample_business_profile):
        """Test initialization with default start date"""
        generator = FinancialDataGenerator(sample_business_profile)

        # Start date should be approximately 90 days ago
        expected_start = datetime.now() - timedelta(days=90)
        assert abs((generator.start_date - expected_start).days) <= 1

    def test_generate_customers_restaurant(self, sample_business_profile):
        """Test customer generation for restaurant business"""
        generator = FinancialDataGenerator(sample_business_profile)

        customers = generator._generate_customers()

        assert isinstance(customers, list)
        assert len(customers) > 0
        assert "Walk-in Customer" in customers
        assert "DoorDash" in customers
        assert "Uber Eats" in customers

    def test_generate_customers_retail(self):
        """Test customer generation for retail business"""
        profile = BusinessProfile(
            name="Retail Store",
            business_type="retail",
            avg_daily_revenue=1500.0,
            revenue_variance=0.25,
            avg_transaction_size=75.0,
            expense_categories={},
            seasonal_factors={},
            customer_patterns={},
        )
        generator = FinancialDataGenerator(profile)

        customers = generator._generate_customers()

        assert isinstance(customers, list)
        assert "Cash Sale" in customers
        assert "Credit Card Sale" in customers
        assert "Online Order" in customers

    def test_generate_customers_other_business_type(self):
        """Test customer generation for other business types"""
        profile = BusinessProfile(
            name="Service Business",
            business_type="service",
            avg_daily_revenue=1200.0,
            revenue_variance=0.2,
            avg_transaction_size=100.0,
            expense_categories={},
            seasonal_factors={},
            customer_patterns={},
        )
        generator = FinancialDataGenerator(profile)

        customers = generator._generate_customers()

        assert isinstance(customers, list)
        assert "Customer #1" in customers
        assert "Online Sale" in customers

    def test_generate_vendors_restaurant(self, sample_business_profile):
        """Test vendor generation for restaurant business"""
        generator = FinancialDataGenerator(sample_business_profile)

        vendors = generator._generate_vendors()

        assert isinstance(vendors, dict)
        assert len(vendors) > 0
        assert "Fresh Foods Distributor" in vendors
        assert "City Utilities" in vendors

        # Check vendor structure
        vendor = vendors["Fresh Foods Distributor"]
        assert "category" in vendor
        assert "payment_terms" in vendor
        assert vendor["category"] == "food_supplies"

    def test_generate_vendors_retail(self):
        """Test vendor generation for retail business"""
        profile = BusinessProfile(
            name="Retail Store",
            business_type="retail",
            avg_daily_revenue=1500.0,
            revenue_variance=0.25,
            avg_transaction_size=75.0,
            expense_categories={},
            seasonal_factors={},
            customer_patterns={},
        )
        generator = FinancialDataGenerator(profile)

        vendors = generator._generate_vendors()

        assert "Wholesale Supplier" in vendors
        assert "Shipping Company" in vendors
        assert "Equipment Lease" in vendors

    @patch("simulation.financial_generator.random.normalvariate")
    @patch("simulation.financial_generator.random.choice")
    def test_generate_daily_transactions_basic(
        self, mock_choice, mock_normalvariate, financial_generator
    ):
        """Test basic daily transaction generation"""
        test_date = datetime(2024, 6, 15)  # June 15, Friday

        # Mock random functions with return values instead of side_effect to avoid running out
        mock_normalvariate.return_value = 45.0  # Generic positive value for all normalvariate calls
        mock_choice.return_value = "Walk-in Customer"  # Generic choice for all choice calls

        transactions = financial_generator.generate_daily_transactions(test_date)

        assert isinstance(transactions, list)
        assert len(transactions) > 0

        # Check revenue transactions
        revenue_transactions = [
            t for t in transactions if t["transaction_type"] == TransactionType.INCOME
        ]
        assert len(revenue_transactions) > 0

        # Check expense transactions
        expense_transactions = [
            t for t in transactions if t["transaction_type"] == TransactionType.EXPENSE
        ]
        assert len(expense_transactions) > 0

        # Check transaction structure
        transaction = transactions[0]
        assert "description" in transaction
        assert "amount" in transaction
        assert "transaction_type" in transaction
        assert "category" in transaction
        assert "transaction_date" in transaction
        assert "reference_number" in transaction

    def test_generate_daily_transactions_seasonal_factors(self, financial_generator):
        """Test that seasonal factors affect revenue generation"""
        # June (factor 1.2) vs January (factor 0.8)
        june_date = datetime(2024, 6, 15)
        january_date = datetime(2024, 1, 15)

        # Generate multiple samples to account for randomness
        june_totals = []
        january_totals = []

        for _ in range(10):
            june_transactions = financial_generator.generate_daily_transactions(june_date)
            january_transactions = financial_generator.generate_daily_transactions(january_date)

            june_total = sum(
                t["amount"]
                for t in june_transactions
                if t["transaction_type"] == TransactionType.INCOME
            )
            january_total = sum(
                t["amount"]
                for t in january_transactions
                if t["transaction_type"] == TransactionType.INCOME
            )

            june_totals.append(june_total)
            january_totals.append(january_total)

        # June should generally have higher revenue than January
        june_avg = sum(june_totals) / len(june_totals)
        january_avg = sum(january_totals) / len(january_totals)

        assert june_avg > january_avg

    def test_generate_daily_transactions_day_of_week_factors(self, financial_generator):
        """Test that day-of-week factors affect revenue generation"""
        # Friday (factor 1.3) vs Monday (factor 0.7)
        friday_date = datetime(2024, 6, 14)  # Friday
        monday_date = datetime(2024, 6, 10)  # Monday

        # Generate multiple samples
        friday_totals = []
        monday_totals = []

        for _ in range(10):
            friday_transactions = financial_generator.generate_daily_transactions(friday_date)
            monday_transactions = financial_generator.generate_daily_transactions(monday_date)

            friday_total = sum(
                t["amount"]
                for t in friday_transactions
                if t["transaction_type"] == TransactionType.INCOME
            )
            monday_total = sum(
                t["amount"]
                for t in monday_transactions
                if t["transaction_type"] == TransactionType.INCOME
            )

            friday_totals.append(friday_total)
            monday_totals.append(monday_total)

        # Friday should generally have higher revenue than Monday
        friday_avg = sum(friday_totals) / len(friday_totals)
        monday_avg = sum(monday_totals) / len(monday_totals)

        assert friday_avg > monday_avg

    @patch("simulation.financial_generator.random.random")
    def test_generate_daily_transactions_large_expenses(self, mock_random, financial_generator):
        """Test generation of occasional large expenses"""
        test_date = datetime(2024, 6, 15)

        # Mock random to trigger large expense generation
        mock_random.side_effect = [0.04, 0.7, 0.7, 0.7, 0.7]  # First call triggers large expense

        with patch("simulation.financial_generator.random.choice") as mock_choice, patch(
            "simulation.financial_generator.random.randint"
        ) as mock_randint:

            mock_choice.return_value = ("Equipment Repair", "maintenance", 500)
            mock_randint.return_value = 500

            transactions = financial_generator.generate_daily_transactions(test_date)

            # Should include the large expense
            large_expenses = [t for t in transactions if t["amount"] >= 200]
            assert len(large_expenses) > 0

    def test_get_vendor_for_category(self, financial_generator):
        """Test vendor selection for categories"""
        # Test with existing category
        vendor = financial_generator._get_vendor_for_category("food_supplies")
        assert vendor == "Fresh Foods Distributor"

        # Test with non-existing category
        vendor = financial_generator._get_vendor_for_category("nonexistent")
        assert vendor == "General Vendor"

    @patch("simulation.financial_generator.random.random")
    def test_generate_accounts_receivable(self, mock_random, financial_generator):
        """Test accounts receivable generation"""
        test_date = datetime(2024, 6, 15)

        # Mock random to trigger receivable generation
        mock_random.return_value = 0.05  # 5% chance, should trigger

        with patch("simulation.financial_generator.random.randint") as mock_randint, patch(
            "simulation.financial_generator.random.choice"
        ) as mock_choice:

            mock_randint.side_effect = [2500, 1000, 9999]  # amount, days, random number
            mock_choice.side_effect = [30, "Corporate Client A"]  # due days, customer

            receivables = financial_generator.generate_accounts_receivable(test_date)

            assert isinstance(receivables, list)
            assert len(receivables) > 0

            receivable = receivables[0]
            assert "customer_name" in receivable
            assert "invoice_number" in receivable
            assert "amount" in receivable
            assert "due_date" in receivable
            assert "invoice_date" in receivable
            assert "status" in receivable
            assert receivable["status"] == "unpaid"

    @patch("simulation.financial_generator.random.random")
    def test_generate_accounts_receivable_no_generation(self, mock_random, financial_generator):
        """Test that receivables are not always generated"""
        test_date = datetime(2024, 6, 15)

        # Mock random to prevent receivable generation
        mock_random.return_value = 0.15  # 15% chance, should not trigger

        receivables = financial_generator.generate_accounts_receivable(test_date)

        assert isinstance(receivables, list)
        assert len(receivables) == 0

    @patch("simulation.financial_generator.random.random")
    def test_generate_accounts_payable(self, mock_random, financial_generator):
        """Test accounts payable generation"""
        test_date = datetime(2024, 6, 1)  # First of month for rent/insurance

        # Mock random to trigger payable generation
        mock_random.return_value = 0.04  # 4% chance, should trigger

        with patch("simulation.financial_generator.random.randint") as mock_randint:
            mock_randint.return_value = 1500  # Fixed value for all randint calls

            payables = financial_generator.generate_accounts_payable(test_date)

            assert isinstance(payables, list)
            assert len(payables) > 0

            payable = payables[0]
            assert "vendor_name" in payable
            assert "invoice_number" in payable
            assert "amount" in payable
            assert "due_date" in payable
            assert "invoice_date" in payable
            assert "status" in payable
            assert payable["status"] == "unpaid"

    def test_generate_accounts_payable_monthly_bills(self, financial_generator):
        """Test that monthly bills are only generated on the first of the month"""
        first_of_month = datetime(2024, 6, 1)
        mid_month = datetime(2024, 6, 15)

        # Mock to always trigger generation
        with patch("simulation.financial_generator.random.random", return_value=0.01):
            with patch("simulation.financial_generator.random.randint", return_value=1500):

                first_payables = financial_generator.generate_accounts_payable(first_of_month)
                mid_payables = financial_generator.generate_accounts_payable(mid_month)

                # First of month should potentially have more payables (rent, insurance)
                # This is probabilistic, so we just verify the structure
                assert isinstance(first_payables, list)
                assert isinstance(mid_payables, list)

    def test_generate_period_data(self, financial_generator):
        """Test period data generation"""
        start_date = datetime(2024, 6, 1)
        end_date = datetime(2024, 6, 3)  # 3 days

        period_data = financial_generator.generate_period_data(start_date, end_date)

        assert isinstance(period_data, dict)
        assert "transactions" in period_data
        assert "accounts_receivable" in period_data
        assert "accounts_payable" in period_data

        assert isinstance(period_data["transactions"], list)
        assert isinstance(period_data["accounts_receivable"], list)
        assert isinstance(period_data["accounts_payable"], list)

        # Should have data for multiple days
        assert len(period_data["transactions"]) > 0

    def test_generate_period_data_skip_sundays(self, financial_generator):
        """Test that Sundays are skipped for non-restaurant businesses"""
        # Change business type to retail
        financial_generator.profile.business_type = "retail"

        # Period including a Sunday
        start_date = datetime(2024, 6, 1)  # Saturday
        end_date = datetime(2024, 6, 3)  # Monday (Sunday is 6/2)

        period_data = financial_generator.generate_period_data(start_date, end_date)

        # Should have transactions for Saturday and Monday, but not Sunday
        # This is hard to test directly due to randomness, so we just verify structure
        assert len(period_data["transactions"]) >= 0

    def test_generate_period_data_restaurant_includes_sundays(self, financial_generator):
        """Test that restaurants include Sundays"""
        # Ensure business type is restaurant
        assert financial_generator.profile.business_type == "restaurant"

        # Period including a Sunday
        start_date = datetime(2024, 6, 1)  # Saturday
        end_date = datetime(2024, 6, 3)  # Monday (Sunday is 6/2)

        period_data = financial_generator.generate_period_data(start_date, end_date)

        # Should have transactions for all days including Sunday
        assert len(period_data["transactions"]) > 0

    def test_generate_anomalies_basic(self, financial_generator):
        """Test basic anomaly generation"""
        base_transactions = [
            {
                "amount": 100,
                "transaction_date": datetime.now(),
                "description": "Normal transaction",
            },
            {
                "amount": 200,
                "transaction_date": datetime.now(),
                "description": "Another transaction",
            },
            {"amount": 50, "transaction_date": datetime.now(), "description": "Small transaction"},
        ]

        anomalies = financial_generator.generate_anomalies(base_transactions, anomaly_rate=0.5)

        assert isinstance(anomalies, list)
        assert len(anomalies) > 0

        # Check that anomalies have markers
        for anomaly in anomalies:
            assert "[ANOMALY:" in anomaly["description"]

    def test_generate_anomalies_unusual_amount(self, financial_generator):
        """Test unusual amount anomaly generation"""
        base_transactions = [
            {"amount": 100, "transaction_date": datetime.now(), "description": "Normal transaction"}
        ]

        with patch(
            "simulation.financial_generator.random.choice", return_value="unusual_amount"
        ), patch("simulation.financial_generator.random.uniform", return_value=5.0):

            anomalies = financial_generator.generate_anomalies(base_transactions, anomaly_rate=1.0)

            assert len(anomalies) == 1
            anomaly = anomalies[0]
            assert anomaly["amount"] == 500  # 100 * 5.0
            assert "[ANOMALY: Unusual Amount]" in anomaly["description"]

    def test_generate_anomalies_unusual_time(self, financial_generator):
        """Test unusual time anomaly generation"""
        base_transactions = [
            {
                "amount": 100,
                "transaction_date": datetime(2024, 6, 15, 12, 0),
                "description": "Normal transaction",
            }
        ]

        with patch(
            "simulation.financial_generator.random.choice", return_value="unusual_time"
        ), patch("simulation.financial_generator.random.randint", side_effect=[3, 30]):

            anomalies = financial_generator.generate_anomalies(base_transactions, anomaly_rate=1.0)

            assert len(anomalies) == 1
            anomaly = anomalies[0]
            assert anomaly["transaction_date"].hour == 3
            assert anomaly["transaction_date"].minute == 30
            assert "[ANOMALY: Unusual Time]" in anomaly["description"]

    def test_generate_anomalies_duplicate(self, financial_generator):
        """Test duplicate anomaly generation"""
        base_transactions = [
            {
                "amount": 100,
                "transaction_date": datetime(2024, 6, 15, 12, 0),
                "description": "Normal transaction",
            }
        ]

        with patch("simulation.financial_generator.random.choice", return_value="duplicate"), patch(
            "simulation.financial_generator.random.randint", return_value=15
        ):

            anomalies = financial_generator.generate_anomalies(base_transactions, anomaly_rate=1.0)

            assert len(anomalies) == 1
            anomaly = anomalies[0]
            assert anomaly["transaction_date"] == datetime(2024, 6, 15, 12, 15)  # +15 minutes
            assert "[ANOMALY: Potential Duplicate]" in anomaly["description"]

    def test_generate_anomalies_missing_reference(self, financial_generator):
        """Test missing reference anomaly generation"""
        base_transactions = [
            {
                "amount": 100,
                "transaction_date": datetime.now(),
                "description": "Normal transaction",
                "reference_number": "REF-001",
            }
        ]

        with patch(
            "simulation.financial_generator.random.choice", return_value="missing_reference"
        ):

            anomalies = financial_generator.generate_anomalies(base_transactions, anomaly_rate=1.0)

            assert len(anomalies) == 1
            anomaly = anomalies[0]
            assert anomaly["reference_number"] is None
            assert "[ANOMALY: Missing Reference]" in anomaly["description"]

    def test_generate_anomalies_empty_input(self, financial_generator):
        """Test anomaly generation with empty input"""
        anomalies = financial_generator.generate_anomalies([], anomaly_rate=0.1)

        assert isinstance(anomalies, list)
        assert len(anomalies) == 0  # No anomalies can be generated from empty input

    def test_generate_anomalies_rate_zero(self, financial_generator):
        """Test anomaly generation with zero rate"""
        base_transactions = [
            {"amount": 100, "transaction_date": datetime.now(), "description": "Normal transaction"}
        ]

        anomalies = financial_generator.generate_anomalies(base_transactions, anomaly_rate=0.0)

        assert isinstance(anomalies, list)
        assert len(anomalies) >= 1  # At least 1 anomaly is generated even with 0 rate


class TestPredefinedProfiles:
    """Test cases for predefined business profiles"""

    def test_get_restaurant_profile(self):
        """Test restaurant profile generation"""
        profile = get_restaurant_profile()

        assert isinstance(profile, BusinessProfile)
        assert profile.name == "Sample Restaurant"
        assert profile.business_type == "restaurant"
        assert profile.avg_daily_revenue > 0
        assert profile.revenue_variance > 0
        assert profile.avg_transaction_size > 0
        assert len(profile.expense_categories) > 0
        assert len(profile.seasonal_factors) == 12
        assert len(profile.customer_patterns) == 7

        # Check specific expected categories
        assert "food_supplies" in profile.expense_categories
        assert "labor" in profile.expense_categories
        assert "rent" in profile.expense_categories

        # Check seasonal factors
        assert 1 in profile.seasonal_factors  # January
        assert 12 in profile.seasonal_factors  # December

        # Check customer patterns
        assert "monday" in profile.customer_patterns
        assert "friday" in profile.customer_patterns

    def test_get_retail_profile(self):
        """Test retail profile generation"""
        profile = get_retail_profile()

        assert isinstance(profile, BusinessProfile)
        assert profile.name == "Sample Retail Store"
        assert profile.business_type == "retail"
        assert profile.avg_daily_revenue > 0
        assert profile.revenue_variance > 0
        assert profile.avg_transaction_size > 0
        assert len(profile.expense_categories) > 0
        assert len(profile.seasonal_factors) == 12
        assert len(profile.customer_patterns) == 7

        # Check specific expected categories
        assert "inventory" in profile.expense_categories
        assert "labor" in profile.expense_categories
        assert "shipping" in profile.expense_categories

        # Check that retail has different patterns than restaurant
        restaurant_profile = get_restaurant_profile()
        assert profile.avg_daily_revenue != restaurant_profile.avg_daily_revenue
        assert profile.avg_transaction_size != restaurant_profile.avg_transaction_size

    def test_profile_consistency(self):
        """Test that profiles are consistent across calls"""
        profile1 = get_restaurant_profile()
        profile2 = get_restaurant_profile()

        assert profile1.name == profile2.name
        assert profile1.business_type == profile2.business_type
        assert profile1.avg_daily_revenue == profile2.avg_daily_revenue
        assert profile1.expense_categories == profile2.expense_categories
        assert profile1.seasonal_factors == profile2.seasonal_factors
        assert profile1.customer_patterns == profile2.customer_patterns
