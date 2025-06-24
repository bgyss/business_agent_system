"""
Performance test configuration and fixtures.

Provides shared fixtures and utilities for performance testing including:
- Database setup with various data sizes
- Agent instances for testing
- Benchmark configuration
- Memory profiling utilities
"""

import os

# Add parent directories to path for imports
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.accounting_agent import AccountingAgent
from agents.hr_agent import HRAgent
from agents.inventory_agent import InventoryAgent
from models.employee import Base as EmployeeBase
from models.employee import Employee
from models.financial import Account
from models.financial import Base as FinancialBase
from models.financial import Transaction, TransactionType
from models.inventory import Base as InventoryBase
from models.inventory import Item
from simulation.business_simulator import BusinessSimulator


@pytest.fixture(scope="session")
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture(scope="session")
def performance_config() -> Dict[str, Any]:
    """Performance test configuration."""
    return {
        "business": {"name": "Performance Test Business", "type": "restaurant"},
        "database": {"url": "sqlite:///performance_test.db"},
        "agents": {
            "accounting": {
                "enabled": True,
                "check_interval": 60,
                "cash_low_threshold": 1000,
                "anomaly_threshold": 0.25,
            },
            "inventory": {"enabled": True, "check_interval": 300, "low_stock_threshold": 10},
            "hr": {"enabled": True, "check_interval": 3600},
        },
        "simulation": {"speed_multiplier": 1.0, "simulation_interval": 10},
    }


@pytest.fixture
def test_db_engine(temp_db_path):
    """Create a test database engine."""
    db_url = f"sqlite:///{temp_db_path}"
    engine = create_engine(db_url)

    # Create all tables
    FinancialBase.metadata.create_all(bind=engine)
    InventoryBase.metadata.create_all(bind=engine)
    EmployeeBase.metadata.create_all(bind=engine)

    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = SessionLocal()

    yield session

    session.close()


@pytest.fixture
def small_dataset(test_db_session):
    """Create a small dataset for basic performance testing (100 records)."""
    # Create test accounts
    accounts = [
        Account(id="test_checking", name="Test Checking", account_type="checking", balance=10000.0),
        Account(id="test_revenue", name="Test Revenue", account_type="revenue", balance=0.0),
        Account(id="test_expense", name="Test Expenses", account_type="expense", balance=0.0),
    ]

    for account in accounts:
        test_db_session.add(account)

    # Create test transactions
    base_date = datetime.now() - timedelta(days=30)
    for i in range(100):
        transaction = Transaction(
            id=f"tx_{i}",
            amount=100.0 + (i * 5),
            transaction_type=TransactionType.INCOME if i % 2 == 0 else TransactionType.EXPENSE,
            transaction_date=base_date + timedelta(days=i % 30),
            description=f"Test transaction {i}",
            account_id="test_checking",
            category="test_category",
        )
        test_db_session.add(transaction)

    # Create test items
    for i in range(20):
        item = Item(
            sku=f"ITEM_{i:03d}",
            name=f"Test Item {i}",
            category="test_category",
            unit_cost=10.0 + i,
            selling_price=20.0 + i,
            current_stock=100 - i,
            reorder_point=20,
            reorder_quantity=50,
        )
        test_db_session.add(item)

    # Create test employees
    for i in range(5):
        employee = Employee(
            employee_id=f"EMP_{i:03d}",
            first_name=f"Test{i}",
            last_name="Employee",
            email=f"test{i}@example.com",
            hire_date=datetime.now().date() - timedelta(days=365),
            position="Test Position",
            department="Test Department",
            hourly_rate=15.0 + i,
            is_full_time=i % 2 == 0,
        )
        test_db_session.add(employee)

    test_db_session.commit()
    return test_db_session


@pytest.fixture
def medium_dataset(test_db_session):
    """Create a medium dataset for load testing (10,000 records)."""
    # Create test accounts
    accounts = [
        Account(
            id="med_checking", name="Medium Checking", account_type="checking", balance=100000.0
        ),
        Account(id="med_revenue", name="Medium Revenue", account_type="revenue", balance=0.0),
        Account(id="med_expense", name="Medium Expenses", account_type="expense", balance=0.0),
    ]

    for account in accounts:
        test_db_session.add(account)

    # Create test transactions in batches for performance
    base_date = datetime.now() - timedelta(days=365)
    batch_size = 1000

    for batch in range(0, 10000, batch_size):
        transactions = []
        for i in range(batch, min(batch + batch_size, 10000)):
            transaction = Transaction(
                id=f"med_tx_{i}",
                amount=50.0 + (i % 1000),
                transaction_type=TransactionType.INCOME if i % 3 == 0 else TransactionType.EXPENSE,
                transaction_date=base_date + timedelta(days=i % 365),
                description=f"Medium test transaction {i}",
                account_id="med_checking",
                category=f"category_{i % 10}",
            )
            transactions.append(transaction)

        test_db_session.add_all(transactions)
        test_db_session.commit()

    # Create test items
    items = []
    for i in range(200):
        item = Item(
            sku=f"MED_ITEM_{i:03d}",
            name=f"Medium Test Item {i}",
            category=f"med_category_{i % 5}",
            unit_cost=5.0 + (i % 50),
            selling_price=10.0 + (i % 50),
            current_stock=1000 - (i * 2),
            reorder_point=50,
            reorder_quantity=100,
        )
        items.append(item)

    test_db_session.add_all(items)

    # Create test employees
    employees = []
    for i in range(50):
        employee = Employee(
            employee_id=f"MED_EMP_{i:03d}",
            first_name=f"MedTest{i}",
            last_name="Employee",
            email=f"medtest{i}@example.com",
            hire_date=datetime.now().date() - timedelta(days=365 - i),
            position=f"Position {i % 5}",
            department=f"Department {i % 3}",
            hourly_rate=12.0 + (i % 20),
            is_full_time=i % 3 == 0,
        )
        employees.append(employee)

    test_db_session.add_all(employees)
    test_db_session.commit()
    return test_db_session


@pytest.fixture
def large_dataset(test_db_session):
    """Create a large dataset for stress testing (100,000+ records)."""
    # Create test accounts
    accounts = [
        Account(
            id="large_checking", name="Large Checking", account_type="checking", balance=1000000.0
        ),
        Account(id="large_revenue", name="Large Revenue", account_type="revenue", balance=0.0),
        Account(id="large_expense", name="Large Expenses", account_type="expense", balance=0.0),
    ]

    for account in accounts:
        test_db_session.add(account)

    # Create test transactions in large batches
    base_date = datetime.now() - timedelta(days=1095)  # 3 years
    batch_size = 5000
    total_transactions = 100000

    for batch in range(0, total_transactions, batch_size):
        transactions = []
        for i in range(batch, min(batch + batch_size, total_transactions)):
            transaction = Transaction(
                id=f"large_tx_{i}",
                amount=25.0 + (i % 5000),
                transaction_type=TransactionType.INCOME if i % 4 == 0 else TransactionType.EXPENSE,
                transaction_date=base_date + timedelta(days=i % 1095),
                description=f"Large test transaction {i}",
                account_id="large_checking",
                category=f"large_category_{i % 20}",
            )
            transactions.append(transaction)

        test_db_session.add_all(transactions)
        test_db_session.commit()

    # Create test items
    items = []
    for i in range(1000):
        item = Item(
            sku=f"LARGE_ITEM_{i:04d}",
            name=f"Large Test Item {i}",
            category=f"large_category_{i % 10}",
            unit_cost=2.0 + (i % 100),
            selling_price=5.0 + (i % 100),
            current_stock=10000 - (i * 5),
            reorder_point=100,
            reorder_quantity=500,
        )
        items.append(item)

    test_db_session.add_all(items)

    # Create test employees
    employees = []
    for i in range(500):
        employee = Employee(
            employee_id=f"LARGE_EMP_{i:04d}",
            first_name=f"LargeTest{i}",
            last_name="Employee",
            email=f"largetest{i}@example.com",
            hire_date=datetime.now().date() - timedelta(days=1095 - (i * 2)),
            position=f"Position {i % 10}",
            department=f"Department {i % 5}",
            hourly_rate=10.0 + (i % 30),
            is_full_time=i % 4 == 0,
        )
        employees.append(employee)

    test_db_session.add_all(employees)
    test_db_session.commit()
    return test_db_session


@pytest.fixture
async def test_accounting_agent(performance_config, temp_db_path):
    """Create a test accounting agent."""
    db_url = f"sqlite:///{temp_db_path}"
    agent = AccountingAgent(
        agent_id="test_accounting_agent",
        api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        config=performance_config["agents"]["accounting"],
        db_url=db_url,
    )
    return agent


@pytest.fixture
async def test_inventory_agent(performance_config, temp_db_path):
    """Create a test inventory agent."""
    db_url = f"sqlite:///{temp_db_path}"
    agent = InventoryAgent(
        agent_id="test_inventory_agent",
        api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        config=performance_config["agents"]["inventory"],
        db_url=db_url,
    )
    return agent


@pytest.fixture
async def test_hr_agent(performance_config, temp_db_path):
    """Create a test HR agent."""
    db_url = f"sqlite:///{temp_db_path}"
    agent = HRAgent(
        agent_id="test_hr_agent",
        api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        config=performance_config["agents"]["hr"],
        db_url=db_url,
    )
    return agent


@pytest.fixture
def test_business_simulator(performance_config, temp_db_path):
    """Create a test business simulator."""
    db_url = f"sqlite:///{temp_db_path}"
    simulator = BusinessSimulator(performance_config, db_url)
    simulator.initialize_business("restaurant")
    return simulator


# Benchmark configuration
def pytest_configure(config):
    """Configure pytest-benchmark settings."""
    config.option.benchmark_json = "tests/performance/benchmark_results.json"
    config.option.benchmark_compare_fail = "min:10%"  # Fail if performance degrades by 10%


# Benchmark comparison hook
def pytest_benchmark_compare_machine_info(config, machine_info, compared_benchmark):
    """Hook for comparing machine info in benchmarks."""
    return True  # Always allow comparison for now


# Custom markers for performance tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")
    config.addinivalue_line("markers", "memory: mark test as a memory usage test")
    config.addinivalue_line("markers", "stress: mark test as a stress test")
    config.addinivalue_line("markers", "database: mark test as a database performance test")
    config.addinivalue_line("markers", "agent: mark test as an agent performance test")
    config.addinivalue_line("markers", "simulation: mark test as a simulation performance test")
