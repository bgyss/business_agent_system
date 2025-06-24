"""Database Performance Tests.

Tests for measuring and benchmarking database query performance,
insert/update operations, and optimization opportunities under various
load conditions.
"""

import os

# Add parent directories to path for imports
import sys
import time
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.agent_decisions import AgentDecisionModel
from models.employee import Employee, TimeRecord
from models.financial import Transaction, TransactionType
from models.inventory import Item, StockMovement


class TestDatabasePerformance:
    """Test suite for database performance benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_transaction_insert_performance(self, benchmark, test_db_session):
        """Benchmark transaction insertion performance."""

        def insert_transactions():
            transactions = []
            for i in range(100):
                transaction = Transaction(
                    id=f"perf_insert_tx_{i}",
                    amount=100.0 + i,
                    transaction_type=TransactionType.INCOME if i % 2 else TransactionType.EXPENSE,
                    transaction_date=datetime.now() - timedelta(days=i % 30),
                    description=f"Performance test transaction {i}",
                    account_id="test_checking",
                    category="performance_test",
                )
                transactions.append(transaction)

            test_db_session.add_all(transactions)
            test_db_session.commit()
            return len(transactions)

        result = benchmark(insert_transactions)

        # Should be able to insert 100 transactions quickly
        assert result == 100
        assert benchmark.stats["mean"] < 0.5  # Less than 500ms

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_transaction_query_performance(self, benchmark, medium_dataset):
        """Benchmark transaction query performance with medium dataset."""

        def query_recent_transactions():
            return (
                medium_dataset.query(Transaction)
                .filter(Transaction.transaction_date >= datetime.now() - timedelta(days=30))
                .order_by(Transaction.transaction_date.desc())
                .limit(100)
                .all()
            )

        results = benchmark(query_recent_transactions)

        # Query should complete quickly
        assert len(results) > 0
        assert benchmark.stats["mean"] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_transaction_aggregation_performance(self, benchmark, medium_dataset):
        """Benchmark transaction aggregation queries."""

        def aggregate_transactions():
            # Revenue by category
            revenue_by_category = (
                medium_dataset.query(
                    Transaction.category, func.sum(Transaction.amount).label("total")
                )
                .filter(Transaction.transaction_type == TransactionType.INCOME)
                .group_by(Transaction.category)
                .all()
            )

            # Monthly totals
            monthly_totals = (
                medium_dataset.query(
                    func.strftime("%Y-%m", Transaction.transaction_date).label("month"),
                    func.sum(Transaction.amount).label("total"),
                )
                .group_by(func.strftime("%Y-%m", Transaction.transaction_date))
                .all()
            )

            return len(revenue_by_category), len(monthly_totals)

        result = benchmark(aggregate_transactions)

        # Aggregation should complete reasonably fast
        assert result[0] > 0  # Should have categories
        assert result[1] > 0  # Should have months
        assert benchmark.stats["mean"] < 0.2  # Less than 200ms

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_inventory_query_performance(self, benchmark, medium_dataset):
        """Benchmark inventory-related query performance."""

        def inventory_queries():
            # Low stock items
            low_stock = (
                medium_dataset.query(Item).filter(Item.current_stock <= Item.reorder_point).all()
            )

            # Stock movements in last week
            week_ago = datetime.now() - timedelta(days=7)
            recent_movements = (
                medium_dataset.query(StockMovement)
                .filter(StockMovement.movement_date >= week_ago)
                .order_by(StockMovement.movement_date.desc())
                .all()
            )

            # Inventory value
            total_value = medium_dataset.query(
                func.sum(Item.current_stock * Item.unit_cost)
            ).scalar()

            return len(low_stock), len(recent_movements), total_value or 0

        result = benchmark(inventory_queries)

        # Inventory queries should be fast
        assert result[2] >= 0  # Total value should be non-negative
        assert benchmark.stats["mean"] < 0.15  # Less than 150ms

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_employee_query_performance(self, benchmark, medium_dataset):
        """Benchmark employee and HR-related query performance."""

        def employee_queries():
            # Active employees
            active_employees = (
                medium_dataset.query(Employee).filter(Employee.status == "active").all()
            )

            # Recent time records
            week_ago = datetime.now() - timedelta(days=7)
            recent_time_records = (
                medium_dataset.query(TimeRecord).filter(TimeRecord.timestamp >= week_ago).all()
            )

            # Payroll calculations
            payroll_total = (
                medium_dataset.query(func.sum(Employee.hourly_rate * 40))  # Assuming 40 hours
                .filter(Employee.is_full_time)
                .scalar()
            )

            return len(active_employees), len(recent_time_records), payroll_total or 0

        result = benchmark(employee_queries)

        # HR queries should be fast
        assert result[2] >= 0  # Payroll total should be non-negative
        assert benchmark.stats["mean"] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_decision_log_query_performance(self, benchmark, test_db_session):
        """Benchmark agent decision log query performance."""
        # First, create some decision log entries
        decisions = []
        for i in range(200):
            decision = AgentDecisionModel(
                agent_id=f"test_agent_{i % 3}",
                decision_type="performance_test",
                context=f'{{"test": {i}}}',
                reasoning=f"Performance test reasoning {i}",
                action=f"Test action {i}",
                confidence=0.8 + (i % 20) / 100,
                timestamp=datetime.now() - timedelta(minutes=i),
            )
            decisions.append(decision)

        test_db_session.add_all(decisions)
        test_db_session.commit()

        def query_recent_decisions():
            return (
                test_db_session.query(AgentDecisionModel)
                .order_by(AgentDecisionModel.timestamp.desc())
                .limit(50)
                .all()
            )

        result = benchmark(query_recent_decisions)

        # Decision queries should be fast
        assert len(result) > 0
        assert benchmark.stats["mean"] < 0.05  # Less than 50ms

    @pytest.mark.database
    @pytest.mark.stress
    def test_large_dataset_query_performance(self, large_dataset):
        """Test query performance with large dataset."""
        start_time = time.time()

        # Complex query with joins and aggregations
        results = (
            large_dataset.query(
                Transaction.category,
                func.count(Transaction.id).label("count"),
                func.sum(Transaction.amount).label("total"),
                func.avg(Transaction.amount).label("average"),
            )
            .filter(Transaction.transaction_date >= datetime.now() - timedelta(days=365))
            .group_by(Transaction.category)
            .order_by(func.sum(Transaction.amount).desc())
            .all()
        )

        end_time = time.time()
        query_time = end_time - start_time

        # Even with large dataset, complex queries should complete in reasonable time
        assert len(results) > 0
        assert query_time < 2.0, f"Complex query took {query_time:.2f}s on large dataset"

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_batch_insert_performance(self, benchmark, test_db_session):
        """Benchmark batch insert performance."""

        def batch_insert():
            # Test batch insert of various record types
            transactions = []
            items = []
            employees = []

            for i in range(50):
                # Transactions
                transaction = Transaction(
                    id=f"batch_tx_{i}",
                    amount=50.0 + i,
                    transaction_type=TransactionType.INCOME if i % 2 else TransactionType.EXPENSE,
                    transaction_date=datetime.now() - timedelta(days=i),
                    description=f"Batch test transaction {i}",
                    account_id="test_checking",
                    category="batch_test",
                )
                transactions.append(transaction)

                # Items
                item = Item(
                    sku=f"BATCH_ITEM_{i}",
                    name=f"Batch Test Item {i}",
                    category="batch_test",
                    unit_cost=10.0 + i,
                    selling_price=20.0 + i,
                    current_stock=100,
                    reorder_point=20,
                    reorder_quantity=50,
                )
                items.append(item)

                # Employees (fewer than other records)
                if i < 10:
                    employee = Employee(
                        employee_id=f"BATCH_EMP_{i}",
                        first_name=f"BatchTest{i}",
                        last_name="Employee",
                        email=f"batchtest{i}@example.com",
                        hire_date=datetime.now().date() - timedelta(days=30),
                        position="Test Position",
                        department="Test Department",
                        hourly_rate=15.0,
                        is_full_time=True,
                    )
                    employees.append(employee)

            # Add all records
            test_db_session.add_all(transactions)
            test_db_session.add_all(items)
            test_db_session.add_all(employees)
            test_db_session.commit()

            return len(transactions) + len(items) + len(employees)

        result = benchmark(batch_insert)

        # Batch insert should be efficient
        assert result == 110  # 50 + 50 + 10
        assert benchmark.stats["mean"] < 1.0  # Less than 1 second

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_update_performance(self, benchmark, small_dataset):
        """Benchmark update operation performance."""

        def update_records():
            # Update transaction amounts
            updated_count = (
                small_dataset.query(Transaction)
                .filter(Transaction.category == "test_category")
                .update({Transaction.amount: Transaction.amount * 1.1}, synchronize_session=False)
            )

            # Update item stock levels
            updated_items = (
                small_dataset.query(Item)
                .filter(Item.category == "test_category")
                .update({Item.current_stock: Item.current_stock - 1}, synchronize_session=False)
            )

            small_dataset.commit()

            return updated_count, updated_items

        result = benchmark(update_records)

        # Updates should be fast
        assert result[0] > 0  # Should have updated some transactions
        assert benchmark.stats["mean"] < 0.1  # Less than 100ms

    @pytest.mark.database
    @pytest.mark.stress
    def test_concurrent_database_access(self, test_db_engine):
        """Test database performance under concurrent access."""
        import queue
        import threading

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def database_worker(worker_id: int):
            """Worker function for concurrent database access."""
            try:
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
                session = SessionLocal()

                start_time = time.time()

                # Perform various database operations
                for i in range(20):
                    # Insert a transaction
                    transaction = Transaction(
                        id=f"concurrent_{worker_id}_{i}",
                        amount=100.0 + i,
                        transaction_type=TransactionType.INCOME,
                        transaction_date=datetime.now(),
                        description=f"Concurrent test {worker_id}_{i}",
                        account_id="test_checking",
                        category="concurrent_test",
                    )
                    session.add(transaction)

                    # Query existing transactions
                    if i % 5 == 0:
                        (
                            session.query(Transaction)
                            .filter(Transaction.category == "concurrent_test")
                            .count()
                        )

                session.commit()
                end_time = time.time()

                results_queue.put(
                    {"worker_id": worker_id, "duration": end_time - start_time, "operations": 20}
                )

                session.close()

            except Exception as e:
                error_queue.put({"worker_id": worker_id, "error": str(e)})

        # Start multiple worker threads
        threads = []
        num_workers = 5

        start_time = time.time()

        for worker_id in range(num_workers):
            thread = threading.Thread(target=database_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        end_time = time.time()
        total_duration = end_time - start_time

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())

        # Assertions
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        assert total_duration < 15, f"Concurrent operations took too long: {total_duration:.2f}s"

        # Calculate throughput
        total_operations = sum(r["operations"] for r in results)
        throughput = total_operations / total_duration
        assert throughput > 5, f"Low concurrent throughput: {throughput:.2f} ops/sec"

    @pytest.mark.benchmark
    @pytest.mark.database
    def test_index_performance_comparison(self, benchmark, test_db_engine):
        """Compare query performance with and without indexes."""
        # This test would ideally create tables with and without indexes
        # and compare performance, but for simplicity we'll test existing indexes

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
        session = SessionLocal()

        # Create test data
        transactions = []
        for i in range(1000):
            transaction = Transaction(
                id=f"index_test_tx_{i}",
                amount=100.0 + i,
                transaction_type=TransactionType.INCOME if i % 2 else TransactionType.EXPENSE,
                transaction_date=datetime.now() - timedelta(days=i % 100),
                description=f"Index test transaction {i}",
                account_id="test_checking",
                category=f"category_{i % 20}",
            )
            transactions.append(transaction)

        session.add_all(transactions)
        session.commit()

        def query_with_date_filter():
            # This should benefit from date index if it exists
            return (
                session.query(Transaction)
                .filter(Transaction.transaction_date >= datetime.now() - timedelta(days=30))
                .count()
            )

        result = benchmark(query_with_date_filter)

        # Date-filtered queries should be fast
        assert result > 0
        assert benchmark.stats["mean"] < 0.1  # Less than 100ms

        session.close()

    @pytest.mark.database
    @pytest.mark.memory
    def test_database_memory_usage(self, large_dataset):
        """Test database memory usage with large queries."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform memory-intensive database operations
        start_time = time.time()

        # Large result set query
        all_transactions = large_dataset.query(Transaction).all()

        # Complex aggregation
        category_stats = (
            large_dataset.query(
                Transaction.category,
                func.count(Transaction.id).label("count"),
                func.sum(Transaction.amount).label("total"),
                func.min(Transaction.amount).label("min_amount"),
                func.max(Transaction.amount).label("max_amount"),
                func.avg(Transaction.amount).label("avg_amount"),
            )
            .group_by(Transaction.category)
            .all()
        )

        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = final_memory - initial_memory
        duration = end_time - start_time

        # Memory usage should be reasonable
        assert len(all_transactions) > 1000  # Should have substantial data
        assert len(category_stats) > 0
        assert memory_increase < 200, f"Excessive memory usage: {memory_increase:.2f}MB"
        assert duration < 5, f"Queries took too long: {duration:.2f}s"
