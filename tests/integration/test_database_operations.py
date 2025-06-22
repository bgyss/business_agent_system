"""
Integration tests for database operations and data persistence.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.agent_decisions import AgentDecision, AgentDecisionModel
from models.employee import Employee
from models.financial import (
    Account,
    Transaction,
)
from models.inventory import Item, StockMovement, Supplier


class TestDatabaseSchema:
    """Test database schema creation and integrity."""

    def test_all_tables_created(self, temp_db, integration_helper):
        """Test that all expected tables are created."""
        tables = integration_helper.verify_database_tables(temp_db)

        # Verify specific important tables
        expected_tables = [
            "accounts",
            "transactions",
            "accounts_receivable",
            "accounts_payable",
            "items",
            "stock_movements",
            "suppliers",
            "purchase_orders",
            "purchase_order_items",
            "employees",
            "time_records",
            "schedules",
            "agent_decisions",
        ]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    def test_table_relationships(self, temp_db):
        """Test database table relationships and foreign keys."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Test financial data relationships
            account = Account(
                id="test_account", name="Test Account", account_type="checking", balance=1000.00
            )
            session.add(account)
            session.flush()

            transaction = Transaction(
                account_id="test_account",
                amount=Decimal("100.00"),
                transaction_type="credit",
                description="Test transaction",
                transaction_date=datetime.now().date(),
            )
            session.add(transaction)

            # Test inventory relationships
            supplier = Supplier(
                name="Test Supplier",
                contact_person="John Doe",
                email="john@supplier.com",
                phone="555-0123",
                lead_time_days=5,
                payment_terms="Net 30",
            )
            session.add(supplier)
            session.flush()

            item = Item(
                sku="TEST001",
                name="Test Item",
                description="Test inventory item",
                unit_cost=Decimal("10.00"),
                unit_of_measure="each",
                category="test",
                supplier_id=supplier.id,
                reorder_point=10,
                reorder_quantity=50,
                current_stock=25,
            )
            session.add(item)
            session.flush()

            stock_movement = StockMovement(
                item_id=item.id,
                movement_type="adjustment",
                quantity=5,
                unit_cost=Decimal("10.00"),
                timestamp=datetime.now(),
                reference="Test adjustment",
            )
            session.add(stock_movement)

            # Test employee data
            employee = Employee(
                employee_id="TEST001",
                first_name="Test",
                last_name="Employee",
                email="test@company.com",
                hire_date=datetime.now().date(),
                position="Test Position",
                department="Test Dept",
                hourly_rate=Decimal("15.00"),
                is_full_time=True,
            )
            session.add(employee)

            session.commit()

            # Verify relationships work
            retrieved_transaction = (
                session.query(Transaction).filter_by(account_id="test_account").first()
            )
            assert retrieved_transaction is not None
            assert retrieved_transaction.account.name == "Test Account"

            retrieved_item = session.query(Item).filter_by(supplier_id=supplier.id).first()
            assert retrieved_item is not None
            assert retrieved_item.supplier.name == "Test Supplier"

        finally:
            session.close()

    def test_database_constraints(self, temp_db):
        """Test database constraints and data validation."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Test unique constraints
            account1 = Account(
                id="unique_test", name="First Account", account_type="checking", balance=1000.00
            )
            session.add(account1)
            session.commit()

            # Try to add another account with same ID
            account2 = Account(
                id="unique_test",  # Same ID
                name="Second Account",
                account_type="savings",
                balance=2000.00,
            )
            session.add(account2)

            with pytest.raises(Exception):  # Should raise integrity error
                session.commit()

            session.rollback()

            # Test not-null constraints
            invalid_transaction = Transaction(
                # Missing required account_id
                amount=Decimal("100.00"),
                transaction_type="credit",
                description="Invalid transaction",
                transaction_date=datetime.now().date(),
            )
            session.add(invalid_transaction)

            with pytest.raises(Exception):  # Should raise not-null constraint error
                session.commit()

        finally:
            session.close()


class TestDataPersistence:
    """Test data persistence and retrieval operations."""

    def test_financial_data_persistence(self, temp_db):
        """Test financial data CRUD operations."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Create account
            account = Account(
                id="persistence_test",
                name="Persistence Test Account",
                account_type="checking",
                balance=5000.00,
                description="Test account for persistence",
            )
            session.add(account)
            session.commit()

            # Create transactions
            transactions = []
            for i in range(10):
                tx = Transaction(
                    account_id="persistence_test",
                    amount=Decimal(f"{100 + i}.00"),
                    transaction_type="credit" if i % 2 == 0 else "debit",
                    description=f"Test transaction {i}",
                    transaction_date=datetime.now().date() - timedelta(days=i),
                    reference=f"REF{i:03d}",
                )
                transactions.append(tx)
                session.add(tx)

            session.commit()

            # Verify persistence
            retrieved_account = session.query(Account).filter_by(id="persistence_test").first()
            assert retrieved_account is not None
            assert retrieved_account.name == "Persistence Test Account"
            assert retrieved_account.balance == Decimal("5000.00")

            retrieved_transactions = (
                session.query(Transaction).filter_by(account_id="persistence_test").all()
            )
            assert len(retrieved_transactions) == 10

            # Test update
            retrieved_account.balance = Decimal("6000.00")
            session.commit()

            updated_account = session.query(Account).filter_by(id="persistence_test").first()
            assert updated_account.balance == Decimal("6000.00")

            # Test delete
            session.delete(retrieved_account)
            session.commit()

            deleted_account = session.query(Account).filter_by(id="persistence_test").first()
            assert deleted_account is None

        finally:
            session.close()

    def test_inventory_data_persistence(self, temp_db):
        """Test inventory data CRUD operations."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Create supplier
            supplier = Supplier(
                name="Test Inventory Supplier",
                contact_person="Jane Smith",
                email="jane@supplier.com",
                phone="555-9876",
                lead_time_days=7,
                payment_terms="Net 15",
            )
            session.add(supplier)
            session.flush()

            # Create items
            items = []
            for i in range(5):
                item = Item(
                    sku=f"INV{i:03d}",
                    name=f"Inventory Item {i}",
                    description=f"Test inventory item {i}",
                    unit_cost=Decimal(f"{10 + i}.50"),
                    unit_of_measure="each",
                    category="test_category",
                    supplier_id=supplier.id,
                    reorder_point=10 + i,
                    reorder_quantity=50 + i * 10,
                    current_stock=25 + i * 5,
                )
                items.append(item)
                session.add(item)

            session.flush()

            # Create stock movements
            for item in items:
                movement = StockMovement(
                    item_id=item.id,
                    movement_type="delivery",
                    quantity=20,
                    unit_cost=item.unit_cost,
                    timestamp=datetime.now(),
                    reference="Test delivery",
                )
                session.add(movement)

            session.commit()

            # Verify persistence
            retrieved_items = session.query(Item).filter(Item.sku.like("INV%")).all()
            assert len(retrieved_items) == 5

            for item in retrieved_items:
                movements = session.query(StockMovement).filter_by(item_id=item.id).all()
                assert len(movements) >= 1
                assert movements[0].movement_type == "delivery"
                assert movements[0].quantity == 20

        finally:
            session.close()

    def test_agent_decision_persistence(self, temp_db):
        """Test agent decision logging and retrieval."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Create test decisions
            decisions = []
            for i in range(5):
                decision = AgentDecision(
                    agent_id="test_agent",
                    decision_type=f"test_decision_{i % 3}",
                    action=f"Take action {i}",
                    reasoning=f"Reason for action {i}",
                    confidence=0.8 + i * 0.02,
                    context={"test_data": f"value_{i}", "number": i},
                    timestamp=datetime.now() - timedelta(minutes=i),
                )
                decisions.append(decision)

                # Convert to DB model and persist
                db_decision = decision.to_db_model()
                session.add(db_decision)

            session.commit()

            # Verify persistence
            retrieved_decisions = (
                session.query(AgentDecisionModel).filter_by(agent_id="test_agent").all()
            )
            assert len(retrieved_decisions) == 5

            # Test ordering by timestamp
            ordered_decisions = (
                session.query(AgentDecisionModel)
                .filter_by(agent_id="test_agent")
                .order_by(AgentDecisionModel.timestamp.desc())
                .all()
            )

            assert len(ordered_decisions) == 5
            # Most recent should be first
            assert ordered_decisions[0].timestamp >= ordered_decisions[-1].timestamp

            # Test filtering by decision type
            type_filtered = (
                session.query(AgentDecisionModel)
                .filter_by(agent_id="test_agent", decision_type="test_decision_0")
                .all()
            )

            assert len(type_filtered) >= 1

            # Test context serialization
            decision_with_context = ordered_decisions[0]
            assert decision_with_context.context is not None
            assert "test_data" in decision_with_context.context
            assert "number" in decision_with_context.context

            # Test conversion back to Pydantic model
            pydantic_decision = AgentDecision.from_db_model(decision_with_context)
            assert pydantic_decision.agent_id == "test_agent"
            assert pydantic_decision.context["test_data"].startswith("value_")

        finally:
            session.close()

    def test_transaction_querying_and_aggregation(self, temp_db):
        """Test complex querying and aggregation operations."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Create test account
            account = Account(
                id="query_test",
                name="Query Test Account",
                account_type="checking",
                balance=10000.00,
            )
            session.add(account)
            session.flush()

            # Create transactions with different dates and amounts
            base_date = datetime.now().date()
            test_transactions = [
                (100.00, "credit", 0),
                (50.00, "debit", 1),
                (200.00, "credit", 2),
                (75.00, "debit", 3),
                (300.00, "credit", 7),  # Week later
                (25.00, "debit", 7),
                (150.00, "credit", 14),  # Two weeks later
            ]

            for amount, tx_type, days_back in test_transactions:
                tx = Transaction(
                    account_id="query_test",
                    amount=Decimal(str(amount)),
                    transaction_type=tx_type,
                    description=f"Test {tx_type} transaction",
                    transaction_date=base_date - timedelta(days=days_back),
                )
                session.add(tx)

            session.commit()

            # Test date range queries
            week_ago = base_date - timedelta(days=7)
            recent_transactions = (
                session.query(Transaction)
                .filter(
                    Transaction.account_id == "query_test", Transaction.transaction_date >= week_ago
                )
                .all()
            )

            assert len(recent_transactions) >= 4  # Transactions from last week

            # Test aggregation queries
            from sqlalchemy import func

            total_credits = (
                session.query(func.sum(Transaction.amount))
                .filter(
                    Transaction.account_id == "query_test", Transaction.transaction_type == "credit"
                )
                .scalar()
            )

            assert total_credits >= Decimal("750.00")  # Sum of credit transactions

            # Test grouping by transaction type
            type_counts = (
                session.query(
                    Transaction.transaction_type,
                    func.count(Transaction.id),
                    func.sum(Transaction.amount),
                )
                .filter(Transaction.account_id == "query_test")
                .group_by(Transaction.transaction_type)
                .all()
            )

            assert len(type_counts) == 2  # Should have both credit and debit

            type_dict = {tx_type: (count, total) for tx_type, count, total in type_counts}
            assert "credit" in type_dict
            assert "debit" in type_dict

        finally:
            session.close()


class TestDatabaseIntegrationWithSimulation:
    """Test database operations in the context of business simulation."""

    def test_simulation_data_generation_and_persistence(self, simulator):
        """Test that simulation properly generates and persists data."""
        simulator.initialize_business("restaurant")
        simulator.simulate_historical_data(days_back=7)

        session = simulator.SessionLocal()
        try:
            # Verify accounts were created
            accounts = session.query(Account).all()
            assert len(accounts) >= 4

            # Verify transactions were generated
            transactions = session.query(Transaction).all()
            assert len(transactions) > 0

            # Verify transaction dates are within expected range
            min_date = datetime.now().date() - timedelta(days=8)
            max_date = datetime.now().date() + timedelta(days=1)

            for transaction in transactions:
                assert min_date <= transaction.transaction_date <= max_date

            # Verify inventory items were created
            items = session.query(Item).all()
            assert len(items) > 0

            # Verify stock movements were created
            movements = session.query(StockMovement).all()
            assert len(movements) > 0

            # Verify suppliers were created
            suppliers = session.query(Supplier).all()
            assert len(suppliers) > 0

            # Verify employees were created
            employees = session.query(Employee).all()
            assert len(employees) >= 3

        finally:
            session.close()

    def test_concurrent_database_access(self, temp_db):
        """Test concurrent database access scenarios."""
        import threading

        engine = create_engine(temp_db, pool_size=10, max_overflow=20)
        SessionLocal = sessionmaker(bind=engine)

        results = []
        errors = []

        def worker_function(worker_id):
            try:
                session = SessionLocal()
                try:
                    # Each worker creates some data
                    account = Account(
                        id=f"worker_{worker_id}",
                        name=f"Worker {worker_id} Account",
                        account_type="checking",
                        balance=1000.00 * worker_id,
                    )
                    session.add(account)
                    session.commit()

                    # Each worker creates transactions
                    for i in range(5):
                        tx = Transaction(
                            account_id=f"worker_{worker_id}",
                            amount=Decimal(f"{10 + i}.00"),
                            transaction_type="credit",
                            description=f"Worker {worker_id} transaction {i}",
                            transaction_date=datetime.now().date(),
                        )
                        session.add(tx)

                    session.commit()
                    results.append(f"Worker {worker_id} completed")

                finally:
                    session.close()

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {str(e)}")

        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 5, f"Expected 5 completions, got {len(results)}"
        assert len(errors) == 0, f"Got errors: {errors}"

        # Verify data was created correctly
        session = SessionLocal()
        try:
            accounts = session.query(Account).filter(Account.id.like("worker_%")).all()
            assert len(accounts) == 5

            transactions = (
                session.query(Transaction).filter(Transaction.account_id.like("worker_%")).all()
            )
            assert len(transactions) == 25  # 5 workers * 5 transactions each

        finally:
            session.close()

    def test_database_error_handling(self, temp_db):
        """Test database error handling and recovery."""
        engine = create_engine(temp_db)
        session = sessionmaker(bind=engine)()

        try:
            # Test transaction rollback on error
            account = Account(
                id="error_test", name="Error Test Account", account_type="checking", balance=1000.00
            )
            session.add(account)
            session.commit()

            # Start a transaction that will fail
            try:
                # Add a valid transaction
                tx1 = Transaction(
                    account_id="error_test",
                    amount=Decimal("100.00"),
                    transaction_type="credit",
                    description="Valid transaction",
                    transaction_date=datetime.now().date(),
                )
                session.add(tx1)

                # Add an invalid transaction (duplicate primary key or constraint violation)
                # This will depend on your specific constraints
                tx2 = Transaction(
                    account_id="nonexistent_account",  # Should violate foreign key
                    amount=Decimal("200.00"),
                    transaction_type="credit",
                    description="Invalid transaction",
                    transaction_date=datetime.now().date(),
                )
                session.add(tx2)

                session.commit()  # This should fail

            except Exception:
                session.rollback()

                # Verify rollback worked - no transactions should be added
                tx_count = session.query(Transaction).filter_by(account_id="error_test").count()
                assert tx_count == 0

                # Session should still be usable after rollback
                valid_tx = Transaction(
                    account_id="error_test",
                    amount=Decimal("150.00"),
                    transaction_type="credit",
                    description="Transaction after rollback",
                    transaction_date=datetime.now().date(),
                )
                session.add(valid_tx)
                session.commit()

                # This should succeed
                final_count = session.query(Transaction).filter_by(account_id="error_test").count()
                assert final_count == 1

        finally:
            session.close()
