"""
Stress Tests

Comprehensive stress tests for high-volume scenarios, memory leak detection,
and system stability under extreme conditions.
"""

import asyncio
import gc
import os
import queue

# Add parent directories to path for imports
import sys
import threading
import time
from datetime import datetime, timedelta

import psutil
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import patch

from agents.accounting_agent import AccountingAgent
from agents.inventory_agent import InventoryAgent
from models.financial import Transaction, TransactionType
from models.inventory import Item
from simulation.business_simulator import BusinessSimulator


class TestStressScenarios:
    """Test suite for stress testing scenarios."""

    @pytest.mark.stress
    @pytest.mark.agent
    def test_high_volume_agent_decisions(self, test_accounting_agent, large_dataset):
        """Stress test agents with high volume of decisions."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        successful_decisions = 0
        failed_decisions = 0
        memory_samples = []

        # Process 5000 decisions
        for i in range(5000):
            try:
                test_data = {
                    "transaction_id": f"stress_tx_{i}",
                    "amount": 100.0 + (i % 1000),
                    "transaction_type": "expense" if i % 3 == 0 else "income",
                    "description": f"Stress test transaction {i}",
                    "category": f"stress_category_{i % 20}"
                }

                # Mock Claude API to avoid costs
                with patch.object(test_accounting_agent, 'analyze_with_claude') as mock_claude:
                    mock_claude.return_value = f"Stress test analysis {i}"
                    result = asyncio.run(test_accounting_agent.process_data(test_data))

                    if result:
                        successful_decisions += 1
                    else:
                        failed_decisions += 1

                # Sample memory every 100 decisions
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(current_memory - initial_memory)

                    # Clear decision log periodically to prevent excessive memory usage
                    if len(test_accounting_agent.decisions_log) > 100:
                        test_accounting_agent.decisions_log = test_accounting_agent.decisions_log[-50:]

            except Exception:
                failed_decisions += 1
                if failed_decisions > 100:  # Stop if too many failures
                    break

        end_time = time.time()
        duration = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Performance assertions
        assert successful_decisions >= 4500, f"Only {successful_decisions} successful decisions"
        assert failed_decisions < 100, f"Too many failed decisions: {failed_decisions}"
        assert duration < 300, f"Stress test took too long: {duration:.2f}s"  # 5 minutes max

        # Memory leak detection
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200, f"Potential memory leak: {memory_increase:.2f}MB increase"

        # Check memory growth pattern
        if len(memory_samples) > 10:
            early_avg = sum(memory_samples[:5]) / 5
            late_avg = sum(memory_samples[-5:]) / 5
            memory_growth_rate = late_avg - early_avg
            assert memory_growth_rate < 50, f"Memory growing too fast: {memory_growth_rate:.2f}MB"

        # Throughput check
        throughput = successful_decisions / duration
        assert throughput > 15, f"Throughput too low: {throughput:.2f} decisions/second"

    @pytest.mark.stress
    @pytest.mark.database
    def test_database_stress_concurrent_access(self, test_db_engine):
        """Stress test database with high concurrent access."""
        num_workers = 10
        operations_per_worker = 200

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def database_stress_worker(worker_id: int):
            """Worker function for database stress testing."""
            try:
                from sqlalchemy.orm import sessionmaker
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)

                session = SessionLocal()
                start_time = time.time()
                operations = 0

                for i in range(operations_per_worker):
                    try:
                        # Mix of operations: inserts, queries, updates
                        operation_type = i % 4

                        if operation_type == 0:  # Insert transaction
                            transaction = Transaction(
                                id=f"stress_worker_{worker_id}_tx_{i}",
                                amount=100.0 + i,
                                transaction_type=TransactionType.INCOME if i % 2 else TransactionType.EXPENSE,
                                transaction_date=datetime.now(),
                                description=f"Stress test worker {worker_id} transaction {i}",
                                account_id="test_checking",
                                category="stress_test"
                            )
                            session.add(transaction)

                        elif operation_type == 1:  # Query transactions
                            count = session.query(Transaction).filter(
                                Transaction.category == "stress_test"
                            ).count()

                        elif operation_type == 2:  # Insert item
                            item = Item(
                                sku=f"STRESS_W{worker_id}_I{i}",
                                name=f"Stress Item W{worker_id} I{i}",
                                category="stress_test",
                                unit_cost=10.0,
                                selling_price=20.0,
                                current_stock=100,
                                reorder_point=20,
                                reorder_quantity=50
                            )
                            session.add(item)

                        else:  # Query items
                            count = session.query(Item).filter(
                                Item.category == "stress_test"
                            ).count()

                        # Commit every 10 operations
                        if i % 10 == 0:
                            session.commit()

                        operations += 1

                    except Exception:
                        session.rollback()
                        # Continue with next operation
                        pass

                # Final commit
                session.commit()
                session.close()

                end_time = time.time()

                results_queue.put({
                    'worker_id': worker_id,
                    'operations': operations,
                    'duration': end_time - start_time,
                    'ops_per_second': operations / (end_time - start_time)
                })

            except Exception as e:
                error_queue.put({
                    'worker_id': worker_id,
                    'error': str(e)
                })

        # Start all workers
        threads = []
        start_time = time.time()

        for worker_id in range(num_workers):
            thread = threading.Thread(target=database_stress_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=120)  # 2 minute timeout per worker

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
        assert len(errors) == 0, f"Database stress test errors: {errors}"
        assert len(results) == num_workers, f"Not all workers completed: {len(results)}/{num_workers}"

        total_operations = sum(r['operations'] for r in results)
        avg_throughput = sum(r['ops_per_second'] for r in results) / len(results)

        assert total_operations >= num_workers * operations_per_worker * 0.9, "Too many failed operations"
        assert total_duration < 180, f"Stress test took too long: {total_duration:.2f}s"
        assert avg_throughput > 5, f"Average throughput too low: {avg_throughput:.2f} ops/sec"

    @pytest.mark.stress
    @pytest.mark.simulation
    def test_simulation_stress_continuous_generation(self, performance_config, temp_db_path):
        """Stress test simulation with continuous data generation."""
        db_url = f"sqlite:///{temp_db_path}"
        simulator = BusinessSimulator(performance_config, db_url)
        simulator.initialize_business("restaurant")

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_samples = []
        generation_times = []

        # Continuous generation for 50 cycles
        for cycle in range(50):
            cycle_start = time.time()

            # Generate data
            simulator.simulate_historical_data(days_back=5)

            cycle_end = time.time()
            cycle_duration = cycle_end - cycle_start
            generation_times.append(cycle_duration)

            # Monitor memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory - initial_memory)

            # Brief pause to allow for any cleanup
            time.sleep(0.1)

            # Force garbage collection every 10 cycles
            if cycle % 10 == 0:
                gc.collect()

        # Performance assertions
        avg_generation_time = sum(generation_times) / len(generation_times)
        max_generation_time = max(generation_times)
        final_memory = memory_samples[-1]

        assert avg_generation_time < 10, f"Average generation time too slow: {avg_generation_time:.2f}s"
        assert max_generation_time < 30, f"Max generation time too slow: {max_generation_time:.2f}s"
        assert final_memory < 500, f"Memory usage too high: {final_memory:.2f}MB"

        # Check for memory stability
        if len(memory_samples) >= 20:
            early_memory = sum(memory_samples[:10]) / 10
            late_memory = sum(memory_samples[-10:]) / 10
            memory_growth = late_memory - early_memory
            assert memory_growth < 100, f"Memory continuously growing: {memory_growth:.2f}MB"

    @pytest.mark.stress
    @pytest.mark.agent
    def test_multi_agent_stress(self, performance_config, temp_db_path):
        """Stress test multiple agents running concurrently."""
        db_url = f"sqlite:///{temp_db_path}"

        # Create multiple agents
        agents = []
        for i in range(3):
            accounting_agent = AccountingAgent(
                agent_id=f"stress_accounting_agent_{i}",
                api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
                config=performance_config["agents"]["accounting"],
                db_url=db_url
            )
            agents.append(('accounting', accounting_agent))

            inventory_agent = InventoryAgent(
                agent_id=f"stress_inventory_agent_{i}",
                api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
                config=performance_config["agents"]["inventory"],
                db_url=db_url
            )
            agents.append(('inventory', inventory_agent))

        async def stress_agent_worker(agent_type: str, agent, worker_id: int):
            """Worker function for agent stress testing."""
            successful_operations = 0

            for i in range(100):
                try:
                    if agent_type == 'accounting':
                        test_data = {
                            "transaction_id": f"multi_stress_{worker_id}_{i}",
                            "amount": 50.0 + i,
                            "transaction_type": "expense" if i % 2 else "income",
                            "description": f"Multi-agent stress test {worker_id} {i}",
                            "category": "multi_stress"
                        }
                    else:  # inventory
                        test_data = {
                            "item_id": i + 1,
                            "current_stock": 10 + (i % 50),
                            "reorder_point": 20,
                            "item_name": f"Stress Item {worker_id} {i}",
                            "supplier": "Stress Supplier"
                        }

                    # Mock Claude API
                    with patch.object(agent, 'analyze_with_claude') as mock_claude:
                        mock_claude.return_value = f"Multi-agent stress analysis {worker_id} {i}"
                        result = await agent.process_data(test_data)

                        if result:
                            successful_operations += 1

                except Exception:
                    pass  # Continue with next operation

            return successful_operations

        async def run_multi_agent_stress():
            """Run stress test with multiple agents."""
            tasks = []

            for i, (agent_type, agent) in enumerate(agents):
                task = stress_agent_worker(agent_type, agent, i)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        start_time = time.time()
        results = asyncio.run(run_multi_agent_stress())
        end_time = time.time()

        duration = end_time - start_time

        # Count successful operations
        successful_results = [r for r in results if not isinstance(r, Exception)]
        total_successful = sum(successful_results)

        # Assertions
        assert len(successful_results) == len(agents), f"Some agents failed: {len(successful_results)}/{len(agents)}"
        assert total_successful >= len(agents) * 80, f"Too few successful operations: {total_successful}"
        assert duration < 120, f"Multi-agent stress took too long: {duration:.2f}s"

        # Check individual agent performance
        for i, result in enumerate(successful_results):
            assert result >= 80, f"Agent {i} had too few successful operations: {result}"

    @pytest.mark.stress
    @pytest.mark.memory
    def test_memory_leak_detection_extended(self, test_accounting_agent):
        """Extended memory leak detection test."""
        process = psutil.Process()

        memory_samples = []
        operation_counts = []

        # Run for 20 iterations with increasing workload
        for iteration in range(20):
            # Force garbage collection
            gc.collect()

            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Increasing workload each iteration
            operations_this_iteration = 50 + (iteration * 5)

            for i in range(operations_this_iteration):
                test_data = {
                    "transaction_id": f"leak_detect_{iteration}_{i}",
                    "amount": 75.0 + i,
                    "transaction_type": "expense",
                    "description": f"Memory leak detection {iteration} {i}",
                    "category": "leak_detection"
                }

                with patch.object(test_accounting_agent, 'analyze_with_claude') as mock_claude:
                    mock_claude.return_value = f"Leak detection analysis {iteration} {i}"
                    asyncio.run(test_accounting_agent.process_data(test_data))

            # Clear decision log to prevent it from being the source of memory "leak"
            test_accounting_agent.decisions_log = test_accounting_agent.decisions_log[-10:]

            # Force another garbage collection
            gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            memory_samples.append(memory_increase)
            operation_counts.append(operations_this_iteration)

        # Analyze memory usage pattern
        # Memory increase should not be strongly correlated with operation count
        if len(memory_samples) >= 10:
            # Calculate trend
            x_values = list(range(len(memory_samples)))
            y_values = memory_samples

            # Simple linear regression to detect trend
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

            # Slope should be small (no significant memory growth trend)
            assert abs(slope) < 1.0, f"Memory leak detected: trend slope = {slope:.3f} MB/iteration"

        # No single iteration should use excessive memory
        max_memory_increase = max(memory_samples)
        assert max_memory_increase < 50, f"Single iteration used too much memory: {max_memory_increase:.2f}MB"

        # Average memory increase should be reasonable
        avg_memory_increase = sum(memory_samples) / len(memory_samples)
        assert avg_memory_increase < 10, f"Average memory increase too high: {avg_memory_increase:.2f}MB"

    @pytest.mark.stress
    @pytest.mark.agent
    def test_agent_error_recovery_stress(self, test_accounting_agent):
        """Stress test agent error recovery capabilities."""
        successful_operations = 0
        error_recoveries = 0
        consecutive_errors = 0
        max_consecutive_errors = 0

        for i in range(1000):
            try:
                # Occasionally inject errors
                if i % 50 == 0:
                    # Force an error scenario
                    test_data = {
                        "transaction_id": f"error_stress_{i}",
                        "amount": "invalid_amount",  # This should cause an error
                        "transaction_type": "invalid_type",
                        "description": f"Error stress test {i}",
                        "category": "error_stress"
                    }
                else:
                    # Normal operation
                    test_data = {
                        "transaction_id": f"normal_stress_{i}",
                        "amount": 100.0 + i,
                        "transaction_type": "expense" if i % 2 else "income",
                        "description": f"Normal stress test {i}",
                        "category": "normal_stress"
                    }

                with patch.object(test_accounting_agent, 'analyze_with_claude') as mock_claude:
                    mock_claude.return_value = f"Error recovery stress analysis {i}"
                    result = await test_accounting_agent.process_data(test_data)

                    if result:
                        successful_operations += 1
                        if consecutive_errors > 0:
                            error_recoveries += 1
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        max_consecutive_errors = max(max_consecutive_errors, consecutive_errors)

            except Exception:
                consecutive_errors += 1
                max_consecutive_errors = max(max_consecutive_errors, consecutive_errors)

        # Assertions for error recovery
        assert successful_operations >= 900, f"Too few successful operations: {successful_operations}"
        assert error_recoveries >= 15, f"Agent should recover from errors: {error_recoveries} recoveries"
        assert max_consecutive_errors <= 10, f"Too many consecutive errors: {max_consecutive_errors}"

        # Agent should still be operational
        health_status = asyncio.run(test_accounting_agent.health_check())
        assert health_status is not None
        assert "agent_id" in health_status

    @pytest.mark.stress
    @pytest.mark.simulation
    def test_simulation_data_volume_stress(self, performance_config, temp_db_path):
        """Stress test simulation with very large data volumes."""
        db_url = f"sqlite:///{temp_db_path}"
        simulator = BusinessSimulator(performance_config, db_url)
        simulator.initialize_business("restaurant")

        start_time = time.time()

        # Generate very large historical dataset
        simulator.simulate_historical_data(days_back=365)  # 1 year of data

        end_time = time.time()
        generation_time = end_time - start_time

        # Check results
        status = simulator.get_simulation_status()

        # Assertions
        assert status['transaction_count'] > 50000, f"Should generate substantial data: {status['transaction_count']}"
        assert generation_time < 180, f"Large data generation took too long: {generation_time:.2f}s"  # 3 minutes max

        # Test querying the large dataset
        query_start = time.time()

        session = simulator.SessionLocal()
        try:
            # Complex query on large dataset
            recent_transactions = session.query(Transaction).filter(
                Transaction.transaction_date >= datetime.now() - timedelta(days=30)
            ).order_by(Transaction.transaction_date.desc()).limit(100).all()

            # Aggregation query
            monthly_revenue = session.query(
                func.strftime('%Y-%m', Transaction.transaction_date).label('month'),
                func.sum(Transaction.amount).label('revenue')
            ).filter(
                Transaction.transaction_type == TransactionType.INCOME
            ).group_by(func.strftime('%Y-%m', Transaction.transaction_date)).all()

        finally:
            session.close()

        query_end = time.time()
        query_time = query_end - query_start

        # Query performance on large dataset should still be reasonable
        assert len(recent_transactions) > 0
        assert len(monthly_revenue) > 0
        assert query_time < 5, f"Queries on large dataset too slow: {query_time:.2f}s"
