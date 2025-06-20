"""
Simulation Performance Tests

Tests for measuring and benchmarking simulation data generation speed,
memory usage, and efficiency under various conditions.
"""

import asyncio
import os

# Add parent directories to path for imports
import sys
import time
from datetime import datetime, timedelta

import psutil
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.financial import Transaction
from models.inventory import Item
from simulation.business_simulator import BusinessSimulator
from simulation.financial_generator import (
    FinancialDataGenerator,
    get_restaurant_profile,
)
from simulation.inventory_simulator import InventorySimulator, get_restaurant_inventory_profile


class TestSimulationPerformance:
    """Test suite for simulation performance benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_business_simulator_initialization(self, benchmark, performance_config, temp_db_path):
        """Benchmark business simulator initialization time."""
        def create_and_initialize_simulator():
            db_url = f"sqlite:///{temp_db_path}"
            simulator = BusinessSimulator(performance_config, db_url)
            simulator.initialize_business("restaurant")
            return simulator

        simulator = benchmark(create_and_initialize_simulator)

        # Initialization should be fast
        assert simulator is not None
        assert benchmark.stats['mean'] < 2.0  # Less than 2 seconds

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_financial_data_generation_speed(self, benchmark):
        """Benchmark financial data generation performance."""
        profile = get_restaurant_profile()
        generator = FinancialDataGenerator(profile)

        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        def generate_financial_data():
            return generator.generate_period_data(start_date, end_date)

        result = benchmark(generate_financial_data)

        # Data generation should be fast
        assert result is not None
        assert len(result['transactions']) > 0
        assert benchmark.stats['mean'] < 1.0  # Less than 1 second for 30 days

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_daily_transaction_generation_speed(self, benchmark):
        """Benchmark daily transaction generation performance."""
        profile = get_restaurant_profile()
        generator = FinancialDataGenerator(profile)

        def generate_daily_transactions():
            return generator.generate_daily_transactions(datetime.now())

        result = benchmark(generate_daily_transactions)

        # Daily generation should be very fast
        assert result is not None
        assert len(result) > 0
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_inventory_simulation_speed(self, benchmark):
        """Benchmark inventory simulation performance."""
        profile = get_restaurant_inventory_profile()
        simulator = InventorySimulator(profile)

        def generate_inventory_data():
            # Generate initial inventory
            items = simulator.generate_initial_inventory()

            # Simulate daily consumption
            consumption = simulator.simulate_daily_consumption(items, datetime.now())

            # Simulate deliveries
            deliveries = simulator.simulate_deliveries(items, datetime.now())

            return len(items), len(consumption), len(deliveries)

        result = benchmark(generate_inventory_data)

        # Inventory simulation should be fast
        assert result[0] > 0  # Should generate items
        assert benchmark.stats['mean'] < 0.5  # Less than 500ms

    @pytest.mark.simulation
    @pytest.mark.stress
    def test_large_scale_data_generation(self, test_business_simulator):
        """Test simulation performance with large-scale data generation."""
        start_time = time.time()

        # Generate 180 days of historical data
        test_business_simulator.simulate_historical_data(days_back=180)

        end_time = time.time()
        generation_time = end_time - start_time

        # Large-scale generation should complete in reasonable time
        assert generation_time < 30, f"Large data generation took {generation_time:.2f}s"

        # Verify data was generated
        status = test_business_simulator.get_simulation_status()
        assert status['transaction_count'] > 1000  # Should have substantial data

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_anomaly_generation_performance(self, benchmark):
        """Benchmark anomaly generation performance."""
        profile = get_restaurant_profile()
        generator = FinancialDataGenerator(profile)

        # Generate base transactions
        base_transactions = generator.generate_daily_transactions(datetime.now())

        def generate_anomalies():
            return generator.generate_anomalies(base_transactions, anomaly_rate=0.1)

        result = benchmark(generate_anomalies)

        # Anomaly generation should be fast
        assert result is not None
        assert benchmark.stats['mean'] < 0.05  # Less than 50ms

    @pytest.mark.simulation
    @pytest.mark.memory
    def test_simulation_memory_usage(self, performance_config, temp_db_path):
        """Test simulation memory usage during data generation."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        db_url = f"sqlite:///{temp_db_path}"
        simulator = BusinessSimulator(performance_config, db_url)
        simulator.initialize_business("restaurant")

        # Monitor memory during data generation
        memory_samples = []

        for days in [30, 60, 90]:
            simulator.simulate_historical_data(days_back=days)
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory - initial_memory)

        # Memory usage should be reasonable and not grow excessively
        max_memory_usage = max(memory_samples)
        assert max_memory_usage < 500, f"Excessive memory usage: {max_memory_usage:.2f}MB"

        # Memory should not grow linearly with data size (should be efficient)
        memory_growth = memory_samples[-1] - memory_samples[0]
        assert memory_growth < 200, f"Memory growth too high: {memory_growth:.2f}MB"

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_real_time_simulation_cycle_performance(self, benchmark, test_business_simulator):
        """Benchmark real-time simulation cycle performance."""
        async def single_simulation_cycle():
            # Simulate a single cycle of real-time data generation
            today = datetime.now()

            # Generate transactions
            daily_transactions = test_business_simulator.financial_generator.generate_daily_transactions(today)

            # Simulate database insertion (limited to avoid overhead)
            session = test_business_simulator.SessionLocal()
            try:
                for tx_data in daily_transactions[:5]:  # Limit to 5 transactions
                    from models.financial import Transaction
                    transaction = Transaction(**tx_data)
                    session.add(transaction)
                session.commit()
            finally:
                session.close()

            return len(daily_transactions)

        def run_cycle():
            return asyncio.run(single_simulation_cycle())

        result = benchmark(run_cycle)

        # Real-time simulation cycles should be very fast
        assert result > 0
        assert benchmark.stats['mean'] < 0.2  # Less than 200ms per cycle

    @pytest.mark.simulation
    @pytest.mark.stress
    def test_concurrent_simulation_performance(self, performance_config, temp_db_path):
        """Test simulation performance under concurrent operations."""
        import queue
        import threading

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def simulation_worker(worker_id: int):
            """Worker function for concurrent simulation."""
            try:
                db_url = f"sqlite:///{temp_db_path}_worker_{worker_id}"
                simulator = BusinessSimulator(performance_config, db_url)
                simulator.initialize_business("restaurant")

                start_time = time.time()

                # Generate data concurrently
                simulator.simulate_historical_data(days_back=30)

                end_time = time.time()

                status = simulator.get_simulation_status()
                results_queue.put({
                    'worker_id': worker_id,
                    'duration': end_time - start_time,
                    'transaction_count': status['transaction_count']
                })

            except Exception as e:
                error_queue.put({
                    'worker_id': worker_id,
                    'error': str(e)
                })

        # Start multiple worker threads
        threads = []
        num_workers = 3

        start_time = time.time()

        for worker_id in range(num_workers):
            thread = threading.Thread(target=simulation_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

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
        assert len(errors) == 0, f"Concurrent simulation errors: {errors}"
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        assert total_duration < 60, f"Concurrent simulation took too long: {total_duration:.2f}s"

        # Verify all workers generated data
        for result in results:
            assert result['transaction_count'] > 0, f"Worker {result['worker_id']} generated no data"

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_different_business_type_performance(self, benchmark, performance_config, temp_db_path):
        """Compare simulation performance between different business types."""
        def simulate_restaurant():
            db_url = f"sqlite:///{temp_db_path}_restaurant"
            simulator = BusinessSimulator(performance_config, db_url)
            simulator.initialize_business("restaurant")
            simulator.simulate_historical_data(days_back=30)
            return simulator.get_simulation_status()

        def simulate_retail():
            db_url = f"sqlite:///{temp_db_path}_retail"
            simulator = BusinessSimulator(performance_config, db_url)
            simulator.initialize_business("retail")
            simulator.simulate_historical_data(days_back=30)
            return simulator.get_simulation_status()

        restaurant_result = benchmark(simulate_restaurant)
        retail_result = benchmark(simulate_retail)

        # Both business types should perform similarly
        assert restaurant_result['transaction_count'] > 0
        assert retail_result['transaction_count'] > 0
        assert benchmark.stats['mean'] < 5.0  # Less than 5 seconds

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_data_persistence_performance(self, benchmark, test_business_simulator):
        """Benchmark data persistence performance during simulation."""
        # Generate data in memory first
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        financial_data = test_business_simulator.financial_generator.generate_period_data(
            start_date, end_date
        )

        inventory_items = test_business_simulator.inventory_simulator.generate_initial_inventory()

        def persist_simulation_data():
            session = test_business_simulator.SessionLocal()
            try:
                # Insert transactions
                for tx_data in financial_data["transactions"][:50]:  # Limit for benchmark
                    transaction = Transaction(**tx_data)
                    session.add(transaction)

                # Insert inventory items
                for item_data in inventory_items[:20]:  # Limit for benchmark
                    item = Item(**item_data)
                    session.add(item)

                session.commit()
                return True
            finally:
                session.close()

        result = benchmark(persist_simulation_data)

        # Data persistence should be efficient
        assert result is True
        assert benchmark.stats['mean'] < 0.5  # Less than 500ms

    @pytest.mark.simulation
    @pytest.mark.stress
    def test_long_running_simulation_stability(self, performance_config, temp_db_path):
        """Test simulation stability over extended periods."""
        db_url = f"sqlite:///{temp_db_path}"
        simulator = BusinessSimulator(performance_config, db_url)
        simulator.initialize_business("restaurant")

        # Monitor memory over time
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_samples = []

        # Simulate extended operation
        for cycle in range(10):
            cycle_start = time.time()

            # Generate data for one week
            simulator.simulate_historical_data(days_back=7)

            cycle_end = time.time()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_samples.append({
                'cycle': cycle,
                'memory_usage': current_memory - initial_memory,
                'cycle_time': cycle_end - cycle_start
            })

            # Check for memory leaks
            if len(memory_samples) > 3:
                recent_memory = memory_samples[-1]['memory_usage']
                early_memory = memory_samples[0]['memory_usage']

                # Memory should not continuously grow
                memory_growth = recent_memory - early_memory
                assert memory_growth < 100, f"Potential memory leak: {memory_growth:.2f}MB growth"

        # Performance should remain stable
        cycle_times = [s['cycle_time'] for s in memory_samples]
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        max_cycle_time = max(cycle_times)

        assert avg_cycle_time < 5.0, f"Average cycle time too high: {avg_cycle_time:.2f}s"
        assert max_cycle_time < 10.0, f"Max cycle time too high: {max_cycle_time:.2f}s"

    @pytest.mark.benchmark
    @pytest.mark.simulation
    def test_scenario_application_performance(self, benchmark, test_business_simulator):
        """Benchmark scenario application performance."""
        scenarios = test_business_simulator.generate_sample_scenarios()

        def apply_scenario():
            # Apply a test scenario
            test_business_simulator.apply_scenario("Cash Flow Crisis")
            return True

        result = benchmark(apply_scenario)

        # Scenario application should be fast
        assert result is True
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.simulation
    @pytest.mark.memory
    def test_simulation_cleanup_efficiency(self, performance_config, temp_db_path):
        """Test simulation cleanup and resource management efficiency."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and destroy multiple simulators
        for i in range(5):
            db_url = f"sqlite:///{temp_db_path}_cleanup_{i}"
            simulator = BusinessSimulator(performance_config, db_url)
            simulator.initialize_business("restaurant")
            simulator.simulate_historical_data(days_back=30)

            # Explicitly clean up
            del simulator

            # Force garbage collection
            import gc
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory should be properly cleaned up
        assert memory_increase < 100, f"Memory not properly cleaned up: {memory_increase:.2f}MB"
