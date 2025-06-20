"""
Dashboard Performance Tests

Tests for measuring and benchmarking dashboard loading times,
rendering performance, and responsiveness under various data loads.
"""

import os

# Add parent directories to path for imports
import sys
import time
from datetime import datetime, timedelta

import psutil
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.app import BusinessDashboard


class TestDashboardPerformance:
    """Test suite for dashboard performance benchmarks."""

    @pytest.fixture
    def test_dashboard(self, performance_config, temp_db_path):
        """Create a test dashboard instance."""
        # Create a temporary config file
        import tempfile

        import yaml

        config = performance_config.copy()
        config["database"]["url"] = f"sqlite:///{temp_db_path}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        dashboard = BusinessDashboard(config_path)

        yield dashboard

        # Cleanup
        try:
            os.unlink(config_path)
        except OSError:
            pass

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_dashboard_initialization_speed(self, benchmark, performance_config, temp_db_path):
        """Benchmark dashboard initialization time."""
        import tempfile

        import yaml

        def create_dashboard():
            config = performance_config.copy()
            config["database"]["url"] = f"sqlite:///{temp_db_path}"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name

            dashboard = BusinessDashboard(config_path)

            # Cleanup
            os.unlink(config_path)

            return dashboard

        dashboard = benchmark(create_dashboard)

        # Dashboard initialization should be fast
        assert dashboard is not None
        assert benchmark.stats['mean'] < 0.5  # Less than 500ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_financial_summary_performance(self, benchmark, test_dashboard, small_dataset):
        """Benchmark financial summary generation performance."""
        def get_financial_summary():
            return test_dashboard.get_financial_summary(days=30)

        result = benchmark(get_financial_summary)

        # Financial summary should load quickly
        assert result is not None
        assert 'total_revenue' in result
        assert 'total_expenses' in result
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_daily_revenue_chart_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark daily revenue chart generation performance."""
        def generate_revenue_chart():
            return test_dashboard.get_daily_revenue_chart(days=30)

        chart = benchmark(generate_revenue_chart)

        # Chart generation should be reasonably fast
        assert chart is not None
        assert benchmark.stats['mean'] < 0.5  # Less than 500ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_expense_breakdown_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark expense breakdown chart generation performance."""
        def generate_expense_chart():
            return test_dashboard.get_expense_breakdown(days=30)

        chart = benchmark(generate_expense_chart)

        # Chart generation should be reasonably fast
        assert chart is not None
        assert benchmark.stats['mean'] < 0.5  # Less than 500ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_inventory_summary_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark inventory summary generation performance."""
        def get_inventory_summary():
            return test_dashboard.get_inventory_summary()

        result = benchmark(get_inventory_summary)

        # Inventory summary should load quickly
        assert result is not None
        assert 'total_items' in result
        assert 'total_value' in result
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_inventory_levels_chart_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark inventory levels chart generation performance."""
        def generate_inventory_chart():
            return test_dashboard.get_inventory_levels_chart()

        chart = benchmark(generate_inventory_chart)

        # Chart generation should be reasonably fast
        assert chart is not None
        assert benchmark.stats['mean'] < 0.3  # Less than 300ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_hr_summary_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark HR summary generation performance."""
        def get_hr_summary():
            return test_dashboard.get_hr_summary()

        result = benchmark(get_hr_summary)

        # HR summary should load quickly
        assert result is not None
        assert 'total_employees' in result
        assert 'active_employees' in result
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_recent_transactions_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark recent transactions loading performance."""
        def get_recent_transactions():
            return test_dashboard.get_recent_transactions(limit=20)

        result = benchmark(get_recent_transactions)

        # Recent transactions should load quickly
        assert result is not None
        assert len(result) >= 0  # May be empty
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_recent_stock_movements_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark recent stock movements loading performance."""
        def get_recent_movements():
            return test_dashboard.get_recent_stock_movements(limit=20)

        result = benchmark(get_recent_movements)

        # Recent movements should load quickly
        assert result is not None
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_agent_decisions_loading_performance(self, benchmark, test_dashboard, small_dataset):
        """Benchmark agent decisions loading performance."""
        def load_agent_decisions():
            return test_dashboard.load_agent_decisions(limit=20)

        result = benchmark(load_agent_decisions)

        # Agent decisions should load quickly
        assert result is not None
        assert isinstance(result, list)
        assert benchmark.stats['mean'] < 0.1  # Less than 100ms

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_agent_status_performance(self, benchmark, test_dashboard):
        """Benchmark agent status checking performance."""
        def get_agent_status():
            return test_dashboard.get_agent_status()

        result = benchmark(get_agent_status)

        # Agent status should be instant
        assert result is not None
        assert isinstance(result, dict)
        assert benchmark.stats['mean'] < 0.01  # Less than 10ms

    @pytest.mark.dashboard
    @pytest.mark.stress
    def test_dashboard_with_large_dataset(self, test_dashboard, large_dataset):
        """Test dashboard performance with large dataset."""
        start_time = time.time()

        # Load all dashboard components with large dataset
        financial_summary = test_dashboard.get_financial_summary(days=90)
        revenue_chart = test_dashboard.get_daily_revenue_chart(days=90)
        expense_chart = test_dashboard.get_expense_breakdown(days=90)
        inventory_summary = test_dashboard.get_inventory_summary()
        inventory_chart = test_dashboard.get_inventory_levels_chart()
        hr_summary = test_dashboard.get_hr_summary()
        recent_transactions = test_dashboard.get_recent_transactions(limit=50)
        agent_decisions = test_dashboard.load_agent_decisions(limit=50)

        end_time = time.time()
        total_time = end_time - start_time

        # All dashboard components should load in reasonable time with large dataset
        assert financial_summary is not None
        assert revenue_chart is not None
        assert expense_chart is not None
        assert inventory_summary is not None
        assert inventory_chart is not None
        assert hr_summary is not None
        assert recent_transactions is not None
        assert agent_decisions is not None

        assert total_time < 5.0, f"Dashboard took too long with large dataset: {total_time:.2f}s"

    @pytest.mark.dashboard
    @pytest.mark.memory
    def test_dashboard_memory_usage(self, test_dashboard, medium_dataset):
        """Test dashboard memory usage during operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load dashboard components multiple times to check for memory leaks
        for iteration in range(10):
            financial_summary = test_dashboard.get_financial_summary(days=30)
            revenue_chart = test_dashboard.get_daily_revenue_chart(days=30)
            expense_chart = test_dashboard.get_expense_breakdown(days=30)
            inventory_summary = test_dashboard.get_inventory_summary()

            # Clear references
            del financial_summary, revenue_chart, expense_chart, inventory_summary

        # Force garbage collection
        import gc
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should be reasonable
        assert memory_increase < 50, f"Dashboard using too much memory: {memory_increase:.2f}MB"

    @pytest.mark.dashboard
    @pytest.mark.stress
    def test_concurrent_dashboard_access(self, performance_config, temp_db_path):
        """Test dashboard performance under concurrent access."""
        import queue
        import tempfile
        import threading

        import yaml

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def dashboard_worker(worker_id: int):
            """Worker function for concurrent dashboard access."""
            try:
                config = performance_config.copy()
                config["database"]["url"] = f"sqlite:///{temp_db_path}"

                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(config, f)
                    config_path = f.name

                dashboard = BusinessDashboard(config_path)

                start_time = time.time()

                # Perform various dashboard operations
                for i in range(5):
                    financial_summary = dashboard.get_financial_summary(days=30)
                    inventory_summary = dashboard.get_inventory_summary()
                    hr_summary = dashboard.get_hr_summary()
                    recent_transactions = dashboard.get_recent_transactions(limit=10)

                end_time = time.time()

                results_queue.put({
                    'worker_id': worker_id,
                    'duration': end_time - start_time,
                    'operations': 5
                })

                # Cleanup
                os.unlink(config_path)

            except Exception as e:
                error_queue.put({
                    'worker_id': worker_id,
                    'error': str(e)
                })

        # Start multiple worker threads
        threads = []
        num_workers = 4

        start_time = time.time()

        for worker_id in range(num_workers):
            thread = threading.Thread(target=dashboard_worker, args=(worker_id,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=20)  # 20 second timeout

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
        assert len(errors) == 0, f"Concurrent dashboard errors: {errors}"
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        assert total_duration < 30, f"Concurrent dashboard operations took too long: {total_duration:.2f}s"

        # Calculate average performance
        avg_duration = sum(r['duration'] for r in results) / len(results)
        assert avg_duration < 5, f"Average dashboard operations too slow: {avg_duration:.2f}s"

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_multiple_time_periods_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark dashboard performance across different time periods."""
        def load_multiple_periods():
            results = {}
            for days in [7, 30, 60, 90]:
                results[days] = {
                    'financial': test_dashboard.get_financial_summary(days=days),
                    'revenue_chart': test_dashboard.get_daily_revenue_chart(days=days),
                    'expense_chart': test_dashboard.get_expense_breakdown(days=days)
                }
            return results

        results = benchmark(load_multiple_periods)

        # Multiple time periods should load in reasonable time
        assert len(results) == 4
        for days, data in results.items():
            assert data['financial'] is not None
            assert data['revenue_chart'] is not None
            assert data['expense_chart'] is not None

        assert benchmark.stats['mean'] < 2.0  # Less than 2 seconds for all periods

    @pytest.mark.dashboard
    @pytest.mark.stress
    def test_dashboard_rapid_refresh_performance(self, test_dashboard, medium_dataset):
        """Test dashboard performance under rapid refresh scenarios."""
        refresh_times = []

        # Simulate rapid dashboard refreshes
        for refresh in range(20):
            start_time = time.time()

            # Load key dashboard components (simulating a refresh)
            financial_summary = test_dashboard.get_financial_summary(days=30)
            inventory_summary = test_dashboard.get_inventory_summary()
            recent_transactions = test_dashboard.get_recent_transactions(limit=10)

            end_time = time.time()
            refresh_times.append(end_time - start_time)

        # Performance should remain stable under rapid refresh
        avg_refresh_time = sum(refresh_times) / len(refresh_times)
        max_refresh_time = max(refresh_times)

        assert avg_refresh_time < 0.5, f"Average refresh time too slow: {avg_refresh_time:.2f}s"
        assert max_refresh_time < 1.0, f"Max refresh time too slow: {max_refresh_time:.2f}s"

        # Performance should not degrade significantly over time
        early_avg = sum(refresh_times[:5]) / 5
        late_avg = sum(refresh_times[-5:]) / 5
        degradation = late_avg - early_avg

        assert degradation < 0.2, f"Performance degraded over time: {degradation:.2f}s"

    @pytest.mark.benchmark
    @pytest.mark.dashboard
    def test_chart_data_processing_performance(self, benchmark, test_dashboard, medium_dataset):
        """Benchmark chart data processing performance."""
        def process_chart_data():
            # This tests the data processing part of chart generation
            import pandas as pd

            # Simulate chart data processing
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)

            # Get raw data
            session = test_dashboard.SessionLocal()
            try:
                from sqlalchemy import func

                from models.financial import Transaction, TransactionType

                daily_revenue = session.query(
                    func.date(Transaction.transaction_date).label('date'),
                    func.sum(Transaction.amount).label('revenue')
                ).filter(
                    Transaction.transaction_type == TransactionType.INCOME,
                    func.date(Transaction.transaction_date) >= start_date
                ).group_by(func.date(Transaction.transaction_date)).all()

                # Process data into DataFrame (simulating chart preparation)
                df = pd.DataFrame(daily_revenue, columns=['date', 'revenue'])
                df['date'] = pd.to_datetime(df['date'])

                return len(df)

            finally:
                session.close()

        result = benchmark(process_chart_data)

        # Chart data processing should be fast
        assert result >= 0
        assert benchmark.stats['mean'] < 0.2  # Less than 200ms
