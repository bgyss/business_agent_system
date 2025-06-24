"""Agent Performance Tests.

Tests for measuring and benchmarking agent decision processing speed,
memory usage, and scalability under various conditions.
"""

import asyncio
import os

# Add parent directories to path for imports
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import patch

# Memory profiling imports
import psutil
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.agent_decisions import AgentDecision
from models.financial import Transaction


class TestAgentPerformance:
    """Test suite for agent performance benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_accounting_agent_decision_speed(self, benchmark, test_accounting_agent, small_dataset):
        """Benchmark accounting agent decision processing speed."""
        # Create test transaction data
        test_data = {
            "transaction_id": "perf_test_tx_001",
            "amount": 5000.0,
            "transaction_type": "expense",
            "description": "Large expense transaction",
            "category": "equipment",
        }

        def process_decision():
            # Mock the Claude API call to avoid API costs during benchmarks
            with patch.object(test_accounting_agent, "analyze_with_claude") as mock_claude:
                mock_claude.return_value = """
                Analysis: This is a significant expense that requires attention.
                Risk Level: Medium
                Recommendation: Review this expense for budget impact.
                Confidence: 0.85
                """
                return asyncio.run(test_accounting_agent.process_data(test_data))

        result = benchmark(process_decision)

        # Assertions
        assert result is not None
        # Performance baseline: should complete within 100ms for mocked API
        assert benchmark.stats["mean"] < 0.1

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_inventory_agent_decision_speed(self, benchmark, test_inventory_agent, small_dataset):
        """Benchmark inventory agent decision processing speed."""
        # Create test inventory data
        test_data = {
            "item_id": 1,
            "current_stock": 5,
            "reorder_point": 20,
            "item_name": "Test Item",
            "supplier": "Test Supplier",
        }

        def process_decision():
            # Mock the Claude API call
            with patch.object(test_inventory_agent, "analyze_with_claude") as mock_claude:
                mock_claude.return_value = """
                Analysis: Stock level is below reorder point.
                Action: Recommend immediate reorder.
                Quantity: 50 units
                Confidence: 0.92
                """
                return asyncio.run(test_inventory_agent.process_data(test_data))

        result = benchmark(process_decision)

        # Assertions
        assert result is not None
        assert benchmark.stats["mean"] < 0.1

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_hr_agent_decision_speed(self, benchmark, test_hr_agent, small_dataset):
        """Benchmark HR agent decision processing speed."""
        # Create test HR data
        test_data = {
            "employee_id": "EMP_001",
            "hours_worked": 45,
            "overtime_hours": 5,
            "department": "Operations",
            "pay_rate": 15.0,
        }

        def process_decision():
            # Mock the Claude API call
            with patch.object(test_hr_agent, "analyze_with_claude") as mock_claude:
                mock_claude.return_value = """
                Analysis: Employee has worked overtime hours.
                Action: Approve overtime pay and review scheduling.
                Amount: $75.00
                Confidence: 0.88
                """
                return asyncio.run(test_hr_agent.process_data(test_data))

        result = benchmark(process_decision)

        # Assertions
        assert result is not None
        assert benchmark.stats["mean"] < 0.1

    @pytest.mark.memory
    @pytest.mark.agent
    def test_accounting_agent_memory_usage(self, test_accounting_agent, medium_dataset):
        """Test accounting agent memory usage during processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple decisions
        test_data_list = []
        for i in range(100):
            test_data_list.append(
                {
                    "transaction_id": f"memory_test_tx_{i}",
                    "amount": 100.0 + i,
                    "transaction_type": "expense" if i % 2 else "income",
                    "description": f"Memory test transaction {i}",
                    "category": "test",
                }
            )

        # Mock Claude API to avoid costs
        with patch.object(test_accounting_agent, "analyze_with_claude") as mock_claude:
            mock_claude.return_value = "Test analysis result"

            # Process all decisions
            for test_data in test_data_list:
                asyncio.run(test_accounting_agent.process_data(test_data))

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should not increase significantly (less than 50MB for 100 decisions)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB"

        # Check decision log size
        assert len(test_accounting_agent.decisions_log) <= 100

    @pytest.mark.stress
    @pytest.mark.agent
    def test_agent_concurrent_processing(self, test_accounting_agent, medium_dataset):
        """Test agent performance under concurrent decision processing."""

        async def process_concurrent_decisions():
            # Create multiple decision tasks
            tasks = []
            for i in range(50):
                test_data = {
                    "transaction_id": f"concurrent_tx_{i}",
                    "amount": 200.0 + i,
                    "transaction_type": "expense",
                    "description": f"Concurrent test transaction {i}",
                    "category": "test",
                }

                # Mock Claude API
                with patch.object(test_accounting_agent, "analyze_with_claude") as mock_claude:
                    mock_claude.return_value = f"Concurrent analysis {i}"
                    task = test_accounting_agent.process_data(test_data)
                    tasks.append(task)

            # Process all tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            return results, end_time - start_time

        results, duration = asyncio.run(process_concurrent_decisions())

        # Check that all tasks completed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 45  # Allow for some failures

        # Performance check: concurrent processing should be faster than sequential
        # With 50 tasks, should complete in reasonable time
        assert duration < 5.0, f"Concurrent processing took {duration:.2f}s"

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_agent_decision_logging_performance(
        self, benchmark, test_accounting_agent, small_dataset
    ):
        """Benchmark agent decision logging performance."""
        # Create test decision
        test_decision = AgentDecision(
            agent_id="test_accounting_agent",
            decision_type="performance_test",
            context={"test": "data"},
            reasoning="Performance test reasoning",
            action="Test action",
            confidence=0.85,
            timestamp=datetime.now(),
        )

        def log_decision():
            test_accounting_agent.log_decision(test_decision)

        benchmark(log_decision)

        # Performance baseline: logging should be fast
        assert benchmark.stats["mean"] < 0.01  # Less than 10ms

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_agent_message_handling_performance(self, benchmark, test_accounting_agent):
        """Benchmark agent message handling performance."""
        from agents.base_agent import AgentMessage

        test_message = AgentMessage(
            sender="test_sender",
            recipient="test_accounting_agent",
            message_type="data_update",
            content={"test": "message_content"},
        )

        def handle_message():
            # Mock the process_data method to avoid Claude API calls
            with patch.object(test_accounting_agent, "process_data") as mock_process:
                mock_process.return_value = None
                return asyncio.run(test_accounting_agent.handle_message(test_message))

        benchmark(handle_message)

        # Message handling should be very fast
        assert benchmark.stats["mean"] < 0.005  # Less than 5ms

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_agent_health_check_performance(self, benchmark, test_accounting_agent):
        """Benchmark agent health check performance."""

        def health_check():
            return asyncio.run(test_accounting_agent.health_check())

        result = benchmark(health_check)

        # Health check should be instantaneous
        assert benchmark.stats["mean"] < 0.001  # Less than 1ms
        assert result is not None
        assert "agent_id" in result
        assert "status" in result

    @pytest.mark.memory
    @pytest.mark.agent
    def test_agent_memory_leak_detection(self, test_accounting_agent, medium_dataset):
        """Test for memory leaks in long-running agent operations."""
        import gc

        process = psutil.Process()
        memory_samples = []

        # Run multiple iterations to detect memory leaks
        for iteration in range(10):
            # Force garbage collection
            gc.collect()

            # Take memory measurement
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Simulate agent work
            for i in range(20):
                test_data = {
                    "transaction_id": f"leak_test_{iteration}_{i}",
                    "amount": 50.0 + i,
                    "transaction_type": "expense",
                    "description": f"Leak test transaction {iteration}_{i}",
                    "category": "test",
                }

                with patch.object(test_accounting_agent, "analyze_with_claude") as mock_claude:
                    mock_claude.return_value = "Leak test analysis"
                    asyncio.run(test_accounting_agent.process_data(test_data))

            # Clear decision log to prevent it from growing indefinitely
            if len(test_accounting_agent.decisions_log) > 50:
                test_accounting_agent.decisions_log = test_accounting_agent.decisions_log[-25:]

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(memory_after - memory_before)

        # Check for memory leak pattern
        # Memory usage should stabilize and not continuously increase
        if len(memory_samples) >= 5:
            recent_average = sum(memory_samples[-5:]) / 5
            early_average = sum(memory_samples[:5]) / 5

            # Memory increase should be minimal over time
            memory_growth = recent_average - early_average
            assert (
                memory_growth < 10
            ), f"Potential memory leak detected: {memory_growth:.2f}MB growth"

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_agent_database_query_performance(
        self, benchmark, test_accounting_agent, medium_dataset
    ):
        """Benchmark agent database query performance."""

        def query_recent_transactions():
            session = test_accounting_agent.SessionLocal()
            try:
                # Simulate typical agent query
                recent_transactions = (
                    session.query(Transaction)
                    .filter(Transaction.transaction_date >= datetime.now() - timedelta(days=7))
                    .order_by(Transaction.transaction_date.desc())
                    .limit(50)
                    .all()
                )
                return len(recent_transactions)
            finally:
                session.close()

        result = benchmark(query_recent_transactions)

        # Database queries should be fast
        assert benchmark.stats["mean"] < 0.05  # Less than 50ms
        assert result >= 0

    @pytest.mark.stress
    @pytest.mark.agent
    def test_agent_high_volume_processing(self, test_accounting_agent, large_dataset):
        """Test agent performance with high volume of decisions."""
        start_time = time.time()
        successful_decisions = 0
        errors = 0

        # Process a large number of decisions
        for i in range(1000):
            try:
                test_data = {
                    "transaction_id": f"volume_test_tx_{i}",
                    "amount": 25.0 + (i % 500),
                    "transaction_type": "expense" if i % 3 else "income",
                    "description": f"Volume test transaction {i}",
                    "category": f"category_{i % 10}",
                }

                with patch.object(test_accounting_agent, "analyze_with_claude") as mock_claude:
                    mock_claude.return_value = f"Volume test analysis {i}"
                    result = asyncio.run(test_accounting_agent.process_data(test_data))

                    if result:
                        successful_decisions += 1

            except Exception:
                errors += 1
                if errors > 50:  # Stop if too many errors
                    break

        end_time = time.time()
        duration = end_time - start_time

        # Performance assertions
        assert successful_decisions >= 900, f"Only {successful_decisions} successful decisions"
        assert errors < 50, f"Too many errors: {errors}"
        assert duration < 60, f"Processing took too long: {duration:.2f}s"

        # Throughput check
        throughput = successful_decisions / duration
        assert throughput > 15, f"Throughput too low: {throughput:.2f} decisions/second"

    @pytest.mark.benchmark
    @pytest.mark.agent
    def test_agent_startup_performance(self, benchmark, performance_config, temp_db_path):
        """Benchmark agent initialization and startup time."""

        def create_agent():
            from agents.accounting_agent import AccountingAgent

            db_url = f"sqlite:///{temp_db_path}"
            agent = AccountingAgent(
                agent_id="startup_test_agent",
                api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
                config=performance_config["agents"]["accounting"],
                db_url=db_url,
            )
            return agent

        agent = benchmark(create_agent)

        # Agent startup should be fast
        assert benchmark.stats["mean"] < 0.1  # Less than 100ms
        assert agent is not None
        assert agent.agent_id == "startup_test_agent"
