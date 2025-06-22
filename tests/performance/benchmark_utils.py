"""
Benchmarking Utilities

Utilities for establishing baseline metrics, tracking performance trends,
and generating performance reports for the Business Agent System.
"""

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import psutil


@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    tags: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetric":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Represents the result of a benchmark run."""

    test_name: str
    metrics: List[PerformanceMetric]
    system_info: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    status: str  # 'success', 'failure', 'warning'
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "metrics": [m.to_dict() for m in self.metrics],
            "system_info": self.system_info,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["metrics"] = [PerformanceMetric.from_dict(m) for m in data["metrics"]]
        return cls(**data)


class SystemProfiler:
    """Utility for collecting system information and resource usage."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            import platform
            import socket

            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage("/").total if os.path.exists("/") else None,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    @staticmethod
    def get_resource_usage() -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            process = psutil.Process()

            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used": psutil.virtual_memory().used,
                "memory_available": psutil.virtual_memory().available,
                "process_memory": process.memory_info().rss,
                "process_cpu_percent": process.cpu_percent(),
                "disk_usage_percent": (
                    psutil.disk_usage("/").percent if os.path.exists("/") else None
                ),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


class PerformanceTracker:
    """Utility for tracking performance metrics over time."""

    def __init__(self, storage_path: str = "tests/performance/metrics_history.json"):
        self.storage_path = storage_path
        self.metrics_history: List[BenchmarkResult] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load metrics history from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self.metrics_history = [BenchmarkResult.from_dict(item) for item in data]
            except Exception as e:
                print(f"Warning: Could not load metrics history: {e}")
                self.metrics_history = []

    def _save_history(self) -> None:
        """Save metrics history to storage."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as f:
                data = [result.to_dict() for result in self.metrics_history]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics history: {e}")

    def record_benchmark(self, result: BenchmarkResult) -> None:
        """Record a benchmark result."""
        self.metrics_history.append(result)
        self._save_history()

    def get_baseline_metrics(self, test_name: str) -> Optional[Dict[str, float]]:
        """Get baseline metrics for a specific test."""
        test_results = [r for r in self.metrics_history if r.test_name == test_name]

        if not test_results:
            return None

        # Use the last 10 successful runs to establish baseline
        successful_results = [r for r in test_results if r.status == "success"][-10:]

        if not successful_results:
            return None

        baseline = {}

        # Calculate baseline for each metric
        for metric_name in {m.name for result in successful_results for m in result.metrics}:
            values = []
            for result in successful_results:
                for metric in result.metrics:
                    if metric.name == metric_name:
                        values.append(metric.value)

            if values:
                baseline[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return baseline

    def detect_regressions(self, test_name: str, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect performance regressions in recent test runs."""
        baseline = self.get_baseline_metrics(test_name)
        if not baseline:
            return []

        # Get the most recent result
        recent_results = [
            r for r in self.metrics_history if r.test_name == test_name and r.status == "success"
        ]
        if not recent_results:
            return []

        latest_result = recent_results[-1]
        regressions = []

        for metric in latest_result.metrics:
            if metric.name in baseline:
                baseline_mean = baseline[metric.name]["mean"]
                current_value = metric.value

                # Check for significant increase (regression)
                if baseline_mean > 0:
                    regression_ratio = (current_value - baseline_mean) / baseline_mean
                    if regression_ratio > threshold:
                        regressions.append(
                            {
                                "metric_name": metric.name,
                                "baseline_value": baseline_mean,
                                "current_value": current_value,
                                "regression_ratio": regression_ratio,
                                "unit": metric.unit,
                                "severity": (
                                    "high"
                                    if regression_ratio > 0.5
                                    else "medium" if regression_ratio > 0.25 else "low"
                                ),
                            }
                        )

        return regressions

    def get_performance_trends(
        self, test_name: str, days: int = 30
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get performance trends for a test over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            r
            for r in self.metrics_history
            if r.test_name == test_name and r.timestamp >= cutoff_date and r.status == "success"
        ]

        trends = {}

        for result in recent_results:
            for metric in result.metrics:
                if metric.name not in trends:
                    trends[metric.name] = []
                trends[metric.name].append((result.timestamp, metric.value))

        # Sort by timestamp
        for metric_name in trends:
            trends[metric_name].sort(key=lambda x: x[0])

        return trends


class BenchmarkRunner:
    """Utility for running and managing performance benchmarks."""

    def __init__(self, tracker: Optional[PerformanceTracker] = None):
        self.tracker = tracker or PerformanceTracker()
        self.profiler = SystemProfiler()

    def run_benchmark(self, test_name: str, test_function, *args, **kwargs) -> BenchmarkResult:
        """Run a benchmark and record the results."""
        start_time = time.time()
        start_resources = self.profiler.get_resource_usage()

        try:
            # Run the test function
            test_function(*args, **kwargs)
            status = "success"
            notes = ""
        except Exception as e:
            status = "failure"
            notes = str(e)

        end_time = time.time()
        end_resources = self.profiler.get_resource_usage()
        execution_time = end_time - start_time

        # Create performance metrics
        metrics = [
            PerformanceMetric(
                name="execution_time",
                value=execution_time,
                unit="seconds",
                timestamp=datetime.now(),
                category="timing",
                tags={"test": test_name},
            )
        ]

        # Add resource usage metrics if available
        if "error" not in start_resources and "error" not in end_resources:
            memory_delta = end_resources["process_memory"] - start_resources["process_memory"]
            metrics.extend(
                [
                    PerformanceMetric(
                        name="memory_delta",
                        value=memory_delta / 1024 / 1024,  # Convert to MB
                        unit="MB",
                        timestamp=datetime.now(),
                        category="memory",
                        tags={"test": test_name},
                    ),
                    PerformanceMetric(
                        name="cpu_usage",
                        value=end_resources["process_cpu_percent"],
                        unit="percent",
                        timestamp=datetime.now(),
                        category="cpu",
                        tags={"test": test_name},
                    ),
                ]
            )

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            test_name=test_name,
            metrics=metrics,
            system_info=self.profiler.get_system_info(),
            execution_time=execution_time,
            timestamp=datetime.now(),
            status=status,
            notes=notes,
        )

        # Record the result
        self.tracker.record_benchmark(benchmark_result)

        return benchmark_result

    def compare_with_baseline(self, test_name: str, tolerance: float = 0.1) -> Dict[str, Any]:
        """Compare latest results with baseline and return comparison report."""
        baseline = self.tracker.get_baseline_metrics(test_name)
        regressions = self.tracker.detect_regressions(test_name, tolerance)

        # Get latest result
        recent_results = [r for r in self.tracker.metrics_history if r.test_name == test_name]
        latest_result = recent_results[-1] if recent_results else None

        return {
            "test_name": test_name,
            "has_baseline": baseline is not None,
            "baseline_metrics": baseline,
            "latest_result": latest_result.to_dict() if latest_result else None,
            "regressions": regressions,
            "regression_count": len(regressions),
            "status": "pass" if not regressions else "regression_detected",
        }


class PerformanceReporter:
    """Utility for generating performance reports and visualizations."""

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker

    def generate_summary_report(
        self, output_path: str = "tests/performance/performance_report.html"
    ) -> None:
        """Generate a comprehensive performance summary report."""
        # Group results by test name
        test_groups = {}
        for result in self.tracker.metrics_history:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)

        # Generate HTML report
        html_content = self._generate_html_report(test_groups)

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Performance report generated: {output_path}")

    def _generate_html_report(self, test_groups: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate HTML content for the performance report."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Agent System - Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .test-section { margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .metric { margin: 10px 0; }
                .regression { color: red; font-weight: bold; }
                .improvement { color: green; font-weight: bold; }
                .stable { color: blue; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Business Agent System - Performance Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Total test runs: {total_runs}</p>
                <p>Test categories: {test_count}</p>
            </div>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_runs=len(self.tracker.metrics_history),
            test_count=len(test_groups),
        )

        # Add summary table
        html_content += """
            <h2>Test Summary</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Last Run</th>
                    <th>Status</th>
                    <th>Avg Execution Time</th>
                    <th>Regressions</th>
                </tr>
        """

        for test_name, results in test_groups.items():
            recent_results = [r for r in results if r.status == "success"][-10:]
            if recent_results:
                avg_time = sum(r.execution_time for r in recent_results) / len(recent_results)
                latest_result = results[-1]
                regressions = self.tracker.detect_regressions(test_name)

                html_content += f"""
                    <tr>
                        <td>{test_name}</td>
                        <td>{latest_result.timestamp.strftime("%Y-%m-%d %H:%M")}</td>
                        <td>{latest_result.status}</td>
                        <td>{avg_time:.3f}s</td>
                        <td>{len(regressions)} detected</td>
                    </tr>
                """

        html_content += "</table>"

        # Add detailed sections for each test
        for test_name, results in test_groups.items():
            html_content += '<div class="test-section">'
            html_content += f"<h3>{test_name}</h3>"

            # Get baseline and trends
            baseline = self.tracker.get_baseline_metrics(test_name)
            regressions = self.tracker.detect_regressions(test_name)
            self.tracker.get_performance_trends(test_name, days=30)

            if baseline:
                html_content += "<h4>Baseline Metrics</h4>"
                html_content += "<table><tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>"
                for metric_name, stats in baseline.items():
                    html_content += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{stats['mean']:.3f}</td>
                            <td>{stats['std']:.3f}</td>
                            <td>{stats['min']:.3f}</td>
                            <td>{stats['max']:.3f}</td>
                        </tr>
                    """
                html_content += "</table>"

            if regressions:
                html_content += "<h4>Performance Regressions</h4>"
                for regression in regressions:
                    html_content += f"""
                        <div class="regression">
                            {regression['metric_name']}: {regression['regression_ratio']:.1%} slower
                            ({regression['baseline_value']:.3f} → {regression['current_value']:.3f} {regression['unit']})
                        </div>
                    """

            html_content += "</div>"

        html_content += "</body></html>"
        return html_content

    def create_trend_charts(
        self, test_name: str, output_dir: str = "tests/performance/charts"
    ) -> List[str]:
        """Create trend charts for a specific test."""
        trends = self.tracker.get_performance_trends(test_name, days=30)

        if not trends:
            return []

        os.makedirs(output_dir, exist_ok=True)
        chart_files = []

        plt.style.use("seaborn-v0_8")

        for metric_name, data_points in trends.items():
            if len(data_points) < 2:
                continue

            timestamps, values = zip(*data_points)

            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, values, marker="o", linewidth=2, markersize=4)
            plt.title(f"{test_name} - {metric_name} Trend")
            plt.xlabel("Time")
            plt.ylabel(f"{metric_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()

            chart_file = os.path.join(output_dir, f"{test_name}_{metric_name}_trend.png")
            plt.savefig(chart_file, dpi=150, bbox_inches="tight")
            plt.close()

            chart_files.append(chart_file)

        return chart_files

    def create_comparison_chart(
        self,
        test_names: List[str],
        metric_name: str = "execution_time",
        output_path: str = "tests/performance/charts/comparison.png",
    ) -> str:
        """Create a comparison chart for multiple tests."""
        plt.figure(figsize=(14, 8))

        test_data = []

        for test_name in test_names:
            trends = self.tracker.get_performance_trends(test_name, days=30)
            if metric_name in trends:
                timestamps, values = zip(*trends[metric_name])
                test_data.append(
                    {"test_name": test_name, "timestamps": timestamps, "values": values}
                )

        for data in test_data:
            plt.plot(
                data["timestamps"],
                data["values"],
                marker="o",
                label=data["test_name"],
                linewidth=2,
                markersize=3,
            )

        plt.title(f"Performance Comparison - {metric_name}")
        plt.xlabel("Time")
        plt.ylabel(metric_name)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path
