"""
Performance Test Runner

Main script for running performance tests, generating reports,
and tracking performance trends over time.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.performance.benchmark_utils import (
    PerformanceTracker, 
    BenchmarkRunner, 
    PerformanceReporter,
    SystemProfiler
)


class PerformanceTestRunner:
    """Main class for running and managing performance tests."""
    
    def __init__(self, output_dir: str = "tests/performance/results"):
        self.output_dir = output_dir
        self.tracker = PerformanceTracker()
        self.runner = BenchmarkRunner(self.tracker)
        self.reporter = PerformanceReporter(self.tracker)
        self.profiler = SystemProfiler()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run_performance_tests(self, test_categories: Optional[List[str]] = None,
                            generate_report: bool = True,
                            compare_baseline: bool = True) -> Dict[str, Any]:
        """Run performance tests and generate reports."""
        
        # Default to all categories if none specified
        if not test_categories:
            test_categories = ["agent", "database", "simulation", "dashboard", "stress"]
        
        print(f"Running performance tests for categories: {', '.join(test_categories)}")
        print(f"Output directory: {self.output_dir}")
        
        # Build pytest command
        pytest_args = [
            "python", "-m", "pytest",
            "tests/performance/",
            "-v",
            "--tb=short",
            f"--benchmark-json={self.output_dir}/benchmark_results.json",
            "--benchmark-compare-fail=min:10%",  # Fail if 10% slower than baseline
            "--benchmark-sort=mean",
        ]
        
        # Add category markers
        if test_categories:
            markers = " or ".join(test_categories)
            pytest_args.extend(["-m", markers])
        
        # Add coverage if requested
        if os.getenv("COVERAGE", "false").lower() == "true":
            pytest_args.extend([
                "--cov=agents",
                "--cov=models", 
                "--cov=simulation",
                "--cov=dashboard",
                f"--cov-report=html:{self.output_dir}/coverage_html",
                f"--cov-report=xml:{self.output_dir}/coverage.xml"
            ])
        
        # Run tests
        print("Starting performance test execution...")
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                pytest_args,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("Performance tests timed out after 1 hour!")
            return {
                "status": "timeout",
                "message": "Tests timed out after 1 hour",
                "duration": 3600
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save test output
        with open(f"{self.output_dir}/test_output.log", "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
        
        print(f"Performance tests completed in {duration:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        # Process results
        test_results = {
            "status": "success" if success else "failure",
            "exit_code": result.returncode,
            "duration": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "categories_tested": test_categories,
            "system_info": self.profiler.get_system_info()
        }
        
        # Load benchmark results if available
        benchmark_file = f"{self.output_dir}/benchmark_results.json"
        if os.path.exists(benchmark_file):
            try:
                with open(benchmark_file, "r") as f:
                    benchmark_data = json.load(f)
                    test_results["benchmark_data"] = benchmark_data
                    
                    # Extract key metrics
                    if "benchmarks" in benchmark_data:
                        test_results["benchmark_summary"] = self._summarize_benchmarks(
                            benchmark_data["benchmarks"]
                        )
            except Exception as e:
                print(f"Warning: Could not load benchmark results: {e}")
        
        # Generate reports if requested
        if generate_report:
            try:
                self._generate_reports(test_results)
                test_results["reports_generated"] = True
            except Exception as e:
                print(f"Warning: Could not generate reports: {e}")
                test_results["reports_generated"] = False
                test_results["report_error"] = str(e)
        
        # Compare with baseline if requested
        if compare_baseline:
            try:
                baseline_comparison = self._compare_with_baseline()
                test_results["baseline_comparison"] = baseline_comparison
            except Exception as e:
                print(f"Warning: Could not compare with baseline: {e}")
                test_results["baseline_comparison_error"] = str(e)
        
        # Save overall test results
        with open(f"{self.output_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        return test_results
    
    def _summarize_benchmarks(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize benchmark results."""
        if not benchmarks:
            return {}
        
        summary = {
            "total_benchmarks": len(benchmarks),
            "fastest_test": None,
            "slowest_test": None,
            "category_averages": {},
            "overall_stats": {}
        }
        
        # Find fastest and slowest
        sorted_by_mean = sorted(benchmarks, key=lambda x: x["stats"]["mean"])
        summary["fastest_test"] = {
            "name": sorted_by_mean[0]["name"],
            "mean_time": sorted_by_mean[0]["stats"]["mean"]
        }
        summary["slowest_test"] = {
            "name": sorted_by_mean[-1]["name"],
            "mean_time": sorted_by_mean[-1]["stats"]["mean"]
        }
        
        # Calculate overall statistics
        all_means = [b["stats"]["mean"] for b in benchmarks]
        summary["overall_stats"] = {
            "mean_of_means": sum(all_means) / len(all_means),
            "total_time": sum(all_means),
            "fastest_mean": min(all_means),
            "slowest_mean": max(all_means)
        }
        
        # Group by category (extract from test name)
        categories = {}
        for benchmark in benchmarks:
            test_name = benchmark["name"]
            category = "unknown"
            
            if "agent" in test_name.lower():
                category = "agent"
            elif "database" in test_name.lower():
                category = "database"
            elif "simulation" in test_name.lower():
                category = "simulation"
            elif "dashboard" in test_name.lower():
                category = "dashboard"
            elif "stress" in test_name.lower():
                category = "stress"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(benchmark["stats"]["mean"])
        
        # Calculate category averages
        for category, times in categories.items():
            summary["category_averages"][category] = {
                "count": len(times),
                "average_time": sum(times) / len(times),
                "total_time": sum(times)
            }
        
        return summary
    
    def _generate_reports(self, test_results: Dict[str, Any]) -> None:
        """Generate performance reports."""
        print("Generating performance reports...")
        
        # Generate summary report
        summary_report_path = f"{self.output_dir}/performance_summary.html"
        self.reporter.generate_summary_report(summary_report_path)
        
        # Generate trend charts if we have enough data
        unique_tests = set()
        for result in self.tracker.metrics_history:
            unique_tests.add(result.test_name)
        
        chart_dir = f"{self.output_dir}/charts"
        os.makedirs(chart_dir, exist_ok=True)
        
        chart_files = []
        for test_name in unique_tests:
            try:
                test_charts = self.reporter.create_trend_charts(test_name, chart_dir)
                chart_files.extend(test_charts)
            except Exception as e:
                print(f"Warning: Could not create charts for {test_name}: {e}")
        
        # Generate comparison chart for key tests
        if len(unique_tests) > 1:
            try:
                key_tests = [t for t in unique_tests if "agent" in t or "database" in t][:5]
                if key_tests:
                    comparison_chart = self.reporter.create_comparison_chart(
                        key_tests, 
                        output_path=f"{chart_dir}/performance_comparison.png"
                    )
                    chart_files.append(comparison_chart)
            except Exception as e:
                print(f"Warning: Could not create comparison chart: {e}")
        
        print(f"Generated {len(chart_files)} chart files")
        
        # Create markdown report
        self._create_markdown_report(test_results, chart_files)
    
    def _create_markdown_report(self, test_results: Dict[str, Any], chart_files: List[str]) -> None:
        """Create a markdown performance report."""
        report_path = f"{self.output_dir}/PERFORMANCE_REPORT.md"
        
        with open(report_path, "w") as f:
            f.write("# Business Agent System - Performance Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test Summary
            f.write("## Test Summary\n\n")
            f.write(f"- **Status:** {test_results['status']}\n")
            f.write(f"- **Duration:** {test_results['duration']:.2f} seconds\n")
            f.write(f"- **Categories Tested:** {', '.join(test_results['categories_tested'])}\n")
            
            if "benchmark_summary" in test_results:
                summary = test_results["benchmark_summary"]
                f.write(f"- **Total Benchmarks:** {summary['total_benchmarks']}\n")
                f.write(f"- **Fastest Test:** {summary['fastest_test']['name']} ({summary['fastest_test']['mean_time']:.3f}s)\n")
                f.write(f"- **Slowest Test:** {summary['slowest_test']['name']} ({summary['slowest_test']['mean_time']:.3f}s)\n")
            
            f.write("\n")
            
            # System Information
            f.write("## System Information\n\n")
            system_info = test_results["system_info"]
            f.write(f"- **Platform:** {system_info.get('platform', 'Unknown')}\n")
            f.write(f"- **Python Version:** {system_info.get('python_version', 'Unknown')}\n")
            f.write(f"- **CPU Count:** {system_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"- **Memory Total:** {system_info.get('memory_total', 0) / 1024 / 1024 / 1024:.1f} GB\n")
            f.write("\n")
            
            # Category Performance
            if "benchmark_summary" in test_results and "category_averages" in test_results["benchmark_summary"]:
                f.write("## Performance by Category\n\n")
                for category, stats in test_results["benchmark_summary"]["category_averages"].items():
                    f.write(f"### {category.title()}\n")
                    f.write(f"- **Test Count:** {stats['count']}\n")
                    f.write(f"- **Average Time:** {stats['average_time']:.3f}s\n")
                    f.write(f"- **Total Time:** {stats['total_time']:.3f}s\n\n")
            
            # Baseline Comparison
            if "baseline_comparison" in test_results:
                f.write("## Baseline Comparison\n\n")
                # Add baseline comparison details here
                f.write("Baseline comparison results available in test_results.json\n\n")
            
            # Charts
            if chart_files:
                f.write("## Performance Charts\n\n")
                for chart_file in chart_files:
                    chart_name = os.path.basename(chart_file)
                    relative_path = os.path.relpath(chart_file, self.output_dir)
                    f.write(f"![{chart_name}]({relative_path})\n\n")
            
            # Files Generated
            f.write("## Generated Files\n\n")
            f.write("- `test_results.json` - Complete test results\n")
            f.write("- `benchmark_results.json` - Detailed benchmark data\n")
            f.write("- `performance_summary.html` - Interactive HTML report\n")
            f.write("- `test_output.log` - Complete test execution log\n")
            f.write("- `charts/` - Performance trend charts\n")
        
        print(f"Markdown report generated: {report_path}")
    
    def _compare_with_baseline(self) -> Dict[str, Any]:
        """Compare current results with baseline metrics."""
        print("Comparing with baseline metrics...")
        
        # This would implement baseline comparison logic
        # For now, return a placeholder
        return {
            "baseline_available": False,
            "regressions_detected": 0,
            "improvements_detected": 0,
            "status": "no_baseline"
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run Business Agent System performance tests")
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["agent", "database", "simulation", "dashboard", "stress"],
        help="Test categories to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="tests/performance/results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = PerformanceTestRunner(args.output_dir)
    
    # Run tests
    results = runner.run_performance_tests(
        test_categories=args.categories,
        generate_report=not args.no_report,
        compare_baseline=not args.no_baseline
    )
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE TEST SUMMARY")
    print("="*50)
    print(f"Status: {results['status']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if "benchmark_summary" in results:
        summary = results["benchmark_summary"]
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"Fastest test: {summary['fastest_test']['name']} ({summary['fastest_test']['mean_time']:.3f}s)")
        print(f"Slowest test: {summary['slowest_test']['name']} ({summary['slowest_test']['mean_time']:.3f}s)")
    
    print(f"\nResults saved to: {args.output_dir}")
    
    # Exit with appropriate code
    sys.exit(0 if results["status"] == "success" else 1)


if __name__ == "__main__":
    main()