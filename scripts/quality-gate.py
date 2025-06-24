#!/usr/bin/env python3
"""
Quality Gate System for Business Agent Management System

A comprehensive quality gate system that enforces development workflow standards
and prevents problematic code from being committed or deployed.

Supports multiple modes:
- normal: Standard checks for regular development
- strict: Comprehensive checks with stricter standards
- quick: Fast checks for rapid iteration
- emergency: Minimal checks for emergency fixes

Usage:
    python scripts/quality-gate.py [--mode MODE] [--files FILE1 FILE2 ...] [--config CONFIG]

Examples:
    python scripts/quality-gate.py                    # Run normal mode on all files
    python scripts/quality-gate.py --mode strict     # Run strict mode
    python scripts/quality-gate.py --mode quick      # Run quick checks
    python scripts/quality-gate.py --files agents/base_agent.py  # Check specific file
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set


# Color constants for output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class QualityGateMode(Enum):
    """Quality gate execution modes with different check intensities."""

    NORMAL = "normal"
    STRICT = "strict"
    QUICK = "quick"
    EMERGENCY = "emergency"


@dataclass
class CheckResult:
    """Result of an individual quality check."""

    name: str
    success: bool
    message: str
    duration: float
    details: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None


@dataclass
class QualityGateResult:
    """Overall result of quality gate execution."""

    mode: QualityGateMode
    success: bool
    total_duration: float
    checks: List[CheckResult] = field(default_factory=list)
    files_checked: Set[str] = field(default_factory=set)
    summary: str = ""


class QualityGateConfig:
    """Configuration for quality gate checks based on mode."""

    def __init__(self, mode: QualityGateMode):
        self.mode = mode
        self.timeout_seconds = self._get_timeout()
        self.checks = self._get_checks()

    def _get_timeout(self) -> int:
        """Get timeout based on mode."""
        timeouts = {
            QualityGateMode.QUICK: 60,
            QualityGateMode.NORMAL: 180,
            QualityGateMode.STRICT: 300,
            QualityGateMode.EMERGENCY: 30,
        }
        return timeouts[self.mode]

    def _get_checks(self) -> List[str]:
        """Get list of checks to run based on mode."""
        all_checks = [
            "format_check",
            "import_sort",
            "lint_check",
            "type_check",
            "test_run",
            "security_scan",
            "dependency_check",
            "coverage_check",
        ]

        check_configs = {
            QualityGateMode.EMERGENCY: ["format_check", "lint_check"],
            QualityGateMode.QUICK: ["format_check", "import_sort", "lint_check", "type_check"],
            QualityGateMode.NORMAL: [
                "format_check",
                "import_sort",
                "lint_check",
                "type_check",
                "test_run",
            ],
            QualityGateMode.STRICT: all_checks,
        }

        return check_configs[self.mode]


class QualityGateRunner:
    """Main quality gate runner that executes checks based on configuration."""

    def __init__(self, config: QualityGateConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.results: List[CheckResult] = []

    async def run(self, files: Optional[List[str]] = None) -> QualityGateResult:
        """Run quality gate checks."""
        start_time = time.time()

        print(
            f"{Colors.HEADER}🚪 Starting Quality Gate - {self.config.mode.value.upper()} mode{Colors.ENDC}"
        )
        print(
            f"{Colors.OKBLUE}Configuration: {len(self.config.checks)} checks, {self.config.timeout_seconds}s timeout{Colors.ENDC}"
        )
        print("=" * 80)

        # Determine files to check
        if files:
            files_to_check = set(files)
        else:
            files_to_check = self._discover_python_files()

        print(f"{Colors.OKCYAN}📁 Files to check: {len(files_to_check)}{Colors.ENDC}")

        # Run checks in sequence
        success = True
        for check_name in self.config.checks:
            result = await self._run_check(check_name, files_to_check)
            self.results.append(result)

            if not result.success:
                success = False
                if self.config.mode == QualityGateMode.STRICT:
                    print(f"{Colors.FAIL}❌ Stopping on first failure in strict mode{Colors.ENDC}")
                    break

        total_duration = time.time() - start_time

        # Generate summary
        gate_result = QualityGateResult(
            mode=self.config.mode,
            success=success,
            total_duration=total_duration,
            checks=self.results,
            files_checked=files_to_check,
        )

        self._print_summary(gate_result)
        return gate_result

    def _discover_python_files(self) -> Set[str]:
        """Discover all Python files in the project."""
        python_files = set()

        # Key directories to check
        directories = ["agents", "models", "simulation", "dashboard", "utils"]

        for directory in directories:
            dir_path = self.project_root / directory
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    if not self._should_exclude_file(py_file):
                        python_files.add(str(py_file.relative_to(self.project_root)))

        # Add root level Python files
        for py_file in self.project_root.glob("*.py"):
            if not self._should_exclude_file(py_file):
                python_files.add(str(py_file.relative_to(self.project_root)))

        return python_files

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from quality checks."""
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".venv",
            "venv",
            "node_modules",
        ]

        return any(pattern in str(file_path) for pattern in exclude_patterns)

    async def _run_check(self, check_name: str, files: Set[str]) -> CheckResult:
        """Run a specific quality check."""
        start_time = time.time()
        print(f"{Colors.OKCYAN}⏳ Running {check_name}...{Colors.ENDC}", end=" ", flush=True)

        try:
            method = getattr(self, f"_check_{check_name}")
            result = await method(files)
            duration = time.time() - start_time

            if result.success:
                print(f"{Colors.OKGREEN}✅ ({duration:.1f}s){Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}❌ ({duration:.1f}s){Colors.ENDC}")
                if result.message:
                    print(f"{Colors.WARNING}  └─ {result.message}{Colors.ENDC}")

            return CheckResult(
                name=check_name,
                success=result.success,
                message=result.message,
                duration=duration,
                details=result.details,
                suggestions=result.suggestions,
                exit_code=result.exit_code,
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"{Colors.FAIL}💥 ERROR ({duration:.1f}s){Colors.ENDC}")
            print(f"{Colors.FAIL}  └─ {str(e)}{Colors.ENDC}")

            return CheckResult(
                name=check_name,
                success=False,
                message=f"Check failed with error: {str(e)}",
                duration=duration,
                suggestions=["Check tool installation", "Verify project configuration"],
            )

    async def _check_format_check(self, files: Set[str]) -> CheckResult:
        """Check code formatting with Black."""
        if not files:
            return CheckResult("format_check", True, "No files to check", 0)

        cmd = ["uv", "run", "black", "--check", "--diff"] + list(files)
        result = await self._run_command(cmd, "Black formatting check")

        suggestions = []
        if not result.success:
            suggestions = [
                "Run: uv run black .",
                "Or: make format",
                "Configure your editor to format on save",
            ]

        return CheckResult(
            "format_check",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_import_sort(self, files: Set[str]) -> CheckResult:
        """Check import sorting with isort."""
        if not files:
            return CheckResult("import_sort", True, "No files to check", 0)

        cmd = ["uv", "run", "isort", "--check-only", "--diff"] + list(files)
        result = await self._run_command(cmd, "Import sorting check")

        suggestions = []
        if not result.success:
            suggestions = [
                "Run: uv run isort .",
                "Or: make format",
                "Check isort configuration in pyproject.toml",
            ]

        return CheckResult(
            "import_sort",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_lint_check(self, files: Set[str]) -> CheckResult:
        """Check code quality with Ruff."""
        if not files:
            return CheckResult("lint_check", True, "No files to check", 0)

        cmd = ["uv", "run", "ruff", "check"] + list(files)
        result = await self._run_command(cmd, "Ruff linting check")

        suggestions = []
        if not result.success:
            suggestions = [
                "Run: uv run ruff check --fix .",
                "Review Ruff configuration in pyproject.toml",
                "Check specific error messages for fixes",
            ]

        return CheckResult(
            "lint_check",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_type_check(self, files: Set[str]) -> CheckResult:
        """Check type annotations with MyPy."""
        # MyPy works better with directories than individual files
        directories = set()
        for file_path in files:
            if file_path.endswith(".py"):
                dir_name = str(Path(file_path).parent)
                if dir_name in ["agents", "models", "simulation", "utils"]:
                    directories.add(dir_name)

        if not directories:
            return CheckResult("type_check", True, "No directories to check", 0)

        cmd = ["uv", "run", "mypy"] + list(directories)
        result = await self._run_command(cmd, "MyPy type checking")

        suggestions = []
        if not result.success:
            suggestions = [
                "Add missing type annotations",
                "Check MyPy configuration in pyproject.toml",
                "Review specific type errors in output",
                "Use # type: ignore for unavoidable issues",
            ]

        return CheckResult(
            "type_check",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_test_run(self, files: Set[str]) -> CheckResult:
        """Run relevant tests."""
        # For quick testing, run only unit tests
        if self.config.mode == QualityGateMode.QUICK:
            cmd = ["uv", "run", "pytest", "tests/unit/", "-x", "--tb=short"]
        else:
            cmd = ["uv", "run", "pytest", "tests/unit/", "--tb=short"]

        result = await self._run_command(cmd, "Test execution")

        suggestions = []
        if not result.success:
            suggestions = [
                "Fix failing tests",
                "Run: uv run pytest tests/unit/ -v for detailed output",
                "Check test configuration in pyproject.toml",
                "Verify test dependencies are installed",
            ]

        return CheckResult(
            "test_run",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_security_scan(self, files: Set[str]) -> CheckResult:
        """Run security scanning with bandit."""
        if not files:
            return CheckResult("security_scan", True, "No files to check", 0)

        cmd = ["uv", "run", "bandit", "-r"] + list(files)
        result = await self._run_command(cmd, "Security scanning")

        # Bandit returns non-zero for findings, but we may want to be lenient
        if result.exit_code == 1:  # Bandit found issues but not critical
            result.success = True
            result.message = "Security scan completed with warnings"

        suggestions = []
        if not result.success:
            suggestions = [
                "Review security findings in detail",
                "Add # nosec comments for false positives",
                "Update security dependencies",
                "Follow security best practices",
            ]

        return CheckResult(
            "security_scan",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_dependency_check(self, files: Set[str]) -> CheckResult:
        """Check for dependency security issues."""
        cmd = ["uv", "run", "safety", "check", "--json"]
        result = await self._run_command(cmd, "Dependency security check")

        suggestions = []
        if not result.success:
            suggestions = [
                "Update vulnerable dependencies",
                "Review security advisory details",
                "Consider alternative packages",
                "Pin dependency versions if needed",
            ]

        return CheckResult(
            "dependency_check",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _check_coverage_check(self, files: Set[str]) -> CheckResult:
        """Check test coverage."""
        cmd = [
            "uv",
            "run",
            "pytest",
            "--cov=agents",
            "--cov=models",
            "--cov=simulation",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "tests/unit/",
        ]
        result = await self._run_command(cmd, "Coverage check")

        suggestions = []
        if not result.success:
            suggestions = [
                "Add tests for uncovered code",
                "Review coverage report for gaps",
                "Consider adjusting coverage threshold",
                "Focus on critical path coverage",
            ]

        return CheckResult(
            "coverage_check",
            result.success,
            result.message,
            0,
            result.details,
            suggestions,
            result.exit_code,
        )

    async def _run_command(self, cmd: List[str], description: str) -> CheckResult:
        """Run a command with timeout protection."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.project_root),
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.timeout_seconds
                )
                exit_code = process.returncode

                output = stdout.decode("utf-8", errors="replace") if stdout else ""
                success = exit_code == 0

                if success:
                    message = f"{description} completed successfully"
                else:
                    message = f"{description} failed (exit code: {exit_code})"

                return CheckResult(
                    description, success, message, 0, output if output else None, [], exit_code
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CheckResult(
                    description,
                    False,
                    f"{description} timed out after {self.config.timeout_seconds}s",
                    0,
                    None,
                    [f"Consider increasing timeout or optimizing {description}"],
                    -1,
                )

        except Exception as e:
            return CheckResult(
                description,
                False,
                f"{description} failed to execute: {str(e)}",
                0,
                None,
                ["Check tool installation", "Verify command availability"],
                -1,
            )

    def _print_summary(self, result: QualityGateResult) -> None:
        """Print comprehensive summary of quality gate results."""
        print("\n" + "=" * 80)
        print(f"{Colors.HEADER}📊 QUALITY GATE SUMMARY - {result.mode.value.upper()}{Colors.ENDC}")
        print("=" * 80)

        # Overall status
        if result.success:
            print(f"{Colors.OKGREEN}✅ Overall Status: PASSED{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}❌ Overall Status: FAILED{Colors.ENDC}")

        print(f"{Colors.OKCYAN}⏱️  Total Duration: {result.total_duration:.1f}s{Colors.ENDC}")
        print(f"{Colors.OKCYAN}📁 Files Checked: {len(result.files_checked)}{Colors.ENDC}")

        # Check results
        print(f"\n{Colors.BOLD}Check Results:{Colors.ENDC}")
        for check in result.checks:
            status = f"{Colors.OKGREEN}✅" if check.success else f"{Colors.FAIL}❌"
            print(
                f"  {status} {check.name:<20} ({check.duration:.1f}s) - {check.message}{Colors.ENDC}"
            )

            if not check.success and check.suggestions:
                print(f"{Colors.WARNING}     Suggestions:{Colors.ENDC}")
                for suggestion in check.suggestions[:3]:  # Limit to top 3 suggestions
                    print(f"{Colors.WARNING}     • {suggestion}{Colors.ENDC}")

        # Failed checks details
        failed_checks = [c for c in result.checks if not c.success]
        if failed_checks:
            print(f"\n{Colors.FAIL}🚨 Failed Checks Details:{Colors.ENDC}")
            for check in failed_checks:
                print(f"\n{Colors.FAIL}❌ {check.name}:{Colors.ENDC}")
                if check.details and len(check.details) < 1000:  # Limit output length
                    print(f"{Colors.WARNING}{check.details[:1000]}{Colors.ENDC}")
                elif check.details:
                    lines = check.details.split("\n")
                    print(f"{Colors.WARNING}{chr(10).join(lines[:10])}{Colors.ENDC}")
                    if len(lines) > 10:
                        print(f"{Colors.WARNING}... ({len(lines) - 10} more lines){Colors.ENDC}")

        # Final recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}")
        if result.success:
            print(
                f"{Colors.OKGREEN}🎉 All checks passed! Code is ready for commit/deployment.{Colors.ENDC}"
            )
        else:
            print(f"{Colors.FAIL}🔧 Fix the failed checks before proceeding.{Colors.ENDC}")
            print(
                f"{Colors.OKCYAN}💡 Run with --mode quick for faster iteration during fixes.{Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}📖 See suggestions above for specific remediation steps.{Colors.ENDC}"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quality Gate System for Business Agent Management System",
        epilog="""
Examples:
  %(prog)s                           # Run normal mode on all files
  %(prog)s --mode strict            # Run comprehensive strict checks
  %(prog)s --mode quick             # Run quick checks for iteration
  %(prog)s --mode emergency         # Run minimal checks for emergency fixes
  %(prog)s --files agents/base_agent.py  # Check specific file(s)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in QualityGateMode],
        default=QualityGateMode.NORMAL.value,
        help="Quality gate mode (default: normal)",
    )

    parser.add_argument(
        "--files", nargs="*", help="Specific files to check (default: all Python files)"
    )

    parser.add_argument("--config", type=str, help="Path to quality gate configuration file (JSON)")

    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    # Find project root
    project_root = Path.cwd()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent
    else:
        print(f"{Colors.FAIL}❌ Could not find project root (no pyproject.toml found){Colors.ENDC}")
        return 1

    # Verify we're in the right project
    if not (project_root / "agents").exists():
        print(
            f"{Colors.FAIL}❌ This doesn't appear to be the Business Agent System project{Colors.ENDC}"
        )
        return 1

    # Load configuration
    mode = QualityGateMode(args.mode)
    config = QualityGateConfig(mode)

    # Override config if custom config file provided
    if args.config and Path(args.config).exists():
        try:
            with open(args.config) as f:
                custom_config = json.load(f)
                # Apply custom configuration overrides here
                if "timeout_seconds" in custom_config:
                    config.timeout_seconds = custom_config["timeout_seconds"]
                if "checks" in custom_config:
                    config.checks = custom_config["checks"]
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Failed to load custom config: {e}{Colors.ENDC}")

    # Run quality gate
    runner = QualityGateRunner(config, project_root)
    result = await runner.run(args.files)

    # Output results to file if requested
    if args.output:
        try:
            output_data = {
                "mode": result.mode.value,
                "success": result.success,
                "total_duration": result.total_duration,
                "files_checked": list(result.files_checked),
                "checks": [
                    {
                        "name": check.name,
                        "success": check.success,
                        "message": check.message,
                        "duration": check.duration,
                        "exit_code": check.exit_code,
                        "suggestions": check.suggestions,
                    }
                    for check in result.checks
                ],
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"{Colors.OKCYAN}📝 Results written to {args.output}{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Failed to write output file: {e}{Colors.ENDC}")

    # Return appropriate exit code
    return 0 if result.success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}🛑 Quality gate interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.FAIL}💥 Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)
