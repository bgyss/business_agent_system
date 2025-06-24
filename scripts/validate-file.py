#!/usr/bin/env python3
"""
Individual File Validation Utility for Business Agent Management System

A focused validation tool for checking individual files quickly during development.
Provides fast feedback for specific files without running full project checks.

Features:
- Quick validation of single files
- Targeted checks relevant to the file type
- Smart test discovery and execution
- Minimal overhead for rapid iteration
- Integration with editor workflows

Usage:
    python scripts/validate-file.py FILE [--checks CHECK1,CHECK2,...] [--with-tests]

Examples:
    python scripts/validate-file.py agents/base_agent.py
    python scripts/validate-file.py models/financial.py --with-tests
    python scripts/validate-file.py agents/inventory_agent.py --checks format,lint,type
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


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


@dataclass
class ValidationResult:
    """Result of a file validation check."""

    check_name: str
    success: bool
    message: str
    duration: float
    output: Optional[str] = None
    suggestions: List[str] = None


class FileValidator:
    """Validates individual files quickly and efficiently."""

    # Default checks for different file patterns
    DEFAULT_CHECKS = {
        "agents/*.py": ["format", "imports", "lint", "type"],
        "models/*.py": ["format", "imports", "lint", "type"],
        "simulation/*.py": ["format", "imports", "lint", "type"],
        "utils/*.py": ["format", "imports", "lint", "type"],
        "dashboard/*.py": ["format", "imports", "lint"],  # Skip type checks for UI
        "tests/*.py": ["format", "imports", "lint"],  # Skip type checks for tests
        "*.py": ["format", "imports", "lint"],  # Default fallback
    }

    def __init__(self, project_root: Path, timeout: int = 30):
        self.project_root = project_root
        self.timeout = timeout
        self.results: List[ValidationResult] = []

    async def validate_file(
        self, file_path: str, checks: Optional[List[str]] = None, with_tests: bool = False
    ) -> Tuple[bool, List[ValidationResult]]:
        """Validate a single file with specified checks."""

        # Resolve file path
        full_path = self.project_root / file_path
        if not full_path.exists():
            return False, [
                ValidationResult(
                    "file_check",
                    False,
                    f"File not found: {file_path}",
                    0.0,
                    suggestions=["Check file path", "Ensure file exists"],
                )
            ]

        # Determine checks to run
        if checks is None:
            checks = self._get_checks_for_file(file_path)

        print(f"{Colors.HEADER}🔍 Validating: {file_path}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Checks: {', '.join(checks)}{Colors.ENDC}")
        print("=" * 60)

        # Run each check
        self.results = []
        overall_success = True

        for check in checks:
            result = await self._run_check(check, file_path)
            self.results.append(result)

            if not result.success:
                overall_success = False

        # Run related tests if requested
        if with_tests:
            test_result = await self._run_related_tests(file_path)
            self.results.append(test_result)
            if not test_result.success:
                overall_success = False

        # Print summary
        self._print_file_summary(file_path, overall_success)

        return overall_success, self.results

    def _get_checks_for_file(self, file_path: str) -> List[str]:
        """Determine appropriate checks for a file based on its path."""

        # Try specific patterns first
        for pattern, checks in self.DEFAULT_CHECKS.items():
            if self._matches_pattern(file_path, pattern):
                return checks

        # Fallback to basic checks
        return ["format", "imports", "lint"]

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches a pattern."""
        if "*" not in pattern:
            return file_path == pattern

        # Simple glob-like matching
        if pattern.startswith("*/"):
            return file_path.endswith(pattern[2:])
        if pattern.endswith("/*"):
            return file_path.startswith(pattern[:-2])
        if "/*." in pattern:
            dir_part, ext_part = pattern.split("/*.")
            return file_path.startswith(dir_part) and file_path.endswith("." + ext_part)

        return False

    async def _run_check(self, check_name: str, file_path: str) -> ValidationResult:
        """Run a specific validation check on the file."""
        start_time = time.time()
        print(f"{Colors.OKCYAN}⏳ {check_name.capitalize()}...{Colors.ENDC}", end=" ", flush=True)

        try:
            if check_name == "format":
                result = await self._check_format(file_path)
            elif check_name == "imports":
                result = await self._check_imports(file_path)
            elif check_name == "lint":
                result = await self._check_lint(file_path)
            elif check_name == "type":
                result = await self._check_types(file_path)
            elif check_name == "syntax":
                result = await self._check_syntax(file_path)
            else:
                result = ValidationResult(
                    check_name,
                    False,
                    f"Unknown check: {check_name}",
                    0.0,
                    suggestions=["Available checks: format, imports, lint, type, syntax"],
                )

            duration = time.time() - start_time
            result.duration = duration

            if result.success:
                print(f"{Colors.OKGREEN}✅ ({duration:.1f}s){Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}❌ ({duration:.1f}s){Colors.ENDC}")
                if result.message:
                    print(f"{Colors.WARNING}  └─ {result.message}{Colors.ENDC}")

            return result

        except Exception as e:
            duration = time.time() - start_time
            print(f"{Colors.FAIL}💥 ERROR ({duration:.1f}s){Colors.ENDC}")
            print(f"{Colors.FAIL}  └─ {str(e)}{Colors.ENDC}")

            return ValidationResult(
                check_name,
                False,
                f"Check failed: {str(e)}",
                duration,
                suggestions=["Check tool installation", "Verify file accessibility"],
            )

    async def _check_format(self, file_path: str) -> ValidationResult:
        """Check code formatting with Black."""
        cmd = ["uv", "run", "black", "--check", "--diff", file_path]
        success, output, exit_code = await self._run_command(cmd)

        suggestions = []
        if not success:
            suggestions = [
                f"Run: uv run black {file_path}",
                "Configure your editor to format on save",
                "Use 'make format' for project-wide formatting",
            ]

        return ValidationResult(
            "format",
            success,
            "Code formatting is correct" if success else "Code formatting issues found",
            0.0,
            output,
            suggestions,
        )

    async def _check_imports(self, file_path: str) -> ValidationResult:
        """Check import sorting with isort."""
        cmd = ["uv", "run", "isort", "--check-only", "--diff", file_path]
        success, output, exit_code = await self._run_command(cmd)

        suggestions = []
        if not success:
            suggestions = [
                f"Run: uv run isort {file_path}",
                "Check isort configuration in pyproject.toml",
                "Use 'make format' for project-wide import sorting",
            ]

        return ValidationResult(
            "imports",
            success,
            "Import sorting is correct" if success else "Import sorting issues found",
            0.0,
            output,
            suggestions,
        )

    async def _check_lint(self, file_path: str) -> ValidationResult:
        """Check code quality with Ruff."""
        cmd = ["uv", "run", "ruff", "check", file_path]
        success, output, exit_code = await self._run_command(cmd)

        suggestions = []
        if not success:
            suggestions = [
                f"Run: uv run ruff check --fix {file_path}",
                "Review specific linting errors",
                "Check Ruff configuration in pyproject.toml",
            ]

        return ValidationResult(
            "lint",
            success,
            "No linting issues found" if success else "Linting issues found",
            0.0,
            output,
            suggestions,
        )

    async def _check_types(self, file_path: str) -> ValidationResult:
        """Check type annotations with MyPy."""
        # MyPy works better with modules, so check the containing directory
        file_dir = str(Path(file_path).parent)

        # Skip type checking for certain directories
        skip_dirs = ["tests", "dashboard"]
        if any(skip_dir in file_dir for skip_dir in skip_dirs):
            return ValidationResult("type", True, "Type checking skipped for this file type", 0.0)

        cmd = ["uv", "run", "mypy", file_path]
        success, output, exit_code = await self._run_command(cmd)

        suggestions = []
        if not success:
            suggestions = [
                "Add missing type annotations",
                "Fix type inconsistencies",
                "Use # type: ignore for unavoidable issues",
                "Check MyPy configuration in pyproject.toml",
            ]

        return ValidationResult(
            "type",
            success,
            "Type checking passed" if success else "Type checking issues found",
            0.0,
            output,
            suggestions,
        )

    async def _check_syntax(self, file_path: str) -> ValidationResult:
        """Check Python syntax by compiling the file."""
        try:
            with open(self.project_root / file_path, encoding="utf-8") as f:
                source = f.read()

            compile(source, file_path, "exec")
            return ValidationResult("syntax", True, "Syntax is valid", 0.0)

        except SyntaxError as e:
            return ValidationResult(
                "syntax",
                False,
                f"Syntax error: {str(e)}",
                0.0,
                suggestions=[
                    "Fix syntax errors",
                    "Check for missing brackets, quotes, or colons",
                    "Verify indentation consistency",
                ],
            )
        except Exception as e:
            return ValidationResult(
                "syntax",
                False,
                f"Failed to check syntax: {str(e)}",
                0.0,
                suggestions=["Check file accessibility", "Verify file encoding"],
            )

    async def _run_related_tests(self, file_path: str) -> ValidationResult:
        """Find and run tests related to the file."""
        related_tests = self._discover_related_tests(file_path)

        if not related_tests:
            return ValidationResult(
                "tests",
                True,
                "No related tests found",
                0.0,
                suggestions=["Consider adding tests for this module"],
            )

        print(
            f"{Colors.OKCYAN}⏳ Running {len(related_tests)} related test(s)...{Colors.ENDC}",
            end=" ",
            flush=True,
        )

        # Run the related tests
        cmd = ["uv", "run", "pytest", "-x", "--tb=short"] + related_tests
        success, output, exit_code = await self._run_command(cmd)

        suggestions = []
        if not success:
            suggestions = [
                "Fix failing tests",
                "Update tests to match code changes",
                "Run tests individually for detailed output",
            ]

        return ValidationResult(
            "tests",
            success,
            (
                f"Related tests passed ({len(related_tests)} tests)"
                if success
                else "Some related tests failed"
            ),
            0.0,
            output,
            suggestions,
        )

    def _discover_related_tests(self, file_path: str) -> List[str]:
        """Discover test files related to the given file."""
        related_tests = []

        # Convert file path to module-like structure
        path_parts = Path(file_path).parts
        if path_parts[0] in ["agents", "models", "simulation", "utils"]:
            module_name = Path(file_path).stem

            # Look for direct test files
            test_patterns = [
                f"tests/unit/test_{module_name}.py",
                f"tests/unit/test_{module_name}_*.py",
                f"tests/integration/test_{module_name}.py",
                f"tests/integration/test_{module_name}_*.py",
            ]

            for pattern in test_patterns:
                if "*" in pattern:
                    # Handle glob patterns
                    pattern_dir = Path(pattern).parent
                    pattern_name = Path(pattern).name.replace("*", "")

                    test_dir = self.project_root / pattern_dir
                    if test_dir.exists():
                        for test_file in test_dir.glob(f"*{pattern_name}*"):
                            if test_file.is_file() and test_file.suffix == ".py":
                                related_tests.append(str(test_file.relative_to(self.project_root)))
                else:
                    test_file = self.project_root / pattern
                    if test_file.exists():
                        related_tests.append(pattern)

        return related_tests

    async def _run_command(self, cmd: List[str]) -> Tuple[bool, str, int]:
        """Run a command and return success status, output, and exit code."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.project_root),
            )

            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.timeout)

                output = stdout.decode("utf-8", errors="replace") if stdout else ""
                success = process.returncode == 0

                return success, output, process.returncode

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False, f"Command timed out after {self.timeout}s", -1

        except Exception as e:
            return False, f"Failed to execute command: {str(e)}", -1

    def _print_file_summary(self, file_path: str, success: bool) -> None:
        """Print summary of file validation results."""
        print("\n" + "=" * 60)
        print(f"{Colors.HEADER}📊 VALIDATION SUMMARY: {file_path}{Colors.ENDC}")
        print("=" * 60)

        # Overall status
        if success:
            print(f"{Colors.OKGREEN}✅ Overall Status: PASSED{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}❌ Overall Status: FAILED{Colors.ENDC}")

        # Individual check results
        print(f"\n{Colors.BOLD}Check Results:{Colors.ENDC}")
        total_duration = sum(r.duration for r in self.results)

        for result in self.results:
            status = f"{Colors.OKGREEN}✅" if result.success else f"{Colors.FAIL}❌"
            print(
                f"  {status} {result.check_name:<12} ({result.duration:.1f}s) - {result.message}{Colors.ENDC}"
            )

        print(f"\n{Colors.OKCYAN}⏱️  Total Duration: {total_duration:.1f}s{Colors.ENDC}")

        # Suggestions for failed checks
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\n{Colors.FAIL}🔧 Suggestions for Failed Checks:{Colors.ENDC}")
            for result in failed_results:
                if result.suggestions:
                    print(f"\n{Colors.FAIL}{result.check_name.upper()}:{Colors.ENDC}")
                    for suggestion in result.suggestions[:2]:  # Limit suggestions
                        print(f"{Colors.WARNING}  • {suggestion}{Colors.ENDC}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Individual File Validation Utility",
        epilog="""
Examples:
  %(prog)s agents/base_agent.py                    # Validate with default checks
  %(prog)s models/financial.py --with-tests       # Include related tests
  %(prog)s agents/inventory_agent.py --checks format,lint  # Specific checks only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("file", nargs="?", help="File to validate (relative to project root)")

    parser.add_argument(
        "--checks",
        type=str,
        help="Comma-separated list of checks to run (format,imports,lint,type,syntax)",
    )

    parser.add_argument(
        "--with-tests", action="store_true", help="Run related tests after validation"
    )

    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout in seconds for each check (default: 30)"
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    parser.add_argument("--list-checks", action="store_true", help="List available checks and exit")

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Handle list checks option
    if args.list_checks:
        print("Available validation checks:")
        print("  format   - Code formatting with Black")
        print("  imports  - Import sorting with isort")
        print("  lint     - Code quality with Ruff")
        print("  type     - Type checking with MyPy")
        print("  syntax   - Python syntax validation")
        return 0

    # Validate that file argument is provided
    if not args.file:
        print(f"{Colors.FAIL}❌ File argument is required{Colors.ENDC}")
        print("Use --list-checks to see available validation checks")
        return 1

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

    # Parse checks
    checks = None
    if args.checks:
        checks = [check.strip() for check in args.checks.split(",")]
        # Validate check names
        valid_checks = {"format", "imports", "lint", "type", "syntax"}
        invalid_checks = set(checks) - valid_checks
        if invalid_checks:
            print(f"{Colors.FAIL}❌ Invalid checks: {', '.join(invalid_checks)}{Colors.ENDC}")
            print(f"{Colors.OKCYAN}Valid checks: {', '.join(sorted(valid_checks))}{Colors.ENDC}")
            return 1

    # Create validator and run
    validator = FileValidator(project_root, args.timeout)
    success, results = await validator.validate_file(args.file, checks, args.with_tests)

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}🛑 File validation interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.FAIL}💥 Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)
