#!/usr/bin/env python3
"""
Watch Mode Validation Script

Monitors file system changes and automatically runs validation steps
(linting, formatting, type checking, testing) on modified Python files.

Features:
- Real-time file monitoring with watchdog
- Incremental validation sequence
- Smart test discovery
- Timeout protection (30 seconds per step)
- Graceful error handling
- Clear emoji-enhanced output
- Continue monitoring after failures
"""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer


class Colors:
    """ANSI color codes for terminal output"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


class ValidationHandler(FileSystemEventHandler):
    """Handles file system events and triggers validation"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.processing: Set[str] = set()
        self.last_processed: Dict[str, float] = {}
        self.debounce_seconds = 2.0  # Avoid processing same file multiple times rapidly

    def on_modified(self, event) -> None:
        """Handle file modification events"""
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith(".py"):
            self._schedule_validation(event.src_path)

    def _schedule_validation(self, file_path: str) -> None:
        """Schedule validation for a file with debouncing"""
        current_time = time.time()

        # Skip if we're already processing this file
        if file_path in self.processing:
            return

        # Skip if we processed this file very recently (debouncing)
        if file_path in self.last_processed:
            if current_time - self.last_processed[file_path] < self.debounce_seconds:
                return

        # Schedule validation in a separate thread
        threading.Thread(target=self._validate_file, args=(file_path,), daemon=True).start()

    def _validate_file(self, file_path: str) -> None:
        """Validate a single file"""
        try:
            self.processing.add(file_path)
            self.last_processed[file_path] = time.time()

            validator = FileValidator(self.root_path)
            validator.validate_single_file(Path(file_path))

        except Exception as e:
            print(f"{Colors.RED}❌ Error validating {file_path}: {e}{Colors.RESET}")
        finally:
            self.processing.discard(file_path)


class FileValidator:
    """Handles validation of individual files"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.timeout = 30  # 30 seconds timeout per validation step

    def validate_single_file(self, file_path: Path) -> None:
        """Run complete validation sequence on a single file"""
        rel_path = file_path.relative_to(self.root_path)
        print(f"\n{Colors.CYAN}🔍 Validating: {rel_path}{Colors.RESET}")
        print(f"{Colors.BLUE}{'=' * 50}{Colors.RESET}")

        # Run validation steps in sequence
        steps = [
            ("Lint Check", self._run_lint_check),
            ("Auto-fix", self._run_auto_fix),
            ("Format Check", self._run_format_check),
            ("Import Sorting", self._run_import_sort),
            ("Type Check", self._run_type_check),
            ("Test Discovery", self._run_related_tests),
        ]

        for step_name, step_func in steps:
            try:
                print(f"{Colors.YELLOW}⏳ {step_name}...{Colors.RESET}", end=" ", flush=True)
                success = step_func(file_path)

                if success:
                    print(f"{Colors.GREEN}✅{Colors.RESET}")
                else:
                    print(f"{Colors.RED}❌{Colors.RESET}")
                    # Continue with other steps even if one fails

            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")

        print(f"{Colors.BLUE}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.GREEN}✨ Validation complete for {rel_path}{Colors.RESET}")

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run a command with timeout protection"""
        try:
            result = subprocess.run(
                cmd, cwd=cwd or self.root_path, capture_output=True, text=True, timeout=self.timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}⏰ Timeout after {self.timeout}s{Colors.RESET}")
            return False
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            return False

    def _run_lint_check(self, file_path: Path) -> bool:
        """Run ruff linting check"""
        return self._run_command(["ruff", "check", str(file_path)])

    def _run_auto_fix(self, file_path: Path) -> bool:
        """Run ruff auto-fix"""
        return self._run_command(["ruff", "check", "--fix", str(file_path)])

    def _run_format_check(self, file_path: Path) -> bool:
        """Run black formatting"""
        return self._run_command(["black", "--check", str(file_path)])

    def _run_import_sort(self, file_path: Path) -> bool:
        """Run isort import sorting"""
        return self._run_command(["isort", "--check-only", str(file_path)])

    def _run_type_check(self, file_path: Path) -> bool:
        """Run mypy type checking"""
        return self._run_command(["mypy", str(file_path)])

    def _run_related_tests(self, file_path: Path) -> bool:
        """Discover and run related tests"""
        test_files = self._discover_related_tests(file_path)

        if not test_files:
            print(f"{Colors.YELLOW}📝 No related tests found{Colors.RESET}")
            return True

        print(f"{Colors.CYAN}🧪 Found {len(test_files)} related test(s){Colors.RESET}")

        # Run each test file
        all_passed = True
        for test_file in test_files:
            if not self._run_command(["pytest", str(test_file), "-v"]):
                all_passed = False

        return all_passed

    def _discover_related_tests(self, file_path: Path) -> List[Path]:
        """Smart test discovery based on file paths"""
        test_files = []
        rel_path = file_path.relative_to(self.root_path)

        # Get the module name without extension
        module_name = rel_path.stem

        # Pattern 1: Direct test file (test_module.py)
        direct_test = self.root_path / "tests" / "unit" / f"test_{module_name}.py"
        if direct_test.exists():
            test_files.append(direct_test)

        # Pattern 2: Module-specific test directory
        if rel_path.parts[0] in ["agents", "models", "simulation"]:
            module_dir = rel_path.parts[0]
            module_test = self.root_path / "tests" / "unit" / f"test_{module_name}.py"
            if module_test.exists() and module_test not in test_files:
                test_files.append(module_test)

        # Pattern 3: Enhanced/advanced test files
        enhanced_patterns = [
            f"test_{module_name}_enhanced.py",
            f"test_{module_name}_enhanced_coverage.py",
            f"test_{module_name}_enhanced_missing_coverage.py",
            f"test_{module_name}_enhanced_additional_coverage.py",
            f"test_{module_name}_enhanced_final_coverage.py",
            f"test_{module_name}_missing_coverage.py",
            f"test_{module_name}_specific_coverage.py",
            f"test_{module_name}_advanced.py",
            f"test_{module_name}_additional_coverage.py",
        ]

        for pattern in enhanced_patterns:
            test_file = self.root_path / "tests" / "unit" / pattern
            if test_file.exists() and test_file not in test_files:
                test_files.append(test_file)

        return test_files


class WatchModeRunner:
    """Main watch mode runner"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.observer = Observer()
        self.handler = ValidationHandler(root_path)
        self.running = False

    def start(self) -> None:
        """Start watch mode"""
        print(f"{Colors.BOLD}{Colors.GREEN}🚀 Starting Watch Mode Validation{Colors.RESET}")
        print(f"{Colors.CYAN}📁 Monitoring: {self.root_path}{Colors.RESET}")
        print(f"{Colors.CYAN}📂 Watching directories: agents/, models/, simulation/{Colors.RESET}")
        print(f"{Colors.YELLOW}⏱️  Timeout per step: 30 seconds{Colors.RESET}")
        print(f"{Colors.MAGENTA}🛑 Press Ctrl+C to stop{Colors.RESET}")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")

        # Watch the specific directories
        for directory in ["agents", "models", "simulation"]:
            watch_path = self.root_path / directory
            if watch_path.exists():
                self.observer.schedule(self.handler, str(watch_path), recursive=True)
                print(f"{Colors.GREEN}👀 Watching: {directory}/ {Colors.RESET}")

        self.observer.start()
        self.running = True

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop watch mode"""
        print(f"\n{Colors.YELLOW}🛑 Stopping watch mode...{Colors.RESET}")
        self.running = False
        self.observer.stop()
        self.observer.join()
        print(f"{Colors.GREEN}✅ Watch mode stopped{Colors.RESET}")


def main() -> None:
    """Main entry point"""
    # Find project root
    current_dir = Path(__file__).parent
    root_path = current_dir.parent

    # Validate we're in the right directory
    if not (root_path / "pyproject.toml").exists():
        print(f"{Colors.RED}❌ Error: Could not find pyproject.toml in {root_path}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}💡 Make sure you're running this from the project root{Colors.RESET}"
        )
        sys.exit(1)

    # Check if required tools are available
    required_tools = ["ruff", "black", "isort", "mypy", "pytest"]
    missing_tools = []

    for tool in required_tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)

    if missing_tools:
        print(f"{Colors.RED}❌ Missing required tools: {', '.join(missing_tools)}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}💡 Run 'make install' or 'uv sync --all-extras' to install dependencies{Colors.RESET}"
        )
        sys.exit(1)

    # Start watch mode
    runner = WatchModeRunner(root_path)
    runner.start()


if __name__ == "__main__":
    main()
