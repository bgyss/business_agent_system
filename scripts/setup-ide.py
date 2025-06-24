#!/usr/bin/env python3
"""
IDE Setup Script for Business Agent Management System

This script helps set up and validate IDE integration for optimal development experience.
"""

import json
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class IDESetupValidator:
    """Validates and sets up IDE integration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.vscode_dir = project_root / ".vscode"

    def run_setup(self) -> bool:
        """Run the complete IDE setup and validation."""
        print(f"{Colors.HEADER}🔧 Business Agent System - IDE Setup{Colors.ENDC}")
        print("=" * 60)

        success = True
        checks = [
            ("Project Structure", self._check_project_structure),
            ("VS Code Configuration", self._check_vscode_config),
            ("Python Environment", self._check_python_environment),
            ("Development Tools", self._check_dev_tools),
            ("Extension Recommendations", self._check_extensions),
            ("Quality Gate Integration", self._check_quality_gate),
        ]

        for check_name, check_func in checks:
            print(f"\n{Colors.OKCYAN}📋 {check_name}...{Colors.ENDC}")
            try:
                if not check_func():
                    success = False
                    print(f"{Colors.FAIL}❌ {check_name} failed{Colors.ENDC}")
                else:
                    print(f"{Colors.OKGREEN}✅ {check_name} passed{Colors.ENDC}")
            except Exception as e:
                success = False
                print(f"{Colors.FAIL}❌ {check_name} error: {e}{Colors.ENDC}")

        self._print_summary(success)
        return success

    def _check_project_structure(self) -> bool:
        """Check if project structure is correct."""
        required_dirs = ["agents", "models", "simulation", "tests", "config"]
        required_files = ["pyproject.toml", "main.py", "Makefile"]

        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                print(f"{Colors.WARNING}  Missing directory: {dir_name}{Colors.ENDC}")
                return False

        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                print(f"{Colors.WARNING}  Missing file: {file_name}{Colors.ENDC}")
                return False

        print(f"{Colors.OKGREEN}  All required project files and directories found{Colors.ENDC}")
        return True

    def _check_vscode_config(self) -> bool:
        """Check VS Code configuration files."""
        required_files = ["settings.json", "tasks.json", "launch.json", "extensions.json"]

        if not self.vscode_dir.exists():
            print(f"{Colors.WARNING}  .vscode directory not found{Colors.ENDC}")
            return False

        missing_files = []
        for file_name in required_files:
            file_path = self.vscode_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
            else:
                # Validate JSON syntax
                try:
                    with open(file_path) as f:
                        json.load(f)
                    print(f"{Colors.OKGREEN}  ✓ {file_name} - valid{Colors.ENDC}")
                except json.JSONDecodeError as e:
                    print(f"{Colors.FAIL}  ✗ {file_name} - invalid JSON: {e}{Colors.ENDC}")
                    return False

        if missing_files:
            print(
                f"{Colors.WARNING}  Missing VS Code config files: {', '.join(missing_files)}{Colors.ENDC}"
            )
            return False

        return True

    def _check_python_environment(self) -> bool:
        """Check Python environment setup."""
        venv_path = self.project_root / ".venv"

        if not venv_path.exists():
            print(f"{Colors.WARNING}  Virtual environment not found at .venv{Colors.ENDC}")
            print(f"{Colors.OKCYAN}  Run: make install{Colors.ENDC}")
            return False

        python_executable = venv_path / "bin" / "python"
        if not python_executable.exists():
            print(
                f"{Colors.WARNING}  Python executable not found in virtual environment{Colors.ENDC}"
            )
            return False

        # Check if required packages are installed
        required_packages = ["black", "isort", "ruff", "mypy", "pytest"]

        missing_packages = []
        for package in required_packages:
            try:
                result = subprocess.run(
                    [str(python_executable), "-c", f"import {package}"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    missing_packages.append(package)
            except Exception:
                missing_packages.append(package)

        if missing_packages:
            print(f"{Colors.WARNING}  Missing packages: {', '.join(missing_packages)}{Colors.ENDC}")
            print(f"{Colors.OKCYAN}  Run: make install{Colors.ENDC}")
            return False

        print(f"{Colors.OKGREEN}  Python environment configured correctly{Colors.ENDC}")
        return True

    def _check_dev_tools(self) -> bool:
        """Check development tools availability."""
        tools = [
            ("uv", "uv --version"),
            ("make", "make --version"),
        ]

        optional_tools = [
            ("nix", "nix --version"),
            ("direnv", "direnv --version"),
        ]

        # Check required tools
        for tool_name, command in tools:
            if not self._check_command(command):
                print(f"{Colors.FAIL}  Required tool missing: {tool_name}{Colors.ENDC}")
                return False
            else:
                print(f"{Colors.OKGREEN}  ✓ {tool_name} available{Colors.ENDC}")

        # Check optional tools
        for tool_name, command in optional_tools:
            if self._check_command(command):
                print(f"{Colors.OKGREEN}  ✓ {tool_name} available (optional){Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}  ○ {tool_name} not available (optional){Colors.ENDC}")

        return True

    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_extensions(self) -> bool:
        """Check VS Code extension recommendations."""
        extensions_file = self.vscode_dir / "extensions.json"

        if not extensions_file.exists():
            print(f"{Colors.WARNING}  extensions.json not found{Colors.ENDC}")
            return False

        try:
            with open(extensions_file) as f:
                extensions_config = json.load(f)

            recommended = extensions_config.get("recommendations", [])
            essential_extensions = [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-python.black-formatter",
                "charliermarsh.ruff",
            ]

            missing_essential = []
            for ext in essential_extensions:
                if ext not in recommended:
                    missing_essential.append(ext)

            if missing_essential:
                print(
                    f"{Colors.WARNING}  Missing essential extensions: {', '.join(missing_essential)}{Colors.ENDC}"
                )
                return False

            print(
                f"{Colors.OKGREEN}  Extension recommendations configured ({len(recommended)} extensions){Colors.ENDC}"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            print(f"{Colors.FAIL}  Invalid extensions.json: {e}{Colors.ENDC}")
            return False

    def _check_quality_gate(self) -> bool:
        """Check quality gate integration."""
        quality_gate_script = self.project_root / "scripts" / "quality-gate.py"

        if not quality_gate_script.exists():
            print(f"{Colors.WARNING}  Quality gate script not found{Colors.ENDC}")
            return False

        # Check if quality gate can run
        try:
            result = subprocess.run(
                ["python", str(quality_gate_script), "--mode", "emergency"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print(f"{Colors.OKGREEN}  Quality gate integration working{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.WARNING}  Quality gate returned non-zero exit code{Colors.ENDC}")
                return False

        except subprocess.TimeoutExpired:
            print(f"{Colors.WARNING}  Quality gate check timed out{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.FAIL}  Quality gate check failed: {e}{Colors.ENDC}")
            return False

    def _print_summary(self, success: bool) -> None:
        """Print setup summary and next steps."""
        print("\n" + "=" * 60)
        print(f"{Colors.HEADER}📊 IDE Setup Summary{Colors.ENDC}")
        print("=" * 60)

        if success:
            print(f"{Colors.OKGREEN}🎉 IDE setup completed successfully!{Colors.ENDC}")
            print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
            print(
                f"{Colors.OKCYAN}1. Open VS Code: code business-agent-system.code-workspace{Colors.ENDC}"
            )
            print(f"{Colors.OKCYAN}2. Install recommended extensions when prompted{Colors.ENDC}")
            print(f"{Colors.OKCYAN}3. Try a quick validation: Ctrl+Shift+Q{Colors.ENDC}")
            print(f"{Colors.OKCYAN}4. Review the IDE setup guide: docs/IDE_SETUP.md{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}❌ IDE setup has issues that need to be resolved.{Colors.ENDC}")
            print(f"\n{Colors.BOLD}Troubleshooting:{Colors.ENDC}")
            print(f"{Colors.WARNING}1. Run: make dev-setup{Colors.ENDC}")
            print(f"{Colors.WARNING}2. Run: make install{Colors.ENDC}")
            print(
                f"{Colors.WARNING}3. Check docs/IDE_SETUP.md for detailed instructions{Colors.ENDC}"
            )
            print(
                f"{Colors.WARNING}4. Re-run this script: python scripts/setup-ide.py{Colors.ENDC}"
            )

        print(f"\n{Colors.BOLD}Available Commands:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}• make quality-gate-quick  - Quick quality check{Colors.ENDC}")
        print(f"{Colors.OKCYAN}• make watch               - Start watch mode{Colors.ENDC}")
        print(f"{Colors.OKCYAN}• make test                - Run all tests{Colors.ENDC}")
        print(f"{Colors.OKCYAN}• make dashboard           - Launch dashboard{Colors.ENDC}")


def main() -> int:
    """Main entry point."""
    # Find project root
    current_dir = Path.cwd()
    project_root = current_dir

    # Look for pyproject.toml to identify project root
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent
    else:
        print(f"{Colors.FAIL}❌ Could not find project root (no pyproject.toml found){Colors.ENDC}")
        print(
            f"{Colors.WARNING}Please run this script from within the project directory{Colors.ENDC}"
        )
        return 1

    # Verify this is the correct project
    if not (project_root / "agents").exists():
        print(
            f"{Colors.FAIL}❌ This doesn't appear to be the Business Agent System project{Colors.ENDC}"
        )
        return 1

    # Run setup validation
    validator = IDESetupValidator(project_root)
    success = validator.run_setup()

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}🛑 Setup interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.FAIL}💥 Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)
