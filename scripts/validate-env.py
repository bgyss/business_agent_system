#!/usr/bin/env python3
"""
Environment validation script for Business Agent Management System.

This script validates that the development environment is properly configured
with all required dependencies, environment variables, and tools.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False


def check_command_exists(command, description):
    """Check if a command exists and is executable."""
    try:
        result = subprocess.run([command, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            print(f"✅ {description}: {version}")
            return True
        else:
            print(f"❌ {description} not working properly")
            return False
    except FileNotFoundError:
        print(f"❌ {description} not found: {command}")
        return False


def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        "anthropic",
        "sqlalchemy",
        "pydantic",
        "streamlit",
        "pytest",
        "black",
        "isort",
        "ruff",
        "mypy",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-c", f"import {package}"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"✅ Python package: {package}")
            else:
                print(f"❌ Python package missing: {package}")
                missing_packages.append(package)
        except Exception as e:
            print(f"❌ Error checking {package}: {e}")
            missing_packages.append(package)

    return len(missing_packages) == 0


def check_environment_variables():
    """Check if required environment variables are set."""
    # Load .env file if it exists
    env_vars = {}
    if Path(".env").exists():
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = []

    for var in required_vars:
        # Check both environment and .env file
        value = os.environ.get(var) or env_vars.get(var)
        if value and value != "your_api_key_here":
            print(f"✅ Environment variable: {var}")
        else:
            print(f"❌ Environment variable missing or invalid: {var}")
            missing_vars.append(var)

    return len(missing_vars) == 0


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Business Agent System environment")
    parser.add_argument(
        "--env-only", action="store_true", help="Only validate environment variables and .env file"
    )
    parser.add_argument(
        "--tools-only", action="store_true", help="Only validate required tools and commands"
    )

    args = parser.parse_args()

    print("🔍 Business Agent System - Environment Validation")
    print("=" * 50)

    all_checks_passed = True

    if args.env_only:
        # Only check environment variables
        print("\n🔧 Environment Configuration:")
        env_vars_ok = check_environment_variables()
        all_checks_passed = env_vars_ok
    elif args.tools_only:
        # Only check tools
        print("\n🛠️ Required Tools:")
        tools_ok = all(
            [
                check_command_exists("python", "Python"),
                check_command_exists("uv", "UV package manager"),
                check_command_exists("git", "Git"),
            ]
        )
        all_checks_passed = tools_ok
    else:
        # Full validation
        # Check required files
        print("\n📋 Required Files:")
        files_ok = all(
            [
                check_file_exists("pyproject.toml", "Project configuration"),
                check_file_exists("flake.nix", "Nix configuration"),
                check_file_exists(".pre-commit-config.yaml", "Pre-commit configuration"),
                check_file_exists("Makefile", "Build configuration"),
            ]
        )
        all_checks_passed = all_checks_passed and files_ok

        # Check .env file (optional but recommended)
        print("\n🔧 Environment Configuration:")
        if check_file_exists(".env", "Environment variables"):
            env_vars_ok = check_environment_variables()
            all_checks_passed = all_checks_passed and env_vars_ok
        else:
            print("💡 Run 'make env-setup' to create .env from template")

        # Check required commands
        print("\n🛠️ Required Tools:")
        tools_ok = all(
            [
                check_command_exists("python", "Python"),
                check_command_exists("uv", "UV package manager"),
                check_command_exists("git", "Git"),
            ]
        )
        all_checks_passed = all_checks_passed and tools_ok

        # Check optional but recommended tools
        print("\n🔧 Optional Tools:")
        check_command_exists("nix", "Nix package manager")
        check_command_exists("direnv", "Directory environment manager")
        check_command_exists("gh", "GitHub CLI")

        # Check Python packages
        print("\n📦 Python Packages:")
        packages_ok = check_python_packages()
        all_checks_passed = all_checks_passed and packages_ok

        # Check directory structure
        print("\n📁 Project Structure:")
        dirs_ok = all(
            [
                check_file_exists("agents/", "Agents directory"),
                check_file_exists("models/", "Models directory"),
                check_file_exists("simulation/", "Simulation directory"),
                check_file_exists("tests/", "Tests directory"),
                check_file_exists("scripts/", "Scripts directory"),
            ]
        )
        all_checks_passed = all_checks_passed and dirs_ok

    # Final result
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ Environment validation passed!")
        if not args.env_only and not args.tools_only:
            print("🚀 Your development environment is ready.")
        return 0
    else:
        print("❌ Environment validation failed!")
        print("💡 Fix the issues above and run validation again.")
        if not args.env_only and not args.tools_only:
            print("💡 Run 'make dev-setup' for automated setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
