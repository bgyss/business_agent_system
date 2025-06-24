"""Test runner for integration tests with proper setup and teardown."""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Suppress some noisy loggers during testing
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def run_integration_tests():
    """Run all integration tests."""
    setup_test_logging()

    # Set test environment variables
    os.environ["PYTHONPATH"] = str(project_root)

    # Ensure ANTHROPIC_API_KEY is set for testing
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key-for-testing"

    # Run integration tests using uv run pytest
    test_args = [
        "uv",
        "run",
        "pytest",
        str(Path(__file__).parent),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Shorter tracebacks
        "--asyncio-mode=auto",  # Handle async tests
        "--durations=10",  # Report slowest 10 tests
    ]

    print("Running Business Agent System Integration Tests...")
    print("=" * 60)
    print(f"Test directory: {Path(__file__).parent}")
    print(f"Project root: {project_root}")
    print(f"Command: {' '.join(test_args)}")
    print("=" * 60)

    # Change to project root directory for uv run
    os.chdir(project_root)

    try:
        result = subprocess.run(
            test_args, capture_output=True, text=True, timeout=300  # 5 minute total timeout
        )

        # Print captured output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        exit_code = result.returncode

    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes!")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
    else:
        print("\n" + "=" * 60)
        print(f"❌ Integration tests failed with exit code {exit_code}")

    return exit_code


def run_specific_test_file(test_file: str):
    """Run a specific test file."""
    setup_test_logging()

    test_path = Path(__file__).parent / test_file
    if not test_path.exists():
        print(f"❌ Test file {test_file} not found at {test_path}!")
        return 1

    # Ensure ANTHROPIC_API_KEY is set for testing
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key-for-testing"

    test_args = [
        "uv",
        "run",
        "pytest",
        str(test_path),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
    ]

    print(f"Running {test_file}...")
    print("=" * 60)

    # Change to project root directory for uv run
    os.chdir(project_root)

    try:
        result = subprocess.run(test_args, capture_output=True, text=True, timeout=180)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode

    except subprocess.TimeoutExpired:
        print(f"❌ Test {test_file} timed out!")
        return 1
    except Exception as e:
        print(f"❌ Error running test {test_file}: {e}")
        return 1


def run_quick_smoke_test():
    """Run a quick smoke test of core functionality."""
    setup_test_logging()

    test_args = [
        "uv",
        "run",
        "pytest",
        str(Path(__file__).parent / "test_system_initialization.py"),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-k",
        "test_config_loading_valid_file or test_agent_initialization_all_enabled",
    ]

    print("Running quick smoke test...")
    print("=" * 40)

    # Change to project root directory for uv run
    os.chdir(project_root)
    return subprocess.run(test_args).returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run integration tests")
    parser.add_argument(
        "--file",
        help="Run specific test file",
        choices=[
            "test_system_initialization.py",
            "test_business_simulation.py",
            "test_agent_coordination.py",
            "test_database_operations.py",
            "test_end_to_end_workflows.py",
            "test_error_scenarios.py",
        ],
    )
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test only")

    args = parser.parse_args()

    if args.smoke:
        exit_code = run_quick_smoke_test()
    elif args.file:
        exit_code = run_specific_test_file(args.file)
    else:
        exit_code = run_integration_tests()

    sys.exit(exit_code)
