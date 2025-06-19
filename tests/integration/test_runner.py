"""
Test runner for integration tests with proper setup and teardown.
"""
import pytest
import asyncio
import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
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
    
    # Run integration tests
    test_args = [
        str(Path(__file__).parent),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Shorter tracebacks
        "-x",  # Stop on first failure
        "--asyncio-mode=auto",  # Handle async tests
    ]
    
    print("Running Business Agent System Integration Tests...")
    print("=" * 60)
    
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
    else:
        print("\n" + "=" * 60)
        print("❌ Some integration tests failed!")
    
    return exit_code


def run_specific_test_file(test_file: str):
    """Run a specific test file."""
    setup_test_logging()
    
    test_path = Path(__file__).parent / test_file
    if not test_path.exists():
        print(f"Test file {test_file} not found!")
        return 1
    
    test_args = [
        str(test_path),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
    ]
    
    print(f"Running {test_file}...")
    print("=" * 60)
    
    return pytest.main(test_args)


def run_quick_smoke_test():
    """Run a quick smoke test of core functionality."""
    setup_test_logging()
    
    test_args = [
        str(Path(__file__).parent / "test_system_initialization.py"),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-k", "test_config_loading_valid_file or test_agent_initialization_all_enabled"
    ]
    
    print("Running quick smoke test...")
    print("=" * 40)
    
    return pytest.main(test_args)


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
            "test_error_scenarios.py"
        ]
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test only"
    )
    
    args = parser.parse_args()
    
    if args.smoke:
        exit_code = run_quick_smoke_test()
    elif args.file:
        exit_code = run_specific_test_file(args.file)
    else:
        exit_code = run_integration_tests()
    
    sys.exit(exit_code)