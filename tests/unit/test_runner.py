#!/usr/bin/env python3
"""
Test runner script for unit tests
"""
import subprocess
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all unit tests with coverage"""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
        "--asyncio-mode=auto",  # Auto handle asyncio tests
    ]
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend([
            "--cov=agents",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit",
            "--cov-fail-under=80"
        ])
        print("Running tests with coverage...")
    except ImportError:
        print("Coverage not available, running tests without coverage...")
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run the tests
    result = subprocess.run(cmd)
    return result.returncode


def run_specific_test(test_name):
    """Run a specific test file or test method"""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "pytest",
        f"{test_dir}/{test_name}",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    print(f"Running specific test: {test_name}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        return run_specific_test(test_name)
    else:
        # Run all tests
        return run_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)