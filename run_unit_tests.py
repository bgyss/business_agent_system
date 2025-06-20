#!/usr/bin/env python3
"""
Simple script to run unit tests with proper environment setup
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run unit tests with proper environment"""
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Activate virtual environment and run tests
    venv_python = project_root / ".venv" / "bin" / "python"

    if not venv_python.exists():
        print("Virtual environment not found. Please run 'uv sync' first.")
        return 1

    cmd = [
        str(venv_python), "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--asyncio-mode=auto"
    ]

    print("Running unit tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
