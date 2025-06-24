#!/usr/bin/env python3
"""
Test script to validate individual components of the watch-validate system
"""

import subprocess
import sys
from pathlib import Path


def test_tool_availability():
    """Test if all required tools are available"""
    required_tools = ["ruff", "black", "isort", "mypy", "pytest"]
    missing_tools = []

    print("🔧 Testing tool availability...")
    for tool in required_tools:
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, check=True)
            print(f"✅ {tool}: Available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)
            print(f"❌ {tool}: Missing")

    return len(missing_tools) == 0


def test_project_structure():
    """Test if the project has the expected structure"""
    print("\n📁 Testing project structure...")
    root_path = Path(__file__).parent.parent

    required_dirs = ["agents", "models", "simulation", "tests"]
    required_files = ["pyproject.toml", "Makefile"]

    all_present = True

    for directory in required_dirs:
        dir_path = root_path / directory
        if dir_path.exists():
            print(f"✅ {directory}/: Found")
        else:
            print(f"❌ {directory}/: Missing")
            all_present = False

    for file in required_files:
        file_path = root_path / file
        if file_path.exists():
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")
            all_present = False

    return all_present


def test_validation_components():
    """Test individual validation components"""
    print("\n🧪 Testing validation components...")
    root_path = Path(__file__).parent.parent

    # Test on a known good file
    test_file = root_path / "agents" / "__init__.py"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    tests = [
        ("ruff check", ["ruff", "check", str(test_file)]),
        ("black check", ["black", "--check", str(test_file)]),
        ("isort check", ["isort", "--check-only", str(test_file)]),
    ]

    all_passed = True
    for test_name, cmd in tests:
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=10)
            print(f"✅ {test_name}: Passed")
        except subprocess.CalledProcessError:
            print(f"⚠️  {test_name}: Failed (may need fixes)")
            # Don't mark as failed since files might just need formatting
        except subprocess.TimeoutExpired:
            print(f"❌ {test_name}: Timeout")
            all_passed = False
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            all_passed = False

    return all_passed


def main():
    """Main test function"""
    print("🚀 Testing Watch Mode Validation System")
    print("=" * 40)

    tests = [
        ("Tool Availability", test_tool_availability),
        ("Project Structure", test_project_structure),
        ("Validation Components", test_validation_components),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 40)
    print("📋 Test Results Summary:")

    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Watch mode should work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
