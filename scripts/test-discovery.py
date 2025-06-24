#!/usr/bin/env python3
"""
Test script to validate the smart test discovery functionality
"""

import sys
from pathlib import Path


# Import the FileValidator class directly by copying the relevant parts
class FileValidator:
    """Handles validation of individual files - simplified for testing"""

    def __init__(self, root_path: Path):
        self.root_path = root_path

    def _discover_related_tests(self, file_path: Path) -> list[Path]:
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


def test_discovery_for_files():
    """Test discovery for various existing files"""
    root_path = Path(__file__).parent.parent
    validator = FileValidator(root_path)

    # Test files to check discovery for
    test_files = [
        "agents/accounting_agent.py",
        "agents/inventory_agent.py",
        "agents/hr_agent.py",
        "models/financial.py",
        "models/inventory.py",
        "simulation/business_simulator.py",
    ]

    print("🔍 Testing Smart Test Discovery")
    print("=" * 50)

    for test_file_str in test_files:
        test_file = root_path / test_file_str
        if test_file.exists():
            print(f"\n📁 {test_file_str}:")
            discovered_tests = validator._discover_related_tests(test_file)

            if discovered_tests:
                for i, test_path in enumerate(discovered_tests, 1):
                    rel_test_path = test_path.relative_to(root_path)
                    exists = "✅" if test_path.exists() else "❌"
                    print(f"  {i}. {exists} {rel_test_path}")
            else:
                print("  📝 No related tests found")
        else:
            print(f"\n❌ {test_file_str}: File not found")


def main():
    """Main test function"""
    try:
        test_discovery_for_files()
        print("\n✅ Test discovery validation completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Error during test discovery: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
