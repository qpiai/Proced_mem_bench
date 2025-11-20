"""
Validation Script for Custom Retrieval Systems

This script helps you verify that your custom retrieval implementation:
1. Implements all required methods correctly
2. Returns properly formatted results
3. Integrates with the benchmark framework

Run: python tests/validate_retrieval.py
"""

import sys
from typing import List, Dict
from pathlib import Path

# Add package to path for testing before installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from procedural_memory_benchmark import (
    RetrievalSystem,
    RetrievedTrajectory,
    BenchmarkRunner
)


class ValidationChecker:
    """Validates custom retrieval system implementations."""

    def __init__(self, retrieval_system: RetrievalSystem):
        self.system = retrieval_system
        self.errors = []
        self.warnings = []
        self.passed = []

    def check_required_methods(self) -> bool:
        """Verify all required methods exist."""
        print("🔍 Checking required methods...")

        required_methods = ['retrieve', 'get_system_name', 'get_system_info']
        all_exist = True

        for method_name in required_methods:
            if not hasattr(self.system, method_name):
                self.errors.append(f"Missing required method: {method_name}")
                all_exist = False
            elif not callable(getattr(self.system, method_name)):
                self.errors.append(f"'{method_name}' exists but is not callable")
                all_exist = False
            else:
                self.passed.append(f"Method '{method_name}' exists")

        return all_exist

    def check_system_name(self) -> bool:
        """Verify get_system_name() returns valid string."""
        print("🔍 Checking system name...")

        try:
            name = self.system.get_system_name()

            if not isinstance(name, str):
                self.errors.append(f"get_system_name() must return str, got {type(name)}")
                return False

            if len(name) == 0:
                self.errors.append("get_system_name() returned empty string")
                return False

            if ' ' in name:
                self.warnings.append("System name contains spaces - consider using underscores")

            self.passed.append(f"System name: '{name}'")
            return True

        except Exception as e:
            self.errors.append(f"get_system_name() raised exception: {e}")
            return False

    def check_system_info(self) -> bool:
        """Verify get_system_info() returns valid dict."""
        print("🔍 Checking system info...")

        try:
            info = self.system.get_system_info()

            if not isinstance(info, dict):
                self.errors.append(f"get_system_info() must return dict, got {type(info)}")
                return False

            if len(info) == 0:
                self.warnings.append("get_system_info() returned empty dict - consider adding metadata")

            self.passed.append(f"System info contains {len(info)} fields")
            return True

        except Exception as e:
            self.errors.append(f"get_system_info() raised exception: {e}")
            return False

    def check_retrieve_basic(self) -> bool:
        """Verify retrieve() works with basic query."""
        print("🔍 Checking retrieve() with test query...")

        test_query = "Put a mug on the coffee maker"
        test_k = 5

        try:
            results = self.system.retrieve(test_query, k=test_k)

            # Check return type
            if not isinstance(results, list):
                self.errors.append(f"retrieve() must return list, got {type(results)}")
                return False

            # Check length
            if len(results) > test_k:
                self.errors.append(f"retrieve() returned {len(results)} results but k={test_k}")
                return False

            if len(results) == 0:
                self.warnings.append("retrieve() returned 0 results - is corpus loaded?")
                return True  # Not an error, just a warning

            self.passed.append(f"retrieve() returned {len(results)} results")
            return True

        except Exception as e:
            self.errors.append(f"retrieve() raised exception: {e}")
            return False

    def check_retrieved_trajectory_format(self) -> bool:
        """Verify results are properly formatted RetrievedTrajectory objects."""
        print("🔍 Checking RetrievedTrajectory format...")

        test_query = "Put a mug on the coffee maker"

        try:
            results = self.system.retrieve(test_query, k=3)

            if len(results) == 0:
                self.warnings.append("No results to check format - skipping")
                return True

            required_fields = [
                'trajectory_id',
                'task_instance_id',
                'task_description',
                'similarity_score',
                'total_steps',
                'document_text'
            ]

            for i, result in enumerate(results):
                # Check if it's a RetrievedTrajectory object
                if not isinstance(result, RetrievedTrajectory):
                    self.errors.append(
                        f"Result {i} is {type(result)}, not RetrievedTrajectory"
                    )
                    continue

                # Check required fields using object attributes
                for field in required_fields:
                    if not hasattr(result, field):
                        self.errors.append(f"Result {i} missing field: {field}")
                    elif getattr(result, field, None) is None:
                        self.warnings.append(f"Result {i} has None value for: {field}")

                # Check field types using object attributes
                if not isinstance(getattr(result, 'trajectory_id', ''), str):
                    self.errors.append(f"Result {i}: trajectory_id must be str")

                if not isinstance(getattr(result, 'similarity_score', 0.0), (int, float)):
                    self.errors.append(f"Result {i}: similarity_score must be numeric")

                if not isinstance(getattr(result, 'total_steps', 0), int):
                    self.errors.append(f"Result {i}: total_steps must be int")

                if not isinstance(getattr(result, 'document_text', ''), str):
                    self.errors.append(f"Result {i}: document_text must be str")
                elif len(getattr(result, 'document_text', '')) == 0:
                    self.errors.append(f"Result {i}: document_text is empty - LLM judge needs content!")

            if not self.errors:
                self.passed.append("All results properly formatted")

            return len(self.errors) == 0

        except Exception as e:
            self.errors.append(f"Format check raised exception: {e}")
            return False

    def check_similarity_scores(self) -> bool:
        """Verify similarity scores are reasonable."""
        print("🔍 Checking similarity scores...")

        test_query = "Put a mug on the coffee maker"

        try:
            results = self.system.retrieve(test_query, k=5)

            if len(results) == 0:
                return True  # Already warned

            scores = [r.similarity_score for r in results]

            # Check ordering (should be descending)
            if scores != sorted(scores, reverse=True):
                self.warnings.append(
                    "Results not sorted by similarity (highest first) - "
                    "benchmark expects descending order"
                )

            # Check for negative scores
            if any(s < 0 for s in scores):
                self.warnings.append("Found negative similarity scores - is this intentional?")

            # Check for all-zero scores
            if all(s == 0 for s in scores):
                self.warnings.append("All similarity scores are 0 - retrieval may not be working")

            self.passed.append(f"Score range: {min(scores):.3f} to {max(scores):.3f}")
            return True

        except Exception as e:
            self.errors.append(f"Score check raised exception: {e}")
            return False

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("=" * 70)
        print("RETRIEVAL SYSTEM VALIDATION")
        print("=" * 70)
        print()

        checks = [
            self.check_required_methods,
            self.check_system_name,
            self.check_system_info,
            self.check_retrieve_basic,
            self.check_retrieved_trajectory_format,
            self.check_similarity_scores
        ]

        all_passed = True
        for check in checks:
            if not check():
                all_passed = False
            print()

        # Print summary
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        if self.passed:
            print(f"\n✅ PASSED ({len(self.passed)}):")
            for msg in self.passed:
                print(f"  • {msg}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  • {msg}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  • {msg}")

        print("\n" + "=" * 70)

        if all_passed and len(self.errors) == 0:
            print("✅ VALIDATION PASSED - System ready for benchmark!")
            print("\nNext steps:")
            print("  1. Run: python examples/quickstart.py")
            print("  2. Or integrate with BenchmarkRunner directly")
            return True
        else:
            print("❌ VALIDATION FAILED - Fix errors before running benchmark")
            print("\nSee docs/TROUBLESHOOTING.md for help")
            return False


def main():
    """Example validation of a custom retrieval system."""
    print("This script validates your custom RetrievalSystem implementation.\n")
    print("USAGE:")
    print("------")
    print("from procedural_memory_benchmark import RetrievalSystem")
    print("from tests.validate_retrieval import ValidationChecker")
    print()
    print("# Create your retrieval system")
    print("my_retrieval = MyCustomRetrieval()")
    print()
    print("# Validate it")
    print("checker = ValidationChecker(my_retrieval)")
    print("checker.run_all_checks()")
    print()
    print("=" * 70)
    print()

    # Example with minimal keyword retrieval
    print("Running example validation with minimal keyword retrieval...")
    print()

    try:
        # Import the minimal example
        sys.path.insert(0, str(Path(__file__).parent.parent / 'examples'))
        from custom_retrieval_minimal import MinimalKeywordRetrieval

        retrieval = MinimalKeywordRetrieval()
        checker = ValidationChecker(retrieval)
        success = checker.run_all_checks()

        sys.exit(0 if success else 1)

    except ImportError as e:
        print(f"⚠️  Could not import example: {e}")
        print("\nTo validate your own system:")
        print("  1. Import your RetrievalSystem class")
        print("  2. Create ValidationChecker instance")
        print("  3. Call run_all_checks()")


if __name__ == "__main__":
    main()
