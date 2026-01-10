#!/usr/bin/env python3
"""Test runner script for donkey_rl package tests.

This script runs all unit tests in the tests directory.
Usage:
    python tests/run_tests.py
    python tests/run_tests.py test_rewards
    python tests/run_tests.py test_rewards.TestHelperFunctions
"""

import os
import sys
import unittest

# Add project root directory to Python path so we can import donkey_rl
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Get the tests directory path
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if len(sys.argv) > 1:
        # Run specific test module or test case
        test_name = sys.argv[1]
        # If it's just a module name without "test_" prefix, add it
        if not test_name.startswith("test_") and "." not in test_name:
            test_name = f"tests.test_{test_name}"
        elif not test_name.startswith("tests."):
            test_name = f"tests.{test_name}"
        try:
            suite = loader.loadTestsFromName(test_name)
        except (AttributeError, ImportError) as e:
            print(f"Error: Could not find test '{test_name}': {e}")
            print("Available test modules: test_rewards, test_obs_preprocess, test_wrappers")
            sys.exit(1)
    else:
        # Discover all tests in the tests directory
        suite = loader.discover(tests_dir, pattern="test_*.py", top_level_dir=project_root)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

