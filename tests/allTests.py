import unittest
import os

# Automatically discover and load test cases
test_loader = unittest.TestLoader()
test_suite = test_loader.discover(os.path.dirname(__file__), pattern="*.py")

if __name__ == "__main__":
    # Run the tests
    test_runner = unittest.TextTestRunner()
    result = test_runner.run(test_suite)