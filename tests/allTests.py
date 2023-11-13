import os
import unittest

# Automatically discover and load test cases
testLoader = unittest.TestLoader()
testSuite = testLoader.discover(os.path.dirname(__file__), pattern="*.py")

if __name__ == "__main__":
    # Run the tests
    testRunner = unittest.TextTestRunner()
    result = testRunner.run(testSuite)
