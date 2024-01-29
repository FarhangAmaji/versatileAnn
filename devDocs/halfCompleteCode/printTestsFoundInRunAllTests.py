"""
- this can be used in order to check what tests are found
- in the past I had mismatch between number of tests which were tested with
runAllTests.py and the number of funcs found by searching('def test') in tests directory
- very important case: note the number of tests should be run can be more "number of funcs found by searching('def test')"
because some tests classes may inherit from other test classes of "number of funcs found by searching('def test')" doesnt involve them

"""
import inspect
import os
import unittest


# Print the file paths of the tests
def print_tests(tests):
    for test in tests:
        if isinstance(test, unittest.TestSuite):
            print_tests(test)
        elif isinstance(test, unittest.loader._FailedTest):
            print('Failed to load test')
        else:
            test_method = getattr(test.__class__, test._testMethodName)
            print('test', os.path.basename(inspect.getfile(test_method)), test._testMethodName)
print_tests(testSuite)