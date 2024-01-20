import unittest
import os
import coverage

# Create a coverage object
cov = coverage.Coverage()

# Automatically discover and load test cases
testLoader = unittest.TestLoader()
testSuite = testLoader.discover(os.path.dirname(__file__), pattern="*.py")

if __name__ == "__main__":
    # Start coverage measurement
    cov.start()

    # Run the tests
    testRunner = unittest.TextTestRunner()
    result = testRunner.run(testSuite)

    # Stop coverage measurement
    cov.stop()

    # Generate HTML reports for each test file
    cov_html_dir = "coverage_html"  # Specify the directory for HTML reports
    cov.html_report(directory=cov_html_dir)

    # Print the coverage summary
    cov.report()
    cov.html_report()

# Optionally, you can save the coverage data to a file for further analysis
cov.save()
