import os
import shutil
import unittest

from projectUtils.generalUtils import getProjectDirectory

# Automatically discover and load test cases
testLoader = unittest.TestLoader()
testSuite = testLoader.discover(os.path.dirname(__file__), pattern="*.py")

if __name__ == "__main__":
    # Run the tests
    testRunner = unittest.TextTestRunner()
    result = testRunner.run(testSuite)

# delete model logs from tests that pollute the project files
project_dir = getProjectDirectory()
dirsToDelete = [["tests", "lightning_logs"], ["tests", "NNDummy"]]

for dir_ in dirsToDelete:
    dirToDelete = os.path.join(project_dir, *dir_)
    if os.path.exists(dirToDelete):
        try:
            shutil.rmtree(dirToDelete)
        except:
            pass
