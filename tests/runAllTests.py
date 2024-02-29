import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# cccUsage
#  note in few lines later I would explain why it's important to call this file from the root;
#  but it's important to know that this file is called from the root
# kkk for windows/linux/mac, add how to run this file in order to be called from the root folder
# ccc1
#  note tests/brazingTorchTests/fitTests.py uses a func which uses pytorchLogger from
#  pytorch lightning, and the path of that logger is dependent on where this code is called so
#  by default this project is designed to be called from root folder
import shutil
import unittest

from projectUtils.misc import getProjectDirectory

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
