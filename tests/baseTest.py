import unittest
import sys
import io
from utils.vAnnGeneralUtils import equalDfs, equalArrays, equalTensors, equalNpDicts, varPasser
# ----
class BaseTestClass(unittest.TestCase):
    def assertPrint(self, testFunc, expectedPrint, **kwargsOfTestFunc):
        capturedOutput = io.StringIO()
        
        # Redirect stdout to the StringIO object
        sys.stdout = capturedOutput

        try:
            # Run the test function
            testFunc(**kwargsOfTestFunc)
            
            # Get the captured output as a string with newline characters
            printed = capturedOutput.getvalue()
            
            # Assert that the expected print was printed
            self.assertIn(expectedPrint, printed)
        finally:
            # Restore the original stdout even if an exception occurs
            sys.stdout = sys.__stdout__

    def equalDfs(self, df1, df2,
                 checkIndex=True, floatApprox=False, floatPrecision=0.0001):
        kwargs_ = varPasser(
            localArgNames=['df1', 'df2', 'checkIndex', 'floatApprox', 'floatPrecision'])
        self.assertTrue(equalDfs(**kwargs_))

    def equalArrays(self, array1, array2,
                    checkType=True, floatApprox=False, floatPrecision=1e-4):
        kwargs_ = varPasser(
            localArgNames=['array1', 'array2', 'checkType', 'floatApprox', 'floatPrecision'])
        self.assertTrue(equalArrays(**kwargs_))

    def equalTensors(self, tensor1, tensor2,
                     checkType=True, floatApprox=False,
                     floatPrecision=1e-4, checkDevice=True):
        kwargs_ = varPasser(
            localArgNames=['tensor1', 'tensor2', 'checkType', 'floatApprox', 'floatPrecision',
                           'checkDevice'])
        self.assertTrue(equalTensors(**kwargs_))
        
    def equalNpDicts(self, npd1, npd2,
                     checkIndex=True, floatApprox=False, floatPrecision=0.0001):
        kwargs_ = varPasser(
            localArgNames=['npd1', 'npd2', 'checkIndex', 'floatApprox', 'floatPrecision'])
        self.assertTrue(equalNpDicts(**kwargs_))