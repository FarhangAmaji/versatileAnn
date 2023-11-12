import unittest
import sys
import io
from utils.vAnnGeneralUtils import equalDfs, equalArrays, equalTensors, equalNpDicts, varPasser
# ----
class BaseTestClass(unittest.TestCase):
    def assertPrint(self, testFunc, expectedPrint):
        capturedOutput = io.StringIO()
        
        # Redirect stdout to the StringIO object
        sys.stdout = capturedOutput

        try:
            # Run the test function
            testFunc()
            
            # Get the captured output as a string with newline characters
            printed = capturedOutput.getvalue()
            
            # Assert that the expected print was printed
            self.assertIn(expectedPrint, printed)
        finally:
            # Restore the original stdout even if an exception occurs
            sys.stdout = sys.__stdout__

    def equalDfs(self, df1, df2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
        kwargs = varPasser(locals())
        self.assertTrue(equalDfs(**kwargs))

    def equalArrays(self, array1, array2, checkType=True, floatApprox=False, floatPrecision=1e-4):
        kwargs = varPasser(locals())
        self.assertTrue(equalArrays(**kwargs))

    def equalTensors(self, tensor1, tensor2, checkType=True, floatApprox=False, floatPrecision=1e-4, checkDevice=True):
        kwargs = varPasser(locals())
        self.assertTrue(equalTensors(**kwargs))
        
    def equalNpDicts(self, npd1, npd2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
        kwargs = varPasser(locals())
        self.assertTrue(equalNpDicts(**kwargs))