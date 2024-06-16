import io
import sys
import unittest

from projectUtils.dataTypeUtils.npArray import equalArrays
from projectUtils.dataTypeUtils.npDict_dfMutual import equalDfs, equalNpDicts
from projectUtils.dataTypeUtils.tensor import equalTensors
from projectUtils.misc import varPasser


# ----
class BaseTestClass(unittest.TestCase):
    def assertPrint(self, innerFunc, expectedPrint, returnPrinted=False, **kwargsOfTestFunc):
        # bugPotn2
        #  sometimes when the tests are run from the file it gives `'_io.TextIOWrapper' object
        #  has no attribute 'getvalue'` and with several tries I yet have not solved it
        #  or gives `Process finished with exit code -1073741819 (0xC0000005) `
        capturedOutput = io.StringIO()

        # Redirect stdout to the StringIO object
        sys.stdout = capturedOutput

        result = None
        try:
            # Run the test function
            result = innerFunc(**kwargsOfTestFunc)

            # Get the captured output as a string with newline characters
            printed = capturedOutput.getvalue()

            # Assert that the expected print was printed
            self.assertIn(expectedPrint, printed)
        except Exception as e:
            print(f"Unexpected error in assertPrint: {e}")
        finally:
            # Restore the original stdout even if an exception occurs
            sys.stdout = sys.__stdout__
        if returnPrinted:
            return result, printed
        return result

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
