import unittest
import sys
import io
#%%
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