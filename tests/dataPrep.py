import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
#%%
from dataPreparation.utils.dataPrepUtils import getDatasetFiles, NormalizerStack, SingleColsStdNormalizer, MultiColStdNormalizer
import pandas as pd
#%%
class TestDataPrep(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'col1': range(0, 11),
            'col2': range(30, 41),
            'col3': range(40, 51),
            'col4': range(80, 91)})
        self.normalizerStack = NormalizerStack(
            SingleColsStdNormalizer(['col1', 'col2']),
            MultiColStdNormalizer(['col3', 'col4']))

    def testFitNTransform(self):
        print('TEST: fitNTransform\n')
        self.normalizerStack.fitNTransform(self.df)

    def testFitAgain(self):
        print('TEST: fitAgain\n')
        self.normalizerStack.fitNTransform(self.df)

    def testInverseTransformCol(self):
        print('TEST: inverse transform cols\n')
        self.df['Col1'] = self.normalizerStack.inverseTransformCol(self.df, 'col1')
        self.df['Col4'] = self.normalizerStack.inverseTransformCol(self.df, 'col4')
        print('TEST: inverse transform cols again\n')
        self.df['Col1'] = self.normalizerStack.inverseTransformCol(self.df, 'col1')

    def testInverseIransform(self):
        print('TEST: Try inverse transform entire df with retransforming cols again\n')
        inverse_transformed_df = self.normalizerStack.inverseTransform(self.df)
#%%
if __name__ == '__main__':
    unittest.main()





