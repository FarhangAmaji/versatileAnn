import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
#%%
from dataPreparation.utils.dataPrepUtils import (NormalizerStack, SingleColsStdNormalizer, MultiColStdNormalizer,
                                                 SingleColsLblEncoder, MultiColLblEncoder, equalDfs, dfToNpDict, makeIntLabelsString)
import pandas as pd
#%% stdNormalizerTest
class stdNormalizerTest(BaseTestClass):
    def __init__(self, *args, **kwargs):
        super(stdNormalizerTest, self).__init__(*args, **kwargs)
        self.expectedPrint={}
        self.expectedPrint['testFitAgain']="""StdScaler stdcol1 is already fitted
StdScaler stdcol1 skipping transform: Mean of dataToFit is between -1 and 1; so seems to be already fitted.
StdScaler stdcol2 is already fitted
StdScaler stdcol2 skipping transform: Mean of dataToFit is between -1 and 1; so seems to be already fitted.
StdScaler stdcol3_col4 is already fitted
StdScaler stdcol3_col4 skipping transform: Mean of dataToFit is between -1 and 1; so seems to be already fitted.
"""
        self.expectedPrint['testInverseTransformColAgain']="StdScaler stdcol1 skipping inverse transform: Mean of dataToInverseTransformed is not between -1 and 1, since seems the dataToInverseTransformed not to be normalized\n"
    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({
            'col1': range(0, 11),
            'col2': range(30, 41),
            'col3': range(40, 51),
            'col4': range(80, 91)}).astype(float)
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
        self.normalizerStack = NormalizerStack(
            SingleColsStdNormalizer(['col1', 'col2']),
            MultiColStdNormalizer(['col3', 'col4']))
        self.transformedDf = pd.DataFrame({'col1': [-1.58113883, -1.26491106, -0.9486833 , -0.63245553, -0.31622777,0.        ,  0.31622777,  0.63245553,  0.9486833 ,  1.26491106,1.58113883],
                                           'col2': [-1.58113883, -1.26491106, -0.9486833 , -0.63245553, -0.31622777,0.        ,  0.31622777,  0.63245553,  0.9486833 ,  1.26491106,1.58113883],
                                           'col3': [-1.234662  , -1.18527552, -1.13588904, -1.08650256, -1.03711608,-0.9877296 , -0.93834312, -0.88895664, -0.83957016, -0.79018368,-0.7407972 ],
                                           'col4': [0.7407972 , 0.79018368, 0.83957016, 0.88895664, 0.93834312,0.9877296 , 1.03711608, 1.08650256, 1.13588904, 1.18527552,1.234662  ]})
        # self.transformedDfUntouched = self.transformedDf.copy()
        # self.floatPrecision= 0.001
        

    def inverseTransformSetUp(self):
        self.transformSetUp()
        self.normalizerStack.fitNTransform(self.dfToDoTest)


    def testFitNTransform(self):
        self.transformSetUp()#kkk if the self.setUp is not used give a warning
        self.normalizerStack.fitNTransform(self.dfToDoTest)
        assert equalDfs(self.dfToDoTest, self.transformedDf)

    def testFitAgain(self):
        #nip#htd this example of checking expectedPrints
        def testFunc():
            self.transformSetUp()
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            assert equalDfs(self.dfToDoTest, self.transformedDf)
        self.assertPrint(testFunc, self.expectedPrint['testFitAgain'])

    def testInverseTransformCol(self):
        self.inverseTransformSetUp()
        self.dfToDoTest['col1'] = self.normalizerStack.inverseTransformCol(self.dfToDoTest, 'col1')
        self.dfToDoTest['col4'] = self.normalizerStack.inverseTransformCol(self.dfToDoTest, 'col4')
        # for assert modification
        self.dfAssertDummy['col2']=self.transformedDf['col2']
        self.dfAssertDummy['col3']=self.transformedDf['col3']
        assert equalDfs(self.dfToDoTest, self.dfAssertDummy)

    def testInverseTransformColAgain(self):
        def testFunc():
            self.inverseTransformSetUp()
            self.dfToDoTest['col1'] = self.normalizerStack.inverseTransformCol(self.dfToDoTest, 'col1')
            self.dfToDoTest['col1'] = self.normalizerStack.inverseTransformCol(self.dfToDoTest, 'col1')
            # for assert modification
            self.dfAssertDummy['col2']=self.transformedDf['col2']
            self.dfAssertDummy['col3']=self.transformedDf['col3']
            self.dfAssertDummy['col4']=self.transformedDf['col4']
            assert equalDfs(self.dfToDoTest, self.dfAssertDummy)
        self.assertPrint(testFunc, self.expectedPrint['testInverseTransformColAgain'])

    def testInverseIransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.inverseTransform(self.dfToDoTest)
        assert equalDfs(self.dfToDoTest, self.dfAssertDummy)

    #kkk add test for addNormalizer in NormalizerStack
    #kkk what meaningful tests can be added??
#%% lblEncoderTest
class lblEncoderTest(stdNormalizerTest):
    def __init__(self, *args, **kwargs):
        super(lblEncoderTest, self).__init__(*args, **kwargs)
        self.expectedPrint={}
        self.expectedPrint['testFitAgain']="""LblEncoder lblcol1 is already fitted
LblEncoder lblcol1 skipping transform: data already seems transformed.
LblEncoder lblcol2 is already fitted\nLblEncoder lblcol2 skipping transform: data already seems transformed.
LblEncoder lblcol3_col4 is already fitted\nLblEncoder lblcol3_col4 skipping transform: data already seems transformed.
"""
        self.expectedPrint['testInverseTransformColAgain']="LabelEncoder lblcol1 skipping inverse transform: data already seems inverse transformed.\n"

    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({
            'col1': ['a','d','ds','s','a'],
            'col2': ['col2sd','col2dsa','col2dsa','21dxs','21dxs'],
            'col3': ['nkcdf','mdeo','nkcdf','cd','a'],
            'col4': ['z11','sc22','oem2','medk3','df']})
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
            
        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1','col2']),
            MultiColLblEncoder(['col3', 'col4']))
        self.transformedDf = pd.DataFrame({'col1': [0,1,2,3,0],
                                           'col2': [2, 1, 1, 0, 0],
                                           'col3': [5, 3, 5, 1, 0],
                                           'col4': [8, 7, 6, 4, 2]})
        # self.transformedDfUntouched = self.transformedDf.copy()
        # self.floatPrecision= 0.001
#%% makeIntLabelsStringTest
class makeIntLabelsStringTest(BaseTestClass):
    def setUp(self):
        self.df =pd.DataFrame({'col1': [5, 3, 5, 1, 0]})
        self.dfAfterMakeIntLabelsString =pd.DataFrame({'col1': ['col10', 'col11', 'col10', 'col12', 'col13']})
        self.dfEncoded =pd.DataFrame({'col1': [0, 1, 0, 2, 3]})
        self.normalizerStack = NormalizerStack(SingleColsLblEncoder(['col1']))

    def testLblEncoderRaiseValErrorOverIntCols(self):
        #nip#htd this example of expecting error with specific error message
        self.setUp()
        with self.assertRaises(ValueError) as context:
            self.normalizerStack.fitNTransform(self.df)
        
        # Check the exception message
        self.assertEqual(str(context.exception), "Integer labels detected. Use makeIntLabelsString to convert them to string labels.")

    def testApplyMakeIntLabelsString(self):
        self.setUp()
        makeIntLabelsString(self.df, 'col1')
        assert equalDfs(self.df, self.dfAfterMakeIntLabelsString)
        self.normalizerStack.fitNTransform(self.df)
        assert equalDfs(self.df, self.dfEncoded)
#%%
if __name__ == '__main__':
    unittest.main()