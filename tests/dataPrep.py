import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
#%%
from dataPreparation.utils.dataPrepUtils import (NormalizerStack, SingleColsStdNormalizer, MultiColStdNormalizer,
                                                 SingleColsLblEncoder, MultiColLblEncoder, equalDfs, dfToNpDict,
                                                 makeIntLabelsString, LblEncoder, LblEncoderValueErrorMsg)
import pandas as pd
#%% stdNormalizerTest
class stdNormalizerTest(BaseTestClass):
    def __init__(self, *args, **kwargs):
        super(stdNormalizerTest, self).__init__(*args, **kwargs)
        self.expectedPrint={}
        self.expectedPrint['testFitAgain']="""SingleColsStdNormalizer+col1_col2 col1 is already fitted
SingleColsStdNormalizer+col1_col2 col2 is already fitted
MultiColStdNormalizer+col3_col4 is already fitted
"""
    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({
            'col1': range(0, 11),
            'col2': range(30, 41),
            'col3': range(40, 51),
            'col4': range(80, 91)},index=range(100, 111)).astype(float)
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
        self.normalizerStack = NormalizerStack(
            SingleColsStdNormalizer(['col1', 'col2']),
            MultiColStdNormalizer(['col3', 'col4']))
        self.transformedDf = pd.DataFrame({'col1': [-1.58113883, -1.26491106, -0.9486833 , -0.63245553, -0.31622777,0.        ,  0.31622777,  0.63245553,  0.9486833 ,  1.26491106,1.58113883],
                                           'col2': [-1.58113883, -1.26491106, -0.9486833 , -0.63245553, -0.31622777,0.        ,  0.31622777,  0.63245553,  0.9486833 ,  1.26491106,1.58113883],
                                           'col3': [-1.234662  , -1.18527552, -1.13588904, -1.08650256, -1.03711608,-0.9877296 , -0.93834312, -0.88895664, -0.83957016, -0.79018368,-0.7407972 ],
                                           'col4': [0.7407972 , 0.79018368, 0.83957016, 0.88895664, 0.93834312,0.9877296 , 1.03711608, 1.08650256, 1.13588904, 1.18527552,1.234662  ]},
                                          index=range(100, 111))
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
        "#ccc this is example of checking expectedPrints in tests"
        def testFunc():
            self.transformSetUp()
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            assert equalDfs(self.dfToDoTest, self.transformedDf)
        self.assertPrint(testFunc, self.expectedPrint['testFitAgain'])

    def testInverseTransformCol(self):
        self.inverseTransformSetUp()
        self.dfToDoTest['col1'] = self.normalizerStack.inverseTransformCol(self.dfToDoTest, 'col1')#SingleColsStdNormalizer
        self.dfToDoTest['col4'] = self.normalizerStack.inverseTransformCol(self.dfToDoTest, 'col4')#MultiColStdNormalizer
        # for assert modification
        self.dfAssertDummy['col2']=self.transformedDf['col2']
        self.dfAssertDummy['col3']=self.transformedDf['col3']
        assert equalDfs(self.dfToDoTest, self.dfAssertDummy)

    def testInverseIransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.inverseTransform(self.dfToDoTest)
        assert equalDfs(self.dfToDoTest, self.dfUntouched)

    #kkk add test for addNormalizer in NormalizerStack
    #kkk what meaningful tests can be added??
#%% lblEncoderTest
class lblEncoderTest(stdNormalizerTest):
    def __init__(self, *args, **kwargs):
        super(lblEncoderTest, self).__init__(*args, **kwargs)
        self.expectedPrint={}
        self.expectedPrint['testFitAgain']="""SingleColsLblEncoder+col1_col2 col1 is already fitted
SingleColsLblEncoder+col1_col2 col2 is already fitted
MultiColLblEncoder+col3_col4 is already fitted
"""

    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({
            'col1': ['a','d','ds','s','a'],
            'col2': ['col2sd','col2dsa','col2dsa','21dxs','21dxs'],
            'col3': ['nkcdf','mdeo','nkcdf','cd','a'],
            'col4': ['z11','sc22','oem2','medk3','df']},index=range(100, 105))
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
            
        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1','col2']),
            MultiColLblEncoder(['col3', 'col4']))
        self.transformedDf = pd.DataFrame({'col1': [0,1,2,3,0],
                                           'col2': [2, 1, 1, 0, 0],
                                           'col3': [5, 3, 5, 1, 0],
                                           'col4': [8, 7, 6, 4, 2]},index=range(100, 105))
        # self.transformedDfUntouched = self.transformedDf.copy()
        # self.floatPrecision= 0.001

    def testUltimateInverseIransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.ultimateInverseTransform(self.dfToDoTest)#!!! 
        assert equalDfs(self.dfToDoTest, self.dfUntouched)

class lblEncoderWithMakeIntLabelsStringTest(BaseTestClass):
    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({'col1': [3, 3, 0, 0, 1, 4],
                                           'col2': [0, 3, 0, 1, 0, 2],
                                           'col3': [2, 1, 0, 3, 4, 0]},index=range(100, 106))
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
        self.transformedDf = pd.DataFrame({'col1': [2, 2, 0, 0, 1, 3],
                                           'col2': [0, 3, 0, 1, 0, 2],
                                           'col3': [2, 1, 0, 3, 4, 0]},index=range(100, 106))
        self.inverseTransforRes = pd.DataFrame({'col1': ['col1:3', 'col1:3', 'col1:0', 'col1:0', 'col1:1', 'col1:2'],
                                            'col2': ['lbl:col2_col3:0', 'lbl:col2_col3:3', 'lbl:col2_col3:0',
                                            'lbl:col2_col3:1', 'lbl:col2_col3:0', 'lbl:col2_col3:2'],
                                            'col3': ['lbl:col2_col3:2', 'lbl:col2_col3:1', 'lbl:col2_col3:0',
                                            'lbl:col2_col3:3', 'lbl:col2_col3:4', 'lbl:col2_col3:0']},index=range(100, 106))
        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1']),
            MultiColLblEncoder(['col2', 'col3']))

    def inverseTransformSetUp(self):
        stdNormalizerTest.inverseTransformSetUp(self)

    def testFitNTransform(self):
        stdNormalizerTest.testFitNTransform(self)

    def testInverseIransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.inverseTransform(self.dfToDoTest)
        assert equalDfs(self.dfToDoTest, self.inverseTransforRes)

    def testUltimateInverseIransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.ultimateInverseTransform(self.dfToDoTest)#!!! 
        assert equalDfs(self.dfToDoTest, self.dfUntouched)
#%% other tests
class otherTests(BaseTestClass):
    def testNormalizerStack_addNormalizer(self):
        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1']),
            MultiColLblEncoder(['col2', 'col3']))
        self.normalizerStack.addNormalizer(SingleColsLblEncoder(['col4']))

    def testLblEncoderRaiseValueError(self):
        lblEnc=LblEncoder()
        with self.assertRaises(ValueError) as context:
            df= pd.DataFrame({'col1': [3, 3, 0, 0, 1, 2]})
            lblEnc.fit(df['col1'])
        self.assertEqual(str(context.exception), LblEncoderValueErrorMsg)
#%%
if __name__ == '__main__':
    unittest.main()