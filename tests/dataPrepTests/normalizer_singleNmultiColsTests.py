# ----


import unittest

import pandas as pd
import pydantic

from dataPrep.normalizers.baseEncoders import _StdScaler, _LblEncoder, _IntLabelsString
from dataPrep.normalizers.multiColNormalizer import MultiColStdNormalizer, MultiColLblEncoder
from dataPrep.normalizers.normalizerStack import NormalizerStack
from dataPrep.normalizers.normalizers_singleColsNormalizer import SingleColsStdNormalizer, \
    SingleColsLblEncoder, _BaseSingleColsNormalizer
from tests.baseTest import BaseTestClass


# ---- stdNormalizerTest
class stdNormalizerTests(BaseTestClass):
    def __init__(self, *args, **kwargs):
        super(stdNormalizerTests, self).__init__(*args, **kwargs)
        self.expectedPrint = {}
        self.expectedPrint[
            'testFitAgain'] = "SingleColsStdNormalizer:col1_col2 col1 is already fitted\n" \
                              "SingleColsStdNormalizer:col1_col2 col2 is already fitted\n" \
                              "MultiColStdNormalizer:col3_col4 is already fitted\n"

    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({
            'col1': range(0, 11),
            'col2': range(30, 41),
            'col3': range(40, 51),
            'col4': range(80, 91)}, index=range(100, 111)).astype(float)
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
        self.normalizerStack = NormalizerStack(
            SingleColsStdNormalizer(['col1', 'col2']),
            MultiColStdNormalizer(['col3', 'col4']))
        self.transformedDf = pd.DataFrame({'col1': [-1.58113883, -1.26491106,
                                                    -0.9486833, -0.63245553,
                                                    -0.31622777, 0., 0.31622777,
                                                    0.63245553, 0.9486833,
                                                    1.26491106, 1.58113883],
                                           'col2': [-1.58113883, -1.26491106,
                                                    -0.9486833, -0.63245553,
                                                    -0.31622777, 0., 0.31622777,
                                                    0.63245553, 0.9486833,
                                                    1.26491106, 1.58113883],
                                           'col3': [-1.234662, -1.18527552,
                                                    -1.13588904, -1.08650256,
                                                    -1.03711608, -0.9877296,
                                                    -0.93834312, -0.88895664,
                                                    -0.83957016, -0.79018368,
                                                    -0.7407972],
                                           'col4': [0.7407972, 0.79018368,
                                                    0.83957016, 0.88895664,
                                                    0.93834312, 0.9877296,
                                                    1.03711608, 1.08650256,
                                                    1.13588904, 1.18527552,
                                                    1.234662]},
                                          index=range(100, 111))
        # self.transformedDfUntouched = self.transformedDf.copy()
        # self.floatPrecision= 0.001

    def inverseTransformSetUp(self):
        self.transformSetUp()
        self.normalizerStack.fitNTransform(self.dfToDoTest)

    def testFitNTransform(self):
        self.transformSetUp()
        # goodToHave2 if the self.setUp is not used give a warning
        self.normalizerStack.fitNTransform(self.dfToDoTest)
        self.equalDfs(self.dfToDoTest, self.transformedDf, floatApprox=True)

    def testFitAgain(self):
        # cccDevStruct this is example of checking expectedPrints in tests

        def innerFunc():
            self.transformSetUp()
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            self.equalDfs(self.dfToDoTest, self.transformedDf, floatApprox=True)

        self.assertPrint(innerFunc, self.expectedPrint['testFitAgain'])

    def testInverseTransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.inverseTransform(self.dfToDoTest)
        self.equalDfs(self.dfToDoTest, self.dfUntouched)

    # addTest2 what meaningful tests can be added??


# ---- lblEncoderTests
class LblEncoderTest(stdNormalizerTests):
    def __init__(self, *args, **kwargs):
        super(LblEncoderTest, self).__init__(*args, **kwargs)
        self.expectedPrint = {}
        self.expectedPrint[
            'testFitAgain'] = "SingleColsLblEncoder:col1_col2 col1 is already fitted\n" \
                              "SingleColsLblEncoder:col1_col2 col2 is already fitted\n" \
                              "MultiColLblEncoder:col3_col4 is already fitted\n"

        self.expectedPrint['testTransformAgain'] = 'LblEncoder applied transform on lbl:col1\n' \
                                                   'LblEncoder applied transform on lbl:col2\n' \
                                                   'LblEncoder applied transform on lbl:col3_col4\n' \
                                                   'LblEncoder applied transform on lbl:col3_col4\n'

        self.expectedPrint[
            'testInverseTransformAgain'] = 'LblEncoder applied transform on lbl:col1\n' \
                                           'LblEncoder applied transform on lbl:col2\n' \
                                           'LblEncoder applied transform on lbl:col3_col4\n' \
                                           'LblEncoder applied transform on lbl:col3_col4\n' \
                                           'LblEncoder applied inverseTransform on lbl:col1\n' \
                                           'LblEncoder applied inverseTransform on lbl:col2\n' \
                                           'LblEncoder applied inverseTransform on lbl:col3_col4\n' \
                                           'LblEncoder applied inverseTransform on lbl:col3_col4\n'

    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({
            'col1': ['a', 'd', 'ds', 's', 'a'],
            'col2': ['col2sd', 'col2dsa', 'col2dsa', '21dxs', '21dxs'],
            'col3': ['nkcdf', 'mdeo', 'nkcdf', 'cd', 'a'],
            'col4': ['z11', 'sc22', 'oem2', 'medk3', 'df']},
            index=range(100, 105))

        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()

        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1', 'col2']),
            MultiColLblEncoder(['col3', 'col4']))
        self.transformedDf = pd.DataFrame({'col1': [0, 1, 2, 3, 0],
                                           'col2': [2, 1, 1, 0, 0],
                                           'col3': [5, 3, 5, 1, 0],
                                           'col4': [8, 7, 6, 4, 2]},
                                          index=range(100, 105))

    # self.transformedDfUntouched = self.transformedDf.copy()
    # self.floatPrecision= 0.001

    def testTransformAgain(self):
        def innerFunc():
            self.transformSetUp()
            self.normalizerStack.fitNTransform(self.dfToDoTest)
            self.normalizerStack.transformCol(self.dfToDoTest, 'col2')
            self.equalDfs(self.dfToDoTest, self.transformedDf, floatApprox=True)

        self.assertPrint(innerFunc, self.expectedPrint['testTransformAgain'])

    def testInverseTransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.inverseTransform(self.dfToDoTest)
        self.equalDfs(self.dfToDoTest, self.dfUntouched)

    def testInverseTransformAgain(self):
        def innerFunc():
            self.inverseTransformSetUp()
            self.normalizerStack.inverseTransform(self.dfToDoTest)
            self.normalizerStack.inverseTransform(self.dfToDoTest)
            self.equalDfs(self.dfToDoTest, self.dfUntouched)

        self.assertPrint(innerFunc, self.expectedPrint['testInverseTransformAgain'])


class lblEncoderWithIntLabelsStringTests(BaseTestClass):
    def transformSetUp(self):
        self.dfUntouched = pd.DataFrame({'col1': [3, 3, 0, 0, 1, 4],
                                         'col2': [0, 3, 0, 1, 0, 2],
                                         'col3': [2, 1, 0, 3, 4, 0]},
                                        index=range(100, 106))
        self.dfToDoTest = self.dfUntouched.copy()
        self.dfAssertDummy = self.dfUntouched.copy()
        self.transformedDf = pd.DataFrame({'col1': [2, 2, 0, 0, 1, 3],
                                           'col2': [0, 3, 0, 1, 0, 2],
                                           'col3': [2, 1, 0, 3, 4, 0]},
                                          index=range(100, 106))

        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1']),
            MultiColLblEncoder(['col2', 'col3']))

    def inverseTransformSetUp(self):
        self.transformSetUp()
        self.normalizerStack.fitNTransform(self.dfToDoTest)

    def testFitNTransform(self):
        stdNormalizerTests.testFitNTransform(self)

    def testNotAllInts_raiseValueError_withNormalizer(self):
        df = pd.DataFrame({'col1': [3, 3.1, 0, 0, 1, 4]},
                          index=range(100, 106))
        normalizerStack = NormalizerStack(SingleColsLblEncoder(['col1']))
        with self.assertRaises(ValueError) as context:
            normalizerStack.fitNTransform(df)
        self.assertEqual(str(context.exception), _LblEncoder.floatDetectedErrorMsg)

    def testInverseTransform(self):
        self.inverseTransformSetUp()
        self.normalizerStack.inverseTransform(self.dfToDoTest)
        self.equalDfs(self.dfToDoTest, self.dfUntouched)


# ---- other tests
class otherTests(BaseTestClass):
    def testNormalizerStack_addNormalizer(self):
        self.normalizerStack = NormalizerStack(
            SingleColsLblEncoder(['col1']),
            MultiColLblEncoder(['col2', 'col3']))
        self.normalizerStack.addNormalizer(SingleColsLblEncoder(['col4']))
        # kkk print or assert sth

    def testNonDfOrSeries_pydanticAssertion(self):
        StdScaler_ = _StdScaler()
        with self.assertRaises(pydantic.ValidationError) as context:
            StdScaler_.fit([5, 3, 7])
        self.assertTrue('validation errors for Fit' in str(context.exception))

    def testLblEncoderIntRaiseValueError(self):
        lblEnc = _LblEncoder()
        with self.assertRaises(ValueError) as context:
            df = pd.DataFrame({'col1': [3, 3, 0, 0, 1, 2]})
            lblEnc.fit(df['col1'])
        self.assertEqual(str(context.exception),
                         _LblEncoder.intDetectedErrorMsg)

    def testNotAllInts_raiseValueError_intLabelsString(self):
        df = pd.DataFrame({'col1': [3, 3.1, 0, 0, 1, 4]},
                          index=range(100, 106))
        intLabelsString = _IntLabelsString('col1')
        with self.assertRaises(ValueError) as context:
            intLabelsString.fit(df)
        self.assertEqual(str(context.exception),
                         "_IntLabelsString col1 All values should be integers.")

    def testLblEncoderFloatRaiseValueError(self):
        lblEnc = _LblEncoder()
        with self.assertRaises(ValueError) as context:
            df = pd.DataFrame({'col1': [3, 3.1, 0, 0, 1, 2]})
            lblEnc.fit(df['col1'])
        self.assertEqual(str(context.exception),
                         _LblEncoder.floatDetectedErrorMsg)

    def testNotInstanceOf_BaseSingleColsNormalizer(self):
        with self.assertRaises(RuntimeError) as context:
            _BaseSingleColsNormalizer()
        self.assertEqual(str(context.exception),
                         'Instances of _BaseSingleColsNormalizer are not allowed')


# ---- run test
if __name__ == '__main__':
    unittest.main()
