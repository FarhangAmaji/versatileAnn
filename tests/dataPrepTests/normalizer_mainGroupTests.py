# ----


import unittest

import pandas as pd

from dataPrep.normalizers_mainGroupNormalizers import (_MainGroupBaseNormalizer,
                                                       MainGroupSingleColsStdNormalizer,
                                                       MainGroupSingleColsLblEncoder)
from dataPrep.normalizers_normalizerStack import NormalizerStack
from tests.baseTest import BaseTestClass


# ---- _MainGroupBaseNormalizer tests
class MainGroupBaseNormalizerTests(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': ['A1', 'A2', 'A3', 'A4', 'A1', 'A3'],
            'B': ['B1', 'B2', 'B4', 'B4', 'B1', 'B2'],
            'C': ['C1', 'C4', 'C4', 'C4', 'C2', 'C2'],
            'col1': [3, 3, 0, 0, 1, 4],
            'col2': [0, 3, 0, 1, 0, 2],
            'col3': [2, 1, 0, 3, 4, 0]}, index=range(100, 106))
        self.getRowsByCombination1Res = pd.DataFrame({'A': ['A1', 'A1'],
                                                      'B': ['B1', 'B1'],
                                                      'C': ['C1', 'C1'],
                                                      'col1': [3, 1],
                                                      'col2': [0, 0],
                                                      'col3': [2, 4]},
                                                     index=[100, 104])
        self.getRowsByCombination2Res = pd.DataFrame({'A': ['A1'],
                                                      'B': ['B1'],
                                                      'C': ['C1'],
                                                      'col1': [3],
                                                      'col2': [0],
                                                      'col3': [2]}, index=[100])

    def UniqueCombos_BaseTestFunc(self, mainGroupColNames,
                                  uniqueCombosAssertComboDefs):
        self.setUp()
        MainGroupBaseNormalizer_ = _MainGroupBaseNormalizer(self.df, mainGroupColNames,
                                                            internalCall=True)
        uniqueCombos = MainGroupBaseNormalizer_.uniqueCombos

        testSuccessRes = True
        for com in uniqueCombosAssertComboDefs:
            if not MainGroupBaseNormalizer_.findMatchingCombo_dictRepr(com):
                testSuccessRes = False

        for com in uniqueCombos:
            if com not in [str(com1) for com1 in uniqueCombosAssertComboDefs]:
                testSuccessRes = False

        self.assertTrue(testSuccessRes)

    def testUniqueCombos1(self):
        uniqueCombosAssertComboDefs = [{'A': 'A1', 'B': 'B1'},
                                       {'A': 'A2', 'B': 'B2'},
                                       {'A': 'A3', 'B': 'B2'},
                                       {'A': 'A3', 'B': 'B4'},
                                       {'A': 'A4', 'B': 'B4'}]
        mainGroupColNames = ["A", "B"]
        self.UniqueCombos_BaseTestFunc(mainGroupColNames,
                                       uniqueCombosAssertComboDefs)

    def testUniqueCombos2(self):
        mainGroupColNames = ["A", "B", "C"]
        uniqueCombosAssertComboDefs = [{'A': 'A1', 'B': 'B1', 'C': 'C1'},
                                       {'A': 'A1', 'B': 'B1', 'C': 'C2'},
                                       {'A': 'A2', 'B': 'B2', 'C': 'C4'},
                                       {'A': 'A3', 'B': 'B2', 'C': 'C2'},
                                       {'A': 'A3', 'B': 'B4', 'C': 'C4'},
                                       {'A': 'A4', 'B': 'B4', 'C': 'C4'}]
        self.UniqueCombos_BaseTestFunc(mainGroupColNames,
                                       uniqueCombosAssertComboDefs)

    def GetRowsByCombination_BaseTestFunc(self, comboToFind, mainGroupColNames,
                                          getRowsByCombinationRes):
        self.setUp()
        MainGroupBaseNormalizer_ = _MainGroupBaseNormalizer(self.df, mainGroupColNames,
                                                            internalCall=True)
        res = MainGroupBaseNormalizer_.getRowsByCombination(self.df,
                                                            comboToFind)
        self.equalDfs(getRowsByCombinationRes, res, floatApprox=True)

    def testGetRowsByCombination1(self):
        comboToFind = {'A': 'A1', 'B': 'B1'}
        mainGroupColNames = ["A", "B"]
        self.GetRowsByCombination_BaseTestFunc(comboToFind, mainGroupColNames,
                                               self.getRowsByCombination1Res)

    def testGetRowsByCombination2(self):
        comboToFind = {'A': 'A1', 'B': 'B1', 'C': 'C1'}
        mainGroupColNames = ["A", "B", "C"]
        self.GetRowsByCombination_BaseTestFunc(comboToFind, mainGroupColNames,
                                               self.getRowsByCombination2Res)


# ---- MainGroupSingleColsNormalizerTests
class MainGroupSingleColsStdNormalizerTests(BaseTestClass):
    def setUp(self):
        # kkk could had a better example
        self.df = pd.DataFrame(data={
            'A': ['A1', 'A2', 'A3', 'A4', 'A1', 'A3'],
            'B': ['B1', 'B2', 'B4', 'B4', 'B1', 'B2'],
            'C': ['C1', 'C4', 'C4', 'C4', 'C1', 'C2'],
            'col1': [3, 3, 0, 0, 1, 4],
            'col2': [0, 3, 0, 1, 0, 2],
            'col3': [2, 1, 0, 3, 4, 0]}, index=range(100, 106))
        self.dfUntouched = self.df.copy()
        self.dfFitNTransform = pd.DataFrame(data={
            'A': ['A1', 'A2', 'A3', 'A4', 'A1', 'A3'],
            'B': ['B1', 'B2', 'B4', 'B4', 'B1', 'B2'],
            'C': ['C1', 'C4', 'C4', 'C4', 'C1', 'C2'],
            'col1': [1, 0, 0, 0, -1, 0],
            'col2': [0, 0, 0, 0, 0, 0],
            'col3': [2, 1, 0, 3, 4, 0]}, index=range(100, 106))

    def normalizerStackSetUp(self):
        self.normalizerStack = NormalizerStack(
            MainGroupSingleColsStdNormalizer(self.df, ['A', 'B'],
                                             ['col1', 'col2']))

    def testStraightFitNTransform(self):
        self.setUp()
        MainGroupBaseNormalizer_ = MainGroupSingleColsStdNormalizer(self.df,
                                                                    ['A', 'B'],
                                                                    ['col1',
                                                                     'col2'])
        MainGroupBaseNormalizer_.fitNTransform(self.df)
        self.equalDfs(self.df, self.dfFitNTransform)

    def testNormalizerStackFitNTransform(self):
        self.setUp()
        self.normalizerStackSetUp()
        self.normalizerStack.fitNTransform(self.df)
        self.equalDfs(self.df, self.dfFitNTransform, floatApprox=True)

    def testNormalizerStackInverseTransform(self):
        self.testNormalizerStackFitNTransform()
        self.normalizerStack.inverseTransform(self.df)
        self.equalDfs(self.df, self.dfUntouched)


class MainGroupSingleColsLblEncoderTests(MainGroupSingleColsStdNormalizerTests):
    def setUp(self):
        # kkk could had a better example
        self.df = pd.DataFrame(data={
            'A': ['A1', 'A2', 'A3', 'A4', 'A1', 'A3', 'A2'],
            'B': ['B1', 'B2', 'B4', 'B4', 'B1', 'B2', 'B2'],
            'C': ['C1', 'C4', 'C4', 'C4', 'C1', 'C2', 'C3'],
            'col1': [3, 3, 0, 0, 1, 4, 4],
            'col2': ['a', 'v', 'a', 'o', 'o', 'v', 'z'],
            'col3': [2, 1, 0, 3, 4, 0, 4]}, index=range(100, 107))
        self.dfUntouched = self.df.copy()
        self.dfFitNTransform = pd.DataFrame(data={
            'A': ['A1', 'A2', 'A3', 'A4', 'A1', 'A3', 'A2'],
            'B': ['B1', 'B2', 'B4', 'B4', 'B1', 'B2', 'B2'],
            'C': ['C1', 'C4', 'C4', 'C4', 'C1', 'C2', 'C3'],
            'col1': [1, 0, 0, 0, 0, 0, 1],
            'col2': [0, 0, 0, 0, 1, 0, 1],
            'col3': [2, 1, 0, 3, 4, 0, 4]}, index=range(100, 107))
        self.dfInverseRes = pd.DataFrame(data={
            'A': ['A1', 'A2', 'A3', 'A4', 'A1', 'A3', 'A2'],
            'B': ['B1', 'B2', 'B4', 'B4', 'B1', 'B2', 'B2'],
            'C': ['C1', 'C4', 'C4', 'C4', 'C1', 'C2', 'C3'],
            'col1': ['col1:1', 'col1:0', 'col1:0', 'col1:0', 'col1:0', 'col1:0',
                     'col1:1'],
            'col2': ['a', 'v', 'a', 'o', 'o', 'v', 'z'],
            'col3': [2, 1, 0, 3, 4, 0, 4]}, index=range(100, 107))

    def normalizerStackSetUp(self):
        self.normalizerStack = NormalizerStack(
            MainGroupSingleColsLblEncoder(self.df, ['A', 'B'],
                                          ['col1', 'col2']))

    def testStraightFitNTransform(self):
        self.setUp()
        MainGroupBaseNormalizer_ = MainGroupSingleColsLblEncoder(self.df,
                                                                 ['A', 'B'],
                                                                 ['col1',
                                                                  'col2'])
        MainGroupBaseNormalizer_.fitNTransform(self.df)
        self.equalDfs(self.df, self.dfFitNTransform, floatApprox=True)


# ---- run test
if __name__ == '__main__':
    unittest.main()
