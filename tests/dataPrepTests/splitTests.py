# ---- 

from tests.baseTest import BaseTestClass
import unittest
from dataPrep.utils import splitTsTrainValTest_DfNNpDict, splitTrainValTest_NSeries
from utils.vAnnGeneralUtils import listRangesToList
from utils.globalVars import tsStartPointColName
import pandas as pd
import numpy as np
# ---- SplitTests
# ----     splitTsTrainValTest_DfNNpDict
class SplitTsTrainValTest_DfNNpDictTests(BaseTestClass):
    def setUp(self):
        np.random.seed(seed=30)
        self.backcastLen=3
        self.forecastLen=2
        self.df = pd.DataFrame({'y1': [i for i in range(10, 110)],
                                'y2': [i for i in range(110, 210)]},index=range(20, 120))
        self.dfWithCond = pd.DataFrame({'y1': [i for i in range(10, 110)],
                                        'y2': [i for i in range(110, 210)],
                                        tsStartPointColName: [True if i>40 and i<106 else False for i in range(10, 110)],
                                        'condCol': [i for i in range(110, 210)]},index=range(20, 120))

    def testSeqLen0_NoShuffle(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTest_DfNNpDict(self.df, trainRatio=.7, valRatio=.2, seqLen=0, shuffle=False)

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(10, 80)],
                                     'y2': [i for i in range(110, 180)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(80, 100)],
                                   'y2': [i for i in range(180, 200)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(100, 110)],
                                    'y2': [i for i in range(200, 210)]})
        trainDfCheck[tsStartPointColName]=True
        valDfCheck[tsStartPointColName]=True
        testDfCheck[tsStartPointColName]=True
        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck)
        self.equalDfs(testDf,testDfCheck)

    def testNoVal(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTest_DfNNpDict(self.df, trainRatio=.7, valRatio=0, seqLen=0, shuffle=False)

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(10, 80)],
                                     'y2': [i for i in range(110, 180)]})
        valDfCheck = pd.DataFrame(columns=['y1', 'y2', '__startPoint__'])
        testDfCheck = pd.DataFrame({'y1': [i for i in range(80, 110)],
                                    'y2': [i for i in range(180, 210)]})

        trainDfCheck[tsStartPointColName]=True
        valDfCheck[tsStartPointColName]=True
        testDfCheck[tsStartPointColName]=True

        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck, floatApprox=True)
        self.equalDfs(testDf,testDfCheck)

    def testTsStartPointColNameCondition(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2, seqLen=0, shuffle=False)

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(41, 86)],
                                     'y2': [i for i in range(141, 186)],
                                     tsStartPointColName: [True for i in range(41, 86)],
                                     'condCol': [i for i in range(141, 186)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(86, 99)],
                                   'y2': [i for i in range(186, 199)],
                                   tsStartPointColName: [True for i in range(86, 99)],
                                   'condCol': [i for i in range(186, 199)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(99, 106)],
                                    'y2': [i for i in range(199, 206)],
                                    tsStartPointColName: [True for i in range(99, 106)],
                                    'condCol': [i for i in range(199, 206)]})
        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck)
        self.equalDfs(testDf,testDfCheck)

    def testWithSeqLen(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
                                                            seqLen=self.backcastLen+self.forecastLen, shuffle=False)

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(41, 90)],
                                     'y2': [i for i in range(141, 190)],
                                tsStartPointColName: [True if i<86 else False for i in range(41, 90)],
                                'condCol': [i for i in range(141, 190)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(86, 103)],
                                   'y2': [i for i in range(186, 203)],
                                   tsStartPointColName: [True if i<99 else False for i in range(86, 103)],
                                   'condCol': [i for i in range(186, 203)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(99, 110)],
                                    'y2': [i for i in range(199, 210)],
                                    tsStartPointColName: [True if i<106 else False for i in range(99, 110)],
                                    'condCol': [i for i in range(199, 210)]})

        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck)
        self.equalDfs(testDf,testDfCheck)

    def testOtherCondition(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
                                                seqLen=self.backcastLen+self.forecastLen, shuffle=False,
                                                conditions=['condCol>125','condCol<200'])

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(26, 81)],
                                     'y2': [i for i in range(126, 181)],
                                     tsStartPointColName: [True if i<77 else False for i in range(26, 81)],
                                     'condCol': [i for i in range(126, 181)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(77, 96)],
                                   'y2': [i for i in range(177, 196)],
                                   tsStartPointColName: [True if i<92 else False for i in range(77, 96)],
                                   'condCol': [i for i in range(177, 196)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(92, 104)],
                                    'y2': [i for i in range(192, 204)],
                                    tsStartPointColName: [True if i<100 else False for i in range(92, 104)],
                                    'condCol': [i for i in range(192, 204)]})
        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck)
        self.equalDfs(testDf,testDfCheck)

    def testShuffle_WithSeqLen_WithOtherCondition(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
                                            seqLen=self.backcastLen+self.forecastLen, shuffle=True,conditions=['condCol>125','condCol<200'])


        getInd2s = lambda ind2s, inds: [ind2s.update([ti + i]) for i in range(self.backcastLen + self.forecastLen) for ti in inds]

        getDf = lambda ind2s, inds: pd.DataFrame({'y1': [i for i in ind2s],
                                                  'y2': [i + 100 for i in ind2s],
                                                  tsStartPointColName: [False if i not in inds else True for i in ind2s],
                                                  'condCol': [i + 100 for i in ind2s]})


        trainCheckIndexes=[26, 30, 31, 32, 34, 35, 36, 37, 39, 40, 41, 42, 45, 46, 47, 48, 50,
               51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 66, 68, 69, 70, 73,
               74, 76, 77, 80, 82, 83, 84, 85, 87, 88, 89, 90, 91, 93, 94, 95, 98]
        trainCheckIndexes2=set(trainCheckIndexes[:])
        getInd2s(trainCheckIndexes2, trainCheckIndexes)
        trainDfCheck = getDf(trainCheckIndexes2, trainCheckIndexes).reset_index(drop=True)
        self.equalDfs(trainDf,trainDfCheck)


        valCheckIndexes=[27, 29, 33, 43, 44, 61, 64, 67, 72, 75, 78, 81, 86, 92, 96]
        valCheckIndexes2=set(valCheckIndexes[:])
        getInd2s(valCheckIndexes2, valCheckIndexes)
        valDfCheck = getDf(valCheckIndexes2, valCheckIndexes)
        self.equalDfs(valDf,valDfCheck)


        testCheckIndexes=[28, 38, 49, 63, 71, 79, 97, 99]
        testCheckIndexes2=set(testCheckIndexes[:])
        getInd2s(testCheckIndexes2, testCheckIndexes)
        testDfCheck = getDf(testCheckIndexes2, testCheckIndexes)
        self.equalDfs(testDf,testDfCheck)
# ----     splitTrainValTest_NSeries
class TestSplitTrainValTest_NSeries(BaseTestClass):
    def setUp(self):
        np.random.seed(seed=30)
        self.df = pd.DataFrame({
            'A': 66*['A1'],
            'B': 27*['B1']+39*['B2'],
            tsStartPointColName: 21*[True]+6*[False]+33*[True]+6*[False],
            'y1': list(range(30, 96)),},index=range(100, 166))

    def testNoStartPointCol_0seqLen_noShuffle(self):
        df = self.df.drop(tsStartPointColName, axis=1)
        trainDf, valDf, testDf=splitTrainValTest_NSeries(df, ["A","B"], trainRatio=.6, valRatio=.2, seqLen=0, shuffle=False)
        
        trainDfCheck = pd.DataFrame({
            'A': 39*['A1'],
            'B': 16*['B1']+23*['B2'],
            tsStartPointColName: 39*[True],
            'y1': listRangesToList([range(30, 46), range(57, 80)])},index=range(0, 39))

        valDfCheck= pd.DataFrame({
            'A': 13*['A1'],
            'B': 5*['B1']+8*['B2'],
            tsStartPointColName: 13*[True],
            'y1': listRangesToList([range(46, 51), range(80, 88)])},index=range(0, 13))

        testDfCheck = pd.DataFrame({
            'A': 14*['A1'],
            'B': 6*['B1']+8*['B2'],
            tsStartPointColName: 14*[True],
            'y1': listRangesToList([range(51, 57), range(88, 96)])},index=range(0, 14))

        self.equalDfs(trainDf,trainDfCheck, floatApprox=True)
        self.equalDfs(valDf,valDfCheck, floatApprox=True)
        self.equalDfs(testDf,testDfCheck, floatApprox=True)

    def testWithStartPointCol_withSeqLen_noShuffle(self):
        trainDf, valDf, testDf=splitTrainValTest_NSeries(self.df, ["A","B"], trainRatio=.6, valRatio=.2, seqLen=7, shuffle=False)
        
        trainDfCheck = pd.DataFrame({
            'A': 43*['A1'],
            'B': 18*['B1']+25*['B2'],
            tsStartPointColName: 12*[True]+6*[False]+19*[True]+6*[False],
            'y1': listRangesToList([range(30, 48), range(57, 82)])},index=range(0, 43))

        valDfCheck= pd.DataFrame({
            'A': 23*['A1'],
            'B': 10*['B1']+13*['B2'],
            tsStartPointColName: [ True,  True,  True,  True, False, False, False, False, False,
                   False,  True,  True,  True,  True,  True,  True,  True, False,
                   False, False, False, False, False],
            'y1': listRangesToList([range(42, 52), range(76, 89)])},index=range(0, 23))

        testDfCheck = pd.DataFrame({
            'A': 24*['A1'],
            'B': 11*['B1']+13*['B2'],
            tsStartPointColName: [ True,  True,  True,  True,  True, False, False, False, False,
                   False, False,  True,  True,  True,  True,  True,  True,  True,
                   False, False, False, False, False, False],
            'y1': listRangesToList([range(46, 57), range(83, 96)])},index=range(0, 24))

        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck)
        self.equalDfs(testDf,testDfCheck)

    def testWithDifferentSeqLen_noShuffle(self):
        # having a condition doesnt make sense in general use cases, with seqlen, but it may does in some cases
        df = self.df.drop(tsStartPointColName, axis=1)
        trainDf, valDf, testDf=splitTrainValTest_NSeries(df, ["A","B"], trainRatio=.6, valRatio=.2,
                                                         trainSeqLen=8, valSeqLen=4, testSeqLen=3, shuffle=False)
        
        trainDfCheck = pd.DataFrame({
            'A': 51*['A1'],
            'B': 22*['B1']+29*['B2'],
            tsStartPointColName: [ True,  True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True, False, False, False,
                   False, False, False, False,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                    True,  True,  True,  True,  True,  True,  True,  True, False,
                   False, False, False, False, False, False],
            'y1': listRangesToList([range(30, 52), range(57, 86)])},index=range(0, 51))

        valDfCheck= pd.DataFrame({
            'A': 18*['A1'],
            'B': 8*['B1']+10*['B2'],
            tsStartPointColName: [ True,  True,  True,  True,  True, False, False, False,  True,
                    True,  True,  True,  True,  True,  True, False, False, False],
            'y1': listRangesToList([range(45, 53), range(79, 89)])},index=range(0, 18))

        testDfCheck = pd.DataFrame({
            'A': 17*['A1'],
            'B': 7*['B1']+10*['B2'],
            tsStartPointColName: [ True,  True,  True,  True,  True, False, False,  True,  True,
                    True,  True,  True,  True,  True,  True, False, False],
            'y1': listRangesToList([range(50, 57), range(86, 96)])},index=range(0, 17))

        self.equalDfs(trainDf,trainDfCheck, floatApprox=True)
        self.equalDfs(valDf,valDfCheck, floatApprox=True)
        self.equalDfs(testDf,testDfCheck, floatApprox=True)

    def testWithStartPointCol_withSeqLen_Shuffle(self):
        trainDf, valDf, testDf=splitTrainValTest_NSeries(self.df, ["A","B"], trainRatio=.6, valRatio=.2, seqLen=7, shuffle=True)
        
        trainDfCheck = pd.DataFrame({
            'A': 61*['A1'],
            'B': 24*['B1']+37*['B2'],
            tsStartPointColName: [ True,  True, False,  True,  True, False,  True, False,  True,
                    True,  True, False, False, False,  True,  True,  True,  True,
                   False, False, False, False, False, False,  True,  True, False,
                    True, False,  True,  True,  True,  True, False, False, False,
                   False, False, False,  True, False,  True,  True,  True,  True,
                   False,  True,  True,  True, False,  True, False,  True,  True,
                    True, False, False, False, False, False, False],
            'y1': listRangesToList([range(30, 54), range(59, 96)])},index=range(0, 61))

        valDfCheck= pd.DataFrame({
            'A': 58*['A1'],
            'B': 23*['B1']+35*['B2'],
            tsStartPointColName: [ True, False, False, False, False,  True, False, False, False,
                    True, False, False, False, False, False, False,  True, False,
                   False, False, False, False, False,  True, False, False,  True,
                   False, False, False, False, False, False,  True, False, False,
                   False,  True, False, False,  True, False, False, False, False,
                    True, False, False, False, False, False,  True, False, False,
                   False, False, False, False],
            'y1': listRangesToList([range(32, 48), range(49, 56), range(58, 93)])},index=range(0, 58))

        testDfCheck = pd.DataFrame({
            'A': 52*['A1'],
            'B': 22*['B1']+30*['B2'],
            tsStartPointColName: [ True, False, False, False, False, False, False,  True,  True,
                   False, False, False, False,  True, False,  True, False, False,
                   False, False, False, False,  True, False, False, False, False,
                   False,  True, False, False, False, False, False,  True,  True,
                    True, False,  True, False, False, False, False, False, False,
                    True, False, False, False, False, False, False],
            'y1': listRangesToList([range(35, 80), range(84, 91)])},index=range(0, 52))

        self.equalDfs(trainDf,trainDfCheck)
        self.equalDfs(valDf,valDfCheck)
        self.equalDfs(testDf,testDfCheck)
# ---- run test
if __name__ == '__main__':
    unittest.main()