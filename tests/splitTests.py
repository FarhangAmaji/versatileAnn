#%% 
from tests.baseTest import BaseTestClass
import unittest
from dataPrep.utils import splitTsTrainValTestDfNNpDict
from utils.vAnnGeneralUtils import equalDfs
from utils.globalVars import tsStartPointColName
import pandas as pd
import numpy as np
#%% SplitTests
class SplitTests(BaseTestClass):
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
        trainDf, valDf, testDf=splitTsTrainValTestDfNNpDict(self.df, trainRatio=.7, valRatio=.2, seqLen=0, shuffle=False)

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(10, 80)],
                                     'y2': [i for i in range(110, 180)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(80, 99)],
                                   'y2': [i for i in range(180, 199)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(99, 110)],
                                    'y2': [i for i in range(199, 210)]})
        trainDfCheck[tsStartPointColName]=True
        valDfCheck[tsStartPointColName]=True
        testDfCheck[tsStartPointColName]=True
        self.assertTrue(equalDfs(trainDf,trainDfCheck))
        self.assertTrue(equalDfs(valDf,valDfCheck))
        self.assertTrue(equalDfs(testDf,testDfCheck))

    def testNoVal(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTestDfNNpDict(self.df, trainRatio=.7, valRatio=0, seqLen=0, shuffle=False)

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(10, 80)],
                                     'y2': [i for i in range(110, 180)]})
        valDfCheck = pd.DataFrame(columns=['y1', 'y2', '__startPoint__'])
        testDfCheck = pd.DataFrame({'y1': [i for i in range(80, 110)],
                                    'y2': [i for i in range(180, 210)]})

        trainDfCheck[tsStartPointColName]=True
        valDfCheck[tsStartPointColName]=True
        testDfCheck[tsStartPointColName]=True

        self.assertTrue(equalDfs(trainDf,trainDfCheck))
        self.assertTrue(equalDfs(valDf,valDfCheck))
        self.assertTrue(equalDfs(testDf,testDfCheck))

    def testTsStartPointColNameCondition(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTestDfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2, seqLen=0, shuffle=False)

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
        self.assertTrue(equalDfs(trainDf,trainDfCheck))
        self.assertTrue(equalDfs(valDf,valDfCheck))
        self.assertTrue(equalDfs(testDf,testDfCheck))

    def testWithSeqLen(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTestDfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
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

        self.assertTrue(equalDfs(trainDf,trainDfCheck))
        self.assertTrue(equalDfs(valDf,valDfCheck))
        self.assertTrue(equalDfs(testDf,testDfCheck))

    def testOtherCondition(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTestDfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
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
        self.assertTrue(equalDfs(trainDf,trainDfCheck))
        self.assertTrue(equalDfs(valDf,valDfCheck))
        self.assertTrue(equalDfs(testDf,testDfCheck))

    def testShuffle_WithSeqLen_WithOtherCondition(self):
        self.setUp()
        trainDf, valDf, testDf=splitTsTrainValTestDfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
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
        self.assertTrue(equalDfs(trainDf,trainDfCheck))


        valCheckIndexes=[27, 29, 33, 43, 44, 61, 64, 67, 72, 75, 78, 81, 86, 92, 96]
        valCheckIndexes2=set(valCheckIndexes[:])
        getInd2s(valCheckIndexes2, valCheckIndexes)
        valDfCheck = getDf(valCheckIndexes2, valCheckIndexes)
        self.assertTrue(equalDfs(valDf,valDfCheck))


        testCheckIndexes=[28, 38, 49, 63, 71, 79, 97, 99]
        testCheckIndexes2=set(testCheckIndexes[:])
        getInd2s(testCheckIndexes2, testCheckIndexes)
        testDfCheck = getDf(testCheckIndexes2, testCheckIndexes)
        self.assertTrue(equalDfs(testDf,testDfCheck))
#%% run test
if __name__ == '__main__':
    unittest.main()