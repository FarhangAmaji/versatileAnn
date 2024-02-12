# ---- 

import random
import unittest

import numpy as np
import pandas as pd

from dataPrep.utils import splitTsTrainValTest_DfNNpDict, splitTrainValTest_mainGroup, \
    combineNSeries, \
    splitToNSeries
from dataPrep.utils_innerFuncs import _split_splitNShuffle_startPointIndexes
from tests.baseTest import BaseTestClass
from utils.dataTypeUtils.list import listRangesToList
from utils.globalVars import tsStartPointColName
from utils.generalUtils import morePreciseFloat


# ---- splitTsTrainValTest_DfNNpDict
class SplitTsTrainValTest_DfNNpDictTests(BaseTestClass):
    def setUp(self):
        np.random.seed(seed=30)
        self.backcastLen = 3
        self.forecastLen = 2
        self.df = pd.DataFrame({'y1': [i for i in range(10, 110)],
                                'y2': [i for i in range(110, 210)]}, index=range(20, 120))
        self.dfWithCond = pd.DataFrame({'y1': [i for i in range(10, 110)],
                                        'y2': [i for i in range(110, 210)],
                                        tsStartPointColName: [True if i > 40 and i < 106 else False \
                                                              for i in range(10, 110)],
                                        'condCol': [i for i in range(110, 210)]},
                                       index=range(20, 120))

    def testSeqLen0_NoShuffle(self):
        self.setUp()
        setDfs = splitTsTrainValTest_DfNNpDict(self.df, trainRatio=.7, valRatio=.2, seqLen=0,
                                               shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(10, 80)],
                                     'y2': [i for i in range(110, 180)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(80, 100)],
                                   'y2': [i for i in range(180, 200)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(100, 110)],
                                    'y2': [i for i in range(200, 210)]})
        trainDfCheck[tsStartPointColName] = True
        valDfCheck[tsStartPointColName] = True
        testDfCheck[tsStartPointColName] = True
        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck)
        self.equalDfs(testDf, testDfCheck)

    def testNoVal(self):
        self.setUp()
        setDfs = splitTsTrainValTest_DfNNpDict(self.df, trainRatio=.7, valRatio=0, seqLen=0,
                                               shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(10, 80)],
                                     'y2': [i for i in range(110, 180)]})
        valDfCheck = pd.DataFrame(columns=['y1', 'y2', '__startPoint__'])
        testDfCheck = pd.DataFrame({'y1': [i for i in range(80, 110)],
                                    'y2': [i for i in range(180, 210)]})

        trainDfCheck[tsStartPointColName] = True
        valDfCheck[tsStartPointColName] = True
        testDfCheck[tsStartPointColName] = True

        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck, floatApprox=True)
        self.equalDfs(testDf, testDfCheck)

    def testTsStartPointColNameCondition(self):
        self.setUp()
        setDfs = splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7,
                                               valRatio=.2, seqLen=0, shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

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
        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck)
        self.equalDfs(testDf, testDfCheck)

    def testWithSeqLen(self):
        self.setUp()
        setDfs = splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7,
                                               valRatio=.2,
                                               seqLen=self.backcastLen + self.forecastLen,
                                               shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(41, 90)],
                                     'y2': [i for i in range(141, 190)],
                                     tsStartPointColName: [True if i < 86 else False for i in
                                                           range(41, 90)],
                                     'condCol': [i for i in range(141, 190)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(86, 103)],
                                   'y2': [i for i in range(186, 203)],
                                   tsStartPointColName: [True if i < 99 else False for i in
                                                         range(86, 103)],
                                   'condCol': [i for i in range(186, 203)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(99, 110)],
                                    'y2': [i for i in range(199, 210)],
                                    tsStartPointColName: [True if i < 106 else False for i in
                                                          range(99, 110)],
                                    'condCol': [i for i in range(199, 210)]})

        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck)
        self.equalDfs(testDf, testDfCheck)

    def testOtherCondition(self):
        self.setUp()
        setDfs = splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7, valRatio=.2,
                                               seqLen=self.backcastLen + self.forecastLen,
                                               shuffle=False,
                                               conditions=['condCol>125', 'condCol<200'])
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({'y1': [i for i in range(26, 80)],
                                     'y2': [i for i in range(126, 180)],
                                     tsStartPointColName: [True if i < 76 else False for i in
                                                           range(26, 80)],
                                     'condCol': [i for i in range(126, 180)]})

        valDfCheck = pd.DataFrame({'y1': [i for i in range(76, 96)],
                                   'y2': [i for i in range(176, 196)],
                                   tsStartPointColName: [True if i < 92 else False for i in
                                                         range(76, 96)],
                                   'condCol': [i for i in range(176, 196)]})

        testDfCheck = pd.DataFrame({'y1': [i for i in range(92, 104)],
                                    'y2': [i for i in range(192, 204)],
                                    tsStartPointColName: [True if i < 100 else False for i in
                                                          range(92, 104)],
                                    'condCol': [i for i in range(192, 204)]})
        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck)
        self.equalDfs(testDf, testDfCheck)

    def testShuffle_WithSeqLen_WithOtherCondition(self):
        self.setUp()
        setDfs = splitTsTrainValTest_DfNNpDict(self.dfWithCond, trainRatio=.7,
                                               valRatio=.2,
                                               seqLen=self.backcastLen + self.forecastLen,
                                               shuffle=True, shuffleSeed=65,
                                               conditions=['condCol>125', 'condCol<200'])
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        getInd2s = lambda ind2s, inds: [ind2s.update([ti + i]) for i in
                                        range(self.backcastLen + self.forecastLen) for ti in inds]

        getDf = lambda ind2s, inds: pd.DataFrame({'y1': [i for i in ind2s],
                                                  'y2': [i + 100 for i in ind2s],
                                                  tsStartPointColName: [
                                                      False if i not in inds else True for i in
                                                      ind2s],
                                                  'condCol': [i + 100 for i in ind2s]})

        trainCheckIndexes = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                             45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                             81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                             99, 100, 101, 102]
        trainCheckIndexes_StartPoints = [27, 30, 31, 33, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 49,
                                         50, 51, 53, 54, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69,
                                         70, 71, 75, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 90,
                                         91, 93, 94, 96, 97, 98]
        trainDfCheck = getDf(trainCheckIndexes, trainCheckIndexes_StartPoints).reset_index(
            drop=True)
        self.equalDfs(trainDf, trainDfCheck)

        valCheckIndexes = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45,
                           52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70,
                           71, 72, 74, 75, 76, 77, 78, 79, 80, 89, 90, 91, 92, 93, 94, 95, 96, 99,
                           100, 101, 102, 103]
        valCheckIndexes_StartPoints = [26, 28, 32, 34, 41, 52, 55, 57, 63, 68, 74, 76, 89, 92, 99]
        valDfCheck = getDf(valCheckIndexes, valCheckIndexes_StartPoints).reset_index(drop=True)
        self.equalDfs(valDf, valDfCheck)

        testCheckIndexes = [29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 72,
                            73, 74, 75, 76, 77, 82, 83, 84, 85, 86, 95, 96, 97, 98, 99]
        testCheckIndexes_StartPoints = [29, 35, 37, 45, 72, 73, 82, 95]
        testDfCheck = getDf(testCheckIndexes, testCheckIndexes_StartPoints).reset_index(drop=True)
        self.equalDfs(testDf, testDfCheck)


# ---- splitTrainValTest_mainGroup
class TestSplitTrainValTest_NSeries(BaseTestClass):
    def setUp(self):
        np.random.seed(seed=30)
        self.df = pd.DataFrame({
            'A': 66 * ['A1'],
            'B': 27 * ['B1'] + 39 * ['B2'],
            tsStartPointColName: 21 * [True] + 6 * [False] + 33 * [True] + 6 * [False],
            'y1': list(range(30, 96)), }, index=range(100, 166))

    def testNoStartPointCol_0seqLen_noShuffle(self):
        df = self.df.drop(tsStartPointColName, axis=1)
        setDfs = splitTrainValTest_mainGroup(df, ["A", "B"], trainRatio=.6,
                                             valRatio=.2, seqLen=0, shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({
            'A': 39 * ['A1'],
            'B': 16 * ['B1'] + 23 * ['B2'],
            tsStartPointColName: 39 * [True],
            'y1': listRangesToList([range(30, 46), range(57, 80)])}, index=range(0, 39))

        valDfCheck = pd.DataFrame({
            'A': 13 * ['A1'],
            'B': 5 * ['B1'] + 8 * ['B2'],
            tsStartPointColName: 13 * [True],
            'y1': listRangesToList([range(46, 51), range(80, 88)])}, index=range(0, 13))

        testDfCheck = pd.DataFrame({
            'A': 14 * ['A1'],
            'B': 6 * ['B1'] + 8 * ['B2'],
            tsStartPointColName: 14 * [True],
            'y1': listRangesToList([range(51, 57), range(88, 96)])}, index=range(0, 14))

        self.equalDfs(trainDf, trainDfCheck, floatApprox=True)
        self.equalDfs(valDf, valDfCheck, floatApprox=True)
        self.equalDfs(testDf, testDfCheck, floatApprox=True)

    def testWithStartPointCol_withSeqLen_noShuffle(self):
        setDfs = splitTrainValTest_mainGroup(self.df, ["A", "B"], trainRatio=.6,
                                             valRatio=.2, seqLen=7, shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({
            'A': 43 * ['A1'],
            'B': 18 * ['B1'] + 25 * ['B2'],
            tsStartPointColName: 12 * [True] + 6 * [False] + 19 * [True] + 6 * [False],
            'y1': listRangesToList([range(30, 48), range(57, 82)])}, index=range(0, 43))

        valDfCheck = pd.DataFrame({
            'A': 23 * ['A1'],
            'B': 10 * ['B1'] + 13 * ['B2'],
            tsStartPointColName: [True, True, True, True, False, False, False, False, False,
                                  False, True, True, True, True, True, True, True, False,
                                  False, False, False, False, False],
            'y1': listRangesToList([range(42, 52), range(76, 89)])}, index=range(0, 23))

        testDfCheck = pd.DataFrame({
            'A': 24 * ['A1'],
            'B': 11 * ['B1'] + 13 * ['B2'],
            tsStartPointColName: [True, True, True, True, True, False, False, False, False,
                                  False, False, True, True, True, True, True, True, True,
                                  False, False, False, False, False, False],
            'y1': listRangesToList([range(46, 57), range(83, 96)])}, index=range(0, 24))

        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck)
        self.equalDfs(testDf, testDfCheck)

    def testWithDifferentSeqLen_noShuffle(self):
        # having a condition doesnt make sense in general use cases, with seqlen, but it may does in some cases
        df = self.df.drop(tsStartPointColName, axis=1)
        setDfs = splitTrainValTest_mainGroup(df, ["A", "B"], trainRatio=.6,
                                             valRatio=.2,
                                             trainSeqLen=8, valSeqLen=4, testSeqLen=3,
                                             shuffle=False)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({
            'A': 50 * ['A1'],
            'B': 22 * ['B1'] + 28 * ['B2'],
            tsStartPointColName: [True, True, True, True, True, True, True, True, True, True, True,
                                  True, True, True, True, False, False, False, False, False, False,
                                  False, True, True, True, True, True, True, True, True, True, True,
                                  True, True, True, True, True, True, True, True, True, True, True,
                                  False, False, False, False, False, False, False],
            'y1': listRangesToList([range(30, 52), range(57, 85)])}, index=range(0, 50))

        valDfCheck = pd.DataFrame({
            'A': 18 * ['A1'],
            'B': 7 * ['B1'] + 11 * ['B2'],
            tsStartPointColName: [True, True, True, True, False, False, False, True, True, True,
                                  True, True, True, True, True, False, False, False],
            'y1': listRangesToList([range(45, 52), range(78, 89)])}, index=range(0, 18))

        testDfCheck = pd.DataFrame({
            'A': 18 * ['A1'],
            'B': 8 * ['B1'] + 10 * ['B2'],
            tsStartPointColName: [True, True, True, True, True, True, False, False, True, True,
                                  True, True, True, True, True, True, False, False],
            'y1': listRangesToList([range(49, 57), range(86, 96)])}, index=range(0, 18))

        self.equalDfs(trainDf, trainDfCheck, floatApprox=True)
        self.equalDfs(valDf, valDfCheck, floatApprox=True)
        self.equalDfs(testDf, testDfCheck, floatApprox=True)

    def testWithStartPointCol_withSeqLen_Shuffle(self):
        setDfs = splitTrainValTest_mainGroup(self.df, ["A", "B"], trainRatio=.6,
                                             valRatio=.2, seqLen=7, shuffle=True)
        trainDf, valDf, testDf = setDfs['train'], setDfs['val'], setDfs['test']

        trainDfCheck = pd.DataFrame({
            'A': 60 * ['A1'],
            'B': 24 * ['B1'] + 36 * ['B2'],
            tsStartPointColName: [True, False, False, False, True, False, True, True, True, True,
                                  True, True, False, True, True, False, True, True, False, False,
                                  False, False, False, False, True, True, False, True, True, True,
                                  False, True, True, False, False, True, False, True, False, True,
                                  False, True, True, False, True, False, True, True, True, False,
                                  False, True, True, True, False, False, False, False, False,
                                  False],
            'y1': listRangesToList([range(30, 54), range(59, 95)])}, index=range(0, 60))

        valDfCheck = pd.DataFrame({
            'A': 51 * ['A1'],
            'B': 22 * ['B1'] + 29 * ['B2'],
            tsStartPointColName: [True, False, False, False, False, False, False, True, False,
                                  False, False, False, False, False, True, True, False, False,
                                  False, False, False, False, True, False, False, True, False,
                                  False, False, True, False, False, True, False, False, True, False,
                                  True, False, False, False, False, False, False, True, False,
                                  False, False, False, False, False],
            'y1': listRangesToList([range(33, 40), range(42, 57), range(58, 80), range(89, 96)])},
            index=range(0, 51))

        testDfCheck = pd.DataFrame({
            'A': 51 * ['A1'],
            'B': 21 * ['B1'] + 30 * ['B2'],
            tsStartPointColName: [True, True, False, False, True, False, False, False, False, False,
                                  False, True, False, False, True, False, False, False, False,
                                  False, False, True, False, False, False, False, False, False,
                                  True, False, False, False, False, False, True, False, False, True,
                                  False, True, False, False, False, True, True, False, False, False,
                                  False, False, False],
            'y1': listRangesToList([range(31, 42), range(45, 55), range(57, 64), range(69, 92)])},
            index=range(0, 51))

        self.equalDfs(trainDf, trainDfCheck)
        self.equalDfs(valDf, valDfCheck)
        self.equalDfs(testDf, testDfCheck)


class _split_splitNShuffleIndexes_Tests(BaseTestClass):
    # cccDevAlgo
    #  - this code is really complex; and its obvious some case are not at all possible(harsher conditions)
    #       than allConditions here; note these are in common in practical use but just for
    #       the case of hypothetical perfection
    #  - I guess worst case scenarios are the ones with:
    #   - test ratio is the highest
    #   - one of the sets has 0 or low seqLen and other 2 have high seqLens(specially)
    #       because _split_indexesNotInvolved wont remove elementsNotToInvolved_inAll3Sets from indexes
    #       therefore more indexes with
    #   - and probably with "result deduction of `1-2` if goes to an index below 0" and in less critical
    #       1. supposed starting point of set if there was no seqLens
    #       2. seqLen of set
    #   - similar to last point but in less critical("doesnt go to an index below 0") but creates
    #       a deficit which sum of these deficits makes problem

    def _setupSetUniqueVals_WithoutMaxN0(self, ratios):
        uniqueVals = list(set(ratios.values()))
        uniqueVals_WithoutMaxN0 = uniqueVals[:]
        maxVal = max(uniqueVals)
        uniqueVals_WithoutMaxN0.remove(maxVal)
        if 0 in uniqueVals:
            uniqueVals_WithoutMaxN0.remove(0)
        return uniqueVals_WithoutMaxN0

    def _setupSetInLoop(self, seqLens, sn, ar, possiblePlusForSet, configsForAr):
        for ppfs in possiblePlusForSet:
            seqLens2 = seqLens.copy()
            # 100 is len of df, hardcoded here
            # cccDevAlgo
            #  note seqLens are not necessarily less than len required('morePreciseFloat(ar[sn])')
            #  and they can be even more
            if ar[sn] != 0 and morePreciseFloat(ar[sn]) * 100 > ppfs:
                seqLens2[sn] = int(ppfs)
            if sn == 'train':
                self._setupSetInLoop(seqLens2, 'val', ar, possiblePlusForSet, configsForAr)
            elif sn == 'val':
                self._setupSetInLoop(seqLens2, 'test', ar, possiblePlusForSet, configsForAr)
            elif sn == 'test':
                configsForAr.append(seqLens2)

    def _setupGetAllRatios(self):
        return [
            # all equal
            {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
            # train is the biggest
            {"train": 0.6, "val": 0.3, "test": 0.1},  # val > test
            {"train": 0.6, "val": 0.1, "test": 0.3},  # val < test
            {"train": 0.6, "val": 0.2, "test": 0.2},  # val = test
            # train is the smallest and val = test
            {"train": 0.2, "val": 0.3, "test": 0.3},

            # val is the biggest
            {"val": 0.6, "train": 0.3, "test": 0.1},  # train > test
            {"val": 0.6, "train": 0.1, "test": 0.3},  # train < test
            {"val": 0.6, "train": 0.2, "test": 0.2},  # train = test
            # val is the smallest and train = test
            {"val": 0.2, "train": 0.3, "test": 0.3},

            # test is the biggest
            {"test": 0.6, "train": 0.3, "val": 0.1},  # train > val
            {"test": 0.6, "train": 0.1, "val": 0.3},  # train < val
            {"test": 0.6, "train": 0.2, "val": 0.2},  # train = val
            # test is the smallest and train = val
            {"test": 0.2, "train": 0.3, "val": 0.3}, ]

    def _setupGet_possiblePlusForSet(self, ratios_):
        uniqueVals_WithoutMaxN0 = self._setupSetUniqueVals_WithoutMaxN0(ratios_)
        if not uniqueVals_WithoutMaxN0:
            return []

        # 100 is len of df, hardcoded here
        possiblePlusForSet = [int(morePreciseFloat(urv) * 100) for urv in uniqueVals_WithoutMaxN0]
        # adding middle points for between
        if len(possiblePlusForSet) > 1:
            lastPointBefore_possiblePlusForSet = possiblePlusForSet[-1]
            for ppfsI, ppfs in enumerate(possiblePlusForSet):
                if ppfs == lastPointBefore_possiblePlusForSet:
                    break
                middlePointToAdded = int(
                    (possiblePlusForSet[ppfsI] + possiblePlusForSet[ppfsI + 1]) / 2)
                possiblePlusForSet.append(middlePointToAdded)
        possiblePlusForSet.sort()
        possiblePlusForSet.append(possiblePlusForSet[-1] + 5)
        possiblePlusForSet.append(possiblePlusForSet[0] - 5)
        possiblePlusForSet = list(set(possiblePlusForSet))
        return possiblePlusForSet

    def _setupGetAllConfigs(self):
        # goodToHave3
        #  assume case {'ratios': {'test': 0.85, 'train': 0.05, 'val': 0.1}, 'seqLens': {'train': 15, 'val': 15, 'test': 25}}
        allConfigs = []
        allRatios = self._setupGetAllRatios()
        for ar in allRatios:
            possiblePlusForSet = self._setupGet_possiblePlusForSet(ar)

            if not possiblePlusForSet:
                continue

            configsForAr = []
            seqLens = {'train': 0, 'val': 0, 'test': 0}
            self._setupSetInLoop(seqLens, 'train', ar, possiblePlusForSet, configsForAr)

            configsForAr = pd.DataFrame(configsForAr)
            configsForAr = configsForAr.drop_duplicates()
            configsForAr = configsForAr.to_dict(orient='records')
            configsForSet_withRatios = []
            for cfs in configsForAr:
                configsForSet_withRatios.append({'ratios': ar, 'seqLens': cfs})
            allConfigs.extend(configsForSet_withRatios)

        return allConfigs

    def setUp_NonSequentIndexes(self) -> None:
        self.setNames = ['train', 'val', 'test']
        self.shuffleSeed = 65
        random.seed(self.shuffleSeed)
        # these are not sequent data like 57,58,59 and may be 57,61,63
        randomIndexes = random.sample(range(201), 100)
        self.df = pd.DataFrame({'a': [i for i in range(100)]}, index=randomIndexes)

    def runAllconfigs(self, shuffle):
        allConfigs = self._setupGetAllConfigs()
        self.setUp_NonSequentIndexes()
        if shuffle:
            shuffleSeed = self.shuffleSeed
        else:
            shuffleSeed = None

        for ac in allConfigs:
            self.setUp_NonSequentIndexes()
            if ac['ratios'] == {'train': 0.6, 'val': 0.3, 'test': 0.1} and ac['seqLens'] == {
                'train': 5, 'val': 20, 'test': 0} and shuffle == False:
                x = 0

            try:
                _split_splitNShuffle_startPointIndexes(self.df, False, ac['ratios'], ac['seqLens'],
                                                       shuffle=shuffle, shuffleSeed=shuffleSeed,
                                                       setNames=self.setNames)
            except:
                raise RuntimeError(f"{ac['ratios']=}, {ac['seqLens']=}, {shuffle=}, {shuffleSeed=}")

    def testAllconfigs_Shuffle(self):
        self.runAllconfigs(shuffle=True)

    def testAllconfigs_noShuffle(self):
        self.runAllconfigs(shuffle=False)

    # ---- assert configs with shuffle
    # addTest1
    #  because there are 312 configs possible and there are so many cases not included in allConfigs
    #  thus the algorithm seems a bit complex, therefore is hard
    #  to understand which configs are really should be asserted in order to provide immunity
    #  to change in the code; anyway here are some good candidates
    def testConfig_NotFromAllConfigs1_shuffle(self):
        # bugPotentialCheck1
        #  super important: what does this config has which my allConfigs does not
        #  note none of testAllconfigs with Shuffle on or off didnt return setsIndexes from
        #  'if not len(indexes)' just after '_assignAsMuchAs_IdxsWith1PossibleSet'
        #  note its not a imminent hazard but the fact the allConfigs has not included
        #  a config type to go out from mentioned point
        # this config is not in allConfigs
        config = {'ratios': {'train': .7, 'val': .2, 'test': .1},
                  'seqLens': {'train': 37, 'val': 16, 'test': 5}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'], shuffle=True,
                                                             shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41, 44, 46,
                      47, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 69, 72, 73, 75, 76, 77,
                      81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94, 96, 100, 101, 106, 107, 109, 110,
                      111, 113, 115, 117, 118, 119, 121],
            'val': [123, 127, 131, 132, 135, 136, 137, 138, 140, 141, 142, 143, 146, 147, 149, 151,
                    152, 158, 159, 162, 163],
            'test': [167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig18shuffle(self):  # == allConfigs[18]
        config = {'ratios': {'test': 0.1, 'train': 0.6, 'val': 0.3},
                  'seqLens': {'test': 0, 'train': 5, 'val': 20}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=True, shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [158, 162, 178, 163, 170, 187, 172, 168, 189, 186, 180, 159, 182, 177, 167, 94,
                      36, 83, 131, 81, 127, 60, 107, 57, 137, 113, 146, 91, 76, 35, 141, 59, 66, 41,
                      143, 85, 142, 12, 33, 34, 132, 111, 140, 101, 117, 67, 55, 75, 109, 136, 86,
                      93, 69, 39, 123, 46, 119, 28, 16, 82],
            'val': [73, 100, 147, 149, 118, 1, 152, 96, 49, 72, 11, 115, 89, 40, 47, 44, 106, 5, 18,
                    77, 50, 64, 65, 10, 135, 56, 31, 151, 110, 121],
            'test': [191, 193, 195, 199, 20, 84, 58, 2, 138, 90]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig83shuffle(self):
        config = {'ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2},
                  'seqLens': {'train': 25, 'val': 15, 'test': 15}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=True, shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [96, 81, 115, 89, 60, 118, 57, 113, 107, 131, 109, 36, 106, 35, 143, 136, 83,
                      41, 137, 101, 73, 141, 1, 85, 94, 12, 135, 33, 49, 34, 72, 11, 91, 127, 40,
                      111, 47, 67, 44, 75, 140, 100, 66, 50, 132, 10, 46, 119, 58, 16, 82],
            'val': [151, 167, 159, 163, 152, 5, 77, 117, 93, 64, 65, 123, 56, 142, 28, 110, 121],
            'test': [146, 158, 149, 162, 147, 55, 18, 86, 59, 76, 69, 39, 20, 84, 31, 2, 138, 90]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig147shuffle(self):
        config = {'ratios': {'val': 0.6, 'train': 0.1, 'test': 0.3},
                  'seqLens': {'train': 0, 'val': 35, 'test': 20}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=True, shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [158, 159, 162, 163, 167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189,
                      191, 193, 195, 199],
            'val': [73, 119, 35, 60, 93, 100, 59, 55, 91, 111, 44, 85, 75, 41, 47, 57, 36, 107, 109,
                    81, 5, 118, 66, 127, 34, 83, 67, 40, 12, 50, 115, 117, 106, 65, 101, 10, 72, 1,
                    33, 11, 20, 46, 56, 84, 89, 96, 31, 58, 28, 69, 123, 77, 64, 18, 82, 90],
            'test': [131, 132, 135, 136, 137, 138, 140, 141, 142, 143, 146, 147, 149, 151, 152, 113,
                     86, 94, 39, 76, 2, 16, 110, 49, 121]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig166shuffle(self):
        config = {'ratios': {'val': 0.6, 'train': 0.1, 'test': 0.3},
                  'seqLens': {'train': 5, 'val': 35, 'test': 10}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=True, shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {'train': [180, 182, 186, 187, 189, 56, 31, 2, 82],
                                'val': [73, 106, 83, 117, 113, 111, 89, 36, 115, 35, 57, 85, 91, 41,
                                        118, 60, 1, 101, 81, 109, 12, 33, 49, 34, 72, 11, 94, 107,
                                        40, 96, 47, 67, 44, 55, 75, 5, 18, 77, 86, 100, 59, 66, 93,
                                        127, 76, 50, 64, 69, 65, 39, 10, 123, 20, 46, 84, 58, 16,
                                        121],
                                'test': [170, 159, 141, 137, 177, 149, 140, 146, 162, 178, 168, 142,
                                         158, 132, 172, 136, 143, 131, 135, 147, 152, 163, 151, 138,
                                         167, 119, 28, 110, 90]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig233shuffle(self):
        config = {'ratios': {'test': 0.6, 'train': 0.3, 'val': 0.1},
                  'seqLens': {'train': 10, 'val': 5, 'test': 35}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=True, shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [170, 159, 141, 137, 177, 149, 140, 146, 162, 178, 168, 142, 158, 132, 172,
                      136, 143, 131, 135, 147, 152, 163, 151, 138, 56, 31, 2, 82],
            'val': [180, 182, 186, 187, 189, 167, 84, 58, 16, 121],
            'test': [73, 106, 83, 117, 113, 111, 89, 36, 115, 35, 57, 85, 91, 41, 118, 60, 1, 101,
                     81, 109, 12, 33, 49, 34, 72, 11, 94, 107, 40, 96, 47, 67, 44, 55, 75, 5, 18,
                     77, 86, 100, 59, 66, 93, 127, 76, 50, 64, 69, 65, 39, 10, 123, 20, 46, 119, 28,
                     110, 90]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig287shuffle(self):
        config = {'ratios': {'test': 0.6, 'train': 0.1, 'val': 0.3},
                  'seqLens': {'train': 5, 'val': 20, 'test': 30}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=True, shuffleSeed=self.shuffleSeed,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [158, 159, 162, 163, 167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189],
            'val': [138, 140, 141, 142, 143, 146, 147, 149, 151, 152, 76, 64, 118, 39, 72, 33, 20,
                    56, 91, 119, 109, 86, 94, 2, 110, 121],
            'test': [117, 60, 127, 132, 115, 136, 106, 137, 12, 101, 77, 113, 73, 123, 41, 34, 83,
                     81, 89, 35, 40, 67, 47, 36, 93, 57, 44, 55, 75, 100, 49, 96, 5, 18, 107, 135,
                     59, 66, 85, 50, 69, 65, 10, 1, 11, 46, 84, 111, 131, 31, 58, 28, 16, 82, 90]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    # ---- assert configs with no shuffle
    def testConfig_NotFromAllConfigs1_noShuffle(self):
        # this config is not in allConfigs
        config = {'ratios': {'train': .7, 'val': .2, 'test': .1},
                  'seqLens': {'train': 37, 'val': 16, 'test': 5}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'], shuffle=False,
                                                             shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41, 44, 46,
                      47, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 69, 72, 73, 75, 76, 77,
                      81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94, 96, 100, 101, 106, 107, 109, 110,
                      111, 113, 115, 117, 118, 119, 121],
            'val': [123, 127, 131, 132, 135, 136, 137, 138, 140, 141, 142, 143, 146, 147, 149, 151,
                    152, 158, 159, 162, 163],
            'test': [167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig18noShuffle(self):  # == allConfigs[18]
        config = {'ratios': {'test': 0.1, 'train': 0.6, 'val': 0.3},
                  'seqLens': {'test': 0, 'train': 5, 'val': 20}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=False, shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41, 44, 46,
                      47, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 69, 72, 73, 75, 76, 77,
                      81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94, 96, 100, 101, 106, 107, 109, 110,
                      111, 113, 115],
            'val': [117, 118, 119, 121, 123, 127, 131, 132, 135, 136, 137, 138, 140, 141, 142, 143,
                    146, 147, 149, 151, 152],
            'test': [158, 159, 162, 163, 167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189, 191,
                     193, 195, 199]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig83noShuffle(self):
        config = {'ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2},
                  'seqLens': {'train': 25, 'val': 15, 'test': 15}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=False, shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41, 44, 46,
                      47, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 69, 72, 73, 75, 76, 77,
                      81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94],
            'val': [96, 100, 101, 106, 107, 109, 110, 111, 113, 115, 117, 118, 119, 121, 123, 127,
                    131, 132],
            'test': [135, 136, 137, 138, 140, 141, 142, 143, 146, 147, 149, 151, 152, 158, 159, 162,
                     163, 167]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig147noShuffle(self):
        config = {'ratios': {'val': 0.6, 'train': 0.1, 'test': 0.3},
                  'seqLens': {'train': 0, 'val': 35, 'test': 20}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=False, shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [158, 159, 162, 163, 167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189,
                      191, 193, 195, 199],
            'val': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41, 44, 46, 47,
                    49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 69, 72, 73, 75, 76, 77, 81, 82,
                    83, 84, 85, 86, 89, 90, 91, 93, 94, 96, 100],
            'test': [101, 106, 107, 109, 110, 111, 113, 115, 117, 118, 119, 121, 123, 127, 131, 132,
                     135, 136, 137, 138, 140, 141, 142, 143, 146, 147, 149, 151, 152]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig166noShuffle(self):
        config = {'ratios': {'val': 0.6, 'train': 0.1, 'test': 0.3},
                  'seqLens': {'train': 5, 'val': 35, 'test': 10}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=False, shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {'train': [1, 2, 5, 180, 182, 186, 187, 189],
                                'val': [10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41,
                                        44, 46, 47, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67,
                                        69, 72, 73, 75, 76, 77, 81, 82, 83, 84, 85, 86, 89, 90, 91,
                                        93, 94, 96, 100, 101, 106, 107, 109, 110, 111, 113, 115,
                                        117, 118],
                                'test': [119, 121, 123, 127, 131, 132, 135, 136, 137, 138, 140, 141,
                                         142, 143, 146, 147, 149, 151, 152, 158, 159, 162, 163, 167,
                                         168, 170, 172, 177, 178]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig233noShuffle(self):
        config = {'ratios': {'test': 0.6, 'train': 0.3, 'val': 0.1},
                  'seqLens': {'train': 10, 'val': 5, 'test': 35}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=False, shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 41, 44, 46,
                      47, 49, 50, 55, 56, 57, 58, 131, 132, 135, 136, 137, 138, 140, 141, 142, 143,
                      146, 147, 149, 151, 152, 158, 159, 162, 163, 167, 168, 170, 172, 177, 178],
            'val': [59, 60, 64, 65, 66, 180, 182, 186, 187, 189],
            'test': [67, 69, 72, 73, 75, 76, 77, 81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94, 96,
                     100, 101, 106, 107, 109, 110, 111, 113, 115, 117, 118, 119, 121, 123, 127]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    def testConfig287noShuffle(self):
        config = {'ratios': {'test': 0.6, 'train': 0.1, 'val': 0.3},
                  'seqLens': {'train': 5, 'val': 20, 'test': 30}}
        self.setUp_NonSequentIndexes()
        setsIndexes = _split_splitNShuffle_startPointIndexes(self.df, False, config['ratios'],
                                                             config['seqLens'],
                                                             shuffle=False, shuffleSeed=None,
                                                             setNames=self.setNames)
        expected_setsIndexes = {
            'train': [158, 159, 162, 163, 167, 168, 170, 172, 177, 178, 180, 182, 186, 187, 189],
            'val': [1, 2, 5, 10, 11, 12, 16, 18, 20, 28, 31, 33, 34, 35, 36, 39, 40, 138, 140, 141,
                    142, 143, 146, 147, 149, 151, 152],
            'test': [41, 44, 46, 47, 49, 50, 55, 56, 57, 58, 59, 60, 64, 65, 66, 67, 69, 72, 73, 75,
                     76, 77, 81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94, 96, 100, 101, 106, 107,
                     109, 110, 111, 113, 115, 117, 118, 119, 121, 123, 127, 131, 132, 135, 136,
                     137]}
        self.assertEquals(setsIndexes, expected_setsIndexes)

    # ---- some other configs
    def testConfig_someOtherConfigs_shuffleNNoShuffle(self):
        self.setUp_NonSequentIndexes()
        # these configs are not in allConfigs
        someOtherConfigs = [{'ratios': {'train': .7, 'val': .2, 'test': .1},
                             'seqLens': {'train': 37, 'val': 16, 'test': 5}},
                            {'ratios': {'train': .7, 'val': .2, 'test': .1},
                             'seqLens': {'train': 37, 'val': 5, 'test': 16}},
                            {'ratios': {'train': .7, 'val': .2, 'test': .1},
                             'seqLens': {'train': 16, 'val': 37, 'test': 5}},
                            {'ratios': {'train': .7, 'val': .2, 'test': .1},
                             'seqLens': {'train': 16, 'val': 5, 'test': 37}},
                            {'ratios': {'train': .7, 'val': .2, 'test': .1},
                             'seqLens': {'train': 5, 'val': 37, 'test': 16}},
                            {'ratios': {'train': .7, 'val': .2, 'test': .1},
                             'seqLens': {'train': 5, 'val': 16, 'test': 37}}, ]
        for soc in someOtherConfigs:
            # shuffle
            self.setUp_NonSequentIndexes()
            # if soc['ratios'] == {'train': 0.7, 'val': 0.2, 'test': 0.1} and soc['seqLens'] == {
            #     'train': 37, 'val': 5, 'test': 16}:
            #     x=0
            _split_splitNShuffle_startPointIndexes(self.df, False, soc['ratios'], soc['seqLens'],
                                                   shuffle=True, shuffleSeed=self.shuffleSeed,
                                                   setNames=self.setNames)
            # noShuffle
            self.setUp_NonSequentIndexes()
            _split_splitNShuffle_startPointIndexes(self.df, False, soc['ratios'], soc['seqLens'],
                                                   shuffle=False, shuffleSeed=None,
                                                   setNames=self.setNames)


# ---- SplitNCombineNSeries
class TestSplitNCombineNSeries(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({
            'dew': [20, 31, 18, 37, 26],
            'Temperature_A': [20, 21, 22, 23, 24],
            'Temperature_B': [30, 31, 32, 33, 34],
            'Pressure_A': [1000, 1001, 1002, 1003, 1004],
            'Pressure_B': [900, 901, 902, 903, 904]})

        self.splittedDf = pd.DataFrame({
            'dew': [20, 31, 18, 37, 26, 20, 31, 18, 37, 26,
                    20, 31, 18, 37, 26, 20, 31, 18, 37, 26],
            'Temperature': [20, 21, 22, 23, 24, 30, 31, 32, 33, 34,
                            20, 21, 22, 23, 24, 30, 31, 32, 33, 34],
            'TemperatureType': ['Temperature_A', 'Temperature_A', 'Temperature_A', 'Temperature_A',
                                'Temperature_A',
                                'Temperature_B', 'Temperature_B', 'Temperature_B', 'Temperature_B',
                                'Temperature_B',
                                'Temperature_A', 'Temperature_A', 'Temperature_A', 'Temperature_A',
                                'Temperature_A',
                                'Temperature_B', 'Temperature_B', 'Temperature_B', 'Temperature_B',
                                'Temperature_B'],
            'Pressure': [1000, 1001, 1002, 1003, 1004, 1000, 1001, 1002, 1003, 1004,
                         900, 901, 902, 903, 904, 900, 901, 902, 903, 904],
            'PressureType': ['Pressure_A', 'Pressure_A', 'Pressure_A', 'Pressure_A', 'Pressure_A',
                             'Pressure_A', 'Pressure_A', 'Pressure_A', 'Pressure_A', 'Pressure_A',
                             'Pressure_B', 'Pressure_B', 'Pressure_B', 'Pressure_B', 'Pressure_B',
                             'Pressure_B', 'Pressure_B', 'Pressure_B', 'Pressure_B', 'Pressure_B']})

    def testSplit(self):
        splitData = splitToNSeries(self.df, ['Temperature_A', 'Temperature_B'], 'Temperature')
        splitData = splitToNSeries(splitData, ['Pressure_A', 'Pressure_B'], 'Pressure')

        self.equalDfs(splitData, self.splittedDf)
        return splitData

    def testCombine(self):
        splitData = self.testSplit()
        comb1 = combineNSeries(splitData, 'Temperature')
        comb2 = combineNSeries(comb1, 'Pressure')

        self.equalDfs(comb2, self.df, floatApprox=True)


# ---- run test
if __name__ == '__main__':
    unittest.main()
