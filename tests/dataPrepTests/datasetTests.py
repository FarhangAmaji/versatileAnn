# ----
"#ccc no tests for getBackForeCastData because it only uses _IdxNdataToLook_WhileFetching and assertIdxInIndexes_dependingOnAllowance,"
"... also getBackForeCastData_general which has it own tests on tsRowFetcherTests"

"#ccc note the test for getBackForeCastData_general may not contain the type of df with useNpDictForDfs,"
"... as the indexes and dataToLook matches the npDict type results"
# ----

import unittest

from tests.baseTest import BaseTestClass
from utils.dataTypeUtils.dotDict_npDict import NpDict

from dataPrep.dataset import VAnnTsDataset
from utils.globalVars import tsStartPointColName
import pandas as pd
import numpy as np


# ---- dataset tests
# ----         VAnnTsDataset_setIndexesTests
# ccc this test also does some of check related to VAnnTsDataset_indexesSettingTests; no problem
# kkk its seems to be enough tests, be more tests are always appriciated
class VAnnTsDataset_setIndexesTests(BaseTestClass):
    # this is related to _setIndexes
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3, 4],
                                'B': [5, 6, 7, 8],
                                '__startPoint__': [False, True, True, False]},
                               index=[8, 9, 10, 11])

    # ---- 0lens
    def testNpArrayWithIndexes_No0lens(self):
        self.setUp()
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indexes = [0, 2]  # Include only rows at index 0 and 2
        dataset = VAnnTsDataset(npArray, backcastLen=1, forecastLen=1, indexes=indexes)
        self.assertEqual(len(dataset), 2)  # Only specified indexes should be included
        self.assertEqual(dataset.indexes, indexes)  # Only specified indexes should be included

    def testDf_noUseNpDictForDfs_NoIndexesPassed_NoStartPoint_0lens(self):
        self.setUp()
        self.df = self.df.drop('__startPoint__', axis=1)
        dataset = VAnnTsDataset(self.df, backcastLen=0, forecastLen=0, useNpDictForDfs=False)
        self.assertEqual(dataset.indexes, [0, 1, 2, 3])
        self.assertEqual(len(dataset), 4)  # All rows should be included

    def testDf_NoIndexesPassed_WithStartPoint_0lens(self):  # not needed
        self.setUp()
        dataset = VAnnTsDataset(self.df, backcastLen=0, forecastLen=0, useNpDictForDfs=False)
        self.assertTrue(list(dataset.indexes) == [9, 10])
        self.assertEqual(len(dataset),
                         2)  # Only rows with '__startPoint__' as True should be included

    def testNpArray_NoIndexesPassed_0lens(self):
        self.setUp()
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = VAnnTsDataset(npArray, backcastLen=0, forecastLen=0)
        self.assertEqual(dataset.indexes, [0, 1, 2])
        self.assertEqual(len(dataset), 3)

    # ---- no0lens NoStartPoint assertion
    def testDf_useNpDictForDfs_NoIndexesPassed_NoStartPoint_No0lens(self):
        self.setUp()
        npDict = self.df.drop('__startPoint__', axis=1)
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(npDict, backcastLen=1, forecastLen=1, useNpDictForDfs=True)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    def testDf_noUseNpDictForDfs_NoIndexesPassed_NoStartPoint_No0lens(self):
        self.setUp()
        self.df = self.df.drop('__startPoint__', axis=1)
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=False)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    def testNpDict_NoIndexesPassed_NoStartPoint_No0lens(self):
        self.setUp()
        npDict = NpDict(self.df.drop('__startPoint__', axis=1))
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(npDict, backcastLen=1, forecastLen=1)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    def testNpArray_NoIndexesPassed_No0lens(self):
        self.setUp()
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(npArray, backcastLen=1, forecastLen=1)
        self.assertEqual(str(context.exception), VAnnTsDataset.noIndexesAssertionMsg)

    # ---- mainTypes No0lens NoIndexesPassed WithStartPoint
    def testDf_useNpDictForDfs_NoIndexesPassed_WithStartPoint_No0lens(self):
        self.setUp()
        dataset = VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=True)
        self.assertTrue(list(dataset.indexes) == [1, 2])  # ccc note indexes has kept their values
        self.assertEqual(len(dataset),
                         2)  # Only rows with '__startPoint__' as True should be included

    def testDf_noUseNpDictForDfs_NoIndexesPassed_WithStartPoint_No0lens(self):
        self.setUp()
        dataset = VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=False)
        self.assertTrue(list(dataset.indexes) == [9, 10])  # ccc note indexes has kept their values
        self.assertEqual(len(dataset),
                         2)  # Only rows with '__startPoint__' as True should be included

    def testNpDict_NoIndexesPassed_WithStartPoint_No0lens(self):
        self.setUp()
        npDict = NpDict(self.df)
        dataset = VAnnTsDataset(npDict, backcastLen=1, forecastLen=1)
        self.assertTrue(list(dataset.indexes) == [1, 2])
        self.assertEqual(len(dataset), 2)
        # Only rows with '__startPoint__' as True should be included


# ----         VAnnTsDataset_indexesSetting_noNSeriesTests
class VAnnTsDataset_indexesSetting_noNSeriesTests(BaseTestClass):
    # this is related to _setIndexes and _assignData_NMainGroupsIdxs
    # ccc test assumes, noIndexes are passed to dataset init and there is some
    # value for backcast and forecast lens
    def setUp(self):
        self.kwargs = {'backcastLen': 3, 'forecastLen': 2}
        self.df = pd.DataFrame({'A': list(range(10)),
                                '__startPoint__': [True, False, True, True, False, True, False,
                                                   True, False, True]},
                               index=list(range(110, 120)))
        self.TruesIndexFromBeginning = [i for i, val in enumerate(self.df['__startPoint__']) if
                                        val == True]

    def assertMainGroupsIdxEmpty(self, dataset):
        self.assertDictEqual(dataset.mainGroupsGeneralIdxs, {})
        self.assertDictEqual(dataset.mainGroupsRelIdxs, {})

    def testDf_useNpDictForDfs_StartPointsInCols(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        self.assertEqual(dataset.indexes, self.TruesIndexFromBeginning)
        # ccc note the indexes are relative from beginning
        self.assertMainGroupsIdxEmpty(dataset)

    def testDf_noUseNpDictForDfs_StartPointsInCols(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        self.assertEqual(dataset.indexes, list(self.df[self.df['__startPoint__'] == True].index))
        self.assertMainGroupsIdxEmpty(dataset)

    def testNpDict_StartPointsInCols(self):
        dataset = VAnnTsDataset(NpDict(self.df), **self.kwargs)
        self.assertEqual(dataset.indexes, self.TruesIndexFromBeginning)
        "#ccc note the indexes are relative from beginning"
        self.assertMainGroupsIdxEmpty(dataset)


# ----         VAnnTsDataset_indexesSetting_NSeriesTests
class VAnnTsDataset_indexesSetting_NSeriesTests(BaseTestClass):
    # this is related to _setIndexes and _assignData_NMainGroupsIdxs
    # ccc test assumes, noIndexes are passed to dataset init and there is
    # some value for backcast and forecast lens

    def setUp(self):
        self.kwargs = {'backcastLen': 3, 'forecastLen': 2, 'mainGroups': ['group']}
        g1startPoints = [True, False, True, True, False, False, False, False]
        g2startPoints = [False, True, False, True, False, False, False, False]
        self.df = pd.DataFrame({'A': list(range(40, 56)),
                                '__startPoint__': g1startPoints + g2startPoints,
                                'group': 8 * ['g1'] + 8 * ['g2']},
                               index=list(range(110, 126)))

        TruesIndexFromBeginning = lambda inpList: [i for i, val in enumerate(inpList) if
                                                   val == True]
        self.TruesIndexFromBeginning = TruesIndexFromBeginning(self.df['__startPoint__'])
        self.mgGeneralIndexes = {'g1': [0, 2, 3],
                                 'g2': [9, 11]}
        self.mgRelIndexes = {'g1': TruesIndexFromBeginning(g1startPoints),
                             'g2': TruesIndexFromBeginning(g2startPoints)}

    def assertionsFor_Df_useNpDictForDfs_AndNpDict(self, dataset):
        self.setUp()
        self.assertEqual(dataset.indexes, self.TruesIndexFromBeginning)
        self.assertDictEqual(dataset.mainGroupsGeneralIdxs, self.mgGeneralIndexes)
        self.assertDictEqual(dataset.mainGroupsRelIdxs, self.mgRelIndexes)

    def testDf_useNpDictForDfs_StartPointsInCols(self):
        self.setUp()
        # this is gonna be similar to testNpDict_StartPointsInCols
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        self.assertionsFor_Df_useNpDictForDfs_AndNpDict(dataset)

    def testDf_noUseNpDictForDfs_StartPointsInCols(self):
        self.setUp()
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        self.assertEqual(dataset.indexes, list(self.df[self.df['__startPoint__'] == True].index))
        # bugPotentialCheck2 # goodToHave3
        #  this is not an important one (because doesn't affect the correctness of code or even
        #  user behavior at any level), but dataset.mainGroupsGeneralIdxs was checked in another
        #  computer which was macbook and the dataset.mainGroupsGeneralIdxs was
        #  `{'g1': [110, 112, 113], 'g2': [119, 121]}`. so this comes either macOs version of python
        #  or maybe related to other python package differences.
        self.assertDictEqual(dataset.mainGroupsGeneralIdxs, {'g1': [110, 112, 113],
                                                             'g2': [119, 121]})
        self.assertDictEqual(dataset.mainGroupsRelIdxs, {'g1': [], 'g2': []})

    def testNpDict_StartPointsInCols(self):
        self.setUp()
        dataset = VAnnTsDataset(NpDict(self.df), **self.kwargs)
        self.assertionsFor_Df_useNpDictForDfs_AndNpDict(dataset)


# ----         VAnnTsDataset_NoNanOrNoneDataAssertionTests
class VAnnTsDataset_NoNanOrNoneDataAssertionTests(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, np.nan, 8],
                                '__startPoint__': [False, True, True, False]},
                               index=[8, 9, 10, 11])

    def testDf_WithStartPoint(self):
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(self.df, backcastLen=1, forecastLen=1, useNpDictForDfs=False)
        self.assertEqual(str(context.exception), "The DataFrame contains NaN values.")

    def testNpDict_WithStartPoint(self):
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(NpDict(self.df), backcastLen=1, forecastLen=1)
        self.assertEqual(str(context.exception), "The NpDict contains NaN values.")

    def testNpArray_WithStartPoint(self):
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]])
        with self.assertRaises(ValueError) as context:
            VAnnTsDataset(npArray, backcastLen=0, forecastLen=0)
        self.assertEqual(str(context.exception), "The NumPy array contains NaN values.")


# ----         VAnnTsDataset_NSeries_assignData
class VAnnTsDataset_NSeries_assignData(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': 26 * ['A1'],
            'B': 7 * ['B1'] + 19 * ['B2'],
            tsStartPointColName: 3 * [True] + 4 * [False] + 15 * [True] + 4 * [False],
            'y1': list(range(30, 56)),
            'y2': list(range(130, 156))}, index=range(200, 226))
        self.dataset = VAnnTsDataset(self.df, backcastLen=2, forecastLen=3, mainGroups=['A', 'B'],
                                     useNpDictForDfs=False)
        self.expectedGroup1 = pd.DataFrame({'A': ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1'],
                                            'B': ['B1', 'B1', 'B1', 'B1', 'B1', 'B1', 'B1'],
                                            '__startPoint__': [True, True, True, False, False,
                                                               False, False],
                                            'y1': [30, 31, 32, 33, 34, 35, 36],
                                            'y2': [130, 131, 132, 133, 134, 135, 136]},
                                           index=list(range(200, 207)))
        self.expectedGroup2 = pd.DataFrame({'A': 19 * ['A1'], 'B': 19 * ['B2'],
                                            '__startPoint__': 15 * [True] + 4 * [False],
                                            'y1': list(range(37, 56)),
                                            'y2': list(range(137, 156))},
                                           index=list(range(207, 226)))

    def test(self):
        self.assertTrue(list(self.dataset.data.keys()) == [('A1', 'B1'), ('A1', 'B2')])
        self.assertTrue(self.expectedGroup1.equals(self.dataset.data[('A1', 'B1')]))
        self.assertTrue(self.expectedGroup2.equals(self.dataset.data[('A1', 'B2')]))


# ----         VAnnTsDataset_IdxNdataToLook_WhileFetching_NoSeriesTests
class VAnnTsDataset_IdxNdataToLook_WhileFetching_NoSeriesTests(BaseTestClass):
    def setUp(self):
        self.kwargs = {'backcastLen': 3, 'forecastLen': 2}
        self.df = pd.DataFrame({'A': list(range(10)),
                                '__startPoint__': [True, False, True, True, False, True, False,
                                                   True, False, True]},
                               index=list(range(110, 120)))

    def assertionsFor_Df_useNpDictForDfs_AndNpDict(self, dataset):
        idx = 5
        dataToLook, newIdx = dataset._IdxNdataToLook_WhileFetching(idx)
        self.equalNpDicts(dataToLook, NpDict(self.df))
        self.assertEqual(newIdx, 5)

    def testDf_useNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        self.assertionsFor_Df_useNpDictForDfs_AndNpDict(dataset)

    def testDf_noUseNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        idx = 112
        dataToLook, newIdx = dataset._IdxNdataToLook_WhileFetching(idx)
        self.equalDfs(dataToLook, self.df)
        self.assertEqual(newIdx, idx)

    def testNpDict(self):
        dataset = VAnnTsDataset(NpDict(self.df), **self.kwargs)
        self.assertionsFor_Df_useNpDictForDfs_AndNpDict(dataset)


# ----         VAnnTsDataset_IdxNdataToLook_WhileFetching_seriesTests
class VAnnTsDataset_IdxNdataToLook_WhileFetching_seriesTests(BaseTestClass):
    def setUp(self):
        g1startPoints = [True, False, True, True, False, False, False, False]
        g2startPoints = [False, True, False, True, False, False, False, False]
        self.df = pd.DataFrame({'A': list(range(40, 56)),
                                '__startPoint__': g1startPoints + g2startPoints,
                                'group': 8 * ['g1'] + 8 * ['g2']},
                               index=list(range(110, 126)))
        self.kwargs = {'backcastLen': 3, 'forecastLen': 2, 'mainGroups': ['group']}
        # kkk in tests only important args should be shown, use this kwargs method elsewhere

    def assertionsFor_Df_useNpDictForDfs_AndNpDict(self, dataset):
        idx = 11
        dataToLook, newIdx = dataset._IdxNdataToLook_WhileFetching(idx)
        expectedDataToLook_dict = {'A': [48, 49, 50, 51, 52, 53, 54, 55],
                                   '__startPoint__': [False, True, False, True] + \
                                                     4 * [False],
                                   'group': 8 * ['g2'], }
        self.assertEqual(dataToLook.getDict(resetDtype=True), expectedDataToLook_dict)
        self.assertEqual(newIdx, 3)

    def testDf_useNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        self.assertionsFor_Df_useNpDictForDfs_AndNpDict(dataset)

    def testDf_noUseNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        idx = 121
        dataToLook, newIdx = dataset._IdxNdataToLook_WhileFetching(idx)
        expectedDataToLook = pd.DataFrame({'A': [48, 49, 50, 51, 52, 53, 54, 55],
                                           '__startPoint__': [False, True, False, True] + \
                                                             4 * [False],
                                           'group': 8 * ['g2'], }, index=list(range(118, 126)))
        self.equalDfs(dataToLook, expectedDataToLook)
        self.assertEqual(newIdx, 121)

    def testNpDict(self):
        dataset = VAnnTsDataset(NpDict(self.df), **self.kwargs)
        self.assertionsFor_Df_useNpDictForDfs_AndNpDict(dataset)


# ----         VAnnTsDataset_NoNSeries_GetItemTests
class VAnnTsDataset_NoNSeries_GetItemTests(BaseTestClass):
    def setUp(self):
        self.kwargs = {'backcastLen': 3, 'forecastLen': 2}
        self.df = pd.DataFrame({'A': list(range(10)),
                                '__startPoint__': [True, False, True, True, False, True, False,
                                                   False, False, False]},
                               index=list(range(110, 120)))

    def testDf_useNpDictForDfs_StartPointsInCols(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        idx = 5
        expected = self.df.loc[115].values
        result = dataset[idx]
        self.equalArrays(expected, result, checkType=False)

    def testDf_useNpDictForDfs_StartPointsInCols_expectedRaisingIndexError_notInTsStartPointIndexes(
            self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        idx = 1
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testDf_noUseNpDictForDfs_StartPointsInCols(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        idx = 113
        expected = self.df.loc[idx]
        result = dataset[idx]
        self.equalDfs(expected, result)

    def testDf_noUseNpDictForDfs_StartPointsInCols_expectedRaisingIndexError_notInTsStartPointIndexes(
            self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        idx = 11  # dataset.indexes==[110, 112, 113, 115]
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testNpDict_StartPointsInCols(self):
        npDict = NpDict(self.df)
        dataset = VAnnTsDataset(npDict, **self.kwargs)
        idx = 2
        "#ccc note indexes for NpDict with no mainGroups, are according to their order from beginning of the arrays"
        expected = npDict[:][idx]
        result = dataset[idx]
        self.equalArrays(expected, result)

    def testNpDict_StartPointsInCols_expectedRaisingIndexError(self):
        npDict = NpDict(self.df)
        dataset = VAnnTsDataset(npDict, **self.kwargs)
        idx = 13  # this not in npDict at all( len of npDict is 10), and the problem is not, not to have tsStartPoint==True
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testNpDict_StartPointsInCols_expectedRaisingIndexError_notInTsStartPointIndexes(self):
        npDict = NpDict(self.df)
        dataset = VAnnTsDataset(npDict, **self.kwargs)
        idx = 1
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testDataTypeError(self):
        dataset = VAnnTsDataset({'a': 10, 'b': 30}, backcastLen=0, forecastLen=0)
        with self.assertRaises(ValueError) as context:
            dataset[0]
        self.assertEqual(str(context.exception),
                         'only datasets with pd.DataFrame, NpDict, np.array and torch.Tensor data can use __getitem__')

    # ---- np array
    def testGetItem_NpArray(self):
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = VAnnTsDataset(npArray, backcastLen=0, forecastLen=0)
        idx = 1
        expected = np.array([4, 5, 6])
        result = dataset[idx]
        self.equalArrays(result, expected)

    def testGetItem_NpArray_NotInIndexes(self):
        npArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = VAnnTsDataset(npArray, backcastLen=0, forecastLen=0, indexes=[0, 2])
        idx = 1
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")


# ----         VAnnTsDataset_NSeries_GetItemTests
class VAnnTsDataset_NSeries_GetItemTests(BaseTestClass):
    def setUp(self):
        self.kwargs = {'backcastLen': 3, 'forecastLen': 2, 'mainGroups': ['group']}
        self.df = pd.DataFrame({'A': list(range(16)),
                                '__startPoint__': [True, False, True, True, False, False, False, False] \
                                                  + [False, True, False, True, False, False, False, False],
                                'group': 8 * ['g1'] + 8 * ['g2']},
                               index=list(range(110, 126)))

    def testDf_StartPointsInCols_useNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        idx = 9  # from g2
        expected = self.df.loc[119].values
        result = dataset[idx]
        self.equalArrays(expected, result, checkType=False)

    def testDf_StartPointsInCols_useNpDictForDfs_expectedRaisingIndexError_notInTsStartPointIndexes(
            self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=True, **self.kwargs)
        idx = 8  # from g2
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testDf_StartPointsInCols_noUseNpDictForDfs(self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        idx = 113  # from g1
        expected = self.df.loc[idx]
        result = dataset[idx]
        self.equalDfs(expected, result)

    def testDf_StartPointsInCols_noUseNpDictForDfs_expectedRaisingIndexError_notInTsStartPointIndexes(
            self):
        dataset = VAnnTsDataset(self.df, useNpDictForDfs=False, **self.kwargs)
        idx = 111  # from g1
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")

    def testNpDict_StartPointsInCols(self):
        npDict = NpDict(self.df)
        dataset = VAnnTsDataset(NpDict(self.df), **self.kwargs)
        idx = 11  # from g2
        "#ccc note with NpDict with mainGroups, we use its df indexes"
        npDictIdx = 11
        expected = npDict[:][idx]
        result = dataset[idx]
        self.equalArrays(expected, result)

    def testNpDict_StartPointsInCols_expectedRaisingIndexError_notInTsStartPointIndexes(self):
        dataset = VAnnTsDataset(NpDict(self.df), **self.kwargs)
        idx = 14  # from g2
        with self.assertRaises(ValueError) as context:
            dataset[idx]
        self.assertEqual(str(context.exception), f"{idx} is not in indexes")


# ---- run test
if __name__ == '__main__':
    unittest.main()
