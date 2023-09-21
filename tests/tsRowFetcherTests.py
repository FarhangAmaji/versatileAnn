#%% imports
'#ccc also test for getBackForeCastData in dataset'
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
import torch
import pandas as pd
import numpy as np
from utils.vAnnGeneralUtils import NpDict, tensorEqualWithDtype
from dataPrep.dataset import TsRowFetcher
#%% TsRowFetcherTests
"""
things have been tested:
    in TestTsRowFetcherDfTests, TestTsRowFetcherNpArrayTests, TestTsRowFetcherNpDictTests, TestTsRowFetcherTorchTensorTests:
                getBackForeCastDataGeneral: for 4 types of DataFrame,npArray, NpDict, tensor|for 4 modes| for specifiedCols or 
                '___all___'|for makeTensor True and False|keeping int dtype

"""
#%%        TestTsRowFetcherDfTests
class TestTsRowFetcherDfTests(BaseTestClass):
    def setUp(self):
        self.fetcher=TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                           'y3': [32, 33, 34, 35, 36, 37, 38, 39]},index=[130, 131, 132, 133, 134, 135, 136, 137])

    def testBackcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 130, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64)))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int64)))

    def testBackcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (3, 2))
        expectedResult = pd.DataFrame({'y1': [2, 3, 4], 'y2': [17, 18, 19]}, index=[131, 132, 133])
        self.assertTrue(expectedResult.equals(result))

    def testForecastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 132, mode='forecast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (2, 2))
        expectedResult = pd.DataFrame({'y1': [6, 7], 'y2': [21, 22]}, index=[135, 136])
        self.assertTrue(expectedResult.equals(result))

    def testForecastModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 133, mode='forecast', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = pd.DataFrame({'y1': [7, 8], 'y2': [22, 23], 'y3': [38, 39]}, index=[136, 137])
        self.assertTrue(expectedResult.equals(result))

    def testForecastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 133, mode='forecast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64)))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 130, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int64)))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64)))

    def testFullcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (5, 2))
        expectedResult = pd.DataFrame({'y1': [2, 3, 4, 5, 6], 'y2': [17, 18, 19, 20, 21]}, index=[131, 132, 133, 134, 135])
        self.assertTrue(expectedResult.equals(result))

    def testSinglePointModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 132, mode='singlePoint', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (1, 2))
        expectedResult = pd.DataFrame({'y1': [3], 'y2': [18]}, index=[132])
        self.assertTrue(expectedResult.equals(result))

    def testSinglePointModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 133, mode='singlePoint', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (1, 3))
        expectedResult = pd.DataFrame({'y1': [4], 'y2': [19], 'y3': [35]}, index=[133])
        self.assertTrue(expectedResult.equals(result))

    def testSinglePointModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 133, mode='singlePoint', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[4, 19]], dtype=torch.int64)))
#%%        TestTsRowFetcherNpArrayTests
class TestTsRowFetcherNpArrayTests(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)
        self.npArray = np.array([[1, 16, 32], [2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37], [7, 22, 38], [8, 23, 39]])

    def testBackcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 0, mode='backcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int32)))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int32)))

    def testBackcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 1, mode='backcast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (3, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 2, mode='forecast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 2))
        expectedResult = np.array([[6, 21], [7, 22]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 3, mode='forecast', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = np.array([[7, 22, 38], [8, 23, 39]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 3, mode='forecast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int32)))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 0, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int32)))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int32)))

    def testFullcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 1, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (5, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 2, mode='singlePoint', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2))
        expectedResult = np.array([[3, 18]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 3, mode='singlePoint', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 3))
        expectedResult = np.array([[4, 19, 35]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 3, mode='singlePoint', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[4, 19]], dtype=torch.int32)))
#%%        TestTsRowFetcherNpDictTests
class TestTsRowFetcherNpDictTests(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],
                                'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39]},
                               index=[130, 131, 132, 133, 134, 135, 136, 137])
        self.npDict = NpDict(self.df)

    def testBackcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 0, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64)))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int64)))

    def testBackcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 1, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (3, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 2, mode='forecast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 2))
        expectedResult = np.array([[6, 21], [7, 22]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 3, mode='forecast', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = np.array([[7, 22, 38], [8, 23, 39]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 3, mode='forecast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64)))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 0, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int64)))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64)))

    def testFullcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 1, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (5, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 2, mode='singlePoint', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2))
        expectedResult = np.array([[3, 18]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 3, mode='singlePoint', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 3))
        expectedResult = np.array([[4, 19, 35]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 3, mode='singlePoint', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[4, 19]], dtype=torch.int64)))
#%%        TestTsRowFetcherTorchTensorTests
class TestTsRowFetcherTorchTensorTests(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],
                                'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39]},
                               index=[130, 131, 132, 133, 134, 135, 136, 137])
        self.tensor = torch.tensor([[1, 16, 32], [2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37], [7, 22, 38], [8, 23, 39]], dtype=torch.int64)

    def testBackcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 0, mode='backcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64)))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int64)))

    def testBackcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 1, mode='backcast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 2, mode='forecast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        expectedResult = np.array([[6, 21], [7, 22]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 3, mode='forecast', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = np.array([[7, 22, 38], [8, 23, 39]])
        np.testing.assert_array_equal(expectedResult, result)

    def testForecastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 3, mode='forecast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64)))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 0, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int64)))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64)))

    def testFullcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 1, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 2, mode='singlePoint', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        expectedResult = np.array([[3, 18]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 3, mode='singlePoint', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 3))
        expectedResult = np.array([[4, 19, 35]])
        np.testing.assert_array_equal(expectedResult, result)

    def testSinglePointModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 3, mode='singlePoint', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([[4, 19]], dtype=torch.int64)))
#%%        TestTsRowFetcherChangeFloatDtypeTests
'#ccc in the tests above we have tested that if type is int, it would keep it int'
class TestTsRowFetcherFloatDtypeChange(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)
    
    def testChangeFloatDtypeFrom64to32(self):
        inputData = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0],
                               [10.0, 11.0, 12.0],
                               [13.0, 14.0, 15.0],
                               [16.0, 17.0, 18.0]], dtype=np.float64)
        result = self.fetcher.getBackForeCastDataGeneral(inputData, idx=1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (3, 3))
        expectedResult = torch.tensor([[4.0, 5.0, 6.0],
                                       [7.0, 8.0, 9.0],
                                       [10.0, 11.0, 12.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(result, expectedResult))
#%%        TestTsRowFetcherShorterLenError
class TestTsRowFetcherShorterLenError(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=8, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39]}, index=[130, 131, 132, 133, 134, 135, 136, 137])

    def testBackcastModeShorterLenNoErrorDf(self):
        self.fetcher.getBackForeCastDataGeneral(self.df, 130, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)

    def testBackcastModeShorterLenErrorDf(self):
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(TsRowFetcher.errMsgs['shorterLen'],str(context.exception))

    def testBackcastModeShorterLenAllowedDf(self):
        res = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='backcast', colsOrIndexes=['y1'],
                                                      makeTensor=False,canHaveShorterLength=True)
        self.assertTrue(res.equals(pd.DataFrame({'y1': [2, 3, 4, 5, 6, 7, 8]}, index=[131, 132, 133, 134, 135, 136, 137])))
        

    def testBackcastModeShorterLenNoErrorNpDict(self):
        self.fetcher.getBackForeCastDataGeneral(NpDict(self.df), 0, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        
    def testBackcastModeShorterLenErrorNpDict(self):#kkk add canHaveShorterLength
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(NpDict(self.df), 1, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(TsRowFetcher.errMsgs['shorterLen'],str(context.exception))

    def testBackcastModeShorterLenAllowedNpDict(self):
        res=self.fetcher.getBackForeCastDataGeneral(NpDict(self.df), 1, mode='backcast',
                                                    colsOrIndexes=['y1'], makeTensor=False,canHaveShorterLength=True)
        np.testing.assert_array_equal(res,np.array([2, 3, 4, 5, 6, 7, 8]))
#%%        TestTsRowFetcherOutOfDataError
'#ccc this is different than, out of dataset indexes. this validate is out of this data(df, npArray,...) indexes or not'
class TestTsRowFetcherOutOfDataError(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=8, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39]}, index=[130, 131, 132, 133, 134, 135, 136, 137])

    def testBackcastOutOfDataErrorDf(self):
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(self.df, 129, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(TsRowFetcher.errMsgs['non-negStartingPointDf'],str(context.exception))

    def testBackcastOutOfDataErrorNpDict(self):
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(NpDict(self.df), -1, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)        
        self.assertTrue(TsRowFetcher.errMsgs['non-negStartingPointNpDict'],str(context.exception))

    def testBackcastOutOfDataErrorNpArray(self):
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(self.df.values, -1, mode='backcast', colsOrIndexes=[0,2], makeTensor=False)
        self.assertTrue(TsRowFetcher.errMsgs['non-negStartingPointNpArray'],str(context.exception))

    def testBackcastOutOfDataErrorTensor(self):
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(torch.tensor(self.df.values), -1, mode='backcast', colsOrIndexes=[0,2], makeTensor=False)
        self.assertTrue(TsRowFetcher.errMsgs['non-negStartingPointTensor'],str(context.exception))
#%%        TestTsRowFetcherSingleFeatureShapeCorrectionTests
class TestTsRowFetcherSingleFeatureShapeCorrectionTests(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)

    def testSingleFeatureShapeCorrection(self):
        input_data = torch.tensor([[1], [2], [3]])
        result = self.fetcher.singleFeatureShapeCorrection(input_data)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3,))
        self.assertTrue(tensorEqualWithDtype(result, torch.tensor([1, 2, 3], dtype=torch.int64)))
#%% run test
if __name__ == '__main__':
    unittest.main()