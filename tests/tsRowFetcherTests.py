#%% imports
'#ccc also test for getBackForeCastData in dataset'
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.baseTest import BaseTestClass
import unittest
import torch
import pandas as pd
import numpy as np
from utils.vAnnGeneralUtils import NpDict
from utils.globalVars import tsStartPointColName
from dataPrep.dataset import TsRowFetcher, VAnnTsDataset
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
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int64))

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
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 130, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int64))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64))

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
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int64))
#%%        TestTsRowFetcherNpArrayTests
class TestTsRowFetcherNpArrayTests(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)
        self.npArray = np.array([[1, 16, 32], [2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37], [7, 22, 38], [8, 23, 39]])

    def testBackcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 0, mode='backcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int32))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int32))

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
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int32))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 0, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int32))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npArray, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int32))

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
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int32))
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
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int64))

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
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 0, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int64))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.npDict, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64))

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
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int64))
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
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64))

    def testBackcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.int64))

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
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64))

    def testFullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 0, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.int64))

    def testFullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastDataGeneral(self.tensor, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64))

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
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int64))
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
        #kkk may have added the result check(maybe not needed ofc, done in other tests)
        
    def testBackcastModeShorterLenErrorNpDict(self):
        with self.assertRaises(AssertionError) as context:
            self.fetcher.getBackForeCastDataGeneral(NpDict(self.df), 1, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(TsRowFetcher.errMsgs['shorterLen'],str(context.exception))

    def testBackcastModeShorterLenAllowedNpDict(self):
        res=self.fetcher.getBackForeCastDataGeneral(NpDict(self.df), 1, mode='backcast',
                                                    colsOrIndexes=['y1'], makeTensor=False,canHaveShorterLength=True)
        np.testing.assert_array_equal(res,np.array([2, 3, 4, 5, 6, 7, 8]))
#%%        TestTsRowFetcherRightPad
class TestTsRowFetcherRightPad(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=8, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39]}, index=[130, 131, 132, 133, 134, 135, 136, 137])

    def testBackcastRightPadDf(self):
        res=self.fetcher.getBackForeCastDataGeneral(self.df, 134, mode='backcast', colsOrIndexes=['y1', 'y2'],
                                                makeTensor=False,shiftForward=2,rightPadIfShorter=True)
        dfRightPadExpectedRes=pd.DataFrame({'y1': [7, 8, 0, 0, 0, 0, 0, 0],'y2': [22, 23, 0, 0, 0, 0, 0, 0]},
                                           index=[136, 137, 138, 139, 140, 141, 142, 143])
        self.assertTrue(res.equals(dfRightPadExpectedRes))

    def testBackcastNotToRightPadDf(self):
        res=self.fetcher.getBackForeCastDataGeneral(self.df, 130, mode='backcast', colsOrIndexes=['y1', 'y2'],
                                                makeTensor=False,shiftForward=0,rightPadIfShorter=True)
        dfRightPadExpectedRes=pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23]},
                                           index=[130, 131, 132, 133, 134, 135, 136, 137])
        self.assertTrue(res.equals(dfRightPadExpectedRes))

    def testBackcastRightPadNpArray(self):
        res=self.fetcher.getBackForeCastDataGeneral(self.df.values, 4, mode='backcast', colsOrIndexes=[0,2],
                                                makeTensor=False,shiftForward=2,rightPadIfShorter=True)
        np.testing.assert_array_equal(res,np.array([[ 7, 38],[ 8, 39],[ 0,  0],[ 0,  0],[ 0,  0],[ 0,  0],[ 0,  0],[ 0,  0]]))

    def testBackcastNotToRightPadNpArray(self):
        res=self.fetcher.getBackForeCastDataGeneral(self.df.values, 0, mode='backcast', colsOrIndexes=[0,2],
                                                makeTensor=False,shiftForward=0,rightPadIfShorter=True)
        np.testing.assert_array_equal(res,np.array([[ 1, 32],[ 2, 33],[ 3, 34],[ 4, 35],[ 5, 36],[ 6, 37],[ 7, 38],[ 8, 39]]))

    def testBackcastRightPadTensor(self):
        res=self.fetcher.getBackForeCastDataGeneral(torch.tensor(self.df.values), 4, mode='backcast',
                                                    colsOrIndexes=[0,2],shiftForward=2, makeTensor=True,rightPadIfShorter=True)
        expectedResult = torch.tensor([[ 7, 38],[ 8, 39],[ 0,  0],[ 0,  0],[ 0,  0],[ 0,  0],[ 0,  0],[ 0,  0]], dtype=torch.int64)
        self.assertTrue(torch.equal(res, expectedResult))
        
    def testBackcastNotToRightPadTensor(self):
        res=self.fetcher.getBackForeCastDataGeneral(torch.tensor(self.df.values), 0, mode='backcast',
                                                    colsOrIndexes=[0,2],shiftForward=0, makeTensor=True,rightPadIfShorter=True)
        expectedResult = torch.tensor([[ 1, 32],[ 2, 33],[ 3, 34],[ 4, 35],[ 5, 36],[ 6, 37],[ 7, 38],[ 8, 39]], dtype=torch.int64)
        self.assertTrue(torch.equal(res, expectedResult))
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
        inputData = torch.tensor(
            [[[[ 4],[ 7],[ 6]],[[ 4],[ 8],[ 3]],[[ 1],[ 3],[ 9]]],
             [[[ 6],[ 3],[ 1]],[[ 1],[ 9],[ 7]],[[ 2],[ 3],[ 1]]],
             [[[10],[ 6],[ 1]],[[ 3],[ 2],[ 9]],[[10],[ 9],[ 3]]]]
            )
        result = self.fetcher.singleFeatureShapeCorrection(inputData)
        self.assertTrue(result.shape==torch.Size([3, 3, 3]))
#%%        TestDataset_NSeries_getBackForeCastData
class TestDataset_NSeries_getBackForeCastData(BaseTestClass):
    #kkk could have added no useNpDictForDfs=True datasets
    def setUp(self):
        self.df = pd.DataFrame({
            'A': 26*['A1'],
            'B': 7*['B1']+19*['B2'],
            tsStartPointColName: 3*[True]+4*[False]+15*[True]+4*[False],
            'y1': list(range(30, 56)),
            'y2': list(range(130, 156))},index=range(100, 126))
        self.dataset=VAnnTsDataset(self.df,backcastLen=2, forecastLen=3, mainGroups=['A','B'], useNpDictForDfs=False)

    def testShift_inGroupBound(self):
        res=self.dataset.getBackForeCastData(101, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                makeTensor=False, shiftForward=-1, rightPadIfShorter=False)
        expectedResult = pd.DataFrame({'y1': list(range(32, 35)), 'y2': list(range(132, 135))},index=range(102, 105))
        self.assertTrue(res.equals(expectedResult))

    def testForecast_NSeries_rightPadIfShorter(self):
        "#ccc ensure dataset with mainGroups can only get data from its group"
        res=self.dataset.getBackForeCastData(103, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                makeTensor=False, shiftForward=0, canBeOutStartIndex=True, rightPadIfShorter=True)
        expectedResult = pd.DataFrame({'y1': [35,36,0], 'y2': [135,136,0]},index=range(105, 108))
        "#ccc note idx==107 exists in self.df, and is another group"
        self.assertTrue(res.equals(expectedResult))

    def test_notInDatasetIndexes(self):
        "#ccc 103 is in dataset but with '__startPoint__'==False"
        "#ccc this test is for self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx)"
        with self.assertRaises(AssertionError) as context:
            self.dataset.getBackForeCastData(103, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                    makeTensor=False, shiftForward=0, rightPadIfShorter=False)
        self.assertTrue("103 is not in indexes",str(context.exception))

    def test_withShift_notInDatasetIndexes(self):
        "#ccc this test is for self.assertIdxInIndexesDependingOnAllowance(canBeOutStartIndex, idx+shiftForward)"
        "#ccc idx+shiftForward also must be in dataset indexes"
        with self.assertRaises(AssertionError) as context:
            self.dataset.getBackForeCastData(106, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                    makeTensor=False, shiftForward=-2, rightPadIfShorter=False)
        self.assertTrue("104 is not in indexes",str(context.exception))
#%%        TestTsRowFetcher_Shift
class TestTsRowFetcher_Shift(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=2, forecastLen=3)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39]}, index=[130, 131, 132, 133, 134, 135, 136, 137])

    def testShift(self):
        res=self.fetcher.getBackForeCastDataGeneral(self.df, 131, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                makeTensor=False, shiftForward=2, rightPadIfShorter=False)
        expectedResult = pd.DataFrame({'y1': [6,7,8], 'y2': [21,22,23]},index=range(135, 138))
        self.assertTrue(res.equals(expectedResult))

    def testShiftNeg(self):
        res=self.fetcher.getBackForeCastDataGeneral(self.df, 134, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                makeTensor=False, shiftForward=-2, rightPadIfShorter=False)
        expectedResult = pd.DataFrame({'y1': [5,6,7], 'y2': [20,21,22]},index=range(134, 137))
        self.assertTrue(res.equals(expectedResult))
#%%     TestDataset_nonNSeries_InDatasetIndexes
class TestDataset_nonNSeries_InDatasetIndexes(BaseTestClass):
    def setUp(self):
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                                'y3': [32, 33, 34, 35, 36, 37, 38, 39], tsStartPointColName:4*[True]+4*[False]},
                               index=[130, 131, 132, 133, 134, 135, 136, 137])
        self.dataset=VAnnTsDataset(self.df,backcastLen=2, forecastLen=3, useNpDictForDfs=False)

    def test_notInDatasetIndexes(self):
        with self.assertRaises(AssertionError) as context:
            self.dataset.getBackForeCastData(134, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                    makeTensor=False, shiftForward=0, rightPadIfShorter=False)
        self.assertTrue("134 is not in indexes",str(context.exception))

    def test_withShift_notInDatasetIndexes(self):
        with self.assertRaises(AssertionError) as context:
            self.dataset.getBackForeCastData(137, mode='forecast', colsOrIndexes=['y1', 'y2'],
                                                    makeTensor=False, shiftForward=-2, rightPadIfShorter=False)
        self.assertTrue("135 is not in indexes",str(context.exception))
#%% run test
if __name__ == '__main__':
    unittest.main()