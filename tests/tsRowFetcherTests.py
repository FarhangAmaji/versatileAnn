from tests.baseTest import BaseTestClass
import unittest
import torch
import pandas as pd
import numpy as np
from utils.vAnnGeneralUtils import NpDict
from dataPrep.dataset import TsRowFetcher
#%% TsRowFetcherDfTests
class TestTsRowFetcherDfTests(BaseTestClass):
    def setUp(self):
        self.fetcher=TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                           'y3': [32, 33, 34, 35, 36, 37, 38, 39]},index=[130, 131, 132, 133, 134, 135, 136, 137])

    def test_backcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 0, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.float32)))

    def test_backcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.df, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(torch.equal(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.float32)))

    def test_backcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 1, mode='backcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (3, 2))
        expected_result = pd.DataFrame({'y1': [2, 3, 4], 'y2': [17, 18, 19]}, index=[131, 132, 133])
        self.assertTrue(expected_result.equals(result))

    def test_forecastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 2, mode='forecast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (2, 2))
        expected_result = pd.DataFrame({'y1': [6, 7], 'y2': [21, 22]}, index=[135, 136])
        self.assertTrue(expected_result.equals(result))

    def test_forecastModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.df, 3, mode='forecast', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 3))
        expected_result = pd.DataFrame({'y1': [7, 8], 'y2': [22, 23], 'y3': [38, 39]}, index=[136, 137])
        self.assertTrue(expected_result.equals(result))

    def test_forecastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 3, mode='forecast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.float32)))

    def test_fullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 0, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.float32)))

    def test_fullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.df, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.assertTrue(torch.equal(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.float32)))

    def test_fullcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 1, mode='fullcast', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (5, 2))
        expected_result = pd.DataFrame({'y1': [2, 3, 4, 5, 6], 'y2': [17, 18, 19, 20, 21]}, index=[131, 132, 133, 134, 135])
        self.assertTrue(expected_result.equals(result))

    def test_singlePointModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 2, mode='singlePoint', colsOrIndexes=['y1', 'y2'], makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (1, 2))
        expected_result = pd.DataFrame({'y1': [3], 'y2': [18]}, index=[132])
        self.assertTrue(expected_result.equals(result))

    def test_singlePointModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.df, 3, mode='singlePoint', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (1, 3))
        expected_result = pd.DataFrame({'y1': [4], 'y2': [19], 'y3': [35]}, index=[133])
        self.assertTrue(expected_result.equals(result))

    def test_singlePointModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.df, 3, mode='singlePoint', colsOrIndexes=['y1', 'y2'], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[4, 19]], dtype=torch.float32)))
#%% TestTsRowFetcherNpArrayTests
class TestTsRowFetcherNpArrayTests(BaseTestClass):
    def setUp(self):
        self.fetcher = TsRowFetcher(backcastLen=3, forecastLen=2)
        self.np_array = np.array([[1, 16, 32], [2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37], [7, 22, 38], [8, 23, 39]])

    def test_backcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 0, mode='backcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.float32)))

    def test_backcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 1, mode='backcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(torch.equal(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35]], dtype=torch.float32)))

    def test_backcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 1, mode='backcast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (3, 2))
        expected_result = np.array([[2, 17], [3, 18], [4, 19]])
        np.testing.assert_array_equal(expected_result, result)

    def test_forecastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 2, mode='forecast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 2))
        expected_result = np.array([[6, 21], [7, 22]])
        np.testing.assert_array_equal(expected_result, result)

    def test_forecastModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 3, mode='forecast', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 3))
        expected_result = np.array([[7, 22, 38], [8, 23, 39]])
        np.testing.assert_array_equal(expected_result, result)

    def test_forecastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 3, mode='forecast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.float32)))

    def test_fullcastModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 0, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]], dtype=torch.float32)))

    def test_fullcastModeMakeTensorTrueColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 1, mode='fullcast', colsOrIndexes='___all___', makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.assertTrue(torch.equal(result, torch.tensor([[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.float32)))

    def test_fullcastModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 1, mode='fullcast', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (5, 2))
        expected_result = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        np.testing.assert_array_equal(expected_result, result)

    def test_singlePointModeMakeTensorFalseColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 2, mode='singlePoint', colsOrIndexes=[0, 1], makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2))
        expected_result = np.array([[3, 18]])
        np.testing.assert_array_equal(expected_result, result)

    def test_singlePointModeMakeTensorFalseColIndexesAll(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 3, mode='singlePoint', colsOrIndexes='___all___', makeTensor=False)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 3))
        expected_result = np.array([[4, 19, 35]])
        np.testing.assert_array_equal(expected_result, result)

    def test_singlePointModeMakeTensorTrueColIndexesList(self):
        result = self.fetcher.getBackForeCastData(self.np_array, 3, mode='singlePoint', colsOrIndexes=[0, 1], makeTensor=True)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[4, 19]], dtype=torch.float32)))
#%% run test
if __name__ == '__main__':
    unittest.main()