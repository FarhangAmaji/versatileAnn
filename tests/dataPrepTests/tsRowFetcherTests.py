# ---- imports
"""
#ccc important::: test names are important as they provide kwargs
            for 'getBackForeCastData_general' through giveKwargsByFuncName_dfSample1
            and giveKwargsByFuncName_npArraySample1

#ccc note 'autoKwargsByFuncName' provides kwargs passed.
        so focus of reader of this file, is on more detailed items
"""

import inspect
import unittest

import numpy as np
import pandas as pd
import torch

from dataPrep.dataset import _TsRowFetcher, VAnnTsDataset
from tests.baseTest import BaseTestClass
from projectUtils.dataTypeUtils.dotDict_npDict import NpDict
from projectUtils.globalVars import tsStartPointColName

# ---- sample variables
npArraySample1 = np.array([[1, 16, 32],
                           [2, 17, 33],
                           [3, 18, 34],
                           [4, 19, 35],
                           [5, 20, 36],
                           [6, 21, 37],
                           [7, 22, 38],
                           [8, 23, 39]])

dfSample1 = pd.DataFrame({'y1': [1, 2, 3, 4, 5, 6, 7, 8],
                          'y2': [16, 17, 18, 19, 20, 21, 22, 23],
                          'y3': [32, 33, 34, 35, 36, 37, 38, 39]},
                         index=[130, 131, 132, 133, 134, 135, 136, 137])


# ---- create dictionaries with args to pass, by func name
def createKwargsDictFromName_base(name):
    dict_ = {}

    # set modeType
    if '_BackcastMode' in name:
        dict_['mode'] = 'backcast'
    elif '_ForecastMode' in name:
        dict_['mode'] = 'forecast'
    elif '_FullcastMode' in name:
        dict_['mode'] = 'fullcast'
    elif '_SinglePointMode' in name:
        dict_['mode'] = 'singlePoint'

    # set colType
    if '_AllColIndexes' in name:
        dict_['colsOrIndexes'] = '___all___'

    # set makeTensorType
    if '_ConvTensor' in name:
        dict_['outputTensor'] = True
    elif '_noConvTensor' in name:
        dict_['outputTensor'] = False

    return dict_


def createKwargsDictFromName_dfSample1(name):
    dict_ = createKwargsDictFromName_base(name)
    if '_someCols' in name:
        dict_['colsOrIndexes'] = ['y1', 'y2']
    return dict_


def createKwargsDictFromName_npArraySample1(name):
    dict_ = createKwargsDictFromName_base(name)
    if '_someCols' in name:
        dict_['colsOrIndexes'] = [0, 1]
    return dict_


def giveKwargsByFuncName_base(operationFunc):
    frame = inspect.currentframe()
    try:
        funcName = frame.f_back.f_back.f_code.co_name
        return operationFunc(funcName)
    except:
        return {}
    finally:
        del frame


def giveKwargsByFuncName_dfSample1():
    return giveKwargsByFuncName_base(createKwargsDictFromName_dfSample1)


def giveKwargsByFuncName_npArraySample1():
    return giveKwargsByFuncName_base(createKwargsDictFromName_npArraySample1)


# ---- common used kwargs
commonkwargs1 = createKwargsDictFromName_base('_BackcastMode_AllColIndexes_ConvTensor_kwargs')
commonkwargs2 = createKwargsDictFromName_dfSample1('_BackcastMode_someCols_noConvTensor_kwargs')
commonkwargs3 = createKwargsDictFromName_dfSample1('_ForecastMode_someCols_noConvTensor_kwargs')
# ---- TsRowFetcherTests
# ----        TsRowFetcherTests_types
"""#ccc
in TsRowFetcherTests_types, including (TestTsRowFetcher_DfTests, TestTsRowFetcher_NpArrayTests,
                                       TestTsRowFetcher_NpDictTests, TestTsRowFetcher_TensorTests)

for each (backcast, forecast, fullcast, singlePoint) castModes
for outputTensor== True or False
for someCols == ['y1', 'y2'](or [0, 1]) or _AllColIndexes=='___all___'
"""


# ----        type: TestTsRowFetcher_DfTests
class TestTsRowFetcher_DfTests(BaseTestClass):
    # ccc note uses giveKwargsByFuncName_dfSample1 for autoKwargsByFuncName
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = dfSample1.copy()

    def test_BackcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        # kkk have added some wrapper to prevent 'autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()' repetition
        result = self.fetcher.getBackForeCastData_general(self.df, 130, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64))

    def test_BackcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 131, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:4], dtype=torch.int64))

    def test_BackcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 131, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (3, 2))
        expectedResult = pd.DataFrame({'y1': [2, 3, 4], 'y2': [17, 18, 19]}, index=[131, 132, 133])
        self.assertTrue(expectedResult.equals(result))

    def test_ForecastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 132, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (2, 2))
        expectedResult = pd.DataFrame({'y1': [6, 7], 'y2': [21, 22]}, index=[135, 136])
        self.assertTrue(expectedResult.equals(result))

    def test_ForecastMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 133, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = pd.DataFrame({'y1': [7, 8], 'y2': [22, 23], 'y3': [38, 39]},
                                      index=[136, 137])
        self.assertTrue(expectedResult.equals(result))

    def test_ForecastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 133, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64))

    def test_FullcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 130, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]],
                                               dtype=torch.int64))

    def test_FullcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 131, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor(
            [[2, 17, 33], [3, 18, 34], [4, 19, 35], [5, 20, 36], [6, 21, 37]], dtype=torch.int64))

    def test_FullcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 131, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (5, 2))
        expectedResult = pd.DataFrame({'y1': [2, 3, 4, 5, 6], 'y2': [17, 18, 19, 20, 21]},
                                      index=[131, 132, 133, 134, 135])
        self.assertTrue(expectedResult.equals(result))

    def test_SinglePointMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 132, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (1, 2))
        expectedResult = pd.DataFrame({'y1': [3], 'y2': [18]}, index=[132])
        self.assertTrue(expectedResult.equals(result))

    def test_SinglePointMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 133, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(result.values.shape, (1, 3))
        expectedResult = pd.DataFrame({'y1': [4], 'y2': [19], 'y3': [35]}, index=[133])
        self.assertTrue(expectedResult.equals(result))

    def test_SinglePointMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.df, 133, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int64))

    def testDoes_TsRowFetcher_getRows_dfWorkWithSeries(self):
        result = self.fetcher.getRows_df(self.df['y1'], 131, lowerBoundGap=0,
                                         upperBoundGap=self.fetcher.backcastLen,
                                         colsOrIndexes='___all___')
        self.equalArrays(result.values, pd.Series([2, 3, 4], dtype=np.int64))


# ----        type: TestTsRowFetcher_NpArrayTests
class TestTsRowFetcher_NpArrayTests(BaseTestClass):
    # ccc note uses giveKwargsByFuncName_npArraySample1 for autoKwargsByFuncName
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=3, forecastLen=2)
        self.npArray = npArraySample1.copy()

    def test_BackcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 0, **autoKwargsByFuncName)
        # checking the dtype is not important; just to make sure it's .int32 as on some devices
        # the default is int64
        result = result.to(torch.int32)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int32))

    def test_BackcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 1, **autoKwargsByFuncName)
        # checking the dtype is not important; just to make sure it's .int32 as on some devices
        # the default is int64
        result = result.to(torch.int32)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:4], dtype=torch.int32))

    def test_BackcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (3, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19]])
        self.equalArrays(expectedResult, result)

    def test_ForecastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 2, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 2))
        expectedResult = np.array([[6, 21], [7, 22]])
        self.equalArrays(expectedResult, result)

    def test_ForecastMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = np.array([[7, 22, 38], [8, 23, 39]])
        self.equalArrays(expectedResult, result)

    def test_ForecastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 3, **autoKwargsByFuncName)
        # checking the dtype is not important; just to make sure it's .int32 as on some devices
        # the default is int64
        result = result.to(torch.int32)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int32))

    def test_FullcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 0, **autoKwargsByFuncName)
        # checking the dtype is not important; just to make sure it's .int32 as on some devices
        # the default is int64
        result = result.to(torch.int32)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]],
                                               dtype=torch.int32))

    def test_FullcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 1, **autoKwargsByFuncName)
        # checking the dtype is not important; just to make sure it's .int32 as on some devices
        # the default is int64
        result = result.to(torch.int32)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:6], dtype=torch.int32))

    def test_FullcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (5, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        self.equalArrays(expectedResult, result)

    def test_SinglePointMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 2, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2))
        expectedResult = np.array([[3, 18]])
        self.equalArrays(expectedResult, result)

    def test_SinglePointMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 3))
        expectedResult = np.array([[4, 19, 35]])
        self.equalArrays(expectedResult, result)

    def test_SinglePointMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.npArray, 3, **autoKwargsByFuncName)
        # checking the dtype is not important; just to make sure it's .int32 as on some devices
        # the default is int64
        result = result.to(torch.int32)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int32))


# ----        type: TestTsRowFetcher_NpDictTests
class TestTsRowFetcher_NpDictTests(BaseTestClass):
    # ccc note uses giveKwargsByFuncName_dfSample1 for autoKwargsByFuncName
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = dfSample1.copy()
        self.npDict = NpDict(self.df)

    def test_BackcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 0, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64))

    def test_BackcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:4], dtype=torch.int64))

    def test_BackcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (3, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_ForecastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 2, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 2))
        expectedResult = np.array([[6, 21], [7, 22]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_ForecastMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = npArraySample1[6:8]
        self.equalArrays(expectedResult, result, checkType=False)

    def test_ForecastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64))

    def test_FullcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 0, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]],
                                               dtype=torch.int64))

    def test_FullcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:6], dtype=torch.int64))

    def test_FullcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (5, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_SinglePointMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 2, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2))
        expectedResult = np.array([[3, 18]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_SinglePointMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 3))
        expectedResult = npArraySample1[3:4]
        self.equalArrays(expectedResult, result, checkType=False)

    def test_SinglePointMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_dfSample1()
        result = self.fetcher.getBackForeCastData_general(self.npDict, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int64))


# ----        type: TestTsRowFetcher_TensorTests
class TestTsRowFetcher_TensorTests(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=3, forecastLen=2)
        self.df = dfSample1.copy()
        self.tensor = torch.tensor(npArraySample1, dtype=torch.int64)

    def test_BackcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 0, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18]], dtype=torch.int64))

    def test_BackcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:4], dtype=torch.int64))

    def test_BackcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_ForecastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 2, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        expectedResult = np.array([[6, 21], [7, 22]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_ForecastMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 3))
        expectedResult = npArraySample1[6:8]
        self.equalArrays(expectedResult, result, checkType=False)

    def test_ForecastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (2, 2))
        self.equalTensors(result, torch.tensor([[7, 22], [8, 23]], dtype=torch.int64))

    def test_FullcastMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 0, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        self.equalTensors(result, torch.tensor([[1, 16], [2, 17], [3, 18], [4, 19], [5, 20]],
                                               dtype=torch.int64))

    def test_FullcastMode_AllColIndexes_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 3))
        self.equalTensors(result, torch.tensor(npArraySample1[1:6], dtype=torch.int64))

    def test_FullcastMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 1, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (5, 2))
        expectedResult = np.array([[2, 17], [3, 18], [4, 19], [5, 20], [6, 21]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_SinglePointMode_someCols_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 2, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        expectedResult = np.array([[3, 18]])
        self.equalArrays(expectedResult, result, checkType=False)

    def test_SinglePointMode_AllColIndexes_noConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 3))
        expectedResult = npArraySample1[3:4]
        self.equalArrays(expectedResult, result, checkType=False)

    def test_SinglePointMode_someCols_ConvTensor(self):
        autoKwargsByFuncName = giveKwargsByFuncName_npArraySample1()
        result = self.fetcher.getBackForeCastData_general(self.tensor, 3, **autoKwargsByFuncName)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (1, 2))
        self.equalTensors(result, torch.tensor([[4, 19]], dtype=torch.int64))


# ----        TestTsRowFetcher_ChangeFloatDtypeTests
'#ccc in the tests above we have tested that if type is int, it would keep it int'


# kkk not sure does adding commonkwargs1, increases or decreases readability and more important focus on main details
class TestTsRowFetcher_ChangeFloatDtypeTests(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=3, forecastLen=2)

    def testChangeFloatDtype_From64to32(self):
        inputData = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0],
                              [10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0],
                              [16.0, 17.0, 18.0]], dtype=np.float64)
        result = self.fetcher.getBackForeCastData_general(inputData, idx=1, **commonkwargs1)

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (3, 3))
        expectedResult = torch.tensor([[4.0, 5.0, 6.0],
                                       [7.0, 8.0, 9.0],
                                       [10.0, 11.0, 12.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(result, expectedResult))


# ----        TestTsRowFetcher_ShorterLen
class TestTsRowFetcher_ShorterLen(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=8, forecastLen=2)
        self.df = dfSample1.copy()

    def test_NoError_Df(self):
        self.fetcher.getBackForeCastData_general(self.df, 130, **commonkwargs2)

    def test_Error_Df(self):
        with self.assertRaises(ValueError) as context:
            self.fetcher.getBackForeCastData_general(self.df, 131, **commonkwargs2)
        self.assertTrue(_TsRowFetcher.errMsgs['shorterLen'], str(context.exception))

    def test_ShorterLenAllowed_Df(self):
        res = self.fetcher.getBackForeCastData_general(self.df, 131, mode='backcast',
                                                       colsOrIndexes=['y1'], outputTensor=False,
                                                       canHaveShorterLength=True)
        expectedResult = pd.DataFrame({'y1': [2, 3, 4, 5, 6, 7, 8]},
                                      index=[131, 132, 133, 134, 135, 136, 137])
        self.assertTrue(res.equals(expectedResult))

    def test_NoError_NpDict(self):
        self.fetcher.getBackForeCastData_general(NpDict(self.df), 0, mode='backcast',
                                                 colsOrIndexes=['y1', 'y2'], outputTensor=False)
        # kkk may have added the result check(maybe not needed ofc, done in other tests)

    def test_Error_NpDict(self):
        with self.assertRaises(ValueError) as context:
            self.fetcher.getBackForeCastData_general(NpDict(self.df), 1, **commonkwargs2)
        self.assertTrue(_TsRowFetcher.errMsgs['shorterLen'], str(context.exception))

    def test_ShorterLenAllowed_NpDict(self):
        res = self.fetcher.getBackForeCastData_general(NpDict(self.df), 1, mode='backcast',
                                                       colsOrIndexes=['y1'], outputTensor=False,
                                                       canHaveShorterLength=True)
        self.equalArrays(res, np.array([2, 3, 4, 5, 6, 7, 8]), checkType=False)


# ----        TestTsRowFetcher_RightPad
class TestTsRowFetcher_RightPad(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=8, forecastLen=2)
        self.df = dfSample1.copy()

    def test_rightPad_Df(self):
        res = self.fetcher.getBackForeCastData_general(self.df, 134, **commonkwargs2,
                                                       shiftForward=2, rightPadIfShorter=True)
        dfRightPadExpectedRes = pd.DataFrame({'y1': [7, 8, 0, 0, 0, 0, 0, 0],
                                              'y2': [22, 23, 0, 0, 0, 0, 0, 0]},
                                             index=[136, 137, 138, 139, 140, 141, 142, 143])
        self.assertTrue(res.equals(dfRightPadExpectedRes))

    def test_noRightPad_Df(self):
        res = self.fetcher.getBackForeCastData_general(self.df, 130, **commonkwargs2,
                                                       shiftForward=0, rightPadIfShorter=True)
        dfRightPadExpectedRes = dfSample1[['y1', 'y2']].copy()
        self.assertTrue(res.equals(dfRightPadExpectedRes))

    def test_rightPad_NpArray(self):
        res = self.fetcher.getBackForeCastData_general(self.df.values, 4, mode='backcast',
                                                       colsOrIndexes=[0, 2], outputTensor=False,
                                                       shiftForward=2, rightPadIfShorter=True)
        self.equalArrays(res, np.array(
            [[7, 38], [8, 39], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]), checkType=False)

    def test_noRightPad_NpArray(self):
        res = self.fetcher.getBackForeCastData_general(self.df.values, 0, mode='backcast',
                                                       colsOrIndexes=[0, 2], outputTensor=False,
                                                       shiftForward=0, rightPadIfShorter=True)
        self.equalArrays(res, np.array(
            [[1, 32], [2, 33], [3, 34], [4, 35], [5, 36], [6, 37], [7, 38], [8, 39]]),
                         checkType=False)

    def test_rightPad_Tensor(self):
        res = self.fetcher.getBackForeCastData_general(torch.tensor(self.df.values), 4,
                                                       mode='backcast',
                                                       colsOrIndexes=[0, 2], shiftForward=2,
                                                       outputTensor=True, rightPadIfShorter=True)
        expectedResult = torch.tensor(
            [[7, 38], [8, 39], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.int64)
        self.assertTrue(torch.equal(res, expectedResult))

    def test_noRightPad_Tensor(self):
        res = self.fetcher.getBackForeCastData_general(torch.tensor(self.df.values), 0,
                                                       mode='backcast',
                                                       colsOrIndexes=[0, 2], shiftForward=0,
                                                       outputTensor=True, rightPadIfShorter=True)
        expectedResult = torch.tensor(
            [[1, 32], [2, 33], [3, 34], [4, 35], [5, 36], [6, 37], [7, 38], [8, 39]],
            dtype=torch.int64)
        self.assertTrue(torch.equal(res, expectedResult))


# ----        TestTsRowFetcher_OutOfDataError
'#ccc this is different than, out of dataset indexes. this validate is out of this data(df, npArray,...) indexes or not'


class TestTsRowFetcher_OutOfDataError(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=8, forecastLen=2)
        self.df = dfSample1.copy()

    def test_Df(self):
        with self.assertRaises(ValueError) as context:
            self.fetcher.getBackForeCastData_general(self.df, 129, **commonkwargs2)
        self.assertTrue(_TsRowFetcher.errMsgs['non-negStartingPointDf'], str(context.exception))

    def test_NpDict(self):
        with self.assertRaises(ValueError) as context:
            self.fetcher.getBackForeCastData_general(NpDict(self.df), -1, **commonkwargs2)
        self.assertTrue(_TsRowFetcher.errMsgs['non-negStartingPointNpDict'], str(context.exception))

    def test_NpArray(self):
        with self.assertRaises(ValueError) as context:
            self.fetcher.getBackForeCastData_general(self.df.values, -1, mode='backcast',
                                                     colsOrIndexes=[0, 2], outputTensor=False)
        self.assertTrue(_TsRowFetcher.errMsgs['non-negStartingPointNpArray'],
                        str(context.exception))

    def test_Tensor(self):
        with self.assertRaises(ValueError) as context:
            self.fetcher.getBackForeCastData_general(torch.tensor(self.df.values), -1,
                                                     mode='backcast',
                                                     colsOrIndexes=[0, 2], outputTensor=False)
        self.assertTrue(_TsRowFetcher.errMsgs['non-negStartingPointTensor'], str(context.exception))


# ----        TestTsRowFetcher_SingleFeatureShapeCorrectionTests
class TestTsRowFetcher_SingleFeatureShapeCorrectionTests(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=3, forecastLen=2)

    def testSingleFeatureShapeCorrection(self):
        inputData = torch.tensor(
            [[[[4], [7], [6]], [[4], [8], [3]], [[1], [3], [9]]],
             [[[6], [3], [1]], [[1], [9], [7]], [[2], [3], [1]]],
             [[[10], [6], [1]], [[3], [2], [9]], [[10], [9], [3]]]])

        result = self.fetcher.singleFeatureShapeCorrection(inputData)
        self.assertEqual(result.shape, torch.Size([3, 3, 3]))


# ----        TestTsRowFetcher_Shift
class TestTsRowFetcher_Shift(BaseTestClass):
    def setUp(self):
        self.fetcher = _TsRowFetcher(backcastLen=2, forecastLen=3)
        self.df = dfSample1.copy()

    def test_posShift(self):
        res = self.fetcher.getBackForeCastData_general(self.df, 131, **commonkwargs3,
                                                       shiftForward=2, rightPadIfShorter=False)
        expectedResult = pd.DataFrame({'y1': [6, 7, 8],
                                       'y2': [21, 22, 23]},
                                      index=range(135, 138))
        self.assertTrue(res.equals(expectedResult))

    def test_negShift(self):
        res = self.fetcher.getBackForeCastData_general(self.df, 134, **commonkwargs3,
                                                       shiftForward=-2, rightPadIfShorter=False)
        expectedResult = pd.DataFrame({'y1': [5, 6, 7], 'y2': [20, 21, 22]}, index=range(134, 137))
        self.assertTrue(res.equals(expectedResult))


# ----     TestDataset_noNSeries_InDatasetIndexes
class TestDataset_noNSeries_InDatasetIndexes(BaseTestClass):
    def setUp(self):
        self.df = dfSample1.copy()
        self.df[tsStartPointColName] = 4 * [True] + 4 * [False]
        self.dataset = VAnnTsDataset(self.df, backcastLen=2, forecastLen=3, useNpDictForDfs=False)

    def test_notInDatasetIndexes(self):
        with self.assertRaises(ValueError) as context:
            self.dataset.getBackForeCastData(134, **commonkwargs3,
                                             shiftForward=0, rightPadIfShorter=False)
        self.assertTrue("134 is not in indexes", str(context.exception))

    def test_withShift_notInDatasetIndexes(self):
        with self.assertRaises(ValueError) as context:
            self.dataset.getBackForeCastData(137, **commonkwargs3,
                                             shiftForward=-2, rightPadIfShorter=False)
        self.assertTrue("135 is not in indexes", str(context.exception))


# ---- run test
if __name__ == '__main__':
    unittest.main()
