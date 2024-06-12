import unittest

import numpy as np
import pandas as pd
import torch

from projectUtils.dataTypeUtils.dotDict_npDict import DotDict, NpDict
from projectUtils.dataTypeUtils.str import snakeToCamel, camelToSnake
from projectUtils.dataTypeUtils.tensor import equalTensors, getDefaultTorchDevice_name
from tests.baseTest import BaseTestClass


class DotDictTests(BaseTestClass):
    def setUp(self) -> None:
        self.dictData = {'a': 1, 'b': 2}
        self.dotDict = DotDict(self.dictData)

    def testInitialization(self):
        self.assertEqual(self.dotDict.dict, self.dictData)
        self.assertEqual(self.dotDict._data, self.dictData)

    def testGettingAttribute(self):
        # ccc1 Getting means not using .get
        self.assertEqual(self.dotDict.a, 1)
        self.assertEqual(self.dotDict.b, 2)

    def testGettingNonexistentAttribute(self):
        with self.assertRaises(AttributeError):
            value = self.dotDict.c

    def testGettingItem(self):
        self.assertEqual(self.dotDict['a'], 1)
        self.assertEqual(self.dotDict['b'], 2)

    def testGettingNonexistentKey(self):
        with self.assertRaises(KeyError):
            value = self.dotDict['c']

    def testSetItem(self):
        self.dotDict['c'] = 3
        self.assertEqual(self.dotDict['c'], 3)

    def testLen(self):
        dotDict = DotDict({'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(len(dotDict), 3)

    def testIteration(self):
        for key, value in self.dotDict:
            self.assertEqual(self.dictData[key], value)

    def testRepr(self):
        self.assertEqual(repr(self.dotDict), f'DotDict: {self.dictData}')

    # ---- .get
    def testGetExistingKey_withGet(self):
        # ccc1 testGet is for .get
        value_a = self.dotDict.get('a', 0)
        self.assertEqual(value_a, 1)

    def testGetNonexistentKey_withDefault_withGet(self):
        value_c_default = self.dotDict.get('c', 0)
        self.assertEqual(value_c_default, 0)

    def testGetNonexistentKey_withoutDefault_withGet(self):
        value_c_noDefault = self.dotDict.get('c')
        self.assertIsNone(value_c_noDefault)

    # ---- .setDefault
    def testSetDefault_existingKey(self):
        value_a_existing = self.dotDict.setDefault('a', 0)
        self.assertEqual(value_a_existing, 1)

    def testSetDefault_nonExistingKey(self):
        value_c_default = self.dotDict.setDefault('c', 0)
        self.assertEqual(value_c_default, 0)

    def testSetDefault_nonExistingKey_withoutDefaultValue(self):
        value_c_noDefault = self.dotDict.setDefault('c')
        self.assertIsNone(value_c_noDefault)


class NpDictTests(BaseTestClass):

    def setUp(self):
        self.dfData = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        self.df = pd.DataFrame(self.dfData)
        self.npDict = NpDict(self.df)

    def testInitialization_fromDataFrame(self):
        self.assertEqual(self.npDict.shape, self.df.shape)
        self.assertEqual(set(self.npDict.cols()), set(self.df.columns))

    def testInitialization_fromDictionary(self):
        npDict_fromDict = NpDict(self.dfData)
        self.assertEqual(npDict_fromDict.shape, self.df.shape)
        self.assertEqual(set(npDict_fromDict.cols()), set(self.df.columns))

    def testGetDict(self):
        for col in self.dfData:
            self.equalArrays(self.npDict.getDict()[col], self.df[col])

    def testPrintDict(self):
        expectedPrint = "{'a': [1, 2, 3],\n" + "'b': [4, 5, 6]}"

        def innerFunc():
            self.npDict.printDict()

        self.assertPrint(innerFunc, expectedPrint)

    def testToDf(self):
        dfFromNpDict = self.npDict.toDf()
        self.equalDfs(dfFromNpDict, self.df)

    def testToDf_resetDtype(self):
        npDict = NpDict({'a': ['s', 1, 2, 3]})
        self.assertEqual(npDict['a'].dtype, np.object_)
        npDict2 = NpDict({'a': npDict['a'][1:]})
        self.assertEqual(npDict2['a'].dtype, np.object_)
        self.assertEqual(npDict2.toDf(resetDtype=True)['a'].dtype, np.int64)

    def testDfProperty(self):
        dfProperty = self.npDict.df
        self.equalDfs(dfProperty, self.df)

    def testGetItem_singleCol(self):
        column_a = self.npDict['a']
        self.equalArrays(column_a, np.array([1, 2, 3]), checkType=False)

    def testGetItem_multipleCols(self):
        selectedColumns = self.npDict[['a', 'b']]
        self.equalArrays(selectedColumns, np.array([[1, 4], [2, 5], [3, 6]]), checkType=False)

    def testGetItem_allColsSlice(self):
        self.equalArrays(self.npDict[:], np.array([[1, 4], [2, 5], [3, 6]]), checkType=False)

    def testGetItem_raiseErrorWithSlice(self):
        with self.assertRaises(ValueError):
            invalid_slice = self.npDict[1:3]  # Slicing other than [:] is not allowed

    def testLen(self):
        self.assertEqual(len(self.npDict), len(self.df))

    def testRepr(self):
        reprString = repr(self.npDict)
        expectedRepr = '   a  b\n0  1  4\n1  2  5\n2  3  6'
        self.assertEqual(reprString, expectedRepr)

    def testSetItemDisabled(self):
        with self.assertRaises(ValueError):
            self.npDict['new_column'] = [7, 8, 9]  # __setitem__ is disabled


class equalTensorsTests(BaseTestClass):
    def testDifferentFloatTypes(self):
        pass

    def testSameValues(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([1.0, 2.0, 3.0])
        result = equalTensors(tensor1, tensor2)
        self.assertTrue(result)

    def testDifferentValues(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([1.1, 2.2, 3.3])
        result = equalTensors(tensor1, tensor2)
        self.assertFalse(result)

    def testDifferentDtype(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = equalTensors(tensor1, tensor2)
        self.assertFalse(result)

    def testDifferentDtype_withCheckTypeFalse(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = equalTensors(tensor1, tensor2, checkType=False)
        self.assertTrue(result)

    def testDifferentDtypeInt_withCheckTypeFalse(self):
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int64)
        tensor2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = equalTensors(tensor1, tensor2, checkType=False)
        self.assertTrue(result)

    def testFloatApproxDifferentDtypes(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor2 = torch.tensor([1.0001, 2.0001, 3.0001], dtype=torch.float64)
        result = equalTensors(tensor1, tensor2, floatApprox=True, floatPrecision=1e-3,
                              checkType=False)
        self.assertTrue(result)

    def testDifferentDevice(self):
        deviceName = getDefaultTorchDevice_name()
        # so if there is no other device available this test can't be done
        if deviceName == 'cpu':
            return

        tensor1 = torch.tensor([1.0, 2.0, 3.0], device=deviceName)
        tensor2 = torch.tensor([1.0, 2.0, 3.0], device='cpu')
        result = equalTensors(tensor1, tensor2)
        self.assertFalse(result)

    def testDifferentDevice_withCheckDeviceFalse(self):
        deviceName = getDefaultTorchDevice_name()
        # so if there is no other device available this test can't be done
        if deviceName == 'cpu':
            return

        tensor1 = torch.tensor([1.0, 2.0, 3.0], device=deviceName)
        tensor2 = torch.tensor([1.0, 2.0, 3.0], device='cpu')
        result = equalTensors(tensor1, tensor2, checkDevice=False)
        self.assertTrue(result)

    def testFloatApprox(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([1.0001, 2.0001, 3.0001])
        result = equalTensors(tensor1, tensor2, floatApprox=True, floatPrecision=1e-3)
        self.assertTrue(result)

    def testFloatApproxNonFloatTensors(self):
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([1, 2, 3])
        result = equalTensors(tensor1, tensor2, floatApprox=True, floatPrecision=1e-3)
        self.assertTrue(result)


class CaseChangeTests(BaseTestClass):
    def testSnakeToCamel(self):
        snakeString = "example_snake_case_string"
        camelCaseResult = snakeToCamel(snakeString)
        self.assertEqual(camelCaseResult, "exampleSnakeCaseString")

    def testCamelToSnake(self):
        camelString = "exampleCamelCaseString"
        snakeCaseResult = camelToSnake(camelString)
        self.assertEqual(snakeCaseResult, "example_camel_case_string")


# ---- run test
if __name__ == '__main__':
    unittest.main()
