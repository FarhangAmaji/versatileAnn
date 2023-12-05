import unittest
from typing import List, Tuple, Union

import torch

from tests.baseTest import BaseTestClass
from utils.typeCheck import typeHintChecker_AListOfSomeType
from utils.vAnnGeneralUtils import equalTensors, DotDict


class DotDictTests(BaseTestClass):
    def setUp(self) -> None:
        self.dictData = {'a': 1, 'b': 2}
        self.dotDict = DotDict(self.dictData)

    def testInitialization(self):
        self.assertEqual(self.dotDict.dict, self.dictData)
        self.assertEqual(self.dotDict._data, self.dictData)

    def testGettingAttribute(self):
        # cccAlgo Getting means not using .get
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
        # cccAlgo testGet is for .get
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


class typeHintChecker_AListOfSomeType_Test(BaseTestClass):
    @typeHintChecker_AListOfSomeType
    def funcWithHints(self, a1: List[str], a2: List[int], a3: List[Tuple],
                      a4, a5: str, a6: Tuple, a7,
                      a8: List[int], a9: List[Union[str, int]], a10: List[Union[str, int]],
                      a11: int, a12: list):
        pass

    def testCorrectInput(self):
        self.funcWithHints(['a', 'b'], [1, 2, 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6), 'other',
                           [3, 4], ['fd', 4], ['funcWithHints', 41], [3, 's'], a11=11)

    def testIncorrectA2Type(self):
        with self.assertRaises(TypeError):
            self.funcWithHints(['a', 'b'], [1, '2', 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6),
                               'other', [3, 4], ['fd', 4], ['funcWithHints', 41], 11, [3, 's'])

    def testIncorrectA10Type(self):
        with self.assertRaises(TypeError):
            self.funcWithHints(['a', 'b'], [1, '2', 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6), 123,
                               [3, 4], ['fd', 4], ['funcWithHints', 41, (3)], 11, [3, 's'])


class typeHintChecker_AListOfSomeType_argValidator_Test(typeHintChecker_AListOfSomeType_Test):
    @typeHintChecker_AListOfSomeType
    def funcWithHints(self, a1: List[str], a2: List[int], a3: List[Tuple], a4, a5: str, a6: Tuple,
                      a7, a8: List[int], a9: List[Union[str, int]], a10: List[Union[str, int]],
                      a11: int, a12: list):
        pass


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
        tensor1 = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
        tensor2 = torch.tensor([1.0, 2.0, 3.0], device='cpu')
        result = equalTensors(tensor1, tensor2)
        self.assertFalse(result)

    def testDifferentDevice_withCheckDeviceFalse(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
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


# ---- run test
if __name__ == '__main__':
    unittest.main()
