import unittest
from typing import List, Tuple, Union

import torch

from tests.baseTest import BaseTestClass
from utils.typeCheck import typeHintChecker_AListOfSomeType
from utils.vAnnGeneralUtils import equalTensors


# ----
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
