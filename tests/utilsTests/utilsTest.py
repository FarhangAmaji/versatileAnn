import unittest
from typing import List, Tuple, Union

from tests.baseTest import BaseTestClass
from utils.typeCheck import typeHintChecker_AListOfSomeType


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


# ---- run test
if __name__ == '__main__':
    unittest.main()
