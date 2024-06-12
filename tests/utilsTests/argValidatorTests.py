import unittest
from typing import List, Tuple, Union

import pandas as pd
import pydantic

from projectUtils.initParentClasses import exclude_selfNArgsNKwargs_fromAllArgs
from projectUtils.typeCheck import typeHintChecker_AListOfSomeType, argValidator
from tests.baseTest import BaseTestClass


class TypeHintChecker_AListOfSomeType_Test(BaseTestClass):

    @typeHintChecker_AListOfSomeType
    def funcWithHints(self, a1: List[str], a2: List[int], a3: List[Tuple],
                      a4, a5: str, a6: Tuple, a7,
                      a8: List[int], a9: List[Union[str, int]], a10: List[Union[str, int]],
                      a11: int, a12: list):
        pass

    def testCorrectInput(self):
        self.funcWithHints(['a', 'b'], [1, 2, 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6), 'other',
                           [3, 4], ['fd', 4], ['funcWithHints', 41], 11, [3, 's'])

    def testIncorrectA2Type(self):
        with self.assertRaises(TypeError):
            self.funcWithHints(['a', 'b'], [1, '2', 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6),
                               'other', [3, 4], ['fd', 4], ['funcWithHints', 41], 11, [3, 's'])

    def testIncorrectA10Type(self):
        with self.assertRaises(TypeError):
            self.funcWithHints(['a', 'b'], [1, 2, 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6), 123,
                               [3, 4], ['fd', 4], ['funcWithHints', 41, (3,)], 11, [3, 's'])

    def testA10HintIs_ListWithUnion_butAllItemsAreOneType(self):
        self.funcWithHints(['a', 'b'], [1, 2, 3], [(1, 2), (3, 4)], 42, 'hello', (5, 6), 123,
                           [3, 4], ['fd', 4], ['funcWithHints', '41'], 11, [3, 's'])

    def test_singleDictArg(self):
        res = exclude_selfNArgsNKwargs_fromAllArgs({'a': 3, 'b': 4})
        self.assertEqual(res, {'a': 3, 'b': 4})


class TypeHintChecker_AListOfSomeType_argValidator_Test(TypeHintChecker_AListOfSomeType_Test):
    @argValidator
    def funcWithHints(self, a1: List[str], a2: List[int], a3: List[Tuple], a4, a5: str, a6: Tuple,
                      a7, a8: List[int], a9: List[Union[str, int]], a10: List[Union[str, int]],
                      a11: int, a12: list):
        pass

    @argValidator
    def funcWithHints2_withDefaultVals(self, a1: List[str], a2: List[int] = [4, 3]):
        return [a1, a2]

    def test_defaultVal(self):
        self.assertEqual([['1', '2'], [4, 3]], self.funcWithHints2_withDefaultVals(['1', '2']))

    @argValidator
    def funcWithHints3_withArgs(self, a1: List[str], *a2):
        return [a1, *a2]

    def test_Args(self):
        self.funcWithHints3_withArgs(['a'], 3, 'f')

    @argValidator
    def funcWithHints4_withArgsWithHint(self, a1: List[str], *a2: int):
        return [a1, *a2]

    def test_ArgsWithHint(self):
        self.funcWithHints4_withArgsWithHint(['a'], 3, 4)

    def test_ArgsWithHintError(self):
        with self.assertRaises(TypeError):
            self.funcWithHints4_withArgsWithHint(['a'], 3, 'f')


class ArgValidatorTest(BaseTestClass):
    @argValidator
    def oneType(self, a1: pd.DataFrame):
        pass

    def testOneType(self):
        self.oneType(pd.DataFrame({'a': [1], 'b': [2]}))

    def testOneTypeError(self):
        with self.assertRaises(pydantic.error_wrappers.ValidationError):
            self.oneType([1, 2])

    @argValidator
    def pandasUnion(self, a1: Union[pd.DataFrame, pd.Series]):
        pass

    def testUnion(self):
        self.pandasUnion(pd.DataFrame({'a': [1], 'b': [2]}))

    def testUnionError(self):
        with self.assertRaises(pydantic.error_wrappers.ValidationError):
            self.pandasUnion([1, 2])

    @argValidator
    def oneTypeInt(self, a1: int):
        pass

    def testOneTypeInt(self):
        self.oneTypeInt(3)

    def testOneTypeInt_notGivingError_whenPassingIntergableString(self):
        # ccc1
        #  note this doesn't give error with pydantic
        self.oneTypeInt('3')

    def testOneTypeIntError(self):
        with self.assertRaises(pydantic.error_wrappers.ValidationError):
            self.oneTypeInt('s')


# ---- run test
if __name__ == '__main__':
    unittest.main()
