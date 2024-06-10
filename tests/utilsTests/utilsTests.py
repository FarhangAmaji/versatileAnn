import os
import unittest
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from dataPrep.normalizers.mainGroupNormalizers import MainGroupSingleColStdNormalizer
from projectUtils.dataTypeUtils.dotDict_npDict import DotDict, NpDict
from projectUtils.dataTypeUtils.str import snakeToCamel, camelToSnake
from projectUtils.dataTypeUtils.tensor import equalTensors, getDefaultTorchDevice_name
from projectUtils.initParentClasses import exclude_selfNArgsNKwargs_fromAllArgs
from projectUtils.misc import getProjectDirectory, findClassDefinition_inADirectory, \
    getClassObjectFromFile
from projectUtils.typeCheck import typeHintChecker_AListOfSomeType, argValidator
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


class typeHintChecker_AListOfSomeType_Test(BaseTestClass):

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


class typeHintChecker_AListOfSomeType_argValidator_Test(typeHintChecker_AListOfSomeType_Test):
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


# ----
class FindClassDefinitionTests(BaseTestClass):
    def testExistingClass(self):
        res = findClassDefinition_inADirectory(getProjectDirectory(),
                                               'MainGroupSingleColStdNormalizer')
        expectedPath = os.path.join(getProjectDirectory(), 'dataPrep', 'normalizers',
                                    'mainGroupNormalizers.py')
        self.assertEqual(res['filePaths'][0], expectedPath)
        expectedDef = """class MainGroupSingleColStdNormalizer(_MainGroupSingleColNormalizer):

    def __init__(self, df, mainGroupColNames, colNames: list):

        super().__init__(SingleColStdNormalizer, df, mainGroupColNames,

                         colNames)



    @argValidator

    def setMeanNStd_ofMainGroups(self, df: pd.DataFrame):

        self._warnToInverseTransform_mainGroups(df)

        # ccc1

        #  for each col, makes f'{col}Mean' and f'{col}Std'

        #  note setMeanNStd_ofMainGroups needs to have unTransformed mainGroups. so if needed,

        #  inverseTransform them and transform them again after applying this func

        for col in self.colNames:

            for _, combo in self.uniqueCombos.items():

                dfToFit = self.getRowsByCombination(df, combo)

                inds = dfToFit.index

                scaler = self.container[col][combo.shortRepr()].encoders[col].scaler

                comboMean = scaler.mean_[0]

                comboStd = scaler.scale_[0]

                df.loc[inds, f'{col}Mean'] = comboMean

                df.loc[inds, f'{col}Std'] = comboStd



    def __repr__(self):

        className = type(self).__name__

        return f"{className}:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"
"""
        self.assertEqual(res['Definitions'][0], expectedDef)

    def testNonExistingClass(self):
        res = findClassDefinition_inADirectory(getProjectDirectory(), 'qqBangBang')
        self.assertEqual(res, {'className': 'qqBangBang', 'Definitions': [], 'filePaths': []})

    def testExistingClass_inMultiplePlaces(self):
        res = findClassDefinition_inADirectory(getProjectDirectory(),
                                               'NNDummyFor_findClassDefinition_inADirectoryTest')
        expectedPaths = [os.path.join(getProjectDirectory(), 'tests', 'utilsTests',
                                      'dummyForTest.py'),
                         os.path.join(getProjectDirectory(), 'tests', 'utilsTests',
                                      'dummyForTest2.py')]
        self.assertEqual(res['filePaths'], expectedPaths)
        def1 = "class NNDummyFor_findClassDefinition_inADirectoryTest:\n\n    def __init__(self):\n\n        self.ke = 78\n\n\n\n    @staticmethod\n\n    def static_Method1():\n\n        print('staticmethod for NNDummyModule1')\n\n\n\n    def instanceMeth1(self):\n\n        print('instancemethod for NNDummyModule1')\n"
        self.assertEqual(res['Definitions'][0], def1)
        self.assertEqual(res['Definitions'][1], def1)


class getClassObjectFromFileTest(BaseTestClass):
    def test(self):
        res = findClassDefinition_inADirectory(getProjectDirectory(),
                                               'MainGroupSingleColStdNormalizer')
        classObj = getClassObjectFromFile('MainGroupSingleColStdNormalizer', res['filePaths'][0])

        self.assertEqual(classObj.__name__, MainGroupSingleColStdNormalizer.__name__)
        self.assertEqual(dir(classObj), dir(MainGroupSingleColStdNormalizer))


# ---- run test
if __name__ == '__main__':
    unittest.main()
