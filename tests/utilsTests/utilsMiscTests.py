import os
import unittest

from dataPrep.normalizers.mainGroupNormalizers import MainGroupSingleColStdNormalizer
from projectUtils.misc import getProjectDirectory, findClassDefinition_inADirectory, \
    getClassObjectFromFile, getStaticmethod_actualClass, isFunctionOrMethod, \
    getActualClassFromMethod, giveOnlyKwargsRelated_toMethod
from tests.baseTest import BaseTestClass
from tests.utilsTests.dummyForTest import DummyClassFor_Test_isFunctionOrMethod, \
    dummyRegularFunctionFor_isFunctionOrMethod


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


# ----
class TestGiveOnlyKwargsRelatedToMethod(BaseTestClass):

    def test_basic(self):
        # Test case where method takes simple kwargs
        def method(arg1, arg2, arg3):
            pass

        updater = {'arg1': 'value1', 'arg2': 'value2', 'arg4': 'value4'}
        result, notRelatedKeys = giveOnlyKwargsRelated_toMethod(method, updater)

        self.assertEqual(result, {'arg1': 'value1', 'arg2': 'value2'})
        self.assertEqual(notRelatedKeys, ['arg4'])

    def test_snakeCaseCompatibility(self):
        # Test case where method takes kwargs with snake_case and updater has camelCase
        def method(my_arg, yourArg):
            return {'my_arg': my_arg, 'yourArg': yourArg}

        updater = {'myArg': 'value1', 'yourArg': 'value2'}
        result, notRelatedKeys = giveOnlyKwargsRelated_toMethod(method, updater)

        self.assertEqual(result, {'my_arg': 'value1', 'yourArg': 'value2'})
        self.assertEqual(notRelatedKeys, [])

    def test_delAfter(self):
        # Test case where delAfter=True deletes keys from updater after updating updatee
        def method(arg1, arg2):
            pass

        updater = {'arg1': 'value1', 'arg2': 'value2'}
        result, notRelatedKeys = giveOnlyKwargsRelated_toMethod(method, updater, delAfter=True)

        self.assertEqual(result, {'arg1': 'value1', 'arg2': 'value2'})
        self.assertEqual(updater, {})  # updater should be empty after deletion
        self.assertEqual(notRelatedKeys, [])

    def test_invalidMethod(self):
        # Test case where an invalid method (non-callable) is passed
        updater = {'arg1': 'value1'}
        with self.assertRaises(ValueError):
            giveOnlyKwargsRelated_toMethod('invalid_method', updater)

    def test_emptyUpdatee(self):
        # Test case where updatee is initially empty
        def method(arg1):
            return {'arg1': arg1}

        updater = {'arg1': 'value1'}
        result, notRelatedKeys = giveOnlyKwargsRelated_toMethod(method, updater, updatee={})

        self.assertEqual(result, {'arg1': 'value1'})
        self.assertEqual(notRelatedKeys, [])


# ----
class Test_isFunctionOrMethod(BaseTestClass):

    def test_staticMethod(self):
        result, typeName = isFunctionOrMethod(DummyClassFor_Test_isFunctionOrMethod.staticMethod)
        self.assertTrue(result)
        self.assertEqual(typeName, "Static Method")

    def test_classMethod(self):
        result, typeName = isFunctionOrMethod(DummyClassFor_Test_isFunctionOrMethod.classMethod)
        self.assertTrue(result)
        self.assertEqual(typeName, "Class Method")

    def test_instanceMethod(self):
        result, typeName = isFunctionOrMethod(
            DummyClassFor_Test_isFunctionOrMethod().instanceMethod)
        self.assertTrue(result)
        self.assertEqual(typeName, "Instance Method")

    def test_privateMethod(self):
        result, typeName = isFunctionOrMethod(
            DummyClassFor_Test_isFunctionOrMethod()._privateMethod)
        self.assertTrue(result)
        self.assertEqual(typeName, "Instance Method")

    def test_magicMethod(self):
        result, typeName = isFunctionOrMethod(
            DummyClassFor_Test_isFunctionOrMethod()._DummyClassFor_Test_isFunctionOrMethod__magicMethod)
        self.assertTrue(result)
        self.assertEqual(typeName, "Instance Method")

    def test_regularFunction(self):
        result, typeName = isFunctionOrMethod(dummyRegularFunctionFor_isFunctionOrMethod)
        self.assertTrue(result)
        self.assertEqual(typeName, "Function")

    def test_regularFunction_insideTheLocal(self):
        # ccc1
        #  right now this is supposed to have error until the isFunctionOrMethod is fixed
        def regularFunction():
            pass

        result, typeName = isFunctionOrMethod(regularFunction)
        self.assertTrue(result)
        self.assertEqual(typeName, "Function")

    def test_notFunctionOrMethod(self):
        result, typeName = isFunctionOrMethod(123)
        self.assertFalse(result)
        self.assertEqual(typeName, "not a method or a func")


class Test_getStaticmethod_actualClass(BaseTestClass):

    def test_getStaticmethod_actualClass_staticMethod(self):
        self.assertEqual(
            getStaticmethod_actualClass(DummyClassFor_Test_isFunctionOrMethod.staticMethod),
            DummyClassFor_Test_isFunctionOrMethod)

    def test_getStaticmethod_actualClass_classMethod(self):
        self.assertIsNone(
            getStaticmethod_actualClass(DummyClassFor_Test_isFunctionOrMethod.classMethod))

    def test_getStaticmethod_actualClass_instanceMethod(self):
        self.assertIsNone(
            getStaticmethod_actualClass(DummyClassFor_Test_isFunctionOrMethod().instanceMethod))


class Test_getActualClassFromMethod(BaseTestClass):

    def test_getActualClassFromMethod_staticMethod(self):
        self.assertEqual(
            getActualClassFromMethod(DummyClassFor_Test_isFunctionOrMethod.staticMethod),
            DummyClassFor_Test_isFunctionOrMethod)

    def test_getActualClassFromMethod_classMethod(self):
        self.assertEqual(
            getActualClassFromMethod(DummyClassFor_Test_isFunctionOrMethod.classMethod),
            DummyClassFor_Test_isFunctionOrMethod)

    def test_getActualClassFromMethod_instanceMethod(self):
        self.assertEqual(
            getActualClassFromMethod(DummyClassFor_Test_isFunctionOrMethod().instanceMethod),
            DummyClassFor_Test_isFunctionOrMethod)

    def test_getActualClassFromMethod_privateMethod(self):
        self.assertEqual(
            getActualClassFromMethod(DummyClassFor_Test_isFunctionOrMethod()._privateMethod),
            DummyClassFor_Test_isFunctionOrMethod)

    def test_getActualClassFromMethod_magicMethod(self):
        self.assertEqual(getActualClassFromMethod(
            DummyClassFor_Test_isFunctionOrMethod()._DummyClassFor_Test_isFunctionOrMethod__magicMethod),
            DummyClassFor_Test_isFunctionOrMethod)


# ---- run test
if __name__ == '__main__':
    unittest.main()
