import os
import unittest

from dataPrep.normalizers.mainGroupNormalizers import MainGroupSingleColStdNormalizer
from projectUtils.misc import getProjectDirectory, findClassDefinition_inADirectory, \
    getClassObjectFromFile
from tests.baseTest import BaseTestClass


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
