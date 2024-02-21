import pandas as pd

from dataPrep.normalizers.baseNormalizer import _BaseNormalizer
from dataPrep.normalizers.normalizers_singleColsNormalizer import SingleColsStdNormalizer, \
    SingleColsLblEncoder
from projectUtils.dataTypeUtils.df_series import pandasGroupbyAlternative
from projectUtils.dataTypeUtils.dotDict_npDict import NpDict
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


# ---- normalizers: MainGroupNormalizers
class _Combo:
    def __init__(self, defDict, mainGroupColNames):
        for key in defDict:
            if key not in mainGroupColNames:
                raise ValueError(f"'{key}' is not a valid column name in mainGroupColNames.")

        for col in mainGroupColNames:
            if col not in defDict:
                raise ValueError(f"'{col}' is missing in combo definition.")

        self.defDict = defDict

    def shortRepr(self):
        return '_'.join([str(item) for item in self.defDict.values()])

    def __repr__(self):
        return str(self.defDict)


class _MainGroupBaseNormalizer:
    def __init__(self, df, mainGroupColNames, internalCall=False):
        if not internalCall:
            _allowOnlyCreationOf_ChildrenInstances(self, _MainGroupSingleColsNormalizer)
        self.mainGroupColNames = mainGroupColNames
        self.uniqueCombos = self._getUniqueCombinations(df)
        self._getMainGroupUniqueVals(df, mainGroupColNames)

    def _getMainGroupUniqueVals(self, df, mainGroupColNames):
        self._mainGroupUniqueVals = {}
        for mainGroup in mainGroupColNames:
            self._mainGroupUniqueVals[mainGroup] = list(df[mainGroup].unique())

    def uniqueCombosShortReprs(self):
        return [combo.shortRepr() for combo in self.uniqueCombos]

    def findMatchingCombo_shortRepr(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.shortRepr():
                return uniqueCombo
        return None

    @property
    def uniqueCombos_dictReprs(self):
        return [combo.defDict for combo in self.uniqueCombos]

    @argValidator
    def findMatchingCombo_dictRepr(self, combo: dict):
        return self.uniqueCombos.get(str(combo), None)

    def isComboInUniqueCombos(self, combo):
        if isinstance(combo, _Combo):
            if combo.__repr__() in self.uniqueCombos.keys():
                return combo
        isInputComboStr_NRepresentsADict = isinstance(combo, str) and type(eval(combo)) == dict
        if isinstance(combo, str) and not isInputComboStr_NRepresentsADict:
            if self.findMatchingCombo_shortRepr(combo):
                return self.findMatchingCombo_shortRepr(combo)
        elif isinstance(combo, dict) or isInputComboStr_NRepresentsADict:
            return self.findMatchingCombo_dictRepr(combo)
        raise ValueError(f'no {str(combo)} is in uniqueCombos')

    def _getUniqueCombinations(self, df):
        comboObjs = {}

        for groupName, groupDf in pandasGroupbyAlternative(df, self.mainGroupColNames):
            comboDict = dict(zip(self.mainGroupColNames, groupName))
            combo = _Combo(comboDict, self.mainGroupColNames)
            comboObjs.update({f'{combo.__repr__()}': combo})

        return comboObjs

    def getRowsByCombination(self, df, combo):
        comboObj = self.isComboInUniqueCombos(combo)
        tempDf = df[
            (df[self.mainGroupColNames] == comboObj.defDict).all(axis=1)]

        # this is to correct dtypes
        npDict = NpDict(tempDf)
        tempDf = npDict.toDf(resetDtype=True)
        return tempDf


# ---- _MainGroupSingleColsNormalizer
class _MainGroupSingleColsNormalizer(_MainGroupBaseNormalizer,
                                     _BaseNormalizer):
    def __init__(self, classType, df, mainGroupColNames, colNames: list):
        _allowOnlyCreationOf_ChildrenInstances(self, _MainGroupSingleColsNormalizer)
        _MainGroupBaseNormalizer.__init__(self, df, mainGroupColNames, internalCall=True)
        self.colNames = colNames
        self.container = {}
        for col in colNames:
            self.container[col] = {}
            for _, combo in self.uniqueCombos.items():
                self.container[col][combo.shortRepr()] = classType([col])

    @argValidator
    def fitCol(self, df: pd.DataFrame, col):
        self._warnToInverseTransform_mainGroups(df)
        for _, combo in self.uniqueCombos.items():
            dfToFit = self.getRowsByCombination(df, combo)
            dfToFit = dfToFit.reset_index(drop=True)
            self.container[col][combo.shortRepr()].fit(dfToFit)

    def _warnToInverseTransform_mainGroups(self, df):
        # addTest2 maybe for other functionalities which use this
        # cccAlgo
        #  methods in this class need data of mainGroups as they were initialized,
        #  usually when mainGroups themselves also transformed, to use methods of this class u need
        #  to inverseTransform mainGroups and retransform after utilizing method wanted
        # goodToHave3
        #  maybe as possible this class may had taken stackNormalizer and had done inverseTransform
        #  and retransform it, automatically
        #  note check for danger of multiple flow between this class and stackNormalizer
        # cccDevAlgo #  addTest1 add test for this
        #  this is an old note on this problem
        #  """normalizer=NormalizerStack(SingleColsLblEncoder(['sku', 'month', 'agency', *specialDays]), MainGroupSingleColsStdNormalizer(df, mainGroups, target))
        #  normalizer.fitNTransform(df)"""
        #  this wont work because the unqiueCombos in MainGroupSingleColsStdNormalizer are determined first and after fitNTransform
        #  of SingleColsLblEncoder, values of mainGroups are changed
        #  kinda correct way right now: normalizer=NormalizerStack(MainGroupSingleColsStdNormalizer(df, mainGroups, target), SingleColsLblEncoder(['sku', 'agency', 'month', *specialDays]))
        #  - for this problem initing all normalizers in init of NormalizerStack doesnt seem to be a good solution
        #  - (should think about it much more) a good solution is that in fitNTransform of normalStack I do fit then transform and if the next uniqueNormalizer has this col in its _colNames or groupNames, undo and redo again
        mainGroupsNeededToInverseTransformed = []
        for mainGroup in self.mainGroupColNames:
            currentUniqueValues = df[mainGroup].unique()
            supposedUniqueValues = self._mainGroupUniqueVals[mainGroup]

            if any(value not in supposedUniqueValues for value in currentUniqueValues):
                mainGroupsNeededToInverseTransformed.append(mainGroup)
            else:
                if not all(value in supposedUniqueValues for value in currentUniqueValues):
                    Warn.warn(f'data distortion Warning: seems some data have been added to' +
                              ' original {mainGroup} column')
        if mainGroupsNeededToInverseTransformed:
            raise RuntimeError('it seems "' + ', '.join(mainGroupsNeededToInverseTransformed) +
                               '" need to be inverseTransformed. if you want them transformed' +
                               ' after using this method, retransform them back again')

    @argValidator
    def fit(self, df: pd.DataFrame):
        self._warnToInverseTransform_mainGroups(df)
        for col in self.colNames:
            self.fitCol(df, col)

    @argValidator
    def transformCol(self, df: pd.DataFrame, col):
        self._warnToInverseTransform_mainGroups(df)
        dfCopy = df.copy()
        for _, combo in self.uniqueCombos.items():
            dfToFit = self.getRowsByCombination(df, combo)
            inds = dfToFit.index
            dfToFit = dfToFit.reset_index(drop=True)
            self.container[col][combo.shortRepr()].transform(dfToFit)
            dfToFit.index = inds
            dfCopy.loc[inds, col] = dfToFit
        return dfCopy[col]

    @argValidator
    def transform(self, df: pd.DataFrame):
        self._warnToInverseTransform_mainGroups(df)
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    @argValidator
    def fitNTransformCol(self, df: pd.DataFrame):
        self._warnToInverseTransform_mainGroups(df)
        self.fitCol(df)
        self.transformCol(df)

    @argValidator
    def fitNTransform(self, df: pd.DataFrame):
        self._warnToInverseTransform_mainGroups(df)
        self.fit(df)
        self.transform(df)

    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        self._warnToInverseTransform_mainGroups(df)
        dfCopy = df.copy()
        for _, combo in self.uniqueCombos.items():
            dfToFit = self.getRowsByCombination(df, combo)
            if dfToFit.empty:
                continue
            inds = dfToFit.index
            dfToFit = dfToFit.reset_index(drop=True)
            dfCopy.loc[inds, col] = self.container[col][combo.shortRepr()].inverseTransformCol(
                dfToFit, col)
        return dfCopy[col]

    @argValidator
    def inverseTransform(self, df: pd.DataFrame):
        self._warnToInverseTransform_mainGroups(df)
        for col in self.colNames:
            df[col] = self.inverseTransformCol(df, col)


class MainGroupSingleColsStdNormalizer(_MainGroupSingleColsNormalizer):
    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames,
                         colNames)

    @argValidator
    def setMeanNStd_ofMainGroups(self, df: pd.DataFrame):
        self._warnToInverseTransform_mainGroups(df)
        # cccAlgo
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
        return f"MainGroupSingleColsStdNormalizer:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"


class MainGroupSingleColsLblEncoder(_MainGroupSingleColsNormalizer):
    # cccAlgo this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used

    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)

    def __repr__(self):
        return f"MainGroupSingleColsLblEncoder:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"
