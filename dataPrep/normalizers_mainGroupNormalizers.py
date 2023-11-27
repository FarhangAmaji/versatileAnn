import pandas as pd

from dataPrep.normalizers_baseNormalizer import _BaseNormalizer
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer, \
    SingleColsLblEncoder
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import NpDict, _allowOnlyCreationOf_ChildrenInstances


# ---- normalizers: MainGroupNormalizers
class _Combo:
    def __init__(self, defDict, mainGroupColNames):
        for key in defDict:
            assert key in mainGroupColNames, f"'{key}' is not a valid column name in mainGroupColNames."

        for col in mainGroupColNames:
            assert col in defDict, f"'{col}' is missing in combo definition."

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
        assert False, f'no {str(combo)} is in uniqueCombos'

    def _getUniqueCombinations(self, df):
        comboObjs = {}

        for groupName, groupDf in df.groupby(self.mainGroupColNames):
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
        super().__init__(df, mainGroupColNames)
        self.colNames = colNames
        self.container = {}
        for col in colNames:
            self.container[col] = {}
            for _, combo in self.uniqueCombos.items():
                self.container[col][combo.shortRepr()] = classType([col])

    @argValidator
    def fitCol(self, df: pd.DataFrame, col):
        for _, combo in self.uniqueCombos.items():
            dfToFit = self.getRowsByCombination(df, combo)
            dfToFit = dfToFit.reset_index(drop=True)
            self.container[col][combo.shortRepr()].fit(dfToFit)

    @argValidator
    def fit(self, df: pd.DataFrame):
        for col in self.colNames:
            self.fitCol(df, col)

    @argValidator
    def transformCol(self, df: pd.DataFrame, col):
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
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    @argValidator
    def fitNTransformCol(self, df: pd.DataFrame):
        self.fitCol(df)
        self.transformCol(df)

    @argValidator
    def fitNTransform(self, df: pd.DataFrame):
        self.fit(df)
        self.transform(df)

    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        dfCopy = df.copy()
        for _, combo in self.uniqueCombos.items():
            dfToFit = self.getRowsByCombination(df, combo)
            inds = dfToFit.index
            dfToFit = dfToFit.reset_index(drop=True)
            dfCopy.loc[inds, col] = self.container[col][combo.shortRepr()].inverseTransformCol(
                dfToFit, col)
        return dfCopy[col]

    @argValidator
    def inverseTransform(self, df: pd.DataFrame):
        for col in self.colNames:
            df[col] = self.inverseTransformCol(df, col)


class MainGroupSingleColsStdNormalizer(_MainGroupSingleColsNormalizer):
    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames,
                         colNames)

    @argValidator
    def getMeanNStd(self, df: pd.DataFrame):
        for col in self.colNames:
            for _, combo in self.uniqueCombos.items():
                dfToFit = self.getRowsByCombination(df, combo)
                inds = dfToFit.index
                scaler = self.container[col][combo.shortRepr()].scalers[col].scaler
                comboMean = scaler.mean_[0]
                comboStd = scaler.scale_[0]
                df.loc[inds, f'{col}Mean'] = comboMean
                df.loc[inds, f'{col}Std'] = comboStd

    # mustHave2
    #  """normalizer=NormalizerStack(SingleColsLblEncoder(['sku', 'month', 'agency', *specialDays]), MainGroupSingleColsStdNormalizer(df, mainGroups, target))
    #  normalizer.fitNTransform(df)"""
    #  this wont work because the unqiueCombos in MainGroupSingleColsStdNormalizer are determined first and after fitNTransform
    #  of SingleColsLblEncoder, values of mainGroups are changed
    #  kinda correct way right now: normalizer=NormalizerStack(MainGroupSingleColsStdNormalizer(df, mainGroups, target), SingleColsLblEncoder(['sku', 'agency', 'month', *specialDays]))
    #  - for this problem initing all normalizers in init of NormalizerStack doesnt seem to be a good solution
    #  - (should think about it much more) a good solution is that in fitNTransform of normalStack I do fit then transform and if the next uniqueNormalizer has this col in its _colNames or groupNames, undo and redo again
    #  addTest1 add test for this
    def __repr__(self):
        return f"MainGroupSingleColsStdNormalizer:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"


class MainGroupSingleColsLblEncoder(_MainGroupSingleColsNormalizer):
    # cccAlgo this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used

    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)

    def __repr__(self):
        return f"MainGroupSingleColsLblEncoder:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"
