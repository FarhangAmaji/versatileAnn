from dataPrep.normalizers_baseNormalizer import _BaseNormalizer
from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer, \
    SingleColsLblEncoder
from utils.vAnnGeneralUtils import NpDict, _allowOnlyCreationOf_ChildrenInstances


# ---- normalizers: MainGroupNormalizers
class Combo:
    def __init__(self, defDict, mainGroupColNames):
        assert isinstance(defDict, dict) and all(key in defDict for key in
                                                 mainGroupColNames), "defDict format is invalid."

        for key in defDict:
            if key not in mainGroupColNames:
                raise ValueError(
                    f"'{key}' is not a valid column name in mainGroupColNames.")

        for col in mainGroupColNames:
            if col not in defDict:
                raise ValueError(f"'{col}' is missing in combo definition.")

        self.defDict = defDict

    def shortRepr_(self):
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
        return [combo.shortRepr_() for combo in self.uniqueCombos]

    def findMatchingShortReprCombo(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.shortRepr_():
                return uniqueCombo
        return None

    def uniqueCombosDictReprs(self):
        return [combo.defDict for combo in self.uniqueCombos]

    def findMatchingDictReprCombo(self, combo):
        for uniqueCombo in self.uniqueCombos:
            if combo == uniqueCombo.defDict:
                return uniqueCombo
        return None

    def comboInUniqueCombos(self, combo):
        if isinstance(combo, Combo):
            if combo in self.uniqueCombos:
                return combo
        elif isinstance(combo, str):
            if self.findMatchingShortReprCombo(combo):
                return self.findMatchingShortReprCombo(combo)
        elif isinstance(combo, dict):
            if self.findMatchingDictReprCombo(combo):
                return self.findMatchingDictReprCombo(combo)
        return

    def _getUniqueCombinations(self, df):
        comboObjs = []

        for groupName, groupDf in df.groupby(self.mainGroupColNames):
            comboDict = dict(zip(self.mainGroupColNames, groupName))
            combo = Combo(comboDict, self.mainGroupColNames)
            comboObjs.append(combo)

        return comboObjs

    def getRowsByCombination(self, df, combo):
        comboObj = self.comboInUniqueCombos(combo)
        assert comboObj, "Combo is not in uniqueCombos"
        tempDf = df[
            (df[self.mainGroupColNames] == comboObj.defDict).all(axis=1)]

        # this is to correct dtypes
        npDict = NpDict(tempDf)
        tempDf = npDict.toDf(resetDtype=True)
        return tempDf


# ---- _MainGroupSingleColsNormalizer
class _MainGroupSingleColsNormalizer(_MainGroupBaseNormalizer,
                                     _BaseNormalizer):
    # goodToHave2 fitNTransformCol, inverseMiddleTransform, inverseTransform
    # goodToHave3 should not be able to have an instance
    def __init__(self, classType, df, mainGroupColNames, colNames: list):
        _allowOnlyCreationOf_ChildrenInstances(self, _MainGroupSingleColsNormalizer)
        super().__init__(df, mainGroupColNames)
        self.colNames = colNames
        self.container = {}
        for col in colNames:
            self.container[col] = {}
            for combo in self.uniqueCombos:
                self.container[col][combo.shortRepr_()] = classType([col])

    def fitCol(self, df, col):
        for combo in self.uniqueCombos:
            dfToFit = self.getRowsByCombination(df, combo)
            dfToFit = dfToFit.reset_index(drop=True)
            # goodToHave3 does it need reset_index
            self.container[col][combo.shortRepr_()].fit(dfToFit)

    def fit(self, df):
        for col in self.colNames:
            self.fitCol(df, col)

    def transformCol(self, df, col):
        dfCopy = df.copy()
        for combo in self.uniqueCombos:
            dfToFit = self.getRowsByCombination(df, combo)
            inds = dfToFit.index
            dfToFit = dfToFit.reset_index(drop=True)
            # goodToHave3 does it need reset_index
            self.container[col][combo.shortRepr_()].transform(dfToFit)
            dfToFit.index = inds
            dfCopy.loc[inds, col] = dfToFit
        return dfCopy[col]

    def transform(self, df):
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    def fitNTransform(self, df):
        self.fit(df)
        self.transform(df)

    def inverseTransformColBase(self, df, col, funcName):
        dfCopy = df.copy()
        for combo in self.uniqueCombos:
            dfToFit = self.getRowsByCombination(df, combo)
            inds = dfToFit.index
            dfToFit = dfToFit.reset_index(drop=True)
            func = getattr(self.container[col][combo.shortRepr_()], funcName)
            invRes = func(dfToFit, col)
            dfCopy.loc[inds, col] = invRes
        return dfCopy[col]

    def inverseMiddleTransformCol(self, df, col):
        return self.inverseTransformColBase(df, col,
                                            'inverseMiddleTransformCol')

    def inverseTransformCol(self, df, col):
        return self.inverseTransformColBase(df, col, 'inverseTransformCol')


class MainGroupSingleColsStdNormalizer(_MainGroupSingleColsNormalizer):
    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames,
                         colNames)

    def getMeanNStd(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit = self.getRowsByCombination(df, combo)
                inds = dfToFit.index
                scaler = self.container[col][combo.shortRepr_()].scalers[col].scaler
                comboMean = scaler.mean_[0]
                comboStd = scaler.scale_[0]
                df.loc[inds, f'{col}Mean'] = comboMean
                df.loc[inds, f'{col}Std'] = comboStd

    # mustHave2 normalizer=NormalizerStack(SingleColsLblEncoder(['sku', 'month', 'agency', *specialDays]), MainGroupSingleColsStdNormalizer(df, mainGroups, target))
    # ... normalizer.fitNTransform(df)
    # ... this wont work because the unqiueCombos in MainGroupSingleColsStdNormalizer are determined first and after fitNTransform
    # ... of SingleColsLblEncoder, values of mainGroups are changed
    # ... kinda correct way right now: normalizer=NormalizerStack(MainGroupSingleColsStdNormalizer(df, mainGroups, target), SingleColsLblEncoder(['sku', 'agency', 'month', *specialDays]))
    # mustHave2 for this problem initing all normalizers in init of NormalizerStack doesnt seem to be a good solution
    # mustHave2 (should think about it much more) a good solution is that in fitNTransform of normalStack I do fit then transform and if the next uniqueNormalizer has this col in its _colNames or groupNames, undo and redo again
    # addTest1 add test for this
    def __repr__(self):
        return f"MainGroupSingleColsStdNormalizer:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"


class MainGroupSingleColsLblEncoder(_MainGroupSingleColsNormalizer):
    "this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used"

    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)

    # goodToHave2 maybe add getClasses()

    def __repr__(self):
        return f"MainGroupSingleColsLblEncoder:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"
