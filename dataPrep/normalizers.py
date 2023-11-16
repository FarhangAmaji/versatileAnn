from dataPrep.normalizers_singleColsNormalizer import SingleColsStdNormalizer, \
    SingleColsLblEncoder
from dataPrep.normalizers_baseEncoders import _LblEncoder, _StdScaler, _IntLabelsString
from dataPrep.normalizers_baseNormalizer import _BaseNormalizer
from utils.vAnnGeneralUtils import NpDict


# ---- normalizers: BaseMultiColNormalizers
class BaseMultiColNormalizer(_BaseNormalizer):
    # goodToHave2 fitcol, fitNTransformCol,  inverseMiddleTransform, inverseTransform
    def __init__(self):
        self.isFitted = False

    def assertColNames(self, df):
        for col in self.colNames:
            self.assertColNameInDf(df, col)

    def areTheseIntCols(self, df):
        return df[self.colNames].apply(
            lambda col: col.apply(lambda x: isinstance(x, int))).all().all()

    def isFittedFunc(self, printFitted=False, printNotFitted=False):
        if self.isFitted:
            if printFitted:
                print(f'{self.__repr__()} is already fitted')
            return True
        if printNotFitted:
            print(f'{self.__repr__()} is not fitted yet; fit it first')
        return False

    def transformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if not self.isFittedFunc(printNotFitted=True):
            return df[col]
        res = df[col]
        if isinstance(self, MultiColLblEncoder) and self.intLabelsString:
            # goodToHave1 oop
            res = self.intLabelsString.transform(res)
        return self.scaler.transform(res)

    def transform(self, df):
        self.assertColNames(df)
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    def fit(self, df):
        self.assertColNames(df)
        if self.isFittedFunc(printFitted=True):
            return
        dfColsCopy = df[self.colNames].copy()
        # goodToHave3 is copying needed
        if isinstance(self, MultiColLblEncoder) and self.areTheseIntCols(df):
            # goodToHave1 oop
            self.intLabelsString = _IntLabelsString(self.shortRep())
            self.intLabelsString.fit(dfColsCopy)
            dfColsCopy = self.intLabelsString.transform(dfColsCopy)
        self.scaler.fit(dfColsCopy)
        self.isFitted = True

    def fitNTransform(self, df):
        if self.isFittedFunc(printFitted=True):
            return
        self.fit(df)
        self.transform(df)

    # goodToHave2 could have add many fit, transform, assert and their other combinations for single col
    # goodToHave2 could have added inverseMiddleTransform and inverseTransform which does inverse on self.colNames in df

    def inverseMiddleTransformCol(self, df, col):
        if not self.isFittedFunc(printNotFitted=True):
            return df[col]
        dataToBeInverseTransformed = df[col]
        return self.scaler.inverseTransform(dataToBeInverseTransformed)

    def inverseTransformCol(self, dataToBeInverseTransformed, col):
        assert col in dataToBeInverseTransformed.columns, 'inverseTransformCol "{self}" "{col}" col is not in df columns'
        if not self.isFittedFunc(printNotFitted=True):
            return dataToBeInverseTransformed[col]
        res = self.inverseMiddleTransformCol(dataToBeInverseTransformed, col)
        if isinstance(self, MultiColLblEncoder) and self.intLabelsString:
            res = self.intLabelsString.inverseTransform(res)
            # bugPotentialCheck2 does the singlecol has done intlabel invTrans after its main transform
        return res


class MultiColStdNormalizer(BaseMultiColNormalizer):
    def __init__(self, colNames):
        super().__init__()
        self.colNames = colNames
        self.scaler = _StdScaler(self.shortRep())

    def shortRep(self):
        return 'std:' + '_'.join(self.colNames)

    def __repr__(self):
        return f"MultiColStdNormalizer:{'_'.join(self.colNames)}"


class MultiColLblEncoder(BaseMultiColNormalizer):
    def __init__(self, colNames):
        super().__init__()
        self.intLabelsString = None
        self.colNames = colNames
        self.encoder = _LblEncoder(self.shortRep())

    @property
    def scaler(self):
        return self.encoder

    def getClasses(self):
        return self.encoder.encoder.classes_

    def shortRep(self):
        return 'lbl:' + '_'.join(self.colNames)

    def __repr__(self):
        return f"MultiColLblEncoder:{'_'.join(self.colNames)}"


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


class MainGroupBaseNormalizer:
    def __init__(self, df, mainGroupColNames):
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


# ---- MainGroupSingleColsNormalizer
class MainGroupSingleColsNormalizer(MainGroupBaseNormalizer,
                                    _BaseNormalizer):
    # goodToHave2 fitNTransformCol, inverseMiddleTransform, inverseTransform
    def __init__(self, classType, df, mainGroupColNames, colNames: list):
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


class MainGroupSingleColsStdNormalizer(MainGroupSingleColsNormalizer):
    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsStdNormalizer, df, mainGroupColNames,
                         colNames)

    def getMeanNStd(self, df):
        for col in self.colNames:
            for combo in self.uniqueCombos:
                dfToFit = self.getRowsByCombination(df, combo)
                inds = dfToFit.index
                scaler = self.container[col][combo.shortRepr_()].scalers[
                    col].scaler
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
    # mustHave2 (should think about it much more) a good solution is that in fitNTransform of normalStack I do fit then transform and if the next uniqueNormalizer has this col in its colNames or groupNames, undo and redo again
    # addTest1 add test for this
    def __repr__(self):
        return f"MainGroupSingleColsStdNormalizer:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"


class MainGroupSingleColsLblEncoder(MainGroupSingleColsNormalizer):
    "this the lblEncoder version of MainGroupSingleColsStdNormalizer; its rarely useful, but in some case maybe used"

    def __init__(self, df, mainGroupColNames, colNames: list):
        super().__init__(SingleColsLblEncoder, df, mainGroupColNames, colNames)

    # goodToHave2 maybe add getClasses()

    def __repr__(self):
        return f"MainGroupSingleColsLblEncoder:{'_'.join(list(map(str, self.uniqueCombos)))}:{'_'.join(self.colNames)}"
