from dataPrep.normalizers_baseEncoders import _LblEncoder, _StdScaler, _IntLabelsString
from dataPrep.normalizers_baseNormalizer import _BaseNormalizer
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances


class _BaseMultiColNormalizer(_BaseNormalizer):
    # goodToHave2 fitcol, fitNTransformCol,  inverseMiddleTransform, inverseTransform
    # goodToHave3 should not be able to have an instance
    def __init__(self):
        _allowOnlyCreationOf_ChildrenInstances(self, _BaseMultiColNormalizer)
        self.isFitted = False

    def assertColNames(self, df):
        for col in self.colNames:
            self._assertColNameInDf(df, col)

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
        self._assertColNameInDf(df, col)
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
    # goodToHave2 could have added inverseMiddleTransform and inverseTransform which does inverse on self._colNames in df

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


class MultiColStdNormalizer(_BaseMultiColNormalizer):
    def __init__(self, colNames):
        super().__init__()
        self.colNames = colNames
        self.scaler = _StdScaler(self.shortRep())

    def shortRep(self):
        return 'std:' + '_'.join(self.colNames)

    def __repr__(self):
        return f"MultiColStdNormalizer:{'_'.join(self.colNames)}"


class MultiColLblEncoder(_BaseMultiColNormalizer):
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
