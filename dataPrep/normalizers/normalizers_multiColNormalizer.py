import pandas as pd

from dataPrep.normalizers.normalizers_baseEncoders import _LblEncoder, _StdScaler, _IntLabelsString
from dataPrep.normalizers.normalizers_baseNormalizer import _BaseNormalizer
from utils.typeCheck import argValidator
from utils.generalUtils import _allowOnlyCreationOf_ChildrenInstances


class _BaseMultiColNormalizer(_BaseNormalizer):
    # cccDevStruct
    #  there many similar funcs within this file and also between here and singleColsNormalizer
    #  which can use some `base func` in order to `reduce duplicity` but this may readability harder, so dont change it
    # cccDevAlgo
    #  fitCol therefore fitNTransformCol doesnt make any sense because its multiCol
    def __init__(self):
        super().__init__()
        self.encoder = None
        _allowOnlyCreationOf_ChildrenInstances(self, _BaseMultiColNormalizer)
        self.isFitted = False

    def _isFittedPlusPrint(self, printFitted=False, printNotFitted=False):
        return self._isFittedPlusPrint_base(printFitted=printFitted, printNotFitted=printNotFitted)

    @argValidator
    def fit(self, df: pd.DataFrame):
        self._assertColNamesInDf(df)
        if self._isFittedPlusPrint(printFitted=True):
            return
        self.encoder.fit(df[self.colNames])
        self.isFitted = True

    @argValidator
    def transform(self, df: pd.DataFrame):
        self._assertColNamesInDf(df)
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    @argValidator
    def fitNTransform(self, df: pd.DataFrame):
        if self._isFittedPlusPrint(printFitted=True):
            return
        self.fit(df)
        self.transform(df)

    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if not self._isFittedPlusPrint(printNotFitted=True):
            return df[col]
        data_ = self.encoder.inverseTransform(df[col])
        return data_

    @argValidator
    def inverseTransform(self, df: pd.DataFrame):
        self._assertColNamesInDf(df)
        for col in self.colNames:
            df[col] = self.inverseTransformCol(df, col)


class MultiColStdNormalizer(_BaseMultiColNormalizer):
    def __init__(self, colNames):
        super().__init__()
        self.colNames = colNames
        self.encoder = _StdScaler(self.shortRep())

    @argValidator
    def transformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if not self._isFittedPlusPrint(printNotFitted=True):
            return df[col]
        return self.encoder.transform(df[col])

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

    @argValidator
    def fit(self, df: pd.DataFrame):
        try:
            super().fit(df)
        except ValueError as e:
            if str(e) == _LblEncoder.intDetectedErrorMsg:
                dfColsData = df[self.colNames]
                self.intLabelsString = _IntLabelsString(self.shortRep())
                self.intLabelsString.fit(dfColsData)
                dfColsData = self.intLabelsString.transform(dfColsData)
                self.encoder.fit(dfColsData)
                self.isFitted = True
            else:
                raise

    @argValidator
    def transformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if not self._isFittedPlusPrint(printNotFitted=True):
            return df[col]
        res = df[col]
        if self.intLabelsString:
            res = self.intLabelsString.transform(res)
        return self.encoder.transform(res)

    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        data_ = super().inverseTransformCol(df, col)
        # cccAlgo
        #  intLabelsStrings inverseTransform apply after the encoder inverseTransform
        if self.intLabelsString:
            data_ = self.intLabelsString.inverseTransform(data_)
        return data_

    def getClasses(self):
        return self.encoder.encoder.classes_

    def shortRep(self):
        return 'lbl:' + '_'.join(self.colNames)

    def __repr__(self):
        return f"MultiColLblEncoder:{'_'.join(self.colNames)}"
