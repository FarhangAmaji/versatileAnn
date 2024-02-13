import pandas as pd

from dataPrep.normalizers.baseEncoders import _LblEncoder, _StdScaler, _IntLabelsString
from dataPrep.normalizers.baseNormalizer import _BaseNormalizer
from utils.typeCheck import argValidator
from utils.generalUtils import _allowOnlyCreationOf_ChildrenInstances


class _BaseSingleColsNormalizer(_BaseNormalizer):
    def __init__(self):
        _allowOnlyCreationOf_ChildrenInstances(self, _BaseSingleColsNormalizer)

        super().__init__()
        self.encoders = {}

    def _init_isFitted(self):
        self.isFitted = {col: False for col in self.colNames}

    @property
    def colNames(self):
        return self.encoders.keys()

    def _isFittedPlusPrint_col(self, col, printFitted=False, printNotFitted=False):
        return self._isFittedPlusPrint_base(col, printFitted=printFitted,
                                            printNotFitted=printNotFitted)

    @argValidator
    def fitCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if self._isFittedPlusPrint_col(col, printFitted=True):
            return
        self.encoders[col].fit(df[col])
        self.isFitted[col] = True

    @argValidator
    def fitNTransformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if self._isFittedPlusPrint_col(col, printFitted=True):
            return
        self.fitCol(df, col)
        df[col] = self.transformCol(df, col)

    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if not self._isFittedPlusPrint_col(col, printNotFitted=True):
            return df
        dfColData = df[col]
        data_ = self.encoders[col].inverseTransform(dfColData)
        return data_

    @argValidator
    def fit(self, df: pd.DataFrame):
        for col in self.colNames:
            self.fitCol(df, col)

    @argValidator
    def transform(self, df: pd.DataFrame):
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    @argValidator
    def fitNTransform(self, df: pd.DataFrame):
        for col in self.colNames:
            self.fitNTransformCol(df, col)

    @argValidator
    def inverseTransform(self, df: pd.DataFrame):
        for col in self.colNames:
            df[col] = self.inverseTransformCol(df, col)


class SingleColsStdNormalizer(_BaseSingleColsNormalizer):
    def __init__(self, colNames: list):
        super().__init__()
        self.encoders = {col: _StdScaler(f'std{col}') for col in colNames}
        self._init_isFitted()

    @argValidator
    def transformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if not self._isFittedPlusPrint_col(col, printNotFitted=True):
            return df[col]
        return self.encoders[col].transform(df[col])

    def __repr__(self):
        return f"SingleColsStdNormalizer:{'_'.join(self.colNames)}"


class SingleColsLblEncoder(_BaseSingleColsNormalizer):
    @argValidator
    def __init__(self, colNames: list):
        super().__init__()
        self.intLabelsStrings = {}
        self.encoders = {col: _LblEncoder(f'lbl:{col}') for col in colNames}
        self._init_isFitted()

    @argValidator
    def fitCol(self, df: pd.DataFrame, col):
        # cccAlgo
        #  note the code tries to fit _LblEncoder but in the case that some `int` is supposed to be a `categorical`
        #  an error would raised by _LblEncoder. after that some _IntLabelsString would be wrapped and applied before fitting
        try:
            super().fitCol(df, col)
        except ValueError as e:
            if str(e) == _LblEncoder.intDetectedErrorMsg:
                self.intLabelsStrings[col] = _IntLabelsString(col)
                self.intLabelsStrings[col].fit(df[col])
                intLabelsStringsTransformed = self.intLabelsStrings[col].transform(df[col])
                self.encoders[col].fit(intLabelsStringsTransformed)
                self.isFitted[col] = True
            else:
                raise

    @argValidator
    def transformCol(self, df: pd.DataFrame, col):
        self._assertColNameInDf(df, col)
        if not self._isFittedPlusPrint_col(col, printNotFitted=True):
            return df[col]
        # cccAlgo
        #  intLabelsStrings transforms apply before the encoder transform
        if col in self.intLabelsStrings.keys():
            data_ = self.intLabelsStrings[col].transform(df[col])
        else:
            data_ = df[col]
        return self.encoders[col].transform(data_)

    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        data_ = super().inverseTransformCol(df, col)
        # cccAlgo
        #  intLabelsStrings inverseTransform apply after the encoder inverseTransform
        if col in self.intLabelsStrings.keys():
            data_ = self.intLabelsStrings[col].inverseTransform(data_)
        return data_

    def getClasses(self):
        return {col: enc.encoder.classes_ for col, enc in self.encoders.items()}

    def __repr__(self):
        return f"SingleColsLblEncoder:{'_'.join(self.colNames)}"
