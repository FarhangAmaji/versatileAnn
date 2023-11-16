from dataPrep.normalizers_baseEncoders import _LblEncoder, _StdScaler, _IntLabelsString
from dataPrep.normalizers_baseNormalizer import _BaseNormalizer


class BaseSingleColsNormalizer(_BaseNormalizer):
    # cccUsage
    #  for instances of SingleColsLblEncoder if they have/need IntLabelsStrings, we wont use 3 transforms.
    #  for i.e. in if we have 5->'colA0'->0, BaseSingleColsNormalizer transforms only the 'colA0'->0 and not 5->'colA0' or 5->0
    # goodToHave2 transformCol, inverseMiddleTransformCol, inverseMiddleTransform, inverseTransform
    # goodToHave1 maybe comment needs a more detailed explanation
    def __init__(self):
        self.isFitted = {col: False for col in self.colNames}

    @property
    def colNames(self):
        return self.scalers.keys()

    def isColFitted(self, col, printFitted=False, printNotFitted=False):
        if self.isFitted[col]:
            if printFitted:
                print(f'{self.__repr__()} {col} is already fitted')
            return True
        if printNotFitted:
            print(f'{self.__repr__()} {col} is not fitted yet; fit it first')
        return False

    def fitCol(self, df, col):
        self.assertColNameInDf(df, col)
        if self.isColFitted(col, printFitted=True):
            return
        self.scalers[col].fit(df[col])
        self.isFitted[col] = True

    def fit(self, df):
        for col in self.colNames:
            self.fitCol(df, col)

    def fitNTransformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if self.isColFitted(col, printFitted=True):
            return
        self.fitCol(df, col)
        df[col] = self.transformCol(df, col)

    def transform(self, df):
        for col in self.colNames:
            df[col] = self.transformCol(df, col)

    def fitNTransform(self, df):
        for col in self.colNames:
            self.fitNTransformCol(df, col)

    def inverseMiddleTransformCol(self, df, col):
        if not self.isColFitted(col, printNotFitted=True):
            return df[col]
        dataToBeInverseTransformed = df[col]
        return self.scalers[col].inverseTransform(dataToBeInverseTransformed)

    def inverseTransformCol(self, dataToBeInverseTransformed, col):
        if not self.isColFitted(col, printNotFitted=True):
            return dataToBeInverseTransformed
        dataToBeInverseTransformed = self.inverseMiddleTransformCol(
            dataToBeInverseTransformed, col)
        if hasattr(self, 'intLabelsStrings'):
            # addTest2 does this part have tests
            if col in self.intLabelsStrings.keys():
                dataToBeInverseTransformed = self.intLabelsStrings[
                    col].inverseTransform(dataToBeInverseTransformed)
        return dataToBeInverseTransformed


class SingleColsStdNormalizer(BaseSingleColsNormalizer):
    def __init__(self, colNames: list):
        self.scalers = {col: _StdScaler(f'std{col}') for col in colNames}
        super().__init__()

    def transformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if not self.isColFitted(col, printNotFitted=True):
            return df[col]
        return self.scalers[col].transform(df[col])

    def __repr__(self):
        return f"SingleColsStdNormalizer:{'_'.join(self.colNames)}"


class SingleColsLblEncoder(BaseSingleColsNormalizer):
    def __init__(self, colNames: list):
        self.intLabelsStrings = {}
        self.encoders = {col: _LblEncoder(f'lbl{col}') for col in colNames}
        super().__init__()

    @property
    def scalers(self):
        return self.encoders

    def fitCol(self, df, col):
        try:
            super().fitCol(df, col)
        except ValueError as e:
            if str(e) == _LblEncoder.intDetectedErrorMsg:
                self.intLabelsStrings[col] = _IntLabelsString(col)
                self.intLabelsStrings[col].fit(df[col])
                intLabelsStringsTransformed = self.intLabelsStrings[
                    col].transform(df[col])
                self.scalers[col].fit(intLabelsStringsTransformed)
                self.isFitted[col] = True
            else:
                raise

    def transformCol(self, df, col):
        self.assertColNameInDf(df, col)
        if not self.isColFitted(col, printNotFitted=True):
            return df[col]
        if col in self.intLabelsStrings.keys():
            df[col] = self.intLabelsStrings[col].transform(df[col])
        return self.scalers[col].transform(df[col])

    def getClasses(self):
        return {col: enc.encoder.classes_ for col, enc in self.encoders.items()}

    def __repr__(self):
        return f"SingleColsLblEncoder:{'_'.join(self.colNames)}"
