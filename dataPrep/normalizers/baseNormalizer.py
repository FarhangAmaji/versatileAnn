from abc import ABC, abstractmethod

import pandas as pd

from projectUtils.typeCheck import argValidator


class _BaseNormalizer(ABC):
    # ccc1
    #  note the command pattern cant be applied here as the users do transforms on each col when they want.
    #  so there is nothing which can be tracked centralized
    def __init__(self):
        pass

    @property
    def colNames(self):
        return self.__colNames

    @colNames.setter
    @argValidator
    def colNames(self, value):
        self.__colNames = value

    @staticmethod
    def _assertColNameInDf(df, col):
        if col not in df.columns:
            raise KeyError(f'{col} is not in df columns')

    def _assertColNamesInDf(self, df):
        for col in self.colNames:
            self._assertColNameInDf(df, col)

    def _isFittedPlusPrint_base(self, col=None, printFitted=False, printNotFitted=False):
        colPrint = ''
        if col:
            isFitted = self.isFitted[col]
            colPrint = f' {col}'
        else:
            isFitted = self.isFitted

        if isFitted:
            if printFitted:
                print(f'{self.__repr__()}{colPrint} is already fitted')
            return True
        if printNotFitted:
            print(f'{self.__repr__()}{colPrint} is not fitted yet; fit it first')
        return False

    # @abstractmethod
    def fitCol(self):
        ...

    @abstractmethod
    def fit(self):
        ...

    # @abstractmethod
    def transformCol(self) -> pd.Series:
        ...

    @abstractmethod
    def transform(self):
        ...

    # @abstractmethod
    def fitNTransformCol(self, df, col):
        # goodToHave2 add NpDict also
        ...

    @abstractmethod
    def fitNTransform(self):
        ...

    # @abstractmethod
    def inverseTransformCol(self, df, col) -> pd.Series:
        ...

    # @abstractmethod
    def inverseTransform(self):
        ...
