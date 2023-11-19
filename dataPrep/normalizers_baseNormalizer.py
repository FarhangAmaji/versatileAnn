from abc import ABC, abstractmethod

from utils.typeCheck import argValidator


class _BaseNormalizer(ABC):
    # cccDevStruct
    #  note the command pattern cant be applied here as the users do transforms on each col when they want.
    #  so there is nothing which can be tracked centralized
    def __init__(self):
        self.__colNames = []

    @property
    def colNames(self):
        return self.__colNames

    @colNames.setter
    @argValidator
    def colNames(self, value):
        self.__colNames = value

    @staticmethod
    def _assertColNameInDf(df, col):
        assert col in df.columns, f'{col} is not in df columns'

    # @abstractmethod
    def fitCol(self):
        ...

    @abstractmethod
    def fit(self):
        ...

    # @abstractmethod
    def transformCol(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    # @abstractmethod
    def fitNTransformCol(self, df, col):
        # goodToHave3 add NpDict also
        ...

    @abstractmethod
    def fitNTransform(self):
        ...

    # @abstractmethod
    def inverseMiddleTransformCol(self, df, col):
        ...

    # @abstractmethod
    def inverseMiddleTransform(self):
        ...

    # @abstractmethod
    def inverseTransformCol(self, df, col):
        ...

    # @abstractmethod
    def inverseTransform(self):
        ...
