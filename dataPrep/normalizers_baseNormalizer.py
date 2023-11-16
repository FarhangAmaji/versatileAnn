from abc import ABC, abstractmethod


class _BaseNormalizer(ABC):
    def assertColNameInDf(self, df, col):
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
    def fitNTransformCol(self):
        ...

    @abstractmethod
    def fitNTransform(self):
        ...

    # @abstractmethod
    def inverseMiddleTransformCol(self):
        ...

    # @abstractmethod
    def inverseMiddleTransform(self):
        ...

    # @abstractmethod
    def inverseTransformCol(self):
        ...

    # @abstractmethod
    def inverseTransform(self):
        ...
