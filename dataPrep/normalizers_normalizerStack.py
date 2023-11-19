import pandas as pd

from dataPrep.normalizers_baseNormalizer import _BaseNormalizer
from utils.typeCheck import argValidator


class NormalizerStack:
    # cccUsage this is a Class to utilize all normalizers uniformly
    # addTest2 doesnt have individual tests but all methods are used in other tests
    @argValidator
    def __init__(self, *normalizers: _BaseNormalizer):
        self._normalizers = {}
        for normalizer in normalizers:
            self.addNormalizer(normalizer)

    @argValidator
    def addNormalizer(self, newNormalizer: _BaseNormalizer):
        for col in newNormalizer.colNames:
            if col not in self._normalizers.keys():
                # mustHave2 add ability to have a col which exists in 2 normalizers(either _colNames or mainGroup)
                self._normalizers.update({col: newNormalizer})
            else:
                print(f'{col} is already in normalizers')

    @property
    def normalizers(self):
        return self._normalizers

    @property
    def uniqueNormalizers(self):
        uniqueNormalizers = []
        [uniqueNormalizers.append(nrm) for nrm in self._normalizers.values() if
         nrm not in uniqueNormalizers]
        return uniqueNormalizers

    @argValidator
    def fitNTransform(self, df: pd.DataFrame):
        for nrm in self.uniqueNormalizers:
            nrm.fitNTransform(df)

    def _baseTransformCol(self, funcName, df: pd.DataFrame, col: str):
        # mustHave2 # bugPotentialCheck2 addTest1 needs tests
        #  later when ability to have a key in 2,... uniqueNormalizers is added; pay attention
        #  so the order of applying transform and invTransforming of them is ok
        # bugPotentialCheck2
        #  if duplicate cols is applied `self._normalizers[col]` is gonna be a list of different normalizers
        if col not in self.normalizers.keys():
            raise ValueError(f'{col} is not in normalizers cols')
        func = getattr(self.normalizers[col], funcName)
        return func(df, col)

    @argValidator
    def transformCol(self, df: pd.DataFrame, col: str):
        return self._baseTransformCol('transformCol', df, col)


    @argValidator
    def inverseTransformCol(self, df: pd.DataFrame, col):
        return self._baseTransformCol('inverseTransformCol', df, col)


    @argValidator
    def inverseTransform(self, df: pd.DataFrame):
        # bugPotentialCheck2
        #  in the past this loop was applied reversed; but later I found no meaningful difference; have this mind later if a bug occurred.
        for col in list(self.normalizers.keys()):
            df[col] = self.inverseTransformCol(df, col)

    def __repr__(self):
        return str(self.uniqueNormalizers)
