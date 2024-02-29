import pandas as pd

from dataPrep.normalizers.baseNormalizer import _BaseNormalizer
from dataPrep.normalizers.mainGroupNormalizers import _MainGroupBaseNormalizer
from projectUtils.typeCheck import argValidator


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
            # goodToHave1
            #  add ability to have a col which exists in 2 normalizers
            #  this means several normalizers in a certain order can be applied to the col data
            #  - probably in the development the order is essential
            #  note with current normalizers this doesnt really make sense
            if col not in self._normalizers.keys():
                self._normalizers.update({col: newNormalizer})
            else:
                raise ValueError(f'{col} is already in normalizers')

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
        # mustHave2 # bugPotn2 addTest1 needs tests
        #  later when ability to have a key in 2,... uniqueNormalizers is added; pay attention
        #  so the order of applying transform and invTransforming of them is ok
        # bugPotn2
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
        # bugPotn2
        #  in the past this loop was applied reversed; but later I found no meaningful difference; have this mind later if a bug occurred.
        mainGroupsInverseTransformedNow = []
        for col in list(self.normalizers.keys()):
            # ccc1
            #  for inversing cols which have maingroups, the mainGroups should inversed first
            if isinstance(self.normalizers[col], _MainGroupBaseNormalizer):
                for mainGroup in self.normalizers[col].mainGroupColNames:
                    if mainGroup in self.normalizers.keys():
                        df[mainGroup] = self.inverseTransformCol(df, mainGroup)
                        mainGroupsInverseTransformedNow.append(mainGroup)
            # ccc1
            #  because only _LblEncoder amongst baseEncoders has the ability to skip inverseTransforms,
            #  in the case that maingroup may not be _LblEncoder we skip those maingroups already
            #  inverseTransformed here in line #LSkmg1
            if col not in mainGroupsInverseTransformedNow: #LSkmg1
                df[col] = self.inverseTransformCol(df, col)

    def __repr__(self):
        return str(self.uniqueNormalizers)
