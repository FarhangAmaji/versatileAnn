from dataPrep.normalizers import MainGroupSingleColsNormalizer
from dataPrep.normalizers_multiColNormalizer import BaseMultiColNormalizer
from dataPrep.normalizers_singleColsNormalizer import BaseSingleColsNormalizer


class NormalizerStack:
    # addTest2 doesnt have individual tests but all methods are used in other tests
    def __init__(self, *stdNormalizers):
        self._normalizers = {}
        for stdNormalizer in stdNormalizers:
            self.addNormalizer(stdNormalizer)

    def addNormalizer(self, newNormalizer):
        assert isinstance(newNormalizer, (
            BaseSingleColsNormalizer, BaseMultiColNormalizer,
            MainGroupSingleColsNormalizer))
        for col in newNormalizer.colNames:
            if col not in self._normalizers.keys():
                # mustHave2 add ability to have a col which exists in 2 normalizers(either colNames or mainGroup)
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

    def fitNTransform(self, df):
        for nrm in self.uniqueNormalizers:
            nrm.fitNTransform(df)

    def transformCol(self, df, col):
        # mustHave2 later when ability to have a key in 2,... uniqueNormalizers is added; pay attention
        # ... so the order of applying and occausionally temp invTransforming them is ok
        # addTest1 needs tests
        return self._normalizers[col].transformCol(df, col)

    def inverseMiddleTransform(self, df):
        for col in list(self.normalizers.keys())[::-1]:
            df[col] = self.inverseMiddleTransformCol(df, col)

    def inverseMiddleTransformCol(self, df, col):
        # mustHave2 same as transformCol; for having a key in multiple uniqueNormalizers
        assert col in self._normalizers.keys(), f'{col} is not in normalizers cols'
        return self._normalizers[col].inverseMiddleTransformCol(df, col)

    def inverseTransform(self, df):
        for col in list(self.normalizers.keys())[::-1]:
            df[col] = self.inverseTransformCol(df, col)

    def inverseTransformCol(self, df, col):
        # mustHave2 same as transformCol; for having a key in multiple uniqueNormalizers
        return self._normalizers[col].inverseTransformCol(df, col)

    def __repr__(self):
        return str(self.uniqueNormalizers)
