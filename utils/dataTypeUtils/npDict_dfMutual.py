import numpy as np
import pandas as pd

from utils.dataTypeUtils.dotDict_npDict import NpDict
from utils.typeCheck import argValidator


# ---- df utils

@argValidator
def dfPrintDict(df: pd.DataFrame):
    npd = NpDict(df)
    npd.printDict()


@argValidator
def dfResetDType(df: pd.DataFrame):
    npd = NpDict(df)
    return npd.toDf(resetDtype=True)


def equalDfs(df1, df2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
    # addTest1
    df1_ = df1.copy()
    df2_ = df2.copy()

    # Check if both DataFrames have the same shape
    if df1_.shape != df2_.shape:
        return False

    if checkIndex:
        if list(df1_.index) != list(df2_.index):
            return False

    if floatApprox:
        # try to make pandas redetect column types; in some cases this may be handy
        df1_ = dfResetDType(df1_)
        df2_ = dfResetDType(df2_)

        for col in df1_.columns:
            if all([pd.api.types.is_numeric_dtype(df1_[col]),
                    pd.api.types.is_numeric_dtype(df2_[col])]):
                # case: the column on both dfs is numeric
                if not np.allclose(df1_[col], df2_[col], rtol=floatPrecision):
                    return False
            else:
                if not df1_[col].equals(df2_[col]):
                    return False

        # If all numeric columns are close, return True
        return True
    else:
        return df1_.equals(df2_)


# ---- npDict utils
def equalNpDicts(npd1, npd2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
    # note equalNpDicts is in df_series.py in order not to make circular imports
    # goodToHave1
    #  beside this in NpDict __equal__ could have been defined
    if npd1.shape != npd2.shape:
        return False

    if checkIndex:
        if list(npd1.__index__) != list(npd2.__index__):
            return False

    return equalDfs(npd1.df, npd2.df, checkIndex=False,
                    floatApprox=floatApprox, floatPrecision=floatPrecision)
