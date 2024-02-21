"""
also check npDict_dfMutual.py
"""
import pandas as pd

from projectUtils.dataTypeUtils.list import areItemsOfList1_InList2
from projectUtils.typeCheck import argValidator


def regularizeBoolCol(df, colName):
    if not areItemsOfList1_InList2(df[colName].unique(), [0., 1., True, False]):
        raise ValueError(f"{colName} col doesnt seem to have boolean values")
    df[colName] = df[colName].astype(bool)


def pandasGroupbyAlternative(df, columns, **kwargs):
    """
    Custom implementation of pandas' groupby method to ensure consistent behavior across
    different versions.

    In some pandas versions, grouping by a single column results in keys as tuples ('g1',).
    In other versions, keys are returned as values from the column ('g1').
    This function ensures group names are always strings, not tuples, regardless of pandas version.
    """
    grouped = df.groupby(columns, **kwargs)
    for groupName, groupDf in grouped:
        if isinstance(groupName, tuple) and len(groupName) == 1:
            groupName = groupName[0]
        yield groupName, groupDf


@argValidator
def tryToConvertSeriesToDatetime(series: pd.Series):
    # if all items in series are string
    if series.apply(lambda x: isinstance(x, str)).all():
        try:
            newSeries = pd.to_datetime(series)

            # Check if the conversion was successful
            # note it seems pd.to_datetime always converts to datetime64
            if pd.api.types.is_datetime64_any_dtype(newSeries):
                return newSeries
            else:
                return series
        except:
            return series
    else:
        return series
