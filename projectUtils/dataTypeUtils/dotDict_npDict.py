"""
also check npDict_dfMutual.py
"""
from typing import Union

import numpy as np
import pandas as pd

from projectUtils.dataTypeUtils.df_series import tryToConvertSeriesToDatetime
from projectUtils.dataTypeUtils.list import hasThisListAnyRange, listToRanges, similarItemsString
from projectUtils.typeCheck import argValidator


class DotDict:
    def __init__(self, data):
        if not hasattr(data, 'keys') or not callable(getattr(data, 'keys')):
            raise ValueError("Input data must be a type that supports keys (e.g., a dictionary)")
        self._data = data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    @property
    def dict(self):
        return {key: self._data[key] for key in self.keys()}

    def copy(self):
        return DotDict(self._data.copy())

    def __len__(self):
        return len(self.keys())

    def __getattr__(self, key):
        if key in self._data.keys():
            return self._data[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __getitem__(self, key):
        if key in self._data.keys():
            return self._data[key]
        else:
            raise KeyError(key)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def setDefault(self, key, default=None):
        if key not in self._data:
            self._data[key] = default
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data.items())

    def __repr__(self):
        return 'DotDict: ' + str(self.dict)


class NpDict(DotDict):
    """
    converts cols of df to a dict of np arrays or
    also helps re-assigning the dtypes of df subsets
    """

    # kkk make sure other functionalities of pd df, except the things defined below are kept
    # kkk maybe also works with pd series(probably not needed)
    # kkk add setItem
    @argValidator
    def __init__(self, df: Union[pd.DataFrame, dict]):
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        super().__init__({col: df[col].values for col in df.columns})
        self.__index__ = df.index
        self.shape = df.shape

    def cols(self):
        keys = list(self._data.keys())
        return keys

    def getDict(self, resetDtype=False):
        if resetDtype:
            res = {}
            for col in self.cols():
                if pd.Series(self[col]).dtype.type == np.object_:
                    # if it's object dtype(note object dtype probably is string by default in pandas)
                    res[col] = tryToConvertSeriesToDatetime(pd.Series(self[col])).tolist()
                elif pd.Series(self[col]).dtype.type == np.datetime64:
                    res[col] = [self[col][i] for i in range(self[col].size)]
                else:
                    res[col] = self[col].tolist()
            return res
        return {col: self[col] for col in self.cols()}

    def printDict(self):
        # cccAlgo
        #  this is super useful to make a very more readable str version of df
        #  example: prints df like `{'__startPoint__': 6 * [True] + 10 * [False] + 6 * [True] + 10 * [False] ,}`
        print('{', end='')
        for i, col in enumerate(self.cols()):
            colRes = list(self[col])
            if hasThisListAnyRange(colRes):
                colRes = listToRanges(colRes)
            colRes = similarItemsString(colRes)
            print(f"'{col}': {colRes}", end=('' if i == len(self.cols()) - 1 else ',\n'))
        print('}')

    def toDf(self, resetDtype=False):
        # cccAlgo
        #  assume col data consists 1 string and 3 int data like ['s',1,2,3]; the nparray has
        #  determined the dtype==object but in the case we have removed the first string data('s'),
        #  the dtype is not going to be changed and remains `object`, but with `resetDtype=True`,
        #  the dtype is going to be determined again, and this time is gonna be int
        # mustHave1
        #  in the tutorial put the example of testToDf_resetDtype test also from getRowsByCombination

        return pd.DataFrame(self.getDict(resetDtype), index=self.__index__, columns=self.cols())

    @property
    def df(self):
        return self.toDf()

    def __getitem__(self, key):
        if key in self.cols():
            return self._data[key]
        elif isinstance(key, list):
            # If a list of keys is provided, return a dictionary with selected columns
            return np.column_stack([self[col] for col in key])
        elif isinstance(key, slice):
            # kkk add number slices
            if key == slice(None, None, None):
                # If the slice is [:], return the stacked data of all columns
                return np.column_stack([self[col] for col in self.cols()])
            else:
                # Raise an error for other slice types
                raise ValueError("Only [:] is allowed for slicing.")
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        raise ValueError("Item assignment is not allowed for NpDict.")

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        tempDf = self.toDf().reset_index(drop=True)
        return str(tempDf)
