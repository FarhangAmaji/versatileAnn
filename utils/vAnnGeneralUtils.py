import pandas as pd
import numpy as np
#%%
class DotDict:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError(key)

class NpDict(DotDict):
    """
    converts cols of df to a dict of np arrays or help reassign
    also helps reassigning the dtypes of df subsets
    """
    def __init__(self, df):
        super().__init__({col: df[col].values for col in df.columns})
        self.__index__ = df.index

    def getDfCols(self):
        keys=list(self.data.keys())
        return keys

    def getDfDict(self, resetDtype=False):
        keys=self.getDfCols()
        if resetDtype:
            return {col: self[col].tolist() for col in keys}
        return {col: self[col] for col in keys}

    def toDf(self, resetDtype=False):
        return pd.DataFrame(self.getDfDict(resetDtype),index=self.__index__,columns=self.getDfCols())

    def __getitem__(self, key):
        if key in self.getDfCols():
            return self.data[key]
        elif isinstance(key, list):
            # If a list of keys is provided, return a dictionary with selected columns
            return np.column_stack([self[col] for col in key])
        elif isinstance(key, slice):
            if key == slice(None, None, None):
                # If the slice is [:], return the stacked data of all columns
                return np.column_stack([self[col] for col in self.getDfCols()])
            else:
                # Raise an error for other slice types
                raise ValueError("Only [:] is allowed for slicing.")
        else:
            raise KeyError(key)
#%% lists
def checkAllItemsInList1ExistInList2(list1, list2):
    setList2 = set(list2)
    for item in list1:
        if item not in setList2:
            return False
    return True
#%% dfs
def equalDfs(df1, df2, floatPrecision=0.0001):
    # Check if both DataFrames have the same shape
    if df1.shape != df2.shape:
        return False

    # Iterate through columns and compare them individually
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            # Check if all elements in the numeric column are close
            if not np.allclose(df1[col], df2[col], rtol=floatPrecision):
                return False
        else:
            if any([pd.api.types.is_numeric_dtype(df1[col]), pd.api.types.is_numeric_dtype(df2[col])]):
                npd1=NpDict(df1).getDfDict(True)
                npd2=NpDict(df2).getDfDict(True)
                if any([pd.api.types.is_numeric_dtype(npd1[col]), pd.api.types.is_numeric_dtype(npd2[col])]):
                    return False
            # If the column is non-numeric, skip the check
            continue

    # If all numeric columns are close, return True
    return True