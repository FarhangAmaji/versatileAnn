import pandas as pd
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