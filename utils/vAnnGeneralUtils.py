import torch
import pandas as pd
import numpy as np
#%% DotDict NpDict
class DotDict:
    def __init__(self, data):
        self.data = data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @property
    def dict(self):
        return {key:self.data[key] for key in self.keys()}

    def __len__(self):
        return len(self.keys())

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

    def __iter__(self):
        return iter(self.data.items())

    def __repr__(self):
        return 'DotDict: '+str(self.dict)

class NpDict(DotDict):
    """
    converts cols of df to a dict of np arrays or help reassign
    also helps reassigning the dtypes of df subsets
    """
    #kkk make sure other functionalities of pd df, except the things defined below are kept
    #kkk maybe also works with pd series(probably not needed)
    #kkk add setItem
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('only pandas DataFrames are accepted')
        super().__init__({col: df[col].values for col in df.columns})
        self.__index__ = df.index

    def cols(self):
        keys=list(self.data.keys())
        return keys

    def getDict(self, resetDtype=False):
        keys=self.cols()
        if resetDtype:
            return {col: self[col].tolist() for col in keys}
        return {col: self[col] for col in keys}

    def toDf(self, resetDtype=False):
        return pd.DataFrame(self.getDict(resetDtype),index=self.__index__,columns=self.cols())

    @property
    def df(self):
        return self.toDf()

    def __getitem__(self, key):
        if key in self.cols():
            return self.data[key]
        elif isinstance(key, list):
            # If a list of keys is provided, return a dictionary with selected columns
            return np.column_stack([self[col] for col in key])
        elif isinstance(key, slice):
            #kkk add number slices
            if key == slice(None, None, None):
                # If the slice is [:], return the stacked data of all columns
                return np.column_stack([self[col] for col in self.cols()])
            else:
                # Raise an error for other slice types
                raise ValueError("Only [:] is allowed for slicing.")
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        tempDf=self.toDf()
        tempDf=tempDf.reset_index(drop=True)
        return str(tempDf)
#%% tensor
def floatDtypeChange(tensor):
    if tensor.dtype == torch.float16 or tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
        #kkk make it compatible to global precision
    return tensor

def tensorEqualWithDtype(tensor1, tensor2):
    if torch.equal(tensor1, tensor2) and tensor1.dtype == tensor2.dtype:
        return True
    return False
#%% dfs
def equalDfs(df1, df2, floatPrecision=0.0001):
    #kkk needs tests
    #kkk its not index sensetive
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
                npd1=NpDict(df1).getDict(True)
                npd2=NpDict(df2).getDict(True)
                if any([pd.api.types.is_numeric_dtype(npd1[col]), pd.api.types.is_numeric_dtype(npd2[col])]):
                    return False
            # If the column is non-numeric, skip the check
            continue

    # If all numeric columns are close, return True
    return True
#%% np array
def npArrayBroadCast(arr, shape):
    shape=tuple(shape)
    arrShape=arr.shape
    arrShapeLen=len(arrShape)
    assert arrShape[:arrShapeLen]==shape[:arrShapeLen], 'np array and the given shape, doesnt have same first dims'
    repeatCount=np.prod(shape[arrShapeLen:])
    res= np.repeat(arr, repeatCount).reshape(shape)
    return res
#%% lists
def checkAllItemsInList1ExistInList2(list1, list2):
    setList2 = set(list2)
    for item in list1:
        if item not in setList2:
            return False
    return True

def isListTupleOrSet(obj):
    return isinstance(obj, (list, tuple, set))

def isIterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def listToRanges(inputList):
    if not inputList:
        return []

    ranges = []
    start = inputList[0]
    end = inputList[0]

    for num in inputList[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(range(start, start + 1))
            else:
                ranges.append(range(start, end + 1))
            start = end = num

    if start == end:
        ranges.append(range(start, start + 1))
    else:
        ranges.append(range(start, end + 1))

    return ranges

def listRangesToList(rangeList):
    if not rangeList:
        return []

    assert all(isinstance(rg, range) for rg in rangeList), 'Not all items are ranges'
    
    res = []
    for rg in rangeList:
        res.extend(range(rg.start, rg.stop))
    
    return res
#%% floats
def morePreciseFloat(num, precisionOrder=6):
    return round(num,precisionOrder)