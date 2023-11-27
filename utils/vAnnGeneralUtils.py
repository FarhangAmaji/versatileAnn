import inspect

import numpy as np
import pandas as pd
import torch


# ---- DotDict NpDict
class DotDict:
    # goodToHave2 add .get and setdefault
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
        return {key: self.data[key] for key in self.keys()}

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
        return 'DotDict: ' + str(self.dict)


class NpDict(DotDict):
    """
    converts cols of df to a dict of np arrays or
    also helps reassigning the dtypes of df subsets
    """

    # kkk make sure other functionalities of pd df, except the things defined below are kept
    # kkk maybe also works with pd series(probably not needed)
    # kkk add setItem
    def __init__(self, df):
        if not (isinstance(df, pd.DataFrame) or isinstance(df, dict)):
            raise ValueError('only pandas DataFrames, or dictionaries are accepted')
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        super().__init__({col: df[col].values for col in df.columns})
        self.__index__ = df.index
        self.shape = df.shape

    def cols(self):
        keys = list(self.data.keys())
        return keys

    def getDict(self, resetDtype=False):
        if resetDtype:
            return {col: self[col].tolist() for col in self.cols()}
        return {col: self[col] for col in self.cols()}

    def printDict(self):
        print('{')
        for col in self.cols():
            colRes = list(self[col])
            if hasThisListAnyRange(colRes):
                colRes = listToRanges(colRes)
            colRes = similarItemsString(colRes)
            print(f"'{col}': {colRes}, ")
        print('}')

    def toDf(self, resetDtype=False):
        return pd.DataFrame(self.getDict(resetDtype), index=self.__index__, columns=self.cols())

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
            # kkk add number slices
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
        tempDf = self.toDf()
        tempDf = tempDf.reset_index(drop=True)
        return str(tempDf)


def equalNpDicts(npd1, npd2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
    if npd1.shape != npd2.shape:
        return False

    if checkIndex:
        if list(npd1.__index__) != list(npd2.__index__):
            return False

    return equalDfs(npd1.df, npd2.df, checkIndex=False,
                    floatApprox=floatApprox, floatPrecision=floatPrecision)


# ---- tensor
def tensor_floatDtypeChangeIfNeeded(tensor):
    if tensor.dtype == torch.float16 or tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
        # kkk make it compatible to global precision
    return tensor


def equalTensors(tensor1, tensor2, checkType=True, floatApprox=False, floatPrecision=1e-4,
                 checkDevice=True):
    dtypeEqual = True
    if checkType:
        dtypeEqual = tensor1.dtype == tensor2.dtype
    if not dtypeEqual:
        return False

    deviceEqual = tensor1.device == tensor2.device
    if not checkDevice:
        if not deviceEqual:  # cccDevStruct even though device check is not need but make both tensors to cpu in order not to get different device error in equal line below
            tensor1 = tensor1.to(torch.device('cpu'))
            tensor2 = tensor2.to(torch.device('cpu'))
        deviceEqual = True
    if not deviceEqual:
        return False

    equalVals = True
    if floatApprox:
        # Check if the tensors are equal with precision
        equalVals = torch.allclose(tensor1, tensor2, atol=floatPrecision)
    else:
        equalVals = torch.equal(tensor1, tensor2)
    return equalVals


# ---- dfs
def equalDfs(df1, df2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
    # kkk needs tests
    # Check if both DataFrames have the same shape
    if df1.shape != df2.shape:
        return False

    if checkIndex:
        if list(df1.index) != list(df2.index):
            return False

    if floatApprox:
        # Iterate through columns and compare them individually
        for col in df1.columns:
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                # Check if all elements in the numeric column are close
                if not np.allclose(df1[col], df2[col], rtol=floatPrecision):
                    return False
            else:
                if any([pd.api.types.is_numeric_dtype(df1[col]),
                        pd.api.types.is_numeric_dtype(df2[col])]):
                    npd1 = NpDict(df1).getDict(True)
                    npd2 = NpDict(df2).getDict(True)
                    if any([pd.api.types.is_numeric_dtype(npd1[col]),
                            pd.api.types.is_numeric_dtype(npd2[col])]):
                        return False
                # If the column is non-numeric, skip the check
                continue

        # If all numeric columns are close, return True
        return True
    else:
        return df1.equals(df2)


def regularizeBoolCol(df, colName):
    if not areItemsOfList1_InList2(df[colName].unique(), [0., 1., True, False]):
        raise ValueError(f"{colName} col doesnt seem to have boolean values")
    df[colName] = df[colName].astype(bool)


# ---- np array
def npArrayBroadCast(arr, shape):
    shape = tuple(shape)
    arrShape = arr.shape
    arrShapeLen = len(arrShape)
    assert arrShape[:arrShapeLen] == shape[
                                     :arrShapeLen], 'np array and the given shape, doesnt have same first dims'
    repeatCount = np.prod(shape[arrShapeLen:])
    res = np.repeat(arr, repeatCount).reshape(shape)
    return res


def equalArrays(array1, array2, checkType=True, floatApprox=False, floatPrecision=1e-4):
    dtypeEqual = True
    if checkType:
        # Check if the data types are equal
        dtypeEqual = array1.dtype == array2.dtype

    equalVals = True
    if floatApprox:
        # Check if the arrays are equal with precision
        equalVals = np.allclose(array1, array2, atol=floatPrecision)
    else:
        equalVals = np.array_equal(array1, array2)

    # Return True if both data type and precision are equal
    return dtypeEqual and equalVals


# ---- lists
def areItemsOfList1_InList2(list1, list2):
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


def hasThisListAnyRange(list_):
    return any([type(item) == range for item in list_])


def similarItemsString(inputList):
    result = []
    currentItem = inputList[0]
    count = 1

    def formatItem(item, count):
        itemStr = f"'{item}'" if isinstance(item, str) else str(item)
        return f"{count}*[{itemStr}]" if count > 1 else itemStr

    for item in inputList[1:]:
        if item == currentItem:
            count += 1
        else:
            result.append(formatItem(currentItem, count))
            currentItem = item
            count = 1

    result.append(formatItem(currentItem, count))

    result2 = []
    currRes2 = []
    currResFormat = lambda currRes2: '[' + ', '.join(currRes2) + ']'

    for item2 in result:
        if '*' not in item2:
            currRes2.append(item2)
        else:
            if currRes2:
                result2.append(currResFormat(currRes2))
                currRes2 = []
            result2.append(item2)
    if currRes2:
        result2.append(currResFormat(currRes2))

    return '+'.join(result2)


# ---- floats
def morePreciseFloat(num, precisionOrder=6):
    return round(num, precisionOrder)


# ---- misc
def gpuMemoryUsed():
    device = torch.device("cuda")

    memory_allocated = torch.cuda.memory_allocated(device) / 1024
    memory_cached = torch.cuda.memory_reserved(device) / 1024
    print(f'Memory Allocated: {memory_allocated:,.2f} KB')
    print(f'Memory Cached: {memory_cached:,.2f} KB')


# kkk create func arg getter, similar to varPasser; gets locals() and a func and gets possible args from locals
def varPasser(*, localArgNames=None, exclude=None):
    if exclude is None:
        exclude = []
    # kkk add var renamer
    locals_ = inspect.currentframe().f_back.f_locals
    dict_ = {}
    if localArgNames:
        for lan in localArgNames:
            if lan not in exclude:  # this is usually not needed; but for the case we have some dictionary and we want to remove some keys
                dict_[lan] = locals_[lan]
    else:
        for lan in locals_.keys():
            if lan == 'self':
                continue
            if lan not in exclude:  # this may be needed much more comparing to prev
                dict_[lan] = locals_[lan]
    return dict_


def _allowOnlyCreationOf_ChildrenInstances(self, cls):
    if type(self) == cls:
        raise RuntimeError(f"Instances of {cls.__name__} are not allowed")


def validate_IsObjOfTypeX_orAListOfTypeX(typeX):
    def func(obj, errMsg=''):
        if not errMsg:
            errMsg = f"the object isn't of type {typeX} or a list of it"

        isObjOfTypeX = isinstance(obj, typeX)
        isAListOfTypeX = (isinstance(obj, list) and
                          all([isinstance(it, typeX) for it in obj]))
        if not (isObjOfTypeX or isAListOfTypeX):
            raise ValueError(errMsg)

    return func


def nLastCallers(n=1):
    import inspect
    frame = inspect.currentframe().f_back
    calling_frame = frame
    print('\nnLastCallers')
    for i in range(n):
        calling_frame = calling_frame.f_back  # Get the frame of the calling function/method
        calling_module = inspect.getmodule(calling_frame)
        calling_function = inspect.getframeinfo(calling_frame).function
        print(f'{i + 1}th last frame\n{calling_frame=}\n{calling_module=}\n{calling_function=}')


def getCurrentFuncName():
    frame = inspect.currentframe()
    try:
        return frame.f_back.f_code.co_name
    finally:
        del frame


def shuffleData(inputData_, seed=None):
    import random
    import copy
    if seed is not None:
        random.seed(seed)
    inputCopy = copy.deepcopy(inputData_)

    if isinstance(inputCopy, list):  # list
        shuffledData = random.sample(inputCopy, len(inputCopy))
    elif isinstance(inputCopy, tuple):
        # Convert tuple to list, shuffle, and convert back to tuple
        shuffledData = tuple(random.sample(list(inputCopy), len(inputCopy)))
    elif isinstance(inputCopy, pd.DataFrame):  # DataFrame
        shuffledData = inputCopy.sample(frac=1).reset_index(drop=True)
    elif isinstance(inputCopy, np.ndarray):
        # Shuffle NumPy array along the first axis (rows)
        shuffledData = np.random.permutation(inputCopy)
    elif isinstance(inputCopy, torch.Tensor):
        # Convert tensor to NumPy array, shuffle, and convert back to tensor
        npArray = inputCopy.numpy()
        np.random.shuffle(npArray)
        shuffledData = torch.from_numpy(npArray)
    else:
        # Handle unsupported data type
        raise ValueError(f"Unsupported data type: {type(inputCopy)}")

    return shuffledData
