import ast
import inspect
import os
import platform
import re
import threading
import types
from typing import Union

import aiohttp
import numpy as np
import pandas as pd
import torch

from utils.typeCheck import argValidator
from utils.warnings import Warn


# goodToHave2
#  split it to multiple files

# ---- DotDict NpDict
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
            return {col: self[col].tolist() for col in self.cols()}
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
    # mustHave3
    #  the whole code which uses this func should be revised to apply pytorch lightning
    #  model precision to tensors and not just changing to float32 by default
    if tensor.dtype == torch.float16 or tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
        # kkk make it compatible to global precision
    return tensor


def equalTensors(tensor1, tensor2, checkType=True, floatApprox=False, floatPrecision=1e-6,
                 checkDevice=True):
    tensor1_ = tensor1.clone()
    tensor2_ = tensor2.clone()

    dtypeEqual = True
    if checkType:
        dtypeEqual = tensor1_.dtype == tensor2_.dtype
    else:
        tensor1_ = tensor1_.to(torch.float32)
        tensor2_ = tensor2_.to(torch.float32)
    if not dtypeEqual:
        return False

    deviceEqual = tensor1_.device == tensor2_.device
    if not checkDevice:
        if not deviceEqual:
            #  even though device check is not need but make both tensors to cpu
            #  in order not to get different device error in equal line below
            tensor1_ = tensor1_.to(torch.device('cpu'))
            tensor2_ = tensor2_.to(torch.device('cpu'))
        deviceEqual = True
    if not deviceEqual:
        return False

    equalVals = True
    if floatApprox:
        # Check if the tensors are equal with precision
        equalVals = torch.allclose(tensor1_, tensor2_, atol=floatPrecision)
    else:
        equalVals = torch.equal(tensor1_, tensor2_)
    return equalVals


def toDevice(tensor, device):
    # cccDevAlgo
    #  'mps' device doesn't support float64 and int64
    # check if the device.type is 'mps' and it's float64 or int64; first change
    # dtype to float32 or int32, after that change device
    if device.type == 'mps':
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
            Warn.info('float64 tensor is changed to float32 to be compatible with mps')
        elif tensor.dtype == torch.int64:
            tensor = tensor.to(torch.int32)
            Warn.info('int64 tensor is changed to int32 to be compatible with mps')
    return tensor.to(device)


# ---- dfs
def equalDfs(df1, df2, checkIndex=True, floatApprox=False, floatPrecision=0.0001):
    # addTest1
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


# ---- np array
def npArrayBroadCast(arr, shape):
    shape = tuple(shape)
    arrShape = arr.shape
    arrShapeLen = len(arrShape)
    if arrShape[:arrShapeLen] != shape[:arrShapeLen]:
        raise ValueError('np array and the given shape, doesnt have same first dims')
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
def areItemsOfList1_InList2(list1, list2, giveNotInvolvedItems=False):
    notInvolvedItems = []

    setList2 = set(list2)
    for item in list1:
        if item not in setList2:
            notInvolvedItems.append(item)

    result = notInvolvedItems == []
    if giveNotInvolvedItems:
        return result, notInvolvedItems
    return result


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

    if not all(isinstance(rg, range) for rg in rangeList):
        raise ValueError('Not all items are ranges')

    res = []
    for rg in rangeList:
        res.extend(range(rg.start, rg.stop))

    return res


def hasThisListAnyRange(list_):
    return any([type(item) == range for item in list_])


def similarItemsString(inputList):
    result = []
    lastItem = inputList[0]
    count = 1

    def formatItem(item, count):
        itemStr = f"'{item}'" if isinstance(item, str) else str(item)
        string = f"{count} * [{itemStr}]" if count > 1 else itemStr
        return string

    for item in inputList[1:]:
        if item == lastItem:
            count += 1
        else:
            result.append(formatItem(lastItem, count))
            lastItem = item
            count = 1

    result.append(formatItem(lastItem, count))

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

    return ' + '.join(result2)


# ---- floats
def morePreciseFloat(num, precisionOrder=6):
    return round(num, precisionOrder)


# ---- dicts
@argValidator
def giveOnlyKwargsRelated_toMethod(method, updater: dict, updatee: dict = None, delAfter=False):
    # cccDevStruct
    #  finds keys in updater that can be passed to method as they are in the args that method takes
    #  updatee is the result which can have some keys from before
    #  - also takes for camelCase adaptibility for i.e. if the method takes `my_arg`
    #       but updater has `myArg`, includes `my_arg` as 'myArg'
    if not callable(method):
        raise ValueError(f'method should be a method or a function.')

    updatee = updatee or {}
    methodArgs = {key: key for key in getMethodArgs(method)}
    for key in methodArgs:
        if key in updater:
            updatee.update({key: updater[key]})
            if delAfter:
                del updater[key]
        elif snakeToCamel(key) in updater:
            updatee.update({key: updater[snakeToCamel(key)]})
            if delAfter:
                del updater[snakeToCamel(key)]
    return updatee


def isNestedDict(dict_):
    if not isinstance(dict_, dict):
        return False

    for value in dict_.values():
        if isinstance(value, dict):
            return True

    return False


# ---- methods and funcs
def getMethodArgs(method):
    return list(inspect.signature(method).parameters.keys())


def isCustomFunction(func):
    # Helper function to check if a function is a custom (user-defined and not Python built-in or from packages) function

    import builtins
    import pkg_resources
    import types

    if func is None or func is types.NoneType:
        return False

    moduleName = getattr(func, '__module__', '')
    return (
            isinstance(func, types.FunctionType) and
            not (
                    func in builtins.__dict__.values()
                    or any(moduleName.startswith(package.key) for package in
                           pkg_resources.working_set)
                    or moduleName.startswith('collections')
            )
    )


# ---- methods and funcs: detect funcs, instance methods or static methods

def isStaticmethod(method):
    if not isinstance(method, types.FunctionType):
        return False

    return inspect.getsource(method).strip().startswith('@staticmethod')


def getStaticmethod_actualClass(method):
    '''
    it's not possible to get actual class of a static method don't have
    access to its instance or class.
    this 'globals().get(className)' doesn't work always, ofc it may work
    sometimes if the className is in globals
    '''
    if not isStaticmethod(method):
        return ''
    qualname = method.__qualname__
    className = qualname.split('.')[0]
    actualClass = globals().get(className)
    return actualClass


def isFunctionOrMethod(obj):
    if isStaticmethod(obj):
        return True, "Static Method"
    elif isinstance(obj, types.FunctionType):
        if len(obj.__qualname__.split('.')) > 1:
            return True, "Instance Method"
        return True, "Function"
    elif isinstance(obj, types.MethodType):
        return True, "Instance Method"
    else:
        return False, "not a method or a func"


def _ifFunctionOrMethod_returnIsClass_AlsoActualClassOrFunc(obj):
    funcOrNot, result = isFunctionOrMethod(obj)
    if not funcOrNot:
        return False, None  # first one is isClass, next one is ClassOrFunc object

    if result == "Static Method":
        return True, getStaticmethod_actualClass(obj)
    elif result == "Function":
        return False, obj
    elif result == "Instance Method":
        if hasattr(obj, '__self__'):
            return True, obj.__self__.__class__
        return True, obj


# ---- classes utils
def isCustomClass(cls_):
    # Helper function to check if a class is a custom(user defined and not python builtin or not from packages) class

    import builtins
    import pkg_resources
    import types
    if cls_ is None or cls_ is types.NoneType:  # kkk
        return False
    moduleName = getattr(cls_, '__module__', '')
    return (
            isinstance(cls_, type) and
            not (
                    cls_ in builtins.__dict__.values()
                    or any(moduleName.startswith(package.key) for package in
                           pkg_resources.working_set)
                    or moduleName.startswith('collections')
            )
    ) and not issubclass(cls_, types.FunctionType)


def findClassDefinition_inADirectory(directoryPath, className):
    """
    This function searches for a specific class definition in all Python files within a given directory.
    """

    filePathsHavingTheDefinitionOfClass = []
    classDefinitions = []

    class ClassVisitor(ast.NodeVisitor):
        """
        This class is a subclass of ast.NodeVisitor used to visit nodes in the AST.
        It overrides the visit_ClassDef method to process class definition nodes.
        """

        def visit_ClassDef(self, node):
            """
            This method is called for each class definition node in the AST.
            If the class name matches the target class name, it appends its source code to lastClassDefinitions.

            Parameters:
            node (ast.ClassDef): The class definition node to process.
            """
            if node.name == className:
                start_line = node.lineno
                # Find the last line of the class definition
                end_line = max((getattr(n, 'end_lineno', start_line) for n in ast.walk(node)),
                               default=start_line)
                classDefinition = lines[start_line - 1:end_line]
                lastClassDefinitions.append('\n'.join(classDefinition))
            self.generic_visit(node)

    for root, dirs, files in os.walk(directoryPath):
        for file in files:
            if file.endswith('.py'):
                filePath = os.path.join(root, file)
                try:
                    with open(filePath, 'r') as f:
                        lines = f.readlines()
                        tree = ast.parse(''.join(lines))
                        lastClassDefinitions = []
                        # reset lastClassDefinitions for this file; as it is the indicator of
                        # 'ClassVisitor().visit(tree)' being successful or not
                        ClassVisitor().visit(tree)
                        if lastClassDefinitions:
                            filePathsHavingTheDefinitionOfClass.append(filePath)
                            classDefinitions.extend(lastClassDefinitions)
                except Exception as e:
                    print(
                        f"findClassDefinition_inADirectory: An error occurred while parsing the file {filePath}: {e}")

    return {'className': className, 'Definitions': classDefinitions,
            'filePaths': filePathsHavingTheDefinitionOfClass}


def getClassObjectFromFile(className, filePath):
    """
    Get the class object based on the provided class name and file path.

    :return: Class object if found, otherwise None.
    """
    import importlib
    import sys
    try:
        # Extract the directory containing the module
        moduleDir = os.path.dirname(filePath)

        # Add the module directory to sys.path
        sys.path.append(moduleDir)

        # Import the module dynamically
        moduleName = os.path.splitext(os.path.basename(filePath))[0]
        module = importlib.import_module(moduleName)

        # Retrieve the class object using the class name
        classObject = getattr(module, className)

        # Check if the retrieved object is a class
        if isinstance(classObject, type):
            return classObject
        else:
            print(f"{className} is not a class in {filePath}.")
    except ImportError:
        print(f"Module {filePath} not found.")
    except AttributeError:
        print(f"Class {className} not found in {filePath}.")
    finally:
        # Remove the module directory from sys.path to avoid conflicts
        sys.path.remove(moduleDir)

    return None


def findClassObject_inADirectory(directoryPath, className):
    result = {'className': className, 'classObjects': [],
              'filePaths': [], 'Definitions': []}
    findClassDefinitionRes = findClassDefinition_inADirectory(directoryPath, className)

    for i, path in enumerate(findClassDefinitionRes['filePaths']):
        result['classObjects'].append(getClassObjectFromFile(className, path))
        result['filePaths'].append(findClassDefinitionRes['filePaths'][i])
        result['Definitions'].append(findClassDefinitionRes['Definitions'][i])

    return result


# ---- download
async def downloadFileAsync(url, destination, event=None):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(destination, 'wb') as f:
                    f.write(content)
                if event:
                    event.set()  # Set the event to signal that the file has been downloaded
            else:
                raise Exception(f"Failed to download file. Status code: {response.status}")


# ---- os utils
def filePathToDirectoryPath(path):
    # cccDevStruct
    #  doesn't work with no file extension paths
    if os.path.exists(path):
        if os.path.isfile(path):
            directory_path = os.path.dirname(path)
            return directory_path
        elif os.path.isdir(path):
            # If it's already a directory path, do nothing
            return path
        else:
            raise ValueError(f"{path} is neither a file nor a directory.")
    else:
        raise ValueError(f"{path} doesn't exist.")


def nFoldersBack(path, n=1):
    # Get the absolute path to handle both relative and absolute paths
    absolutePath = os.path.abspath(path)

    # Navigate three levels up for directories
    for _ in range(n):
        absolutePath = os.path.dirname(absolutePath)

    # If the original path is a file, go up one more level
    if os.path.isfile(path):
        absolutePath = os.path.dirname(absolutePath)

    return absolutePath


def getProjectDirectory():
    return nFoldersBack(os.path.abspath(__file__))


# ---- torch utils
def getTorchDevice():
    # bugPotentialCheck1
    #  this func may still not work with macbooks; ofc in general they don't work with 'cuda' but
    #  may also not work with Mps
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # Check if MPS is available (only for MacOS with Metal support)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    # Check if XLA is available (only if `torch_xla` package is installed)
    elif 'xla' in torch.__dict__:
        device = torch.device('xla')
    else:
        device = torch.device('cpu')
    return device


def getDefaultTorchDevice_name():
    device_ = torch.tensor([1, 2]).to(getTorchDevice()).device
    deviceName = ''

    if device_.type == 'cuda':
        if hasattr(device_, 'index'):
            deviceName = f"{device_.type}:{device_.index}"
        else:
            deviceName = f"{device_.type}"

    elif device_.type == 'mps':
        deviceName = f'{device_.type}'
    elif device_.type == 'cpu':
        deviceName = 'cpu'
    # bugPotentialCheck2
    #  not sure about xla devices

    return deviceName


def getDefaultTorchDevice_printName():
    # gives printing device name of getTorchDevice()
    deviceName = getDefaultTorchDevice_name()
    devicePrintName = ''
    if 'cuda' in deviceName or 'mps' in deviceName:
        devicePrintName = f", device='{deviceName}'"
    elif 'cpu' in deviceName:
        devicePrintName = ''
    # bugPotentialCheck2
    #  not sure about xla devices

    return devicePrintName


# ---- str utils

def camelToSnake(camelString):
    # Use regular expression to insert underscores before capital letters
    snakeString = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', camelString)
    # Convert to lowercase
    snakeString = snakeString.lower()
    return snakeString


def snakeToCamel(snakeString):
    # Use regular expression to capitalize letters following underscores
    camelString = re.sub(r'(?!^)_([a-zA-Z])', lambda x: x.group(1).upper(), snakeString)
    return camelString


@argValidator
def joinListWithComma(list_: list, doubleQuoteItems=True):
    if doubleQuoteItems:
        return '"' + '", "'.join(list_) + '"'
    return ', '.join(list_)


def randomStringGenerator(stringLength=4, characters=None):
    import random
    import string
    characters = characters or (string.ascii_letters + string.digits)

    return ''.join(random.choices(characters, k=stringLength))


@argValidator
def spellPluralS(list_: list, string="", es=False):
    if len(list_) > 1:
        if es:
            string += "es"
        else:
            string += "s"
    return string


# ---- variable
def NoneToNullValueOrValue(var, nullVal):
    '''
    it's a very important to prevent bugs
    'a=a or []' must be '[] if a is None else a' of as this func is
    specially in recursive funcs and the variable is mutable so we have counted
    on the fact it is mutable.
    it's common that first few iters may not populate the variable
    so when the one populates it is a different variable than the variable in the upper func(the
    one has called this func) so it is a variable at different memory location
    '''
    if var is None:
        return nullVal
    return var


# ---- misc
def gpuMemoryUsed():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")

    memory_allocated = torch.cuda.memory_allocated(device) / 1024
    memory_cached = torch.cuda.memory_reserved(device) / 1024
    print(f'Memory Allocated: {memory_allocated:,.2f} KB')
    print(f'Memory Cached: {memory_cached:,.2f} KB')


def varPasser(*, localArgNames=None, exclude=None, rename=None):
    # cccDevAlgo
    #  in order not to pass many local variables which have the same name to another
    #  func by mentioning `func1(var1=var1, var2=var2,....)` with use this func
    #  and put local variables to a dictionary and in a more clean way with pass that dict as kwargs
    #  This function is used to pass variables from one scope to another.
    #  It takes three optional arguments: localArgNames, exclude, and rename.
    #  localArgNames is a list of variable names to be passed.
    #  exclude is a list of variable names to be excluded from being passed.
    #  rename is a dictionary where the keys are the current variable names and the values are
    #  the new names.
    #  simple example:
    #  kwargs_=varPasser(localArgNames=['var1', 'var2', 'var3', 'var4', 'var5', 'var6'])
    #  func1(**kwargs_)
    localArgNames = localArgNames or []
    exclude = exclude or []
    rename = rename or {}

    # bugPotentialCheck1
    #  this has a big bug pontential in macOs maybe other non windows Oses when the
    #  localArgNames is None
    if platform.system() != 'Windows' and localArgNames is None:
        Warn.warn(f'for {platform.system()} Os(operating system) is better to pass ' + \
                  'localArgNames and not depend on automatic variable detection of varPasser')

    # Get local variables from the calling function's frame
    locals_ = inspect.currentframe().f_back.f_locals

    # Check if variables specified for renaming exist in local variables
    for rn in rename.keys():
        if rn not in locals_.keys():
            raise ValueError(f'{rn} is not in the local vars')

    # Collect variables to include in the result
    argsToGet = set(localArgNames[:] + list(rename.keys()))
    result = {}
    for atg in argsToGet:
        if atg not in locals_.keys():
            raise ValueError(f'{atg} is not in the local vars')
        if atg not in exclude:
            result[atg] = locals_[atg]

    # Include all local variables if localArgNames is not provided
    if not localArgNames:
        for loc in locals_.keys():
            if loc == 'self':
                continue
            if loc not in exclude:
                result[loc] = locals_[loc]

    # Rename variables as specified in the 'rename' dictionary
    for rn, rnVal in rename.items():
        result[rnVal] = result[rn]
        del result[rn]
    return result


def _allowOnlyCreationOf_ChildrenInstances(self, cls):
    if type(self) == cls:
        raise RuntimeError(f"Instances of {cls.__name__} are not allowed")


def validate_IsObjOfTypeX_orAListOfTypeX(typeX):
    # cccAlgo
    #  currently argValidator decorator with use of hints like List[typeX] or List[Union[typeX, int]]
    #  does this sort of validation
    def func(obj, errMsg=''):
        if not errMsg:
            errMsg = f"the object isn't of type {typeX} or a list of it"

        isObjOfTypeX = isinstance(obj, typeX)
        isAListOfTypeX = (isinstance(obj, list) and
                          all([isinstance(it, typeX) for it in obj]))
        if not (isObjOfTypeX or isAListOfTypeX):
            raise ValueError(errMsg)

    return func


def giveDateTimeStr():
    import datetime
    return datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")


def inputTimeout(prompt, timeout=30):
    """
        Display a prompt to the user and wait for input with a specified timeout.

        Parameters:
        - prompt (str): The message to display as the input prompt.
        - timeout (int, optional): The time limit (in seconds) for the user to input data.
                                  Defaults to 30 seconds.

        Returns:
        - str or False: If input is received within the timeout, returns the user's input as a string.
                        If no input is received within the timeout, returns False.
    """
    print(prompt)
    user_input = [None]

    def inputThread():
        user_input[0] = input()

    # Start the input thread
    thread = threading.Thread(target=inputThread)
    thread.start()

    # Wait for the thread to finish or timeout
    thread.join(timeout)

    # Check if input was received or timeout occurred
    if thread.is_alive():
        # print("No input received. Continuing the code...")
        return False
    else:
        return user_input[0]


def nLastCallers(n=1):
    # cccAlgo this is useful for debugging
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
