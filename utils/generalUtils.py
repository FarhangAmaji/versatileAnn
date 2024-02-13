import ast
import inspect
import os
import platform
import threading
import types

import aiohttp
import numpy as np
import pandas as pd
import torch

from utils.warnings import Warn


# ---- floats
def morePreciseFloat(num, precisionOrder=6):
    return round(num, precisionOrder)


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


class ClassVisitor(ast.NodeVisitor):
    # goodToHave2
    #  make a similar version for funcs
    """
    This class is a subclass of ast.NodeVisitor used to visit nodes in the AST.
    It overrides the visit_ClassDef method to process class definition nodes.
    """

    def __init__(self, className, lines):
        self.className = className
        self.classDefinitions = []
        self.lines = lines

    def visit_ClassDef(self, node):
        """
        This method is called for each class definition node in the AST.
        If the class name matches the target class name, it appends its source code to lastClassDefinitions.

        Parameters:
        node (ast.ClassDef): The class definition node to process.
        """
        if node.name == self.className:
            start_line = node.lineno
            # Find the last line of the class definition
            end_line = max((getattr(n, 'end_lineno', start_line) for n in ast.walk(node)),
                           default=start_line)
            classDefinition = self.lines[start_line - 1:end_line]
            self.classDefinitions.append('\n'.join(classDefinition))
        self.generic_visit(node)


def findClassDefinition_inAFile(filePath, className, printOff=False):
    try:
        with open(filePath, 'r') as f:
            lines = f.readlines()
            tree = ast.parse(''.join(lines))
            visitor = ClassVisitor(className, lines)
            visitor.visit(tree)
            return visitor.classDefinitions
    except Exception as e:
        if not printOff:
            print(
                f"findClassDefinition_inAFile: An error occurred while parsing the file {filePath}: {e}")
        return []


def findClassDefinition_inADirectory(directoryPath, className, printOff=False):
    filePathsHavingTheDefinitionOfClass = []
    classDefinitions = []

    for root, dirs, files in os.walk(directoryPath):
        for file in files:
            if file.endswith('.py'):
                filePath = os.path.join(root, file)
                fileClassDefinitions = findClassDefinition_inAFile(filePath, className, printOff)
                if fileClassDefinitions:
                    filePathsHavingTheDefinitionOfClass.append(filePath)
                    classDefinitions.extend(fileClassDefinitions)

    return {'className': className, 'Definitions': classDefinitions,
            'filePaths': filePathsHavingTheDefinitionOfClass}


def getClassObjectFromFile(className, filePath, printOff=False):
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
            if not printOff:
                print(f"{className} is not a class in {filePath}.")
    except ImportError:
        if not printOff:
            print(f"Module {filePath} not found.")
    except AttributeError:
        if not printOff:
            print(f"Class {className} not found in {filePath}.")
    finally:
        # Remove the module directory from sys.path to avoid conflicts
        sys.path.remove(moduleDir)

    return None


def findClassObject_inADirectory(directoryPath, className, printOff=False):
    result = {'className': className, 'classObjects': [],
              'filePaths': [], 'Definitions': []}
    findClassDefinitionRes = findClassDefinition_inADirectory(directoryPath, className,
                                                              printOff=printOff)

    for i, path in enumerate(findClassDefinitionRes['filePaths']):
        result['classObjects'].append(getClassObjectFromFile(className, path, printOff=printOff))
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
    # this is supposed to bring same result from anywhere this func is called
    return nFoldersBack(os.path.abspath(__file__), n=1)


# ---- variable
def setDefaultIfNone(var, defaultVal):
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
        return defaultVal
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

    # goodToHave3
    #  think about removing this; because it's not the case
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
