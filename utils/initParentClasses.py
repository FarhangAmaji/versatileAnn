import inspect
from typing import Union, List

from utils.typeCheck import argValidator
from utils.generalUtils import getMethodArgs


def getArgsOfClasses(classesDict, originalKwargs=None):
    # cccDevStruct
    #  classesDict is a dict with {'className':classObj}
    _assertClassesDictStruct(classesDict)

    allArgs = {}
    for pc, pcObj in classesDict.items():
        for arg in getMethodArgs(pcObj.__init__):
            if arg not in allArgs:
                allArgs[arg] = {}
                allArgs[arg]['classes'] = [pc]
            else:
                allArgs[arg]['classes'].append(pc)
            if originalKwargs:
                if arg in originalKwargs:
                    allArgs[arg]['value'] = originalKwargs[arg]
    return allArgs


@argValidator
def exclude_selfNArgsNKwargs_fromAllArgs(allArgs: dict):
    for removeArg in ['self', 'args', 'kwargs']:
        if removeArg in allArgs:
            allArgs.pop(removeArg)
    return allArgs


@argValidator
def getArgsRelatedToAClass_fromAllArgs(className: str, allArgs):
    argsRelatedToAClass = {}
    for arg, argVal in allArgs.items():
        if className in argVal['classes']:
            if 'value' in argVal:
                argsRelatedToAClass[arg] = allArgs[arg]['value']
    return argsRelatedToAClass


def initClasses_withAllArgs(obj, classes, allArgs,
                            exceptions: Union[List[str], None] = None,
                            just: Union[List[str], None] = None):
    # cccDevStruct
    #  this func gets all args and classes and initiates them with their related args
    exceptions = exceptions or []
    just = just or []

    if just:
        for justClsName in just:
            if justClsName in exceptions:
                continue
            clsObj = classes[justClsName]
            argsRelatedToClass = getArgsRelatedToAClass_fromAllArgs(justClsName, allArgs)
            clsObj.__init__(obj, **argsRelatedToClass)
    else:
        for clsName, clsObj in classes.items():
            if clsName in exceptions:
                continue
            argsRelatedToClass = getArgsRelatedToAClass_fromAllArgs(clsName, allArgs)
            clsObj.__init__(obj, **argsRelatedToClass)


def orderClassNames_soChildIsAlways_afterItsParents(classesDict: dict):
    # bugPotentialCheck2
    #  is it possible for this func to have error
    #  I assume that as the classes should be one way and have some order so this code cannot be wrong

    # pre-check if the expected classesDict struct is passed
    _assertClassesDictStruct(classesDict)

    classesNames = list(classesDict.keys())
    orderClasses_names = []

    # code is not optimized but is good enough
    for _ in range(len(classesNames) ** 2):
        for cd, clsObj in classesDict.items():
            if cd in orderClasses_names:
                continue
            allParentsAreIncluded = True
            for pc in clsObj.__bases__:
                if pc.__name__ in classesNames:
                    if pc.__name__ not in orderClasses_names:
                        allParentsAreIncluded = False
            if allParentsAreIncluded:
                orderClasses_names.append(cd)

        if len(orderClasses_names) == len(classesNames):
            break

    if len(orderClasses_names) != len(classesNames):
        raise RuntimeError('internalError: logic error.')

    return orderClasses_names


def _assertClassesDictStruct(classesDict):
    for cd, clsObj in classesDict.items():
        if not inspect.isclass(clsObj):
            raise TypeError(
                f'classesDict must have class objects as values. but {cd} is not a class object.')


def findParentClasses_OfAClass_tillAnotherClass(cls_, classTillThatIsWanted,
                                                parentClasses: dict = None):
    # this is similar to
    parentClasses = parentClasses or {}
    # goodToHave3 bugPotentialCheck2
    #  some classes may have same .__name__ but are actually different classes
    #  but I am not counting for that(maybe later).
    #  so for now each class is going to be captured in a dict with {class.__name__:classObj}

    if str(cls_) == str(classTillThatIsWanted):
        return parentClasses
    elif cls_ is object:
        return parentClasses

    parentClasses.update({cls_.__name__: cls_})
    parentsOfThisClass = cls_.__bases__
    for potc in parentsOfThisClass:
        parentClasses = findParentClasses_OfAClass_tillAnotherClass(
            potc,
            classTillThatIsWanted,
            parentClasses)
    return parentClasses


def checkIfAClassIs_initingItsParentClasses_inItsInit(cls_):
    # Check if the __init__ method is defined in the given class
    if '__init__' not in cls_.__dict__:
        return False

    # Get the source code of the __init__ method
    initSourceLines = inspect.getsourcelines(cls_.__dict__['__init__'])[0]
    initSourceFlat = ''.join(initSourceLines)

    parentClasses = cls_.__bases__
    for parentClass in parentClasses:
        if f'{parentClass.__name__}.__init__' in initSourceFlat:
            return True

    for line in initSourceLines:
        if line.strip().startswith('super('):
            return True

    return False
