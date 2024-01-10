import inspect
from typing import List

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import getMethodArgs, areItemsOfList1_InList2
# goodToHave2 add comments

@argValidator
def initParentClasses(cls, kwargs, obj,
                      exceptions: List[str] = None, just: List[str] = None):
    if not inspect.isclass(cls):
        raise ValueError('cls must be a class.')

    exceptions = exceptions or []
    just = just or []

    parentClasses = cls.__bases__
    parentClassesNames = [pc.__name__ for pc in parentClasses]
    if not areItemsOfList1_InList2(exceptions, parentClassesNames):
        raise ValueError('exceptions must be a list of parent classes.')
    if not areItemsOfList1_InList2(just, parentClassesNames):
        raise ValueError('just must be a list of parent classes.')

    _detectParentClasses_argConflicts(parentClasses)
    classArgs = _getEachParentClasse_args(parentClasses, kwargs)

    if just:
        for just_ in just:
            if just_ in exceptions:
                continue
            just_.__init__(obj, **classArgs[just_])
    else:
        for pc in parentClasses:
            if pc.__name__ in exceptions:
                continue
            pc.__init__(obj, **classArgs[pc.__name__])


def _getAllParentClassesArgs(parentClasses):
    allArgs = {}
    for pc in parentClasses:
        for arg in getMethodArgs(pc.__init__):
            if arg not in allArgs:
                allArgs[arg] = [pc.__name__]
            else:
                allArgs[arg].append(pc.__name__)
    return allArgs


def _detectParentClasses_argConflicts(parentClasses):
    allArgs = _getAllParentClassesArgs(parentClasses)
    for arg, argBaseClasses in allArgs.items():
        if arg in ['self', 'args', 'kwargs']:
            # cccDevStruct
            #  note 'args' may be acceptable by parent, but are just passing things
            #  with keyword kwargs
            continue
        else:
            if len(argBaseClasses) > 1:
                raise RuntimeError(
                    "internalError: this is for development:" +
                    "\nparentClasses must not have args with similar names in their __init__."
                    f'\narg "{arg}" is used in more than one base classes: {argBaseClasses}.')


def _getEachParentClasse_args(parentClasses, kwargs):
    allArgs = _getAllParentClassesArgs(parentClasses)
    classArgs = {pc.__name__: {} for pc in parentClasses}

    for arg, argBaseClasses in allArgs.items():
        if arg in ['self', 'args', 'kwargs']:
            continue
        if arg in kwargs.keys():
            classArgs[argBaseClasses[0]].update({arg: kwargs[arg]})

    return classArgs
