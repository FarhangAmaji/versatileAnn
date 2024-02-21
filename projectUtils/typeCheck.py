import typing
from typing import get_type_hints

from pydantic import validate_arguments

from projectUtils.customErrors import InternalLogicError


def isHintTypeOfAListOfSomeType(typ):
    # cccAlgo
    #  detects these patterns: `typing.List[str]`, `typing.List[int]`,
    #  `typing.List[Union[str,tuple]]` or even `typing.List[customType]`
    if isinstance(typ, typing._GenericAlias) and typ.__origin__ is list:
        inner_type = typ.__args__[0]
        if hasattr(inner_type, '__origin__') and inner_type.__origin__ is typing.Union:
            return True, inner_type.__args__
        return True, [inner_type]
    return False, []


def typeHintChecker_AListOfSomeType(func):
    def wrapper(*args, **kwargs):
        allArgs = getAllArgs(args, kwargs)
        hints = get_type_hints(func)
        for argName, argVal in allArgs.items():
            hintType = hints.get(argName, '')
            isListOfSomeType, innerListTypes = isHintTypeOfAListOfSomeType(hintType)
            if isListOfSomeType and not doItemsOfListObeyHinting(argVal, innerListTypes):
                raise TypeError(f"values passed for {argName} don't obey {hintType}")

    def getAllArgs(args, kwargs):
        allFuncParams = list(func.__code__.co_varnames)
        argParams = allFuncParams[:]
        for k in kwargs:
            argParams.remove(k)
        if len(argParams) != len(args):
            raise InternalLogicError
        allArgs = {}
        for i, ap in enumerate(argParams):
            allArgs[ap] = args[i]
        allArgs.update(kwargs)
        return allArgs

    def doItemsOfListObeyHinting(argVal, innerListTypes):
        uniqueTypes = set(type(arg) for arg in argVal)
        for ut in uniqueTypes:
            if not any([issubclass(ut, ilt) for ilt in innerListTypes]):
                return False
        return True

    return wrapper


def argValidator(func):
    # 1st decorator to raise error
    typeHintChecker_AListOfSomeType(func)
    # 2nd decorator to raise error
    return validate_arguments(config={'arbitrary_types_allowed': True})(func)
