import typing
from typing import get_type_hints

from pydantic import validate_arguments

from projectUtils.customErrors import InternalLogicError


def isHintTypeOfAListOfSomeType(typ):
    # ccc1
    #  detects these patterns: `typing.List[str]`, `typing.List[int]`,
    #  `typing.List[Union[str,tuple]]` or even `typing.List[customType]`
    if isinstance(typ, typing._GenericAlias) and typ.__origin__ is list:
        innerType = typ.__args__[0]
        if hasattr(innerType, '__origin__') and innerType.__origin__ is typing.Union:
            # checks compatible for List[Union[str, int]] like
            return True, innerType.__args__
        return True, [innerType]
    return False, []


def typeHintChecker_AListOfSomeType(func):
    """
    a decorator which raises error when the hint is List[someType] and the argument passed for
    that argument doesn't follow the hinting
    """

    def wrapper(*args, **kwargs):
        allArgs = getAllArgs(args, kwargs)
        hints = get_type_hints(func)
        for argName, argVal in allArgs.items():
            hintType = hints.get(argName, '')
            isListOfSomeType, innerListTypes = isHintTypeOfAListOfSomeType(hintType)
            if isListOfSomeType and not doItemsOfListObeyHinting(argVal, innerListTypes):
                raise TypeError(f"values passed for '{argName}' don't obey {hintType}")

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
        if len(allFuncParams) != len(allArgs):
            raise InternalLogicError
        return allArgs

    def doItemsOfListObeyHinting(argVals, innerListTypes):
        for arg in argVals:
            if not any([issubclass(type(arg), ilt) for ilt in innerListTypes]):
                return False
        return True

    return wrapper


def argValidator(func):
    # 1st decorator to raise error
    func = typeHintChecker_AListOfSomeType(func)
    # 2nd decorator to raise error
    return validate_arguments(config={'arbitrary_types_allowed': True})(func)
