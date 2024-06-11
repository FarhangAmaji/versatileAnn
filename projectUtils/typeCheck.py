import inspect
import typing
from typing import get_type_hints

from pydantic import validate_arguments


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
        args_ = args[:]
        kwargs_ = kwargs.copy()
        allArgs, starArgVar = getAllArgs(args_)  # starArgVar is *args variable
        hints = get_type_hints(func)
        for argName, argVal in allArgs.items():
            hintType = hints.get(argName, '')
            if argName == starArgVar:  # to check *args hints
                if hintType:
                    if not doItemsOfListObeyHinting(allArgs[starArgVar],
                                                    [hints.get(starArgVar, '')]):
                        raise TypeError(f"values passed for *'{argName}' don't obey {hintType}")

            isListOfSomeType, innerListTypes = isHintTypeOfAListOfSomeType(hintType)
            if isListOfSomeType and not doItemsOfListObeyHinting(argVal, innerListTypes):
                raise TypeError(f"values passed for '{argName}' don't obey {hintType}")

        return func(*args, **kwargs)

    def getAllArgs(args):
        sig = inspect.signature(func)
        params = sig.parameters
        allArgs = {}
        argsIndex = 0
        starArgVar = None

        for paramName, param in params.items():
            if param.kind == param.VAR_POSITIONAL:
                allArgs[paramName] = args[argsIndex:]
                starArgVar = paramName
                break
            elif param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}:
                if argsIndex < len(args):
                    allArgs[paramName] = args[argsIndex]
                    argsIndex += 1
        return allArgs, starArgVar

    def doItemsOfListObeyHinting(argVals, innerListTypes):
        for arg in argVals:
            if not any([issubclass(type(arg), ilt) for ilt in innerListTypes]):
                return False
        return True

    return wrapper


def argValidator(func):
    # Apply Pydantic validation first
    func = validate_arguments(config={'arbitrary_types_allowed': True})(func)
    # Then apply the custom type hint checker
    return typeHintChecker_AListOfSomeType(func)
