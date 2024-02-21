from projectUtils.dataTypeUtils.str import snakeToCamel
from projectUtils.misc import getMethodArgs
from projectUtils.typeCheck import argValidator


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


@argValidator
def stringValuedDictsEqual(dict1: dict, dict2: dict):
    """
    Check if two dictionaries with string values are equal.
    """
    for key, value in dict1.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for key '{key}' is not a string.")

    for key, value in dict2.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for key '{key}' is not a string.")

    if len(dict1) != len(dict2):
        return False

    for key, value in dict1.items():
        if key not in dict2 or dict2[key] != value:
            return False

    return True
