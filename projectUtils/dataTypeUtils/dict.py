from projectUtils.typeCheck import argValidator


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
