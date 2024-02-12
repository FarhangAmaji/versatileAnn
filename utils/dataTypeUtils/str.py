import re

from utils.typeCheck import argValidator


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
