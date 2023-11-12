from pydantic import validate_arguments

def argValidator(func):
    return validate_arguments(config={'arbitrary_types_allowed':True})(func)