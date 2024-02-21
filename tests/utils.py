from unittest.mock import patch


def simulateInput(inputValue):
    """
    Simulates user input with a specified delay.

    Args:
        inputValue (str): The input value to provide.
    """

    return patch('builtins.input', return_value=inputValue)
