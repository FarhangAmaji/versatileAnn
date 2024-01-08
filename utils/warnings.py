import warnings

from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)


# Custom warning levels
class CustomWarning(UserWarning):
    pass


# Custom Warn class
class Warn:
    @staticmethod
    def _printWithColorNBackground(message, wordsColor, backgroundColor):
        style = Style.NORMAL
        warningMessage = f"{style}{wordsColor}{backgroundColor} {message} {Style.RESET_ALL}"
        warnings.warn(warningMessage, CustomWarning)

    @staticmethod
    def info(message):
        Warn._printWithColorNBackground(message, Fore.BLUE, Back.BLACK)

    @staticmethod
    def warn(message):
        Warn._printWithColorNBackground(message, Fore.YELLOW, Back.BLACK)

    @staticmethod
    def error(message):
        Warn._printWithColorNBackground(message, Fore.BLACK, Back.RED)
