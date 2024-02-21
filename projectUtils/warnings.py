import warnings
import colorama

from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)


# Custom warning levels
class CusWarn(UserWarning):
    pass


# Custom Warn class
class Warn:
    @staticmethod
    def _printWithColorNBackground(message, wordsColor, backgroundColor):
        style = Style.NORMAL
        warningMessage = f"\n{style}{wordsColor}{backgroundColor} {message} {Style.RESET_ALL}"
        warnings.warn(warningMessage, CusWarn)

    @staticmethod
    def info(message):
        Warn._printWithColorNBackground(message, Fore.BLACK, Back.BLUE)

    @staticmethod
    def warn(message):
        Warn._printWithColorNBackground(message, Fore.BLACK, Back.YELLOW)

    @staticmethod
    def error(message):
        Warn._printWithColorNBackground(message, Fore.BLACK, Back.RED)
