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
    def _print_warning(message, level, foreground_color, background_color):
        style = Style.NORMAL
        warning_message = f"{style}{foreground_color}{background_color} {message} {Style.RESET_ALL}"
        warnings.warn(warning_message, CustomWarning)

    @staticmethod
    def info(message):
        Warn._print_warning(message, 'info', Fore.BLUE, Back.BLACK)

    @staticmethod
    def warn(message):
        Warn._print_warning(message, 'warning', Fore.YELLOW, Back.BLACK)

    @staticmethod
    def error(message):
        Warn._print_warning(message, 'error', Fore.BLACK, Back.RED)