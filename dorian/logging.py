import sys


class Colors:
    """ANSI color codes for terminal output, with TTY detection."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    _enabled = True

    @classmethod
    def _check_tty(cls):
        if not sys.stdout.isatty():
            cls.disable()

    @classmethod
    def disable(cls):
        cls._enabled = False
        cls.BLUE = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.DIM = ""
        cls.BOLD = ""
        cls.RESET = ""


Colors._check_tty()


def info(message):
    """Print an info message with blue [INFO] prefix."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")


def success(message):
    """Print a success message with green checkmark."""
    print(f"{Colors.GREEN}\u2714{Colors.RESET} {message}")


def warning(message):
    """Print a warning message with yellow [WARNING] prefix."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")


def error(message):
    """Print an error message with red [ERROR] prefix to stderr."""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}", file=sys.stderr)


def debug(message):
    """Print a debug message with dim [DEBUG] prefix and separator lines."""
    sep = Colors.DIM + "-" * 40 + Colors.RESET
    print(f"{sep}\n{Colors.DIM}[DEBUG]{Colors.RESET} {message}\n{sep}")


def banner(message):
    """Print a separator banner."""
    line = "=" * 60
    print(f"{Colors.BOLD}{line}\n  {message}\n{line}{Colors.RESET}")
