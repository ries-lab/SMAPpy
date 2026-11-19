import functools
from typing import Optional

from structlog import get_logger

logger = get_logger()


def experimental(tested: bool = False, level: Optional[str] = "info"):
    """
    Decorator for experimental functions.

    Args:
        tested: indicate whether there is a test for the function
        level: log level
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_fn = getattr(logger, level)
            log_fn(f"Running experimental function `{func.__name__}`", tested=tested)
            return func(*args, **kwargs)

        return wrapper
    return decorator
