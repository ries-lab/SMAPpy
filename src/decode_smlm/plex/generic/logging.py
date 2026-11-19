# plain alias to later on allow changing the logger implementation
from functools import lru_cache

from structlog import get_logger as get_logger_structlog


def get_logger(name: str, *args, **kwargs):
    return get_logger_structlog(name, *args, **kwargs)


@lru_cache(10)
def warn_once(logger, msg: str, **kwargs):
    logger.warning(msg, **kwargs)
