# plain alias to later on allow changing the logger implementation
#
# The implementation changed (decode_smlm.plex, docs/decode/plex-plan.md §3): DECODE-Plex's
# structlog gave way to the standard library, so that reading a model needs no structlog.  Its
# loggers were called structlog's way, a message and keyword arguments (`logger.info("Loading
# transformation.", reference=reference)`); `KeywordLogger` takes that call and writes the keywords
# after the message.
import logging
from functools import lru_cache


class KeywordLogger:
    """A `logging.Logger` called as structlog's: ``logger.info(msg, key=value, ...)``."""

    def __init__(self, name: str | None = None):
        self._logger = logging.getLogger(name if name else "decode_smlm.plex")

    def _log(self, level: int, msg, *args, **kwargs):
        if not self._logger.isEnabledFor(level):
            return
        exc_info = kwargs.pop("exc_info", None)
        if kwargs:
            msg = f"{msg} " + " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        self._logger.log(level, msg, *args, exc_info=exc_info, stacklevel=3)

    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    warn = warning

    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)


def get_logger(name: str | None = None, *args, **kwargs):
    return KeywordLogger(name)


@lru_cache(10)
def warn_once(logger, msg: str, **kwargs):
    logger.warning(msg, **kwargs)
