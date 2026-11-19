from abc import abstractmethod
from typing import Any, Callable, Literal

from decode.generic import logging


class Action:
    @abstractmethod
    def __call__(self, ctxt: dict[str, Any]):
        raise NotImplementedError


class LogAction(Action):
    def __init__(
        self,
        level: str,
        logger=None,
        msg: str | None = None,
        ctxt_fn: Callable | Literal["unpack"] | None = "unpack",
    ):
        self.level = level
        self.logger = logger if logger is not None else logging.get_logger(__name__)
        self.msg = msg
        self.ctxt_fn = ctxt_fn

    def __call__(self, ctxt: dict[str, Any]):
        args = (self.msg,) if self.msg is not None else tuple()
        kwargs = {}
        if self.ctxt_fn is None:
            args = args + (ctxt,)
        elif self.ctxt_fn == "unpack":
            kwargs = ctxt
        else:
            args = args + (self.ctxt_fn(ctxt),)

        getattr(self.logger, self.level)(*args, **kwargs)
