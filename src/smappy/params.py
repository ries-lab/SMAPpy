"""How a settings field is shown: `ParamInfo`, and `param` to attach it.

A module of its own, with no smappy imports, so that the algorithm modules
(`pipeline`, `drift`, `rcc`, `group`, `calibrate`) can declare how their
settings are shown on the fields themselves -- the one way there is -- without
importing the plugin machinery.  `smappy.plugins` re-exports both.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any, Optional, Sequence

_META = "smappy"


@dataclass
class ParamInfo:
    """How a settings field is presented.  Everything is optional."""
    label: Optional[str] = None
    unit: Optional[str] = None
    help: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[Sequence] = None   # values, or (value, label) pairs; may be a callable
    advanced: bool = False      # hidden behind "more"
    hidden: bool = False        # scripting only
    # "open_file", "save_file", "dir": a path with a browse button.
    # "text": a line that is given the width of the form, for something
    # written rather than dialled -- an expression, a list of names
    kind: Optional[str] = None
    file_filter: str = ""       # a Qt-style name filter for those, "TIFF (*.tif)"
    # for a part: shown folded until it is opened.  A chain's steps are, so a
    # chain of eight plugins reads as eight lines rather than eight forms
    collapsed: bool = False
    # a bool that turns on optional work which costs time per run -- a
    # bootstrap, a fit beside the numbers -- and that a run over many regions
    # starts without (`fast_values`; docs/regions.md, "Slow, optional parts")
    slow: bool = False


def param(default: Any = MISSING, *, default_factory: Any = MISSING, **info):
    """A dataclass ``field`` carrying :class:`ParamInfo` for the GUI."""
    metadata = {_META: ParamInfo(**info)}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)
