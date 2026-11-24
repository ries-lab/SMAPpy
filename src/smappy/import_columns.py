"""Which columns of a localization file to read: the choice, and when to ask.

A file often holds many more columns than anything reads -- 30 in a real
SMAP ``_sml.mat``, 13 of them used -- and a large one does not fit a
microscope PC: the 57 M localization file needs about 9 GB with every column.
So the GUI reads a file one of three ways (*columns* on the File tab's *Load*
line, `plugins.file.OpenSettings`):

* **standard** (the default): every column but those the format writes and
  nothing reads (`io.formats.Reader.drop`) -- and if even that would take
  more than `SHARE` of the memory available, the column dialog asks first,
  with the standard ones ticked and the memory each takes beside it;
* **all**: every column;
* **choose on import**: the dialog every time.

The decision is here, without Qt, so it can be tested and a script can ask
it (`plan`); the dialog is `gui.columns_dialog`.  A format that cannot list
its columns without reading the file (a csv, a MATLAB v5 file, MINFLUX) is
read by the policy alone: there is nothing to put in a dialog before reading.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

STANDARD, ALL, ASK = "standard", "all", "ask"
CHOICES = ((STANDARD, "standard, ask if large"), (ALL, "all"),
           (ASK, "choose on import"))

#: a file whose standard columns would take more than this share of the
#: memory available is asked about: the rest is for merging, rendering and
#: an undo step, each a fraction of the table again
SHARE = 0.5


@dataclass
class Plan:
    """What to do with one file: read it with ``columns`` (`io.formats.load`),
    or -- ``ask`` -- show the dialog on ``found`` first, saying ``why``."""
    columns: object = STANDARD
    ask: bool = False
    found: Optional[list] = None          # `io.formats.Column`s
    why: str = ""
    available: Optional[int] = None       # bytes


def available_memory() -> Optional[int]:
    """Bytes of memory available, or None where it cannot be told.  (macOS
    gives only the total, `undo.memory`'s fallback; there it is the total.)"""
    try:
        from .undo import memory
        return int(memory()[1])
    except Exception:          # noqa: BLE001 -- a guess is no reason to refuse a file
        return None


def plan(path, policy: str = STANDARD, reader=None,
         available: Optional[int] = None) -> Plan:
    """How to read ``path`` under ``policy`` (`STANDARD`, `ALL`, `ASK`).

    ``available`` is the memory to compare with, `available_memory` if not
    given.  A file whose format cannot list its columns is never asked about.
    """
    from .io.formats import catalog, estimate_bytes, reader_for
    if policy == ALL:
        return Plan(ALL)
    try:
        reader = reader or reader_for(path)
    except ValueError:
        return Plan(STANDARD)
    found = catalog(path, reader)
    if not found:
        return Plan(STANDARD)
    if policy == ASK:
        return Plan(STANDARD, True, found, f"{Path(path).name}: which columns to read?")
    available = available_memory() if available is None else available
    need = estimate_bytes(found, STANDARD)
    if available and need > SHARE * available:
        return Plan(STANDARD, True, found,
                    f"{Path(path).name} needs {gigabytes(need)} with its standard "
                    f"columns, of {gigabytes(available)} available: leave out the "
                    f"columns you will not use", available)
    return Plan(STANDARD, available=available)


def gigabytes(n: float) -> str:
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


def standard_names(found: List) -> List[str]:
    """The columns *standard* reads, of a file's `io.formats.Column`s."""
    from .io.formats import REQUIRED
    return [c.name for c in found if c.standard or c.name in REQUIRED]
