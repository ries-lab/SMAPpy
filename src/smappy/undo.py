"""The undo stack: the tables a session replaced, and what replaced them.

A step is one *table* change -- a plugin that rewrote the localizations, a
file added or removed -- and nothing else.  Filters, layers, display settings
and the drawn ROI are cheap to put back by hand, and mixing them in would mean
Ctrl+Z sometimes moves a slider and sometimes undoes twenty minutes of drift
correction; that unpredictability is what makes an undo frightening to use.

**Whole tables, bounded by bytes.**  A step keeps a reference to the table it
replaced rather than a diff of it.  That is affordable because of how the
plugins are written: a new table is ``dict(locs.columns)`` with one or two
arrays replaced (`remove_locs.hide`, `math_parser`, `assign_colors`), so the
unchanged columns are *the same arrays* in both tables and cost nothing to
keep.  Only the operations that change the rows -- a deletion, a file appended
-- really pay for a copy.  So the stack is limited by what it actually holds,
counting each array once however many steps share it, and not by a step count
that would be wildly wrong in both directions.

The cap is a fraction of physical memory rather than of *available* memory:
the stack's own arrays are part of what is unavailable, so a fraction of the
free memory would shrink as the history grew and the depth a user got would
depend on the order things happened in and on what else the machine was doing.
A fixed share of the machine is predictable and explainable.  `MAX_STEPS` is
on top of that, because past ten or twenty steps nobody is navigating a list
anyway.
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from .locs import Localizations

# Past this nobody is reading the menu, whatever the memory allows.
MAX_STEPS = 20
# The share of physical memory the stack may hold, and the floor for it on a
# small machine.  A quarter leaves the working table, its grouped form and the
# renderer their room on the 16-32 GB a microscope computer has.
DEFAULT_FRACTION = 0.25
MIN_BUDGET = 512 * 1024 * 1024
# What must stay free whatever the budget says: pushing a step that swaps is
# worse than losing the history.
HEADROOM = 1024 * 1024 * 1024
# What to assume when the machine will not say how much memory it has.
ASSUMED_RAM = 4 * 1024 * 1024 * 1024


def _from_sysconf(*names: str) -> Optional[int]:
    """Bytes from ``os.sysconf``, or None if this platform has no such name."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    for name in names:
        try:
            pages = os.sysconf(name)
        except (ValueError, OSError, AttributeError):
            continue
        if pages and pages > 0:
            return int(pages) * int(page)
    return None


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _windows_memory() -> Optional[tuple]:
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except Exception:
        return None


def memory() -> tuple:
    """``(total, available)`` bytes, best effort.

    `psutil` if the environment happens to have it -- it is not a dependency
    and will not become one for this -- then the platform's own call, then an
    assumption, because a stack that cannot size itself must still work.
    """
    try:
        import psutil                                    # noqa: PLC0415
        info = psutil.virtual_memory()
        return int(info.total), int(info.available)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        found = _windows_memory()
        if found:
            return found
    total = _from_sysconf("SC_PHYS_PAGES")
    # macOS has no SC_AVPHYS_PAGES; there the available figure falls back to
    # the total, and only HEADROOM keeps the stack off a full machine.
    available = _from_sysconf("SC_AVPHYS_PAGES")
    if total:
        return total, (available or total)
    return ASSUMED_RAM, ASSUMED_RAM


def budget(fraction: float = DEFAULT_FRACTION) -> int:
    """How many bytes the stack may hold on this machine."""
    total, _ = memory()
    return max(int(total * fraction), MIN_BUDGET)


def configured() -> "UndoStack":
    """A stack sized by the user's configuration, or by the defaults.

    ``undo_fraction`` (of physical memory) and ``undo_steps`` in the config
    file.  Read here rather than in the session so that a script gets the same
    history a window does; ``undo_fraction: 0`` leaves only `MIN_BUDGET`,
    which on a large table is the single step the stack always keeps.
    """
    from . import config                                     # noqa: PLC0415

    try:
        fraction = float(config.get("undo_fraction", DEFAULT_FRACTION))
        steps = int(config.get("undo_steps", MAX_STEPS))
    except (TypeError, ValueError):
        fraction, steps = DEFAULT_FRACTION, MAX_STEPS
    return UndoStack(fraction=max(0.0, fraction), max_steps=max(1, steps))


def table_bytes(locs: Optional[Localizations], seen: Optional[set] = None) -> int:
    """What a table costs *in addition to* the arrays already in ``seen``.

    Columns are counted by the buffer they are views of (`Localizations.extend`
    keeps a column as a view of a larger buffer), and each one once: two steps
    that differ in a single column cost one column, which is the whole reason
    whole tables are affordable here.
    """
    if locs is None:
        return 0
    seen = seen if seen is not None else set()
    total = 0
    for column in locs.columns.values():
        base = column
        while getattr(base, "base", None) is not None:
            base = base.base
        key = id(base)
        if key in seen:
            continue
        seen.add(key)
        total += int(getattr(base, "nbytes", 0))
    return total


@dataclass
class Edit:
    """One table change, and what it takes to go back to before it.

    ``locs`` is the table as it was *before*; the label is what the menu says
    ("Undo drift correction"), and `text` is the plugin's own one-line result.

    The table is the whole of the step.  Which files the session had is in
    ``locs.metadata["files"]``, written there by `Session.add_file`, so a step
    that added or removed a file restores the list by being put back -- there
    is nothing to keep in step with.
    """
    label: str
    locs: Localizations
    text: str = ""

    def nbytes(self, seen: Optional[set] = None) -> int:
        return table_bytes(self.locs, seen)


class UndoStack:
    """Steps back and steps forward, under a memory cap.

    Nothing here knows about the session: an `Edit` is data, and the caller
    decides what putting one back means.  That is what makes it testable
    without a table, a layer or a window.
    """

    def __init__(self, fraction: float = DEFAULT_FRACTION,
                 max_steps: int = MAX_STEPS, limit: Optional[int] = None):
        """``limit`` overrides the machine's share outright, in bytes; it is
        how a test pins a budget that does not depend on the machine."""
        self.fraction = fraction
        self.max_steps = max_steps
        self.limit = limit
        self._undo: List[Edit] = []
        self._redo: List[Edit] = []

    # ------------------------------------------------------------- state
    def __len__(self) -> int:
        return len(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def entries(self) -> List[Edit]:
        """The undo steps, most recent first -- what the submenu lists."""
        return list(reversed(self._undo))

    def redo_entries(self) -> List[Edit]:
        return list(reversed(self._redo))

    def labels(self) -> List[str]:
        return [e.label for e in self.entries()]

    def redo_labels(self) -> List[str]:
        return [e.label for e in self.redo_entries()]

    def nbytes(self) -> int:
        """What the whole history holds, counting a shared array once."""
        seen: set = set()
        return sum(e.nbytes(seen) for e in self._undo + self._redo)

    # ------------------------------------------------------------ editing
    def push(self, edit: Edit) -> None:
        """Record a step, and drop the redo history: it is a different future."""
        self._redo.clear()
        self._undo.append(edit)
        self.trim()

    def undo(self, redo_of: Optional[Edit] = None) -> Optional[Edit]:
        """Take the most recent step off; ``redo_of`` is the state it undoes.

        The caller hands in the *current* state as ``redo_of`` so that redo has
        somewhere to come back to -- the stack never reads a session.
        """
        if not self._undo:
            return None
        edit = self._undo.pop()
        if redo_of is not None:
            self._redo.append(redo_of)
            self.trim()
        return edit

    def redo(self, undo_of: Optional[Edit] = None) -> Optional[Edit]:
        if not self._redo:
            return None
        edit = self._redo.pop()
        if undo_of is not None:
            self._undo.append(undo_of)
            self.trim()
        return edit

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    # ------------------------------------------------------------ the cap
    def trim(self) -> None:
        """Drop the oldest steps until the history fits.

        The most recent step always survives, whatever it costs: a single
        table larger than the whole budget must still be undoable once, which
        is the behaviour this stack replaced.  The redo side goes first --
        having to redo something is rarer than having to undo it, and the
        alternative is losing a step that was actually taken.
        """
        if self.limit is not None:
            limit, crowded = self.limit, False
        else:
            limit = budget(self.fraction)
            _, available = memory()
            crowded = available < HEADROOM
        while len(self._undo) > self.max_steps:
            self._undo.pop(0)
        while len(self._redo) > self.max_steps:
            self._redo.pop(0)
        while len(self._undo) + len(self._redo) > 1 and self.nbytes() > limit:
            if self._redo:
                self._redo.pop(0)
            else:
                self._undo.pop(0)
        # a machine that is nearly full keeps one step and no more, rather
        # than pushing the working set into swap for the sake of a history
        if crowded:
            while len(self._undo) > 1:
                self._undo.pop(0)
            self._redo.clear()
