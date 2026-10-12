"""Handing a session to a new instance of the program: *Open in a new window*.

A fit still appending to its table cannot take a drift correction, and two
analyses of one dataset are easier side by side than one after the other.
Both want the session as it is now, in a window of its own that does not
change with this one.  That is a second instance, a process of its own, not
a second set of windows in this one: closing, quitting and a crash then stay
where they happen -- Ctrl+Q in the copy does not end the acquisition's fit --
and the copy competes with the live view for nothing.

**The table goes through shared memory; everything else as JSON.**  Measured
on 5M localizations x 15 columns: 0.36 s to copy 320 MB into a shared block,
against 11 s to write the same table as `Session.save` does (lzf) and 2.6 s
to read it back.  The fit's own file is not read instead: it is open for
writing, and HDF5 refuses a second process (file locking) or, with locking
off and no SWMR, hands it torn reads -- 22 of 25 in a test while blocks were
being appended.  And it holds only the raw localizations, not the layers,
filters, ROIs and names the copy should open with.

The protocol is three files' worth of state, all temporary:

1. `offer` copies the columns into a shared block and writes a JSON file:
   the block's name and layout, the table's metadata as a file would carry
   it, the layers, the view, the workspace and the tool results.
2. `launch` starts ``smappy-gui --handoff FILE``.
3. The new instance (`receive`) copies the columns out, closes the block
   and deletes the JSON file.  That deletion is the signal: `Offer.taken`
   sees it, and `Offer.release` closes and unlinks the block -- which on
   Windows must not happen sooner, because a block there lives only while a
   handle to it is open.  An offer nobody takes is released after
   `Offer.timeout` all the same.

The raw camera frames are not handed on: they can be gigabytes, and the copy
is for working with the localizations.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .locs import Localizations

VERSION = 1


# ------------------------------------------------------------- the state
def _plain(value):
    """JSON-able: dataclasses as dicts, arrays as lists, tuples as lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_default(value):
    out = _plain(value)
    if out is value:
        raise TypeError(f"not JSON-able: {type(value).__name__}")
    return out


def _from_plain(cls, data):
    """A dataclass ``cls`` from what `_plain` made of one; unknown keys are
    dropped and missing ones keep their defaults, so a copy opened by another
    version still opens."""
    if not isinstance(data, dict):
        return cls()
    hints = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, value in data.items():
        f = hints.get(name)
        if f is None or not f.init:
            continue
        default = (f.default_factory() if f.default_factory is not dataclasses.MISSING
                   else f.default)
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            value = _from_plain(type(default), value)
        elif isinstance(value, list) and isinstance(default, tuple):
            value = tuple(value)
        elif isinstance(value, list) and name.endswith("range"):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def session_state(session, gui_state=None, view=None) -> Dict[str, Any]:
    """Everything the new instance opens with, but the table's columns."""
    layers = []
    for layer in session.layers:
        if layer.is_image:
            continue
        ranges = layer.ranges
        layers.append({
            "name": layer.name, "visible": bool(layer.visible),
            "grouped": bool(layer.grouped),
            "bounds": {f: [lo, hi] for f, (lo, hi) in ranges.items()},
            "files": None if layer.files is None else list(layer.files),
            "render": _plain(layer.state.settings),
            "display": _plain(layer.get_display()),
        })
    return {
        "version": VERSION,
        "metadata": _plain(session.file_metadata()),
        "files": [_plain(f.to_dict()) for f in session.files],
        "results": _plain(session.results),
        "layers": layers,
        "group_settings": _plain(session.group_settings),
        "view": _plain(view) if view is not None else None,
        "gui_state": _plain(gui_state) if gui_state is not None else None,
        # what the copy was taken from, for its log
        "source": str(session.path) if session.path else session.display_name,
    }


def snapshot_identity(session) -> Tuple[str, str]:
    """A name and colour for a copy of ``session``: its name, marked as a
    snapshot, and a colour other than its own."""
    from .session import COLOURS
    name = session.display_name or "session"
    if session.live and len(session.locs) and "frame" in session.locs:
        frame = int(np.max(np.asarray(session.locs["frame"])))
        name = f"{name}, snapshot at frame {frame:,}"
    else:
        name = f"{name}, copy"
    order = ("orange", "green", "purple", "red", "blue", "yellow", "brown")
    return name, next(c for c in order if c in COLOURS and c != session.colour)


# ------------------------------------------------------------- offering
class Offer:
    """A session put up for a new instance to take; see the module notes."""

    timeout = 120.0                 # seconds a taker has to start and copy

    def __init__(self, path: Path, block):
        self.path = path
        self.block = block
        self.made = time.monotonic()

    @property
    def taken(self) -> bool:
        return not self.path.exists()

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.made > self.timeout

    def release(self) -> None:
        """Free the block (and the JSON, if nobody took it).  Idempotent."""
        if self.block is not None:
            self.block.close()
            try:
                _unlink(self.block)
            except FileNotFoundError:
                pass
            self.block = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _shared_memory(size: int, name: Optional[str] = None):
    """A block that this process's resource tracker leaves alone.

    The tracker unlinks a block at exit -- in the taker too, before Python
    3.13, where merely attaching registered it -- and the protocol decides
    who unlinks: the offering side, once the block is taken.
    """
    from multiprocessing import shared_memory
    create = name is None
    try:
        block = shared_memory.SharedMemory(name=name, create=create,
                                           size=max(size, 1), track=False)
    except TypeError:                # Python < 3.13: no `track`
        block = shared_memory.SharedMemory(name=name, create=create,
                                           size=max(size, 1))
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(block._name, "shared_memory")
        except Exception:
            pass
    return block


def _unlink(block) -> None:
    """Unlink a block made by `_shared_memory`.

    Before Python 3.13 `unlink` also unregisters the block from the resource
    tracker, which `_shared_memory` has already done; the tracker then prints
    a KeyError traceback for every handoff.  Registering it again first keeps
    the two balanced.
    """
    if getattr(block, "_track", True) and os.name == "posix":
        try:
            from multiprocessing import resource_tracker
            resource_tracker.register(block._name, "shared_memory")
        except Exception:
            pass
    block.unlink()


def offer(session, *, name: str = "", colour: str = "", gui_state=None,
          view=None, run=None, folder=None) -> Offer:
    """Put ``session`` up for a new instance: its table in a shared block,
    the rest in a JSON file whose path `launch` hands on.

    ``run`` is ``(plugin, settings)`` for the new instance to run once it has
    the session: a table change asked for while a fit was still appending
    here, sent where it can be applied."""
    locs = session.locs.snapshot() if session.live else session.locs
    columns = {k: np.ascontiguousarray(v) for k, v in locs.columns.items()}
    size = sum(c.nbytes for c in columns.values())
    block = _shared_memory(size)
    layout, offset = [], 0
    for key, column in columns.items():
        np.ndarray(column.shape, column.dtype, block.buf, offset)[...] = column
        layout.append({"name": key, "dtype": column.dtype.str,
                       "shape": list(column.shape), "offset": offset})
        offset += column.nbytes
    state = session_state(session, gui_state, view)
    state["identity"] = {"name": name, "colour": colour}
    if run is not None:
        from .plugins import settings_values
        plugin, settings = run
        state["run"] = {"plugin": plugin.path,
                        "values": settings_values(settings) if settings is not None else {}}
    state["block"] = {"name": block.name, "size": size, "columns": layout}
    handle, where = tempfile.mkstemp(prefix="smappy-handoff-", suffix=".json",
                                     dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(state, f, default=_json_default)
    except Exception:
        block.close()
        _unlink(block)
        Path(where).unlink(missing_ok=True)
        raise
    return Offer(Path(where), block)


def launch(path) -> subprocess.Popen:
    """Start ``smappy-gui --handoff path`` in a process of its own."""
    command = [sys.executable, "-m", "smappy.cli.gui", "--handoff", str(path)]
    kwargs: Dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True   # not ended with this one's terminal
    return subprocess.Popen(command, **kwargs)


# ------------------------------------------------------------- taking
def receive(path) -> Tuple[Localizations, Dict[str, Any]]:
    """Take an offer: the table (copied out of the block) and the state.

    Deletes the JSON file once the columns are copied, which tells the
    offering side the block can go.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    if state.get("version") != VERSION:
        raise ValueError(f"handed a session of version {state.get('version')}, "
                         f"this program reads version {VERSION}")
    spec = state.pop("block")
    block = _shared_memory(spec["size"], name=spec["name"])
    try:
        columns = {}
        for c in spec["columns"]:
            view = np.ndarray(tuple(c["shape"]), np.dtype(c["dtype"]),
                              block.buf, c["offset"])
            columns[c["name"]] = np.array(view, copy=True)
            del view                 # no export may outlive `close`
    finally:
        block.close()
    path.unlink(missing_ok=True)
    return Localizations(columns, dict(state.get("metadata") or {})), state


def take(path, group_settings=None, progress=None):
    """`receive`, and link the table as opening a file would:
    ``(locs, state, grouped)``.  The slow half, for a worker thread."""
    from .session import DEFAULT_GROUP_SETTINGS, GROUPED_BY_DEFAULT
    locs, state = receive(path)
    grouped = None
    if GROUPED_BY_DEFAULT and len(locs) and "frame" in locs:
        from .group import group as link
        settings = _group_settings(state, group_settings or DEFAULT_GROUP_SETTINGS)
        grouped, _ = link(locs, settings,
                          progress=(lambda text, _f: progress(f"Grouper: {text}"))
                          if progress else None)
    return locs, state, grouped


def _group_settings(state: Dict[str, Any], default):
    """The link settings the state was taken with: the session's, or -- from
    an instance that kept them per layer -- the first layer's."""
    from .group import GroupSettings
    saved = state.get("group_settings")
    if not saved:
        saved = next(iter(state.get("layers") or []), {}).get("group_settings")
    return _from_plain(GroupSettings, saved) if saved else default


def apply_state(session, locs: Localizations, state: Dict[str, Any],
                grouped: Optional[Localizations] = None) -> None:
    """Make an empty ``session`` the one that was handed over (Qt-free).

    Goes through `add_file`, as opening a file would, so the log, the ROI,
    the ROI manager's project and the name come from the metadata the same
    way; then the files, results and layers are put back as they were.
    """
    from .io.formats import FileInfo
    from .render import DisplaySettings, RenderSettings
    identity = state.get("identity") or {}
    name = identity.get("name") or "snapshot"
    # before the table: the grouped table handed in was linked with these
    settings = _group_settings(state, session.group_settings)
    if settings != session.group_settings:
        session.set_group_settings(settings)
    session.add_file(locs, FileInfo(name=name, path="", format="snapshot",
                                    n=len(locs)), grouped=grouped)
    files = [FileInfo(**{k: v for k, v in f.items() if k != "n"})
             for f in state.get("files") or []]
    if files:
        session.files = files
        session.locs.metadata["files"] = [f.to_dict() for f in files]
        if "filenumber" in locs:
            # `add_file` numbered every row as one file; the table's own
            # numbers are the ones its layers and file list refer to
            session.locs.columns["filenumber"] = np.asarray(locs["filenumber"])
    session.path = None            # nowhere to save back to until asked
    # its log says what it is a copy of, in place of the "load" of a file
    # with no name that `add_file` wrote
    if session.history and session.history[-1].get("what") == "load":
        session.history.pop()
    session.log("copy", f"{len(locs)} localizations of {state.get('source') or 'a session'}")
    session.results = dict(state.get("results") or {})
    session.changed("results")
    configs = state.get("layers") or []
    session.set_layer_configs([{"grouped": c.get("grouped"),
                                "bounds": c.get("bounds") or {},
                                "files": c.get("files")} for c in configs])
    layers = [l for l in session.layers if not l.is_image]
    for layer, config in zip(layers, configs):
        layer.name = config.get("name") or layer.name
        layer.visible = bool(config.get("visible", True))
        if config.get("render"):
            layer.state.settings = _from_plain(RenderSettings, config["render"])
        if config.get("display"):
            layer.set_display(_from_plain(DisplaySettings, config["display"]))
    session.rename(identity.get("name") or "", colour=identity.get("colour") or "")
    session.changed("layers")
