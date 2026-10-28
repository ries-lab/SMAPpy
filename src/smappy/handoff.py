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

The layers and the 3D view are serialised once, here, for two uses: the
handoff, and the session block a saved file carries (`session_block`,
`/gui/session`) for *File > Open and restore* to put back (`gui.restore`).
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


def layer_states(session) -> list:
    """Every layer as it stands, in order: a localization layer's filter,
    files, grouping, render and display settings; an image layer's display,
    which image it shows (`_image_key`) and how."""
    layers = []
    for layer in session.layers:
        if layer.is_image:
            if layer.image is None or layer.image.is_empty:
                continue
            layers.append({
                "kind": "image", "name": layer.name, "visible": bool(layer.visible),
                "image": _image_key(session, layer.image),
                "frame": int(layer.frame), "image_show": layer.image_show,
                "image_planes": None if layer.image_planes is None
                else list(layer.image_planes),
                "follow_z": bool(layer.follow_z), "image_3d": layer.image_3d,
                "display": _plain(layer.get_display()),
            })
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
    return layers


def _image_key(session, image) -> Dict[str, Any]:
    """What finds an image again in the session a file opens: its name,
    kind and shape, and which of the images so called it is.  Not the object
    nor a place in a list -- a file's images come back as new objects, in
    the order `saved_images` wrote them."""
    same = [im for im in session.images
            if (im.name, im.kind) == (image.name, image.kind)]
    nth = next((n for n, im in enumerate(same) if im is image), 0)
    return {"name": image.name or "", "kind": image.kind,
            "shape": list(np.shape(image.data)), "nth": nth}


def _find_image(session, key: Dict[str, Any]):
    """The session's image `_image_key` describes, or None."""
    same = [im for im in session.images if not im.is_empty
            and (im.name or "", im.kind) == (key.get("name") or "", key.get("kind"))]
    shaped = [im for im in same
              if list(np.shape(im.data)) == list(key.get("shape") or [])]
    pool = shaped or same
    if not pool:
        return None
    nth = int(key.get("nth") or 0)
    return pool[nth] if nth < len(pool) else pool[0]


def apply_layers(session, configs) -> None:
    """Make the session's layers those `layer_states` described, in its
    order.  An image layer whose image the session does not have (a session
    handed over without its frames) is left out; localization layers beyond
    the list are removed, so the count is the saved one."""
    from .render import DisplaySettings, RenderSettings
    configs = list(configs or [])
    locs_configs = [c for c in configs if c.get("kind", "locs") != "image"]
    if not locs_configs:
        return
    session.set_layer_configs([{"grouped": c.get("grouped"),
                                "bounds": c.get("bounds") or {},
                                "files": c.get("files")} for c in locs_configs])
    locs_layers = [l for l in session.layers if not l.is_image]
    for layer, config in zip(locs_layers, locs_configs):
        layer.name = config.get("name") or layer.name
        layer.visible = bool(config.get("visible", True))
        if config.get("render"):
            layer.state.settings = _from_plain(RenderSettings, config["render"])
        if config.get("display"):
            layer.set_display(_from_plain(DisplaySettings, config["display"]))
    locs_layers = locs_layers[:len(locs_configs)]
    # the image layers there are, matched to the saved ones by their image;
    # a saved one with no match here is made, one with no image dropped
    images = [l for l in session.layers if l.is_image]
    ordered, kept = [], iter(locs_layers)
    from .session import Layer
    for config in configs:
        if config.get("kind", "locs") != "image":
            ordered.append(next(kept))
            continue
        image = _find_image(session, config.get("image") or {})
        if image is None:
            continue
        layer = next((l for l in images if l.image is image), None)
        if layer is None:
            layer = Layer.from_image(image)
        else:
            images.remove(layer)
        layer.name = config.get("name") or layer.name
        layer.visible = bool(config.get("visible", True))
        layer.frame = int(config.get("frame") or 0)
        planes = config.get("image_planes")
        layer.set_image_view(config.get("image_show") or "slice",
                             tuple(planes) if planes else None,
                             bool(config.get("follow_z")),
                             config.get("image_3d") or "plane")
        if config.get("display"):
            layer.set_display(_from_plain(DisplaySettings, config["display"]))
        ordered.append(layer)
    session.layers = ordered
    session.changed("layers")


def view_state(session, projection=None) -> Dict[str, Any]:
    """The 3D view: its projection (``projection``, the window's own copy,
    when it has one) and the slab -- the session's, so a script has it too."""
    slab = session.slab
    return {
        "projection": _plain(projection if projection is not None
                             else session.projection),
        "slab": None if slab is None else {"center": _plain(slab.center),
                                           "size": _plain(slab.size),
                                           "angle": float(slab.angle)},
        "slab_follows_roi": bool(session.slab_follows_roi),
        "slab_z": None if session.slab_z is None else list(session.slab_z),
        "select_in_slab": bool(session.select_in_slab),
    }


def apply_view_state(session, state: Optional[Dict[str, Any]]) -> None:
    """Put back what `view_state` kept."""
    if not state:
        return
    from .view3d import Projection, Slab
    if state.get("projection"):
        session.set_projection(_from_plain(Projection, state["projection"]))
    session.slab_z = tuple(state["slab_z"]) if state.get("slab_z") else None
    session.select_in_slab = bool(state.get("select_in_slab"))
    slab = state.get("slab")
    follows = bool(state.get("slab_follows_roi", True))
    if slab:
        session.set_slab(Slab(slab["center"], slab["size"], float(slab.get("angle") or 0.0)),
                         follow_roi=follows)
    else:
        session.slab_follows_roi = follows


#: the version of the session block a saved file carries (`session_block`)
BLOCK_VERSION = 1


def session_block(session, view=None, projection=None, windows=None) -> Dict[str, Any]:
    """What a saved file needs beyond its table, ROIs and tabs to bring the
    session back (*File > Open and restore*): the layers, the merge settings,
    the 3D view, and -- from the GUI -- the 2D view's range and the windows.
    The table's metadata, files and results are in the file already."""
    return {
        "version": BLOCK_VERSION,
        "layers": layer_states(session),
        "group_settings": _plain(session.group_settings),
        "view": _plain(view) if view is not None else None,
        "view3d": view_state(session, projection),
        "windows": _plain(windows) if windows is not None else None,
    }


def apply_session_block(session, block: Optional[Dict[str, Any]]) -> None:
    """The Qt-free half of a restore: merge settings, layers and 3D view.
    The 2D range and the windows are the GUI's (`gui.restore`)."""
    if not block:
        return
    from .group import GroupSettings
    if block.get("group_settings"):
        settings = _from_plain(GroupSettings, block["group_settings"])
        if settings != session.group_settings:
            session.set_group_settings(settings)
    apply_layers(session, block.get("layers"))
    apply_view_state(session, block.get("view3d"))


def session_state(session, gui_state=None, view=None) -> Dict[str, Any]:
    """Everything the new instance opens with, but the table's columns."""
    return {
        "version": VERSION,
        "metadata": _plain(session.file_metadata()),
        "files": [_plain(f.to_dict()) for f in session.files],
        "results": _plain(session.results),
        "layers": layer_states(session),
        "group_settings": _plain(session.group_settings),
        "view": _plain(view) if view is not None else None,
        "view3d": view_state(session),
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


def own_session(windows: Optional[bool] = None) -> Dict[str, Any]:
    """Popen's arguments for a window that is not ended with this one's terminal.

    On POSIX a session of its own.  On Windows the copy otherwise shared the
    original's console, so Ctrl+C there or closing it ended both: a group of
    its own takes it out of Ctrl+C, and a console of its own (hidden, so
    that what it prints still goes somewhere) out of the close.  The same
    two flags as `updater._detached`.
    """
    if sys.platform.startswith("win") if windows is None else windows:
        return {"creationflags": 0x00000200 | 0x08000000}   # NEW_PROCESS_GROUP | NO_WINDOW
    return {"start_new_session": True}


def launch(path) -> subprocess.Popen:
    """Start ``smappy-gui --handoff path`` in a process of its own."""
    command = [sys.executable, "-m", "smappy.cli.gui", "--handoff", str(path)]
    return subprocess.Popen(command, **own_session())


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
    """`receive`, and link the table if a layer handed over shows merged
    blinks: ``(locs, state, grouped)``.  The slow half, for a worker thread
    -- linked here and not when the layers are read, so the new window does
    not wait for it, and not at all when no layer shows blinks."""
    from .session import DEFAULT_GROUP_SETTINGS
    locs, state = receive(path)
    grouped = None
    wanted = any(c.get("grouped") for c in state.get("layers") or [])
    if wanted and len(locs) and "frame" in locs:
        from .group import group as link
        settings = _group_settings(state, group_settings or DEFAULT_GROUP_SETTINGS)
        grouped, _ = link(locs, settings,
                          progress=(lambda text, _f: progress(f"merging blinks: {text}"))
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
    identity = state.get("identity") or {}
    name = identity.get("name") or "snapshot"
    # before the table: the grouped table handed in was linked with these
    settings = _group_settings(state, session.group_settings)
    if settings != session.group_settings:
        session.set_group_settings(settings)
    session.add_file(locs, FileInfo(name=name, path="", format="snapshot",
                                    n=len(locs)), grouped=grouped)
    files = [FileInfo.from_dict(f) for f in state.get("files") or []]
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
    apply_layers(session, state.get("layers"))
    apply_view_state(session, state.get("view3d"))
    session.rename(identity.get("name") or "", colour=identity.get("colour") or "")
    session.changed("layers")
