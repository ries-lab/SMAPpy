"""The state a GUI or a script works on: one table, its layers, undo, history.

No Qt in here.  The GUI subscribes to `on_change` and redraws; a script uses
it the same way without subscribing to anything.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .filter import LocFilter
from .locs import Localizations
from .plugins import Plugin, Result, Selection
from .render import DisplaySettings, RenderSettings
from .viewer import DEFAULT_BOUNDS, ViewState


class Layer:
    """A filter and how its localizations are drawn.

    Wraps a `ViewState`, which owns the filter, the spatial index and the
    render/display settings -- everything the render view needs per layer.
    """

    def __init__(self, locs: Localizations, name: str = "layer 1",
                 defaults: bool = True,
                 settings: Optional[RenderSettings] = None,
                 display: Optional[DisplaySettings] = None):
        self.name = name
        self.visible = True
        self.state = ViewState(locs, settings, display)
        if defaults:
            for field, (lo, hi) in DEFAULT_BOUNDS.items():
                if field in locs:
                    self.filter.set(field, lo, hi)

    @property
    def locs(self) -> Localizations:
        """The table this layer draws: grouped or not."""
        return self.state.locs

    @property
    def filter(self) -> LocFilter:
        return self.state.filter

    def rebind(self, locs: Localizations) -> None:
        """Point at a new table, keeping the bounds and display the user set."""
        old = self.state
        self.state = ViewState(locs, old.settings, old.display)
        for field, (lo, hi) in old.sets["ungrouped"].filter.ranges.items():
            if field in locs:
                self.filter.set(field, lo, hi)
        if old.use_grouped:
            self.state.show_grouped(True)

    @property
    def grouped(self) -> bool:
        return self.state.use_grouped

    def show_grouped(self, on: bool) -> None:
        """Draw one entry per blink instead of one per frame.  Links on first use."""
        self.state.show_grouped(on)

    def selection(self, index: int = 0) -> Selection:
        """The *ungrouped* localizations this layer's filter keeps.

        Plugins work on the full table, so a grouped layer hands back the
        ungrouped filter, which is what it would apply to it.
        """
        f = self.state.sets["ungrouped"].filter
        return Selection(f.mask, layer=index, name=self.name)


class Session:
    def __init__(self, locs: Optional[Localizations] = None, path=None):
        self.locs = locs if locs is not None else Localizations({}, {})
        self.path: Optional[Path] = Path(path) if path else None
        self.layers: List[Layer] = [Layer(self.locs)]
        self.history: List[Dict] = []
        self._undo: Optional[Localizations] = None
        self._listeners: List[Callable[[str], None]] = []

    # ----------------------------------------------------------- observers
    def on_change(self, callback: Callable[[str], None]) -> None:
        """``callback(what)``: "locs" (new table), "layer" (a filter or display
        setting changed), "layers" (added/removed/visibility) or "history"."""
        self._listeners.append(callback)

    def changed(self, what: str) -> None:
        for cb in self._listeners:
            cb(what)

    # ---------------------------------------------------------------- data
    def load(self, path) -> None:
        from .io.hdf5 import load_localizations
        self.path = Path(path)
        self.set_locs(load_localizations(path), undoable=False)
        self.history.clear()
        self.log("load", str(self.path))

    def save(self, path=None) -> Path:
        from .io.hdf5 import save_localizations
        path = Path(path or self.path)
        metadata = dict(self.locs.metadata)
        metadata["history"] = self.history
        save_localizations(path, self.locs, metadata)
        self.path = path
        return path

    def set_locs(self, locs: Localizations, undoable: bool = True) -> None:
        self._undo = self.locs if undoable else None
        self.locs = locs
        if undoable:                       # the same data, corrected: keep the layers
            for layer in self.layers:
                layer.rebind(locs)
        else:                              # a new file: start over with one layer
            self.layers = [Layer(locs)]
        self.changed("locs")

    def add_layer(self) -> Layer:
        """A new layer on the same table, with a fresh (default) filter."""
        layer = Layer(self.locs, name=f"layer {len(self.layers) + 1}")
        self.layers.append(layer)
        self.changed("layers")
        return layer

    def remove_layer(self, index: int) -> None:
        if len(self.layers) > 1:
            del self.layers[index]
            self.changed("layers")

    @property
    def can_undo(self) -> bool:
        return self._undo is not None

    def undo(self) -> None:
        if self._undo is not None:
            self.locs, self._undo = self._undo, None
            for layer in self.layers:
                layer.rebind(self.locs)
            self.log("undo")
            self.changed("locs")

    # ------------------------------------------------------------- plugins
    def selection(self, layer: int = 0) -> Selection:
        return self.layers[layer].selection(layer)

    def run(self, plugin: Plugin, settings=None, layer: int = 0,
            progress: Optional[Callable[[str], None]] = None) -> Result:
        """Run a plugin on this session's table; apply what comes back.

        The plugin runs on whatever thread calls this; only `apply` touches
        the session, so a GUI can run the plugin in a worker and apply here.
        """
        result = plugin.run(self.locs, self.selection(layer), settings, progress)
        self.apply(plugin, result)
        return result

    def apply(self, plugin: Plugin, result: Result) -> None:
        settings = result.settings
        self.log(plugin.path, result.text,
                 settings=asdict(settings) if is_dataclass(settings) else settings)
        if result.locs is not None:
            self.set_locs(result.locs)

    def log(self, what: str, text: str = "", **extra) -> None:
        self.history.append({"time": datetime.now().isoformat(timespec="seconds"),
                             "what": what, "text": text, **extra})
        self.changed("history")
