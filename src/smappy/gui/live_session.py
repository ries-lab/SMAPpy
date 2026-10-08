"""The GUI, opened on a fit that is already running, for a program that drives it.

MicroClaw runs the microscope and starts an analysis while the acquisition is
being written; the SMAPpy package it installs (`ries-lab/smappy-microclaw`)
calls this to open the ordinary GUI on that dataset.  It is the same session,
the same two windows and the same Fit plugin a user would start by hand -- the
point is that the person at the microscope sees the reconstruction build up
and can work with it in the program they already know -- with three things a
hand-started fit does not need:

* **The run is started here, with no question asked.**  Nobody may be at the
  keyboard, so the plugin's preflight is skipped; the caller checks what it
  would have asked before calling.
* **It is controlled from another thread.**  `stop` ends the fit early and
  keeps the file; `writer_finished` says the acquisition is complete, after
  which the watch reads what is left and the fit ends -- not after an idle
  timeout, because a pause in an acquisition is not its end
  (`smappy.io.watch`).  Both are `threading.Event`s, so a reader thread can set
  them without touching Qt.
* **The fit's file is frozen once it is done.**  The caller hands it on with a
  digest, and the window stays open, so the session refuses to save over it
  (`Session.protected`); what the user does afterwards goes to a new file.

`on_finished` is called exactly once, on the GUI thread, when the fit has
ended and its file is closed -- also when the windows were closed while it ran,
which is a cancellation.  `run` returns only once the windows are closed.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import plugins
from ..plugins import param_specs, settings_from, settings_values


@dataclass
class LiveOutcome:
    """How a fit started by `LiveSession` ended."""

    state: str                    # "succeeded", "cancelled" or "failed"
    complete: bool                # every frame the writer wrote was read
    path: Optional[Path] = None   # the fit's file, closed; None: nothing saved
    stats: Dict[str, Any] = field(default_factory=dict)
    error: str = ""               # one line, for "failed"
    traceback: str = ""


def fit_settings(plugin_path: str, values: Optional[Dict[str, Any]] = None):
    """The settings of the fitter at ``plugin_path``: its defaults under ``values``.

    ``values`` is a flat dotted map (``{"fit.roisize": 13}``).  Unlike
    `plugins.settings_from`, which forgives a saved name the plugin no longer
    has, this refuses one: the values come from a call somebody just made, and
    a misspelt camera offset that silently kept its default would be a wrong
    fit with nothing to say so.
    """
    cls = plugins.get(plugin_path)
    values = dict(values or {})
    known = set(_dotted_names(cls.Settings))
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{plugin_path} has no setting {', '.join(unknown)}; "
                         f"it has {', '.join(sorted(known))}")
    settings = settings_from(cls.Settings, values)
    got = settings_values(settings)
    refused = sorted(k for k, v in values.items() if got.get(k) != v)
    if refused:
        # settings_from falls back to the defaults on a value the class
        # refuses; here that is an error, not a reset
        raise ValueError(f"{plugin_path} refused {', '.join(refused)}")
    return settings


def _dotted_names(settings_cls, prefix: str = "") -> List[str]:
    out = []
    for name, spec in param_specs(settings_cls).items():
        if spec.children is not None:
            out += _dotted_names(spec.type, f"{prefix}{name}.")
        else:
            out.append(prefix + name)
    return out


class LiveSession:
    """The smappy GUI with the fitter at ``plugin_path`` running ``settings``."""

    def __init__(self, plugin_path: str, settings, *,
                 on_finished: Optional[Callable[[LiveOutcome], None]] = None,
                 on_progress: Optional[Callable[[str], None]] = None,
                 on_closed: Optional[Callable[[], None]] = None):
        self.plugin_path = plugin_path
        self.settings = settings
        self.on_finished = on_finished
        self.on_progress = on_progress
        self.on_closed = on_closed
        self.stop = threading.Event()
        self.writer_finished = threading.Event()
        self.outcome: Optional[LiveOutcome] = None
        self.session = None
        self.control = None
        self.render = None
        self.panel = None
        self._window = None

    # ------------------------------------------------------------- running
    def open(self) -> None:
        """Build the windows and start the fit; the event loop is the caller's.

        `run` is this plus the loop.  Split so a test can drive the loop
        itself.
        """
        from PySide6.QtWidgets import QApplication

        from ..session import Session
        from .app import ControlWindow, RenderWindow
        from .collector import collect_on_gui_thread
        from .widgets import apply_style

        app = QApplication.instance() or QApplication([sys.argv[0]])
        apply_style(app)
        collect_on_gui_thread(app)
        self.session = Session()
        out = self.settings.output.resolve(self.settings.source.path)
        if out is not None:
            self.session.protected.add(Path(out))
        self.render = RenderWindow(self.session)
        self.control = ControlWindow(self.session, self.render)
        screen = app.primaryScreen().availableGeometry()
        self.control.move(screen.left(), screen.top())
        self.render.move(screen.left() + self.control.width() + 20, screen.top())
        self.render.show()
        self.control.show()

        self.panel = self._panel()
        self.panel.form.set(self.settings)
        self.panel.ended.connect(self._ended)
        self.panel.progressed.connect(self._progressed)
        if not self.panel.start_run(ask=False, stop=self.stop,
                                    writer_finished=self.writer_finished):
            self._finish(LiveOutcome("failed", False,
                                     error=self.panel.status.text() or
                                     "the fit did not start"))

    def run(self) -> int:
        """Open, and run the event loop until the windows are closed."""
        from PySide6.QtWidgets import QApplication

        try:
            self.open()
        except Exception as error:
            # a failure to build the windows is the fit's failure: the caller
            # is owed an outcome whatever happens
            self._finish(LiveOutcome("failed", False, error=_one_line(error),
                                     traceback=traceback.format_exc()))
            if self.on_closed is not None:
                self.on_closed()
            return 1
        code = QApplication.instance().exec()
        self.wait()
        if self.on_closed is not None:
            self.on_closed()
        return code

    def wait(self) -> None:
        """After the windows have closed: end a fit that is still running.

        Closing the windows before the fit has finished is a cancellation --
        the user has decided not to look -- and the fit closes its file and
        reports as a stopped run does.  Its last signals are queued for the
        GUI thread, so they are delivered here by hand.
        """
        from PySide6.QtWidgets import QApplication

        if self.outcome is None:
            self.stop.set()
        app = QApplication.instance()
        while self.outcome is None:
            app.processEvents()
            time.sleep(0.02)
        thread = getattr(self.panel, "_thread", None)
        if thread is not None:
            thread.wait()

    # ------------------------------------------------------------ internals
    def _panel(self):
        """The fitter's panel in the Localize tab, or a window of its own.

        The tab is where a user would look for it; a workspace that has
        unpinned it still gets the fit, in the window the Plugins menu opens.
        """
        for index in range(self.control.tabs.count()):
            tab = self.control.tabs.widget(index)
            slots = getattr(tab, "slots", None)
            if not slots:
                continue
            for instance in tab.tab.instances:
                if instance.plugin != self.plugin_path:
                    continue
                panel = slots[instance.id].ensure()
                if panel is not None:
                    self.control.tabs.setCurrentIndex(index)
                    tab.open_section(instance.id)
                    return panel
        from .plugin_panel import PluginWindow
        self._window = PluginWindow(plugins.get(self.plugin_path), self.session,
                                    self.control)
        self._window.show()
        return self._window.panel

    def _progressed(self, text: str) -> None:
        if self.on_progress is not None and self.outcome is None:
            self.on_progress(text)

    def _ended(self, kind: str, payload) -> None:
        if self.outcome is not None:
            return                       # a later run the user started by hand
        if kind == "failed":
            lines = str(payload).strip().splitlines()
            # stopped before it had a file -- while the dataset was still
            # being waited for, say -- is still a stop, not a broken fit
            state = "cancelled" if self.stop.is_set() else "failed"
            self._finish(LiveOutcome(state, False,
                                     error=lines[-1] if lines else "failed",
                                     traceback=str(payload)))
            return
        data = payload.data or {}
        stopped = bool(data.get("stopped"))
        live = self.settings.source.live
        complete = not stopped and (self.writer_finished.is_set() or not live)
        path = data.get("path")
        self._finish(LiveOutcome("cancelled" if stopped else "succeeded", complete,
                                 path=Path(path) if path else None,
                                 stats=dict(data.get("stats") or {})))

    def _finish(self, outcome: LiveOutcome) -> None:
        self.outcome = outcome
        if self.on_finished is not None:
            self.on_finished(outcome)


def _one_line(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".splitlines()[0]
