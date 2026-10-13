"""Garbage is collected on the GUI thread, on a timer, and nowhere else.

Python's cyclic collector runs in whichever thread happens to allocate when
its counters overflow.  In this program that is often a worker: a plugin run,
a fit's reader, a render.  If what it collects is a Qt widget that was left in
a reference cycle -- a closed result window, a panel rebuilt and dropped --
the widget's destructor runs on that worker.  Destroying a widget off the GUI
thread is undefined in Qt, and in practice `QWindow::close` waits there for
the GUI thread to flush window-system events: a hang, or a crash.  The test
suite deadlocked exactly so (a widget from a GUI test, freed under `tifffile`
in a live fit's reader thread; see tests/conftest.py).

So automatic collection is switched off and a timer on the GUI thread does it
instead, a generation at a time as the counters say it is due -- pyqtgraph's
`GarbageCollector`, for the same reason.  Reference counting is untouched:
only cycles wait, at most `INTERVAL_MS`.

Before each collection the screens are held (`keep_screens`): collecting
widgets can otherwise delete Qt's own `QScreen`.

It does not matter what a worker holds.  Whichever thread allocates past the
threshold collects every cycle in the process, a window it never saw
included.  PySide hands a plain `QWidget` or `QWindow` collected off the main
thread back to it (`Py_AddPendingCall`, 6.6 to 6.12), but not a Python
subclass of one -- and every widget here is a subclass: a `PluginPanel` left
in a cycle, then a worker that only allocated dicts, segfaulted while the
GUI thread ran its events and hung while it waited for the worker.  So the
render threads drawing a snapshot (`layer_state.LayerSnapshot`) did not make
it safe to switch automatic collection back on.  Tried in 2026-10 (PySide
6.12.0, Python 3.13): `--slow -n auto` lost a worker to a segfault in each
of 9 runs, in `test_camera_frames` (the fit's TIFF reader collecting while
the test waited on it) or `test_stop_button` (the linking pool); the
GUI-heavy files alone, 5 of 20 runs.
"""
from __future__ import annotations

import gc

INTERVAL_MS = 1000

_timer = None
_screens: list = []


def collect_on_gui_thread(app, interval_ms: int = INTERVAL_MS):
    """Install the collector on ``app``'s thread; a second call is a no-op.

    Returns the timer, which the caller need not keep: it is parented to the
    application.
    """
    global _timer
    if _timer is not None:
        return _timer
    from PySide6.QtCore import QTimer
    gc.disable()
    _timer = QTimer(app)
    _timer.timeout.connect(lambda: _collect_due(app))
    _timer.start(interval_ms)
    return _timer


def _collect_due(app=None) -> None:
    """The generation the automatic collector would have collected by now."""
    counts, thresholds = gc.get_count(), gc.get_threshold()
    generation = -1
    for g in range(3):
        if counts[g] <= thresholds[g]:
            break
        generation = g
    if generation >= 0:
        if app is not None:
            keep_screens(app)
        gc.collect(generation)


def keep_screens(app) -> None:
    """Hold a Python reference to every `QScreen`; call it before collecting.

    `window.windowHandle().screen()` -- which matplotlib's canvas calls when
    it is shown -- hands back the screen's wrapper and makes it a shiboken
    child of the window's wrapper, then of the next window's.  When the
    cyclic collector frees two such widgets together, PySide deletes the C++
    `QScreen` itself (its `destroyed` fires inside `gc.collect`), Qt's screen
    list keeps the dangling pointer, and the next window to hide -- one
    destroyed in the same collection, say -- segfaults in
    `QCursor::pos(primaryScreen())`.  Before 6.9 matplotlib's DPI connections
    held a reference to the wrapper and so hid this; without them 6.8.3
    crashes too.  A wrapper that something still references is never in the
    collector's garbage, so it is never deleted.

    Renewed before every collection rather than taken once: when one of
    those windows is destroyed by Qt, shiboken invalidates the screen's
    wrapper with it, though the screen lives on, and `screen()` then hands
    out a new wrapper that nothing holds.  A full single-process run lost
    its screen so after 600 tests.
    """
    _screens[:] = app.screens()
