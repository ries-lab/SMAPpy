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
"""
from __future__ import annotations

import gc

INTERVAL_MS = 1000

_timer = None


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
    _timer.timeout.connect(_collect_due)
    _timer.start(interval_ms)
    return _timer


def _collect_due() -> None:
    """The generation the automatic collector would have collected by now."""
    counts, thresholds = gc.get_count(), gc.get_threshold()
    generation = -1
    for g in range(3):
        if counts[g] <= thresholds[g]:
            break
        generation = g
    if generation >= 0:
        gc.collect(generation)
