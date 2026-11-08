"""Import the heavy libraries in the background, after the windows are up.

Plugins import scipy and matplotlib inside their functions so that startup
stays quick, but the cost then lands on the first click: after a restart the
first fitter's panel took 0.8 s to open (scipy.signal, optimize, interpolate),
Simulate's 0.5 s, the first figure another 0.25 s for matplotlib.  Importing
them on a thread a few seconds in moves that to a moment nobody waits on.

A plain import is safe off the GUI thread: these modules make no widgets, and
Python's import lock makes a click that needs one mid-import wait for it,
never import it twice -- so at worst a click waits as long as it did before.
Nothing here touches Qt; matplotlib's Qt backend is left to `gui/canvas.py`.
"""
from __future__ import annotations

import importlib
import threading

# What the first panels and runs were measured to import (2026-11, on an
# M-series Mac, cold): together about 1.2 s of the GUI thread's time.
MODULES = (
    "scipy.signal",            # pulls in scipy.stats
    "scipy.interpolate",
    "scipy.optimize",
    "scipy.spatial",
    "smappy.calibrate.core",   # the fitters' and Simulate's forms read these
    "smappy.calibrate.dual",
    "smappy.calibrate.transform",
    "matplotlib.figure",       # every plugin's figure
)

DELAY_MS = 2000                # after the windows are drawn and the files opened


def _import_all(modules=MODULES) -> None:
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:      # an optional piece missing: its own click says so
            pass


def start(modules=MODULES) -> threading.Thread:
    """Import `modules` on a daemon thread, which an exit does not wait for."""
    thread = threading.Thread(target=_import_all, args=(modules,),
                              name="smappy-warmup", daemon=True)
    thread.start()
    return thread


def start_later(delay_ms: int = DELAY_MS) -> None:
    """`start` once the event loop has run for `delay_ms`."""
    from PySide6.QtCore import QTimer

    def go():
        # a singleShot callable must never raise (CLAUDE.md): PySide would
        # leave the exception pending and the next override would crash
        try:
            start()
        except Exception as error:
            print(f"warm-up not started: {error}")
    QTimer.singleShot(delay_ms, go)
