"""Who takes the keyboard: a log of the windows that get the focus, and of
every raise, activate and show the GUI's Python asks for, with where it
asked from.

For a bug that only a real window system shows -- on a Mac, the main window
coming to the front while one steps through the ROIs from their figures --
and that the tests' offscreen platform cannot: it moves no focus at all.
Switched on by the environment, ``SMAPPY_DEBUG_FOCUS=1`` (a log in the
config directory, ``focus.log``) or ``SMAPPY_DEBUG_FOCUS=/a/file``; off, it
costs nothing.  A focus change with no raise before it was asked for by Qt
or the window system itself, not by SMAPpy's code -- which is worth knowing
too.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Optional

ENV = "SMAPPY_DEBUG_FOCUS"
#: the calls watched, on every widget
WATCHED = ("raise_", "activateWindow", "show", "setFocus")
_installed: Optional[Path] = None           # once per process: make_app runs again


def log_path() -> Optional[Path]:
    """Where the log goes, or None when it is off."""
    value = os.environ.get(ENV, "").strip()
    if not value or value == "0":
        return None
    if value in ("1", "yes", "on"):
        from ..config import config_dir
        return Path(config_dir()) / "focus.log"
    return Path(value).expanduser()


def _where() -> str:
    """SMAPpy's part of the stack, innermost first."""
    frames = [f for f in traceback.extract_stack()[:-2] if "smappy" in f.filename
              and not f.filename.endswith("focus_log.py")]
    return " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames[::-1][:6])


def _name(widget) -> str:
    try:
        title = widget.window().windowTitle()
    except RuntimeError:                # deleted on the C++ side
        return "?"
    return f"{type(widget).__name__} [{title[:40]}]"


def install(app, path: Optional[Path] = None) -> Optional[Path]:
    """Start logging, when the environment asks for it; the log's path, or None."""
    global _installed
    path = path or log_path()
    if path is None or _installed is not None:
        return _installed if path is not None else None
    _installed = path
    from PySide6.QtWidgets import QWidget
    path.parent.mkdir(parents=True, exist_ok=True)
    out = open(path, "a", buffering=1, encoding="utf-8")
    out.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} SMAPpy focus log\n")

    def note(what: str) -> None:
        try:
            out.write(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} "
                      f"{what}\n")
        except Exception:               # noqa: BLE001 -- a log never breaks the GUI
            pass

    def watch(name: str) -> None:
        original = getattr(QWidget, name)

        def call(self, *args, **kwargs):
            if name != "setFocus" or self.isWindow():
                note(f"{name:<15} {_name(self)}  <- {_where()}")
            return original(self, *args, **kwargs)
        setattr(QWidget, name, call)

    for name in WATCHED:
        watch(name)

    def focused(window) -> None:
        title = window.title() if window is not None else "(none: another application)"
        note(f"{'FOCUS':<15} {title[:60]}  <- {_where() or 'Qt / the window system'}")
    app.focusWindowChanged.connect(focused)
    return path
