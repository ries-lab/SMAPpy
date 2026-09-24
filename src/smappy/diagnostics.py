"""A log a user can send when something crashed or came out wrong.

"It crashed" or "the numbers look off" is where a bug report usually starts,
and the questions that follow are always the same: which version, which
Python and Qt, what was open, what ran with which settings, and what the
traceback said.  All of that is known to the program at the moment it
happens and forgotten a moment later, so it is written down as it goes:

* ``smappy.log`` -- the environment once per start, every plugin run with its
  settings, the size of what it ran on, how long it took, and the traceback
  if it failed; every uncaught exception, on any thread; and whatever the
  program prints, which is where most of its own error messages still go.
  Rotated at `MAX_BYTES`, `BACKUPS` old ones kept, so it never grows without
  bound and the last few sessions are always there.
* ``crash.log`` -- `faulthandler`'s stack of every thread when the process
  dies in native code (the C++ fitter, HDF5, Qt, a GPU driver), which no
  Python handler can catch because there is no Python left to run it.

`bug_report` puts both in one zip with the environment, the configuration
and a description of the session -- its history, columns and files, never the
localizations themselves -- which is the file to attach to an issue or hand
to an AI assistant along with what was expected.

Kept out of Qt: the batch runner and a script can use it too.  Nothing is
written until `start` is called, and a program that never calls it behaves
exactly as before.
"""
from __future__ import annotations

import faulthandler
import io
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional

LOG_NAME = "smappy.log"
CRASH_NAME = "crash.log"
MAX_BYTES = 2_000_000          # ~ a day of heavy use; the tail is what matters
BACKUPS = 3
# what the environment section asks after, the packages a bug usually lives near
PACKAGES = ("numpy", "scipy", "h5py", "tifffile", "PySide6", "pyqtgraph",
            "matplotlib", "numba", "torch", "wgpu", "PyYAML")

logger = logging.getLogger("smappy")
# Without a handler, `logging` prints warnings and errors to stderr through
# its "last resort" -- which in a script that never started the log would turn
# every failed plugin run into a traceback on the console.  This one swallows.
logger.addHandler(logging.NullHandler())

_state: dict = {}
_lock = threading.Lock()


def log_dir() -> Path:
    """Beside the configuration: ``<config dir>/logs``."""
    from .config import config_dir
    return config_dir() / "logs"


def log_file() -> Path:
    return log_dir() / LOG_NAME


def crash_file() -> Path:
    return log_dir() / CRASH_NAME


def started() -> bool:
    return bool(_state)


def start(program: str = "smappy", capture_output: bool = True) -> Path:
    """Begin writing the log; returns its path.  Safe to call twice.

    ``capture_output`` also copies stdout and stderr into the log, still
    printing them as before.  A GUI wants that -- its errors are prints that
    nobody sees on a desktop with no terminal -- and a test does not.
    """
    with _lock:
        if _state:
            return _state["path"]
        folder = log_dir()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / LOG_NAME
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _state.update(path=path, handler=handler, program=program)

        # native crashes: faulthandler writes straight to the file descriptor,
        # so the file has to stay open for as long as the process lives
        crash = open(folder / CRASH_NAME, "a", encoding="utf-8")
        crash.write(f"\n=== {program} started {datetime.now():%Y-%m-%d %H:%M:%S}, "
                    f"pid {os.getpid()} ===\n")
        crash.flush()
        faulthandler.enable(file=crash, all_threads=True)
        _state["crash"] = crash

        previous_hook = sys.excepthook

        def excepthook(kind, value, tb):
            logger.critical("uncaught exception\n%s",
                            "".join(traceback.format_exception(kind, value, tb)))
            previous_hook(kind, value, tb)

        sys.excepthook = excepthook
        previous_thread_hook = threading.excepthook

        def thread_hook(args):
            logger.critical("uncaught exception in thread %s\n%s",
                            getattr(args.thread, "name", "?"),
                            "".join(traceback.format_exception(
                                args.exc_type, args.exc_value, args.exc_traceback)))
            previous_thread_hook(args)

        threading.excepthook = thread_hook
        _state["hooks"] = (previous_hook, previous_thread_hook)

        if capture_output:
            _state["streams"] = (sys.stdout, sys.stderr)
            sys.stdout = _Tee(sys.stdout, logging.INFO, "out")
            sys.stderr = _Tee(sys.stderr, logging.WARNING, "err")

    logger.info("%s started\n%s", program, environment())
    return path


def stop() -> None:
    """Undo `start`: for tests, and for a host that embeds smappy."""
    with _lock:
        if not _state:
            return
        if "streams" in _state:
            for tee in (sys.stdout, sys.stderr):
                if isinstance(tee, _Tee):
                    tee.flush()
            sys.stdout, sys.stderr = _state["streams"]
        sys.excepthook, threading.excepthook = _state["hooks"]
        faulthandler.disable()
        _state["crash"].close()
        logger.removeHandler(_state["handler"])
        _state["handler"].close()
        _state.clear()


class _Tee(io.TextIOBase):
    """A stream that still prints and also logs, one record per line."""

    def __init__(self, stream, level: int, name: str):
        super().__init__()
        self.stream = stream
        self.level = level
        self.name = name
        self._buffer = ""
        self._busy = threading.local()   # a logging error prints to stderr

    def write(self, text: str) -> int:
        if self.stream is not None:
            try:
                self.stream.write(text)
            except Exception:
                pass
        if getattr(self._busy, "on", False):
            return len(text)
        self._busy.on = True
        try:
            self._buffer += text
            *lines, self._buffer = self._buffer.split("\n")
            for line in lines:
                if line.strip():
                    logger.log(self.level, "(%s) %s", self.name, line.rstrip())
        finally:
            self._busy.on = False
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            logger.log(self.level, "(%s) %s", self.name, self._buffer.rstrip())
        self._buffer = ""
        if self.stream is not None:
            try:
                self.stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return bool(self.stream is not None and self.stream.isatty())

    def fileno(self) -> int:
        return self.stream.fileno()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def _package_version(name: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment() -> str:
    """What a bug report is asked for first, as ``name: value`` lines."""
    try:
        from . import __version__ as smappy_version
    except Exception:                                     # never the reason to fail
        smappy_version = "unknown"
    lines = [f"smappy: {smappy_version}",
             f"python: {sys.version.split()[0]} ({platform.python_implementation()}) "
             f"at {sys.executable}",
             f"platform: {platform.platform()} ({platform.machine()})",
             f"cpus: {os.cpu_count()}"]
    found = [f"{name} {v}" for name in PACKAGES if (v := _package_version(name))]
    lines.append("packages: " + ", ".join(found))
    lines.append(f"log: {log_file()}")
    return "\n".join(lines)


def _describe_settings(settings) -> str:
    from .plugins import settings_values
    values = settings_values(settings)
    return ", ".join(f"{k}={v!r}" for k, v in values.items()) if values else "-"


@contextmanager
def running(plugin, job: str = "run", settings: Any = None,
            n: Optional[int] = None) -> Iterator[None]:
    """Log one plugin run: what, with which settings, on how much, how long.

    A failure is logged with its traceback and raised again, so the caller
    reports it as it always did.  Costs nothing when the log is not started.
    """
    path = getattr(plugin, "path", "") or type(plugin).__name__
    version = getattr(plugin, "version", "?")
    size = f" on {n} localizations" if n is not None else ""
    logger.info("%s %s (v%s)%s: %s", job, path, version, size, _describe_settings(settings))
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.error("%s %s failed after %.2f s\n%s", job, path,
                     time.perf_counter() - t0, traceback.format_exc())
        raise
    logger.info("%s %s done in %.2f s", job, path, time.perf_counter() - t0)


def describe_session(session) -> str:
    """What was open: the table's shape and provenance, not its contents."""
    if session is None:
        return "no session"
    lines: List[str] = []
    locs = session.locs
    lines.append(f"path: {session.path}")
    lines.append(f"localizations: {len(locs)}")
    lines.append("columns: " + ", ".join(
        f"{k} ({locs[k].dtype})" for k in locs.keys()))
    lines.append("files: " + ", ".join(getattr(f, "name", str(f)) for f in session.files))
    for i, layer in enumerate(session.layers):
        try:
            ranges = layer.filter.ranges
        except Exception:
            ranges = {}
        lines.append(f"layer {i}: visible={getattr(layer, 'visible', '?')} "
                     f"grouped={getattr(layer, 'grouped', '?')} filter={ranges}")
    lines.append(f"roi: {session.roi}")
    lines.append(f"slab: {session.slab}  plugins use it: "
                 f"{getattr(session, 'selects_slab', False)}")
    lines.append("history:")
    for entry in session.history:
        lines.append(f"  {entry}")
    return "\n".join(lines)


def bug_report(path, session=None, note: str = "") -> Path:
    """One zip with everything a bug needs but the data; returns its path.

    The logs (the current one and those rotated out), the crash stacks, the
    environment, the configuration and `describe_session`.  File names and
    folders are in it, as they are in the log; localizations are not.
    """
    path = Path(path)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    for handler in logger.handlers:
        handler.flush()
    from .config import config_file
    folder = log_dir()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.txt",
                   f"smappy bug report, {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
                   + (f"note:\n{note}\n\n" if note else "")
                   + f"environment:\n{environment()}\n\n"
                   + f"session:\n{describe_session(session)}\n")
        for name in [LOG_NAME] + [f"{LOG_NAME}.{i}" for i in range(1, BACKUPS + 1)] \
                + [CRASH_NAME]:
            if (folder / name).exists():
                z.write(folder / name, name)
        if config_file().exists():
            z.write(config_file(), "config.yaml")
    logger.info("bug report written to %s", path)
    return path
