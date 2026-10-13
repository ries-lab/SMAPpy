"""Fitting an acquisition while it runs: `LiveFit`, the headless handle.

The fitter is `LocalizationEngine` driven by a file that is still being
written (`smappy.io.watch`); whoever shows the result -- the GUI's live fit,
a script -- drains what it has queued.  The join is a **queue**.

* The fitter runs in its own thread and puts each finished block on a queue.
  It never touches the view.  Nothing needs a lock, because a block of
  localizations is handed over and never looked at again by the thread that
  made it -- and because the C++ that dominates the fit releases the GIL, that
  thread does not compete with the window for it either.
* Whoever shows it drains the queue when it likes (`LiveFit.drain`).

An HDF5 file is written at the same time and is the actual result.  The GUI
shows a live fit through the Fit plugin (`gui/live_session.py`); the
matplotlib viewer this module once drove went in 2026-10.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .io.watch import WatchSettings, open_growing_stack
from .locs import Localizations
from .metadata import CameraMetadata
from .pipeline import FitSettings, LocalizationEngine, prefetch, provenance


@dataclass
class LiveSettings:
    """The timing of an online run: how often things are handed on."""

    chunk: int = 100             # frames read from the file at a time
    update_seconds: float = 3.0  # how often the view takes in what is queued
    flush_seconds: float = 5.0   # fit the buffered ROIs at least this often
    read_ahead: int = 2          # blocks the reader may run ahead of the fit
    watch: WatchSettings = field(default_factory=WatchSettings)


class LiveFit:
    """The pipeline, running in a thread, handing finished blocks over a queue.

    The loop is `fit_stack`'s with one addition: the engine is flushed on a
    timer as well as when its ROI buffer fills.  Without that, a sparse
    acquisition would show nothing for minutes -- the buffer holds 15000 ROIs
    and a quiet sample does not produce them quickly.  The timer is checked
    once per block, so ``frames`` should send empty blocks while it waits
    (``watch(..., idle_blocks=True)``); otherwise what
    was buffered before a pause is fitted only once the pause is over.
    """

    def __init__(self, frames: Iterable[Tuple[int, np.ndarray]],
                 camera: CameraMetadata, finder, model,
                 settings: Optional[FitSettings] = None,
                 live: Optional[LiveSettings] = None,
                 sink=None, stop_event: Optional[threading.Event] = None):
        self.engine = LocalizationEngine(camera, finder, model,
                                         settings or FitSettings())
        self.frames = frames
        self.live = live or LiveSettings()
        self.sink = sink
        self.stop_event = stop_event or threading.Event()
        self.results: "queue.Queue[Localizations]" = queue.Queue()
        self.finished = threading.Event()
        self.error: Optional[BaseException] = None
        self.n_emitted = 0
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.monotonic()

    # ----------------------------------------------------------------- running
    def start(self) -> "LiveFit":
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="smappy-live",
                                        daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            last_flush = time.monotonic()
            for first_frame, block in prefetch(self.frames,
                                               self.live.read_ahead):
                if self.stop_event.is_set():
                    break
                # an empty block is the watch saying nothing has arrived:
                # only the clock below has work
                if len(block):
                    self._emit(self.engine.push(block, first_frame))
                if time.monotonic() - last_flush >= self.live.flush_seconds:
                    self._emit(self.engine.flush())
                    last_flush = time.monotonic()
            self._emit(self.engine.flush())
        except BaseException as error:       # reported, not swallowed
            self.error = error
            traceback.print_exc(file=sys.stderr)
        finally:
            self.finished.set()

    def _emit(self, locs: Optional[Localizations]) -> None:
        if locs is None or len(locs) == 0:
            return
        if self.sink is not None:
            self.sink(locs)                  # the file first: it is the result
        self.n_emitted += len(locs)
        self.results.put(locs)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the fit to end and wait for it, at most ``timeout``."""
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)

    # ----------------------------------------------------------------- reading
    def drain(self) -> List[Localizations]:
        """Every block queued since the last call."""
        blocks = []
        while True:
            try:
                blocks.append(self.results.get_nowait())
            except queue.Empty:
                return blocks

    @property
    def running(self) -> bool:
        return not self.finished.is_set()

    @property
    def status(self) -> str:
        """One line for the title: how far the fit has got."""
        stats = self.engine.stats
        elapsed = max(time.monotonic() - self._t0, 1e-9)
        state = "fitting" if self.running else "acquisition ended"
        if self.error is not None:
            state = f"stopped: {type(self.error).__name__}"
        return (f"{state} · {stats['frames']:,} frames "
                f"({stats['frames'] / elapsed:.0f}/s)")


def camera_extent(camera: CameraMetadata, shape
                  ) -> Optional[Tuple[float, float, float, float]]:
    """The rectangle the camera can see, in nm, as the table's positions are.

    Fixing the view to this instead of to the data means the image is framed
    the same way from the first localization to the last.
    """
    height, width = int(shape[0]), int(shape[1])
    x0, y0 = camera.roi_offset
    if camera.roi is not None and len(camera.roi) >= 4:
        width, height = int(camera.roi[2]), int(camera.roi[3])
    (sx, sy), _ = camera.pixelsize_nm_or_default
    return (x0 * sx, y0 * sy, (x0 + width) * sx, (y0 + height) * sy)
