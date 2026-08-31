"""Frames handed in from somewhere else: a camera API, an acquisition hook.

The file-backed sources answer "what has been written?" by looking at a file.
Some acquisitions never write one -- pycro-manager hands each image to a
callback, and a control program may have the frames in memory already.  This is
the same `ImageSource` interface for that case: the producer calls `push`, and
the pipeline reads blocks from `watch` exactly as it reads them from a growing
TIFF, so `LiveFit` and `live_view` work unchanged.

The producer and the consumer are different threads and neither waits for the
other: `push` hands a frame over and returns, and the fitting side takes
whatever has accumulated.  A `maxsize` bounds that queue for a producer that can
outrun the fit -- pushing then blocks, which is the only honest answer when the
alternative is growing until memory runs out.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np

from .tiff import ImageSource

_CLOSED = object()


@dataclass
class QueueSource(ImageSource):
    """An acquisition that is pushed in rather than read from a file.

    ``shape`` and ``dtype`` describe one frame and are given up front, because
    the pipeline is set up before the first image exists -- the camera knows
    them, and a live view needs the field of view from the start.  They may be
    left out, in which case the first pushed frame settles them.
    """

    maxsize: int = 0            # 0: unbounded; otherwise push blocks when full
    _queue: "queue.Queue" = field(default=None, repr=False)
    _pushed: int = field(default=0, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._queue = queue.Queue(maxsize=self.maxsize)

    # ---------------------------------------------------------------- pushing
    def push(self, frames: np.ndarray, first_frame: Optional[int] = None,
             timeout: Optional[float] = None) -> int:
        """Hand over one frame ``(y, x)`` or a block ``(n, y, x)``.

        Frames are numbered in the order they are pushed; ``first_frame`` sets
        the number of this one and the count continues from it, for a producer
        that knows the acquisition's own frame indices.  Returns the index the
        next push will get.
        """
        if self._closed:
            raise ValueError("push after close")
        block = np.asarray(frames)
        if block.ndim == 2:
            block = block[None]
        if block.ndim != 3:
            raise ValueError(f"expected (y, x) or (n, y, x), got {block.shape}")

        if self.shape is None:
            self.shape = tuple(block.shape[1:])
            self.dtype = block.dtype
        if first_frame is not None:
            self._pushed = int(first_frame)

        self._queue.put((self._pushed, block), timeout=timeout)
        self._pushed += len(block)
        self.n_frames = self._pushed
        return self._pushed

    def close(self) -> None:
        """No more frames.  A reader drains what is left and then stops."""
        if not self._closed:
            self._closed = True
            self._queue.put(_CLOSED)

    @property
    def closed(self) -> bool:
        return self._closed

    # ---------------------------------------------------------------- reading
    def watch(self, chunk: int = 100, settings=None, start: int = 0,
              stop: Optional[int] = None, stop_event=None, on_wait=None
              ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(first_frame, block)`` as frames are pushed.

        Blocks are up to ``chunk`` frames, and whatever has accumulated is handed
        over as soon as the queue runs dry -- waiting for a full block would hold
        the last seconds of a sparse acquisition back indefinitely.

        It ends when `close` has been called and everything pushed has been
        yielded, when ``stop_event`` is set, or when nothing has arrived for
        ``settings.timeout`` seconds, which is the same rule the file-backed
        sources use for an acquisition that simply stopped.
        """
        from .watch import WatchSettings

        settings = settings or WatchSettings()
        pending: list = []
        pending_start = 0        # the frame number the pending block begins at
        next_number = None       # what the pending block expects next
        last_new = time.monotonic()
        done = False

        def ready():
            """The pending frames as one block, and forget them."""
            nonlocal pending
            if not pending:
                return None
            block = np.concatenate(pending) if len(pending) > 1 else pending[0]
            pending = []
            return pending_start, block

        while not _stopped(stop_event) and not done:
            try:
                item = self._queue.get(timeout=settings.poll)
            except queue.Empty:
                block = ready()
                if block is not None:      # hand over what is there
                    yield block
                    last_new = time.monotonic()
                    continue
                if self._closed or time.monotonic() - last_new >= settings.timeout:
                    break
                if on_wait is not None:
                    on_wait(time.monotonic() - last_new)
                continue

            if item is _CLOSED:
                break

            first, frames = item
            last_new = time.monotonic()
            for offset, frame in enumerate(frames):
                number = first + offset
                if number < start:
                    continue
                if stop is not None and number >= stop:
                    done = True
                    break
                # a gap in the numbering ends the block: what is handed on has
                # to be the frames it says it is
                if pending and number != next_number:
                    yield ready()
                if not pending:
                    pending_start = number
                pending.append(frame[None])
                next_number = number + 1
                if len(pending) == chunk:
                    yield ready()

        block = ready()
        if block is not None:
            yield block

    # frames() is watch(): a queue has no past to re-read
    frames = watch

    def __str__(self) -> str:
        shape = "x".join(str(n) for n in self.shape) if self.shape else "?"
        return f"pushed frames ({shape} {self.dtype}, {self.n_frames} so far)"

    def frame(self, index: int) -> np.ndarray:
        raise NotImplementedError(
            "a QueueSource holds no frames: they are handed on as they arrive. "
            "Keep what you need on the producing side.")


def _stopped(stop_event) -> bool:
    return stop_event is not None and stop_event.is_set()


def queue_source(shape=None, dtype=np.uint16, maxsize: int = 0) -> QueueSource:
    """A `QueueSource` for frames of ``shape``, ready to be pushed into."""
    return QueueSource(files=[], shape=shape, dtype=np.dtype(dtype), n_frames=0,
                       n_frames_declared=None, mm_metadata={}, summary={},
                       maxsize=maxsize)
