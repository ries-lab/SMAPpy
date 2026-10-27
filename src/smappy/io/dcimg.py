"""Hamamatsu DCIMG files, memory-mapped.

DCIMG is what HCImage and DCAM write; the 4Pi microscopes this package
supports record all four phase images side by side in one frame of it.  Two
versions exist and are told apart by the word at byte 8:

* ``0x7`` (old): a file header, a session header and the frames back to back,
  with the frame stamps, time stamps and the "4 pixels" in a footer after the
  data;
* ``0x1000000`` (new): each frame followed by its own 32-byte footer -- frame
  number, time stamp and the 4 pixels.

The 4 pixels: the camera stores the first four pixels of one row of every
frame -- row 0 in old files, the sensor's row 1023 in new ones, so the
middle row of a centred ROI -- apart from the image, and the image holds other values
there.  They are put back on reading (`DcimgFile.frames`), which is the
"4-pixel correction" the 4Pi data need: without it a hot spot of four pixels
sits in the middle of the strip, in one of the channels.

A file still being written (`DcimgSource.watch`): the header's frame count
is not to be trusted then -- it may be what was planned, or nothing yet -- so
a new-format file counts the frames that are there, by their footers: frame
k is written once its footer, which follows its pixels, holds the frame
number after frame k - 1's (and a time, where the camera records them).  A file the writer has grown ahead of
the frames reads zeros there, which never pass.  An old-format file keeps
its stamps in a footer written at the end and cannot be followed.

The layout follows Giacomo Mazzamuto's ``dcimg`` package (MIT), which pieced
it together from earlier readers; nothing here imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .tiff import ImageSource

OLD, NEW = 0x7, 0x1000000

_FILE_HEADER = np.dtype([("signature", "S8"), ("version", "<u4"), ("skip", "5<u4"),
                         ("n_sessions", "<u4"), ("n_frames", "<u4"),
                         ("header_size", "<u4"), ("skip2", "<u4"), ("file_size", "<u8")])
_OLD_SESSION = np.dtype([("session_size", "<u8"), ("skip1", "6<u4"), ("n_frames", "<u4"),
                         ("byte_depth", "<u4"), ("skip2", "<u4"), ("width", "<u4"),
                         ("bytes_per_row", "<u4"), ("height", "<u4"),
                         ("bytes_per_image", "<u4"), ("skip3", "2<u4"),
                         ("offset_to_data", "<u4"), ("session_data_size", "<u8")])
_NEW_SESSION = np.dtype([("session_size", "<u8"), ("skip1", "13<u4"), ("n_frames", "<u4"),
                         ("byte_depth", "<u4"), ("skip2", "<u4"), ("width", "<u4"),
                         ("height", "<u4"), ("bytes_per_row", "<u4"),
                         ("bytes_per_image", "<u4"), ("skip3", "2<u4"),
                         ("offset_to_data", "<u8")])
_OLD_FOOTER = np.dtype([("version", "<u4"), ("skip0", "<u4"), ("offset_to_2nd", "<u8"),
                        ("skip1", "2<u4"), ("skip1b", "<u8"), ("skip2", "2<u4"),
                        ("footer_size", "<u4")])
_OLD_FOOTER2 = np.dtype([("skip0", "<u8"), ("s0", "2<u4"), ("skip1", "<u8"), ("s1", "2<u4"),
                         ("skip2", "<u8"), ("s2", "2<u4"), ("offset_to_frame_counts", "<u8"),
                         ("s3", "2<u4"), ("offset_to_timestamps", "<u8"), ("s4", "4<u4"),
                         ("offset_to_4px", "<u8"), ("s5", "<u4"),
                         ("offset_of_4px_in_frame", "<u4")])
_NEW_FRAME_FOOTER = 32          # bytes after each frame of a new file
_NEW_CROP_AT = 712              # bytes into the session header: x0, w, y0, h (u2)


class DcimgFile:
    """One DCIMG file: its frames, their numbers and time stamps.

    ``frames(start, stop)`` reads frames as (n, height, width) with the 4
    pixels put back; ``len()`` is the number of frames.  Nothing is read until
    asked for, so a long acquisition costs no memory to open.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._mm = np.memmap(self.path, mode="r")
        head = np.ndarray((1,), _FILE_HEADER, self._mm)[0]
        if not bytes(head["signature"]).startswith(b"DCIMG"):
            raise ValueError(f"{self.path.name} is not a DCIMG file")
        self.version = int(head["version"])
        start = int(head["header_size"])
        if self.version == OLD:
            s = np.ndarray((1,), _OLD_SESSION, self._mm, start)[0]
        elif self.version == NEW:
            s = np.ndarray((1,), _NEW_SESSION, self._mm, start)[0]
        else:
            raise ValueError(f"{self.path.name}: DCIMG version {self.version:#x} is not known")
        self.n_frames = int(s["n_frames"])
        self.height, self.width = int(s["height"]), int(s["width"])
        depth = int(s["byte_depth"])
        if depth not in (1, 2):
            raise ValueError(f"{self.path.name}: {depth} bytes per pixel is not supported")
        self.dtype = np.dtype(np.uint8 if depth == 1 else "<u2")
        row, image = int(s["bytes_per_row"]), int(s["bytes_per_image"])
        data = start + int(s["offset_to_data"])
        self.x0 = self.y0 = 0
        self.binning = 1
        self._stamps = self._times = None

        if self.version == NEW:
            crop = np.ndarray((4,), "<u2", self._mm, start + _NEW_CROP_AT)
            if crop[1] and crop[3]:
                self.x0, self.y0 = int(crop[0]), int(crop[2])
                self.binning = max(int(crop[1]) // self.width, 1)
            else:
                # Some files leave the crop record empty (the 4Pi bead stack
                # of zenodo 14287640: 168 rows, the bad pixels in row 83).
                # DCAM centres a subarray's rows on the sensor's, so the ROI
                # starts at 1024 - height / 2.
                self.y0 = 1024 - self.height // 2
            stride = image + _NEW_FRAME_FOOTER
            self.n_frames = _written(self._mm, data, image, stride, self.n_frames)
            self._data = np.ndarray((self.n_frames, self.height, self.width), self.dtype,
                                    self._mm, data, (stride, row, depth))
            self._stamps = np.ndarray((self.n_frames,), "<u4", self._mm, data + image,
                                      (stride,))
            self._times = np.ndarray((self.n_frames, 2), "<u4", self._mm, data + image + 4,
                                     (stride, 4))
            self._four = np.ndarray((self.n_frames, 4), self.dtype, self._mm,
                                    data + image + 12, (stride, depth))
            # the sensor's row 1023, in this ROI's rows
            self._row = (1023 - self.y0) // self.binning
        else:
            self._data = np.ndarray((self.n_frames, self.height, self.width), self.dtype,
                                    self._mm, data, (image, row, depth))
            self._four, self._row = None, -1
            footer_at = start + int(s["session_data_size"])
            if footer_at + _OLD_FOOTER.itemsize <= len(self._mm):
                f1 = np.ndarray((1,), _OLD_FOOTER, self._mm, footer_at)[0]
                at2 = footer_at + int(f1["offset_to_2nd"])
                if at2 + _OLD_FOOTER2.itemsize <= len(self._mm):
                    f2 = np.ndarray((1,), _OLD_FOOTER2, self._mm, at2)[0]
                    four_at = int(f2["offset_to_4px"])
                    if int(f1["footer_size"]) == four_at + 4 * depth * self.n_frames:
                        self._four = np.ndarray((self.n_frames, 4), self.dtype, self._mm,
                                                footer_at + four_at)
                        self._row = int(f2["offset_of_4px_in_frame"]) // row
                    self._stamps = np.ndarray((self.n_frames,), "<u4", self._mm,
                                              footer_at + 272)
                    self._times = np.ndarray((self.n_frames, 2), "<u4", self._mm,
                                             footer_at + 272 + 4 * self.n_frames)

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple:
        return self.n_frames, self.height, self.width

    def frames(self, start: int = 0, stop: int = None) -> np.ndarray:
        """Frames ``start:stop`` as (n, height, width), the 4 pixels restored."""
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        out = np.array(self._data[start:stop])
        if self._four is not None and 0 <= self._row < self.height:
            out[:, self._row, :4] = self._four[start:stop, :min(4, self.width)]
        return out

    def timestamps(self) -> np.ndarray:
        """Each frame's time stamp in seconds, or None when the file has none."""
        if self._times is None:
            return None
        t = np.asarray(self._times, float)
        return t[:, 0] + t[:, 1] * 1e-6

    def close(self) -> None:
        self._mm = self._data = self._four = self._stamps = self._times = None


def _written(mm, data: int, image: int, stride: int, declared: int) -> int:
    """How many frames of a new-format file are written (the module
    docstring): those the file is long enough for, up to the first whose
    footer's frame number does not follow on or whose time is zero."""
    fits = max((len(mm) - data) // stride, 0)
    if fits == 0:
        return 0
    stamps = np.ndarray((fits,), "<u4", mm, data + image, (stride,))
    seconds = np.ndarray((fits,), "<u4", mm, data + image + 4, (stride,))
    good = stamps == stamps[0] + np.arange(fits, dtype=np.uint32)
    if seconds.any():
        # a camera that records times: a first frame not yet written has none
        good &= seconds != 0
    bad = np.flatnonzero(~good)
    n = int(bad[0]) if len(bad) else fits
    # a finished file says how many it has; a longer one's tail is not frames
    return min(n, declared) if declared and n >= declared else n


def read_dcimg(paths: Sequence) -> np.ndarray:
    """Every frame of one file, or of a list of files in the order given --
    a z stack written one file per step, as the 4Pi bead scans are."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return np.concatenate([DcimgFile(p).frames() for p in paths])


def is_dcimg(path) -> bool:
    path = Path(path)
    if path.is_dir():
        return any(path.glob("*.dcimg"))
    return path.suffix.lower() == ".dcimg"


def open_dcimg(path) -> "DcimgSource":
    """A DCIMG acquisition: one file of many frames, or a z stack written one
    file per step -- a folder, or any file of the series, stands for it all
    (`smappy.io.quadrants.numbered_series`)."""
    from .quadrants import numbered_series
    path = Path(path)
    if path.is_dir():
        path = sorted(path.glob("*.dcimg"))[0]
    files = numbered_series(path)
    counts, shape, dtype = [], None, None
    for f in files:
        d = DcimgFile(f)
        counts.append(len(d))
        if shape is None:
            shape, dtype = (d.height, d.width), d.dtype
        elif (d.height, d.width) != shape:
            raise ValueError(f"{f.name} has frames of {d.width} x {d.height}, the series "
                             f"{shape[1]} x {shape[0]}")
    return DcimgSource(files=files, shape=shape, dtype=dtype, n_frames=int(sum(counts)),
                       n_frames_declared=None, mm_metadata={}, summary={}, counts=counts)


@dataclass
class DcimgSource(ImageSource):
    """One acquisition in DCIMG files, read as frames."""

    counts: list = field(default_factory=list)

    def frames(self, chunk: int = 100, start: int = 0, stop: int = None):
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        first, at, pending = 0, start, []
        for path, count in zip(self.files, self.counts):
            last = first + count
            if last > start and first < stop:
                d = DcimgFile(path)
                a, b = max(start, first) - first, min(stop, last) - first
                for s in range(a, b, chunk):
                    pending.append(d.frames(s, min(s + chunk, b)))
                    while sum(len(p) for p in pending) >= chunk:
                        joined = np.concatenate(pending)
                        yield at, joined[:chunk]
                        at += chunk
                        pending = [joined[chunk:]] if len(joined) > chunk else []
                d.close()
            first = last
        if pending:
            yield at, np.concatenate(pending)

    def watch(self, chunk: int = 100, settings=None, start: int = 0, stop: int = None,
              stop_event=None, on_wait=None, tags=None, finished=None, idle_blocks: bool = False,
              on_behind=None):
        """Like `frames`, for an acquisition the camera is still writing:
        `smappy.io.watch.watch_stack`'s stream and options, the frames counted
        by their footers (the module docstring), the series' files listed
        again on every look.  ``tags`` is accepted and left alone: a DCIMG
        records no metadata per frame."""
        import time

        from .quadrants import numbered_series
        from .watch import WatchSettings, _sleep, _stopped
        settings = settings or WatchSettings()
        at, last_new, final = start, time.monotonic(), False
        empty = np.empty((0,) + tuple(self.shape), self.dtype)
        while not _stopped(stop_event):
            if _stopped(finished):
                final = True          # read before the frames: nothing written is left behind
            files = numbered_series(self.files[0])
            counts = []
            for f in files:
                d = DcimgFile(f)
                if d.version != NEW:
                    raise ValueError(f"{f.name} is an old-format DCIMG, which keeps its frame "
                                     "stamps at the end: it can be fitted once written, not followed")
                counts.append(len(d))
                d.close()
            there = sum(counts) if stop is None else min(sum(counts), stop)
            if on_behind is not None:
                on_behind(max(there - at, 0))
            if there > at:
                grown = DcimgSource(files=files, shape=self.shape, dtype=self.dtype,
                                    n_frames=there, n_frames_declared=None, mm_metadata={},
                                    summary={}, counts=counts)
                for first, block in grown.frames(chunk=chunk, start=at, stop=there):
                    if _stopped(stop_event):
                        return
                    yield first, block
                at, last_new = there, time.monotonic()
                if not final:
                    continue          # there may be more already: look again at once
            if final or (stop is not None and at >= stop):
                return
            waited = time.monotonic() - last_new
            if finished is None and waited >= settings.timeout:
                final = True
                continue
            if on_wait is not None:
                on_wait(waited)
            if idle_blocks:
                yield at, empty
            _sleep(settings.poll, stop_event)
