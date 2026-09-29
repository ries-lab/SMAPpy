"""The camera frames a fit keeps beside its localizations, as SMAP does.

A localization table says where the molecules were; it cannot say what the
camera saw.  Whether a structure is really there, whether the sample drifted
out of focus, where the cell edge lies -- those are questions for the frames,
and the stack they came from is often gigabytes on another disk by the time
anyone asks.  So a fit keeps a few of them in the localization file: image 0
is the average of every frame fitted, then the first fitted frame, then the
rest evenly spaced over the fitted range, all in photons.

SMAP's CameraConverter keeps every ``diffrawframes``-th frame after the
average; here a *number* of frames is asked for instead, because a spacing
that suits a 2,000-frame stack keeps a thousand frames of a 500,000-frame one.
The first fitted frame is always among them: it is the one a user compares
with the start of the table.

The average is accumulated as the blocks go past, in ADU and in float64, and
converted once at the end: `camera.to_photons` is linear, so the average of
the converted frames and the conversion of the averaged one are the same
numbers, and one conversion is cheaper than one per frame.  The photons are
`to_photons`'s -- counts minus offset, times the conversion -- without the
fitter's division by the EM excess-noise factor, which is a device of the
fit's noise model rather than something the camera measured.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .images import ImageData
from .metadata import CameraMetadata

AVERAGE = -1          # the frame number the average is stored under


def spaced(start: int, stop: int, n: int) -> np.ndarray:
    """``n`` frame numbers evenly spaced over ``[start, stop)``, the first
    always among them; fewer when the range is shorter than ``n``."""
    if n <= 0 or stop <= start:
        return np.zeros(0, np.int64)
    if n == 1:
        return np.array([start], np.int64)
    return np.unique(np.round(np.linspace(start, stop - 1, n)).astype(np.int64))


class RawFrameKeeper:
    """Watches the blocks a fit reads and keeps what goes into the file.

    ``stop`` is the end of the fitted range when it is known.  When it is not
    -- a live acquisition, which ends when the microscope stops -- the frames
    cannot be spaced in advance, so every ``stride``-th is kept and the stride
    doubles, dropping every second frame kept so far, whenever there are more
    than ``keep``: the frames stay evenly spaced over whatever was acquired,
    from the first, and there are never more than ``keep`` of them.
    """

    def __init__(self, keep: int, start: int = 0, stop: Optional[int] = None):
        self.keep = max(int(keep), 0)
        self.start = int(start)
        self.wanted = None if stop is None else set(spaced(self.start, int(stop),
                                                           self.keep).tolist())
        self.stride = 1
        self.kept: dict = {}                 # frame number -> raw frame
        self.total: Optional[np.ndarray] = None
        self.n = 0
        self.first: Optional[int] = None
        self.last: Optional[int] = None

    def watch(self, blocks: Iterable[Tuple[int, np.ndarray]]
              ) -> Iterator[Tuple[int, np.ndarray]]:
        """The same blocks, passed on unchanged once they have been looked at."""
        for first, block in blocks:
            if self.keep:
                self.see(first, block)
            yield first, block

    def see(self, first: int, block: np.ndarray) -> None:
        block = np.asarray(block)
        if block.ndim == 2:
            block = block[None]
        if not len(block):
            return
        summed = block.sum(axis=0, dtype=np.float64)
        self.total = summed if self.total is None else self.total + summed
        self.n += len(block)
        self.first = first if self.first is None else self.first
        self.last = first + len(block) - 1
        for i in range(len(block)):
            number = first + i
            if self.wanted is not None:
                if number in self.wanted:
                    self.kept[number] = block[i].copy()
            elif (number - self.start) % self.stride == 0:
                self.kept[number] = block[i].copy()
                while len(self.kept) > self.keep:
                    self.stride *= 2
                    self.kept = {k: v for k, v in self.kept.items()
                                 if (k - self.start) % self.stride == 0}

    def image(self, camera: CameraMetadata, unit: str = "nm",
              name: str = "raw frames", source: str = "") -> Optional[ImageData]:
        """The average and the kept frames, in photons, placed under the table.

        ``unit`` is the table's: "nm" puts the image in nanometres, anything
        else in camera pixels.  A localization's ``x_pix`` is the chip column
        of the pixel whose *centre* it sits on (the camera ROI offset
        included), so pixel 0 of the frame spans ``roi_x - 0.5`` to
        ``roi_x + 0.5``.
        """
        if not self.n:
            return None
        from .camera import to_photons
        numbers = sorted(self.kept)
        planes = [self.total / self.n] + [self.kept[k] for k in numbers]
        data = to_photons(np.stack(planes), camera)
        sizes = camera.pixelsize_nm_xy if unit != "pixel" else None
        px, py = sizes if sizes is not None else (1.0, 1.0)
        rx, ry = camera.roi_offset
        return ImageData(
            data=data, pixelsize=float(px), pixelsize_y=float(py),
            x0=(rx - 0.5) * px, y0=(ry - 0.5) * py, name=name,
            path=str(source) if source else None,
            kind="raw", frames=np.array([AVERAGE] + numbers, np.int64),
            metadata={"n_averaged": int(self.n), "first": int(self.first),
                      "last": int(self.last), "units": "photons"})


def raw_images(images: Iterable[ImageData]) -> List[ImageData]:
    return [im for im in images if im.kind == "raw"]
