"""A fit by a network that sees each frame with its neighbours: the engine of the DECODE fitter.

`WindowEngine` takes blocks of raw frames as `pipeline.LocalizationEngine` does (`push`,
`flush`, driven by `pipeline.drive`), but there are no candidates and no ROIs: a *predictor*
localizes whole frames.  A predictor (`decode_smlm.predict.Predictor`, plan §5) is called with
frames in photons that include ``window`` frames of context on either side, and returns pixel
columns for the frames between.

**Blocks join without a seam.**  The last ``2 window`` frames of a block are kept and put in
front of the next, and the frames whose later neighbours have not arrived yet wait for them.  So
a movie read in blocks of 37 frames is localized from exactly the frames it is localized from in
one block of 500, and a live fit, whose blocks are whatever the camera has written, gives the
table a finished file gives.  At the ends of the movie -- the first block, and `end` -- the
first and last frames are repeated, as DECODE's ``InferenceDataset`` did; a jump in the frame
numbers is an end and a new start.

`flush` is called on a timer during a live fit and holds nothing back that it could localize:
what is left are the last ``window`` frames, which need frames not yet written.  `end` is the
source's end (`drive` calls it), and localizes them.

The positions go on the camera chip, the acquisition's ROI added, and into nm as every fitter's
do (`locs.to_nm`).  The photons are ``(counts - offset) * e/ADU / EM gain``, without the
excess-noise factor a likelihood fit divides by: the network was trained on frames simulated
with EM noise, and sees them as they are.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .camera import to_photons
from .locs import Localizations, to_nm
from .metadata import CameraMetadata


class WindowEngine:
    """Frames -> localizations through a predictor that needs ``window`` frames of context."""

    def __init__(self, camera: CameraMetadata, predictor):
        self.camera = camera
        self.predictor = predictor
        self.window = int(predictor.window)
        self._context: Optional[np.ndarray] = None   # photons of frames next - window .. last seen
        self._next: Optional[int] = None             # the first frame not yet localized
        self.stats = {"frames": 0, "localizations": 0, "network_seconds": 0.0}
        self.chip = None

    # ------------------------------------------------------------------ driving it
    def push(self, frames: np.ndarray, first_frame: int = 0) -> Optional[Localizations]:
        frames = np.asarray(frames)
        if frames.ndim == 2:
            frames = frames[None]
        if not len(frames):
            return None
        photons = to_photons(frames, self.camera)
        self.stats["frames"] += len(photons)
        done = []
        if self._next is not None and first_frame != self._next + self._waiting():
            done.append(self.end())                  # a gap: the run before it ends there
        if self._next is None:
            k = self.window
            self._context = np.repeat(photons[:1], k, axis=0)
            self._next = int(first_frame)
        buffer = np.concatenate([self._context, photons]) if len(self._context) else photons
        done.append(self._localize(buffer))
        return _joined(done)

    def flush(self) -> Optional[Localizations]:
        """Nothing is held that could be localized now: the frames waiting need later frames."""
        return None

    def end(self) -> Optional[Localizations]:
        """The source has ended: the frames waiting for later neighbours, with the last repeated."""
        if self._next is None:
            return None
        k = self.window
        out = None
        if k and self._waiting():
            buffer = np.concatenate([self._context, np.repeat(self._context[-1:], k, axis=0)])
            out = self._localize(buffer)
        self._context, self._next = None, None
        return out

    # ------------------------------------------------------------------ the work
    def _waiting(self) -> int:
        """Frames seen and not yet localized (they wait for their later neighbours)."""
        return 0 if self._context is None else len(self._context) - self.window

    def _localize(self, buffer: np.ndarray) -> Optional[Localizations]:
        """Localize the frames of ``buffer`` (photons, its first frame ``_next - window``) that have
        all their neighbours; keep the last ``2 window`` as the next block's context."""
        k = self.window
        n = len(buffer) - 2 * k
        out = None
        if n > 0:
            t0 = time.perf_counter()
            cols = self.predictor(buffer, self._next)
            self.stats["network_seconds"] += time.perf_counter() - t0
            out = self._table(cols)
            self._next += n
            self._context = buffer[len(buffer) - 2 * k:] if k else buffer[:0]
        else:
            self._context = buffer
        return out

    def _table(self, cols: dict) -> Optional[Localizations]:
        n = len(cols["frame"])
        if not n:
            return None
        ox, oy = self.camera.roi_offset
        cols = dict(cols)
        cols["x_pix"] = np.asarray(cols["x_pix"], np.float64) + ox
        cols["y_pix"] = np.asarray(cols["y_pix"], np.float64) + oy
        for name, values in cols.items():
            cols[name] = (np.asarray(values, np.int64) if name == "frame"
                          else np.asarray(values, np.float32))
        self.stats["localizations"] += n
        sizes, assumed = self.camera.pixelsize_nm_or_default
        return to_nm(Localizations(cols, {"units": "pixel"}), sizes, assumed=assumed, keep_pixels=False)

    def __str__(self) -> str:
        s = self.stats
        return (f"{s['localizations']} localizations from {s['frames']} frames "
                f"({s['network_seconds']:.1f} s in the network)")


def _joined(parts) -> Optional[Localizations]:
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    out = Localizations()
    for p in parts:
        if not out.columns:
            out.metadata.update(p.metadata)
        out.extend(p)
    return out.compact()
