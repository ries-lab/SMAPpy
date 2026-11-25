"""Cutting fit ROIs around candidate positions: the boxes a fitter fits, not
the analysis ROIs of the ROI manager (`roi_manager`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Candidates


@dataclass
class ROIStack:
    """Square ROIs cut around candidates, ready for the fitter."""

    images: np.ndarray  # (n, roisize, roisize) float32, C-contiguous
    candidates: Candidates  # the candidate each ROI belongs to
    roisize: int

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def half(self) -> int:
        return (self.roisize - 1) // 2

    def to_image_x(self, x_in_roi: np.ndarray) -> np.ndarray:
        """Map a fitted x inside the ROI back to image coordinates."""
        return np.asarray(x_in_roi, dtype=np.float64) - self.half + self.candidates.x

    def to_image_y(self, y_in_roi: np.ndarray) -> np.ndarray:
        return np.asarray(y_in_roi, dtype=np.float64) - self.half + self.candidates.y


def cut_rois(photons: np.ndarray, candidates: Candidates, roisize: int,
             first_frame: int = 0, counts=None, n_threads: int = 0) -> ROIStack:
    """Cut ``roisize`` x ``roisize`` ROIs centred on each candidate.

    Candidates whose ROI would extend past the image border are dropped -- as
    ``simplefitter_cspline`` does.  SMAP's workflow instead shifts such ROIs
    inward and fits them anyway, which biases those localizations because the
    emitter is then no longer near the ROI centre.

    ``first_frame`` is the absolute frame number of ``photons[0]``, so that the
    absolute frame numbers carried by the candidates can be mapped onto this
    block.

    With ``counts`` (`detect.Counts`) the frames are raw counts, and only the
    boxes are converted to photons -- the same photons, to the last bit, as
    converting the frames whole.
    """
    if roisize % 2 == 0:
        raise ValueError(f"roisize must be odd, got {roisize}")

    photons = np.asarray(photons)
    if counts is None and photons.dtype != np.float32:
        photons = photons.astype(np.float32)
    if photons.ndim == 2:
        photons = photons[None]
    n_frames, height, width = photons.shape
    half = (roisize - 1) // 2

    x, y = candidates.x, candidates.y
    inside = ((x >= half) & (x < width - half)
              & (y >= half) & (y < height - half))
    kept = candidates[inside]
    if len(kept) == 0:
        return ROIStack(np.empty((0, roisize, roisize), np.float32), kept, roisize)

    frame_index = _block_index(kept.frame, first_frame, n_frames)
    images = cut_boxes(photons, frame_index, kept.y - half, kept.x - half, roisize, counts,
                       n_threads)
    return ROIStack(images, kept, roisize)


def cut_boxes(frames: np.ndarray, index: np.ndarray, top: np.ndarray, left: np.ndarray,
              size: int, counts=None, n_threads: int = 0) -> np.ndarray:
    """(n, size, size) float32 boxes of ``frames[index]`` from (``top``,
    ``left``), every box inside its frame; converted to photons by ``counts``
    (`detect.Counts`) if given.

    In the extension's threaded kernel for uint16 and float32 frames: numpy's
    fancy indexing was 2 ms a 2048 x 2048 frame (1920 boxes) on one core.  The
    conversion is `camera.convert`'s arithmetic, so the photons are the same to
    the last bit either way (tested).
    """
    from .detect import _fit3d
    kernel = getattr(_fit3d, "cut_boxes", None)
    frames = np.asarray(frames)
    if (kernel is not None and frames.ndim == 3 and frames.flags.c_contiguous
            and frames.dtype in (np.uint16, np.float32)
            and (counts is not None or frames.dtype == np.float32)
            and (counts is None or np.ndim(counts.offset) == 0)):
        index, top, left = (np.ascontiguousarray(a, t) for a, t in
                            ((index, np.int64), (top, np.int32), (left, np.int32)))
        n, h, w = frames.shape
        if index.size and (index.min() < 0 or index.max() >= n or top.min() < 0 or left.min() < 0
                           or top.max() + size > h or left.max() + size > w):
            raise IndexError("a box lies outside its frame")
        convert = counts is not None
        return kernel(frames, index, top, left, int(size), convert,
                      float(np.float32(counts.offset)) if convert else 0.0,
                      float(np.float32(counts.gain)) if convert else 1.0,
                      float(np.float32(counts.excess)) if convert else 1.0, int(n_threads))
    span = np.arange(size)
    rows = np.asarray(top)[:, None] + span  # (n, size)
    cols = np.asarray(left)[:, None] + span
    images = frames[np.asarray(index)[:, None, None], rows[:, :, None], cols[:, None, :]]
    if counts is not None:
        images = counts.photons(images)
    return np.ascontiguousarray(images, dtype=np.float32)


def _block_index(frame: np.ndarray, first_frame: int, n_frames: int) -> np.ndarray:
    """Absolute frame numbers -> position within the current block."""
    index = frame - first_frame
    if index.size and (index.min() < 0 or index.max() >= n_frames):
        raise ValueError(
            f"candidate frames {int(frame.min())}..{int(frame.max())} do not fit "
            f"the block starting at {first_frame} with {n_frames} frames")
    return index
