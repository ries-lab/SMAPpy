"""Conversion of raw camera counts into photons."""

from __future__ import annotations

import numpy as np

from .metadata import CameraMetadata

try:
    from . import _fit3d
except ImportError:  # pure numpy if the extension is not built
    _fit3d = None

# the raw types the threaded kernel reads; anything else goes through numpy
_KERNEL_TYPES = {np.dtype(t) for t in (np.uint8, np.uint16, np.int16, np.uint32, np.int32,
                                       np.float32, np.float64)}


def to_photons(frames: np.ndarray, cam: CameraMetadata, n_threads: int = 0) -> np.ndarray:
    """Convert raw camera counts to photons.

    ``frames`` may be a single image or a block of images ``(n, y, x)``.
    The result is float32.

    EM excess noise is *not* applied here.  The fitter assumes pure Poisson
    statistics; the EMCCD excess-noise factor is applied around the fit
    (see :attr:`CameraMetadata.excess_noise`) so that the fitter itself stays
    free of camera-specific assumptions.
    """
    cam.require("conversion", "offset")
    return convert(frames, cam.offset, cam.adu_to_photons, n_threads=n_threads)


def fit_photons(frames: np.ndarray, cam: CameraMetadata, noise=None,
                n_threads: int = 0) -> np.ndarray:
    """The photons the fitters read: `to_photons` over the excess noise, or,
    with a noise map (`scmos.FitNoise`), each pixel's own offset taken off."""
    if noise is not None:
        return noise.to_photons(frames, cam.adu_to_photons, n_threads=n_threads)
    cam.require("conversion", "offset")
    return convert(frames, cam.offset, cam.adu_to_photons, cam.excess_noise, n_threads)


def convert(frames: np.ndarray, offset, gain: float, excess: float = 1.0,
            n_threads: int = 0) -> np.ndarray:
    """``((frames - offset) * gain) / excess`` in float32, ``offset`` one
    value or one per pixel of the frames' trailing dimensions.

    In the extension's threaded kernel (`csrc/photons.hpp`) where it can, which
    computes exactly numpy's float32 steps -- the same photons to the last bit
    (tested) -- on every core: on one core this was a quarter of peak finding's
    time on 2048 x 2048 frames.  numpy otherwise, as it always was.
    """
    frames = np.asarray(frames)
    offset32 = np.asarray(offset, np.float32)
    kernel = getattr(_fit3d, "to_photons", None)
    if (kernel is not None and frames.dtype in _KERNEL_TYPES and frames.ndim >= 1
            and (offset32.size == 1 or offset32.shape == frames.shape[frames.ndim - offset32.ndim:])):
        return kernel(np.ascontiguousarray(frames), offset32.reshape(-1) if offset32.size == 1
                      else np.ascontiguousarray(offset32), float(np.float32(gain)),
                      float(np.float32(excess)), int(n_threads))
    out = np.asarray(frames, dtype=np.float32) - (np.float32(offset) if offset32.size == 1
                                                   and offset32.ndim == 0 else offset32)
    out *= np.float32(gain)
    if excess != 1.0:
        out = out / np.float32(excess)
    return out
