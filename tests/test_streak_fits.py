"""A single-channel fit of a streak far from focus: both ends are peaks, one molecule is kept.

The two-colour fitters' version of this is `test_dual_fit.py`'s
``test_a_streak_far_from_focus_found_at_both_ends_is_fitted_once``; here it is
`pipeline.LocalizationEngine`, which Spline 3D and Gaussian 2D run on.
"""
import numpy as np
import pytest

from smappy.calibrate.core import spline_coefficients
from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
from smappy.io.calibration import SplineCalibration, evaluate_spline
from smappy.locs import in_pixels
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings, LocalizationEngine
from smappy.psf import GaussianPSF, SplinePSF

SHAPE = (40, 64)
TRUTH = np.array([(18.0, 14.0), (44.0, 24.0)])


def streak_spline():
    """A PSF that, far from focus, is a streak with a maximum at each end, as
    real astigmatic calibrations are: two spots 2.4 |t| px apart along x on
    one side of focus and along y on the other."""
    nz, size = 41, 17
    z, y, x = np.mgrid[:nz, :size, :size].astype(float)
    t = (z - (nz - 1) / 2) / 8
    half, along_x, c = 1.2 * np.abs(t), t > 0, size // 2
    volume = np.zeros_like(x)
    for s in (-1, 1):
        dx, dy = np.where(along_x, s * half, 0.0), np.where(along_x, 0.0, s * half)
        volume += np.exp(-.5 * (((x - c - dx) / 1.3) ** 2 + ((y - c - dy) / 1.3) ** 2))
    volume /= volume.sum(axis=(1, 2), keepdims=True)
    return SplineCalibration(spline_coefficients(volume), 10., 20., x0=8.0, psf=volume,
                             parameters={})


def frames_of(cal, z_index, n=10, photons=10000., background=20., size=15, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        frame = np.full(SHAPE, background)
        for cx, cy in TRUTH:
            x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
            frame[y0:y0 + size, x0:x0 + size] += evaluate_spline(cal, cx - x0, cy - y0, z_index,
                                                                 size) * photons
        out.append(rng.poisson(frame))
    return np.asarray(out, np.float32)


@pytest.mark.parametrize("model, z_index", [
    ("spline", 2.0), ("spline", 20.0), ("spline", 38.0),
    # an elliptical Gaussian is drawn to one lobe where they are 5 px apart (z_index 2),
    # its two fits 1.6 px apart: no longer one place, so it is tried where they are 3.6 px
    ("elliptical", 8.0), ("elliptical", 20.0), ("elliptical", 32.0)])
def test_a_streak_far_from_focus_found_at_both_ends_is_fitted_once(model, z_index):
    """Far from focus each end of the streak is a peak, both boxes hold the whole
    spot and both fits slide to the molecule: one localization is kept, and no
    molecule is lost."""
    cal = streak_spline()
    frames = frames_of(cal, z_index)
    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(20.0))
    far = z_index != 20.0
    found, _ = finder(frames[:1])
    assert len(found) == (4 if far else 2)             # far from focus: two maxima a spot

    psf = SplinePSF(cal) if model == "spline" else GaussianPSF(sigma=1.5, elliptical=True)
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1, em_on=False)
    engine = LocalizationEngine(camera, finder, psf, FitSettings(roisize=13))
    engine.push(frames, 0)
    locs = engine.flush()

    assert len(locs) == len(frames) * len(TRUTH)
    x, y = in_pixels(locs, "x_pix", "y_pix")
    for f in range(len(frames)):
        here = np.asarray(locs["frame"]) == f
        got = np.c_[x[here], y[here]]
        nearest = np.hypot(*(got[:, None] - TRUTH[None]).transpose(2, 0, 1)).min(axis=0)
        assert (nearest < 0.3).all()                  # each molecule has its one fit
    assert engine.stats["duplicates"] == (len(frames) * len(TRUTH) if far else 0)
