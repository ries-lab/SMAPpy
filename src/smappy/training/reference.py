"""The precision a training could reach at best: the CRLB of the spots it draws, for the Follow curves.

DECODE's test set reports the RMSE of the emitters it found (`decode_smlm.training`, laterally the
root mean square of the distance in x and y, axially of z).  How close that is to what the data
allow is what the curve is for, so the Train window draws the Cramer-Rao lower bound beside it:

**The spots** are drawn as the training's ``Simulation`` section draws them -- photons normal with
its mean and spread, the background uniform in its range, z uniform in its extent, x and y uniform
within a pixel -- each alone in a ROI of 13 x 13 pixels, from the job's own calibration
(`simulate.camera._draw_spline`), without noise.  **The bound** is SMAPpy's spline fitter's
(`psf.SplinePSF`): fitted on the noise-free spot it stops at the truth, and its CRLB there is the
Poisson bound.  With EM gain the bound is doubled in variance: the excess noise factor of an EM
register at high gain is 2, which the training's camera model draws and the Poisson bound leaves out.
**The reference** is the median spot's bound: laterally ``sqrt(CRLB_x + CRLB_y)`` in nm (the pixel's
two sizes), axially ``sqrt(CRLB_z)``.

**Per molecule** the training divides each test-set localization's error by its own molecule's
bound every epoch -- the bound of the training's own PSF, `decode_smlm.training.test_crlbs` -- and
reports the width of that (`robust_width`), and where in their pixel its localizations fall
(`subpixel_histogram`, `ks_uniform`): the Follow curves.

It is a guide, not a bound on the test set's number: the test frames hold several spots that
overlap and molecules on for part of a frame, and the RMSE is over the spots found -- the dim ones
and those far from focus, whose bound is largest, are found less.  The root mean square of the
bound, which they dominate, was above the RMSE a model reached on the SMLM Challenge's dim set (73
against 46 nm laterally, at a recall of 0.33); the median spot's is 39.  The photons are drawn no
lower than a tenth of their mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROI = 13
SPOTS = 2000
EXCESS_NOISE = 2.0                      # an EM register's at high gain, in variance


def crlb_reference(params: dict, calibration, spots: int = SPOTS, seed: int = 0) -> dict:
    """``{"lateral_nm", "axial_nm"}``: the median spot's CRLB, the spots drawn as ``params``
    (a job's ``method.yaml``) draws them, with ``calibration`` (a `SplineCalibration` or its path)."""
    from ..io.calibration import load_spline_calibration
    from ..psf import SplinePSF
    from ..simulate.camera import _draw_spline
    cal = calibration if hasattr(calibration, "coeff") else load_spline_calibration(calibration)
    sim, cam = params["Simulation"], params["Camera"]
    rng = np.random.default_rng(seed)
    mu, sd = (float(v) for v in sim["intensity_mu_sig"])
    photons = np.clip(rng.normal(mu, sd, spots), mu / 10, None)
    lo, hi = (float(v) for v in sim["bg_uniform"])
    background = rng.uniform(lo, hi, spots)
    z_lo, z_hi = (float(v) for v in sim["emitter_extent"][2])
    reach = min(cal.z0, cal.coeff.shape[1] - 1 - cal.z0) * cal.dz * 0.95     # inside the spline
    z = rng.uniform(max(z_lo, -reach), min(z_hi, reach), spots)
    xy = ROI // 2 + rng.uniform(-0.5, 0.5, (spots, 2))
    image = np.repeat(background[:, None, None], ROI, axis=1).repeat(ROI, axis=2)
    _draw_spline(image, np.arange(spots), xy, z, photons, cal)
    psf = SplinePSF(cal)
    found = psf.unpack(psf.fit(np.ascontiguousarray(image, np.float32)))
    py, px = (float(v) for v in cam["px_size"])           # DECODE's order: rows, columns
    lateral = (found["x_err_pix"] * px) ** 2 + (found["y_err_pix"] * py) ** 2
    axial = found["z_err_nm"] ** 2
    excess = EXCESS_NOISE if cam.get("em_gain") else 1.0
    def median(v):
        v = v[np.isfinite(v)]
        return float(np.sqrt(excess * np.median(v))) if len(v) else float("nan")
    return {"lateral_nm": median(lateral), "axial_nm": median(axial)}


def for_job(folder) -> dict | None:
    """`crlb_reference` of a job folder's ``method.yaml`` and calibration; None where it has neither."""
    import yaml

    from .job import CALIBRATION
    folder = Path(folder)
    try:
        params = yaml.safe_load((folder / "method.yaml").read_text(encoding="utf-8"))
        return crlb_reference(params, folder / CALIBRATION)
    except (OSError, KeyError, TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- the test set, per molecule
#: the sub-pixel histogram's bins: 20 across a pixel
PIXEL_BINS = 20


def robust_width(values) -> float | None:
    """The standard deviation a Gaussian would have, from the median absolute deviation (1.4826
    MAD): what error / CRLB scatters by without its tails -- a mismatch or two molecules merged --
    with no cut-off to choose, and from the first epochs on, where most errors are large (it holds
    while fewer than half are outliers)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def subpixel_histogram(positions_px, bins: int = PIXEL_BINS) -> list:
    """Where in their pixel positions fall, ``bins`` counts from the pixel's centre (0) through its
    edge (0.5) back to the centre: uniform for unbiased positions, a peak at 0 and 1 for positions
    pulled to the pixel centres (pixelation)."""
    v = np.asarray(positions_px, float)
    v = v[np.isfinite(v)] % 1.0
    return np.histogram(v, bins, (0.0, 1.0))[0].astype(int).tolist()


def ks_uniform(positions_px) -> float | None:
    """The Kolmogorov--Smirnov distance of the positions' place within their pixel from uniform:
    0 for none, a pixelation measure in one number."""
    u = np.sort(np.asarray(positions_px, float) % 1.0)
    u = u[np.isfinite(u)]
    n = len(u)
    if n < 10:
        return None
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))
