"""The Cramér-Rao lower bound of a spline PSF against z: how precisely a fit could localize with it.

SMAP's 3D calibration draws this beside the PSF, and it is the one plot that
says what the calibration is worth before any data are fitted: a 3D PSF
whose z precision is poor somewhere in its range is poor there for every
molecule.

The bound is the Fisher information of the fit's own model, an emitter on a
uniform background with Poisson noise, ``mu_k = N psf_k(x, y, z) + b``, in
the fitter's box (`pipeline` default, 13 px) and at the box's centre:

    I_ij = sum_k (d mu_k / d theta_i)(d mu_k / d theta_j) / mu_k,
    CRLB = diag(I^-1),  theta = (x, y, N, z, b).

N and b are fitted with the position, as they are in a fit, so the bound
includes what not knowing them costs.  The derivatives of the spline are
central differences of `io.calibration.evaluate_spline`, the model the fitter
evaluates: the spline is twice continuously differentiable, so a step of 0.01
(pixels or z planes) is exact to about 1e-4, far below what a plot shows; and
the bound of the C++ fitter's Fisher matrix at the true position agrees with
it to within a percent (`tests/test_calibration_crlb.py`).  An EMCCD's excess
noise, which doubles the variance, is not in it: the bound is for photons.

A dual calibration has two splines, normalised together so that their sum is
one emitter's photons (`psf.signal_integral`); the joint bound is the sum of
both channels' Fisher information with x, y, z and N shared and a background
each, which is what Spline 3D 2C fits, on the splines as the channel layer
sees them (`DualColorCalibration.layer_splines`, the secondary flipped back
along the mirror).
"""
from __future__ import annotations

import numpy as np

#: the photons and background per pixel the page draws the bound for: a typical bright blink
PHOTONS = 2000.0
BACKGROUND = 20.0
#: the pixel size taken for x and y when neither the beads' file nor the camera database gives one
ASSUMED_PIXEL_NM = 100.0
#: the fitter's box, `pipeline.FitSettings.roisize`'s default
ROISIZE = 13
#: the step of the central differences, in pixels and z planes
STEP = 0.01


def _derivatives(cal, z_index: float, roisize: int):
    """The normalised model at the box's centre and its derivatives along x, y, z (index units),
    each (roisize, roisize), in float64."""
    from ..io.calibration import evaluate_spline

    def psf(x, y, z):
        return np.asarray(evaluate_spline(cal, x, y, z, roisize), np.float64)

    c, h = (roisize - 1) / 2, STEP
    return (psf(c, c, z_index),
            (psf(c + h, c, z_index) - psf(c - h, c, z_index)) / (2 * h),
            (psf(c, c + h, z_index) - psf(c, c - h, z_index)) / (2 * h),
            (psf(c, c, z_index + h) - psf(c, c, z_index - h)) / (2 * h))


def _channel_fisher(cal, z_index, photons, background, roisize, n_backgrounds=1, which=0):
    """One channel's Fisher matrix over (x, y, N, z, b_0 .. b_{n-1}), its background at ``which``."""
    p, dx, dy, dz = _derivatives(cal, z_index, roisize)
    mu = np.maximum(photons * p + background, 1e-12)
    grads = [photons * dx, photons * dy, p, photons * dz]
    for k in range(n_backgrounds):
        grads.append(np.ones_like(p) if k == which else np.zeros_like(p))
    g = np.stack([d.ravel() for d in grads])
    return (g / mu.ravel()) @ g.T


def _bounds(fisher, dz_nm):
    """Standard deviations (x px, y px, photons, z nm, background) from a Fisher matrix; NaN where it
    is singular (a flat spline, outside the stack)."""
    try:
        var = np.diag(np.linalg.inv(fisher))
    except np.linalg.LinAlgError:
        return np.full(5, np.nan)
    sd = np.sqrt(np.where(var > 0, var, np.nan))
    return np.array([sd[0], sd[1], sd[2], sd[3] * dz_nm, sd[4]])


def z_range_nm(cal) -> np.ndarray:
    """The z positions the bound is drawn at: every plane of the spline but the outermost, whose
    cells extrapolate, in nm (ascending)."""
    nz = cal.coeff.shape[1]
    return np.sort(cal.z_index_to_nm(np.arange(1, nz - 1, dtype=float)))


def spline_crlb(cal, z_nm, photons: float = PHOTONS, background: float = BACKGROUND,
                roisize: int = ROISIZE) -> dict:
    """The CRLB of one spline at each ``z_nm``, as standard deviations: ``x_px``, ``y_px``, ``z_nm``,
    ``photons``, ``background``.  ``photons`` are the fitted photons, so a dual calibration's channel
    is drawn for that many photons in it (its share, `signal_integral`, divided out)."""
    from ..psf import signal_integral
    z_nm = np.atleast_1d(np.asarray(z_nm, float))
    n = photons / signal_integral(cal)
    rows = [_bounds(_channel_fisher(cal, float(cal.z_nm_to_index(z)), n, background, roisize), cal.dz)
            for z in z_nm]
    return _named(np.array(rows).reshape(len(z_nm), 5))


def dual_crlb(main, secondary, z_nm, photons: float = PHOTONS, background: float = BACKGROUND,
              roisize: int = ROISIZE) -> dict:
    """The joint CRLB of a dual calibration's two splines, as a Spline 3D 2C fit has it: ``photons``
    the emitter's in both channels together, ``background`` per pixel in each; ``background`` is
    the main channel's."""
    z_nm = np.atleast_1d(np.asarray(z_nm, float))
    rows = []
    for z in z_nm:
        fisher = sum(_channel_fisher(cal, float(cal.z_nm_to_index(z)), photons, background, roisize,
                                     n_backgrounds=2, which=c)
                     for c, cal in enumerate((main, secondary)))
        rows.append(_bounds(fisher, main.dz)[:5])
    return _named(np.array(rows).reshape(len(z_nm), 5))


def _named(rows):
    return dict(zip(("x_px", "y_px", "photons", "z_nm", "background"), rows.T))


def spline_pixel_nm(cal) -> float | None:
    """The pixel size a spline (or a dual calibration's main channel) records, nm, or None: the pupil
    model's, in situ's and the molecules' splines carry their camera's, the bead average's may not."""
    main = getattr(cal, "main", cal)
    size = getattr(main, "pixelsize_um", None)
    if size is None:
        return None
    size = np.atleast_1d(np.asarray(size, float))
    return float(np.mean(size)) * 1000 if size.size and np.all(np.isfinite(size)) and np.all(size > 0) else None


def pixel_size_nm(result) -> float | None:
    """The beads' pixel size in the sample, from the first stack's camera (its file, then the camera
    database, which knows the microscope where the file names it), or None: the calibration itself
    is in pixels."""
    try:
        from ..io.tiff import camera_metadata, open_stack
        path = result.beads.sources[0]["path"]
        size = camera_metadata(open_stack(path), require=False).pixelsize_um
    except Exception:  # noqa: BLE001 -- a stack given as arrays, a file moved since: the size assumed
        return None
    if size is None:
        return None
    size = np.atleast_1d(np.asarray(size, float))
    return float(np.mean(size)) * 1000 if np.all(np.isfinite(size)) and np.all(size > 0) else None
