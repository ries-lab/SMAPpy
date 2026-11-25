"""A Gaussian PSF as a spline calibration, its width the third axis: what DECODE trains on without beads.

Two-dimensional data have no bead calibration as a rule, and a network needs a PSF to learn.  A
Gaussian is what Gaussian 2D fits such data with, and the width it fits says how wide the spots
are.  So the training draws Gaussians over the widths the data's spots have.

**A spline, not a renderer of its own.**  Everything a training and the DECODE fitter does goes
through a spline file: the job's ``calibration.h5`` and its sha256, netfit's `render.SplineRenderer`,
the port's ``CubicSplinePSF``, the model's check.  A stack of pixel-integrated Gaussians sampled into
a `SplineCalibration` goes through all of it unchanged, in both networks -- as SMAP did with
``make2Dastig``.

**The third axis is the width, not z.**  A molecule out of focus is wider, but a Gaussian widens
alike above and below focus: a z axis would draw a molecule at +z and -z alike, and a network
cannot learn which it was, the mixture either smearing z or splitting the molecule in two.  So the
planes are widths, monotonic: plane ``k`` has ``sigma = sigma_ref + z``, ``z`` in nm of width as
`SplineCalibration.z_index_to_nm` makes it, ``sigma_ref`` the middle of the range.  The network's
"z" is then the width's offset from ``sigma_ref``, which the fitter turns back into ``sigma_nm``
(`as_widths`), a column Gaussian 2D also writes.

**Each plane sums to one** (the pixels of an infinite plane; the drawn square cuts less than 1e-4),
not the bead average's brightest plane: so the network's photons are Gaussian 2D's photons.  The
pixels are integrated, ``erf`` differences, as Gaussian 2D's model is (`csrc` ``fit_gauss_free``).
The spline is exact on the pixels' centres; between them its cubic follows a Gaussian 1.2 pixels wide
to 1 % of the peak (0.4 % at 1.4 pixels), the format's limit, which a bead spline has as well.

The width's range comes from the data -- Gaussian 2D's fits of the prefit (`sigma_range`) -- or is
typed.
"""
from __future__ import annotations

import math

import numpy as np

#: the calibration's parameters key and kind
KEY = "gaussian"
#: planes this far apart in width, nm: the spline's cubic in z then follows the width's change exactly
STEP_NM = 5.0
#: planes beyond each end of the range, so that the range's ends are not the spline's edge cells
MARGIN = 2
#: the narrowest width a plane is drawn with, pixels: below it the pixel's integral is a box
MIN_SIGMA_PX = 0.3
#: the region's size, pixels: the drawn square (``roi - 4``) reaches 4 widths from the centre, at least 13
MIN_ROI = 13

#: the width's range from the prefit: the 2nd to 98th percentile of the fully-on fits, widened by 15 % of
#: that on either side (`decode_smlm.prefit.z_range`'s rule for z), and never narrower than
#: ``SIGMA_AT_LEAST`` of the median on either side: a model is used again on spots a little sharper or
#: wider than one movie's
SIGMA_QUANTILES, SIGMA_WIDEN, SIGMA_AT_LEAST = (0.02, 0.98), 0.15, 0.15


def _pixel_profile(offsets, sigma_px):
    """The integral over each pixel of a unit 1D Gaussian of width ``sigma_px`` (n,) at ``offsets``
    (m,) pixel centres from it: (n, m)."""
    from scipy.special import erf
    s = np.sqrt(2.0) * np.asarray(sigma_px, float)[:, None]
    o = np.asarray(offsets, float)[None]
    return 0.5 * (erf((o + 0.5) / s) - erf((o - 0.5) / s))


def gaussian_stack(sigma_x_px, sigma_y_px, roi: int) -> np.ndarray:
    """(n, roi, roi): pixel-integrated unit Gaussians centred on the middle pixel, plane ``k`` of widths
    ``sigma_x_px[k]``, ``sigma_y_px[k]``."""
    offsets = np.arange(roi) - (roi - 1) / 2
    px, py = _pixel_profile(offsets, sigma_x_px), _pixel_profile(offsets, sigma_y_px)
    return py[:, :, None] * px[:, None, :]


def _roi(sigma_px: float) -> int:
    roi = max(MIN_ROI, math.ceil(8 * sigma_px) + 5)
    return roi + (roi % 2 == 0)


def gaussian_calibration(sigma_nm, pixel_nm, *, step_nm: float = STEP_NM, margin: int = MARGIN, em_on=None,
                         ratio: float = 1.0, widest: float = 1.0):
    """The `SplineCalibration` of Gaussians from ``sigma_nm[0]`` to ``sigma_nm[1]`` wide (in the sample, nm),
    on pixels ``pixel_nm`` (one number, or x and y), planes ``step_nm`` apart with ``margin`` more beyond
    each end.  Its z is the width less the range's middle, ``sigma_ref_nm`` (`width_of`).

    ``ratio`` draws every plane ``ratio`` times as wide on the same z grid: a second channel's Gaussians,
    its z still the first channel's width less its middle (`dual_calibration`); ``widest`` sizes the region
    for Gaussians that many times as wide, so that two channels share one grid."""
    from ..io.calibration import SplineCalibration
    from .core import spline_coefficients
    lo, hi = sorted(float(v) for v in np.ravel(sigma_nm)[:2])
    if not lo > 0:
        raise ValueError(f"a Gaussian's width must be positive, not {lo:g} nm")
    px, py = (float(v) for v in np.broadcast_to(np.asarray(pixel_nm, float).ravel(), (2,)))
    ref = (lo + hi) / 2
    n_inside = max(math.ceil((hi - lo) / step_nm), 0)
    # planes from the widest to the narrowest: z = -(k - z0) dz grows with the width
    top = ref + (n_inside / 2 + margin) * step_nm
    n = n_inside + 2 * margin + 1
    sigmas = top - step_nm * np.arange(n)
    sigmas = sigmas * float(ratio)
    floor = MIN_SIGMA_PX * max(px, py)
    if sigmas[-1] < floor:
        raise ValueError(f"the range {lo:g} -- {hi:g} nm with its margin reaches below {floor:.0f} nm "
                         f"({MIN_SIGMA_PX} pixels): start it higher")
    z0 = (top - ref) / step_nm
    roi = _roi(sigmas[0] * max(float(widest), 1.0) / float(ratio) / min(px, py))
    psf = gaussian_stack(sigmas / px, sigmas / py, roi).astype(np.float32)
    parameters = {KEY: {"sigma_nm": [lo, hi], "sigma_ref_nm": ref, "step_nm": float(step_nm),
                        "pixel_nm": [px, py], **({"ratio": float(ratio)} if ratio != 1 else {})},
                  "normalization": "each plane sums to one"}
    return SplineCalibration(spline_coefficients(psf), float(step_nm), float(z0), x0=(roi - 1) / 2, psf=psf,
                             em_on=em_on, pixelsize_um=np.array([px, py]) / 1000, parameters=parameters)


def record(cal) -> dict | None:
    """The Gaussian's record (``sigma_nm``, ``sigma_ref_nm``, ...) of a calibration, or None for a PSF's."""
    params = getattr(cal, "parameters", None) or {}
    value = params.get(KEY)
    return dict(value) if isinstance(value, dict) else None


def record_of_file(path) -> dict | None:
    """`record` of the calibration file at ``path``, one channel's or two (`dual_calibration`); None for a
    file of a PSF, or none."""
    from ..io.calibration import load_spline_calibration
    try:
        from ..netfit.prefit_two import is_dual
        if is_dual(path):
            from .dual import load_dual_color_calibration
            return record(load_dual_color_calibration(path))
        return record(load_spline_calibration(path))
    except (OSError, ValueError, KeyError, NotImplementedError):   # missing, or not a spline
        return None


def dual_calibration(sigma_nm, ratio: float, pixel_nm, transform, *, em_on=None):
    """A two-channel calibration (`calibrate.dual.DualColorCalibration`) of Gaussians on one width axis: the
    main channel ``sigma_nm[0]`` to ``sigma_nm[1]`` wide, the other ``ratio`` times as wide at each z, and
    ``transform``'s registration (a `ChannelTransform`, Gaussian 2D 2C's).

    One axis because a molecule's defocus is one: it widens both channels' spots together.  The other
    channel's width as a fixed ratio of the main's is the simplest relation that holds at focus (the
    wavelengths') and widens with it.  Its record is the main channel's, with ``ratio``: the network's z
    is the main channel's width less its middle."""
    from .dual import DualColorCalibration, with_registration
    widest = max(float(ratio), 1.0)
    main = gaussian_calibration(sigma_nm, pixel_nm, em_on=em_on, widest=widest)
    second = gaussian_calibration(sigma_nm, pixel_nm, em_on=em_on, ratio=ratio, widest=widest)
    rec = dict(record(main), ratio=float(ratio))
    base = DualColorCalibration(main, second, np.asarray(transform.transformation, float), dict(transform.geometry),
                                {KEY: rec, "normalization": "each plane sums to one"})
    return with_registration(base, transform)


def width_ratio(sigma_ch0_nm, sigma_ch1_nm, mask=None) -> float:
    """The second channel's widths over the main's, the median of each molecule's ratio."""
    a, b = np.asarray(sigma_ch0_nm, float), np.asarray(sigma_ch1_nm, float)
    keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    if not keep.any():
        raise ValueError("no fitted widths in both channels to take their ratio from")
    return float(np.median(b[keep] / a[keep]))


def width_of(z_nm, rec: dict):
    """The width a Gaussian calibration's ``z_nm`` stands for, nm."""
    return np.asarray(z_nm, float) + float(rec["sigma_ref_nm"])


def z_of(sigma_nm, rec: dict):
    """The calibration's z of a width ``sigma_nm``: `width_of` turned round."""
    return np.asarray(sigma_nm, float) - float(rec["sigma_ref_nm"])


#: the predictor's columns of z and what they are for a Gaussian model
WIDTH_COLUMNS = {"z_err_nm": "sigma_err_nm", "z_err_mix_nm": "sigma_err_mix_nm", "crlb_z_nm": "crlb_sigma_nm"}


def as_widths(cols: dict, rec: dict | None) -> dict:
    """A predictor's columns with a Gaussian model's z turned into the width, ``sigma_nm`` and its errors;
    ``cols`` itself for a model trained on a PSF (``rec`` None)."""
    if not rec or "z_nm" not in cols:
        return cols
    out = {k: v for k, v in cols.items() if k != "z_nm" and k not in WIDTH_COLUMNS}
    out["sigma_nm"] = width_of(cols["z_nm"], rec)
    for old, new in WIDTH_COLUMNS.items():
        if old in cols:
            out[new] = cols[old]
    return out


def sigma_range(sigma_nm, mask=None) -> tuple:
    """The widths a training draws, nm, from Gaussian 2D's fitted widths ``sigma_nm`` (those ``mask`` keeps):
    the rule of ``SIGMA_QUANTILES``, ``SIGMA_WIDEN`` and ``SIGMA_AT_LEAST``."""
    s = np.asarray(sigma_nm, float)
    if mask is not None:
        s = s[np.asarray(mask, bool)]
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) == 0:
        raise ValueError("no fitted widths to estimate the Gaussian's range from: type it")
    lo, hi = np.quantile(s, SIGMA_QUANTILES)
    width = hi - lo
    lo, hi = lo - SIGMA_WIDEN * width, hi + SIGMA_WIDEN * width
    med = float(np.median(s))
    lo, hi = min(lo, med * (1 - SIGMA_AT_LEAST)), max(hi, med * (1 + SIGMA_AT_LEAST))
    return float(lo), float(hi)


def describe(rec: dict) -> str:
    """``Gaussians 110 -- 170 nm wide``: what a model was trained on, in words."""
    lo, hi = rec["sigma_nm"]
    return f"Gaussians {lo:.0f} -- {hi:.0f} nm wide"
