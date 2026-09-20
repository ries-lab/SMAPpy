"""PSF models: what to fit, and how to read the fitted parameters.

A model knows how to call the fitter and how to turn the raw parameter vector
into named quantities in physical units.  Everything else in the pipeline is
model-agnostic, so adding a model means adding a class here (and a kernel in
``csrc/models.hpp``), not touching the pipeline.

Fitted parameters are always ordered ``(x, y, photons, background, ...)`` with
x the ROI column and y the row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import _fit3d
from .io.calibration import SplineCalibration


@dataclass
class FitResult:
    """Raw fitter output for a stack of ROIs."""

    theta: np.ndarray  # (n, NV) fitted parameters
    crlb: np.ndarray  # (n, NV) Cramer-Rao lower bounds (variances)
    logl: np.ndarray  # (n,) log-likelihood
    iterations: np.ndarray  # (n,) iterations used

    def __len__(self) -> int:
        return self.theta.shape[0]


class PSFModel:
    """Base class; see :class:`GaussianPSF` and :class:`SplinePSF`."""

    name = "psf"

    def fit(self, rois: np.ndarray, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        raise NotImplementedError

    def unpack(self, result: FitResult) -> Dict[str, np.ndarray]:
        """Named quantities derived from the raw parameters."""
        raise NotImplementedError

    @property
    def is_3d(self) -> bool:
        return False


@dataclass
class GaussianPSF(PSFModel):
    """Gaussian PSF with a free width.

    ``elliptical=False`` fits one width (``sigma``); ``elliptical=True`` fits
    ``sigma_x`` and ``sigma_y`` separately, which is what an astigmatic
    calibration measurement needs.
    """

    sigma: float = 1.0
    elliptical: bool = False
    name = "gauss"

    def fit(self, rois: np.ndarray, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        fn = _fit3d.fit_gauss_xy if self.elliptical else _fit3d.fit_gauss_free
        return FitResult(*fn(rois, float(self.sigma), int(iterations),
                             int(n_threads)))

    def unpack(self, result: FitResult) -> Dict[str, np.ndarray]:
        p, crlb = result.theta, np.clip(result.crlb, 0, None)
        out = {
            "x_roi": p[:, 0], "y_roi": p[:, 1],
            "photons": p[:, 2], "background": p[:, 3],
            "x_err_pix": np.sqrt(crlb[:, 0]), "y_err_pix": np.sqrt(crlb[:, 1]),
            "photons_err": np.sqrt(crlb[:, 2]),
            "background_err": np.sqrt(crlb[:, 3]),
        }
        if self.elliptical:
            out["sigma_x_pix"] = p[:, 4]
            out["sigma_y_pix"] = p[:, 5]
        else:
            out["sigma_pix"] = p[:, 4]
            out["sigma_err_pix"] = np.sqrt(crlb[:, 4])
        return out

    def __str__(self) -> str:
        kind = "elliptical" if self.elliptical else "free sigma"
        return f"GaussianPSF({kind}, start sigma={self.sigma:g} pix)"


class _GlobalPSF(PSFModel):
    """What the global fits share: the layout of the linked parameter vector.

    One emitter seen in several channels, with the parameters marked in
    ``shared`` fitted once for all of them and the rest once per channel, so
    the fitted vector is not a fixed width: use `slot` to find a parameter, or
    `unpack`, which names them.  See `smappy.dualfit` for where the link comes
    from.

    The parameters are ``(x, y, photons, background, *)``, the fifth being z
    for a spline and sigma for a Gaussian.  That is not a coincidence of
    numbering: because both models put a single fifth parameter in the same
    slot, one link array serves both, and `dualfit.build_link` -- which leaves
    that slot at offset 0, factor 1 -- needs to know nothing about the model.
    """

    shared: tuple = (True, True, False, False, True)   # x, y, photons, bg, *
    # the fifth parameter, as (value column, error column); `_fifth` converts
    # the raw fitted number and its error into them
    fifth = ("", "")

    @property
    def n_channels(self) -> int:
        raise NotImplementedError

    def _check_shared(self) -> tuple:
        shared = tuple(bool(f) for f in self.shared)
        if len(shared) != 5:
            raise ValueError("shared needs a flag for x, y, photons, "
                             "background and the model's fifth parameter")
        return shared

    def slot(self, parameter: int, channel: int = 0) -> int:
        """Where a parameter of one channel sits in the fitted vector."""
        at = 0
        for p in range(parameter):
            at += 1 if self.shared[p] else self.n_channels
        return at if self.shared[parameter] else at + channel

    @property
    def n_values(self) -> int:
        return self.slot(4, self.n_channels - 1) + 1

    def _fifth(self, value: np.ndarray, error: np.ndarray):
        """The fifth parameter in its own unit.  Identity unless overridden."""
        return value, error

    def unpack(self, result: FitResult) -> Dict[str, np.ndarray]:
        """Named quantities.  A free parameter gets one column per channel,
        suffixed ``_ch0``, ``_ch1``, ...; a shared one gets a single column.

        ``photons`` is always the total the emitter gave off as channel 0 sees
        it: shared, that is the fitted value; free, it is the sum over the
        channels.  ``ratio`` is the fraction in the last channel, which is
        what a ratiometric splitter measures colour with.
        """
        p, crlb = result.theta, np.clip(result.crlb, 0, None)
        out: Dict[str, np.ndarray] = {}

        def take(index, channel):
            column = self.slot(index, channel)
            return p[:, column], np.sqrt(crlb[:, column])

        for index, (name, error) in enumerate((("x_roi", "x_err_pix"),
                                               ("y_roi", "y_err_pix"),
                                               ("photons", "photons_err"),
                                               ("background", "background_err"),
                                               self.fifth)):
            for channel in range(1 if self.shared[index] else self.n_channels):
                suffix = "" if self.shared[index] else f"_ch{channel}"
                value, sigma = take(index, channel)
                if index == 4:
                    value, sigma = self._fifth(value, sigma)
                out[name + suffix] = value
                out[error + suffix] = sigma

        # x, y and the fifth carry no suffix downstream: the reference
        # channel's are the localization's, whether they were shared or fitted
        # per channel
        for name in ("x_roi", "y_roi", self.fifth[0],
                     "x_err_pix", "y_err_pix", self.fifth[1]):
            if name not in out:
                out[name] = out[f"{name}_ch0"]

        per_channel = [out.get(f"photons_ch{c}") for c in range(self.n_channels)]
        if per_channel[0] is None:                      # photons were shared
            out["ratio"] = np.zeros(len(p), float)
        else:
            total = np.sum(per_channel, axis=0)
            out["photons"] = total
            out["photons_err"] = np.sqrt(sum(
                out[f"photons_err_ch{c}"] ** 2 for c in range(self.n_channels)))
            with np.errstate(invalid="ignore", divide="ignore"):
                out["ratio"] = np.where(total > 0, per_channel[-1] / total, np.nan)
        if "background" not in out:
            out["background"] = out["background_ch0"]
            out["background_err"] = out["background_err_ch0"]
        return out

    def _linked(self) -> str:
        return ", ".join(n for n, on in zip(("x", "y", "photons", "background",
                                             self.fifth[0].rsplit("_", 1)[0]),
                                            self.shared) if on)


@dataclass
class GlobalSplinePSF(_GlobalPSF):
    """One emitter in several channels, each with its own spline calibration.

    The fifth parameter is z, shared by default: that link is the sqrt(2) a
    two-channel 3D fit buys.  Every channel's spline is checked to sit on the
    same z grid, which is what lets the link leave z at offset 0.
    """

    calibrations: tuple
    shared: tuple = (True, True, False, False, True)   # x, y, photons, bg, z
    z_start_nm: float = 0.0
    name = "cspline-global"
    fifth = ("z_nm", "z_err_nm")

    def __post_init__(self):
        self.calibrations = tuple(self.calibrations)
        self.shared = self._check_shared()
        first = self.calibrations[0]
        for other in self.calibrations[1:]:
            if other.shape != first.shape or other.dz != first.dz or other.z0 != first.z0:
                raise ValueError("every channel's spline must share one z grid")
        self._coeff = np.ascontiguousarray(
            np.stack([c.coeff for c in self.calibrations]), np.float32)

    @property
    def is_3d(self) -> bool:
        return True

    @property
    def n_channels(self) -> int:
        return len(self.calibrations)

    @property
    def calibration(self):
        """The reference channel's, which sets the z scale for all of them."""
        return self.calibrations[0]

    @property
    def mirror(self) -> bool:
        return bool(self.calibration.em_mirror)

    def fit(self, rois: np.ndarray, link: np.ndarray = None, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        if link is None:
            raise ValueError("a global fit needs the link between its channels")
        z_start = float(self.calibration.z_nm_to_index(self.z_start_nm))
        return FitResult(*_fit3d.fit_cspline_global(
            np.ascontiguousarray(rois, np.float32), self._coeff,
            np.ascontiguousarray(link, np.float32),
            np.array(self.shared, np.int32), z_start, int(iterations),
            int(n_threads)))

    def _fifth(self, value, error):
        cal = self.calibration
        return cal.z_index_to_nm(value), error * cal.dz

    def __str__(self) -> str:
        nz, ny, nx = self.calibration.shape
        return (f"GlobalSplinePSF({self.n_channels} channels, {nx}x{ny}x{nz} grid, "
                f"linked: {self._linked() or 'nothing'})")


@dataclass
class GlobalGaussianPSF(_GlobalPSF):
    """One emitter in several channels, each fitted with a free-width Gaussian.

    The 2D two-colour fit: the same machinery as `GlobalSplinePSF` with no PSF
    model to calibrate.  The fifth parameter is sigma, and it is free by
    default -- the two halves of a ratiometric splitter see different
    wavelengths and rarely share a focus, so one width for both would bias the
    photon split, which is the very thing that carries the colour.

    ``sigma`` is the starting width, one per channel; a single number is used
    for all of them.  Linking x and y (the default) is what makes this a
    two-channel fit at all: it is the registration that pins the emitter to
    one position, and the photons are then free to divide as they will.
    """

    sigma: tuple = (1.0, 1.0)
    shared: tuple = (True, True, False, False, False)  # x, y, photons, bg, sigma
    channels: int = 2
    name = "gauss-global"
    fifth = ("sigma_pix", "sigma_err_pix")

    def __post_init__(self):
        self.shared = self._check_shared()
        sigma = ((float(self.sigma),) * int(self.channels)
                 if np.isscalar(self.sigma) else
                 tuple(float(s) for s in self.sigma))
        if len(sigma) != int(self.channels):
            raise ValueError("sigma needs one starting width per channel, or one "
                             "number for all of them")
        if not all(s > 0 for s in sigma):
            raise ValueError("a starting width must be positive")
        self.sigma = sigma
        self._sigma = np.ascontiguousarray(sigma, np.float32)

    @property
    def n_channels(self) -> int:
        return len(self.sigma)

    def fit(self, rois: np.ndarray, link: np.ndarray = None, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        if link is None:
            raise ValueError("a global fit needs the link between its channels")
        return FitResult(*_fit3d.fit_gauss_global(
            np.ascontiguousarray(rois, np.float32), self._sigma,
            np.ascontiguousarray(link, np.float32),
            np.array(self.shared, np.int32), int(iterations), int(n_threads)))

    def __str__(self) -> str:
        widths = ", ".join(f"{s:g}" for s in self.sigma)
        return (f"GlobalGaussianPSF({self.n_channels} channels, start sigma "
                f"{widths} pix, linked: {self._linked() or 'nothing'})")


@dataclass
class SplinePSF(PSFModel):
    """Experimental PSF interpolated by a cubic spline, from a SMAP calibration.

    ``z_start`` is the starting z in nm relative to focus; it is converted to
    the spline's index units.  ``mirror`` reports whether the calibration was
    computed from mirrored bead images, in which case the ROIs must be flipped
    in x before fitting (see :func:`smappy.roi.cut_rois`).
    """

    calibration: SplineCalibration
    z_start_nm: float = 0.0
    name = "cspline"

    @property
    def is_3d(self) -> bool:
        return True

    @property
    def mirror(self) -> bool:
        """Whether ROIs must be x-flipped to match this calibration."""
        return bool(self.calibration.em_mirror)

    def fit(self, rois: np.ndarray, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        z_start = float(self.calibration.z_nm_to_index(self.z_start_nm))
        return FitResult(*_fit3d.fit_cspline(rois, self.calibration.coeff,
                                             z_start, int(iterations),
                                             int(n_threads)))

    def unpack(self, result: FitResult) -> Dict[str, np.ndarray]:
        p, crlb = result.theta, np.clip(result.crlb, 0, None)
        cal = self.calibration
        return {
            "x_roi": p[:, 0], "y_roi": p[:, 1],
            "photons": p[:, 2], "background": p[:, 3],
            "z_nm": cal.z_index_to_nm(p[:, 4]),
            "x_err_pix": np.sqrt(crlb[:, 0]), "y_err_pix": np.sqrt(crlb[:, 1]),
            "photons_err": np.sqrt(crlb[:, 2]),
            "background_err": np.sqrt(crlb[:, 3]),
            "z_err_nm": np.sqrt(crlb[:, 4]) * cal.dz,
        }

    def __str__(self) -> str:
        nz, ny, nx = self.calibration.shape
        return (f"SplinePSF({nx}x{ny}x{nz} grid, dz={self.calibration.dz:g} nm"
                f"{', mirrored calibration' if self.mirror else ''})")
