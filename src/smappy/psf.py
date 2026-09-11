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


@dataclass
class GlobalSplinePSF(PSFModel):
    """One emitter in several channels, each with its own spline calibration.

    The parameters marked in ``shared`` are fitted once for all channels and
    the rest once per channel, so the fitted vector is not a fixed width: use
    `slot` to find a parameter, or `unpack`, which names them.  See
    `smappy.dualfit` for where the link comes from.
    """

    calibrations: tuple
    shared: tuple = (True, True, False, False, True)   # x, y, photons, bg, z
    z_start_nm: float = 0.0
    name = "cspline-global"

    def __post_init__(self):
        self.calibrations = tuple(self.calibrations)
        self.shared = tuple(bool(f) for f in self.shared)
        if len(self.shared) != 5:
            raise ValueError("shared needs a flag for x, y, photons, background, z")
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

    def slot(self, parameter: int, channel: int = 0) -> int:
        """Where a parameter of one channel sits in the fitted vector."""
        at = 0
        for p in range(parameter):
            at += 1 if self.shared[p] else self.n_channels
        return at if self.shared[parameter] else at + channel

    @property
    def n_values(self) -> int:
        return self.slot(4, self.n_channels - 1) + 1

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

    def unpack(self, result: FitResult) -> Dict[str, np.ndarray]:
        """Named quantities.  A free parameter gets one column per channel,
        suffixed ``_ch0``, ``_ch1``, ...; a shared one gets a single column.

        ``photons`` is always the total the emitter gave off as channel 0 sees
        it: shared, that is the fitted value; free, it is the sum over the
        channels.  ``ratio`` is the fraction in the last channel, which is
        what a ratiometric splitter measures colour with.
        """
        p, crlb = result.theta, np.clip(result.crlb, 0, None)
        cal = self.calibration
        out: Dict[str, np.ndarray] = {}

        def take(index, channel):
            column = self.slot(index, channel)
            return p[:, column], np.sqrt(crlb[:, column])

        for index, (name, error) in enumerate((("x_roi", "x_err_pix"),
                                               ("y_roi", "y_err_pix"),
                                               ("photons", "photons_err"),
                                               ("background", "background_err"),
                                               ("z", "z_err"))):
            for channel in range(1 if self.shared[index] else self.n_channels):
                suffix = "" if self.shared[index] else f"_ch{channel}"
                value, sigma = take(index, channel)
                if name == "z":
                    out["z_nm" + suffix] = cal.z_index_to_nm(value)
                    out["z_err_nm" + suffix] = sigma * cal.dz
                else:
                    out[name + suffix] = value
                    out[error.replace("_err", "_err") + suffix] = sigma

        # x, y and z carry no suffix downstream: the reference channel's are
        # the localization's, whether they were shared or fitted per channel
        for name in ("x_roi", "y_roi", "z_nm", "x_err_pix", "y_err_pix", "z_err_nm"):
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

    def __str__(self) -> str:
        nz, ny, nx = self.calibration.shape
        linked = ", ".join(n for n, on in zip(("x", "y", "photons", "background", "z"),
                                              self.shared) if on)
        return (f"GlobalSplinePSF({self.n_channels} channels, {nx}x{ny}x{nz} grid, "
                f"linked: {linked or 'nothing'})")


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
