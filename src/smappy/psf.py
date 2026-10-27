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
            # the widths' errors: what Astigmatic z weighs the two widths by, and its z error
            out["sigma_x_err_pix"] = np.sqrt(crlb[:, 4])
            out["sigma_y_err_pix"] = np.sqrt(crlb[:, 5])
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
    # parameters after the fifth, as (value, error) columns: the 4Pi phase;
    # `_extra` converts them
    extra: tuple = ()

    @property
    def parameters(self) -> tuple:
        """(value column, error column) of each fitted parameter, in order."""
        return (("x_roi", "x_err_pix"), ("y_roi", "y_err_pix"),
                ("photons", "photons_err"), ("background", "background_err"),
                self.fifth) + tuple(self.extra)

    @property
    def n_channels(self) -> int:
        raise NotImplementedError

    def _check_shared(self) -> tuple:
        shared = tuple(bool(f) for f in self.shared)
        if len(shared) != len(self.parameters):
            names = ", ".join(v.rsplit("_", 1)[0] for v, _ in self.parameters[4:])
            raise ValueError("shared needs a flag for x, y, photons, "
                             f"background and {names}")
        return shared

    def slot(self, parameter: int, channel: int = 0) -> int:
        """Where a parameter of one channel sits in the fitted vector."""
        at = 0
        for p in range(parameter):
            at += 1 if self.shared[p] else self.n_channels
        return at if self.shared[parameter] else at + channel

    @property
    def n_values(self) -> int:
        return self.slot(len(self.parameters) - 1, self.n_channels - 1) + 1

    def _fifth(self, value: np.ndarray, error: np.ndarray):
        """The fifth parameter in its own unit.  Identity unless overridden."""
        return value, error

    def _extra(self, index: int, value: np.ndarray, error: np.ndarray):
        """A parameter after the fifth in its own unit.  Identity unless
        overridden."""
        return value, error

    def unpack(self, result: FitResult, link: np.ndarray = None
               ) -> Dict[str, np.ndarray]:
        """Named quantities.  A free parameter gets one column per channel,
        suffixed ``_ch0``, ``_ch1``, ...; a shared one gets a single column.

        ``photons`` is always the emitter's total over the channels.  Free,
        it is the sum of each channel's photons, each fitted number times what
        its model gives one photon (`photon_scales`).  Shared, one number N
        was fitted and channel c saw ``factor_c * N`` of its model -- the
        link's photon factor, the split either typed in or carried by the
        PSFs -- so the total is ``N * sum_c factor_c * scale_c``; for a
        jointly normalised spline pair and no ratio given, that sum is one
        and N is the total itself.  ``link`` is the fit's; without it the
        factors are taken to be one.  ``ratio`` is the fraction in the last
        channel, which is what a ratiometric splitter measures colour with,
        and 0 when the photons were shared: there is no split left to
        measure.
        """
        p, crlb = result.theta, np.clip(result.crlb, 0, None)
        out: Dict[str, np.ndarray] = {}
        scales = self.photon_scales()
        if self.shared[2]:
            factors = (np.ones((len(p), self.n_channels)) if link is None else
                       np.asarray(link, float)[:, 1, :, 2])
            total = factors @ scales

        def take(index, channel):
            column = self.slot(index, channel)
            value, sigma = p[:, column], np.sqrt(crlb[:, column])
            if index == 2:
                # a photon number is in its channel's model units; a shared
                # one is spread over every channel by the link
                k = total if self.shared[2] else scales[channel]
                value, sigma = value * k, sigma * k
            return value, sigma

        for index, (name, error) in enumerate(self.parameters):
            for channel in range(1 if self.shared[index] else self.n_channels):
                suffix = "" if self.shared[index] else f"_ch{channel}"
                value, sigma = take(index, channel)
                if index == 4:
                    value, sigma = self._fifth(value, sigma)
                elif index > 4:
                    value, sigma = self._extra(index, value, sigma)
                out[name + suffix] = value
                out[error + suffix] = sigma

        # x, y, the fifth and what follows carry no suffix downstream: the
        # reference channel's are the localization's, whether they were
        # shared or fitted per channel
        later = [n for pair in self.parameters[4:] for n in pair]
        for name in ("x_roi", "y_roi", "x_err_pix", "y_err_pix", *later):
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

    def photon_scales(self) -> np.ndarray:
        """What one fitted photon of each channel is, in photons.

        One, unless a channel's model is not normalised to one photon: see
        `GlobalSplinePSF.photon_scales`.
        """
        return np.ones(self.n_channels)

    def _linked(self) -> str:
        names = ("x", "y", "photons", "background",
                 *(v.rsplit("_", 1)[0] for v, _ in self.parameters[4:]))
        return ", ".join(n for n, on in zip(names, self.shared) if on)


def signal_integral(calibration) -> float:
    """What a spline gives a fitted photon of, at focus.

    A dual bead calibration normalises its two PSFs together, so that their
    signal sums to one and the pair keeps the beads' split; each channel's
    share is stored with it, ``photon_normalization``
    (`calibrate.core.positive_pair_models`).  A calibration without that
    record (SMAP's calibrations, one built by hand) is taken to be normalised
    to one -- unless it is a dual bead calibration of this program's, whose
    shares are then unknown and which has to be rebuilt.
    """
    p = getattr(calibration, "parameters", None) or {}
    if "photon_normalization" in p:              # stored with the PSF
        value = float(p["photon_normalization"])
        return value if np.isfinite(value) and value > 0 else 1.0
    if str(p.get("method", "")).startswith("smappy_dual_bead"):
        source = getattr(calibration, "source", None)
        raise ValueError(f"{source or 'this dual calibration'} does not store "
                         "its channels' photon shares (photon_normalization); "
                         "rebuild it from the bead stacks")
    return 1.0


def photon_scales(calibrations) -> np.ndarray:
    """`signal_integral` of each channel's spline, over the first's."""
    values = np.array([signal_integral(c) for c in calibrations], float)
    return values / values[0]


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

    def photon_scales(self) -> np.ndarray:
        """What one fitted photon of each channel is: `signal_integral`.

        A dual bead calibration normalises both halves by the *main* half's
        integral (`calibrate.core.positive_pair_models`), so that the pair
        keeps the beads' brightness split: the secondary spline integrates to
        that split, not to one.  A photon number fitted against it is then in
        bead units, and read as photons it put ``photons_ch1`` out by the
        split -- the total 3.4x the truth for a splitter sending a fifth of
        the light to the second half -- and the ratio against the beads'
        rather than the dye's.  SMAP divides by the same number
        (``PhotonRatios / normf``).  A calibration without the record (SMAP's,
        one built by hand) is taken to be normalised per channel.
        """
        return np.array([signal_integral(c) for c in self.calibrations], float)

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
class Global4PiPSF(_GlobalPSF):
    """One emitter in every phase channel of a 4Pi microscope: the IAB model.

    Six parameters, ``(x, y, photons, background, z, phase)`` -- see
    `smappy.fourpi` for the model and its conventions.  By default everything
    but the background is shared: the channels' shares of the light are in
    their splines, so one photon number is the emitter's total, and linking it
    is worth up to 1.5x in z (Li et al., Nat. Commun. 13, 3133 (2022)).
    More than that: the phase *is* the photons' split between the channels,
    so a photon number free in each channel absorbs the interference and
    leaves the phase undetermined -- exactly so where A and B have the
    envelope's shape, nearly so in a real PSF.  SMAP's 4Pi fitter unlinked
    them by default.

    z is the envelope's -- unambiguous, and as precise as the PSF's shape
    makes it -- and is written as ``z_nm`` and ``z_env_nm`` both: the phase,
    wrapped to (-pi, pi], gives z to a few nanometres but only modulo the
    fringe, and unwrapping it replaces ``z_nm`` (`smappy.fourpi`).

    ``n_starts`` z planes of the phasor start are tried, each with
    ``phase_starts`` phases around the circle, keeping the best likelihood.
    """

    calibration: object             # a smappy.fourpi.FourPiCalibration
    shared: tuple = (True, True, True, False, True, True)  # x, y, N, bg, z, phase
    n_starts: int = 2
    phase_starts: int = 1
    z_step: int = 1
    name = "cspline-4pi"
    fifth = ("z_nm", "z_err_nm")
    extra = (("phase_rad", "phase_err_rad"),)

    def __post_init__(self):
        self.shared = self._check_shared()

    @property
    def is_3d(self) -> bool:
        return True

    @property
    def n_channels(self) -> int:
        return self.calibration.n_channels

    def fit(self, rois: np.ndarray, link: np.ndarray = None, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        if link is None:
            from .fourpi import link as plain
            link = plain(len(rois), self.n_channels)
        return FitResult(*_fit3d.fit_cspline4pi_global(
            np.ascontiguousarray(rois, np.float32), self.calibration.coeff,
            np.ascontiguousarray(link, np.float32), np.array(self.shared, np.int32),
            int(self.n_starts), int(self.phase_starts), int(self.z_step),
            int(iterations), int(n_threads)))

    def _fifth(self, value, error):
        cal = self.calibration
        return cal.z_index_to_nm(value), error * cal.dz

    def _extra(self, index, value, error):
        return np.angle(np.exp(1j * value)), error

    def photon_scales(self) -> np.ndarray:
        """A channel's share of the light, so that photons fitted per channel
        add up to the emitter's and a shared N (factor one in every channel)
        is the total: the shares sum to one."""
        return self.calibration.photon_shares()

    def unpack(self, result: FitResult, link: np.ndarray = None) -> Dict[str, np.ndarray]:
        out = super().unpack(result, link)
        if self.shared[2]:
            # the shared N is already the total: the link's factors are one
            # and the shares sum to one, not the channel count
            out["photons"] = result.theta[:, self.slot(2)]
            out["photons_err"] = np.sqrt(np.clip(result.crlb[:, self.slot(2)], 0, None))
        # A fit that ended on an end of the model's z grid was clamped there:
        # its true z could be anywhere beyond, and its CRLB, taken at the
        # clamp, says nothing.  On the Nup96 example a few per cent piled up
        # at the top of the grid, every one with a small error.  As in
        # Astigmatic z, such a fit gets the grid's whole range as its error,
        # which 4Pi z then weighs at nothing.
        nz = self.calibration.shape[0]
        z_index = result.theta[:, self.slot(4)]
        edge = (z_index < 0.5) | (z_index > nz - 0.5)
        span = nz * self.calibration.dz
        for name in [n for n in out if n.startswith("z_err_nm")]:
            out[name] = np.where(edge, span, out[name])
        out["z_env_nm"], out["z_env_err_nm"] = out["z_nm"], out["z_err_nm"]
        return out

    def __str__(self) -> str:
        nz, ny, nx = self.calibration.shape
        return (f"Global4PiPSF({self.n_channels} channels, {nx}x{ny}x{nz} grid, "
                f"linked: {self._linked() or 'nothing'})")


@dataclass
class SplinePSF(PSFModel):
    """Experimental PSF interpolated by a cubic spline, from a SMAP calibration.

    ``z_start`` is the starting z in nm relative to focus; it is converted to
    the spline's index units.
    """

    calibration: SplineCalibration
    z_start_nm: float = 0.0
    name = "cspline"

    @property
    def is_3d(self) -> bool:
        return True

    def fit(self, rois: np.ndarray, iterations: int = 50,
            n_threads: int = 0) -> FitResult:
        z_start = float(self.calibration.z_nm_to_index(self.z_start_nm))
        return FitResult(*_fit3d.fit_cspline(rois, self.calibration.coeff,
                                             z_start, int(iterations),
                                             int(n_threads)))

    def unpack(self, result: FitResult) -> Dict[str, np.ndarray]:
        p, crlb = result.theta, np.clip(result.crlb, 0, None)
        cal = self.calibration
        # one for a single-channel calibration; for one half of a dual one,
        # its share of the pair's joint normalisation (`signal_integral`)
        scale = signal_integral(cal)
        return {
            "x_roi": p[:, 0], "y_roi": p[:, 1],
            "photons": p[:, 2] * scale, "background": p[:, 3],
            "z_nm": cal.z_index_to_nm(p[:, 4]),
            "x_err_pix": np.sqrt(crlb[:, 0]), "y_err_pix": np.sqrt(crlb[:, 1]),
            "photons_err": np.sqrt(crlb[:, 2]) * scale,
            "background_err": np.sqrt(crlb[:, 3]),
            "z_err_nm": np.sqrt(crlb[:, 4]) * cal.dz,
        }

    def __str__(self) -> str:
        nz, ny, nx = self.calibration.shape
        return f"SplinePSF({nx}x{ny}x{nz} grid, dz={self.calibration.dz:g} nm)"
