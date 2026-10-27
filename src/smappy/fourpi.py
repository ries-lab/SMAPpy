"""4Pi: the IAB spline model, its calibration and the global fit's link.

In a 4Pi microscope two opposing objectives collect one molecule's light,
the two beams interfere and the microscope splits the result into (usually)
four images whose interference phases differ by about a quarter turn each.
The model of one channel c is the IAB model (Li et al., Opt. Lett. 45, 3765
(2020)):

    mu_c(x, y, z, phase) = N * (I_c + A_c cos(phase) + B_c sin(phase)) + bg_c

with I_c the incoherent part and A_c, B_c the interference *demodulated by
the fringe*: the fast oscillation along z is carried by `phase`, so the three
volumes vary as slowly as the envelope does and can be splined like any PSF.
z and the phase are separate fit parameters.  The phase gives z to a few
nanometres but only modulo the fringe period; the envelope -- astigmatic on
the microscopes this was written for -- gives z without that ambiguity, and
unwrapping is a step after the fit.

Two conventions, chosen once here:

* **Each channel's phase offset and its share of the light are inside its own
  I, A and B**, so the channels share one phase with no offset.  That is how
  uiPSF v1 exports a model, and it is what lets an offset that is not exactly
  pi/2, an imbalance between the arms or an s channel with less modulation be
  modelled at all: they are simply other splines.  (Bates et al., Nat. Methods
  19, 603 (2022), measured 5-10 nm of z bias from assuming equal amplitudes.)
* **Photons**: the channels' I are normalised together so that their sum over
  the focal plane is one -- `normalise` -- and a fitted N is then the photons
  detected in all channels together.  uiPSF v1 normalised to two, so its N was
  half the photons; its models are renormalised when they are read.

The phase is tied to z by `phase_per_nm`: an emitter that moves by dz (in
`z_nm`, this package's sign) moves its phase by ``phase_per_nm * dz``.  Its
magnitude is 2 pi over the fringe period; its sign is the calibration's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

FORMAT = "smappy-4pi-calibration"
VERSION = 1


@dataclass
class FourPiCalibration:
    """The IAB spline model of every channel, on one z grid.

    ``coeff`` is (C, 3, 64, nz, ny, nx): channel, then I, A, B, then the
    fitter's coefficient layout (`smappy.io.calibration`).  ``volumes`` are the
    knots the splines were made from, (C, 3, nz + 1, ny + 1, nx + 1), kept to
    draw and to check the model.  z is in spline index units in the fitter and
    in nm outside it, with `SplineCalibration`'s sign: z_nm grows as the index
    falls.
    """

    coeff: np.ndarray
    dz: float                          # nm per z index
    z0: float                          # index of the focal plane
    period_nm: float                   # the fringe period z_T
    phase_per_nm: float                # d(phase) / d(z_nm), +-2 pi / period
    volumes: Optional[np.ndarray] = None
    transforms: Optional[np.ndarray] = None   # (C, 3, 3) channel c -> channel 0, pixels
    #: the channel layout the transforms were measured on (`ChannelLayout.to_dict`): the
    #: transforms are between its channel images, and hold for another layout only if
    #: its channels have the same size
    layout: Optional[dict] = None
    pixelsize_nm: Optional[tuple] = None      # (x, y)
    source: Optional[Path] = None
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        self.coeff = np.ascontiguousarray(self.coeff, np.float32)
        if self.coeff.ndim != 6 or self.coeff.shape[1] != 3 or self.coeff.shape[2] != 64:
            raise ValueError("4Pi coefficients must have shape (channels, 3, 64, nz, ny, nx)")
        if not 1 <= self.coeff.shape[0] <= 4:
            raise ValueError("a 4Pi model has one to four channels")
        if not (self.dz > 0 and self.period_nm > 0):
            raise ValueError("dz and the fringe period must be positive")

    @property
    def n_channels(self) -> int:
        return self.coeff.shape[0]

    #: what a fit's provenance records of a spline calibration; a 4Pi model
    #: says nothing about EM gain
    em_on = None

    def maps(self) -> Optional[list]:
        """The channels' registration kept with the model, as
        `ChannelLayout` maps (channel c -> 0), or None: the native file's
        ``transforms``, or a uiPSF v1 model's affines (`maps_from_uipsf_v1`)."""
        if self.transforms is not None:
            return [np.asarray(t, float) for t in self.transforms]
        v1 = self.parameters.get("uipsf_v1_T")
        centre = (self.parameters.get("uipsf_v1") or {}).get("imgcenter")
        if v1 is not None and centre is not None:
            from .channels import maps_from_uipsf_v1
            return maps_from_uipsf_v1(v1, centre, bool(self.parameters.get("transposed_from_v1")))
        return None

    @property
    def shape(self) -> tuple:
        """The spline grid, (nz, ny, nx)."""
        return self.coeff.shape[3:]

    def z_index_to_nm(self, z_index):
        return -(np.asarray(z_index) - self.z0) * self.dz

    def z_nm_to_index(self, z_nm):
        return -np.asarray(z_nm) / self.dz + self.z0

    def photon_shares(self) -> np.ndarray:
        """Each channel's share of the light: its I summed over the focal
        plane.  They sum to one for a normalised model."""
        if self.volumes is not None:
            plane = int(round(self.z0))
            return self.volumes[:, 0, plane].sum(axis=(1, 2)).astype(float)
        return np.full(self.n_channels, 1.0 / self.n_channels)


# ---------------------------------------------------------------- building

def normalise(I, A, B, focal_plane: int):
    """Offset and scale the channels' volumes together, as the fitter needs.

    ``I``, ``A`` and ``B`` are (C, nz, ny, nx).  A learned model can sit on a
    small pedestal (its background was not quite zero): the largest constant
    that keeps every channel's PSF non-negative at every phase,
    min(I - sqrt(A^2 + B^2)), is taken off I -- uiPSF v1 does the same.  Then
    everything is divided by the sum of I over the focal plane, so that N is
    the photons of all channels together.
    """
    I, A, B = (np.asarray(v, float) for v in (I, A, B))
    I = I - float(np.min(I - np.hypot(A, B)))
    scale = float(I[:, focal_plane].sum())
    if not scale > 0:
        raise ValueError("the model has no light in its focal plane")
    return I / scale, A / scale, B / scale


def from_volumes(I, A, B, dz: float, z0: float, period_nm: float, phase_per_nm: float,
                 **kwargs) -> FourPiCalibration:
    """A calibration from each channel's I, A and B sampled on a grid.

    ``I``, ``A``, ``B`` are (C, nz, ny, nx) knots, z0 the knot index of the
    focal plane.  They are normalised (`normalise`) and splined with the
    bead calibration's own `spline_coefficients`, so a 4Pi model and a plain
    one share the coefficient layout.
    """
    from .calibrate.core import spline_coefficients
    # the knots as the file keeps them, so that a saved model splines back to
    # the same coefficients
    I, A, B = (v.astype(np.float32) for v in normalise(I, A, B, int(round(z0))))
    coeff = np.stack([np.stack([spline_coefficients(v[c]) for v in (I, A, B)])
                      for c in range(I.shape[0])])
    return FourPiCalibration(coeff=coeff, dz=float(dz), z0=float(z0),
                             period_nm=float(period_nm), phase_per_nm=float(phase_per_nm),
                             volumes=np.stack([I, A, B], axis=1),
                             **kwargs)


def model_calibration(channels: int = 4, size: int = 27, dz: float = 20.0,
                      z_range_nm: float = 600.0, period_nm: float = 260.0,
                      sigma_px: float = 1.3, astigmatism_nm: float = 300.0,
                      depth_nm: float = 400.0, modulation: float = 0.8,
                      offsets: Sequence[float] = None) -> FourPiCalibration:
    """An astigmatic 4Pi model in closed form, for tests and the pages.

    The envelope is an elliptical Gaussian whose widths grow away from foci
    at +-``astigmatism_nm``/2; each channel carries a quarter of the light and
    its interference is ``modulation * I * cos(phase + offset_c)``, with
    offsets 0, pi/2, pi, 3 pi/2 unless given.  So A_c = m I cos(offset_c)
    and B_c = -m I sin(offset_c), the demodulated form: no fringe in the
    volumes, all of it in the phase.
    """
    offsets = (np.arange(channels) * np.pi / 2 if offsets is None
               else np.asarray(offsets, float))
    z = np.arange(-z_range_nm, z_range_nm + dz / 2, dz)
    centre = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size] - centre

    def width(shift):
        return sigma_px * np.sqrt(1 + ((z - shift) / depth_nm) ** 2)

    sx, sy = width(astigmatism_nm / 2), width(-astigmatism_nm / 2)
    envelope = np.exp(-xx ** 2 / (2 * sx[:, None, None] ** 2)
                      - yy ** 2 / (2 * sy[:, None, None] ** 2))
    envelope /= envelope.sum(axis=(1, 2), keepdims=True)
    I = np.stack([envelope / channels] * channels)
    A = np.stack([modulation * envelope / channels * np.cos(o) for o in offsets])
    B = np.stack([-modulation * envelope / channels * np.sin(o) for o in offsets])
    # the knots run with the index, z_nm against it
    I, A, B = I[:, ::-1], A[:, ::-1], B[:, ::-1]
    return from_volumes(I, A, B, dz=dz, z0=(len(z) - 1) / 2, period_nm=period_nm,
                        phase_per_nm=-2 * np.pi / period_nm,
                        parameters={"method": "closed-form astigmatic 4Pi model",
                                    "modulation": modulation,
                                    "offsets_rad": [float(o) for o in offsets]})


# ---------------------------------------------------------------- evaluation

def evaluate(cal: FourPiCalibration, channel: int, x: float, y: float, z_index: float,
             phase: float, roisize: int) -> np.ndarray:
    """One channel's model, for one photon, in a (roisize, roisize) ROI.

    x and y are ROI coordinates (pixel k centred on k), z in index units --
    the fitter's evaluation, `evaluate_spline` of each of I, A and B.
    """
    from .io.calibration import evaluate_spline
    parts = [evaluate_spline(SimpleNamespace(coeff=cal.coeff[channel, k]), x, y, z_index,
                             roisize) for k in range(3)]
    return parts[0] + parts[1] * np.cos(phase) + parts[2] * np.sin(phase)


def spots(cal: FourPiCalibration, x, y, z_index, phase, roisize: int,
          chunk: int = 512) -> np.ndarray:
    """Every channel's model for many emitters at once, one photon each:
    (n, C, roisize, roisize).

    The fitter's evaluation (`evaluate`), vectorised over the emitters: each
    one's 64 monomials against the coefficient block its ROI reads.  x and y
    are ROI coordinates, z in index units; a page simulates thousands of
    spots this way in a second.
    """
    x, y, z, phase = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, z_index, phase))
    n = len(x)
    C = cal.n_channels
    nz, ny, nx = cal.shape
    out = np.empty((n, C, roisize, roisize), np.float32)
    off_x = int(np.floor((nx + 1 - roisize) / 2))
    off_y = int(np.floor((ny + 1 - roisize) / 2))
    powers = np.arange(4)
    pixels = np.arange(roisize)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        xc = -((x[sl] - roisize / 2) + 0.5)
        yc = -((y[sl] - roisize / 2) + 0.5)
        x0, y0, z0 = np.floor(xc).astype(int), np.floor(yc).astype(int), np.floor(z[sl]).astype(int)
        dx, dy, dz = xc - x0, yc - y0, z[sl] - z0
        delta = (dz[:, None, None, None] ** powers[None, :, None, None]
                 * dy[:, None, None, None] ** powers[None, None, :, None]
                 * dx[:, None, None, None] ** powers[None, None, None, :]).reshape(-1, 64)
        ix = np.clip(pixels[None, :] + x0[:, None] + off_x, 0, nx - 1)
        iy = np.clip(pixels[None, :] + y0[:, None] + off_y, 0, ny - 1)
        iz = np.clip(z0, 0, nz - 1)
        cos, sin = np.cos(phase[sl]), np.sin(phase[sl])
        for c in range(C):
            parts = []
            for k in range(3):
                plane = cal.coeff[c, k]                       # (64, nz, ny, nx)
                block = plane[:, iz[:, None, None], iy[:, :, None], ix[:, None, :]]
                parts.append(np.einsum("kn,knij->nij", delta.T, block))
            out[sl, c] = (parts[0] + parts[1] * cos[:, None, None]
                          + parts[2] * sin[:, None, None])
    return out


def simulate_rois(cal: FourPiCalibration, n: int, roisize: int = 13,
                  photons=3000.0, background: float = 10.0,
                  z_nm=(-400.0, 400.0), phase0=None, xy_spread: float = 0.5,
                  seed: int = 0):
    """Poisson ROIs of single emitters in every channel, with their truth.

    Returns ``rois`` (n, C, roisize, roisize) and a dict of the truth: x, y
    (ROI pixels), z_nm, the phase the fitter should find, photons and the
    background per channel.  ``z_nm`` is a range to draw from or a value per
    emitter, ``photons`` a number or one per emitter.  Each emitter's phase is
    tied to its z by `phase_per_nm`, plus ``phase0``, the cavity's phase: one
    number, one per emitter (a drift, `cavity_drift`), or None for a random
    one each, which is what a test of the fitter alone wants.
    """
    rng = np.random.default_rng(seed)
    centre = (roisize - 1) / 2
    x = centre + rng.uniform(-xy_spread, xy_spread, n)
    y = centre + rng.uniform(-xy_spread, xy_spread, n)
    z = rng.uniform(*z_nm, n) if np.ndim(z_nm) == 1 and len(z_nm) == 2 else \
        np.broadcast_to(np.asarray(z_nm, float), (n,)).copy()
    offset = (rng.uniform(-np.pi, np.pi, n) if phase0 is None
              else np.broadcast_to(np.asarray(phase0, float), (n,)))
    phase = cal.phase_per_nm * z + offset
    photons = np.broadcast_to(np.asarray(photons, float), (n,))
    model = spots(cal, x, y, cal.z_nm_to_index(z), phase, roisize)
    mu = photons[:, None, None, None] * np.clip(model, 0, None) + background
    rois = rng.poisson(mu).astype(np.float32)
    truth = {"x": x, "y": y, "z_nm": z, "phase": np.angle(np.exp(1j * phase)),
             "phase0": np.angle(np.exp(1j * offset)),
             "photons": photons.copy(), "background": np.full(n, float(background))}
    return rois, truth


def cavity_drift(n_frames: int, rate_rad: float = 0.0, wander_rad: float = 0.0,
                 start_rad: float = 0.0, seed: int = 0) -> np.ndarray:
    """The cavity's phase in each frame: a steady drift of ``rate_rad`` per
    frame and a random walk of ``wander_rad`` per frame on top -- the slow
    change of the arms' path difference that 4Pi data have to follow
    (0.04 rad/s was measured by Liu & Huang 2020)."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, wander_rad, n_frames)) if wander_rad else 0.0
    return start_rad + rate_rad * np.arange(n_frames) + walk


def link(n: int, channels: int, dx=None, dy=None) -> np.ndarray:
    """The global fit's link for n ROIs: (n, 2, C, 6) offsets, then factors.

    ``dx``, ``dy`` (n, C) are each channel's sub-pixel offset of the emitter
    from its ROI's position in channel 0's -- what is left after cutting the
    ROIs where the registration puts them.  Factors are one: the channels'
    shares of the light are in their splines.
    """
    out = np.zeros((n, 2, channels, 6), np.float32)
    out[:, 1] = 1.0
    if dx is not None:
        out[:, 0, :, 0] = dx
    if dy is not None:
        out[:, 0, :, 1] = dy
    return out


# ---------------------------------------------------------------- files

def save_fourpi_calibration(path, cal: FourPiCalibration, overwrite: bool = False) -> None:
    """Write the native HDF5 file, atomically.

    The file holds the knots -- each channel's I, A and B on the grid -- and
    not the coefficients, which are a fixed function of them and 64 times the
    size (125 MB for four channels on a 60 x 26 x 26 grid); `load` splines
    them again, in a fraction of a second.
    """
    import os
    import tempfile

    import h5py

    from .io.calibration import _jsonable
    path = Path(path)
    if cal.volumes is None:
        raise ValueError("only a calibration made from its knots (from_volumes) can be saved")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs.update(format=FORMAT, version=VERSION,
                           knots="C,IAB,z,y,x", lateral_unit="pixel", axial_unit="nm", dz=cal.dz, z0=cal.z0,
                           period_nm=cal.period_nm, phase_per_nm=cal.phase_per_nm)
            if cal.pixelsize_nm is not None:
                f.attrs["pixelsize_nm"] = np.asarray(cal.pixelsize_nm, float)
            f.create_dataset("volumes", data=cal.volumes, compression="gzip")
            if cal.transforms is not None:
                f.create_dataset("transforms", data=np.asarray(cal.transforms, float))
            if cal.layout is not None:
                f.create_dataset("layout_json", data=json.dumps(cal.layout))
            f.create_dataset("parameters_json", data=json.dumps(cal.parameters,
                                                                default=_jsonable))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_fourpi_calibration(path) -> FourPiCalibration:
    """Read a native 4Pi calibration, or a uiPSF v1 4Pi model (`load_uipsf_v1`)."""
    import h5py
    path = Path(path)
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            if "res" in f and "channel0" in f["res"] and "A_model" in f["res/channel0"]:
                return load_uipsf_v1(path)
            raise ValueError(f"{path.name} is not a 4Pi calibration")
        if f.attrs.get("version") != VERSION:
            raise ValueError(f"{path.name}: unsupported 4Pi calibration version")
        volumes = f["volumes"][...]
        return from_volumes(
            volumes[:, 0], volumes[:, 1], volumes[:, 2],
            dz=float(f.attrs["dz"]), z0=float(f.attrs["z0"]),
            period_nm=float(f.attrs["period_nm"]),
            phase_per_nm=float(f.attrs["phase_per_nm"]),
            transforms=f["transforms"][...] if "transforms" in f else None,
            layout=json.loads(f["layout_json"][()]) if "layout_json" in f else None,
            pixelsize_nm=tuple(f.attrs["pixelsize_nm"]) if "pixelsize_nm" in f.attrs else None,
            source=path, parameters=json.loads(f["parameters_json"][()]))


def is_fourpi_calibration(path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return (f.attrs.get("format") == FORMAT or
                    ("res" in f and "channel0" in f["res"] and "A_model" in f["res/channel0"]))
    except OSError:
        return False


def load_uipsf_v1(path, transpose: bool = None) -> FourPiCalibration:
    """A uiPSF v1 4Pi model (``psfmodel_*_4pi.h5``), as this package's model.

    v1 keeps each channel's incoherent part ``I_model`` and its complex,
    demodulated interference ``A_model`` (``res/channel<c>/``).  A channel at
    the bead phase theta was ``I + 2 Re(A e^{i(2 pi s / z_T - theta)})``, s the
    slice from the stack's centre, so A = 2 Re(A_model) and B = -2 Im(A_model)
    with the fitter's phase 2 pi s / z_T - theta: the phase grows with the
    slice, so it falls with z_nm.  That holds for a model of beads; v1's
    in-situ model (``datatype`` ``smlm``) builds the two arms' waves and
    demodulates with the opposite sign (``Zphase / zT``, not ``-Zphase /
    zT``), so its phase grows with z_nm -- read with the beads' sign, the
    Nup96 data of uiPSF's examples put z from the phase at minus the
    envelope's over the whole range.  The splines are made again from those
    volumes rather than read from ``locres/coeff``, which v1 normalised to two
    photons and wrote in its own coefficient order; z_T is the period v1 was
    given (``fpi.modulation_period``), which it did not measure.

    v1 read a MATLAB file's images as h5py hands them over, rows and columns
    swapped (`smappy.io.quadrants`), so a model it learned from ``.mat`` beads
    is transposed against the images as this package reads them.
    ``transpose`` puts it the right way round; None decides from v1's own
    record (``format`` ``.mat`` and no ``swapxy``).  Its channel transforms,
    kept in ``parameters``, stay in v1's frame.
    """
    import h5py
    path = Path(path)
    with h5py.File(path, "r") as f:
        params = json.loads(f.attrs["params"]) if "params" in f.attrs else {}
        res = f["res"]
        channels = sorted((k for k in res if k.startswith("channel")),
                          key=lambda k: int(k[len("channel"):]))
        I = np.stack([np.real(res[c]["I_model"][...]) for c in channels])
        Ac = np.stack([res[c]["A_model"][...] for c in channels])
        T = res["T"][...] if "T" in res else None
        extra = {k: res[k][...].tolist() for k in ("imgcenter", "xyshift") if k in res}
    if transpose is None:
        transpose = (params.get("format") == ".mat" and not params.get("swapxy", False))
    if transpose:
        I, Ac = np.swapaxes(I, -1, -2), np.swapaxes(Ac, -1, -2)
    fpi = params.get("fpi", {})
    period_nm = float(fpi.get("modulation_period", 0.26)) * 1000
    dz = float(params.get("pixel_size", {}).get("z", 0.05)) * 1000
    px = params.get("pixel_size", {})
    pixelsize = (float(px.get("x", 0)) * 1000, float(px.get("y", 0)) * 1000) if px else None
    if pixelsize and transpose:
        pixelsize = pixelsize[::-1]
    nz = I.shape[1]
    sign = 1.0 if params.get("datatype") == "smlm" else -1.0
    cal = from_volumes(I, 2 * np.real(Ac), -2 * np.imag(Ac), dz=dz, z0=(nz - 1) / 2,
                       period_nm=period_nm, phase_per_nm=sign * 2 * np.pi / period_nm,
                       source=path,
                       pixelsize_nm=pixelsize,
                       parameters={"method": "uiPSF v1 4Pi model", "uipsf_v1": extra,
                                   "transposed_from_v1": bool(transpose),
                                   "uipsf_v1_params": params})
    if T is not None:
        cal.parameters["uipsf_v1_T"] = np.asarray(T).tolist()
    return cal


# ---------------------------------------------------------------- phase to z
#
# The fit gives two answers for z: the envelope's, z_env, unambiguous and as
# precise as the PSF's shape makes it (tens of nm), and the phase, a few nm
# but only modulo the fringe period.  With k = `phase_per_nm`, an emitter at
# z has the phase k z + phase0(t), phase0 the cavity's, which drifts.  So
#
#   1. phase0(t) is the circular mean of phase - k z_env over a time window
#      (`cavity_phase`); and
#   2. each localization's z is the candidate (phase - phase0 + 2 pi n) / k
#      nearest its z_env -- and how near, against z_env's error, is the
#      probability that it is the right fringe (`unwrap`).
#
# SMAP's Phase2z4Pi did both with a hard rounding and no use of the errors;
# its reported error ignored the chance of a wrong fringe, which is what a
# ghost is.

@dataclass
class CavityPhase:
    """phase0 per time window, and per localization by interpolation."""

    centres: np.ndarray        # frame at each window's centre
    phase0: np.ndarray         # unwrapped along the windows, rad
    error: np.ndarray          # rad
    counts: np.ndarray         # localizations in each window
    at: np.ndarray             # phase0 at each localization, rad


def _weights(phase_err, z_env_err_nm, k):
    """How much a localization says about phase0: one over the variance of
    phase - k z_env.  One whose z_env is uncertain by more than a fringe says
    nearly nothing, and is weighted so."""
    var = np.asarray(phase_err, float) ** 2 + (k * np.asarray(z_env_err_nm, float)) ** 2
    w = np.where(np.isfinite(var) & (var > 0), 1.0 / np.maximum(var, 1e-12), 0.0)
    return w


def cavity_phase(phase, phase_err, z_env_nm, z_env_err_nm, frame, phase_per_nm: float,
                 window_frames: int = 1000, smoothing: int = 1,
                 min_count: int = 20) -> CavityPhase:
    """The cavity's phase over time, from the localizations themselves.

    In each window of ``window_frames`` frames, phase0 is the argument of
    sum w exp(i (phase - k z_env)) -- the maximum-likelihood mean of a von
    Mises spread, weighted by `_weights`.  Its error is the circular spread
    over the effective count.  Windows with fewer than ``min_count``
    localizations are skipped; ``smoothing`` > 1 is a running mean over that
    many windows of the complex sums, so a window's estimate borrows from its
    neighbours without the jump at +-pi a mean of angles would have.  Between
    window centres phase0 is interpolated linearly (unwrapped), and held
    beyond the first and last.
    """
    phase, z_env, frame = (np.asarray(v, float) for v in (phase, z_env_nm, frame))
    w = _weights(phase_err, z_env_err_nm, phase_per_nm)
    ok = np.isfinite(phase) & np.isfinite(z_env) & np.isfinite(frame) & (w > 0)
    psi = np.exp(1j * (phase - phase_per_nm * z_env))
    first = np.nanmin(frame[ok]) if ok.any() else 0.0
    window = np.floor((frame - first) / max(int(window_frames), 1)).astype(int)
    n_windows = int(window[ok].max()) + 1 if ok.any() else 0
    sums = np.zeros(n_windows, complex)
    weights = np.zeros(n_windows)
    squares = np.zeros(n_windows)
    counts = np.zeros(n_windows, int)
    np.add.at(sums, window[ok], w[ok] * psi[ok])
    np.add.at(weights, window[ok], w[ok])
    np.add.at(squares, window[ok], w[ok] ** 2)
    np.add.at(counts, window[ok], 1)
    if smoothing > 1:
        kernel = np.ones(int(smoothing))
        sums, weights, squares, counts = (np.convolve(v, kernel, mode="same")
                                          for v in (sums, weights, squares, counts))
        counts = counts.astype(int)
    keep = counts >= min_count
    if not keep.any():
        raise ValueError(f"fewer than {min_count} localizations with a phase and a z in "
                         "every window: make the window longer")
    centres = first + (np.arange(n_windows) + 0.5) * window_frames
    mean = sums[keep] / weights[keep]
    length = np.clip(np.abs(mean), 1e-9, 1.0)
    n_eff = weights[keep] ** 2 / squares[keep]
    error = np.sqrt(-2 * np.log(length)) / np.sqrt(n_eff)
    phase0 = np.unwrap(np.angle(mean))
    return CavityPhase(centres[keep], phase0, error, counts[keep],
                       _interpolate(frame, centres[keep], phase0))


def _interpolate(frame, centres, phase0):
    """Linear between window centres and continued with the end windows'
    slope beyond them: a cavity drifting steadily keeps drifting through the
    first and last half windows (held constant, a drift of 1.5 rad per
    window left 0.18 rad of error there)."""
    at = np.interp(frame, centres, phase0)
    if len(centres) > 1:
        for side, (a, b) in ((frame < centres[0], (0, 1)), (frame > centres[-1], (-2, -1))):
            slope = (phase0[b] - phase0[a]) / (centres[b] - centres[a])
            edge = centres[b] if b == -1 else centres[a]
            value = phase0[b] if b == -1 else phase0[a]
            at[side] = value + slope * (frame[side] - edge)
    return at


@dataclass
class Unwrapped:
    """z from the phase on the most probable fringe."""

    z_nm: np.ndarray
    z_err_nm: np.ndarray
    fringe: np.ndarray          # which fringe, counted from the one through z = 0
    probability: np.ndarray     # that it is the right one
    envelope_scale: float       # how much wider z_env scatters than its error says


def unwrap(phase, phase_err, z_env_nm, z_env_err_nm, phase0, phase_per_nm: float,
           calibrate_errors: bool = True, reach: int = 2) -> Unwrapped:
    """Each localization's z on the fringe its envelope z makes most probable.

    The candidates are z_n = (phase - phase0 + 2 pi n) / k; with z_env
    Gaussian about the true z by its error, the probability of fringe n is
    proportional to exp(-(z_env - z_n)^2 / 2 s^2), normalised over the
    ``reach`` fringes either side of the nearest.  ``calibrate_errors``
    measures s first: the CRLB of z_env is the least it can scatter, and a
    model that fits the data imperfectly scatters more -- then every
    probability would be too confident.  The scatter of z_env about the
    chosen z, in units of its error, is measured robustly (1.4826 times the
    median absolute deviation, which the few wrong fringes do not move) and
    the errors are scaled up by it when it is above one.

    z's error is the phase's, |phase_err / k|: on the right fringe the
    envelope adds nothing worth having.  The probability says how likely the
    fringe is right; one minus it is the chance of a ghost a fringe away.
    """
    k = float(phase_per_nm)
    period = 2 * np.pi / abs(k)
    phase, z_env = np.asarray(phase, float), np.asarray(z_env_nm, float)
    env_err = np.asarray(z_env_err_nm, float)
    base = (phase - np.asarray(phase0, float)) / k          # the candidate of n = 0
    nearest = np.round((z_env - base) / period)
    n = nearest[:, None] + np.arange(-reach, reach + 1)[None, :]
    candidates = base[:, None] + n * period

    def choose(scale):
        s = np.maximum(env_err * scale, 1e-3)[:, None]
        logp = -0.5 * ((z_env[:, None] - candidates) / s) ** 2
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        best = np.argmax(p, axis=1)
        rows = np.arange(len(best))
        return candidates[rows, best], n[rows, best], p[rows, best]

    scale = 1.0
    z, fringe, probability = choose(scale)
    if calibrate_errors:
        chi = (z_env - z) / env_err
        chi = chi[np.isfinite(chi)]
        if len(chi) >= 20:
            scale = max(1.4826 * float(np.median(np.abs(chi - np.median(chi)))), 1.0)
            z, fringe, probability = choose(scale)
    bad = ~(np.isfinite(z) & np.isfinite(env_err))
    z[bad] = np.nan
    probability[bad] = np.nan
    fringe = np.where(bad, 0, fringe).astype(int)
    return Unwrapped(z, np.abs(np.asarray(phase_err, float) / k), fringe, probability, scale)


def refine_period(phase, phase_err, z_env_nm, z_env_err_nm, frame, phase_per_nm: float,
                  window_frames: int = 1000, span: float = 0.1, steps: int = 201) -> float:
    """The phase's slope in z that the data agree with best.

    For each trial k within +-``span`` of the calibration's, the windows'
    resultants |sum w exp(i (phase - k z_env))| are added: the right slope
    lines the phases of a window up into one direction at every z, a wrong
    one spreads them by (k - k_true) z.  The bead calibration's period is
    the stage's, and a dye a little redder than the beads, or a sample whose
    index differs from the immersion's, has another.  Needs z_env over well
    more than a fringe, and is otherwise flat; returns the calibration's
    then.

    z_env's own noise biases the slope low, as an error in x flattens a
    regression line: the resultant of a localization whose z_env errs by s
    shrinks by exp(-k^2 s^2 / 2), more for a steeper k, so the trial slopes
    are compared with each term divided by that factor -- measured on
    simulations, it took the answer from 0.991 of the true slope to within
    0.3 % of it.  Only localizations with |k| s < 1 take part, where the correction
    stays small.
    """
    phase, z_env, frame, env_err = (np.asarray(v, float)
                                    for v in (phase, z_env_nm, frame, z_env_err_nm))
    w = _weights(phase_err, env_err, phase_per_nm)
    ok = (np.isfinite(phase) & np.isfinite(z_env) & np.isfinite(frame) & (w > 0)
          & (abs(phase_per_nm) * env_err < 1))
    phase, z_env, frame, w, env_err = phase[ok], z_env[ok], frame[ok], w[ok], env_err[ok]
    if len(phase) < 50 or np.ptp(z_env) < 2 * np.pi / abs(phase_per_nm):
        return float(phase_per_nm)
    window = np.floor((frame - frame.min()) / max(int(window_frames), 1)).astype(int)
    trials = phase_per_nm * (1 + np.linspace(-span, span, steps))
    score = np.empty(steps)
    for i, k in enumerate(trials):
        sums = np.zeros(window.max() + 1, complex)
        np.add.at(sums, window, w * np.exp(1j * (phase - k * z_env) + 0.5 * (k * env_err) ** 2))
        score[i] = np.abs(sums).sum()
    best = int(np.argmax(score))
    if 0 < best < steps - 1:                      # a parabola through the peak
        a, b, c = score[best - 1:best + 2]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(trials[best] + shift * (trials[1] - trials[0]))
    return float(trials[best])


def render_frames(cal: FourPiCalibration, layout, frame, x, y, z_nm, photons, phase0,
                  n_frames: int, background: float = 10.0, seed: int = 0,
                  frame_shape=None, roisize: int = None):
    """Camera frames of a 4Pi acquisition, in photons with Poisson noise.

    Emitter k is in ``frame[k]`` at channel-0 pixel ``(x[k], y[k])`` (of the
    cut, unflipped channel), at ``z_nm[k]``, with the cavity's ``phase0[k]``.
    Each channel draws it where the layout's map puts it, and the channels
    are put on the frame in their rectangles with their flips -- what a
    camera behind the microscope would record, for the engine to take apart
    again.  Returns (n_frames, H, W) float32; the frame is the layout's
    bounding box unless ``frame_shape`` is given.
    """
    rng = np.random.default_rng(seed)
    w, h = layout.size
    C = layout.n_channels
    if frame_shape is None:
        frame_shape = (max(r[1] + r[3] for r in layout.regions),
                       max(r[0] + r[2] for r in layout.regions))
    roisize = roisize or min(cal.shape[1], cal.shape[2]) - 3
    half = roisize // 2
    frame, x, y = np.asarray(frame, int), np.asarray(x, float), np.asarray(y, float)
    z = np.asarray(z_nm, float)
    phase = cal.phase_per_nm * z + np.asarray(phase0, float)
    photons = np.broadcast_to(np.asarray(photons, float), x.shape)
    channels = np.zeros((n_frames, C, h, w), np.float32)
    zi = cal.z_nm_to_index(z)
    for c in range(C):
        at = np.c_[x, y] if c == 0 else layout.to_channel(c, np.c_[x, y])
        corner = np.rint(at).astype(int) - half
        local = at - corner
        model = spots(cal, local[:, 0], local[:, 1], zi, phase, roisize)[:, c]
        for k in range(len(x)):
            cx, cy = corner[k]
            x0, y0, x1, y1 = max(cx, 0), max(cy, 0), min(cx + roisize, w), min(cy + roisize, h)
            if x1 > x0 and y1 > y0:
                channels[frame[k], c, y0:y1, x0:x1] += photons[k] * np.clip(
                    model[k, y0 - cy:y1 - cy, x0 - cx:x1 - cx], 0, None)
    frames = np.zeros((n_frames,) + tuple(frame_shape), np.float32)
    for c, ((rx, ry, rw, rh), (fx, fy)) in enumerate(zip(layout.regions, layout.flips)):
        part = channels[:, c] + background
        if fx:
            part = part[:, :, ::-1]
        if fy:
            part = part[:, ::-1, :]
        frames[:, ry:ry + rh, rx:rx + rw] = part
    return rng.poisson(frames).astype(np.float32)
