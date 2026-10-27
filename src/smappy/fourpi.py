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


def simulate_rois(cal: FourPiCalibration, n: int, roisize: int = 13,
                  photons: float = 3000.0, background: float = 10.0,
                  z_nm=(-400.0, 400.0), phase0: float = None, xy_spread: float = 0.5,
                  seed: int = 0):
    """Poisson ROIs of single emitters in every channel, with their truth.

    Returns ``rois`` (n, C, roisize, roisize) and a dict of the truth: x, y
    (ROI pixels), z_nm, the phase the fitter should find, photons and the
    background per channel.  Each emitter's phase is tied to its z by
    `phase_per_nm`, plus ``phase0`` (the cavity's phase; random per emitter
    when None, which is what a test of the fitter alone wants).
    """
    rng = np.random.default_rng(seed)
    C = cal.n_channels
    centre = (roisize - 1) / 2
    x = centre + rng.uniform(-xy_spread, xy_spread, n)
    y = centre + rng.uniform(-xy_spread, xy_spread, n)
    z = rng.uniform(*z_nm, n) if np.ndim(z_nm) == 1 and len(z_nm) == 2 else \
        np.broadcast_to(np.asarray(z_nm, float), (n,)).copy()
    offset = rng.uniform(-np.pi, np.pi, n) if phase0 is None else np.full(n, float(phase0))
    phase = cal.phase_per_nm * z + offset
    zi = cal.z_nm_to_index(z)
    rois = np.empty((n, C, roisize, roisize), np.float32)
    for i in range(n):
        for c in range(C):
            mu = photons * np.clip(evaluate(cal, c, x[i], y[i], zi[i], phase[i], roisize),
                                   0, None) + background
            rois[i, c] = rng.poisson(mu)
    truth = {"x": x, "y": y, "z_nm": z, "phase": np.angle(np.exp(1j * phase)),
             "photons": np.full(n, float(photons)), "background": np.full(n, float(background))}
    return rois, truth


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
    slice, so it falls with z_nm.  The splines are made again from those
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
    cal = from_volumes(I, 2 * np.real(Ac), -2 * np.imag(Ac), dz=dz, z0=(nz - 1) / 2,
                       period_nm=period_nm, phase_per_nm=-2 * np.pi / period_nm,
                       source=path,
                       pixelsize_nm=pixelsize,
                       parameters={"method": "uiPSF v1 4Pi model", "uipsf_v1": extra,
                                   "transposed_from_v1": bool(transpose),
                                   "uipsf_v1_params": params})
    if T is not None:
        cal.parameters["uipsf_v1_T"] = np.asarray(T).tolist()
    return cal
