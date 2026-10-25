"""The *elliptical Gaussian* calibration: the widths of beads through focus, for Astigmatic z.

No Qt: `psf_methods.EllipticalGaussian` is the window's entry around it, and
a script calibrates with `bead_widths` and `calibrate_widths` on the beads
`core.collect_beads` finds.

Each plane of each bead is fitted with the elliptical Gaussian the data will
be fitted with (`psf.GaussianPSF`, the same box), because a Gaussian's width
on a PSF that is not one depends on the estimator and the box: the curves
have to be measured the way the widths they are applied to are.  The bead
volumes `collect_beads` cuts are normalised; they are turned back into
photons with each bead's brightness and background and the camera's
conversion, so the fit's Poisson weights and its width errors are a real
fit's.

Beads on a coverslip are not all at one height -- a tilt of the coverslip is
tens of nanometres across the field -- so each bead's z is moved to put its
own round spot (where its widths cross) at 0 before the curves are fitted to
all of them (`align_beads`).  A bead whose widths never cross in the stack is
left out.  z in the sample is the objective's position turned round: raised
by dz, it leaves the bead dz below focus (`simulate.bead_stacks`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..params import param


@dataclass
class EllipticalCalibrationSettings:
    fit_box_px: int = param(13, label="fit box", unit="px", min=5,
                            help="the box the data will be fitted in: a Gaussian's widths depend on it")
    higher_orders: bool = param(True, label="higher orders",
                                help="fit Huang's A and B as well; off: the plain defocus curve")
    start_sigma_px: float = param(1.2, label="start sigma", unit="px", min=0.1, advanced=True,
                                  help="the width the fits start from")


@dataclass
class BeadWidths:
    """Every plane of every bead, fitted: (bead, plane) arrays, z in the sample (nm), widths and their
    errors (nm), photons; each bead's z moved to its own round spot (``z0_nm``, NaN when it has none)."""
    z_nm: np.ndarray
    sx_nm: np.ndarray
    sy_nm: np.ndarray
    sx_err_nm: np.ndarray
    sy_err_nm: np.ndarray
    photons: np.ndarray
    z0_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def aligned_z(self) -> np.ndarray:
        return self.z_nm - self.z0_nm[:, None]


def _rois(beads, camera, box: int) -> np.ndarray:
    """(bead, plane, box, box) in photons: each bead's volume cut to the fit box around its centre."""
    volumes = np.asarray(beads.volumes, dtype=np.float64)
    size = volumes.shape[-1]
    if box > size:
        raise ValueError(f"the fit box ({box} px) is larger than the bead box ({size} px)")
    start = (size - box) // 2
    cut = volumes[:, :, start:start + box, start:start + box]
    brightness = np.array([r["brightness_adu"] for r in beads.records], dtype=np.float64)
    background = np.array([r["background_adu"] for r in beads.records], dtype=np.float64)
    adu = cut * brightness[:, None, None, None] + background[:, None, None, None]
    if camera is not None and camera.conversion is not None and camera.offset is not None:
        return (adu - camera.offset) * camera.adu_to_photons
    # no camera: ADU above the background's own level, scaled as if one ADU were one photon -- the
    # widths barely change, their errors are then only relative
    return adu


def plane_z(beads) -> np.ndarray:
    """(bead, plane): each plane's z in the sample, the objective's position turned round."""
    out = []
    for record in beads.records:
        z = np.asarray(beads.sources[record["stack"]]["z_nm"], dtype=np.float64)
        start = record.get("start_plane", 0)
        out.append(-z[start:start + beads.volumes.shape[1]])
    return np.array(out)


def bead_widths(beads, camera, pixel_nm, settings: Optional[EllipticalCalibrationSettings] = None,
                progress=None) -> BeadWidths:
    """Every plane of every bead fitted with the elliptical Gaussian; ``pixel_nm`` (x, y)."""
    from ..psf import GaussianPSF
    settings = settings or EllipticalCalibrationSettings()
    report = progress or (lambda text: None)
    rois = _rois(beads, camera, int(settings.fit_box_px))
    n, planes = rois.shape[:2]
    report(f"fitting {n * planes} planes of {n} beads")
    psf = GaussianPSF(sigma=float(settings.start_sigma_px), elliptical=True)
    flat = np.ascontiguousarray(rois.reshape(n * planes, *rois.shape[2:]), dtype=np.float32)
    out = psf.unpack(psf.fit(np.maximum(flat, 0)))
    px, py = float(pixel_nm[0]), float(pixel_nm[1])

    def shaped(name, scale=1.0):
        return np.asarray(out[name], dtype=np.float64).reshape(n, planes) * scale
    widths = BeadWidths(plane_z(beads), shaped("sigma_x_pix", px), shaped("sigma_y_pix", py),
                        shaped("sigma_x_err_pix", px), shaped("sigma_y_err_pix", py), shaped("photons"))
    widths.z0_nm = align_beads(widths)
    return widths


def round_spot(z, sx, sy) -> float:
    """Where one bead's widths cross, nearest the middle of its stack: linear between the planes
    that bracket it.  NaN when they never cross."""
    diff = np.asarray(sx, float) - np.asarray(sy, float)
    ok = np.isfinite(diff)
    z, diff = np.asarray(z, float)[ok], diff[ok]
    if len(z) < 2:
        return float("nan")
    order = np.argsort(z)
    z, diff = z[order], diff[order]
    flips = np.flatnonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)
    if not len(flips):
        return float("nan")
    i = flips[np.argmin(np.abs(z[flips] - np.median(z)))]
    return float(z[i] + (z[i + 1] - z[i]) * diff[i] / (diff[i] - diff[i + 1]))


def align_beads(widths: BeadWidths) -> np.ndarray:
    """Each bead's round spot, its z0 (NaN for a bead whose widths never cross)."""
    return np.array([round_spot(widths.z_nm[b], widths.sx_nm[b], widths.sy_nm[b])
                     for b in range(len(widths.z_nm))])


def calibrate_widths(widths: BeadWidths, higher_orders: bool = True, excluded=(), source: str = ""):
    """The curves from the beads not ``excluded`` and with a round spot: `astigmatic_z.fit_calibration`."""
    from ..astigmatic_z import fit_calibration
    use = np.isfinite(widths.z0_nm)
    use[list(excluded)] = False
    if not use.any():
        raise ValueError("no bead's widths cross in its stack: the beads are not astigmatic, or the stacks "
                         "do not reach the round spot between the foci")
    z = widths.aligned_z[use]
    return fit_calibration(z, widths.sx_nm[use], widths.sy_nm[use], widths.sx_err_nm[use],
                           widths.sy_err_nm[use], higher_orders=higher_orders, source=source,
                           beads=int(use.sum()))


def bead_residuals(widths: BeadWidths, cal) -> np.ndarray:
    """Each bead's rms distance of its widths from the curves, nm (NaN for an unaligned one)."""
    sx, sy = cal.widths(widths.aligned_z)
    inside = (widths.aligned_z >= cal.z_range[0]) & (widths.aligned_z <= cal.z_range[1])
    sq = np.where(inside, (widths.sx_nm - sx) ** 2 + (widths.sy_nm - sy) ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(sq, axis=1) / 2)


def save_path(paths) -> Path:
    """Beside the stacks, under the bead calibration's name with ``_astigz``, never over a file."""
    from .core import calibration_save_defaults
    folder, name = calibration_save_defaults(paths, dual=False)
    path, n = Path(folder) / f"{name}_astigz.h5", 2
    while path.exists():
        path, n = Path(folder) / f"{name}_astigz_{n}.h5", n + 1
    return path


def draw_widths(figure, widths: BeadWidths, cal, selected=(), excluded=()):
    """Every bead's widths in z over the fitted curves, the selected beads drawn over the rest."""
    ax = figure.subplots()
    use = np.isfinite(widths.z0_nm)
    use[list(excluded)] = False
    z = widths.aligned_z
    ax.plot(z[use].T, widths.sx_nm[use].T, ".", color="C0", ms=2, alpha=0.4)
    ax.plot(z[use].T, widths.sy_nm[use].T, ".", color="C3", ms=2, alpha=0.4)
    for b in selected:
        if 0 <= b < len(z) and np.isfinite(widths.z0_nm[b]):
            ax.plot(z[b], widths.sx_nm[b], "o", color="C0", mfc="none", ms=5)
            ax.plot(z[b], widths.sy_nm[b], "o", color="C3", mfc="none", ms=5)
    grid = np.linspace(*cal.z_range, 400)
    cx, cy = cal.widths(grid)
    ax.plot(grid, cx, color="C0", lw=2, label="x")
    ax.plot(grid, cy, color="C3", lw=2, label="y")
    ax.set_xlabel("z in the sample (nm), 0 where the spot is round")
    ax.set_ylabel("width (nm)")
    top = np.nanpercentile(np.r_[widths.sx_nm[use].ravel(), widths.sy_nm[use].ravel()], 99) if use.any() else 1
    ax.set_ylim(0, 1.15 * top)
    ax.legend(frameon=False, loc="upper center")
    return f"{int(use.sum())} beads; rms residual {cal.residual_nm[0]:.1f} nm in x, {cal.residual_nm[1]:.1f} nm in y"


def draw_z_back(figure, widths: BeadWidths, cal, selected=(), excluded=()):
    """The beads' own widths turned back into z: against the z they were taken at, and the error."""
    from ..astigmatic_z import z_from_widths
    left, right = figure.subplots(1, 2)
    use = np.isfinite(widths.z0_nm)
    use[list(excluded)] = False
    z = widths.aligned_z[use]
    found = z_from_widths(widths.sx_nm[use], widths.sy_nm[use], cal, widths.sx_err_nm[use],
                          widths.sy_err_nm[use])
    inside = (z >= cal.z_range[0]) & (z <= cal.z_range[1])
    left.plot(z[inside], found.z_nm.reshape(z.shape)[inside] - z[inside], ".", ms=2, color="0.3")
    left.axhline(0, color="C1", lw=1)
    left.set_xlabel("z of the plane (nm)")
    left.set_ylabel("z from its widths - z (nm)")
    err = found.z_err_nm.reshape(z.shape)
    order = np.argsort(z[inside])
    right.plot(z[inside][order], err[inside][order], ".", ms=2, color="0.3")
    right.set_yscale("log")
    right.set_xlabel("z of the plane (nm)")
    right.set_ylabel("z error at the beads' photons (nm)")
    spread = np.nanstd(found.z_nm.reshape(z.shape)[inside] - z[inside])
    return f"z from the beads' own widths: rms {spread:.1f} nm about the stage's z"
