"""The shift between two layers, by cross-correlation: *Register layers*.

Wave 5, step 5b (`docs/smap_porting/wave5_design.md`), SMAP's RegisterLocs
cross-correlation and XYZdisplacement: how far a *target* layer lies from a
*reference* layer, in x and y and optionally z.  Each layer is a picture on
one common grid -- localizations as they show, binned at *bin*; an image
layer looked up on the same grid (`ImageData.resample`, its placement
honoured) -- and the shift is where their cross-correlation peaks.

**No third correlation.**  The correlation is `correlation._xcorr`, the FFT
correlation of Spatial statistics' correlation image (step 17c), and the
peak is RCC's (`rcc._peak`: a 5x5 box smoothing to pick the pixel, a
quadratic surface over the pixels around it for the sub-pixel position).
z, when asked for, is RCC's axial pass (`rcc._axial_shifts`): z histograms
in 200 nm tiles of the field, correlated tile by tile and summed -- SMAP's
XYZdisplacement correlated slices of the field the same way -- after the
lateral shift has been taken out of the target.

**The pictures have their mean subtracted** before they are correlated.  A
widefield image sits on a pedestal of background, and correlated as it is
that pedestal's overlap with the other picture -- largest at zero shift --
outweighs the structure; RCC, correlating sparse histograms of one
acquisition with itself, has no pedestal to remove.

**Only a shift**, no rotation or scale (the design: later).  The result is
the shift of the target against the reference; what is applied, by
*Transform*, is its opposite, a `Transformation` that moves the target onto
the reference (`Registration.transformation`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .correlation import MAX_PIXELS, _xcorr
from .rcc import RCCSettings, _axial_shifts, _peak
from .render import FieldOfView

#: the tiles of the axial pass, RCC's (see `RCCSettings.tile_nm`)
Z_TILE_NM = 200.0
#: the half-width of the peak fit, in bins: RCC's default
FIT_HALF_WIDTH = 3
#: correlations again with the shift taken out, at most (`register`)
REFINE = 3


@dataclass
class Grid:
    """The common grid both layers are pictured on: the outer edge of bin
    (0, 0) at (x0, y0), square bins."""
    x0: float
    y0: float
    bin_nm: float
    nx: int
    ny: int

    @property
    def fov(self) -> FieldOfView:
        return FieldOfView(self.x0, self.y0, self.bin_nm, self.nx, self.ny)


def common_grid(boxes: Sequence[Tuple[float, float, float, float]], bin_nm: float,
                lags: int = 0) -> Grid:
    """A grid over every box ``(x0, y0, x1, y1)``; refused, naming a bin that
    would do, when it and its padding for ``lags`` would exceed
    `correlation.MAX_PIXELS`."""
    b = np.asarray(boxes, float)
    x0, y0 = float(b[:, 0].min()), float(b[:, 1].min())
    x1, y1 = float(b[:, 2].max()), float(b[:, 3].max())
    nx = max(1, int(np.ceil((x1 - x0) / bin_nm)) + 1)
    ny = max(1, int(np.ceil((y1 - y0) / bin_nm)) + 1)
    if (nx + lags + 1) * (ny + lags + 1) > MAX_PIXELS:
        need = bin_nm * np.sqrt((nx + lags + 1) * (ny + lags + 1) / MAX_PIXELS)
        raise ValueError(f"{nx} x {ny} bins of {bin_nm:g} nm are too many to "
                         f"correlate: use a bin of at least {np.ceil(need):g} nm, "
                         f"or a smaller max shift")
    return Grid(x0, y0, float(bin_nm), nx, ny)


def histogram(x, y, grid: Grid) -> np.ndarray:
    """Localizations counted in the grid's bins, ``[row (y), column (x)]``."""
    ix = np.floor((np.asarray(x, float) - grid.x0) / grid.bin_nm).astype(np.int64)
    iy = np.floor((np.asarray(y, float) - grid.y0) / grid.bin_nm).astype(np.int64)
    keep = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    out = np.zeros((grid.ny, grid.nx))
    np.add.at(out, (iy[keep], ix[keep]), 1.0)
    return out


def image_on(image, grid: Grid, frame: Optional[int] = None) -> np.ndarray:
    """An `ImageData` looked up on the grid (nearest pixel, its placement
    honoured); zero outside it."""
    return np.asarray(image.resample(grid.fov, frame).weight, float)


def lateral_shift(reference: np.ndarray, target: np.ndarray, bin_nm: float,
                  lags: int) -> Tuple[float, float, np.ndarray]:
    """``(dx, dy, correlation)``: how far ``target`` lies from ``reference``
    (two pictures on one grid, ``[row, column]``), in nm, searched within
    ``lags`` bins.  The correlation has the zero shift in its middle."""
    a = reference - reference.mean()
    b = target - target.mean()
    if not np.any(a) or not np.any(b):
        raise ValueError("a layer is empty on the grid: nothing to correlate")
    lags = int(max(2, min(lags, max(reference.shape) - 1)))
    correlation = _xcorr(a, b, lags)
    row, col = _peak(correlation, lags, FIT_HALF_WIDTH)
    return col * bin_nm, row * bin_nm, correlation


def axial_shift(reference_xyz: Sequence[np.ndarray], target_xyz: Sequence[np.ndarray],
                z_bin_nm: float, max_shift_nm: float) -> float:
    """How far the target lies above the reference in z, in nm: RCC's tiled
    z histograms (`rcc._axial_shifts`) of the two, the reference as one time
    window and the target as the other.  The lateral shift must have been
    taken out of the target first."""
    x = np.r_[reference_xyz[0], target_xyz[0]].astype(float)
    y = np.r_[reference_xyz[1], target_xyz[1]].astype(float)
    z = np.r_[reference_xyz[2], target_xyz[2]].astype(float)
    if not np.isfinite(z).all():
        keep = np.isfinite(z)
    else:
        keep = slice(None)
    window = np.r_[np.zeros(len(reference_xyz[0]), np.int64),
                   np.ones(len(target_xyz[0]), np.int64)]
    # two distinct layers: the zero shift is a measurement like any other,
    # not RCC's spike of a molecule on in two windows of one acquisition
    settings = RCCSettings(z_pixelsize_nm=z_bin_nm, tile_nm=Z_TILE_NM,
                           axial_max_drift_nm=max_shift_nm, exclude_zero_lag=False,
                           fit_window=FIT_HALF_WIDTH)
    shifts = _axial_shifts(x[keep], y[keep], z[keep], window[keep],
                           np.array([0, 1]), settings, False)
    return float(shifts[1, 0])


@dataclass
class Registration:
    """The measured shift of the target against the reference, and what it
    was measured from, to draw."""
    dx_nm: float
    dy_nm: float
    dz_nm: Optional[float]
    correlation: np.ndarray            # lateral, the zero shift in the middle
    bin_nm: float
    reference: np.ndarray              # the two pictures, for the figure
    target: np.ndarray
    grid: Grid

    @property
    def shift_nm(self) -> Tuple[float, float, float]:
        return (self.dx_nm, self.dy_nm, self.dz_nm or 0.0)

    def transformation(self, name: str = "", origin: Optional[dict] = None):
        """The `Transformation` that moves the target onto the reference:
        the shift's opposite."""
        from .transformation import typed
        t = typed((-self.dx_nm, -self.dy_nm, -(self.dz_nm or 0.0)))
        t.origin = {"register layers": dict(origin or {}),
                    "shift_nm": [float(v) for v in self.shift_nm]}
        if name:
            t.description = f"{name}: {t}"
        return t

    def __str__(self) -> str:
        z = "" if self.dz_nm is None else f", dz {self.dz_nm:.1f}"
        return f"target shifted by dx {self.dx_nm:.1f}, dy {self.dy_nm:.1f}{z} nm"


def register(reference: dict, target: dict, bin_nm: float = 20.0,
             max_shift_nm: Optional[float] = 2500.0, use_z: bool = False,
             z_bin_nm: float = 10.0, max_z_shift_nm: float = 300.0) -> Registration:
    """The shift of ``target`` against ``reference``.

    Each is ``{"x", "y"}`` (and ``"z"``) arrays of localizations in nm, or
    ``{"image": ImageData, "frame": plane}`` (the plane: None, the image's
    own).  ``max_shift_nm`` None searches up to half the
    grid.  z needs both to be localizations with z.
    """
    def box(layer):
        if "image" in layer:
            return layer["image"].bounds
        x, y = np.asarray(layer["x"], float), np.asarray(layer["y"], float)
        if not len(x):
            raise ValueError("a layer has no localizations to register")
        return (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

    boxes = [box(reference), box(target)]
    lags = None if max_shift_nm is None else int(np.ceil(max_shift_nm / bin_nm))
    grid = common_grid(boxes, bin_nm, lags or 0)
    if lags is None:
        lags = max(grid.nx, grid.ny) // 2

    def picture(layer):
        if "image" in layer:
            return image_on(layer["image"], grid, layer.get("frame"))
        return histogram(layer["x"], layer["y"], grid)

    a, b = picture(reference), picture(target)
    dx, dy, correlation = lateral_shift(a, b, bin_nm, lags)
    # Refined: the shift taken out of whichever side is localizations, and
    # correlated again (`REFINE`).  The quadratic peak fit is biased towards
    # whole bins by up to a fifth of one (4 nm at 20 nm bins, measured on an
    # exact copy); at a residual near zero it is gone -- within 0.9 nm on the
    # same copies, at three shifts.
    moving = target if "image" not in target else (
        reference if "image" not in reference else None)
    sign = -1.0 if moving is target else 1.0
    for _ in range(REFINE if moving is not None else 0):
        moved = histogram(np.asarray(moving["x"], float) + sign * dx,
                          np.asarray(moving["y"], float) + sign * dy, grid)
        rx, ry, _ = lateral_shift(*((a, moved) if moving is target else (moved, b)),
                                  bin_nm, FIT_HALF_WIDTH + 3)
        dx, dy = dx + rx, dy + ry
        if np.hypot(rx, ry) < 0.01 * bin_nm:
            break
    dz = None
    if use_z:
        if "image" in reference or "image" in target or "z" not in reference \
                or "z" not in target:
            raise ValueError("a shift in z needs two layers of localizations "
                             "with z_nm: an image has no z")
        dz = axial_shift((reference["x"], reference["y"], reference["z"]),
                         (np.asarray(target["x"], float) - dx,
                          np.asarray(target["y"], float) - dy, target["z"]),
                         z_bin_nm, max_z_shift_nm)
    return Registration(dx, dy, dz, correlation, bin_nm, a, b, grid)
