"""Rendering localizations into a superresolution image.

Two stages, deliberately separate:

* **accumulation** produces float planes -- a weight plane (localizations, or
  photons, per superresolution pixel) and, when localizations carry individual
  colours, three colour planes.  This is the expensive part and depends on the
  data and the field of view.
* **display** turns those planes into RGB: normalise, gamma, look-up table.
  This is cheap and depends only on the settings, so a viewer can change the
  contrast or the colour map without re-rendering.

SMAP splits the same way (``renderSMAP`` / ``drawerSMAP``).

Colour: an intensity image is recoloured at display time, one lookup per output
pixel.  Colouring by a field (z, frame, ...) cannot work that way, because two
localizations of different colour in one pixel have to mix, so each one
contributes its own colour during accumulation.  We keep **four** planes then --
the three colour-weighted ones plus the plain weight: ``colour_sum /
weight_sum`` gives the hue, ``weight_sum`` the brightness, so saturation and
gamma act on brightness only and leave the hue alone.  (SMAP displays
``colour_sum`` directly, which per-channel clipping pulls towards white in
dense regions; that composite went with the matplotlib viewer in 2026-10.)

What the two axes *are* is `RenderAxes`: normally the table's positions, but
any column against any other, each divided by a scale of its own.  The division
is the whole mechanism -- the grid stays square in those units, so nothing
downstream has to know -- and it is what SMAP's ``VersatileRenderer`` does.

Coordinates are in whatever unit the localizations are in (nm, normally).  A
pixel of the rendered image covers ``[x0 + j*p, x0 + (j+1)*p)``, so a position
maps to pixel ``floor((x - x0) / p)``.  SMAP instead rounds and shifts the range
by half a pixel beforehand; the result is the same grid, expressed once instead
of twice.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Union

import numpy as np

from . import lut as luts
from .locs import Localizations


def finite_range(values) -> Tuple[float, float]:
    """What the renderer colours over when no range is set: min to max of
    the values there are -- NaN left out, (0, 1) when there are none."""
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

try:
    from . import _render
except ImportError:  # pure-Python fallback if the extension is not built
    _render = None


@dataclass(frozen=True)
class FieldOfView:
    """The rendered grid: origin, pixel size and shape.

    ``x0``/``y0`` are the coordinates of the *outer edge* of pixel (0, 0).
    Pixels are square, which keeps the image undistorted under any zoom.
    """

    x0: float
    y0: float
    pixelsize: float
    nx: int
    ny: int

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.pixelsize

    @property
    def y1(self) -> float:
        return self.y0 + self.ny * self.pixelsize

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """``(left, right, bottom, top)``, ready for ``imshow(extent=...)``."""
        return (self.x0, self.x1, self.y1, self.y0)

    @classmethod
    def from_range(cls, xrange, yrange, pixelsize: float) -> "FieldOfView":
        """Cover the given coordinate ranges at a fixed pixel size."""
        nx = max(int(round((xrange[1] - xrange[0]) / pixelsize)), 1)
        ny = max(int(round((yrange[1] - yrange[0]) / pixelsize)), 1)
        return cls(float(xrange[0]), float(yrange[0]), float(pixelsize), nx, ny)

    @classmethod
    def fit(cls, xrange, yrange, nx: int, ny: int) -> "FieldOfView":
        """Fit the ranges into a canvas of ``nx`` x ``ny`` pixels.

        The pixel size follows from the canvas, which is what a viewer wants:
        the cost of a render then depends on the window, not on the zoom.  The
        ranges are centred, and the larger one decides the pixel size so
        nothing is cut off.
        """
        pixelsize = max((xrange[1] - xrange[0]) / nx, (yrange[1] - yrange[0]) / ny)
        cx = 0.5 * (xrange[0] + xrange[1])
        cy = 0.5 * (yrange[0] + yrange[1])
        return cls(cx - 0.5 * nx * pixelsize, cy - 0.5 * ny * pixelsize,
                   float(pixelsize), int(nx), int(ny))

    @classmethod
    def around(cls, x, y, pixelsize: float, margin: float = 0.0) -> "FieldOfView":
        """The smallest field of view containing all of ``x``, ``y``."""
        return cls.from_range((np.min(x) - margin, np.max(x) + margin),
                              (np.min(y) - margin, np.max(y) + margin), pixelsize)

    def to_pixels(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Data coordinates -> fractional pixel coordinates."""
        inv = 1.0 / self.pixelsize
        return ((np.asarray(x, dtype=np.float64) - self.x0) * inv,
                (np.asarray(y, dtype=np.float64) - self.y0) * inv)

    def zoomed(self, factor: float, cx: Optional[float] = None,
               cy: Optional[float] = None) -> "FieldOfView":
        """Same canvas, pixel size scaled by ``factor`` about ``(cx, cy)``.

        ``factor < 1`` zooms in.  The centre defaults to the centre of the
        current view; passing the cursor position is what makes scroll-zoom
        feel right.
        """
        cx = 0.5 * (self.x0 + self.x1) if cx is None else cx
        cy = 0.5 * (self.y0 + self.y1) if cy is None else cy
        p = self.pixelsize * factor
        return FieldOfView(cx - (cx - self.x0) * factor,
                           cy - (cy - self.y0) * factor, p, self.nx, self.ny)

    def moved(self, dx: float, dy: float) -> "FieldOfView":
        return FieldOfView(self.x0 + dx, self.y0 + dy, self.pixelsize,
                           self.nx, self.ny)


@dataclass
class RenderedImage:
    """Accumulated planes plus the geometry they were rendered on."""

    fov: FieldOfView
    weight: np.ndarray                      # (ny, nx)
    color: Optional[np.ndarray] = None      # (ny, nx, 3), colour-weighted
    n_locs: int = 0
    # the area a pixel's value is counted over, in squared data units: what
    # makes an absolute maximum per um^2 (`DisplaySettings.maximum`) the same
    # brightness at every zoom.  None is the pixel's own area -- a histogram
    # or a Gaussian render counts localizations per pixel, so a pixel twice
    # as wide holds four times as many.  `PER_UM2` is a value that does not
    # grow with the pixel: a Voronoi density (already per um^2), a track's
    # line (one wherever it passes) or a camera image.
    count_area: Optional[float] = None

    @property
    def is_colored(self) -> bool:
        return self.color is not None


def render_histogram(x, y, fov: FieldOfView, weights=None, colors=None
                     ) -> RenderedImage:
    """Count localizations (or their weights) per pixel.

    ``colors`` is an ``(n, 3)`` array of per-localization RGB; pass it to get a
    field-coloured image.
    """
    px, py = fov.to_pixels(x, y)
    col = np.floor(px).astype(np.int64)
    row = np.floor(py).astype(np.int64)
    inside = (col >= 0) & (col < fov.nx) & (row >= 0) & (row < fov.ny)

    index = (row[inside] * fov.nx + col[inside])
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[inside]
    n_pixels = fov.nx * fov.ny

    weight = np.bincount(index, weights=w, minlength=n_pixels)
    weight = weight.reshape(fov.shape).astype(np.float32)

    color = None
    if colors is not None:
        c = np.asarray(colors, dtype=np.float64)[inside]
        if w is not None:
            c = c * w[:, None]
        color = np.stack(
            [np.bincount(index, weights=c[:, k], minlength=n_pixels)
             for k in range(3)], axis=-1).reshape(fov.ny, fov.nx, 3).astype(np.float32)

    return RenderedImage(fov, weight, color, int(inside.sum()))


# SMAP's dynamic contrast, and its parameterisation: a single number p that
# saturates 10^-p of the pixels.  The value distribution of a superresolution
# image is extremely heavy-tailed -- a fiducial bead outweighs the structure by
# three orders of magnitude -- so a fixed maximum is useless and a quantile is
# the only scale that transfers between datasets.  p is the natural handle
# because the useful range spans decades: 2 is bright, 4 is dark.
DEFAULT_CONTRAST = 3.0


#: squared data units in a um^2 (nm axes): the area an absolute maximum is per
PER_UM2 = 1e6


def saturated_fraction(contrast: float) -> float:
    """The fraction of pixels driven to full scale: 10^-p."""
    return 10.0 ** -float(contrast)


def normalize(image: np.ndarray, imax: Optional[float] = None,
              contrast: float = DEFAULT_CONTRAST) -> Tuple[np.ndarray, float]:
    """Scale to [0, 1] and clip.  Returns the image and the value used as 1.

    ``imax`` overrides the dynamic contrast with an absolute value.  As in SMAP,
    a quantile that lands on zero -- a very sparse image -- falls back to the
    maximum.
    """
    if imax is None:
        imax = float(np.quantile(image, 1.0 - saturated_fraction(contrast)))
        if imax <= 0:
            imax = float(image.max())
    if imax <= 0:
        return np.zeros_like(image), 0.0
    return np.clip(image / imax, 0.0, 1.0), imax


def absolute_imax(rendered: RenderedImage, maximum: float) -> float:
    """The pixel value an absolute maximum per um^2 comes to on this render.

    SMAP's absolute Imax is in localizations per pixel, so it is tied to one
    zoom: a pixel twice as wide holds four times the localizations and the
    picture saturates.  Given as a density instead, it is scaled here by the
    area a pixel's value counts over (`RenderedImage.count_area`), and a
    structure keeps its brightness as one zooms -- save where the rendering
    width itself changes with the pixel (the precision mode's floor of 0.7
    pixels).
    """
    area = rendered.count_area
    if area is None:
        area = rendered.fov.pixelsize ** 2
    return float(maximum) * area / PER_UM2


def to_rgb(rendered: RenderedImage, lut: luts.LUT = "hot", invert=False,
           imax: Optional[float] = None, contrast: float = DEFAULT_CONTRAST,
           gamma: float = 1.0, white_background: bool = False) -> np.ndarray:
    """Turn accumulated planes into a float32 ``(ny, nx, 3)`` RGB image in [0, 1].

    For an intensity image ``lut`` recolours the normalised weight plane.  For a
    field-coloured one the LUT was already applied per localization (see the
    module docstring).

    ``white_background`` turns the finished image over (`smappy.lut.on_white`):
    the empty ground goes white and the colours keep their hue.  It is done to
    the image rather than to the LUT so that a field-coloured render, where the
    LUT is baked into the accumulation, is turned over as well -- and, where
    several layers are summed, so that it can be done once to the sum instead
    of once per layer, which would wash the picture out.
    """
    rgb = _to_rgb(rendered, lut, invert, imax, contrast, gamma)
    return luts.on_white(rgb) if white_background else rgb


def _to_rgb(rendered: RenderedImage, lut, invert, imax, contrast, gamma
            ) -> np.ndarray:
    if rendered.color is None:
        norm, _ = normalize(rendered.weight, imax, contrast)
        if gamma != 1.0:
            norm = norm ** gamma
        table = luts.get(lut, invert)
        index = np.clip((norm * len(table)).astype(np.int32), 0, len(table) - 1)
        return table[index]

    brightness, _ = normalize(rendered.weight, imax, contrast)
    if gamma != 1.0:
        brightness = brightness ** gamma
    with np.errstate(invalid="ignore", divide="ignore"):
        hue = rendered.color / rendered.weight[..., None]
    hue[~np.isfinite(hue)] = 0.0
    return (hue * brightness[..., None]).astype(np.float32)


# --------------------------------------------------------------- sigma policy
@dataclass(frozen=True)
class SigmaSettings:
    """How a localization precision becomes a rendering sigma.

    SMAP's policy, because it is what makes precision-weighted images
    readable rather than a field of single bright pixels: blow the precision up
    a little (``factor``), never let a blob fall below a floor -- an absolute
    one in data units and one tied to the rendered pixel size -- and cap the
    outliers, since a handful of localizations with an absurd precision would
    otherwise dominate the render time.  SMAP caps at 400 nm; the cap here is
    relative, a multiple of the median, so that it means the same on a table
    in pixels or on custom axes.
    """

    factor: float = 1.0
    min_sigma: float = 0.0          # data units, SMAP's mingaussnm
    min_sigma_pixels: float = 0.7   # rendered pixels, SMAP's mingausspix
    max_factor: float = 10.0        # cap at this multiple of the median

    def apply(self, precision, pixelsize: float) -> np.ndarray:
        """Localization precision -> rendering sigma, in data units."""
        sigma = np.asarray(precision, dtype=np.float32) * np.float32(self.factor)
        finite = sigma[np.isfinite(sigma)]
        cap = (self.max_factor * np.median(finite)) if finite.size else np.inf
        floor = max(self.min_sigma, self.min_sigma_pixels * pixelsize)
        return np.clip(np.nan_to_num(sigma, nan=floor), floor, cap).astype(np.float32)


# SMAP's roiks: the kernel is cut off at this many sigma.
ROI_SIGMA = 2.7


def _sigma_array(value, n: int, what: str) -> np.ndarray:
    a = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
    if a.size not in (1, n):
        raise ValueError(f"{what} must be a scalar or one value per localization")
    return a


def render(x, y, fov: FieldOfView, sigma=None, sigma_y=None, weights=None,
           colors=None, roi_sigma: float = ROI_SIGMA, n_threads: int = 0,
           use_extension: bool = True) -> RenderedImage:
    """Accumulate localizations into ``fov``.

    ``sigma`` selects the mode and is in the units of the grid: ``None``
    renders a histogram, a scalar renders every localization with the same
    Gaussian, and an array renders each with its own -- normally a
    localization precision put through :class:`SigmaSettings`.  There is one
    kernel behind all three; see ``csrc/render.hpp``.

    ``sigma_y`` gives the vertical direction a width of its own, for a grid
    whose two axes are not the same quantity; ``None`` is the same width in
    both.  Either may be zero, which bins along that axis instead of blurring.

    ``weights`` is the intensity per localization (one value broadcasts), and
    ``colors`` an ``(n, 3)`` array for field colouring.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be matching 1-D arrays")

    gaussian = sigma is not None or sigma_y is not None
    sigma_x = sigma_y_ = np.zeros(1, np.float32)
    if gaussian:
        sigma_x = _sigma_array(0.0 if sigma is None else sigma, x.size, "sigma")
        sigma_y_ = sigma_x if sigma_y is None else \
            _sigma_array(sigma_y, x.size, "sigma_y")
        if sigma_x.size != sigma_y_.size:
            # the kernel reads both through one stride, so a per-localization
            # width on one axis and a constant on the other have to meet
            n = max(sigma_x.size, sigma_y_.size)
            sigma_x = np.ascontiguousarray(np.broadcast_to(sigma_x, (n,)), np.float32)
            sigma_y_ = np.ascontiguousarray(np.broadcast_to(sigma_y_, (n,)), np.float32)
        if sigma_x.size == 1 and sigma_y_.size == 1 and sigma_x[0] <= 0 and sigma_y_[0] <= 0:
            # no width either way is a histogram, and the histogram is both
            # the cheaper way to say so and the one that skips the kernel
            gaussian = False

    w = np.ones(1, np.float32) if weights is None else \
        np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
    c = np.zeros((0, 3), np.float32) if colors is None else \
        np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 3)
    if c.size and c.shape[0] != x.size:
        raise ValueError("colours must have one row per localization")

    if use_extension and _render is not None:
        weight, color, n = _render.render(
            x, y, sigma_x, sigma_y_, w, c, fov.x0, fov.y0, fov.pixelsize,
            fov.nx, fov.ny, gaussian, roi_sigma, 512, n_threads)
        return RenderedImage(fov, weight, color, int(n))

    return _render_numpy(x, y, fov, sigma_x if gaussian else None,
                         sigma_y_ if gaussian else None, w, c, roi_sigma)


def _render_numpy(x, y, fov, sigma, sigma_y, weights, colors, roi_sigma) -> RenderedImage:
    """Reference implementation of :func:`render`; the extension must match it."""
    from scipy.special import erf

    if sigma is None:
        w = None if weights.size == 1 else weights
        img = render_histogram(x, y, fov, weights=w,
                               colors=colors if colors.size else None)
        if weights.size == 1:
            img.weight *= weights[0]
            if img.color is not None:
                img.color *= weights[0]
        return img

    weight = np.zeros(fov.shape, np.float64)
    color = np.zeros((fov.ny, fov.nx, 3), np.float64) if colors.size else None
    px, py = fov.to_pixels(x, y)
    n_locs = 0

    def kernel(center, position, s):
        """Pixel-integrated Gaussian over the ROI, and its sum.

        A sigma below a hundredth of a pixel is one pixel -- the integral has
        collapsed onto it -- which is how the extension reads it too, and is
        what an axis with no width at all (plain binning) comes down to.
        """
        if s <= 0.01:
            return np.array([center]), np.ones(1), 1.0
        d = int(roi_sigma * s + 1.0)
        edges = (np.arange(-d, d + 2) - (position - center)) / (np.sqrt(2) * s)
        k = 0.5 * np.diff(erf(edges))
        return np.arange(center - d, center + d + 1), k, k.sum()

    for i in range(x.size):
        s = (sigma[0] if sigma.size == 1 else sigma[i]) / fov.pixelsize
        sy_i = (sigma_y[0] if sigma_y.size == 1 else sigma_y[i]) / fov.pixelsize
        cx, cy = int(np.floor(px[i])), int(np.floor(py[i]))
        if 0 <= cx < fov.nx and 0 <= cy < fov.ny:
            n_locs += 1
        cols, kx, sx = kernel(cx, px[i], s)
        rows, ky, sy = kernel(cy, py[i], sy_i)
        tile = np.outer(ky, kx) * (weights[0] if weights.size == 1 else weights[i]) \
            / (sx * sy)
        keep_r = (rows >= 0) & (rows < fov.ny)
        keep_c = (cols >= 0) & (cols < fov.nx)
        if not keep_r.any() or not keep_c.any():
            continue
        block = np.ix_(rows[keep_r], cols[keep_c])
        tile = tile[np.ix_(keep_r, keep_c)]
        weight[block] += tile
        if color is not None:
            color[block] += tile[..., None] * colors[i]

    return RenderedImage(fov, weight.astype(np.float32),
                         None if color is None else color.astype(np.float32), n_locs)


# ------------------------------------------------------------------ front door
# The columns positions and precisions are read from: nm, always.  A fit writes
# nm only (`locs.to_nm`) and a loader gives a pixel-only file nm ones
# (`io.formats.with_nm`), so nothing downstream has two cases.
POSITION_FIELDS = (("x_nm", "y_nm"),)
PRECISION_FIELDS = ("xy_err_nm",)
LATERAL_FIELDS = tuple(n for pair in POSITION_FIELDS for n in pair)
# z is always in nm (`locs.to_nm` leaves it alone) and has a precision of its own
AXIAL_FIELD = "z_nm"
AXIAL_PRECISION_FIELD = "z_err_nm"


def _pick(locs: Localizations, names, what: str):
    for name in names:
        if (all(n in locs for n in name) if isinstance(name, tuple)
                else name in locs):
            return name
    raise KeyError(f"no {what} column in the table; looked for {names}")


def positions(locs: Localizations, select=None):
    """The position columns of a table, in nm."""
    x_name, y_name = _pick(locs, POSITION_FIELDS, "position")
    x, y = locs[x_name], locs[y_name]
    return (x, y) if select is None else (x[select], y[select])


def axis_unit(name: str) -> str:
    """What one unit of an axis is called, for a scale bar.

    A position column is a length and says so; anything else is counted in
    itself -- "2000 photons", "500 frame" -- which is the honest label when the
    axis is not a distance at all.
    """
    if name.endswith("_nm"):
        return "nm"
    if name.endswith("_pix"):
        return "pixels"
    return name


def precision_for(locs: Localizations, name: str) -> Optional[str]:
    """The localization precision belonging to a position column, if there is one.

    Returns None for a column that is not a position at all -- a photon count
    has no precision to be blurred with -- which is what makes the per-axis
    sigma rule in `render_sigmas` fall through to an explicit width.

    A z axis takes the axial precision where the fit produced one and the
    lateral precision otherwise, which understates the blur along z -- it is
    what every rotated view in the program already does, and a side view that
    suddenly sharpened or softened against the 3D one would be worse.
    """
    if name == AXIAL_FIELD:
        if AXIAL_PRECISION_FIELD in locs:
            return AXIAL_PRECISION_FIELD
        return next((n for n in PRECISION_FIELDS if n in locs), None)
    if name in LATERAL_FIELDS:
        return next((n for n in PRECISION_FIELDS if n in locs), None)
    return None


def is_position(name: str) -> bool:
    return name in LATERAL_FIELDS or name == AXIAL_FIELD


@dataclass(frozen=True)
class RenderAxes:
    """Which columns the render grid's axes are, and the scale of each.

    The default -- the table's own position columns, unscaled -- is the
    superresolution image everybody means.  Any other pair turns the same
    renderer into SMAP's versatile renderer: photons against frame, z against
    x, one fit parameter against another, drawn with the same layers, LUTs,
    contrast, ROIs and 3D box as a picture of a cell.

    A coordinate is the column divided by that axis's ``scale``, so the grid
    stays *square in render units* and the anisotropy lives in the two scales.
    Everything downstream -- the field of view, the zoom, the ROI shapes, the
    projection, the GPU kernels -- is then untouched and never has to learn
    about aspect ratios.  It is how SMAP does it too (``sr_pixrec = 1`` with
    each field divided by a pixel size of its own).

    ``None`` means "whatever the table carries", so the default costs nothing
    and a pixel-unit table keeps working.
    """

    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    x_scale: float = 1.0        # native units of the column per render unit
    y_scale: float = 1.0
    z_scale: float = 1.0

    @property
    def is_default(self) -> bool:
        """The table's own positions at their own scale: nothing to do."""
        return (self.x is None and self.y is None and self.z is None
                and self.x_scale == 1.0 and self.y_scale == 1.0
                and self.z_scale == 1.0)

    def names(self, locs: Localizations) -> Tuple[str, str]:
        """The two columns, resolving the defaults against the table."""
        if self.x is not None and self.y is not None:
            return (self.x, self.y)
        x_name, y_name = _pick(locs, POSITION_FIELDS, "position")
        return (self.x or x_name, self.y or y_name)

    def depth_name(self, locs: Localizations) -> Optional[str]:
        """The third axis, or None when the table has nothing to put there."""
        name = self.z if self.z is not None else AXIAL_FIELD
        return name if name in locs else None

    def coordinates(self, locs: Localizations, select=None):
        """The x and y of the render grid, in render units."""
        x_name, y_name = self.names(locs)
        return (self._values(locs, x_name, self.x_scale, select),
                self._values(locs, y_name, self.y_scale, select))

    def depth(self, locs: Localizations, select=None) -> Optional[np.ndarray]:
        name = self.depth_name(locs)
        return None if name is None else self._values(locs, name, self.z_scale, select)

    def normalised(self, locs: Localizations) -> "RenderAxes":
        """The same axes, with a spelled-out default folded back into one.

        Picking ``x_nm`` and ``y_nm`` by hand is the ordinary picture, and it
        should stay the ordinary picture -- index culling, one scale bar --
        rather than the versatile path that happens to look the same.
        """
        if self.is_default:
            return self
        x_name, y_name = _pick(locs, POSITION_FIELDS, "position")
        if ((self.x, self.y) == (x_name, y_name)
                and self.z in (None, AXIAL_FIELD)
                and self.x_scale == self.y_scale == self.z_scale == 1.0):
            return RenderAxes()
        return self

    def scale_of(self, locs: Localizations, axis: str) -> float:
        return {"x": self.x_scale, "y": self.y_scale, "z": self.z_scale}[axis]

    @staticmethod
    def _values(locs: Localizations, name: str, scale: float, select):
        if name not in locs:
            raise KeyError(f"no column {name!r} in the table; it has "
                           f"{', '.join(sorted(locs.keys()))}")
        values = locs[name]
        if select is not None:
            values = values[select]
        if scale == 1.0 and name in LATERAL_FIELDS:
            return values                       # the usual case: not even a copy
        return np.asarray(values, dtype=np.float32) / np.float32(scale)


@dataclass
class RenderSettings:
    """What to accumulate: the kernel, and what colours the localizations.

    The two sigmas are per axis because the axes need not be the same
    quantity: on a picture of photons against frame there is no localization
    precision to blur with, and a width in nanometres says nothing about one
    in photons.  `render_sigmas` has the rule; zero is plain binning along
    that axis.
    """

    mode: str = "precision"     # "hist", "gauss" (one sigma), "precision", "dl", "tracks", "voronoi"
    sigma: float = 10.0         # native units of the x axis, for mode="gauss"
    sigma_y: Optional[float] = None      # None: `sigma`, where that means anything
    sigma_settings: "SigmaSettings" = field(default_factory=lambda: SigmaSettings())
    roi_sigma: float = ROI_SIGMA
    color_field: Optional[str] = None    # None: an intensity image, recoloured later
    color_range: Optional[Tuple[float, float]] = None
    weight_field: Optional[str] = None   # None: one count per localization
    axes: RenderAxes = field(default_factory=RenderAxes)
    # mode="tracks": the id column whose localizations are one track ("":
    # auto, `tracking.columns`), and the lines' width in pixels of the picture
    track_column: str = ""
    line_width: float = 1.0
    # mode="tracks": whether lines add up where they overlap.  Off, a pixel a
    # line covers is one line whatever crosses it: the joints of a track and
    # its crossings with others are no brighter than the line, and the
    # quantile contrast is set by the lines rather than by their knots
    lines_add: bool = False


@dataclass
class DisplaySettings:
    """How to turn accumulated planes into RGB.  Changing these re-displays, not
    re-renders -- except ``lut``/``invert`` when colouring by a field, because
    then the colour is baked into the accumulation."""

    lut: luts.LUT = "hot"
    # False, or which inversion (`lut.INVERSIONS`): they are not the same
    # picture past a ramp of one hue.  A plain True, from a script that does
    # not choose, means `lut.DEFAULT_INVERSION`.
    invert: Union[bool, str] = False
    contrast: float = DEFAULT_CONTRAST   # saturate 10^-contrast of the pixels
    # SMAP's toggle between the quantile and an absolute scale: with
    # ``absolute`` the picture is full scale at ``maximum`` localizations per
    # um^2, the same at every zoom (`absolute_imax`); ``contrast`` is kept
    # for when it is switched back
    absolute: bool = False
    maximum: float = 1000.0
    # the value that is full scale, in the planes' own units, overriding both:
    # what the 3D view sets to show every slice on one scale
    imax: Optional[float] = None
    gamma: float = 1.0
    # the picture on white paper, whatever the LUT.  A property of the picture
    # rather than of one layer -- what sums several of them turns the sum over
    # once, with `white_background=False` per layer -- but it is kept here so
    # that a single-layer render (an export, a script) needs nothing else.
    white_background: bool = False

    def apply(self, rendered: RenderedImage,
              white_background: Optional[bool] = None) -> np.ndarray:
        white = self.white_background if white_background is None else white_background
        return to_rgb(rendered, self.lut, self.invert, self.pixel_max(rendered),
                      self.contrast, self.gamma, white)

    def pixel_max(self, rendered: RenderedImage) -> Optional[float]:
        """The value shown at full scale on this render, or None for the
        quantile (`normalize`)."""
        if self.imax is not None:
            return self.imax
        if self.absolute:
            return absolute_imax(rendered, self.maximum)
        return None

    def as_absolute(self, rendered: RenderedImage) -> "DisplaySettings":
        """These settings on the absolute scale, at the maximum that shows
        ``rendered`` as they do now: SMAP's toggle, which starts the absolute
        Imax at the one the quantile had found."""
        _, imax = normalize(rendered.weight, self.pixel_max(rendered), self.contrast)
        per_pixel = absolute_imax(rendered, 1.0)
        maximum = imax / per_pixel if imax > 0 and per_pixel > 0 else self.maximum
        return replace(self, absolute=True, maximum=float(maximum))


def explicit_sigmas(locs: Localizations, settings: "RenderSettings") -> Tuple[float, float]:
    """The two widths the settings state, in render units.

    ``sigma_y`` unset means the same width as x where the two axes are the
    same quantity, and zero -- plain binning -- where they are not: a width in
    nanometres says nothing about one in photons.
    """
    axes = settings.axes
    x_name, y_name = axes.names(locs)
    sigma_y = settings.sigma_y
    if sigma_y is None:
        sigma_y = settings.sigma if axis_unit(x_name) == axis_unit(y_name) else 0.0
    return (settings.sigma / axes.x_scale, sigma_y / axes.y_scale)


def render_sigmas(locs: Localizations, settings: "RenderSettings", fov: FieldOfView,
                  select=None):
    """The rendering width along each axis, in render units.

    ``(None, None)`` for a histogram.  The rule is per axis because the axes
    need not be positions:

    * mode ``"gauss"`` is the explicit pair everywhere -- that is what a
      constant Gaussian means;
    * mode ``"precision"`` uses the localization precision on an axis that
      *is* a position (``z_err_nm`` for a z axis, when the fit
      produced one), and the explicit width on any other axis, because there
      is no precision for a photon count or a frame number;
    * ``sigma_y`` unset means the same width as x where the two axes are the
      same quantity, and zero -- plain binning -- where they are not.
    """
    if settings.mode in ("hist", "tracks", "voronoi"):
        return None, None
    if settings.mode == "dl":
        return psf_sigmas(locs, select)
    if settings.mode not in MODES:
        raise ValueError(f"unknown mode {settings.mode!r}; "
                         f"use one of {', '.join(repr(m) for m in MODES)}")
    axes = settings.axes
    x_name, y_name = axes.names(locs)
    explicit_x, explicit_y = explicit_sigmas(locs, settings)

    out, fields = [], []
    for name, scale, explicit in ((x_name, axes.x_scale, explicit_x),
                                  (y_name, axes.y_scale, explicit_y)):
        field_name = precision_for(locs, name) if settings.mode == "precision" else None
        if field_name is None:
            if settings.mode == "precision" and is_position(name):
                raise KeyError("no localization precision column in the table; "
                               f"looked for {PRECISION_FIELDS}")
            out.append(np.float32(explicit))          # already in render units
        else:
            values = locs[field_name]
            values = values if select is None else values[select]
            values = np.asarray(values, dtype=np.float32) / np.float32(scale)
            out.append(settings.sigma_settings.apply(values, fov.pixelsize))
        fields.append((field_name, scale))
    # the usual picture has the same precision on both axes: hand it over once
    # rather than putting the same millions of values through twice
    if fields[0] == fields[1] and fields[0][0] is not None:
        return out[0], None
    return out[0], out[1]


def render_locs(locs: Localizations, fov: FieldOfView,
                settings: Optional[RenderSettings] = None,
                display: Optional[DisplaySettings] = None, select=None,
                n_threads: int = 0, use_extension: bool = True) -> RenderedImage:
    """Render a localization table.

    ``select`` narrows the table to a subset: a :class:`~smappy.filter.LocFilter`,
    a boolean mask, or an index array.  Only the three or four columns the render
    needs are indexed, so filtering never copies the table.
    """
    settings = settings or RenderSettings()
    display = display or DisplaySettings()
    select = getattr(select, "indices", select)  # a LocFilter, or a mask/indices

    # A live fit's table exists before its first block does, with no columns
    # yet, and the view asks for a picture of it the moment the fit starts:
    # that is an empty picture, not a missing position column.
    if not len(locs):
        return RenderedImage(fov, np.zeros((fov.ny, fov.nx)))

    if settings.mode == "tracks":
        return render_tracks(locs, fov, settings, display, select, n_threads,
                             use_extension)
    if settings.mode == "voronoi":
        return render_voronoi(locs, fov, settings, display, select, n_threads)
    x, y = settings.axes.coordinates(locs, select)
    sigma, sigma_y = render_sigmas(locs, settings, fov, select)

    weights = None
    if settings.weight_field is not None:
        weights = locs[settings.weight_field]
        weights = weights if select is None else weights[select]

    colors = None
    if settings.color_field is not None:
        values = locs[settings.color_field]
        values = values if select is None else values[select]
        lo, hi = settings.color_range or finite_range(values)
        colors = luts.colors(values, display.lut, lo, hi, display.invert)

    if settings.mode == "dl":
        if not settings.axes.is_default:
            raise ValueError("DL rendering draws what the camera saw, so it needs "
                             "the positions on both axes")
        camera = camera_grid(fov, camera_pixelsize_nm(locs))
        coarse = render(x, y, camera, sigma=sigma, sigma_y=sigma_y, weights=weights,
                        colors=colors, roi_sigma=settings.roi_sigma,
                        n_threads=n_threads, use_extension=use_extension)
        return resample_nearest(coarse, fov)
    return render(x, y, fov, sigma=sigma, sigma_y=sigma_y, weights=weights,
                  colors=colors, roi_sigma=settings.roi_sigma, n_threads=n_threads,
                  use_extension=use_extension)


# ------------------------------------------------------------ diffraction limited
#
# SMAP's "DL" mode: every localization drawn as the spot the camera saw -- a
# Gaussian as wide as its fitted PSF, on the camera's own pixels -- so a
# super-resolved picture can be set beside the widefield image it came from,
# or the two told apart in a figure.  It is the Gaussian renderer on a grid of
# camera pixels, blown up to the view without interpolation so the pixels
# stay visible.  As SMAP (renderSMAP.m, case 'dl'): the width is `sigma_nm`
# (and `sigma_y_nm`, astigmatic), and one camera pixel where the fit left no
# usable width.

MODES = ("hist", "gauss", "precision", "dl", "tracks", "voronoi")
DL_PIXELSIZE_NM = 100.0     # a camera pixel, when the table records none
DL_MAX_SIGMA_NM = 1500.0    # SMAP's: a wider fitted PSF is a failed fit


def camera_pixelsize_nm(locs: Localizations) -> float:
    """The camera pixel the table was fitted from, in nm.

    From the fit's metadata (``pixelsize_nm``, or the camera's
    ``pixelsize_um``), a simulation's optics, or `DL_PIXELSIZE_NM`.
    """
    meta = locs.metadata
    value = meta.get("pixelsize_nm")
    if value is None:
        camera = meta.get("camera")
        micron = camera.get("pixelsize_um") if isinstance(camera, dict) else None
        if isinstance(micron, (list, tuple)):
            micron = micron[0] if micron else None
        value = None if micron is None else float(micron) * 1000.0
    if value is None:
        sim = meta.get("simulation_settings")
        optics = sim.get("optics") if isinstance(sim, dict) else None
        value = optics.get("pixelsize_nm") if isinstance(optics, dict) else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DL_PIXELSIZE_NM
    return value if np.isfinite(value) and value > 0 else DL_PIXELSIZE_NM


def psf_sigmas(locs: Localizations, select=None):
    """Each localization's PSF width, x and y, in nm, for the DL mode.

    A width that is missing, not positive or beyond `DL_MAX_SIGMA_NM` is one
    camera pixel instead.  ``sigma_y`` is ``None`` when the fit had one width.
    """
    pixel = camera_pixelsize_nm(locs)

    def width(name):
        if name not in locs:
            return None
        values = np.asarray(locs[name] if select is None else locs[name][select],
                            dtype=np.float32)
        bad = ~(np.isfinite(values) & (values > 0) & (values < DL_MAX_SIGMA_NM))
        return np.where(bad, np.float32(pixel), values).astype(np.float32)

    sx = width("sigma_nm")
    if sx is None:
        return np.float32(pixel), None
    return sx, width("sigma_y_nm")


def camera_grid(fov: FieldOfView, pixelsize: float) -> FieldOfView:
    """Camera pixels covering ``fov``, their edges on multiples of the pixel."""
    ix0, iy0 = math.floor(fov.x0 / pixelsize), math.floor(fov.y0 / pixelsize)
    ix1, iy1 = math.ceil(fov.x1 / pixelsize), math.ceil(fov.y1 / pixelsize)
    return FieldOfView(ix0 * pixelsize, iy0 * pixelsize, float(pixelsize),
                       max(ix1 - ix0, 1), max(iy1 - iy0, 1))


def resample_nearest(image: RenderedImage, fov: FieldOfView) -> RenderedImage:
    """``image``'s planes onto ``fov``, each view pixel from the one it lies in."""
    src = image.fov
    xs = fov.x0 + (np.arange(fov.nx) + 0.5) * fov.pixelsize
    ys = fov.y0 + (np.arange(fov.ny) + 0.5) * fov.pixelsize
    ix = np.clip(np.floor((xs - src.x0) / src.pixelsize).astype(np.int64), 0, src.nx - 1)
    iy = np.clip(np.floor((ys - src.y0) / src.pixelsize).astype(np.int64), 0, src.ny - 1)
    weight = image.weight[np.ix_(iy, ix)]
    color = None if image.color is None else image.color[np.ix_(iy, ix)]
    # counted over the source's pixels, however finely they are shown
    area = image.count_area if image.count_area is not None else src.pixelsize ** 2
    return RenderedImage(fov, weight, color, image.n_locs, count_area=area)


def save_image(locs, path, pixelsize: float = 10.0,
               settings: Optional[RenderSettings] = None,
               display: Optional[DisplaySettings] = None, select=None,
               fov: Optional[FieldOfView] = None, margin: float = 0.0) -> "Path":
    """Render a table and write it as an image file.  Returns the path.

    The whole picture with no window and no event loop, for a program that has
    its own -- a report, a thumbnail, a tile in a web page.  ``pixelsize`` is the
    rendered pixel in the units of the table (nm, normally); ``fov`` frames it
    somewhere other than around the data.

    Needs Pillow, which is what writes the file.
    """
    from pathlib import Path as _Path

    try:
        import PIL  # noqa: F401
    except ImportError:            # pragma: no cover - depends on the install
        raise ImportError("save_image writes the file with Pillow: "
                          "pip install pillow") from None

    if isinstance(locs, (str, _Path)):
        from .io.hdf5 import load_localizations
        locs = load_localizations(locs)

    if fov is None:
        x, y = (settings or RenderSettings()).axes.coordinates(
            locs, getattr(select, "indices", select))
        fov = FieldOfView.around(x, y, pixelsize=pixelsize, margin=margin)
    # drawn and written as *Export Image* draws and writes the window's
    # picture (`smappy.picture`): one layer, nothing over it
    from .layer_state import LayerSnapshot
    from .picture import render_picture, write_picture
    layer = LayerSnapshot(name="localizations", locs=locs, mask=_select_mask(select, len(locs)),
                          settings=settings or RenderSettings(),
                          display=display or DisplaySettings())
    (picture,) = render_picture((layer,), fov, what="composite", scale_bar=False,
                                colour_bars=False, labels=False)
    return write_picture(picture, _Path(path))


def _select_mask(select, n: int) -> np.ndarray:
    """A selection as `render_locs` takes it -- None, a filter, a mask or
    rows -- as a mask."""
    if select is None:
        return np.ones(n, bool)
    select = np.asarray(getattr(select, "mask", select))
    if select.dtype == bool:
        return select
    mask = np.zeros(n, bool)
    mask[select.astype(np.int64)] = True
    return mask


# ------------------------------------------------------------------- tracks
#
# The fifth mode: each track drawn as the line through its localizations in
# time order -- a track is the localizations sharing an id column's value
# above 0 (`track_id` from Link tracks, MINFLUX's `tid`).  A raster like the
# others, so the export, the overlays and the sum of layers need nothing new;
# a vector layer was rejected because the exports would miss it.  The view
# draws each track whole within the rows the layer's filter keeps; its
# localizations are not culled to the view first (`LayerSnapshot.select`), as
# a segment crossing the view can have both ends outside it -- whole tracks
# are culled instead, by their bounding boxes.


def track_segments(locs: Localizations, settings: "RenderSettings", select=None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """The segments of every track among the rows ``select`` keeps: the
    rows of their two ends, ``(a, b)``, consecutive in time within a track."""
    from .tracking.columns import track_column
    column = track_column(locs, settings.track_column)
    rows = (np.arange(len(locs)) if select is None
            else np.asarray(select) if np.asarray(select).dtype != bool
            else np.flatnonzero(select))
    ids = np.asarray(locs[column])[rows]
    keep = ids > 0
    rows, ids = rows[keep], ids[keep]
    time = np.asarray(locs["time_s"] if "time_s" in locs else locs["frame"])[rows]
    order = np.lexsort((time, ids))
    rows, ids = rows[order], ids[order]
    same = ids[1:] == ids[:-1]
    return rows[:-1][same], rows[1:][same]


def _cull_tracks(a, b, ids, xa, ya, xb, yb, fov: FieldOfView, reach: float):
    """The segments of tracks whose bounding box reaches into ``fov``."""
    if not len(a):
        return np.zeros(0, bool)
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    lo_x = np.minimum.reduceat(np.minimum(xa, xb), starts)
    hi_x = np.maximum.reduceat(np.maximum(xa, xb), starts)
    lo_y = np.minimum.reduceat(np.minimum(ya, yb), starts)
    hi_y = np.maximum.reduceat(np.maximum(ya, yb), starts)
    seen = ((hi_x >= fov.x0 - reach) & (lo_x <= fov.x1 + reach)
            & (hi_y >= fov.y0 - reach) & (lo_y <= fov.y1 + reach))
    track = np.cumsum(np.isin(np.arange(len(a)), starts)) - 1
    return seen[track]


def render_tracks(locs: Localizations, fov: FieldOfView, settings: "RenderSettings",
                  display: Optional["DisplaySettings"] = None, select=None,
                  n_threads: int = 0, use_extension: bool = True) -> RenderedImage:
    """Every track as antialiased lines ``settings.line_width`` pixels wide.

    A line is one wherever it passes, at every zoom (`render_segments`), so
    its pixels do not count over an area (`RenderedImage.count_area`): an
    absolute maximum is lines over a pixel.  A segment weighs one (or the mean of its ends' ``weight_field``) and,
    coloured by a field, takes the mean of its ends' values: a track coloured
    by time changes colour along itself.
    """
    display = display or DisplaySettings()
    a, b = track_segments(locs, settings, select)
    x, y = settings.axes.coordinates(locs)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    from .tracking.columns import track_column
    ids = np.asarray(locs[track_column(locs, settings.track_column)])[a]
    reach = (0.5 * settings.line_width + 1.0) * fov.pixelsize
    seen = _cull_tracks(a, b, ids, x[a], y[a], x[b], y[b], fov, reach)
    a, b = a[seen], b[seen]
    weights = None
    if settings.weight_field is not None:
        w = np.asarray(locs[settings.weight_field], np.float64)
        weights = 0.5 * (w[a] + w[b])
    colors = None
    if settings.color_field is not None:
        values = np.asarray(locs[settings.color_field], np.float64)
        mid = 0.5 * (values[a] + values[b])
        lo, hi = settings.color_range or finite_range(values)
        colors = luts.colors(mid, display.lut, lo, hi, display.invert)
    out = render_segments(x[a], y[a], x[b], y[b], fov, settings.line_width, weights,
                          colors, n_threads, use_extension)
    if not settings.lines_add:
        top = 1.0 if weights is None or not len(weights) else float(np.nanmax(weights))
        out = opaque_lines(out, top)
    drawn = np.zeros(len(locs), bool)          # the localizations the lines join
    drawn[a] = drawn[b] = True
    out.n_locs = int(drawn.sum())
    return out


def opaque_lines(image: RenderedImage, top: float = 1.0) -> RenderedImage:
    """Lines that cover rather than add: each pixel's weight capped at
    ``top``, one line's (the heaviest segment's, with a weight field), and its
    colour scaled with it, so the colour stays the mean of the lines there.

    Done on the accumulated planes, after the kernel: an antialiased edge
    keeps its fraction of a line, and where segments overlap -- a joint, a
    crossing, a track that turns back on itself -- the pixel is simply full.
    """
    weight = image.weight
    over = weight > top
    if not over.any():
        return image
    scale = np.where(over, top / np.where(over, weight, 1.0), 1.0).astype(np.float32)
    color = None if image.color is None else image.color * scale[..., None]
    return replace(image, weight=weight * scale, color=color)


def render_segments(xa, ya, xb, yb, fov: FieldOfView, width: float = 1.0,
                    weights=None, colors=None, n_threads: int = 0,
                    use_extension: bool = True) -> RenderedImage:
    """Accumulate line segments from (xa, ya) to (xb, yb) into ``fov``,
    ``width`` pixels wide, antialiased (``csrc/render.cpp``).

    A pixel takes from a segment its coverage across the line -- 1 within
    ``width / 2 - 0.5`` of it, falling to 0 at ``width / 2 + 0.5`` -- times
    how much of a pixel-long stretch about the pixel's centre, along the
    segment, lies on the segment.  The ends are square and soft, so the
    segments of a track tile it: a straight line is one along its length
    however many segments it is cut into.  With round ends, as before
    2026-10, every segment added a whole line over its ends, and zoomed out
    -- a track's steps shorter than a pixel -- a slow track piled up into a
    blob hundreds of times brighter than a line, the quantile contrast was
    set by those blobs, and every other track went dim.  A segment of no
    length -- a track along the line of sight, in a projection -- is a
    round dot.
    """
    xa, ya, xb, yb = (np.ascontiguousarray(v, np.float32) for v in (xa, ya, xb, yb))
    w = (np.ones(1, np.float32) if weights is None
         else np.ascontiguousarray(weights, np.float32).reshape(-1))
    c = (np.zeros((0, 3), np.float32) if colors is None
         else np.ascontiguousarray(colors, np.float32).reshape(-1, 3))
    # an extension built before the tracks mode lacks the kernel: draw, slowly
    if use_extension and hasattr(_render, "render_segments"):
        weight, color = _render.render_segments(xa, ya, xb, yb, w, c, fov.x0, fov.y0,
                                                fov.pixelsize, fov.nx, fov.ny,
                                                float(width), n_threads)
        return RenderedImage(fov, weight, color, len(xa), count_area=PER_UM2)
    return _segments_numpy(xa, ya, xb, yb, fov, float(width), w, c)


def _segments_numpy(xa, ya, xb, yb, fov, width, weights, colors) -> RenderedImage:
    """Reference implementation of `render_segments`: every pixel of each
    segment's bounding box, the same coverage rule."""
    weight = np.zeros(fov.shape, np.float64)
    color = np.zeros((fov.ny, fov.nx, 3), np.float64) if colors.size else None
    reach = 0.5 * max(width, 0.0) + 0.5
    for i in range(len(xa)):
        ax, ay = fov.to_pixels(xa[i], ya[i])
        bx, by = fov.to_pixels(xb[i], yb[i])
        c0 = max(0, int(np.floor(min(ax, bx) - reach)))
        c1 = min(fov.nx - 1, int(np.floor(max(ax, bx) + reach)))
        r0 = max(0, int(np.floor(min(ay, by) - reach)))
        r1 = min(fov.ny - 1, int(np.floor(max(ay, by) + reach)))
        if c0 > c1 or r0 > r1:
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1) + 0.5, np.arange(r0, r1 + 1) + 0.5)
        dx, dy = bx - ax, by - ay
        length = np.hypot(dx, dy)
        if length > 0:
            along = ((cc - ax) * dx + (rr - ay) * dy) / length
            across = np.abs((cc - ax) * dy - (rr - ay) * dx) / length
            on = np.clip(np.minimum(along + 0.5, length) - np.maximum(along - 0.5, 0.0),
                         0, 1)
        else:                                   # along the line of sight: a dot
            across, on = np.hypot(cc - ax, rr - ay), 1.0
        cover = (np.clip(reach - across, 0, 1) * on
                 * (weights[0] if weights.size == 1 else weights[i]))
        weight[r0:r1 + 1, c0:c1 + 1] += cover
        if color is not None:
            color[r0:r1 + 1, c0:c1 + 1] += cover[..., None] * colors[i]
    return RenderedImage(fov, weight.astype(np.float32),
                         None if color is None else color.astype(np.float32), len(xa),
                         count_area=PER_UM2)


# ------------------------------------------------------------------ voronoi
#
# The sixth mode (SharpViSu, Andronov et al. 2016; SMAP's VoronoiRenderer):
# each pixel takes the density, one over the area, of the Voronoi cell it
# lies in -- the cell of the localization nearest to it -- so a dense
# structure is bright and flat rather than a pile of spots, with no width to
# choose.  The cells are Neighbourhood's (`voronoi_areas`): open cells, along
# the edge of the field, have no area and are left empty.  A raster like the
# others, so exports, overlays and movies need nothing new.
#
# The cells are of every localization the layer's filter keeps, never of
# the ones in view: culled first, the cells along the view's edge would
# change with every pan.  They take about 12 us a localization, so they are
# cached per table and filter (`voronoi_cells_for`) and a pan or a zoom only
# rasterises: the pixels are culled to the view, the localizations are not.
#
# Zoomed out, a pixel holds many cells and a sample at its centre would pick
# one of them at random -- a picture of noise.  The mean density over the
# pixel is what is wanted, and a cell's density times its area is its
# localizations: so cells smaller than a pixel are counted into the pixel
# their localization is in (a histogram, over the pixel's area), and the
# rest of the pixel's area takes the density of the cell its centre lies
# in.  Zoomed in, that is SharpViSu's picture exactly; zoomed out, the
# histogram in localizations per area.

VORONOI_CACHE_SIZE = 4      # tables and filters whose cells are kept
VORONOI_CHUNK = 1 << 20     # pixel centres looked up at a time


@dataclass
class VoronoiCells:
    """The Voronoi cells of a set of localizations, in render units.

    ``sites`` are the distinct positions, ``cell`` the site of each
    localization (positions repeated share a cell, each of them counted, as
    SMAP's ``VorArea``), ``area`` each cell's area (NaN open) and
    ``density`` its localizations over its area per 10^6 square units: per
    square micrometre for positions in nm.
    """

    sites: np.ndarray
    cell: np.ndarray
    repeats: np.ndarray
    area: np.ndarray
    density: np.ndarray
    _tree: object = field(default=None, repr=False)

    @classmethod
    def of(cls, x, y) -> "VoronoiCells":
        from .plugins.neighbourhood import voronoi_areas
        points = np.column_stack([np.asarray(x, np.float64), np.asarray(y, np.float64)])
        if not len(points):
            empty = np.zeros(0)
            return cls(np.zeros((0, 2)), np.zeros(0, np.int64), empty, empty, empty)
        sites, cell, repeats = np.unique(points, axis=0, return_inverse=True,
                                         return_counts=True)
        area = (voronoi_areas(sites) if len(sites) >= 4
                else np.full(len(sites), np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(area > 0, repeats * 1e6 / area, np.nan)
        area = np.where(area > 0, area, np.nan)
        return cls(sites, np.ravel(cell), repeats, area, density)

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.sites)
        return self._tree


_voronoi_cache: "OrderedDict[tuple, VoronoiCells]" = OrderedDict()
_voronoi_lock = threading.Lock()


def _voronoi_key(x: np.ndarray, y: np.ndarray) -> tuple:
    """What the cells depend on, cheaply: the count, the sums and a strided
    sample of the positions.  Hashing every byte took a quarter of a second
    at 10 M localizations, on every pan; a different filter or table that
    agrees on all of these is not one that occurs."""
    step = max(1, len(x) // 65536)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(x[::step]).tobytes())
    digest.update(np.ascontiguousarray(y[::step]).tobytes())
    return (len(x), float(np.sum(x)), float(np.sum(y)), digest.hexdigest())


def voronoi_cells_for(x, y) -> VoronoiCells:
    """`VoronoiCells.of`, kept for the last `VORONOI_CACHE_SIZE` sets of
    positions: a view panned, zoomed or exported reuses them."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    key = _voronoi_key(x, y)
    with _voronoi_lock:
        found = _voronoi_cache.get(key)
        if found is not None:
            _voronoi_cache.move_to_end(key)
            return found
    # outside the lock: two threads may both compute it, neither waits
    cells = VoronoiCells.of(x, y)
    with _voronoi_lock:
        _voronoi_cache[key] = cells
        while len(_voronoi_cache) > VORONOI_CACHE_SIZE:
            _voronoi_cache.popitem(last=False)
    return cells


def render_voronoi(locs: Localizations, fov: FieldOfView, settings: "RenderSettings",
                   display: Optional["DisplaySettings"] = None, select=None,
                   n_threads: int = 0) -> RenderedImage:
    """Each pixel the mean density of the Voronoi cells over it, in
    localizations per 10^6 square units (per square micrometre in nm).

    A cell smaller than a pixel is counted into the pixel of its
    localizations; the rest of a pixel's area is the cell its centre lies
    in, empty for an open one.  Coloured by a field, a counted localization
    brings its own colour and a cell the mean of its localizations'.  A
    weight column is not used: the density is of localizations.
    """
    display = display or DisplaySettings()
    x, y = settings.axes.coordinates(locs, select)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    cells = voronoi_cells_for(x, y)
    pixel_area = fov.pixelsize ** 2
    per_area = 1e6 / pixel_area
    values = None
    if settings.color_field is not None:
        values = np.asarray(locs[settings.color_field], np.float64)
        values = values if select is None else values[select]
        lo, hi = settings.color_range or finite_range(values)

    # the cells smaller than a pixel, counted where their localizations are
    small = np.isfinite(cells.area) & (cells.area < pixel_area)
    counted = small[cells.cell] if len(cells.cell) else np.zeros(0, bool)
    colors = (None if values is None else
              luts.colors(values[counted], display.lut, lo, hi, display.invert))
    mass = render_histogram(x[counted], y[counted], fov, colors=colors)
    share = cells.area[cells.cell[counted]] / cells.repeats[cells.cell[counted]]
    covered = render_histogram(x[counted], y[counted], fov, weights=share).weight
    weight = mass.weight.astype(np.float64) * per_area
    color = None if mass.color is None else mass.color.astype(np.float64) * per_area

    # the rest of each pixel: the cell its centre is in
    open_pixels = np.flatnonzero(covered.ravel() < pixel_area)
    if len(open_pixels) and len(cells.sites):
        tree = cells.tree()
        site_values = None
        if values is not None:
            site_values = (np.bincount(cells.cell, weights=values,
                                       minlength=len(cells.sites)) / cells.repeats)
        flat_w = weight.ravel()
        flat_c = None if color is None else color.reshape(-1, 3)
        rest = 1.0 - covered.ravel() / pixel_area
        for start in range(0, len(open_pixels), VORONOI_CHUNK):
            chunk = open_pixels[start:start + VORONOI_CHUNK]
            row, col = np.divmod(chunk, fov.nx)
            centres = np.column_stack([fov.x0 + (col + 0.5) * fov.pixelsize,
                                       fov.y0 + (row + 0.5) * fov.pixelsize])
            _, site = tree.query(centres, workers=n_threads if n_threads > 0 else -1)
            rho = cells.density[site]
            # a centre in a small cell is already counted, an open one is empty
            use = np.isfinite(rho) & ~small[site]
            add = rho[use] * rest[chunk[use]]
            flat_w[chunk[use]] += add
            if flat_c is not None:
                flat_c[chunk[use]] += add[:, None] * luts.colors(
                    site_values[site[use]], display.lut, lo, hi, display.invert)
    inside = ((x >= fov.x0) & (x < fov.x1) & (y >= fov.y0) & (y < fov.y1))
    return RenderedImage(fov, weight.astype(np.float32),
                         None if color is None else color.astype(np.float32),
                         int(inside.sum()), count_area=PER_UM2)
