"""The 3D view without a window: a projection, a slab, and engine A.

A `Projection` turns data coordinates into view coordinates ``(x', y',
depth)``; a `Slab` is the box of data that is shown, the 3D ROI.  Engine A
rotates the slab's localizations and renders ``(x', y')`` with the 2D
kernel, so the picture is the 2D image of the tilted slab, with every
layer's own settings.  Nothing here imports Qt.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .locs import Localizations
from .regions import Region
from .render import (DisplaySettings, FieldOfView, RenderAxes, RenderSettings,
                     RenderedImage, explicit_sigmas, is_position, normalize,
                     render_locs)

PREVIEW_POINTS = 2_000_000       # at most this many while the mouse drags
PREVIEW_SCALE = 2                # and at this many times coarser pixels
HIST_POINTS = 200_000            # the depth histogram is taken from this many
HIST_BINS = 64

# A drag frame draws a sample, never the whole table.  Two things bound it: what
# can be seen, and what the machine can do in time.
PREVIEW_PER_PIXEL = 13.0         # points per rendered pixel; beyond this the
                                 # extra ones land on top of each other
TARGET_FPS = 27.0                # the middle of a 25-30 fps band


def _view_dtype(*arrays) -> np.dtype:
    """The precision to project a batch of coordinates in.

    float32 for a float32 table, which is what every reader produces, and
    float64 only if something handed in needs it (the slab's corners, a test
    with Python floats).  Localization coordinates are nanometres over a
    field of a few hundred microns: float32 resolves that to ~4 pm, far below
    a pixel of the finest render, so the promotion these functions used to do
    bought nothing and cost a copy of the table per call.
    """
    seen = [np.asarray(a).dtype for a in arrays if a is not None]
    return np.dtype(np.result_type(np.float32, *seen) if seen else np.float32)


# ------------------------------------------------------------------- slab
@dataclass
class Slab:
    """A box in data coordinates: the volume the 3D view shows.

    Axis-aligned in z; rotated by ``angle`` (degrees) about z in the plane,
    so a line ROI's long axis can be the box's x.
    """
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.ones(3) * 1000.0)
    angle: float = 0.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.size = np.asarray(self.size, dtype=np.float64).reshape(3)

    @classmethod
    def from_bounds(cls, x0, x1, y0, y1, z0, z1) -> "Slab":
        return cls([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
                   [x1 - x0, y1 - y0, z1 - z0])

    @classmethod
    def from_region(cls, region: Region, z_range: Tuple[float, float]) -> "Slab":
        """A rectangle ROI is the footprint; a line ROI a rotated one."""
        z0, z1 = z_range
        if region.kind == "line":
            p = region.points
            start, end = (p[0] + p[3]) / 2, (p[1] + p[2]) / 2
            d = end - start
            return cls([*((start + end) / 2), (z0 + z1) / 2],
                       [float(np.hypot(*d)), region.width, z1 - z0],
                       angle=math.degrees(math.atan2(d[1], d[0])))
        x0, y0, x1, y1 = region.bounds
        return cls.from_bounds(x0, x1, y0, y1, z0, z1)

    def to_local(self, x, y, z=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Data -> slab frame: centred, in-plane rotation undone.

        Kept in the inputs' precision (`_view_dtype`): a float32 table used to
        be promoted to float64 here, which on a 46 M localization file is
        three 366 MB temporaries per call for no accuracy that survives into a
        pixel.  ``z`` None means the slab's own centre plane, and comes back
        as the scalar 0.0 rather than an array of them.
        """
        dtype = _view_dtype(x, y, z)
        c = dtype.type(math.cos(math.radians(self.angle)))
        s = dtype.type(math.sin(math.radians(self.angle)))
        dx = np.subtract(x, float(self.center[0]), dtype=dtype)
        dy = np.subtract(y, float(self.center[1]), dtype=dtype)
        w = 0.0 if z is None else np.subtract(z, float(self.center[2]), dtype=dtype)
        return c * dx + s * dy, -s * dx + c * dy, w

    def mask(self, x, y, z=None) -> np.ndarray:
        u, v, w = self.to_local(x, y, z)
        h = self.size / 2
        inside = (np.abs(u) <= h[0]) & (np.abs(v) <= h[1])
        # z None is the centre plane, which is inside every slab: no array for it
        return inside if z is None else inside & (np.abs(w) <= h[2])

    def corners(self) -> np.ndarray:
        """The 8 corners in data coordinates, (8, 3)."""
        h = self.size / 2
        local = np.array([[sx, sy, sz] for sx in (-h[0], h[0]) for sy in (-h[1], h[1])
                          for sz in (-h[2], h[2])])
        c, s = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return local @ rot.T + self.center

    def axis_unit(self, axis: int) -> np.ndarray:
        """The data-space direction of the slab's own axis 0, 1 or 2."""
        c, s = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]][axis])

    def axis_range(self, axis: int) -> Tuple[float, float]:
        """The extent along the slab's own axis, as a coordinate on that axis
        (for an unrotated slab: the plain x, y or z range)."""
        mid = float(self.center @ self.axis_unit(axis))
        return (mid - self.size[axis] / 2, mid + self.size[axis] / 2)

    def set_axis_range(self, axis: int, lo: float, hi: float) -> None:
        """Move only what changes: the other face stays where it is."""
        lo, hi = sorted((float(lo), float(hi)))
        unit = self.axis_unit(axis)
        mid = float(self.center @ unit)
        self.center = self.center + unit * ((lo + hi) / 2 - mid)
        self.size[axis] = max(hi - lo, 1e-6)

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "size": self.size.tolist(), "angle": self.angle}

    @classmethod
    def from_dict(cls, d: dict) -> "Slab":
        return cls(d["center"], d["size"], d.get("angle", 0.0))

    def __str__(self) -> str:
        return "slab {:.0f} x {:.0f} x {:.0f} nm".format(*self.size)


# ------------------------------------------------------------- projection
def _rz(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rx(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


PRESETS = {"top": (0.0, 0.0, 0.0), "front": (0.0, 90.0, 0.0), "side": (90.0, 90.0, 0.0)}


def _ry(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def euler_zxz(R: np.ndarray) -> Tuple[float, float, float]:
    """(azimuth, elevation, roll) with ``R = Rz(roll) Rx(elevation) Rz(azimuth)``,
    elevation in [0, 180]."""
    el = math.degrees(math.atan2(math.hypot(R[2, 0], R[2, 1]), R[2, 2]))
    if abs(math.sin(math.radians(el))) < 1e-9:      # gimbal: all of it is one turn about z
        az = math.degrees(math.atan2(R[1, 0], R[0, 0])) * (1 if R[2, 2] > 0 else -1)
        return az % 360, el, 0.0
    az = math.degrees(math.atan2(R[2, 0], R[2, 1]))          # R[2] = sinEl (sinAz, cosAz, .)
    roll = math.degrees(math.atan2(R[0, 2], -R[1, 2]))       # R[:, 2] = sinEl (sinRoll, -cosRoll, .)
    return az % 360, el, roll % 360


@dataclass
class Projection:
    """Data -> view: ``v = R (p - pivot)``; ``(v_x, v_y)`` on screen, ``v_z``
    is depth (towards the viewer positive).

    ``azimuth`` (about data z), ``elevation`` (tilt about view x) and ``roll``
    (about the depth axis) compose ``R``; `rotate_view` turns about the
    view's own axes and re-derives them, so a mouse can rotate without end.
    """
    azimuth: float = 0.0        # about the data z axis
    elevation: float = 0.0      # tilt about the view x axis, 0 = top view
    roll: float = 0.0           # about the view depth axis
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zoom: float = 10.0          # nm per screen pixel
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))   # pan, view nm
    focal: Optional[float] = None   # perspective: None = orthographic
    depth_lambda: Optional[float] = None   # attenuation length; None = off
    opacity: float = 0.0            # 0: plain sum; 1: the front hides the back
    slices: int = 32                # depth slices for the opacity compositing
    color_by_depth: bool = False    # override the layers' colour field
    fix_roll: bool = True           # turntable: turn about data z, tilt z forward / back
    engine: str = "cpu"             # "cpu", "gpu" (same image), "points" or "spheres" (GPU)
    point_size: float = 0.0         # points mode: radius in nm; 0 = the median precision
    point_alpha: float = 0.05       # points mode: sprite opacity
    ssao_strength: float = 0.7      # spheres mode: ambient occlusion, 0 = off
    ssao_radius: float = 0.0        # spheres mode: in nm; 0 = 3 x the sphere radius

    def __post_init__(self):
        self.pivot = np.asarray(self.pivot, dtype=np.float64).reshape(3)
        self.offset = np.asarray(self.offset, dtype=np.float64).reshape(2)

    @property
    def matrix(self) -> np.ndarray:
        return _rz(self.roll) @ _rx(self.elevation) @ _rz(self.azimuth)

    def apply(self, x, y, z=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x', y', depth) for data points; z defaults to the pivot's.

        The rotation is written out row by row rather than done as
        ``column_stack(...) @ R.T``: the stack and the product are each an
        (N, 3) array, so on a 46 M localization table that pair was 2.2 GB of
        temporaries per call in float64.  Row by row in the table's own
        precision (`_view_dtype`) it is three N-vectors, and 2.2x faster.
        """
        dtype = _view_dtype(x, y, z)
        m = self.matrix.astype(dtype)
        dx = np.subtract(x, float(self.pivot[0]), dtype=dtype)
        dy = np.subtract(y, float(self.pivot[1]), dtype=dtype)
        dz = None if z is None else np.subtract(z, float(self.pivot[2]), dtype=dtype)

        def row(i):
            # z None is the pivot's own plane: its term is zero, not an array
            v = m[i, 0] * dx + m[i, 1] * dy
            return v if dz is None else v + m[i, 2] * dz

        xv, yv, d = row(0), row(1), row(2)
        if self.focal:
            s = 1.0 / np.maximum(1.0 - d / self.focal, 0.05)
            xv, yv = xv * s, yv * s
        return xv, yv, d

    def view_axis(self, axis: int) -> np.ndarray:
        """The data-space direction of view axis 0 (x'), 1 (y') or 2 (depth)."""
        return self.matrix[axis]

    def rotate_by(self, d_azimuth: float, d_elevation: float, d_roll: float = 0.0) -> None:
        """Change the angles themselves (the dials)."""
        self.azimuth = (self.azimuth + d_azimuth) % 360
        self.elevation = (self.elevation + d_elevation) % 360
        self.roll = (self.roll + d_roll) % 360

    def set_matrix(self, R: np.ndarray) -> None:
        self.azimuth, self.elevation, self.roll = euler_zxz(np.asarray(R, float))

    def rotate_view(self, about_vertical: float, about_horizontal: float) -> None:
        """A mouse drag (degrees).  With ``fix_roll`` (turntable) a horizontal
        drag turns about the data's z axis and a vertical one tilts that axis
        forward or back, and the horizon stays level.  Otherwise a trackball:
        the drag turns about the screen's own axes, continuous, no pole."""
        if self.fix_roll:
            self.roll = 0.0
            self.azimuth = (self.azimuth + about_vertical) % 360
            self.elevation = (self.elevation + about_horizontal) % 360
            return
        self.set_matrix(_ry(about_vertical) @ _rx(about_horizontal) @ self.matrix)

    def move_pivot(self, pivot) -> None:
        """A new centre of rotation, without the image moving."""
        pivot = np.asarray(pivot, float).reshape(3)
        shift = self.matrix @ (self.pivot - pivot)
        self.offset = self.offset + shift[:2]
        self.pivot = pivot

    def preset(self, name: str, slab_angle: float = 0.0) -> None:
        """Top / front / side of the *slab*: its long axis is the screen's x
        in the front view, its depth in the side view."""
        az, el, roll = PRESETS[name]
        self.azimuth, self.elevation, self.roll = (az - slab_angle) % 360, el, roll

    def fov(self, nx: int, ny: int, scale: int = 1) -> FieldOfView:
        """The render grid for a canvas of ``nx`` x ``ny`` pixels, view centred."""
        px = self.zoom * scale
        nx, ny = max(nx // scale, 1), max(ny // scale, 1)
        return FieldOfView(self.offset[0] - nx * px / 2, self.offset[1] - ny * px / 2,
                           px, nx, ny)

    def fit(self, slab: Slab, nx: int, ny: int, margin: float = 1.05) -> None:
        """Centre on the slab and zoom so it fits the canvas."""
        self.pivot = slab.center.copy()
        self.offset[:] = 0
        c = slab.corners()
        xv, yv, _ = self.apply(c[:, 0], c[:, 1], c[:, 2])
        w, h = np.ptp(xv), np.ptp(yv)
        self.zoom = max(w / nx, h / ny, 1e-3) * margin

    def to_dict(self) -> dict:
        return {"azimuth": self.azimuth, "elevation": self.elevation, "roll": self.roll,
                "pivot": self.pivot.tolist(), "zoom": self.zoom,
                "offset": self.offset.tolist(), "focal": self.focal,
                "depth_lambda": self.depth_lambda}


# --------------------------------------------------------------- engine A
def _common_scale(axes: RenderAxes, locs: Localizations) -> Optional[float]:
    """The one scale a rotated view can use, or None if the axes disagree.

    A projection mixes the axes, so it has room for one width only.  That
    still holds when every axis is a position at the same scale -- the picture
    is then the usual one, stretched -- and not when one of them is a photon
    count.
    """
    names = list(axes.names(locs))
    depth = axes.depth_name(locs)
    scales = [axes.x_scale, axes.y_scale]
    if depth is not None:
        names.append(depth)
        scales.append(axes.z_scale)
    if all(is_position(n) for n in names) and len(set(scales)) == 1:
        return scales[0]
    return None


def projected_settings(locs: Localizations, settings: RenderSettings
                       ) -> Tuple[RenderSettings, float]:
    """``settings`` as they apply to a rotated view, and the precision scale.

    The coordinates a 3D engine is handed are already in render units, so the
    settings that go with them are on the *default* axes.  What the mapping
    leaves behind is the width:

    * the usual axes change nothing at all;
    * positions at one common scale keep the localization precision, with the
      scale folded into `SigmaSettings` (sigma = precision x factor, so a
      factor divided by the scale is the precision divided by it) and returned
      for the median the GPU caps with;
    * anything else -- photons against frame -- cannot keep a width per axis
      once they are mixed, so the two explicit widths are averaged into one.
    """
    axes = settings.axes
    if axes.is_default:
        return settings, 1.0
    scale = _common_scale(axes, locs)
    if scale is not None and settings.mode == "precision":
        ss = settings.sigma_settings
        return replace(settings, axes=RenderAxes(),
                       sigma_settings=replace(ss, factor=ss.factor / scale,
                                              min_sigma=ss.min_sigma / scale)), scale
    sigma = 0.5 * sum(explicit_sigmas(locs, settings))
    return replace(settings, axes=RenderAxes(), sigma=sigma,
                   mode="gauss" if sigma > 0 else "hist"), 1.0


def slab_candidates(locs: Localizations, select: np.ndarray, slab: Optional[Slab],
                    index=None) -> np.ndarray:
    """The rows worth looking at: the filter's, narrowed to the slab's footprint.

    Without ``index`` this is every row the filter keeps, and finding the few
    thousand inside a small slab then costs a pass over the whole table -- on
    a 46 M localization file, half a second per frame that a crop does not
    reduce, which is what made the 3D view slow however little of it was
    shown.  The index (`smappy.spatial`, the same one the 2D view queries)
    answers the slab's *bounding rectangle* in time proportional to the
    answer; the slab's own mask still decides, so this only avoids reading
    rows that cannot be inside it.

    ``index`` must be over the same table as ``locs``; None is always correct,
    only slower.  A non-boolean ``select`` is already a list of rows, and is
    passed through rather than intersected.
    """
    if select.dtype != bool:
        return np.asarray(select)
    if index is not None and slab is not None:
        c = slab.corners()
        cand = index.query(float(c[:, 0].min()), float(c[:, 0].max()),
                           float(c[:, 1].min()), float(c[:, 1].max()))
        if cand.size < index.n_localizations:      # a crop: the index earned its keep
            # not sorted back into table order: sorting 3 M indices costs more
            # than the tidier gather saves, and the 2D view queries the same
            # index the same way.  Cell order is piecewise ascending anyway.
            return cand[select[cand]]
    return np.flatnonzero(select)


_rows_cache: dict = {}


def _slab_key(slab: Optional[Slab]):
    """A hashable slab: a cache on it survives a rotation, not a moved box."""
    return None if slab is None else (tuple(slab.center), tuple(slab.size), slab.angle)


def selection_rows(locs: Localizations, select: np.ndarray, slab: Optional[Slab],
                   index=None) -> np.ndarray:
    """`slab_candidates`, memoised on the filter and the slab.

    A rotation changes neither -- only the projection -- so every frame of a
    drag asks the same question, and each engine asks it more than once (the
    render and the depth histogram both need these rows).  Answering it costs
    an index query, or `np.flatnonzero` over the whole table when the slab is
    everything (31 ms at 35 M).

    On the GPU this also keeps the *buffer*: `GPUEngine.selection` is keyed on
    the identity of the array handed to it, so returning the same array means
    the selection is uploaded once per slab rather than once per frame.

    The array is shared -- callers index it, and must not write to it.
    """
    key = _table_key(locs, id(select), _slab_key(slab))
    hit = _rows_cache.get(key)
    if hit is not None and hit[1] is select:
        return hit[0]
    rows = slab_candidates(locs, select, slab, index)
    if len(_rows_cache) > 4:
        _rows_cache.clear()
    # the mask is kept beside the rows so its id cannot be reused meanwhile
    _rows_cache[key] = (rows, select)
    return rows


def thinned(rows: np.ndarray, budget: Optional[int]) -> np.ndarray:
    """``budget`` of these rows, evenly spaced; all of them if they already fit.

    Evenly spaced rather than drawn at random.  A draw is what you would reach
    for, but `Generator.choice(n, k, replace=False)` permutes n first (73 ms at
    35 M) and comes back unsorted, so the gather that follows it scatters over
    the table.  `linspace` costs the k it returns, comes back ascending, and is
    deterministic -- so the same budget asks the GPU for the same selection and
    the buffer is reused rather than re-uploaded.

    It is a systematic sample, which is only as good as the row order is
    unrelated to position.  Localization tables are in acquisition order, and
    grouping renumbers but does not sort by position, so it holds: measured on
    a 3 M row file against a random draw of the same size, the sampled density
    over a 64 x 64 grid differed by L1 0.2305 against the draw's 0.2349, and
    both correlate with the whole table at 0.95.  A reader that ever returns a
    spatially sorted table would break this, and want a shuffle instead.

    Thinning is unbiased: every pixel keeps the same fraction, so a preview is
    what a shorter acquisition would have looked like -- noisier, never
    rearranged.  Merging neighbours instead would hold the total but move it,
    which can make a sparse filament look continuous; see NOTES.md.
    """
    if budget is None or rows.size <= budget or budget < 1:
        return rows
    return rows[np.linspace(0, rows.size - 1, int(budget)).astype(np.int64)]


@dataclass
class PreviewBudget:
    """How many localizations a drag frame may draw, learned from the clock.

    One number cannot serve a base M1 and a 3090 -- and the same machine is an
    order of magnitude apart between a sub-pixel blob zoomed out and a fat one
    zoomed in -- so the budget is what this machine actually got through, at
    this zoom, in ``1 / TARGET_FPS`` seconds.  `record` is fed by `render_3d`.

    Two bounds, and the smaller wins:

    * **what can be seen.**  ``per_pixel`` points per rendered pixel; past that
      they land on each other.  This one applies from the first frame, before
      anything has been timed, and is what stops the opening frame of a whole
      field being the 18 M points it used to be.
    * **what there is time for.**  ``rate`` is points per second, an EMA over
      recent drag frames, so a slower machine simply draws fewer.
    """
    per_pixel: float = PREVIEW_PER_PIXEL
    target: float = 1.0 / TARGET_FPS
    minimum: int = 50_000            # below this, thin no further: it is cheap
    smoothing: float = 0.35          # EMA weight on the newest frame
    rate: Optional[float] = None     # points per second, None until timed

    def points(self, fov: FieldOfView) -> int:
        """The budget for a frame on this grid."""
        seen = self.per_pixel * fov.nx * fov.ny
        timed = self.rate * self.target if self.rate else seen
        return max(self.minimum, int(min(seen, timed)))

    def record(self, drawn: int, seconds: float) -> None:
        """What the last drag frame managed, to size the next one."""
        if drawn <= 0 or seconds <= 0:
            return
        rate = drawn / seconds
        self.rate = rate if self.rate is None else \
            self.smoothing * rate + (1.0 - self.smoothing) * self.rate


def depth_sample(locs: Localizations, select: np.ndarray, projection: Projection,
                 slab: Optional[Slab], index=None, limit: int = HIST_POINTS,
                 axes: Optional[RenderAxes] = None) -> np.ndarray:
    """Depths of up to ``limit`` of the slab's localizations.

    Only the depth *histogram* wants this, and that histogram is a shape --
    the panel hides its count axis -- so a sample says the same thing as every
    row.  It used to be a second full `project_layer` per layer per frame,
    which is the projection the render had just done and thrown away; on a
    46 M localization file that was a second of every frame, and because it
    runs on the CPU whatever `Projection.engine` says, the GPU engines paid it
    too.  Sampling the candidate *rows* before touching a coordinate keeps it
    at ``limit`` however large the table.
    """
    axes = axes or RenderAxes()
    x, y = axes.coordinates(locs)
    z = axes.depth(locs)
    idx = selection_rows(locs, select, slab, index)
    if index is None and slab is not None and idx.size:
        # With an index the candidates are already the slab's own neighbourhood,
        # so thinning them first keeps the sample inside it.  Without one they
        # are the whole table, and a stride over that would land almost
        # entirely outside a small slab -- so the slab has to be applied first,
        # at the cost of the full pass the index exists to avoid.
        idx = idx[slab.mask(x[idx], y[idx], None if z is None else z[idx])]
    if idx.size > limit:
        # a stride, not a draw: `choice(n, k, replace=False)` permutes n first,
        # which is the cost this is here to avoid.  Rows are in acquisition
        # order, which is not an order in depth.
        idx = idx[::max(1, idx.size // limit)]
    if not idx.size:
        return np.empty(0, np.float32)
    xs, ys = x[idx], y[idx]
    zs = None if z is None else z[idx]
    if slab is not None:
        keep = slab.mask(xs, ys, zs)
        xs, ys = xs[keep], ys[keep]
        zs = None if zs is None else zs[keep]
    return projection.apply(xs, ys, zs)[2]


def depth_histogram(depths: Sequence[np.ndarray]) -> np.ndarray:
    """``(HIST_BINS, 2)`` of bin centre and count, over every layer's depths."""
    hist = np.zeros((HIST_BINS, 2), np.float64)
    d = np.concatenate(list(depths)) if depths else np.empty(0, np.float32)
    if d.size:
        counts, edges = np.histogram(d, bins=HIST_BINS)
        hist[:, 0], hist[:, 1] = (edges[:-1] + edges[1:]) / 2, counts
    return hist


def project_layer(locs: Localizations, select: np.ndarray, projection: Projection,
                  slab: Optional[Slab], settings: RenderSettings,
                  preview: bool = False, front: Optional[float] = None,
                  index=None, budget: Optional[int] = None
                  ) -> Tuple[Localizations, np.ndarray]:
    """The selected rows of a table, rotated into a 2D table for the renderer.

    Returns the table (``x_nm``/``y_nm`` are the view coordinates, ``depth``
    a column, the precision and any colour/weight column carried over) and
    the indices it was built from.  ``index`` is the layer's spatial index,
    which narrows the rows the slab is tested against (`slab_candidates`);
    ``budget`` caps how many are drawn (`thinned`), and defaults on a preview
    to `PREVIEW_POINTS` so an unbudgeted caller behaves as it always did.
    """
    if preview and budget is None:
        budget = PREVIEW_POINTS
    x, y = settings.axes.coordinates(locs)
    z = settings.axes.depth(locs)
    # thinned before the mask and the gather, not after: the budget is there to
    # bound the work, and the rows are already the slab's neighbourhood
    idx = thinned(selection_rows(locs, select, slab, index), budget)
    if slab is not None and idx.size:
        keep = slab.mask(x[idx], y[idx], None if z is None else z[idx])
        idx = idx[keep]
    xv, yv, depth = projection.apply(x[idx], y[idx], None if z is None else z[idx])
    columns = {"x_nm": xv.astype(np.float32), "y_nm": yv.astype(np.float32),
               "depth": depth.astype(np.float32)}
    for name in ("loc_precision_nm", "loc_precision_pix", settings.color_field,
                 settings.weight_field):
        if name and name in locs and name not in columns:
            columns[name] = np.asarray(locs[name])[idx]
    if projection.depth_lambda:
        # dimmer towards the back; the front of the slab is at full weight
        if front is None:
            front = depth.max() if depth.size else 0.0
        attenuation = np.exp(-(front - depth) / projection.depth_lambda).astype(np.float32)
        base = columns.get(settings.weight_field) if settings.weight_field else None
        columns["_weight"] = attenuation if base is None else attenuation * np.asarray(base, np.float32)
    return Localizations(columns, {"units": "nm"}), idx


def column_range(locs: Localizations, name: str) -> Tuple[float, float]:
    values = np.asarray(locs[name], np.float32)
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)


def render_layer_3d(locs: Localizations, select: np.ndarray, projection: Projection,
                    slab: Optional[Slab], fov: FieldOfView, settings: RenderSettings,
                    display: DisplaySettings, preview: bool = False,
                    n_threads: int = 0, index=None,
                    budget: Optional[int] = None) -> Tuple[np.ndarray, RenderedImage]:
    """Engine A for one layer: RGB in [0, 1] and the planes.

    With ``projection.opacity`` > 0 the slab is rendered in depth slices and
    composited front to back on the accumulated planes: each slice hides
    what is behind it by ``opacity`` times its own coverage, where coverage
    is the slice's intensity on the scale the plain image would be shown at.
    At 0 this is exactly the plain sum.
    """
    if projection.color_by_depth:
        settings = replace(settings, color_field="depth", color_range=None)
    elif settings.color_field and settings.color_range is None and settings.color_field in locs:
        # one colour scale for the whole table, whatever the slab, slice or
        # preview holds -- as the GPU engine has it
        settings = replace(settings, color_range=column_range(locs, settings.color_field))
    # depth is defined by the slab's corners (or the table's box), so that the
    # colour scale, the slices and the attenuation do not move with the filter
    front, drange = _depth_front_and_range(projection, slab, locs, settings.axes)
    table, _ = project_layer(locs, select, projection, slab, settings, preview, front,
                             index, budget)
    # the projected table is in render units on plain x/y: the mapping has
    # already been applied, and only the width still has to follow it
    settings, _ = projected_settings(locs, settings)
    weight = "_weight" if "_weight" in table else settings.weight_field
    if settings.color_field == "depth":
        settings = replace(settings, color_range=settings.color_range or drange)
    settings = replace(settings, weight_field=weight)
    if projection.opacity <= 0 or projection.slices < 2 or len(table) < 2:
        rendered = render_locs(table, fov, settings, display, n_threads=n_threads)
        return display.apply(rendered, white_background=False), rendered
    rendered = composite_slices(table, fov, settings, display, projection.opacity,
                                max(2, projection.slices // (2 if preview else 1)), n_threads,
                                drange)
    return display.apply(rendered, white_background=False), rendered


def composite_slices(table: Localizations, fov: FieldOfView, settings: RenderSettings,
                     display: DisplaySettings, opacity: float, slices: int,
                     n_threads: int = 0, depth_range=None) -> RenderedImage:
    """Front-to-back compositing of depth slices on the linear planes (CPU)."""
    depth = np.asarray(table["depth"])
    whole = render_locs(table, fov, settings, display, n_threads=n_threads)
    lo, hi = depth_range or (depth.min(), depth.max())
    edges = np.linspace(lo, hi, slices + 1)
    order = np.argsort(depth)
    bins = np.searchsorted(edges[1:-1], depth[order])
    starts = np.searchsorted(bins, np.arange(slices + 1))

    def render_slice(k: int) -> Optional[RenderedImage]:
        idx = order[starts[k]:starts[k + 1]]
        if idx.size == 0:
            return None
        return render_locs(table, fov, settings, display, select=idx, n_threads=n_threads)

    return composite_depth(whole, slices, render_slice, display, opacity)


def composite_depth(whole: RenderedImage, slices: int, render_slice, display: DisplaySettings,
                    opacity: float) -> RenderedImage:
    """Painter's order: from the back (slice 0, smallest depth) to the front
    (largest depth, towards the viewer).  Each slice hides what is already
    there behind it by ``opacity`` x its coverage on the scale the plain image
    is shown at, and adds itself on top."""
    _, imax = normalize(whole.weight, display.imax, display.contrast)
    if imax <= 0:
        return whole
    fov = whole.fov
    weight = np.zeros(fov.shape, np.float32)
    color = np.zeros((*fov.shape, 3), np.float32) if whole.is_colored else None
    for k in range(slices):
        part = render_slice(k)
        if part is None:
            continue
        cover = np.clip(part.weight / imax, 0.0, 1.0) * opacity
        weight = weight * (1.0 - cover) + part.weight
        if color is not None and part.color is not None:
            color = color * (1.0 - cover)[..., None] + part.color
    return RenderedImage(fov, weight, color, n_locs=whole.n_locs)


# ------------------------------------------------------------ engine: GPU
def _depth_front_and_range(projection: Projection, slab: Optional[Slab],
                           locs: Localizations, axes: Optional[RenderAxes] = None):
    """The depth range of the slab's corners (or the table's bounding box)."""
    if slab is not None:
        c = slab.corners()
    else:
        axes = axes or RenderAxes()
        x, y = axes.coordinates(locs)
        z = axes.depth(locs)
        z = np.zeros(1) if z is None else z
        lo = [np.nanmin(x), np.nanmin(y), np.nanmin(z)]
        hi = [np.nanmax(x), np.nanmax(y), np.nanmax(z)]
        c = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])
                      for d in (lo[2], hi[2])])
    _, _, d = projection.apply(c[:, 0], c[:, 1], c[:, 2])
    return float(d.max()), (float(d.min()), float(d.max()))


def _table_key(locs: Localizations, *extra):
    """A key that dies with the table: an object stored on it, not a bare id."""
    token = locs.metadata.get("_gpu_token")
    if token is None:
        token = object()
        locs.metadata["_gpu_token"] = token
    return (id(token), len(locs)) + extra


_median_cache: dict = {}


def point_radius_nm(locs: Localizations, select: np.ndarray, prec_name: Optional[str],
                    projection: Projection) -> float:
    """The sprite radius: what was asked, or the shown points' median precision."""
    if projection.point_size > 0:
        return projection.point_size
    if prec_name is None:
        return 10.0
    key = _table_key(locs, id(select), prec_name)
    if key not in _median_cache:
        values = np.asarray(locs[prec_name], np.float32)
        values = values[select] if select.dtype == bool else values[np.asarray(select)]
        values = values[np.isfinite(values)]
        if len(_median_cache) > 8:
            _median_cache.clear()
        # the mask is kept with the value so its id cannot be reused meanwhile
        _median_cache[key] = (float(np.median(values)) if values.size else 10.0, select)
    return _median_cache[key][0]


def sphere_draw(engine, locs: Localizations, select: np.ndarray, projection: Projection,
                slab: Optional[Slab], fov: FieldOfView, settings: RenderSettings,
                display: DisplaySettings, index=None):
    """One layer's contribution to `GPUEngine.render_spheres`."""
    axes = settings.axes
    x, y = axes.coordinates(locs)
    z = axes.depth(locs)
    settings, _ = projected_settings(locs, settings)
    prec_name = next((n for n in ("loc_precision_nm", "loc_precision_pix") if n in locs), None)
    color_field = "depth" if projection.color_by_depth else settings.color_field
    key = _table_key(locs, prec_name, settings.weight_field, color_field)
    cvalues = (locs[color_field] if color_field and color_field != "depth" and color_field in locs
               else None)
    engine.table(key, x, y, z, locs[prec_name] if prec_name else None,
                 locs[settings.weight_field] if settings.weight_field else None, cvalues)
    idx = selection_rows(locs, select, slab, index)
    sel = engine.selection((key, id(select), _slab_key(slab)), idx)
    front, drange = _depth_front_and_range(projection, slab, locs, axes)
    if color_field == "depth":
        color_mode, color_range = 2, (settings.color_range or drange)
    elif cvalues is not None:
        color_mode, color_range = 1, (settings.color_range or column_range(locs, color_field))
    else:
        color_mode, color_range = 0, (0.0, 1.0)
    radius_nm = point_radius_nm(locs, select, prec_name, projection)
    radius_px = radius_nm / fov.pixelsize
    # How far the occlusion looks, by default.  Three sphere radii is the
    # contact shadow of one sphere against the next -- a few nanometres, which
    # at the zoom a whole box is looked at is a percent or two of the pixel it
    # lands in, so both occlusion controls read as doing nothing.  The cavities
    # worth seeing are the cloud's, not one sphere's, so the default reaches a
    # twentieth of the box as well.
    ssao_nm = projection.ssao_radius
    if not ssao_nm:
        ssao_nm = 3.0 * radius_nm
        if slab is not None:
            ssao_nm = max(ssao_nm, 0.05 * float(np.min(slab.size)))
    pad = radius_nm * 2
    params = engine.params(
        fov=fov, matrix=projection.matrix, pivot=projection.pivot, focal=projection.focal,
        slab=slab, sigma_mode=settings.mode, sigma=settings.sigma, use_weight=False,
        depth_lambda=projection.depth_lambda, depth_front=front, color_mode=color_mode,
        color_range=color_range, n=sel[1], radius=radius_px,
        depth_range=(drange[0] - pad, drange[1] + pad),
        ssao_radius=ssao_nm / fov.pixelsize, ssao_strength=projection.ssao_strength)
    return (key, sel, params, display.lut, display.invert), params


def render_layer_gpu(engine, locs: Localizations, select: np.ndarray, projection: Projection,
                     slab: Optional[Slab], fov: FieldOfView, settings: RenderSettings,
                     display: DisplaySettings, median_precision: float = 0.0,
                     preview: bool = False, index=None, budget: Optional[int] = None):
    """Engine A on the GPU for one layer: the same planes as `render_layer_3d`,
    or, with ``projection.engine == "points"``, an RGB sprite image."""
    axes = settings.axes
    x, y = axes.coordinates(locs)
    z = axes.depth(locs)
    settings, precision_scale = projected_settings(locs, settings)
    median_precision = median_precision / precision_scale
    prec_name = next((n for n in ("loc_precision_nm", "loc_precision_pix") if n in locs), None)
    color_field = "depth" if projection.color_by_depth else settings.color_field
    key = _table_key(locs, prec_name, settings.weight_field, color_field)
    cvalues = (locs[color_field] if color_field and color_field != "depth" and color_field in locs
               else None)
    engine.table(key, x, y, z, locs[prec_name] if prec_name else None,
                 locs[settings.weight_field] if settings.weight_field else None, cvalues)
    sel_key = (key, id(select), _slab_key(slab), budget)
    front, drange = _depth_front_and_range(projection, slab, locs, axes)
    if color_field == "depth":
        color_mode, color_range = 2, (settings.color_range or drange)
    elif cvalues is not None:
        color_mode = 1
        color_range = settings.color_range or column_range(locs, color_field)
    else:
        color_mode, color_range = 0, (0.0, 1.0)
    ss = settings.sigma_settings
    floor = max(ss.min_sigma, ss.min_sigma_pixels * fov.pixelsize)
    cap = ss.max_factor * median_precision if median_precision > 0 else 1e30
    base = dict(fov=fov, matrix=projection.matrix, pivot=projection.pivot, focal=projection.focal,
                slab=slab, sigma_mode=settings.mode, sigma=settings.sigma, factor=ss.factor,
                floor=floor, cap=cap, use_weight=settings.weight_field is not None,
                depth_lambda=projection.depth_lambda, depth_front=front,
                color_mode=color_mode, color_range=color_range,
                # the radius is in nm, so it scales with the zoom (and is the
                # same on the coarser preview grid); 0 = the median precision
                radius=point_radius_nm(locs, select, prec_name, projection) / fov.pixelsize,
                alpha=projection.point_alpha)
    lut, invert = display.lut, display.invert
    idx = thinned(selection_rows(locs, select, slab, index), budget)
    if projection.engine == "points":
        if not preview and idx.size <= PREVIEW_POINTS:
            # Back to front, so the alpha reads right -- but not while the mouse
            # is down: the order depends on the angle, so a drag would re-sort
            # and re-upload every frame, which is why a small selection used to
            # rotate *slower* than a large one (8 fps at 1 M, 45 at 2 M).
            _, _, d = projection.apply(x[idx], y[idx], None if z is None else z[idx])
            idx = idx[np.argsort(d)]
            sel = engine.selection((sel_key, "sorted", projection.azimuth, projection.elevation,
                                    projection.roll), idx)
        else:
            sel = engine.selection(sel_key, idx)
        params = engine.params(n=sel[1], **base)
        return engine.render_points(key, sel, fov, params, lut, invert), None
    sel = engine.selection(sel_key, idx)
    colored = color_mode > 0
    whole = engine.render_planes(key, sel, fov, engine.params(n=sel[1], **base), lut, invert,
                                 colored)
    if projection.opacity > 0 and projection.slices >= 2:
        slices = max(2, projection.slices // (2 if preview else 1))
        edges = np.linspace(drange[0], drange[1], slices + 1)

        def render_slice(k: int):
            params = engine.params(n=sel[1], depth_clip=(edges[k], edges[k + 1]), **base)
            return engine.render_planes(key, sel, fov, params, lut, invert, colored)

        whole = composite_depth(whole, slices, render_slice, display, projection.opacity)
    return display.apply(whole, white_background=False), whole


def render_3d(layers, projection: Projection, slab: Optional[Slab], fov: FieldOfView,
              preview: bool = False, engine=None,
              budget: Optional[PreviewBudget] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Every visible localization layer, added up; also the depth histogram.

    ``layers`` are session layers.  ``engine`` is a `smappy.gpu.GPUEngine`
    for ``projection.engine`` "gpu" or "points"; None means the CPU.
    Returns the RGB image and a (64, 2) array of depth-bin centres and
    counts over the slab's points.

    ``budget`` bounds a *preview* frame and is timed by it, so a drag settles
    at `TARGET_FPS` on whatever machine it is running on; None draws every row
    the slab holds.  A final frame is never budgeted -- the image you measure
    or export is the whole table, and only the drag is a sample.
    """
    rgb = np.zeros((fov.ny, fov.nx, 3), np.float32)
    depths: List[np.ndarray] = []
    limit = budget.points(fov) if (preview and budget is not None) else None
    started, drawn = time.perf_counter(), 0
    # A final frame is timed too, though it is never thinned: it gives the first
    # drag a rate to start from instead of the visible cap, which on a whole
    # field is one slow frame.  Its grid is PREVIEW_SCALE times finer, so the
    # rate it reports is pessimistic -- the safe direction to be wrong in.
    if engine is not None and projection.engine == "spheres":
        draws, shade_params = [], None
        for layer in layers:
            if layer.visible and not layer.is_image:
                st = layer.state
                draw, shade_params = sphere_draw(engine, st.locs, st.filter.mask, projection, slab,
                                                 fov, st.settings, st.display,
                                                 index=getattr(st, "cull_index", None))  # opaque: no budget
                draws.append(draw)
        if draws:
            rgb = engine.render_spheres(draws, fov, shade_params)
    for layer in layers:
        if not layer.visible or layer.is_image:
            continue
        state = layer.state
        # None on custom axes: the index is in the table's own coordinates
        index = getattr(state, "cull_index", None)
        # memoised, so this is the count the engines are about to work on
        drawn += min(selection_rows(state.locs, state.filter.mask, slab, index).size,
                     limit if limit is not None else np.iinfo(np.int64).max)
        if engine is not None and projection.engine == "spheres":
            image = 0.0
        elif engine is not None and projection.engine in ("gpu", "points"):
            image, _ = render_layer_gpu(engine, state.locs, state.filter.mask, projection, slab,
                                        fov, state.settings, state.display,
                                        state.current.median_precision, preview, index=index,
                                        budget=limit)
        else:
            image, _ = render_layer_3d(state.locs, state.filter.mask, projection, slab, fov,
                                       state.settings, state.display, preview,
                                       n_threads=state.n_threads, index=index, budget=limit)
        rgb += image
        depths.append(depth_sample(state.locs, state.filter.mask, projection, slab, index,
                                   axes=state.settings.axes))
    rgb = np.clip(rgb, 0, 1)
    if any(l.visible and not l.is_image and l.get_display().white_background
           for l in layers):
        from .lut import on_white       # once, to the sum, as the 2D view does
        rgb = on_white(rgb)
    out = rgb, depth_histogram(depths)
    if budget is not None:
        budget.record(drawn, time.perf_counter() - started)
    return out


def upscale(rgb: np.ndarray, scale: int, ny: int, nx: int) -> np.ndarray:
    """A preview rendered ``scale`` times coarser, blown up to the canvas."""
    if scale == 1:
        return rgb
    big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    out = np.zeros((ny, nx, 3), np.float32)
    h, w = min(ny, big.shape[0]), min(nx, big.shape[1])
    out[:h, :w] = big[:h, :w]
    return out
