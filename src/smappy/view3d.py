"""The 3D view without a window: a projection, a slab, and engine A.

A `Projection` turns data coordinates into view coordinates ``(x', y',
depth)``; a `Slab` is the box of data that is shown, the 3D ROI.  Engine A
rotates the slab's localizations and renders ``(x', y')`` with the 2D
kernel, so the picture is the 2D image of the tilted slab, with every
layer's own settings.  Nothing here imports Qt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .locs import Localizations
from .regions import Region
from .render import (DisplaySettings, FieldOfView, RenderSettings, RenderedImage,
                     positions, render_locs)

PREVIEW_POINTS = 2_000_000       # at most this many while the mouse drags
PREVIEW_SCALE = 2                # and at this many times coarser pixels


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

    def to_local(self, x, y, z) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Data -> slab frame: centred, in-plane rotation undone."""
        c, s = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        dx, dy = np.asarray(x, float) - self.center[0], np.asarray(y, float) - self.center[1]
        return c * dx + s * dy, -s * dx + c * dy, np.asarray(z, float) - self.center[2]

    def mask(self, x, y, z=None) -> np.ndarray:
        if z is None:
            z = np.zeros_like(np.asarray(x, float)) + self.center[2]
        u, v, w = self.to_local(x, y, z)
        h = self.size / 2
        return (np.abs(u) <= h[0]) & (np.abs(v) <= h[1]) & (np.abs(w) <= h[2])

    def corners(self) -> np.ndarray:
        """The 8 corners in data coordinates, (8, 3)."""
        h = self.size / 2
        local = np.array([[sx, sy, sz] for sx in (-h[0], h[0]) for sy in (-h[1], h[1])
                          for sz in (-h[2], h[2])])
        c, s = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return local @ rot.T + self.center

    def axis_range(self, axis: int) -> Tuple[float, float]:
        return (self.center[axis] - self.size[axis] / 2, self.center[axis] + self.size[axis] / 2)

    def set_axis_range(self, axis: int, lo: float, hi: float) -> None:
        lo, hi = sorted((float(lo), float(hi)))
        self.center[axis], self.size[axis] = (lo + hi) / 2, max(hi - lo, 1e-6)

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


@dataclass
class Projection:
    """Data -> view: ``v = R (p - pivot)``; ``(v_x, v_y)`` on screen, ``v_z``
    is depth (towards the viewer positive)."""
    azimuth: float = 0.0        # about the data z axis
    elevation: float = 0.0      # tilt about the view x axis
    roll: float = 0.0           # about the view depth axis
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zoom: float = 10.0          # nm per screen pixel
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))   # pan, view nm
    focal: Optional[float] = None   # perspective: None = orthographic
    depth_lambda: Optional[float] = None   # attenuation length; None = off

    def __post_init__(self):
        self.pivot = np.asarray(self.pivot, dtype=np.float64).reshape(3)
        self.offset = np.asarray(self.offset, dtype=np.float64).reshape(2)

    @property
    def matrix(self) -> np.ndarray:
        return _rz(self.roll) @ _rx(self.elevation) @ _rz(self.azimuth)

    def apply(self, x, y, z=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x', y', depth) for data points; z defaults to the pivot's."""
        x = np.asarray(x, np.float64)
        z = np.full_like(x, self.pivot[2]) if z is None else np.asarray(z, np.float64)
        p = np.column_stack([x - self.pivot[0], np.asarray(y, np.float64) - self.pivot[1],
                             z - self.pivot[2]])
        v = p @ self.matrix.T
        xv, yv, d = v[:, 0], v[:, 1], v[:, 2]
        if self.focal:
            s = 1.0 / np.maximum(1.0 - d / self.focal, 0.05)
            xv, yv = xv * s, yv * s
        return xv, yv, d

    def view_axis(self, axis: int) -> np.ndarray:
        """The data-space direction of view axis 0 (x'), 1 (y') or 2 (depth)."""
        return self.matrix[axis]

    def rotate_by(self, d_azimuth: float, d_elevation: float, d_roll: float = 0.0) -> None:
        self.azimuth = (self.azimuth + d_azimuth) % 360
        self.elevation = max(-90.0, min(90.0, self.elevation + d_elevation))
        self.roll = (self.roll + d_roll) % 360

    def preset(self, name: str) -> None:
        self.azimuth, self.elevation, self.roll = PRESETS[name]

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
def project_layer(locs: Localizations, select: np.ndarray, projection: Projection,
                  slab: Optional[Slab], settings: RenderSettings,
                  preview: bool = False) -> Tuple[Localizations, np.ndarray]:
    """The selected rows of a table, rotated into a 2D table for the renderer.

    Returns the table (``x_nm``/``y_nm`` are the view coordinates, ``depth``
    a column, the precision and any colour/weight column carried over) and
    the indices it was built from.
    """
    x, y = positions(locs)
    z = locs["z_nm"] if "z_nm" in locs else None
    idx = np.flatnonzero(select) if select.dtype == bool else np.asarray(select)
    if slab is not None and idx.size:
        keep = slab.mask(x[idx], y[idx], None if z is None else z[idx])
        idx = idx[keep]
    if preview and idx.size > PREVIEW_POINTS:
        idx = idx[np.random.default_rng(0).choice(idx.size, PREVIEW_POINTS, replace=False)]
        idx.sort()
    xv, yv, depth = projection.apply(x[idx], y[idx], None if z is None else z[idx])
    columns = {"x_nm": xv.astype(np.float32), "y_nm": yv.astype(np.float32),
               "depth": depth.astype(np.float32)}
    for name in ("loc_precision_nm", "loc_precision_pix", settings.color_field,
                 settings.weight_field):
        if name and name in locs and name not in columns:
            columns[name] = np.asarray(locs[name])[idx]
    if projection.depth_lambda:
        # dimmer towards the back; the front of the slab is at full weight
        front = depth.max() if depth.size else 0.0
        attenuation = np.exp(-(front - depth) / projection.depth_lambda).astype(np.float32)
        base = columns.get(settings.weight_field) if settings.weight_field else None
        columns["_weight"] = attenuation if base is None else attenuation * np.asarray(base, np.float32)
    return Localizations(columns, {"units": "nm"}), idx


def render_layer_3d(locs: Localizations, select: np.ndarray, projection: Projection,
                    slab: Optional[Slab], fov: FieldOfView, settings: RenderSettings,
                    display: DisplaySettings, preview: bool = False,
                    n_threads: int = 0) -> Tuple[np.ndarray, RenderedImage]:
    """Engine A for one layer: RGB in [0, 1] and the planes."""
    table, _ = project_layer(locs, select, projection, slab, settings, preview)
    weight = "_weight" if "_weight" in table else settings.weight_field
    color = settings.color_field
    if color == "depth":
        color_range = settings.color_range or (
            (float(table["depth"].min()), float(table["depth"].max())) if len(table) else None)
        settings = replace(settings, color_range=color_range)
    settings = replace(settings, weight_field=weight)
    rendered = render_locs(table, fov, settings, display, n_threads=n_threads)
    return display.apply(rendered), rendered


def upscale(rgb: np.ndarray, scale: int, ny: int, nx: int) -> np.ndarray:
    """A preview rendered ``scale`` times coarser, blown up to the canvas."""
    if scale == 1:
        return rgb
    big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    out = np.zeros((ny, nx, 3), np.float32)
    h, w = min(ny, big.shape[0]), min(nx, big.shape[1])
    out[:h, :w] = big[:h, :w]
    return out
