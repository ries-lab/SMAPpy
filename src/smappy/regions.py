"""Regions of interest: a shape in the coordinates of the table, and its mask.

A `Region` is what a viewer draws, what a plugin's `Selection` carries and
what an ROI of the ROI manager is the shape of; it knows nothing about
widgets.  A line is a rectangle of a given width around the segment, as in
SMAP, and is stored as its four corners.  A circle is its centre and radius.

One shape type for both, rather than the session's drawn region and the ROI
manager's ROI each with their own geometry: `docs/regions.md` has the
reasons -- a measurement reads one ``ctx.region``, however the region came
to be -- and this is the first step of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# the points of a drawn circle: enough to look round at any zoom the ROI
# manager offers, few enough to redraw while one is dragged
CIRCLE_POINTS = 65


@dataclass
class Region:
    kind: str                               # "rect", "polygon", "line" or "circle"
    points: np.ndarray                      # (n, 2): rect corners, vertices, or the centre
    width: float = 0.0                      # line only, in data units
    radius: float = 0.0                     # circle only, in data units

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)

    @classmethod
    def rect(cls, x0, y0, x1, y1) -> "Region":
        return cls("rect", [[min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]])

    @classmethod
    def line(cls, p0, p1, width: float) -> "Region":
        """The rectangle of ``width`` around the segment ``p0``-``p1``."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1.0) * (width / 2)
        return cls("line", [p0 + n, p1 + n, p1 - n, p0 - n], width=float(width))

    @classmethod
    def polygon(cls, vertices) -> "Region":
        return cls("polygon", vertices)

    @classmethod
    def circle(cls, center, radius: float) -> "Region":
        return cls("circle", [center], radius=float(radius))

    @classmethod
    def square(cls, center, size: float) -> "Region":
        """The square of side ``size`` centred on ``center``."""
        cx, cy = np.asarray(center, float)
        h = size / 2
        return cls.rect(cx - h, cy - h, cx + h, cy + h)

    @classmethod
    def site(cls, center, shape: str, size: float, polygon=None) -> "Region":
        """An ROI manager ROI: its own polygon, or the project's shape -- a
        circle of diameter ``size`` or a square of side ``size`` -- around its
        centre."""
        if polygon is not None:
            return cls.polygon(polygon)
        if shape == "circle":
            return cls.circle(center, size / 2)
        return cls.square(center, size)

    @property
    def bounds(self):
        if self.kind == "circle":
            (cx, cy), r = self.points[0], self.radius
            return cx - r, cy - r, cx + r, cy + r
        (x0, y0), (x1, y1) = self.points.min(axis=0), self.points.max(axis=0)
        return x0, y0, x1, y1

    @property
    def center(self) -> np.ndarray:
        """The centre of the bounding box -- a circle's and a square's own."""
        x0, y0, x1, y1 = self.bounds
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def vertices(self) -> np.ndarray:
        """The outline as polygon vertices, not closed: a rect's four
        corners, a circle as `CIRCLE_POINTS` - 1 points round it."""
        if self.kind == "rect":
            x0, y0, x1, y1 = self.bounds
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if self.kind == "circle":
            t = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)[:-1]
            (cx, cy), r = self.points[0], self.radius
            return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))
        return self.points.copy()

    def outline(self) -> Tuple[np.ndarray, np.ndarray]:
        """The closed outline, ``(xs, ys)``, for drawing."""
        if self.kind == "circle":
            t = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
            (cx, cy), r = self.points[0], self.radius
            return cx + r * np.cos(t), cy + r * np.sin(t)
        v = self.vertices()
        return np.r_[v[:, 0], v[0, 0]], np.r_[v[:, 1], v[0, 1]]

    def mask(self, x, y) -> np.ndarray:
        """Inside, with the outline counted as inside."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.kind == "rect":
            x0, y0, x1, y1 = self.bounds
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if self.kind == "circle":
            (cx, cy), r = self.points[0], self.radius
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        return inside_polygon(x, y, self.points)

    def rim_distance(self, x: float, y: float) -> float:
        """Distance from a point to the outline, in data units.

        The ROI manager selects by this rather than by "inside", so that a
        click in the middle of an ROI is free to start a new, overlapping one.
        """
        p = np.array([x, y], float)
        if self.kind == "circle":
            return abs(float(np.hypot(*(p - self.points[0]))) - self.radius)
        if self.kind == "rect":
            q = np.abs(p - self.center) - (np.array(self.bounds[2:]) - self.bounds[:2]) / 2
            inside = min(max(q[0], q[1]), 0.0)       # signed distance to the box
            return abs(float(np.linalg.norm(np.maximum(q, 0.0)) + inside))
        v = self.points
        a, b = v, np.roll(v, -1, axis=0)
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum((ab ** 2).sum(1), 1e-12),
                    0, 1)
        return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))

    def to_dict(self) -> Dict:
        d = {"kind": self.kind, "points": self.points.tolist(), "width": self.width}
        if self.kind == "circle":
            d["radius"] = self.radius
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Region":
        return cls(d["kind"], d["points"], d.get("width", 0.0), d.get("radius", 0.0))

    def __str__(self) -> str:
        x0, y0, x1, y1 = self.bounds
        return f"{self.kind} ROI {x1 - x0:.0f} x {y1 - y0:.0f}"


def inside_polygon(x, y, vertices) -> np.ndarray:
    """Even-odd fill with the boundary counted as inside.

    The one point-in-polygon test: the session's drawn regions and the ROI
    manager's polygons both come here.  The boundary is inside so that a
    localization exactly on a drawn edge -- a rect's, a polygon's -- is
    treated the same whichever kind drew it.  Only the points in the
    polygon's bounding box are tested; the polygon is small and the table
    is not.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.asarray(vertices, dtype=float)
    out = np.zeros(x.shape, dtype=bool)
    x0, y0 = v.min(axis=0)
    x1, y1 = v.max(axis=0)
    box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    xb, yb = x[box], y[box]
    inside = np.zeros(xb.shape, dtype=bool)
    boundary = inside.copy()
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        dx, dy = b - a
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        cross = (xb - a[0]) * dy - (yb - a[1]) * dx
        dot = (xb - a[0]) * dx + (yb - a[1]) * dy
        boundary |= ((np.abs(cross) <= 1e-7 * length) &
                     (dot >= -1e-7 * length) & (dot <= length**2 + 1e-7 * length))
        if dy != 0:
            inside ^= (((a[1] > yb) != (b[1] > yb)) &
                       (xb < a[0] + (yb - a[1]) * dx / dy))
    out[box] = inside | boundary
    return out
