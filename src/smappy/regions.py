"""Regions of interest: a shape in the coordinates of the table, and its mask.

A `Region` is what a viewer draws, what a plugin's `Selection` carries and
what an ROI of the ROI manager is the shape of; it knows nothing about
widgets.  A line is a rectangle of a given width around the segment, as in
SMAP, and is stored as its four corners.  A circle is its centre and radius;
an ellipse its centre, its two semi-axes and its tilt, so that what it holds
is decided exactly rather than by a polygon round it.

One shape type for both, rather than the session's drawn region and the ROI
manager's ROI each with their own geometry: `docs/regions.md` has the
reasons -- a measurement reads one ``ctx.region``, however the region came
to be -- and this is the first step of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# the points of a drawn circle: enough to look round at any zoom the ROI
# manager offers, few enough to redraw while one is dragged
CIRCLE_POINTS = 65


@dataclass
class Region:
    kind: str                    # "rect", "polygon", "line", "circle" or "ellipse"
    points: np.ndarray                      # (n, 2): rect corners, vertices, or the centre
    width: float = 0.0                      # line only, in data units
    radius: float = 0.0                     # circle only, in data units
    # the point the region was placed at, when that is not the middle of its
    # outline: an ROI's centre, which a polygon ROI keeps wherever it was
    # put and which a measurement may want (NPC Corners counts round it)
    anchor: Optional[np.ndarray] = None
    # ellipse only: the semi-axes (a, b), in data units, and the tilt of the
    # first, in degrees anticlockwise from +x in data coordinates (y up) --
    # as a line's angle (`roi_manager.core.line_angle_length`)
    axes: Optional[Tuple[float, float]] = None
    tilt: float = 0.0

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        if self.anchor is not None:
            self.anchor = np.asarray(self.anchor, dtype=np.float64).reshape(2)
        if self.axes is not None:
            self.axes = (float(self.axes[0]), float(self.axes[1]))

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

    def ends(self) -> Tuple[np.ndarray, np.ndarray]:
        """A line's two ends, the middles of its short sides."""
        p = self.points
        return (p[0] + p[3]) / 2, (p[1] + p[2]) / 2

    def turned(self, degrees: float) -> "Region":
        """A line turned about its middle, anticlockwise in data coordinates
        (y up); its length and width kept."""
        if self.kind != "line":
            raise ValueError(f"only a line is turned, not a {self.kind}")
        p0, p1 = self.ends()
        mid, half = (p0 + p1) / 2, (p1 - p0) / 2
        a = np.radians(degrees)
        turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        half = turn @ half
        return Region.line(mid - half, mid + half, self.width)

    @classmethod
    def polygon(cls, vertices) -> "Region":
        return cls("polygon", vertices)

    @classmethod
    def circle(cls, center, radius: float) -> "Region":
        return cls("circle", [center], radius=float(radius))

    @classmethod
    def ellipse(cls, center, a: float, b: float, tilt: float = 0.0) -> "Region":
        """Semi-axes ``a`` along ``tilt`` degrees and ``b`` across it."""
        return cls("ellipse", [center], axes=(abs(float(a)), abs(float(b))),
                   tilt=float(tilt))

    def _ellipse_frame(self, x, y):
        """Points in the ellipse's own axes, in units of its semi-axes."""
        (cx, cy), (a, b) = self.points[0], self.axes
        t = np.radians(self.tilt)
        dx, dy = np.asarray(x, float) - cx, np.asarray(y, float) - cy
        return ((dx * np.cos(t) + dy * np.sin(t)) / max(a, 1e-12),
                (-dx * np.sin(t) + dy * np.cos(t)) / max(b, 1e-12))

    @classmethod
    def square(cls, center, size: float) -> "Region":
        """The square of side ``size`` centred on ``center``."""
        cx, cy = np.asarray(center, float)
        h = size / 2
        return cls.rect(cx - h, cy - h, cx + h, cy + h)

    @classmethod
    def site(cls, center, shape: str, size: float, polygon=None,
             angle: float = 0.0, ellipse=None) -> "Region":
        """An ROI manager ROI: its own polygon or ellipse, or the project's
        shape -- a circle of diameter ``size`` or a square of side ``size``
        -- around its centre.  ``angle`` (degrees) turns the square with the
        structure, so that it is upright in the ROI's own frame
        (`SiteFrame`): the ROI manager shows the ROI turned by -``angle``,
        and the square stays square on the screen while the localizations
        turn under it.  A polygon and an ellipse are drawn round
        localizations and are never turned by it.  ``ellipse`` is
        ``{"center", "axes", "tilt"}`` (`roi_manager.core.ROI.ellipse`)."""
        if polygon is not None:
            region = cls.polygon(polygon)
        elif ellipse is not None:
            region = cls.ellipse(ellipse["center"], *ellipse["axes"],
                                 ellipse.get("tilt", 0.0))
        elif shape == "circle":
            region = cls.circle(center, size / 2)
        elif angle % 360:
            h = size / 2
            corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
            region = cls.polygon(SiteFrame(center, angle).to_data(corners))
        else:
            region = cls.square(center, size)
        region.anchor = np.asarray(center, dtype=np.float64).reshape(2)
        return region

    @property
    def bounds(self):
        if self.kind == "circle":
            (cx, cy), r = self.points[0], self.radius
            return cx - r, cy - r, cx + r, cy + r
        if self.kind == "ellipse":
            (cx, cy), (a, b) = self.points[0], self.axes
            t = np.radians(self.tilt)
            hx = float(np.hypot(a * np.cos(t), b * np.sin(t)))
            hy = float(np.hypot(a * np.sin(t), b * np.cos(t)))
            return cx - hx, cy - hy, cx + hx, cy + hy
        (x0, y0), (x1, y1) = self.points.min(axis=0), self.points.max(axis=0)
        return x0, y0, x1, y1

    @property
    def center(self) -> np.ndarray:
        """Where the region was placed (`anchor`), else the middle of its
        bounding box -- a circle's and a square's own centre either way."""
        if self.anchor is not None:
            return self.anchor.copy()
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
        if self.kind == "ellipse":
            t = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)[:-1]
            return self._ellipse_points(t)
        return self.points.copy()

    def _ellipse_points(self, t) -> np.ndarray:
        (cx, cy), (a, b) = self.points[0], self.axes
        u, v = a * np.cos(t), b * np.sin(t)
        c, s = np.cos(np.radians(self.tilt)), np.sin(np.radians(self.tilt))
        return np.column_stack((cx + u * c - v * s, cy + u * s + v * c))

    def outline(self) -> Tuple[np.ndarray, np.ndarray]:
        """The closed outline, ``(xs, ys)``, for drawing."""
        if self.kind == "circle":
            t = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
            (cx, cy), r = self.points[0], self.radius
            return cx + r * np.cos(t), cy + r * np.sin(t)
        if self.kind == "ellipse":
            p = self._ellipse_points(np.linspace(0, 2 * np.pi, CIRCLE_POINTS))
            return p[:, 0], p[:, 1]
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
        if self.kind == "ellipse":
            u, v = self._ellipse_frame(x, y)
            return u ** 2 + v ** 2 <= 1.0
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
            x0, y0, x1, y1 = self.bounds
            middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            q = np.abs(p - middle) - np.array([x1 - x0, y1 - y0]) / 2
            inside = min(max(q[0], q[1]), 0.0)       # signed distance to the box
            return abs(float(np.linalg.norm(np.maximum(q, 0.0)) + inside))
        if self.kind == "ellipse":
            # to the drawn outline: near enough for a click on the rim
            return polyline_distance(self.vertices(), x, y, closed=True)
        return polyline_distance(self.points, x, y, closed=True)

    def to_dict(self) -> Dict:
        d = {"kind": self.kind, "points": self.points.tolist(), "width": self.width}
        if self.kind == "circle":
            d["radius"] = self.radius
        if self.kind == "ellipse":
            d["axes"] = list(self.axes)
            d["tilt"] = self.tilt
        if self.anchor is not None:
            d["anchor"] = self.anchor.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Region":
        return cls(d["kind"], d["points"], d.get("width", 0.0), d.get("radius", 0.0),
                   d.get("anchor"), d.get("axes"), d.get("tilt", 0.0))

    def __str__(self) -> str:
        if self.kind == "ellipse":
            a, b = self.axes
            return f"ellipse ROI {2 * a:.0f} x {2 * b:.0f}"
        x0, y0, x1, y1 = self.bounds
        return f"{self.kind} ROI {x1 - x0:.0f} x {y1 - y0:.0f}"


# the columns of an ROI's own frame (`SiteFrame`) that a measurement in an
# ROI gets beside ``x_nm`` / ``y_nm`` (`ROIProject._region_tables`)
SITE_X, SITE_Y = "x_site_nm", "y_site_nm"


@dataclass
class SiteFrame:
    """An ROI's own coordinates: positions relative to its ``center``,
    turned by ``-angle`` so that a structure lying at ``angle`` (degrees,
    ``atan2(dy, dx)`` as a line's, SMAP's ``pos2angle``) lies along +x.

    What the ROI manager shows when *rotate* is on, and what a measurement
    in an ROI gets as `SITE_X` / `SITE_Y` beside the absolute ``x_nm`` /
    ``y_nm``, which never change meaning: a plugin that reports a position
    to be found again in the file (a circle's centre) keeps using those,
    one that wants the structure's own axes uses these -- SMAP's
    ``xnmrot`` / ``ynmrot``.  ``angle`` is the angle in use, 0 when the
    ROIs are not rotated, so the frame is always the one on the screen.
    """
    center: np.ndarray
    angle: float = 0.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(2)
        self.angle = float(self.angle)

    def _turn(self, degrees: float) -> np.ndarray:
        a = np.radians(degrees)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def to_site(self, xy) -> np.ndarray:
        """Absolute positions, ``(n, 2)`` (or one ``(2,)``), in the frame."""
        xy = np.asarray(xy, dtype=np.float64)
        return (xy - self.center) @ self._turn(-self.angle).T

    def to_data(self, uv) -> np.ndarray:
        """Positions in the frame back to absolute ones."""
        uv = np.asarray(uv, dtype=np.float64)
        return uv @ self._turn(self.angle).T + self.center

    def columns(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """`SITE_X` and `SITE_Y` for columns ``x``, ``y``: one pass, float64."""
        dx = np.asarray(x, dtype=np.float64) - self.center[0]
        dy = np.asarray(y, dtype=np.float64) - self.center[1]
        if not self.angle % 360:
            return dx, dy
        c, s = np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))
        return c * dx + s * dy, -s * dx + c * dy


def polyline_distance(points, x: float, y: float, closed: bool = False) -> float:
    """Distance from a point to the line through ``points`` ((n, 2)), in
    data units; ``closed`` joins the last point to the first.  A single
    point is its own distance."""
    v = np.asarray(points, float).reshape(-1, 2)
    p = np.array([x, y], float)
    if len(v) == 0:
        return float("inf")
    if len(v) == 1:
        return float(np.hypot(*(p - v[0])))
    a, b = (v, np.roll(v, -1, axis=0)) if closed else (v[:-1], v[1:])
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum((ab ** 2).sum(1), 1e-12),
                0, 1)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


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
