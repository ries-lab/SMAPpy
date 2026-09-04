"""Regions of interest: a shape in the coordinates of the table, and its mask.

A `Region` is what a viewer draws and what a plugin's `Selection` carries;
it knows nothing about widgets.  A line is a rectangle of a given width
around the segment, as in SMAP, and is stored as its four corners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class Region:
    kind: str                               # "rect", "polygon" or "line"
    points: np.ndarray                      # (n, 2): rect corners, or vertices
    width: float = 0.0                      # line only, in data units

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

    @property
    def bounds(self):
        (x0, y0), (x1, y1) = self.points.min(axis=0), self.points.max(axis=0)
        return x0, y0, x1, y1

    def mask(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.kind == "rect":
            x0, y0, x1, y1 = self.bounds
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return _in_polygon(x, y, self.points)

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "points": self.points.tolist(), "width": self.width}

    @classmethod
    def from_dict(cls, d: Dict) -> "Region":
        return cls(d["kind"], d["points"], d.get("width", 0.0))

    def __str__(self) -> str:
        x0, y0, x1, y1 = self.bounds
        return f"{self.kind} ROI {x1 - x0:.0f} x {y1 - y0:.0f}"


def _in_polygon(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over the points; the polygon is small."""
    inside = np.zeros(x.shape, dtype=bool)
    n = len(vertices)
    x0, y0, x1, y1 = *vertices.min(axis=0), *vertices.max(axis=0)
    box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    xb, yb = x[box], y[box]
    hit = np.zeros(xb.shape, dtype=bool)
    for i in range(n):
        (xi, yi), (xj, yj) = vertices[i], vertices[(i + 1) % n]
        crosses = (yi > yb) != (yj > yb)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = xi + (yb - yi) * (xj - xi) / (yj - yi)
        hit ^= crosses & (xb < x_at)
    inside[box] = hit
    return inside
