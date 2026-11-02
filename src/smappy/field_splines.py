"""Splines that vary over the field of view: one per node of a grid over the chip, blended per spot.

A PSF calibration learnt with *field dependence* (uipsf's `FieldPolynomial`) draws a different PSF at each
place on the chip.  It is kept as the splines of a regular grid of chip positions, and each spot is fitted
with the bilinear blend of the four nodes around it.  A spline is linear in its coefficients, so blending
coefficients is exactly the spline of the blended PSFs -- the fitter blends them per voxel
(`csrc/models.hpp`, `CSplineField`).  On simulated fields a 5 × 5 to 7 × 7 grid reached the exact spline's
bias where one spline for the field was 40 nm off in z at the corners, and M5's present fields need 3 × 3
(`docs/uipsf/field-dependence.md`, §5 and §7).

The grid is chosen from the learnt model (`grid_size`): the smallest whose blend is within a tolerance of
the model between the nodes, so a model with little field dependence costs one spline, as before.

Two conditions the blend needs: every node's spline has one normalisation (the model's photons per photon,
never one of its own), and the nodes' PSFs are centred alike -- the field's tilt and defocus, a distortion
and a focal surface, are not in them.  `calibrate.pupil.field_spline_calibration` makes them so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldSplines:
    """The node splines of a field-dependent calibration."""

    coeff: np.ndarray       # (G, 64, nz, ny, nx) float32, node g = j * len(xs) + i
    xs: np.ndarray          # (gx,) the nodes' chip x, pixels, increasing
    ys: np.ndarray          # (gy,) the nodes' chip y
    blend_error: float = float("nan")   # the grid's largest blend error, of the PSF's peak (`choose_grid`)

    def __post_init__(self):
        self.coeff = np.ascontiguousarray(self.coeff, dtype=np.float32)
        self.xs = np.asarray(self.xs, float).reshape(-1)
        self.ys = np.asarray(self.ys, float).reshape(-1)
        if self.coeff.ndim != 5 or self.coeff.shape[1] != 64:
            raise ValueError("field spline coefficients must have shape (G, 64, nz, ny, nx)")
        if self.coeff.shape[0] != len(self.xs) * len(self.ys):
            raise ValueError(f"{self.coeff.shape[0]} node splines for a {len(self.ys)} × {len(self.xs)} grid")
        for a in (self.xs, self.ys):
            if len(a) > 1 and not np.all(np.diff(a) > 0):
                raise ValueError("the grid's node positions must increase")

    @property
    def shape(self) -> tuple:
        """(gy, gx)."""
        return len(self.ys), len(self.xs)


def _axis(values, nodes):
    """(lower node index, weight of the upper one) along one axis, positions outside the grid clamped."""
    if len(nodes) == 1:
        return np.zeros(len(values), int), np.zeros(len(values))
    v = np.clip(values, nodes[0], nodes[-1])
    i = np.clip(np.searchsorted(nodes, v, side="right") - 1, 0, len(nodes) - 2)
    return i, (v - nodes[i]) / (nodes[i + 1] - nodes[i])


def bilinear(field: FieldSplines, chip_xy) -> tuple:
    """(nodes (n, 4) int32, weights (n, 4) float32): the four nodes around each chip position (x, y) pixels
    and their bilinear weights; a position outside the grid takes its edge's (no extrapolation)."""
    xy = np.asarray(chip_xy, float).reshape(-1, 2)
    i, a = _axis(xy[:, 0], field.xs)
    j, b = _axis(xy[:, 1], field.ys)
    gx = len(field.xs)
    di, dj = int(gx > 1), int(len(field.ys) > 1)       # a single row or column has no neighbour
    nodes = np.stack([j * gx + i, j * gx + i + di, (j + dj) * gx + i, (j + dj) * gx + i + di], 1)
    weights = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1)
    return nodes.astype(np.int32), weights.astype(np.float32)


def blend(field: FieldSplines, chip_xy) -> np.ndarray:
    """The blended coefficients (64, nz, ny, nx) at one chip position: what the fitter uses there."""
    nodes, weights = bilinear(field, [chip_xy])
    return np.tensordot(weights[0], field.coeff[nodes[0]], axes=(0, 0)).astype(np.float32)


def grid_nodes(lo, hi, n: int) -> np.ndarray:
    """``n`` node positions spanning [lo, hi]; one node in the middle for n = 1."""
    return np.array([(lo + hi) / 2.0]) if n == 1 else np.linspace(lo, hi, n)


def blend_error(psf_at, lo_xy, hi_xy, n: int, points: int = 3) -> float:
    """The largest relative difference between the bilinear blend of an n × n grid's PSFs and the PSF itself,
    at the midpoints between nodes (where a blend is worst), ``points`` per cell side; ``psf_at(x, y)`` the
    model's PSF stack there.  Relative to the PSF's peak."""
    xs, ys = grid_nodes(lo_xy[0], hi_xy[0], n), grid_nodes(lo_xy[1], hi_xy[1], n)
    nodes = {(i, j): psf_at(x, y) for j, y in enumerate(ys) for i, x in enumerate(xs)}
    peak = max(float(np.abs(v).max()) for v in nodes.values())
    worst = 0.0
    # a single node stands for the whole field: compare it with the corners
    if n == 1:
        centre = nodes[0, 0]
        for x in (lo_xy[0], hi_xy[0]):
            for y in (lo_xy[1], hi_xy[1]):
                worst = max(worst, float(np.abs(psf_at(x, y) - centre).max()) / peak)
        return worst
    f = np.linspace(0, 1, points + 2)[1:-1]
    for j in range(n - 1):
        for i in range(n - 1):
            for a in f:
                for b in f:
                    x = xs[i] + a * (xs[i + 1] - xs[i])
                    y = ys[j] + b * (ys[j + 1] - ys[j])
                    mixed = ((1 - a) * (1 - b) * nodes[i, j] + a * (1 - b) * nodes[i + 1, j]
                             + (1 - a) * b * nodes[i, j + 1] + a * b * nodes[i + 1, j + 1])
                    worst = max(worst, float(np.abs(psf_at(x, y) - mixed).max()) / peak)
    return worst


#: the largest blend error, relative to the PSF's peak, a grid may leave; 0.5 % is about where M5's
#: linear astigmatism needed 3 × 3 rather than 2 × 2 (excess χ² 0.007 against 0.11 at 5000 photons)
TOLERANCE = 0.005
GRIDS = (1, 2, 3, 5, 7, 9)


def choose_grid(psf_at, lo_xy, hi_xy, tolerance: float = TOLERANCE) -> tuple:
    """(n, its blend error): the smallest grid of `GRIDS` whose blend stays within ``tolerance`` of the model
    over the field, or the largest with its error above it -- which `limit_warning` then says."""
    for n in GRIDS:
        error = blend_error(psf_at, lo_xy, hi_xy, n)
        if error <= tolerance:
            return n, error
    return GRIDS[-1], error


def grid_size(psf_at, lo_xy, hi_xy, tolerance: float = TOLERANCE) -> int:
    """The smallest grid of `GRIDS` whose blend stays within ``tolerance`` of the model over the field."""
    return choose_grid(psf_at, lo_xy, hi_xy, tolerance)[0]


def limit_warning(n: int, error: float, tolerance: float = TOLERANCE) -> str | None:
    """What to say when the largest grid still misses the model by more than ``tolerance``; None otherwise.

    The cap is memory (81 splines of a 27 px box at 20 nm planes are about 2 GB) and the time to make them.
    Over Zenodo's 177 µm field (about 1.4 rad of astigmatism across it) 9 × 9 left 2.7 % of the peak, yet
    a spline drawn exactly at each bead fitted no better than the blend (`docs/uipsf/field-dependence.md`,
    §9): the warning says the grid is coarse for the field, not that the fit is wrong."""
    if n < GRIDS[-1] or error <= tolerance:
        return None
    return (f"the field changes the PSF faster than a {n} × {n} grid of splines follows: between the nodes the "
            f"blended PSF differs from the model by up to {100 * error:.1f} % of its peak (tolerance "
            f"{100 * tolerance:.1f} %); spots far from the field's centre may be fitted slightly less well")
