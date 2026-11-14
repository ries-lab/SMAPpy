"""Equalising the sub-pixel histogram: positions moved so that, within groups of like localizations,
their place within the camera pixel is uniform.

A network that places molecules from a grid of output cells pulls an uncertain molecule toward the
cell it is placed from (pixelation, `training.reference.pixel_modulation`).  Wherever the sample
does not know the camera's grid -- every real sample -- the true places are uniform, so the found
places' distribution says how they were pulled, and its cumulative distribution ``F`` maps them
back: ``u -> F(u)`` makes them uniform.  ``F`` is monotone and keeps the pixel's edges (``F(0) = 0,
F(1) = 1``), so no localization crosses into another pixel and the order within a pixel is kept.

The pull depends on how sure the network is, so ``F`` is learnt per group: |z| in bins (100 nm by
default) and the localization's predicted uncertainty along that axis in quantile groups -- as
DECODE-Plex does (``docs/decode.md``; written here from that description, not from its code).
x and y are equalised apart, each with its own uncertainty.

**What it does not do: make the localizations more precise** (2026-11-14, netfit's models on test
set B, fitted on the even frames and judged on the odd).  It flattens the held-out histogram
(`pixel_modulation` 0.06 - 0.22 down to 0.00 - 0.02), but moves each localization by 1.4 - 4.4 nm
RMS, as often toward its truth as away from it (50 - 52 %), and the RMSE grows by 0.03 - 0.17 nm.
The reason is in the errors: their mean against the *true* place within the pixel is a pull of at
most about 2 nm toward the pixel's centre (1.0 - 1.6 nm RMS), which equalising leaves where it is,
against histogram peaks of 10 - 20 % -- the found places cluster at the cells' centres because of
how each molecule's errors are spread, not because molecules are shifted there.  Even the best correction from the found place alone, the mean error per group and place
learnt from the truth, made the RMSE 0.5 - 0.9 nm worse.  What equalising buys is a rendered image
without the camera's grid in it; it is no substitute for a network that does not pixelate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Z_BIN_NM = 100.0
SIGMA_GROUPS = 4
KNOTS = 40                  # quantiles of a group's places the map passes through
MIN_PER_GROUP = 400         # fewer: the group is left as it is (its map the identity)


def place(p):
    """A position's place within its pixel, 0 at the pixel's lower edge, 1 at its upper (pixel
    centres at whole numbers, as everywhere in SMAPpy)."""
    return (np.asarray(p, float) + 0.5) % 1.0


def cdf_knots(u, knots: int = KNOTS):
    """The map's knots for places ``u``: the places' quantiles at ``0, 1/knots, ..., 1``, with the
    pixel's edges fixed at 0 and 1."""
    q = np.quantile(u, np.linspace(0, 1, knots + 1))
    q[0], q[-1] = 0.0, 1.0
    return np.maximum.accumulate(q)


def apply_knots(u, q):
    """``F(u)``: each place moved to where its quantile is in a uniform distribution."""
    return np.interp(u, q, np.linspace(0, 1, len(q)))


@dataclass
class Axis:
    """One axis's maps: the |z| and uncertainty edges of the groups, and each group's knots
    (``None`` where too few localizations made one)."""
    z_edges: np.ndarray
    sigma_edges: list                 # per z bin: the uncertainty quantiles' edges
    knots: dict = field(default_factory=dict)       # (z bin, sigma group) -> knots or None

    def groups(self, z_nm, sigma):
        zb = np.clip(np.digitize(np.abs(z_nm), self.z_edges) - 1, 0, len(self.z_edges) - 2)
        sg = np.zeros(len(zb), int)
        for b, edges in enumerate(self.sigma_edges):
            m = zb == b
            sg[m] = np.clip(np.digitize(sigma[m], edges[1:-1]), 0, len(edges) - 2)
        return zb, sg


def fit_axis(p, z_nm, sigma, z_bin_nm: float = Z_BIN_NM, sigma_groups: int = SIGMA_GROUPS,
             knots: int = KNOTS, min_per_group: int = MIN_PER_GROUP) -> Axis:
    p, z_nm, sigma = (np.asarray(v, float) for v in (p, z_nm, sigma))
    top = max(float(np.max(np.abs(z_nm))), z_bin_nm) if len(z_nm) else z_bin_nm
    z_edges = np.arange(0.0, top + z_bin_nm, z_bin_nm)
    zb = np.clip(np.digitize(np.abs(z_nm), z_edges) - 1, 0, len(z_edges) - 2)
    sigma_edges = []
    for b in range(len(z_edges) - 1):
        s = sigma[zb == b]
        sigma_edges.append(np.quantile(s, np.linspace(0, 1, sigma_groups + 1)) if len(s) else np.array([0.0, np.inf]))
    axis = Axis(z_edges, sigma_edges)
    zb, sg = axis.groups(z_nm, sigma)
    u = place(p)
    for b in range(len(z_edges) - 1):
        for g in range(len(sigma_edges[b]) - 1):
            m = (zb == b) & (sg == g)
            axis.knots[(b, g)] = cdf_knots(u[m], knots) if m.sum() >= min_per_group else None
    return axis


def apply_axis(axis: Axis, p, z_nm, sigma):
    """The positions with each place moved by its group's map."""
    p = np.asarray(p, float)
    zb, sg = axis.groups(np.asarray(z_nm, float), np.asarray(sigma, float))
    u = place(p)
    v = u.copy()
    for (b, g), q in axis.knots.items():
        if q is not None:
            m = (zb == b) & (sg == g)
            v[m] = apply_knots(u[m], q)
    return p + (v - u)


@dataclass
class Equaliser:
    x: Axis
    y: Axis

    def __call__(self, x, y, z_nm, x_err, y_err):
        """``x``, ``y`` in pixels, their uncertainties ``x_err``, ``y_err`` in any one unit."""
        return apply_axis(self.x, x, z_nm, x_err), apply_axis(self.y, y, z_nm, y_err)


def fit(x, y, z_nm, x_err, y_err, **options) -> Equaliser:
    """The maps that make the places of these localizations uniform in each group (`fit_axis`)."""
    return Equaliser(fit_axis(x, z_nm, x_err, **options), fit_axis(y, z_nm, y_err, **options))


def equalise(x, y, z_nm, x_err, y_err, **options):
    """Fit on the localizations and move them: what a data set does to itself."""
    return fit(x, y, z_nm, x_err, y_err, **options)(x, y, z_nm, x_err, y_err)
