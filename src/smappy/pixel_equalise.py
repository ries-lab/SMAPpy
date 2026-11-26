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

**Two estimators of the map.**  `fit` is the one above, 40 quantile knots per cell.  `fit_smooth`,
what *Pixel equalisation* uses, has no cells: the density's Fourier coefficients are smooth functions
of signed z and the uncertainty's rank, fitted to every localization at once, with only the
harmonics that stand out from counting noise kept (`fit_smooth_axis`).  Compared on netfit's models
and DECODE's example movie (benchmarks/decode/pixelation/estimators.py, 2026-11-26) it flattens as
much and moves less at random: the map's own noise 0.6 - 0.9 nm against the knots' 0.7 - 1.4 nm, and
on a fit with nothing to correct (Spline 3D) 0.45 nm of movement against 0.77 nm.  Neither improves
the positions: the spread of a molecule's two localizations in frames that share nothing grows by
0.01 - 0.6 nm with either.
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


# ---------------------------------------------------------------------------------------------------
# the smooth map: what *Pixel equalisation* uses

#: the harmonics of the place's density a map may use; the data keep the ones that stand out
HARMONICS = 10
#: B-splines along z and along the uncertainty's rank: the coarsest of those tried that flattened as
#: much as any (benchmarks/decode/pixelation/estimators.py, 2026-11-26); finer ones flattened less at
#: 50 - 600 k localizations, because the shrinkage grows with the parameters
Z_SPLINES = 6
SIGMA_SPLINES = 4
#: a harmonic's surface is kept when its chi-squared is this many standard deviations above noise
SIGNIFICANCE = 3.0
#: the density a map may make is kept above this, so that it stays monotone
DENSITY_FLOOR = 0.2
TWO_PI = 2 * np.pi


def fourier_cdf(u, a, b):
    """``F(u)`` for the density ``1 + sum_k a_k cos 2 pi k u + b_k sin 2 pi k u``: ``u`` (n,), the
    coefficients (n, K) one row per place, or (K,) shared."""
    u = np.asarray(u, float)
    k = np.arange(1, np.shape(a)[-1] + 1)
    w = TWO_PI * k * u[:, None]
    return u + ((a * np.sin(w) + b * (1 - np.cos(w))) / (TWO_PI * k)).sum(-1)


def keep_monotone(a, b, floor: float = DENSITY_FLOOR):
    """The coefficients scaled down where their density would fall below ``floor``: noise at a rare
    z or uncertainty could otherwise fold F back on itself and swap two localizations' order."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not a.size:
        return a, b
    grid = np.linspace(0, 1, 64, endpoint=False)
    k = np.arange(1, a.shape[-1] + 1)
    w = TWO_PI * k * grid[:, None]
    p = 1 + np.einsum("...k,gk->...g", a, np.cos(w)) + np.einsum("...k,gk->...g", b, np.sin(w))
    low = p.min(-1)
    s = np.where(low < floor, (1 - floor) / np.maximum(1 - low, 1e-9), 1.0)
    return a * s[..., None], b * s[..., None]


def _splines(x, n: int, degree: int = 3):
    """``n`` B-splines on [0, 1] at ``x`` (clipped), (len(x), n); one constant for ``n`` < 4."""
    x = np.asarray(x, float)
    if n < degree + 1:
        return np.ones((len(x), 1))
    from scipy.interpolate import BSpline
    t = np.r_[[0.0] * degree, np.linspace(0, 1, n - degree + 1), [1.0] * degree]
    return BSpline.design_matrix(np.clip(x, 0, 1 - 1e-12), t, degree).toarray()


@dataclass
class SmoothAxis:
    """One axis's map: the density's Fourier coefficients as smooth functions of z and of the
    uncertainty's rank, each a tensor product of B-splines.

    No cells: each localization's ``2 cos 2 pi k u`` and ``2 sin 2 pi k u`` -- whose means are the
    coefficients -- are fitted by least squares over all localizations at once, so neighbouring z
    and uncertainties share their counts and the map has no edge where it jumps.  A harmonic's
    surface is kept only where it stands out from counting noise, and shrunk by what the noise adds
    to its size (`fit_smooth_axis`): with nothing to correct the map is the identity."""
    z_range: tuple                       # z mapped to [0, 1]; equal ends: no z
    sigma_quantiles: np.ndarray          # the uncertainty's quantiles, for its rank
    z_splines: int
    sigma_splines: int
    a: np.ndarray                        # (basis, K): cos coefficients
    b: np.ndarray                        # (basis, K): sin coefficients

    def _design(self, z_nm, sigma):
        z0, z1 = self.z_range
        nz = self.z_splines if z1 > z0 else 1
        bz = _splines((np.asarray(z_nm, float) - z0) / max(z1 - z0, 1e-9), nz)
        q = self.sigma_quantiles
        ns = self.sigma_splines if q[-1] > q[0] else 1
        rank = np.interp(np.asarray(sigma, float), q, np.linspace(0, 1, len(q)))
        bs = _splines(rank, ns)
        return (bz[:, :, None] * bs[:, None, :]).reshape(len(bz), -1)

    def coefficients(self, z_nm, sigma):
        """Each localization's coefficients, (n, K) twice."""
        X = self._design(z_nm, sigma)
        return keep_monotone(X @ self.a, X @ self.b)

    @property
    def harmonics_kept(self) -> int:
        return int((np.any(self.a != 0, 0) | np.any(self.b != 0, 0)).sum())

    def __call__(self, p, z_nm, sigma):
        """The positions ``p`` (pixels) with each place moved by its map; non-finite ones kept."""
        p = np.asarray(p, float)
        out = p.copy()
        ok = np.isfinite(p) & np.isfinite(z_nm) & np.isfinite(sigma)
        if ok.any() and self.harmonics_kept:
            u = place(p[ok])
            a, b = self.coefficients(np.asarray(z_nm, float)[ok], np.asarray(sigma, float)[ok])
            out[ok] = p[ok] + fourier_cdf(u, a, b) - u
        return out

    def to_dict(self) -> dict:
        return {"z_range": [float(v) for v in self.z_range],
                "sigma_quantiles": [float(v) for v in self.sigma_quantiles],
                "z_splines": int(self.z_splines), "sigma_splines": int(self.sigma_splines),
                "a": np.asarray(self.a).tolist(), "b": np.asarray(self.b).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "SmoothAxis":
        return cls(tuple(d["z_range"]), np.asarray(d["sigma_quantiles"], float), int(d["z_splines"]),
                   int(d["sigma_splines"]), np.asarray(d["a"], float), np.asarray(d["b"], float))


def fit_smooth_axis(p, z_nm, sigma, z_splines: int = Z_SPLINES, sigma_splines: int = SIGMA_SPLINES,
                    harmonics: int = HARMONICS, significance: float = SIGNIFICANCE) -> SmoothAxis:
    """The map that makes the places of positions ``p`` (pixels) uniform, given z and each one's
    uncertainty along this axis (any unit).  ``z_nm`` may be all zeros (a 2D fit), ``sigma`` all
    equal: that dimension then has one constant in place of its splines.

    The noise: an observation ``2 cos(...)`` has variance about 2, so the least-squares coefficients
    ``c`` of one harmonic's surface have the covariance ``2 G^-1`` (``G = X'X``), and ``c' G c / 2``
    is chi-squared with ``P`` degrees of freedom where the surface is noise alone.  A surface less
    than ``significance`` standard deviations (``sqrt(2 P)``) above ``P`` is dropped; the rest are
    shrunk by ``1 - P / chi2`` (James-Stein), so a weak harmonic is corrected only in part."""
    p, z_nm, sigma = (np.asarray(v, float) for v in (p, z_nm, sigma))
    ok = np.isfinite(p) & np.isfinite(z_nm) & np.isfinite(sigma)
    p, z_nm, sigma = p[ok], z_nm[ok], sigma[ok]
    if len(p) >= 2:
        z_range = (float(np.quantile(z_nm, 0.001)), float(np.quantile(z_nm, 0.999)))
        if z_range[1] - z_range[0] < 1.0:          # no z worth the name: a 2D fit, or z all alike
            z_range = (z_range[0], z_range[0])
        q = np.quantile(sigma, np.linspace(0, 1, 101))
    else:
        z_range, q = (0.0, 0.0), np.zeros(2)
    axis = SmoothAxis(z_range, q, z_splines, sigma_splines,
                      np.zeros((0, harmonics)), np.zeros((0, harmonics)))
    X = axis._design(z_nm, sigma)
    P = X.shape[1]
    axis.a = np.zeros((P, harmonics))
    axis.b = np.zeros((P, harmonics))
    if len(p) <= 4 * P:
        return axis
    k = np.arange(1, harmonics + 1)
    w = TWO_PI * k * place(p)[:, None]
    Y = np.c_[2 * np.cos(w), 2 * np.sin(w)]
    G = X.T @ X
    C = np.linalg.solve(G + np.eye(P), X.T @ Y)
    chi2 = np.einsum("pc,pq,qc->c", C, G, C) / 2
    keep = chi2 - P > significance * np.sqrt(2 * P)
    C = C * np.where(keep, np.clip(1 - P / np.maximum(chi2, 1e-12), 0, 1), 0.0)
    axis.a, axis.b = C[:, :harmonics], C[:, harmonics:]
    return axis


@dataclass
class SmoothEqualiser:
    x: SmoothAxis
    y: SmoothAxis

    def __call__(self, x, y, z_nm, x_err, y_err):
        """``x``, ``y`` in pixels, their uncertainties in any one unit."""
        return self.x(x, z_nm, x_err), self.y(y, z_nm, y_err)

    def to_dict(self) -> dict:
        return {"x": self.x.to_dict(), "y": self.y.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "SmoothEqualiser":
        return cls(SmoothAxis.from_dict(d["x"]), SmoothAxis.from_dict(d["y"]))


def fit_smooth(x, y, z_nm, x_err, y_err, **options) -> SmoothEqualiser:
    """The smooth maps of both axes (`fit_smooth_axis`), each with its own uncertainty."""
    return SmoothEqualiser(fit_smooth_axis(x, z_nm, x_err, **options),
                           fit_smooth_axis(y, z_nm, y_err, **options))
