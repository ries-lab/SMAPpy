"""Spatial statistics: are the localizations clustered, and at what scale?

SMAP's ``Analyze/cluster/SpatialPointPattern`` (pair correlation and Ripley's
K, after Sengupta et al. 2011), its helper ``myRipleysK``, and the 2D pair
correlation of ``Analyze/measure/correlation`` (``paircrosscorrelationfunction``,
Veatch et al. 2012), as one plugin with a list of *approaches*.  This module
holds the radial ones -- the pair correlation g(r), Ripley's K(r) and
L(r) - r -- which are one computation shown three ways; the directional ones
(a 2D correlation image, the correlation along a line) are to join them as
approaches of their own.  `docs/smap_porting/wave4_design.md`, step 17a, has
the decisions.

**One computation.**  Every pair of localizations closer than r_max is
binned by its distance: the pair counts H(r).  Under complete spatial
randomness (CSR) in a window W of area |W|, the expected count in a ring
[r0, r1] is

    E(r0, r1) = n (n - 1) / |W|^2 * integral over the ring of gamma_W(h) dh,

where gamma_W(h) = |W ∩ (W + h)| is the window's **set covariance** -- the
area the window shares with itself shifted by h.  g = H / E, K is the
integral of g over the disc, L = sqrt(K / pi).  The ratio is Ohser's
translation edge correction made isotropic, and it is what Veatch's
normalisation does in Fourier space: SMAP's ``paircrosscorrelationfunction``
divides the image's autocorrelation by the mask's.  It holds for any window
shape, so an ROI of the ROI manager, a polygon, a disc or the cell's own
outline are all corrected exactly, where ``myRipleysK``'s minus sampling
(only points farther than r_max from the box's edge) works for a rectangle
alone and throws most of a small ROI away.

Where it departs from SMAP's, and why:

* **The pairs are counted, not Fourier transformed.**  SMAP binned the
  localizations at 2 nm and took the FFT of the whole picture -- 400 million
  pixels for a 40 um field.  Here the pairs come from Neighbourhood's C++
  cell search (`csrc/neighbours.cpp`, `Index.pair_histogram`): n rho pi
  r_max^2 pairs, 10^9 for a million localizations at r_max = 500 nm, in
  about a second, with the exact distances (no 2 nm grid).  Only the window
  is Fourier transformed, on a coarse grid, for gamma_W.
* **Self-pairs are left out**, so g(0) is not an outcome; SMAP's r = 0 bin
  held them, and its plots dropped the first bins by hand.
* **The window** is the region's shape, else the drawn ROI, else the area
  the localizations occupy -- a mask of them binned at about 200 nm, closed
  -- or their bounding box.  SMAP used the field of view, which makes a cell
  in the middle of it look clustered (its outside is empty).
* **Envelopes.**  Neither SMAP plugin said what g = 1.3 means.  Here the
  same number of points is redistributed in the same window, n times (99),
  and the curves of these simulations give a pointwise band and one global
  test (the maximum absolute deviation of L - r, Diggle's MAD test).  Two
  nulls: CSR of the rows the layer shows, and *keep blinks*: each blink's
  localizations moved together (through ``group_id``), so the null blinks as
  the data do and only the arrangement of the molecules is tested.  For a
  cross-correlation layer A stays where it is and layer B is redistributed.
* **Layers**: every visible localization layer as it shows (its filter, its
  *merge blinks*), the auto-correlation of each and the cross-correlation of
  each pair, as SMAP's ``correlation``.  ``SpatialPointPattern``'s own cross
  term multiplied the magnitudes of the two transforms and so lost the
  phase -- it was not a cross-correlation; that is not ported.
* **Per file**: pairs never join two files; each file's counts and expected
  counts are added, so g of several files is pooled, not averaged.
* Sengupta's PC-PALM fit of the blinking is not ported.

``per_region``: in a region the values (the first maximum of g and of L - r,
both at chosen distances, the envelope test) go to the region table, and the
curves with their pair and expected counts to its arrays, so a summary over
regions can pool them (sum over sum) rather than average ratios.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import (Context, FormState, Plot, Plugin, PreflightChoice, PreflightQuestion,
               Result, param, register)

# the window's raster for a drawn shape: this many pixels along its longer
# side.  gamma_W of a raster is exact (bilinear between the lags), so this
# only decides how well the raster follows a curved outline: 0.2 % of its
# size, a bias of g well below the noise of any real data set
WINDOW_PIXELS = 512
# angles over a half circle for the isotropic set covariance (gamma_W(h) =
# gamma_W(-h), so half is all)
ANGLES = 90
# the first maximum: the curve smoothed over this many bins, and a maximum
# counted when it rises at least this fraction of the highest one's height
SMOOTH_BINS = 5
PEAK_FRACTION = 0.5
# keep-blinks placements: a blink put where part of it falls outside the
# window is put again, up to this many times
PLACEMENT_TRIES = 30
# above this many seconds a run asks first (`preflight`)
ASK_ABOVE_S = 5.0
# the points drawn in the window's figure
FIGURE_POINTS = 30_000
# the occupied area's pixel when nobody chooses one (Neighbourhood, the
# directional approaches)
OCCUPIED_BIN_NM = 200.0

APPROACHES = (("pair_correlation", "g(r)"), ("ripley_k", "K(r)"),
              ("ripley_l", "L(r) - r"))
DIRECTIONAL = (("image", "correlation image"), ("line", "correlation along a line"))


# --------------------------------------------------------------- the window

@dataclass
class Window:
    """A window as a raster: ``mask[iy, ix]`` is the pixel from
    ``(x0 + ix px, y0 + iy py)``.  Everything about the window -- its area,
    which points are in it, uniform points in it, its set covariance -- is
    of this raster, so the normalisation and the simulations agree exactly.
    """
    mask: np.ndarray
    x0: float
    y0: float
    px: float
    py: float
    _cov: Optional[np.ndarray] = field(default=None, repr=False)
    _cov_reach: float = field(default=0.0, repr=False)

    @property
    def area(self) -> float:
        return float(self.mask.sum()) * self.px * self.py

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        ny, nx = self.mask.shape
        return self.x0, self.y0, self.x0 + nx * self.px, self.y0 + ny * self.py

    def pixel_of(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.mask.shape
        ix = np.floor((np.asarray(x, float) - self.x0) / self.px).astype(np.int64)
        iy = np.floor((np.asarray(y, float) - self.y0) / self.py).astype(np.int64)
        # a point on the far edge is in the last pixel, as histogram2d has it
        ix[ix == nx] = nx - 1
        iy[iy == ny] = ny - 1
        return ix, iy

    def contains(self, x, y) -> np.ndarray:
        ix, iy = self.pixel_of(x, y)
        ny, nx = self.mask.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(len(ix), dtype=bool)
        out[inside] = self.mask[iy[inside], ix[inside]]
        return out

    def uniform(self, n: int, rng) -> np.ndarray:
        """``n`` points uniform in the window, (n, 2)."""
        cells = np.flatnonzero(self.mask.ravel())
        if not len(cells):
            raise ValueError("the window is empty")
        pick = cells[rng.integers(0, len(cells), n)]
        iy, ix = np.divmod(pick, self.mask.shape[1])
        return np.column_stack([self.x0 + (ix + rng.random(n)) * self.px,
                                self.y0 + (iy + rng.random(n)) * self.py])

    def covariance(self, r: np.ndarray) -> np.ndarray:
        """The set covariance averaged over directions, gamma(r) in nm^2:
        the area the window shares with itself shifted by r, as the mean over
        `ANGLES` directions.  gamma(0) is the area; beyond the window's
        diameter it is 0."""
        from scipy.ndimage import map_coordinates
        r = np.asarray(r, dtype=float)
        reach = float(r.max()) if r.size else 0.0
        if self._cov is None or reach > self._cov_reach:
            self._cov = self._autocorrelation(reach)
            self._cov_reach = reach
        theta = (np.arange(ANGLES) + 0.5) * np.pi / ANGLES
        dx = np.outer(r, np.cos(theta)) / self.px
        dy = np.outer(r, np.sin(theta)) / self.py
        values = map_coordinates(self._cov, [dy.ravel(), dx.ravel()], order=1,
                                 mode="grid-wrap")
        return np.clip(values.reshape(dx.shape).mean(axis=1), 0.0, None)

    def _autocorrelation(self, reach: float) -> np.ndarray:
        """``sum_x m(x) m(x + k)`` px py over integer lags k, wrapped: the FFT
        of the mask padded by ``reach`` so no lag up to it wraps onto another."""
        from scipy.fft import irfft2, next_fast_len, rfft2
        ny, nx = self.mask.shape
        shape = (next_fast_len(ny + int(np.ceil(reach / self.py)) + 2),
                 next_fast_len(nx + int(np.ceil(reach / self.px)) + 2))
        f = rfft2(self.mask.astype(float), shape)
        c = irfft2(f * np.conj(f), shape)
        return np.clip(c, 0.0, None) * (self.px * self.py)


def _grid(lo: float, hi: float, step: float) -> Tuple[float, int]:
    """Pixels from ``lo`` to ``hi`` of about ``step``: (pixel, count), the
    pixel adjusted so they end exactly at ``hi``."""
    span = max(hi - lo, 0.0)
    n = max(1, int(round(span / step)))
    return (span / n if span > 0 else step), n


def window_of_region(region, pixels: int = WINDOW_PIXELS) -> Window:
    """The raster of a shape (`regions.Region`): its bounding box in about
    ``pixels`` pixels along the longer side, a pixel in when its centre is
    inside the shape -- a rectangle exactly."""
    x0, y0, x1, y1 = (float(v) for v in region.bounds)
    step = max(x1 - x0, y1 - y0, 1e-9) / pixels
    px, nx = _grid(x0, x1, step)
    py, ny = _grid(y0, y1, step)
    cx = x0 + (np.arange(nx) + 0.5) * px
    cy = y0 + (np.arange(ny) + 0.5) * py
    gx, gy = np.meshgrid(cx, cy)
    mask = np.asarray(region.mask(gx.ravel(), gy.ravel()), dtype=bool).reshape(ny, nx)
    if not mask.any():
        raise ValueError(f"the {region} has no area")
    return Window(mask, x0, y0, px, py)


def window_of_points(x, y, how: str = "occupied", bin_nm: float = 200.0) -> Window:
    """The window the localizations themselves give: their bounding box
    (``"box"``), or the area they occupy (``"occupied"``) -- binned at about
    ``bin_nm``, a bin kept when it holds one, and closed (a dilation then an
    erosion by a pixel, twice) so the gaps between molecules are not taken
    for holes in the cell.  The grid starts and ends at the outermost points,
    so a uniform rectangle's window is the rectangle."""
    from scipy.ndimage import binary_closing
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x):
        raise ValueError("no localizations to make a window of")
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    if how == "box":
        step = max(x1 - x0, y1 - y0, 1e-9) / WINDOW_PIXELS
        px, nx = _grid(x0, x1, step)
        py, ny = _grid(y0, y1, step)
        return Window(np.ones((ny, nx), dtype=bool), x0, y0, px, py)
    if how != "occupied":
        raise ValueError(f"no window {how!r}: occupied or box")
    px, nx = _grid(x0, x1, bin_nm)
    py, ny = _grid(y0, y1, bin_nm)
    w = Window(np.zeros((ny, nx), dtype=bool), x0, y0, px, py)
    ix, iy = w.pixel_of(x, y)
    w.mask[iy, ix] = True
    pad = 3
    big = np.pad(w.mask, pad)
    big = binary_closing(big, structure=np.ones((3, 3), bool), iterations=2)
    w.mask = big[pad:-pad, pad:-pad] | w.mask
    return w


# ------------------------------------------------------------- pair counts

def pair_counts(a: np.ndarray, b: Optional[np.ndarray], r_max: float, dr: float,
                same: Optional[np.ndarray] = None, threads: int = 0) -> np.ndarray:
    """The histogram of the distances from each point of ``a`` to each point
    of ``b`` closer than ``r_max``, in bins of ``dr`` from 0 (the last bin
    ends at r_max, which should be a whole number of bins).  Each ordered
    pair once: ``b`` None is ``a`` against itself, every pair twice and a
    point never with itself.  ``same[i]``, given for two sets, is the row of
    ``b`` that is point ``i`` of ``a``, or -1 (two layers over one table).

    Lateral only: ``a`` and ``b`` are (n, 2) or wider, x and y first.
    """
    from .._neighbours import Index, cell_order
    bins = int(round(r_max / dr))
    if bins < 1 or not r_max > 0:
        raise ValueError("r_max must be at least one bin of dr")
    reach = bins * dr
    a = np.ascontiguousarray(np.asarray(a, dtype=float)[:, :2])
    if b is None:
        b = a
        same = np.arange(len(a), dtype=np.int64)
    else:
        b = np.ascontiguousarray(np.asarray(b, dtype=float)[:, :2])
    if not len(a) or not len(b):
        return np.zeros(bins)
    x0 = float(min(a[:, 0].min(), b[:, 0].min()))
    y0 = float(min(a[:, 1].min(), b[:, 1].min()))
    x1 = float(max(a[:, 0].max(), b[:, 0].max()))
    index = Index(b[:, 0], b[:, 1], None, None, x0, y0, x1, reach, threads)
    order = cell_order(a[:, 0], a[:, 1], x0, y0, x1, reach, threads)
    s = None if same is None else np.asarray(same, dtype=np.int64)[order]
    return index.pair_histogram(a[order, 0], a[order, 1], s, float(dr), bins, threads)


def expected_counts(window: Window, pairs: float, edges: np.ndarray) -> np.ndarray:
    """The pair counts CSR expects in each ring between ``edges``:
    ``pairs`` / |W|^2 times the integral of the set covariance over the ring,
    with ``pairs`` the number of ordered pairs -- n (n - 1) for a set
    against itself, n_a n_b for two."""
    edges = np.asarray(edges, dtype=float)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ring = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    area = window.area
    if area <= 0:
        return np.zeros(len(mid))
    return pairs / area ** 2 * ring * window.covariance(mid)


def curves_of(counts: np.ndarray, expected: np.ndarray, edges: np.ndarray
              ) -> Dict[str, np.ndarray]:
    """g, K and L - r from pair and expected counts, at the bin centres.

    g is their ratio; K(r) = integral_0^r 2 pi s g(s) ds, summed ring by ring
    to the bin edges and taken at the centres halfway through the ring's
    area; L = sqrt(K / pi).  Where nothing is expected (beyond the window),
    NaN."""
    edges = np.asarray(edges, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(expected > 0, counts / expected, np.nan)
    ring = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    mid = 0.5 * (edges[1:] + edges[:-1])
    filled = np.where(np.isfinite(g), g, np.nan) * ring
    k_edges = np.concatenate([[0.0], np.cumsum(filled)])
    inner = np.pi * (mid ** 2 - edges[:-1] ** 2)
    k = k_edges[:-1] + np.where(np.isfinite(g), g, np.nan) * inner
    l_r = np.sqrt(np.clip(k, 0.0, None) / np.pi) - mid
    return {"g": g, "K": k, "L_r": l_r}


# -------------------------------------------------------------- the nulls

def csr(window: Window, n: int, rng) -> np.ndarray:
    """``n`` points uniform in the window."""
    return window.uniform(n, rng)


def keep_blinks(window: Window, xy: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """``xy`` with each group (blink) moved as a whole: its centre to a
    uniform place in the window, its localizations keeping their offsets from
    it.  A blink with a localization outside the window is placed again, up
    to `PLACEMENT_TRIES` times."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    ids, inverse = np.unique(np.asarray(groups), return_inverse=True)
    inverse = np.ravel(inverse)
    m = len(ids)
    count = np.bincount(inverse, minlength=m).astype(float)
    centre = np.column_stack([np.bincount(inverse, xy[:, 0], m),
                              np.bincount(inverse, xy[:, 1], m)]) / count[:, None]
    offset = xy - centre[inverse]
    place = window.uniform(m, rng)
    todo = np.arange(m)
    for _ in range(PLACEMENT_TRIES):
        p = place[inverse] + offset
        outside = np.zeros(m, dtype=bool)
        np.logical_or.at(outside, inverse, ~window.contains(p[:, 0], p[:, 1]))
        todo = np.flatnonzero(outside)
        if not len(todo):
            break
        place[todo] = window.uniform(len(todo), rng)
    return place[inverse] + offset


# ---------------------------------------------------------- the envelopes

def pointwise_band(sims: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """The (1 - level) / 2 and (1 + level) / 2 quantiles of the simulated
    curves at each r, ignoring NaN."""
    lo, hi = (1 - level) / 2, (1 + level) / 2
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanquantile(sims, lo, axis=0), np.nanquantile(sims, hi, axis=0))


def mad_test(observed: np.ndarray, sims: np.ndarray, use: Optional[np.ndarray] = None,
             level: float = 0.95) -> Dict[str, Any]:
    """Diggle's maximum absolute deviation test over the r where ``use``.

    T = max |curve(r) - mean(r)| for the data, against the same of each
    simulation from the mean of the others: p = (1 + #(T_i >= T)) / (n + 1).
    The global band is the mean +- the ``level`` quantile of the T_i: a curve
    leaves it somewhere exactly when the test rejects at 1 - level."""
    sims = np.asarray(sims, dtype=float)
    n = len(sims)
    if use is None:
        use = np.ones(sims.shape[1], dtype=bool)
    use = use & np.isfinite(observed) & np.isfinite(sims).all(axis=0)
    nan = float("nan")
    if n < 2 or not use.any():
        return {"p": nan, "T": nan, "T_crit": nan, "mean": np.full(sims.shape[1], nan)}
    mean = sims.mean(axis=0)
    others = (n * mean - sims) / (n - 1)
    t_sims = np.abs(sims - others)[:, use].max(axis=1)
    t = float(np.abs(observed - mean)[use].max())
    p = (1 + int(np.sum(t_sims >= t))) / (n + 1)
    # the k-th smallest of n with k = ceil(level (n + 1)), the usual rank rule
    k = min(n, max(1, int(np.ceil(level * (n + 1)))))
    t_crit = float(np.sort(t_sims)[k - 1])
    return {"p": p, "T": t, "T_crit": t_crit, "mean": mean}


def first_maximum(r: np.ndarray, y: np.ndarray, null: float, r_min: float = 0.0
                  ) -> Tuple[float, float]:
    """``(position, height)`` of the first maximum of ``y`` above ``null``.

    The curve is smoothed over `SMOOTH_BINS` bins; among its local maxima at
    r >= r_min (the first bin counts when the curve falls from it) the first
    that rises at least `PEAK_FRACTION` of the highest one's height above
    ``null``.  The height is the smoothed curve's.  NaN when nothing rises
    above ``null``."""
    r, y = np.asarray(r, float), np.asarray(y, float)
    keep = (r >= r_min) & np.isfinite(y)
    nan = (float("nan"), float("nan"))
    if keep.sum() < 2:
        return nan
    r, y = r[keep], y[keep]
    k = min(SMOOTH_BINS, len(y))
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    s = np.convolve(pad, np.ones(k) / k, mode="valid")
    left = np.r_[-np.inf, s[:-1]]
    right = np.r_[s[1:], -np.inf]
    peaks = np.flatnonzero((s >= left) & (s > right))
    if not len(peaks):
        return nan
    heights = s[peaks] - null
    top = heights.max()
    if not top > 0:
        return nan
    first = peaks[np.flatnonzero(heights >= PEAK_FRACTION * top)[0]]
    return float(r[first]), float(s[first])


# ------------------------------------------------------- the whole thing

@dataclass
class Points:
    """One layer's points in one file: positions (n, 2), the blink each is
    in (or None), and the row of the layer's table each is (to recognise
    two layers' common rows)."""
    xy: np.ndarray
    groups: Optional[np.ndarray] = None
    rows: Optional[np.ndarray] = None
    table: Any = None
    z: Optional[np.ndarray] = None


@dataclass
class Curve:
    """One auto- or cross-correlation: what `spatial_statistics` measured."""
    name: str
    key: str                       # "" / "_L2" / "_L1xL2": the values' suffix
    a: int
    b: int
    counts: np.ndarray
    expected: np.ndarray
    g: np.ndarray
    K: np.ndarray
    L_r: np.ndarray
    sims: Optional[Dict[str, np.ndarray]] = None   # name -> (n_sims, bins)
    band: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    test: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_cross(self) -> bool:
        return self.a != self.b


def _same_rows(pa: Points, pb: Points) -> Optional[np.ndarray]:
    """For two layers over one table: per point of a, the point of b that is
    the same row, or -1."""
    if pa.table is None or pa.table is not pb.table or pa.rows is None or pb.rows is None:
        return None
    where = np.full(int(max(pa.rows.max(initial=-1), pb.rows.max(initial=-1))) + 1, -1,
                    dtype=np.int64)
    where[pb.rows] = np.arange(len(pb.rows))
    return where[pa.rows]


def spatial_statistics(layers: Sequence[Sequence[Points]], windows: Sequence[Window],
                       r_max: float, dr: float, cross: bool = True, null: str = "csr",
                       n_sims: int = 99, level: float = 0.95, r_min: float = 0.0,
                       seed: int = 0, names: Optional[Sequence[str]] = None,
                       progress: Optional[Callable[[float], None]] = None
                       ) -> Tuple[np.ndarray, List[Curve]]:
    """The radial statistics of every layer and pair of layers.

    ``layers[i][f]`` is layer i's `Points` in file f, ``windows[f]`` file
    f's window.  Each file's pair counts and expected counts are added, and
    g, K and L - r are of the sums.  ``null`` is "none", "csr" or
    "keep_blinks" (each layer's `Points.groups`); for a cross-correlation
    the second layer is redistributed and the first stays.  Returns the bin
    edges and one `Curve` per auto-correlation, then per pair.
    ``progress(fraction)`` may raise to stop.
    """
    bins = int(round(r_max / dr))
    edges = dr * np.arange(bins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    n_layers = len(layers)
    names = list(names) if names else [f"layer {i + 1}" for i in range(n_layers)]
    jobs = [(i, i) for i in range(n_layers)]
    if cross:
        jobs += [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]
    sims_on = null != "none" and n_sims > 0
    total = len(jobs) * (1 + (n_sims if sims_on else 0))
    done = 0
    rng = np.random.default_rng(seed)
    out = []
    for a, b in jobs:
        counts = np.zeros(bins)
        expected = np.zeros(bins)
        sim_counts = np.zeros((n_sims, bins)) if sims_on else None
        sim_expected = np.zeros(bins)
        for f, window in enumerate(windows):
            pa, pb = layers[a][f], layers[b][f]
            na, nb = len(pa.xy), len(pb.xy)
            if a == b:
                if na < 2:
                    continue
                counts += pair_counts(pa.xy, None, r_max, dr)
                ordered = na * (na - 1.0)
            else:
                if not na or not nb:
                    continue
                same = _same_rows(pa, pb)
                counts += pair_counts(pa.xy, pb.xy, r_max, dr, same)
                ordered = na * float(nb) - (0 if same is None else np.sum(same >= 0))
            e = expected_counts(window, ordered, edges)
            expected += e
            if not sims_on:
                continue
            if null == "keep_blinks" and pb.groups is None:
                raise ValueError(f"{names[b]} has no group_id to keep its blinks "
                                 "together: merge blinks on the layer once, or "
                                 "choose CSR as the null")
            sim_expected += expected_counts(window, na * (na - 1.0) if a == b
                                            else na * float(nb), edges)
            for k in range(n_sims):
                moved = (keep_blinks(window, pb.xy, pb.groups, rng)
                         if null == "keep_blinks" else csr(window, nb, rng))
                sim_counts[k] += pair_counts(moved, None, r_max, dr) if a == b else \
                    pair_counts(pa.xy, moved, r_max, dr)
                if progress is not None:
                    progress((done + (f + (k + 1) / n_sims) / len(windows) * n_sims)
                             / total)
        done += 1 + (n_sims if sims_on else 0)
        if progress is not None:
            progress(done / total)
        c = curves_of(counts, expected, edges)
        if a == b:
            name, key = names[a], ("" if a == 0 else f"_L{a + 1}")
        else:
            name, key = f"{names[a]} x {names[b]}", f"_L{a + 1}xL{b + 1}"
        curve = Curve(name, key, a, b, counts, expected, c["g"], c["K"], c["L_r"])
        if sims_on and sim_expected.any():
            per = [curves_of(s, sim_expected, edges) for s in sim_counts]
            curve.sims = {q: np.array([p[q] for p in per]) for q in ("g", "K", "L_r")}
            curve.band = {q: pointwise_band(curve.sims[q], level) for q in curve.sims}
            curve.test = mad_test(curve.L_r, curve.sims["L_r"], mid >= r_min, level)
        out.append(curve)
    return edges, out


def curve_values(curve: Curve, mid: np.ndarray, at: Sequence[float], r_min: float,
                 approaches: Sequence[str]) -> Dict[str, float]:
    """The numbers of one curve for the region table (`Result.values`)."""
    k = curve.key
    v: Dict[str, float] = {}
    if "pair_correlation" in approaches:
        r, h = first_maximum(mid, curve.g, 1.0, r_min)
        v[f"g_max_r_nm{k}"], v[f"g_max{k}"] = r, h
        for d in at:
            v[f"g_at_{d:g}nm{k}"] = _at(mid, curve.g, d)
    if "ripley_l" in approaches:
        r, h = first_maximum(mid, curve.L_r, 0.0, r_min)
        v[f"lr_max_r_nm{k}"], v[f"lr_max_nm{k}"] = r, h
        for d in at:
            v[f"lr_at_{d:g}nm{k}"] = _at(mid, curve.L_r, d)
    if curve.test:
        p = curve.test["p"]
        v[f"p_csr{k}"] = p
        v[f"leaves_envelope{k}"] = (float(curve.test["T"] > curve.test["T_crit"])
                                    if np.isfinite(p) else float("nan"))
    return v


def _at(r: np.ndarray, y: np.ndarray, d: float) -> float:
    ok = np.isfinite(y)
    if not ok.any() or d < r[ok][0] or d > r[ok][-1]:
        return float("nan")
    return float(np.interp(d, r[ok], y[ok]))


def parse_distances(text: str) -> List[float]:
    """"190, 380" -> [190.0, 380.0]; empty -> []."""
    out = []
    for part in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        if part.strip():
            try:
                value = float(part)
            except ValueError:
                raise ValueError(f"at distances: {part!r} is not a number") from None
            if value > 0:
                out.append(value)
    return out


def expected_seconds(n_points: Sequence[int], areas: Sequence[float], r_max: float,
                     n_sims: int, cross: bool) -> float:
    """A rough run time.  Measured on 4 cores: about 0.25 us a point to sort
    and index, and 1.5 ns a candidate pair -- a point tests the nine cells
    r_max wide around it -- plus 0.15 us a point to place it in a
    simulation."""
    passes = []
    k = len(n_points)
    for i in range(k):
        passes.append((n_points[i], n_points[i], areas[i]))
    if cross:
        for i in range(k):
            for j in range(i + 1, k):
                passes.append((n_points[i], n_points[j], areas[j]))
    seconds = 0.0
    for na, nb, area in passes:
        one = 0.25e-6 * (na + nb) + 1.5e-9 * na * 9 * r_max ** 2 * nb / max(area, 1.0)
        seconds += one * (1 + n_sims) + 0.15e-6 * nb * n_sims
    return seconds


# ---------------------------------------------------------------- drawing

def draw_curves(ax, mid: np.ndarray, curves: Sequence[Curve], quantity: str,
                colours: Sequence[str]) -> None:
    """One quantity (``g``, ``K``, ``L_r``) of every curve, with its
    pointwise band shaded, the global band of L - r dashed, and the value
    CSR expects as a thin black line."""
    labels = {"g": "g(r)", "K": "K(r) (nm²)", "L_r": "L(r) - r (nm)"}
    for curve, colour in zip(curves, colours):
        y = getattr(curve, quantity)
        style = "--" if curve.is_cross else "-"
        if quantity in curve.band:
            lo, hi = curve.band[quantity]
            ax.fill_between(mid, lo, hi, color=colour, alpha=0.18, linewidth=0)
        if quantity == "L_r" and curve.test and np.isfinite(curve.test["T_crit"]):
            mean, t = curve.test["mean"], curve.test["T_crit"]
            ax.plot(mid, mean + t, ":", color=colour, linewidth=0.8)
            ax.plot(mid, mean - t, ":", color=colour, linewidth=0.8)
        label = curve.name
        if curve.test and np.isfinite(curve.test["p"]):
            label += f" (p = {curve.test['p']:.2g})"
        ax.plot(mid, y, style, color=colour, linewidth=1.2, label=label)
    if quantity == "g":
        ax.axhline(1.0, color="k", linewidth=0.6)
    elif quantity == "L_r":
        ax.axhline(0.0, color="k", linewidth=0.6)
    else:
        ax.plot(mid, np.pi * mid ** 2, color="k", linewidth=0.6)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(labels[quantity])
    ax.set_xlim(0, mid[-1] + 0.5 * (mid[1] - mid[0] if len(mid) > 1 else 1))
    ax.legend(fontsize=7, frameon=False)


def draw_window(ax, window: Window, points: Sequence[Tuple[np.ndarray, str]]) -> None:
    """The window (grey) and the points measured in it."""
    x0, y0, x1, y1 = window.extent
    ax.imshow(window.mask, origin="lower", extent=(x0, x1, y0, y1), cmap="Greys",
              vmin=0, vmax=4, interpolation="nearest")
    for xy, colour in points:
        step = max(1, len(xy) // FIGURE_POINTS)
        ax.plot(xy[::step, 0], xy[::step, 1], ".", color=colour, markersize=1.5,
                alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")


def _axis_label(axis: str, turned: bool) -> str:
    if axis == "z":
        return "shift in z (nm)"
    if turned:
        return f"shift {'along' if axis == 'x' else 'across'} (nm)"
    return f"shift in {axis} (nm)"


def draw_images(figure, found, jobs, turned: bool = False) -> None:
    """Each correlation image, g in colour, shift 0 in the middle."""
    rows, across = int(np.ceil(len(jobs) / 3)), min(len(jobs), 3)
    axes = figure.subplots(rows, across, squeeze=False).ravel()
    lim = float(found.lags[-1])
    a, b = found.image_axes
    for ax, (i, j) in zip(axes, jobs):
        g = found.images[(i, j)]
        finite = g[np.isfinite(g)]
        top = float(np.percentile(finite, 99.5)) if len(finite) else 1.0
        shown = ax.imshow(g.T, origin="lower", extent=(-lim, lim, -lim, lim),
                          cmap="magma", vmin=0, vmax=max(top, 1.0),
                          interpolation="nearest")
        figure.colorbar(shown, ax=ax, shrink=0.8, label="g")
        name = found.names[i] if i == j else f"{found.names[i]} x {found.names[j]}"
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(_axis_label(a, turned), fontsize=8)
        ax.set_ylabel(_axis_label(b, turned), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(jobs):]:
        ax.set_visible(False)


def draw_line(figure, found, colours: Sequence[str], turned: bool = False) -> None:
    """The profiles, the correlations with their periods marked, and the
    spectra."""
    prof, corr, spec = figure.subplots(1, 3)
    axis = found.line_axis
    for i, (x, counts) in enumerate(found.profiles):
        if len(x):
            prof.stairs(counts, np.r_[x - found.bin / 2, x[-1] + found.bin / 2],
                        color=colours[i], label=found.names[i])
    for (i, j), g in found.line.items():
        colour = colours[i] if i == j else "0.3"
        at, height = found.periods[(i, j)]
        label = (f"{found.names[i]}: {at:.0f} nm" if i == j
                 else f"{found.names[i]} x {found.names[j]}: {at:.0f} nm")
        corr.plot(found.lags, g, "-" if i == j else "--", color=colour, linewidth=1.1,
                  label=label)
        if np.isfinite(at):
            corr.plot([at], [height], "o", color=colour, markersize=4)
    corr.axhline(1.0, color="k", linewidth=0.5)
    for i, (f, mag) in enumerate(found.spectra):
        if len(f) > 1:
            spec.plot(f[1:] * 1000, mag[1:], color=colours[i], linewidth=1.0)
            p, m = found.fft_periods[i]
            if np.isfinite(p):
                spec.plot([1000 / p], [m], "o", color=colours[i], markersize=4)
    where = ("along the line" if turned and axis == "x" else
             "across the line" if turned and axis == "y" else axis)
    prof.set_xlabel(f"{where} (nm)", fontsize=8)
    prof.set_ylabel("localizations per bin", fontsize=8)
    corr.set_xlabel(_axis_label(axis, turned), fontsize=8)
    corr.set_ylabel("g", fontsize=8)
    spec.set_xlabel("frequency (1/µm)", fontsize=8)
    spec.set_ylabel("magnitude", fontsize=8)
    corr.legend(fontsize=7, frameon=False)
    for ax in (prof, corr, spec):
        ax.tick_params(labelsize=7)


# ---------------------------------------------------------------- the plugin

@dataclass
class RadialSettings:
    r_max_nm: float = param(500.0, label="up to r", unit="nm", min=1.0,
                            help="the largest distance; pairs farther apart are "
                                 "not counted")
    dr_nm: float = param(5.0, label="bin width", unit="nm", min=0.1,
                         help="the width of the rings the pairs are counted in")
    cross: bool = param(True, label="cross-correlations",
                        help="also every pair of visible layers, not only each "
                             "layer with itself")
    at_nm: str = param("", label="at distances", unit="nm",
                       help="g and L - r at these distances as values, e.g. "
                            "'95, 190'")
    window: str = param("occupied", label="without an ROI",
                        choices=(("occupied", "the occupied area"),
                                 ("box", "the bounding box")),
                        help="the window when neither a region nor a drawn ROI "
                             "gives one")
    occupied_bin_nm: float = param(200.0, label="occupied bin", unit="nm", min=1.0,
                                   advanced=True,
                                   help="the pixel of the occupied area's mask")
    null: str = param("csr", label="envelope",
                      choices=(("none", "none"), ("csr", "random (CSR)"),
                               ("keep_blinks", "random, keep blinks")),
                      help="what the curves are compared with: points placed at "
                           "random in the window, or whole blinks placed at random")
    simulations: int = param(99, label="simulations", min=1,
                             help="random placements for the envelope")
    level: float = param(0.95, label="level", min=0.5, max=0.999, advanced=True,
                         help="of the envelope and the test")
    r_min_nm: float = param(0.0, label="from r", unit="nm", min=0.0, advanced=True,
                            help="the maxima and the test look from here on")
    seed: int = param(0, label="seed", min=0, advanced=True,
                      help="of the random placements")


def _roi_lines():
    return [("", "the first it has")]


@dataclass
class DirectionalSettings:
    plane: str = param("xy", label="image plane",
                       choices=(("xy", "x-y"), ("xz", "x-z"), ("yz", "y-z")),
                       help="x along a line ROI, y across it; z from z_nm")
    axis: str = param("x", label="line along", choices=(("x", "x"), ("y", "y"),
                                                       ("z", "z")),
                      help="x along a line ROI, y across it; z from z_nm")
    bin_nm: float = param(5.0, label="bin", unit="nm", min=0.1,
                          help="the localizations are counted in bins this wide")
    max_shift_nm: float = param(500.0, label="shifts up to", unit="nm", min=1.0,
                                help="the largest shift correlated")
    period_nm: float = param(0.0, label="period guess", unit="nm", min=0.0,
                             help="the peak nearest it is the period; 0: the "
                                  "first strong one")
    cross: bool = param(True, label="cross-correlations",
                        help="also every pair of visible layers")
    window: str = param("occupied", label="without an ROI",
                        choices=(("occupied", "the occupied area"),
                                 ("box", "the bounding box")),
                        help="the window when neither a region nor a drawn ROI "
                             "gives one")
    roi_line: str = param("", label="ROI line", choices=_roi_lines,
                          help="in an ROI of the ROI manager, which of its named "
                               "lines x runs along")


@dataclass
class SpatialStatisticsSettings:
    pair_correlation: bool = param(True, label="pair correlation g(r)",
                                   help="pairs at distance r over what chance gives")
    ripley_k: bool = param(False, label="Ripley's K(r)",
                           help="neighbours within r over the density")
    ripley_l: bool = param(True, label="L(r) - r",
                           help="K made linear: 0 for random, above it clustered")
    radial: RadialSettings = param(default_factory=RadialSettings,
                                   label="radial approaches")
    image: bool = param(False, label="correlation image",
                        help="the 2D auto- and cross-correlation, every "
                             "direction apart")
    line: bool = param(False, label="along a line",
                       help="the correlation along one axis, its period "
                            "and spectrum")
    directional: DirectionalSettings = param(default_factory=DirectionalSettings,
                                             label="directional approaches")


def _directional(settings: SpatialStatisticsSettings) -> List[str]:
    return [key for key, _ in DIRECTIONAL if getattr(settings, key)]


def _approaches(settings: SpatialStatisticsSettings) -> List[str]:
    return [key for key, _ in APPROACHES if getattr(settings, key)]


@register("Analysis/Measure/Spatial statistics")
class SpatialStatistics(Plugin):
    """Whether the localizations are clustered, and at what scale: the pair
    correlation g(r), Ripley's K(r) and L(r) - r of every visible layer and
    every pair, edge-corrected for the window, with envelopes from random
    placements."""

    Settings = SpatialStatisticsSettings
    changes_locs = False
    per_region = True
    version = "2"

    def describe(self, ctx: Optional[Context], settings: SpatialStatisticsSettings
                 ) -> FormState:
        s = settings.radial
        state = FormState(shown={"radial": bool(_approaches(settings)),
                                 "directional": bool(_directional(settings))})
        sims = s.null != "none"
        state.shown.update({"radial.simulations": sims, "radial.level": sims,
                            "radial.seed": sims,
                            "radial.occupied_bin_nm": s.window == "occupied",
                            "directional.plane": settings.image,
                            "directional.axis": settings.line,
                            "directional.period_nm": settings.line})
        project = getattr(ctx, "rois", None) if ctx is not None else None
        names = list(getattr(project, "line_names", None) or [])
        state.choices["directional.roi_line"] = ([("", "the first it has")]
                                                 + [(n, n) for n in names])
        return state

    # -- reading the layers

    def _layers(self, ctx: Context):
        from .histogram import _layers
        return _layers(ctx)

    def _drawn_roi(self, ctx: Context, selection):
        """The shape of the window when the run has one: the region's, or
        the drawn ROI's that cut the selection."""
        from ..regions import Region
        if ctx.region is not None:
            return ctx.region if isinstance(ctx.region, Region) else None
        if ctx.per_region:
            return None
        roi = getattr(ctx.session, "roi", None)
        if isinstance(roi, Region) and getattr(selection, "roi", None) is not None:
            return roi
        return None

    def _gather(self, ctx: Context, how: str = "occupied",
                bin_nm: float = OCCUPIED_BIN_NM):
        """``(names, colours, layers[i][f] Points, windows[f], files, shape)``:
        the windows from the drawn shape, else the localizations (``how``,
        `window_of_points`)."""
        found = self._layers(ctx)
        names, colours, columns = [], [], []
        shape = None
        for name, colour, (table, selection) in found:
            for c in ("x_nm", "y_nm"):
                if c not in table:
                    raise ValueError(f"layer {name} has no column {c!r}; it has "
                                     + ", ".join(sorted(table.keys())))
            mask = np.asarray(selection.mask, dtype=bool)
            x = np.asarray(table["x_nm"], dtype=float)
            y = np.asarray(table["y_nm"], dtype=float)
            mask = mask & np.isfinite(x) & np.isfinite(y)
            rows = np.flatnonzero(mask)
            files = (np.asarray(table["filenumber"])[rows] if "filenumber" in table
                     else np.zeros(len(rows), dtype=np.int64))
            groups = (np.asarray(table["group_id"])[rows] if "group_id" in table
                      else None)
            z = np.asarray(table["z_nm"], dtype=float)[rows] if "z_nm" in table else None
            names.append(name)
            colours.append(colour)
            columns.append((table, rows, x[rows], y[rows], files, groups, z))
            if shape is None:
                shape = self._drawn_roi(ctx, selection)
        all_files = np.unique(np.concatenate([c[4] for c in columns])) if columns else []
        layers: List[List[Points]] = [[] for _ in columns]
        windows: List[Window] = []
        used_files = []
        for f in all_files:
            per = []
            for table, rows, x, y, files, groups, z in columns:
                here = files == f
                per.append(Points(np.column_stack([x[here], y[here]]),
                                  None if groups is None else groups[here],
                                  rows[here], table, None if z is None else z[here]))
            if not any(len(p.xy) for p in per):
                continue
            if shape is not None:
                window = window_of_region(shape)
            else:
                xy = np.concatenate([p.xy for p in per])
                window = window_of_points(xy[:, 0], xy[:, 1], how, bin_nm)
            for i, p in enumerate(per):
                layers[i].append(p)
            windows.append(window)
            used_files.append(f)
        return names, colours, layers, windows, used_files, shape

    def pool(self, sums: Dict[str, np.ndarray], x: np.ndarray) -> Dict[str, np.ndarray]:
        """g, K and L - r of many regions from their pair counts and expected
        counts summed (`Plugin.pool`): the regions' pairs over the pairs CSR
        expects in all of them, which is g of their union when their windows
        do not overlap -- each curve's numerator and denominator are sums
        over files already, and regions are pooled the same way."""
        x = np.asarray(x, dtype=float)
        if len(x) < 1:
            return {}
        dr = 2.0 * x[0]                       # the centres start half a bin in
        edges = np.concatenate([[0.0], x + dr / 2])
        out: Dict[str, np.ndarray] = {}
        for name in sums:
            if not name.startswith("pairs"):
                continue
            k = name[len("pairs"):]
            if f"expected{k}" not in sums:
                continue
            c = curves_of(np.asarray(sums[name], float),
                          np.asarray(sums[f"expected{k}"], float), edges)
            out.update({f"g{k}": c["g"], f"K_nm2{k}": c["K"], f"lr_nm{k}": c["L_r"]})
        return out

    def preflight(self, ctx: Context, settings: SpatialStatisticsSettings):
        if not _approaches(settings) or ctx.per_region:
            return None
        s = settings.radial
        try:
            found = self._layers(ctx)
            counts, areas = [], []
            for _, _, (table, selection) in found:
                mask = np.asarray(selection.mask, bool)
                x = np.asarray(table["x_nm"], float)[mask]
                y = np.asarray(table["y_nm"], float)[mask]
                counts.append(int(mask.sum()))
                areas.append(float(max(np.ptp(x), 1.0) * max(np.ptp(y), 1.0))
                             if len(x) else 1.0)
        except Exception:                                  # noqa: BLE001
            return None             # the run says what is wrong
        sims = s.simulations if s.null != "none" else 0
        seconds = expected_seconds(counts, areas, s.r_max_nm, sims, s.cross)
        if seconds < ASK_ABOVE_S:
            return None
        took = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} minutes"
        choices = []
        if sims:
            bare = expected_seconds(counts, areas, s.r_max_nm, 0, s.cross)
            choices.append(PreflightChoice(
                label="without the envelope",
                settings=replace(settings, radial=replace(s, null="none")),
                help=f"the curves alone, about {bare:.0f} s"))
            if sims > 19:
                few = expected_seconds(counts, areas, s.r_max_nm, 19, s.cross)
                choices.append(PreflightChoice(
                    label="19 simulations",
                    settings=replace(settings, radial=replace(s, simulations=19)),
                    help=f"a coarser envelope (5 % from its extremes), about "
                         f"{few:.0f} s"))
        return PreflightQuestion(
            text=f"{sum(counts):,} localizations to r = {s.r_max_nm:g} nm"
                 + (f" with {sims} simulations" if sims else "")
                 + f": about {took}.  A smaller r is faster.", choices=choices)

    def run(self, ctx: Context, settings: SpatialStatisticsSettings) -> Result:
        """The radial approaches, then the directional ones, as one result."""
        radial, directional = _approaches(settings), _directional(settings)
        if not radial and not directional:
            raise ValueError("tick an approach")
        parts = []
        if radial:
            parts.append(self._run_radial(ctx, settings))
        if directional:
            parts.append(self._run_directional(ctx, settings))
        if len(parts) == 1:
            return parts[0]
        first, second = parts
        plots = dict(first.plots)
        plots[second.plot.name or "directional"] = second.plot
        plots.update(second.plots)
        return Result(text=first.text + "\n" + second.text, plot=first.plot, plots=plots,
                      values={**first.values, **second.values},
                      arrays={**first.arrays, **second.arrays}, settings=settings,
                      data={**first.data, **second.data}, title="spatial statistics")

    def _run_radial(self, ctx: Context, settings: SpatialStatisticsSettings) -> Result:
        approaches = _approaches(settings)
        s = settings.radial
        if s.dr_nm <= 0 or s.r_max_nm < s.dr_nm:
            raise ValueError("up to r must be at least one bin width")
        at = parse_distances(s.at_nm)
        names, colours, layers, windows, files, shape = self._gather(
            ctx, s.window, s.occupied_bin_nm)
        if not windows:
            if ctx.per_region:
                return Result(values={}, settings=settings)
            raise ValueError("no localizations to measure")
        started = time.monotonic()

        def progress(fraction: float) -> None:
            if ctx.stopped:
                from . import Stopped
                raise Stopped("spatial statistics stopped: nothing measured")
            if not ctx.per_region:
                ctx.report(f"spatial statistics: {100 * fraction:.0f}%")

        edges, curves = spatial_statistics(
            layers, windows, s.r_max_nm, s.dr_nm, s.cross, s.null, s.simulations,
            s.level, s.r_min_nm, s.seed, names, progress)
        mid = 0.5 * (edges[1:] + edges[:-1])
        values: Dict[str, float] = {}
        arrays: Dict[str, np.ndarray] = {"r_nm": mid}
        lines = []
        for curve in curves:
            values.update(curve_values(curve, mid, at, s.r_min_nm, approaches))
            k = curve.key
            arrays.update({f"pairs{k}": curve.counts, f"expected{k}": curve.expected,
                           f"g{k}": curve.g, f"K_nm2{k}": curve.K, f"lr_nm{k}": curve.L_r})
            for q, short in (("g", "g"), ("L_r", "lr_nm")):
                if q in curve.band:
                    arrays[f"{short}_lo{k}"], arrays[f"{short}_hi{k}"] = curve.band[q]
            lines.append(self._line(curve, mid, approaches, s.r_min_nm))
        area = sum(w.area for w in windows)
        where = (f"in the {shape}" if shape is not None else
                 f"in the {'occupied area' if s.window == 'occupied' else 'bounding box'}")
        head = (f"{where}, {area / 1e6:.4g} µm² over {len(windows)} file"
                f"{'s' if len(windows) != 1 else ''}; r to {edges[-1]:g} nm in "
                f"{s.dr_nm:g} nm bins"
                + ("" if s.null == "none" else
                   f"; {s.simulations} simulations ({'keep blinks' if s.null == 'keep_blinks' else 'CSR'}), "
                   f"test at {100 * (1 - s.level):.3g} %")
                + f"; {time.monotonic() - started:.1f} s")
        auto_colours = [colours[c.a] if not c.is_cross else
                        ("0.25" if i % 2 == 0 else "0.55")
                        for i, c in enumerate(curves)]
        titles = {"pair_correlation": ("g(r)", "g"), "ripley_k": ("K(r)", "K"),
                  "ripley_l": ("L(r) - r", "L_r")}
        plots: Dict[str, Any] = {}
        first = None
        for key in approaches:
            title, q = titles[key]

            def draw(ax, q=q):
                draw_curves(ax, mid, curves, q, auto_colours)
            p = Plot(draw, name=title, size=(5.2, 3.6))
            if first is None:
                first = p
            else:
                plots[title] = p
        shown = windows[0]
        points = [(layers[i][0].xy, colours[i]) for i in range(len(layers))]

        def window_plot(ax):
            draw_window(ax, shown, points)
            ax.set_title("the window" + (" (first file)" if len(windows) > 1 else ""),
                         fontsize=9)
        plots["window"] = Plot(window_plot, name="window", size=(4.2, 4.0))
        return Result(text="\n".join([head] + lines), plot=first, plots=plots,
                      values=values, arrays=arrays, settings=settings,
                      data={"r_nm": mid, "edges": edges, "curves": curves,
                            "area_nm2": area, "files": list(files)},
                      title="spatial statistics")

    def _frame(self, ctx: Context, d: "DirectionalSettings", shape):
        """``(frame, shape)``: the coordinates x and y are taken in, and the
        window's shape.  A line -- the drawn line ROI, the region when it is
        a line, else the ROI's named line (Line Profile's `region_line`) --
        turns x along it and y across; anything else leaves x and y the
        table's."""
        from ..correlation import Frame
        from ..regions import Region
        from .line_profile import line_ends, region_line
        if ctx.per_region:
            region = ctx.region if isinstance(ctx.region, Region) else None
            if region is not None and (region.kind == "line" or d.roi_line
                                       or self._has_lines(ctx)):
                p0, p1, _ = line_ends(region_line(ctx, d.roi_line))
                return Frame.of_line(p0, p1), region
            return Frame(), region
        if shape is not None and shape.kind == "line":
            p0, p1, _ = line_ends(shape)
            return Frame.of_line(p0, p1), shape
        return Frame(), shape

    @staticmethod
    def _has_lines(ctx: Context) -> bool:
        project = ctx.rois
        roi = (project.rois.get(ctx.region_id) if project is not None
               and ctx.region_id is not None else None)
        return bool(getattr(roi, "lines", None))

    def _run_directional(self, ctx: Context, settings: SpatialStatisticsSettings
                         ) -> Result:
        from ..correlation import directional, lateral_mask
        d = settings.directional
        if d.max_shift_nm < d.bin_nm:
            raise ValueError("shifts up to must be at least one bin")
        names, colours, layers, windows, files, shape = self._gather(
            ctx, d.window, OCCUPIED_BIN_NM)
        if not windows:
            if ctx.per_region:
                return Result(values={}, settings=settings)
            raise ValueError("no localizations to measure")
        frame, shape = self._frame(ctx, d, shape)
        masks, framed = [], [[] for _ in layers]
        for f, window in enumerate(windows):
            if shape is not None:
                masks.append(lateral_mask(shape, frame, d.bin_nm))
            else:
                x0, y0, x1, y1 = window.extent
                u, v = frame.to([x0, x1, x1, x0], [y0, y0, y1, y1])
                masks.append(lateral_mask(None, frame, d.bin_nm, window.contains,
                                          (u.min(), v.min(), u.max(), v.max())))
            for i, per in enumerate(layers):
                p = per[f]
                u, v = frame.to(p.xy[:, 0], p.xy[:, 1])
                one = {"u": u, "v": v}
                if p.z is not None:
                    one["z"] = p.z
                framed[i].append(one)
        image = d.plane if settings.image else None
        line = d.axis if settings.line else None
        found = directional(framed, masks, d.bin_nm, d.max_shift_nm, image, line,
                            d.cross, d.period_nm, names)
        values: Dict[str, float] = {}
        arrays: Dict[str, np.ndarray] = {}
        lines = [self._directional_head(found, frame, shape, image, line)]
        keys = {}
        for (i, j) in (found.line or found.images):
            keys[(i, j)] = ("" if i == 0 else f"_L{i + 1}") if i == j else \
                f"_L{i + 1}xL{j + 1}"
        if line:
            arrays["lag_nm"] = found.lags
            for (i, j), g in found.line.items():
                k = keys[(i, j)]
                arrays[f"line_g{k}"] = g
                at, height = found.periods[(i, j)]
                if i == j:
                    fp, fm = found.fft_periods[i]
                    values.update({f"period_nm{k}": at, f"period_g{k}": height,
                                   f"period_fft_nm{k}": fp, f"period_fft_mag{k}": fm})
                    lines.append(f"{names[i]}: period {at:.4g} nm (g {height:.3g}); "
                                 f"spectrum {fp:.4g} nm")
                else:
                    values.update({f"shift_nm{k}": at, f"shift_g{k}": height})
                    lines.append(f"{names[i]} x {names[j]}: strongest at "
                                 f"{at:.4g} nm (g {height:.3g})")
        plots: Dict[str, Any] = {}
        first = None
        cols = [colours[i] for i in range(len(names))]
        if image:
            jobs = list(found.images)
            p = Plot(lambda fig: draw_images(fig, found, jobs, frame.turned),
                     name="correlation image", panels=len(jobs),
                     size=(3.2 * min(len(jobs), 3), 3.0 * int(np.ceil(len(jobs) / 3))))
            first = p
        if line:
            p = Plot(lambda fig: draw_line(fig, found, cols, frame.turned),
                     name="along a line", panels=3, size=(9.6, 3.0))
            if first is None:
                first = p
            else:
                plots["along a line"] = p
        return Result(text="\n".join(lines), plot=first, plots=plots, values=values,
                      arrays=arrays, settings=settings,
                      data={"directional": found, "frame": frame},
                      title="spatial statistics")

    @staticmethod
    def _directional_head(found, frame, shape, image, line) -> str:
        axes = ("x along the line, y across it" if frame.turned else "x and y")
        parts = [f"{axes}; bins of {found.bin:g} nm, shifts to {found.lags[-1]:g} nm"]
        if found.z is not None:
            parts.append(f"z from {found.z[0]:.0f} to {found.z[1]:.0f} nm")
        if image:
            parts.append(f"image {image[0]}-{image[1]}")
        if line:
            parts.append(f"line along {line}")
        return "directional: " + "; ".join(parts)

    @staticmethod
    def _line(curve: Curve, mid: np.ndarray, approaches, r_min: float) -> str:
        parts = []
        if "pair_correlation" in approaches:
            r, h = first_maximum(mid, curve.g, 1.0, r_min)
            parts.append(f"g max {h:.3g} at {r:.0f} nm" if np.isfinite(r)
                         else "g never above 1")
        if "ripley_l" in approaches:
            r, h = first_maximum(mid, curve.L_r, 0.0, r_min)
            parts.append(f"L - r max {h:.3g} nm at {r:.0f} nm" if np.isfinite(r)
                         else "L - r never above 0")
        if curve.test and np.isfinite(curve.test["p"]):
            out = curve.test["T"] > curve.test["T_crit"]
            parts.append(f"p = {curve.test['p']:.2g}, "
                         + ("leaves the envelope" if out else "inside the envelope"))
        return f"{curve.name}: " + "; ".join(parts)


# ---------------------------------------- what Neighbourhood's count expects

# Gauss-Legendre nodes per pixel column for the points near the window's
# edge (`window_coverage`), and points done at once
COVERAGE_NODES = 8
COVERAGE_CHUNK = 4000


def window_coverage(window: Window, x, y, radius: float, gaussian: bool = False
                    ) -> np.ndarray:
    """At each point, how much of a circle of ``radius`` around it lies in
    the window, in nm^2 -- pi r^2 well inside, less near the edge -- or, with
    ``gaussian``, the integral of exp(-d^2 / 2 r^2) over the circle of 3 r
    (as Neighbourhood cuts it) that lies in the window.

    A point whose circle is inside the window -- its pixel in the mask eroded
    by the circle and a pixel more -- gets the whole circle's.  For the rest,
    the points near an edge, the integral over the window's pixels: across
    each pixel column the circle's chord is split by the pixel rows and
    integrated exactly in y; along x, `COVERAGE_NODES` Gauss-Legendre nodes
    per column.  (A grid of points over the circle was the first way, and
    missed a half circle by 3 %: the window's edges run along the grid.)"""
    from math import sqrt
    from scipy.ndimage import binary_erosion
    from scipy.special import erf as verf
    x, y = np.asarray(x, float), np.asarray(y, float)
    reach = 3.0 * radius if gaussian else radius
    if gaussian:
        full = 2 * np.pi * radius ** 2 * (1 - np.exp(-reach ** 2 / (2 * radius ** 2)))
    else:
        full = np.pi * radius ** 2
    px, py = window.px, window.py
    kx = int(np.ceil(reach / px)) + 1
    ky = int(np.ceil(reach / py)) + 1
    gx, gy = np.meshgrid(np.arange(-kx, kx + 1) * px, np.arange(-ky, ky + 1) * py)
    disc = np.hypot(np.maximum(np.abs(gx) - px, 0), np.maximum(np.abs(gy) - py, 0)) <= reach
    padded = np.pad(window.mask, ((ky, ky), (kx, kx)))
    core = binary_erosion(padded, structure=disc, border_value=0)[ky:-ky, kx:-kx]
    out = np.full(len(x), full)
    ix, iy = window.pixel_of(x, y)
    ny, nx = window.mask.shape
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    deep = np.zeros(len(x), dtype=bool)
    deep[ok] = core[iy[ok], ix[ok]]
    near = np.flatnonzero(~deep)
    nodes, weights = np.polynomial.legendre.leggauss(COVERAGE_NODES)
    cols = np.arange(-kx, kx + 1)
    rows = np.arange(-ky, ky + 1)
    s2 = sqrt(2.0) * radius
    for start in range(0, len(near), COVERAGE_CHUNK):
        at = near[start:start + COVERAGE_CHUNK]
        cx, cy, jx, jy = x[at], y[at], ix[at], iy[at]
        # each pixel column's span in x, relative to the point, cut to the circle
        lo = window.x0 + (jx[:, None] + cols[None, :]) * px - cx[:, None]
        a = np.clip(lo, -reach, reach)
        b = np.clip(lo + px, -reach, reach)
        half = 0.5 * (b - a)
        u = 0.5 * (a + b)[..., None] + half[..., None] * nodes          # (P, C, N)
        w = half[..., None] * weights
        chord = np.sqrt(np.clip(reach ** 2 - u ** 2, 0.0, None))
        # each pixel row's span in y, relative to the point
        ylo = window.y0 + (jy[:, None] + rows[None, :]) * py - cy[:, None]   # (P, R)
        lo_y = np.maximum(ylo[:, None, None, :], -chord[..., None])
        hi_y = np.minimum(ylo[:, None, None, :] + py, chord[..., None])
        if gaussian:
            inner = 0.5 * np.sqrt(np.pi) * radius * sqrt(2.0) * (
                verf(hi_y / s2) - verf(lo_y / s2))
            inner = np.where(hi_y > lo_y, inner, 0.0) * np.exp(-u ** 2 / s2 ** 2)[..., None]
        else:
            inner = np.clip(hi_y - lo_y, 0.0, None)                     # (P, C, N, R)
        m = padded[(jy[:, None] + rows[None, :] + ky)[:, None, :],
                   (jx[:, None] + cols[None, :] + kx)[:, :, None]]      # (P, C, R)
        out[at] = np.einsum("pcnr,pcn,pcr->p", inner, w, m.astype(float))
    return out
