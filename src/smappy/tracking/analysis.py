"""How tracks move: the mean squared displacement (MSD) and the distribution
of jumps, per track and over all of them.

Wave 4, step 19c (`docs/smap_porting/wave4_design.md`).  The table is sorted
once by track and then time, and every quantity is a ``bincount`` over it,
so 10^5 tracks cost what one long one would: no loop over tracks.

**Time.**  A step is one frame of the file's frame time (`metadata.
file_frame_time`).  A table with its own clock (MINFLUX's ``time_s``, where
``frame`` is only a rank) is put on a grid of its median interval within
tracks, each localization at the nearest step: `steps_of`.

**MSD.**  For lag k (frames), the mean over every pair of a track's
localizations exactly k frames apart of their squared distance -- all pairs,
overlapping, the usual estimator; pairs across a dark frame are counted at
their true lag.  For Brownian motion seen with a localization error sigma
per axis, in d dimensions,

    MSD(k) = 2 d D k dt + 2 sum_axes sigma_axis^2,

the offset 4 sigma^2 in 2D.  Two ways to the offset (*offset*):

* **from the precisions** (the default): the mean of the localizations'
  ``xy_err_nm^2`` (and ``z_err_nm^2``), so D is a fit of one parameter, the
  slope through the offset.  Michalet and Berglund (2012) show that a known
  sigma makes D more precise; the fitter's precision is what the rest of
  smappy trusts too.  It is the static error: motion blur lowers the true
  offset (Michalet 2010, its Appendix C), which this does not model;
* **fitted**: the intercept fitted with the slope, Michalet's (2010) case.

**How many lags.**  Michalet (2010) showed that the error of D from a fit
of the first p MSD points has a minimum in p, which depends on the track's
length N and on the reduced localization error x = sigma^2 / (D dt) -- 2
for x << 1, more when the error dominates -- and that any other choice can
be far worse.  His paper gives an empirical formula for that p, which could
not be checked here (the equations are missing from every text that was
reachable), so the optimum is computed instead, exactly, from the model:
the covariance of the MSD points of a Brownian track with Gaussian errors
follows from Isserlis' theorem (`msd_covariance`), the variance of either
estimator of D for each p from it, and p is the one with the least
(`optimal_lags`).  Since x needs D, Michalet's iteration: a first D with
p = N / 10, x from it, p from x, until p repeats (at most 10 rounds) --
vectorised over all tracks at once, the optimum tabulated per length.
A track with gaps is treated as the gap-free track over its span, which is
what Michalet suggests for few missing frames.

**alpha** (anomalous exponent) is the slope of log(MSD - offset) over
log(lag), on the same lags or two at least: 1 for free diffusion, below for
confined, above for directed motion.  It needs MSD above the offset, so it
is NaN for a track that hardly moves.

**Jump distances.**  The distances of every pair 1, 2, ... frames apart
(each lag up to *lags*, fitted together) are a mixture of 1 to 3 diffusive
states; in d dimensions a state's jump r over lag k has per axis the
variance s^2 = 2 D k dt + sigma_a^2 + sigma_b^2 (the two localizations'
errors, from the precisions), so r^2 / s^2 is chi-square with d degrees of
freedom.  D and the fraction of each state are fitted by maximum
likelihood on the jumps themselves (`fit_jumps`, by expectation
maximisation), not on a histogram (SMAP's AnalyzeSPT fitted a histogram
of D, Spot-On fits histograms with one localization error); the jumps of
one track are not independent, which the likelihood ignores, so its errors
would be too small and none are given.

**The focus.**  In 2D tracking a particle is seen only while it is within
a few hundred nm of the focal plane, and a fast one leaves within a few
frames: its state makes fewer jumps than its number of particles, and
fewer the longer the lag -- the fast fraction comes out tens of percent
low (Hansen et al. 2018, Spot-On).  With a focal range, a state's share of
the jumps at lag k is its fraction times the fraction of its particles
still in focus k frames on (`in_focus`, saSPT's frame-by-frame form,
Heckert et al. 2022), so the fractions fitted are of the particles.  Both
methods are re-implemented from their papers' descriptions, not ported.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

# the most MSD lags fitted; Michalet's optimum stays well below for the
# tracks single-molecule tracking gives
MAX_LAGS = 50
# tracks longer than this are taken as this long when the lags are chosen:
# the optimum hardly moves beyond
LONGEST = 2000
# the reduced localization errors the optimum is tabulated at
X_GRID = 10.0 ** np.arange(-3.0, 4.01, 0.125)


# ------------------------------------------------------------- the tracks

@dataclass
class Tracks:
    """A table's tracks, sorted by track then time: ``rows`` into the table,
    ``track`` (0, 1, ... per id in ``ids``), ``step`` (frames since the
    track's first), positions and each localization's error variance per
    axis summed over the axes (nm^2)."""
    ids: np.ndarray
    rows: np.ndarray
    track: np.ndarray
    step: np.ndarray
    xyz: np.ndarray
    err2: np.ndarray            # sum over axes of sigma_axis^2, per row
    dt_s: float
    err_axis: Optional[np.ndarray] = None     # sigma_axis^2, per row and axis

    def axis_errors(self) -> np.ndarray:
        """Each localization's error variance per axis (rows x axes, nm^2):
        kept by `gather`, else the sum shared out evenly."""
        if self.err_axis is not None:
            return self.err_axis
        return np.repeat((self.err2 / self.dims)[:, None], self.dims, axis=1)

    @property
    def dims(self) -> int:
        return self.xyz.shape[1]

    @property
    def n_tracks(self) -> int:
        return len(self.ids)

    def counts(self) -> np.ndarray:
        return np.bincount(self.track, minlength=self.n_tracks)

    def spans(self) -> np.ndarray:
        """Frames from a track's first localization to its last, plus one."""
        last = np.zeros(self.n_tracks, np.int64)
        np.maximum.at(last, self.track, self.step)
        return last + 1


def steps_of(ids: np.ndarray, time_s: np.ndarray) -> Tuple[np.ndarray, float]:
    """An irregular clock put on a grid: each localization's step (an
    integer, from 0 within its track) and the grid's interval, the median
    interval between consecutive localizations of a track."""
    order = np.lexsort((time_s, ids))
    t, i = time_s[order], ids[order]
    same = i[1:] == i[:-1]
    gaps = np.diff(t)[same]
    gaps = gaps[gaps > 0]
    if not len(gaps):
        raise ValueError("no track has two localizations at different times")
    dt = float(np.median(gaps))
    first = np.empty(len(t))
    starts = np.concatenate([[True], ~same])
    first_t = t[starts]
    first[:] = first_t[np.cumsum(starts) - 1]
    step = np.empty(len(t), np.int64)
    step[order] = np.round((t - first) / dt).astype(np.int64)
    return step, dt


def gather(ids, frame, x, y, z=None, err_xy=None, err_z=None,
           frame_time_s: float = 1.0, time_s=None) -> Tracks:
    """The localizations with an id (-1, `ids.NONE`, is none) and finite positions, as
    `Tracks`.  ``err_xy`` is the lateral precision per axis (``xy_err_nm``),
    ``err_z`` the axial; missing ones count as 0.  With ``time_s`` the clock
    is that column's, on a grid (`steps_of`), not the frames."""
    ids = np.asarray(ids, np.int64)
    coords = [np.asarray(x, float), np.asarray(y, float)]
    if z is not None:
        coords.append(np.asarray(z, float))
    xyz = np.column_stack(coords)
    keep = (ids >= 0) & np.isfinite(xyz).all(axis=1)
    rows = np.flatnonzero(keep)
    ids_k = ids[rows]
    if time_s is not None:
        step, dt = steps_of(ids_k, np.asarray(time_s, float)[rows])
    else:
        step, dt = np.asarray(frame, np.int64)[rows], float(frame_time_s)
    order = np.lexsort((step, ids_k))
    rows, ids_k, step = rows[order], ids_k[order], step[order]
    unique, track = np.unique(ids_k, return_inverse=True)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(track)) + 1]) if len(track) else \
        np.zeros(0, np.int64)
    step = step - step[starts][track] if len(track) else step
    axes = np.zeros((len(rows), xyz.shape[1]))
    if err_xy is not None:
        axes[:, :2] = np.nan_to_num(np.asarray(err_xy, float)[rows])[:, None] ** 2
    if z is not None and err_z is not None:
        axes[:, 2] = np.nan_to_num(np.asarray(err_z, float)[rows]) ** 2
    return Tracks(unique, rows, track, step, xyz[rows], axes.sum(axis=1), dt, axes)


def lag_sums(t: Tracks, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per track and lag 1..``max_lag`` (frames), the sum of squared
    displacements (nm^2) and the number of pairs: two (tracks, lags)
    arrays.  Every pair exactly k frames apart; a track's localizations are
    in increasing frames, so a pair k frames apart is at most k rows apart."""
    T, K = t.n_tracks, int(max_lag)
    sums = np.zeros(T * K)
    counts = np.zeros(T * K)
    for m in range(1, K + 1):
        if m >= len(t.track):
            break
        same = t.track[m:] == t.track[:-m]
        lag = t.step[m:] - t.step[:-m]
        use = same & (lag <= K) & (lag >= 1)
        if not use.any():
            continue
        a = np.flatnonzero(use)
        d2 = ((t.xyz[a + m] - t.xyz[a]) ** 2).sum(axis=1)
        where = t.track[a] * K + lag[a] - 1
        sums += np.bincount(where, weights=d2, minlength=T * K)
        counts += np.bincount(where, minlength=T * K)
    return sums.reshape(T, K), counts.reshape(T, K)


# --------------------------------------------- how many lags (Michalet 2010)

@lru_cache(maxsize=256)
def _msd_sums(n: int, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The parts of the MSD's covariance that do not depend on x, for a
    gap-free track of n positions and lags 1..p: per pair of lags (m, l),
    sums over every pair of displacements of the number of frames they
    share squared (S1), shared times the noise term (S2) and the noise term
    squared (S3).  See `msd_covariance`."""
    s1, s2, s3 = (np.zeros((p, p)) for _ in range(3))
    for m in range(1, p + 1):
        for l in range(m, p + 1):
            s = np.arange(-(n - m - 1), n - l)                 # j - i
            count = np.minimum(n - m, n - l - s) - np.maximum(0, -s)
            count = np.maximum(count, 0)
            shared = np.maximum(0, np.minimum(m, s + l) - np.maximum(0, s))
            noise = ((s == m - l).astype(float) - (s == m) - (s == -l) + (s == 0))
            v = (np.sum(count * shared ** 2), np.sum(count * shared * noise),
                 np.sum(count * noise ** 2))
            for out, value in zip((s1, s2, s3), v):
                out[m - 1, l - 1] = out[l - 1, m - 1] = value
    return s1, s2, s3


def msd_covariance(n: int, p: int, x: float) -> np.ndarray:
    """The covariance of the MSD at lags 1..p, per axis, of a Brownian
    track of n positions with D dt = 1 and a localization error of variance
    x per axis (so x is Michalet's reduced error sigma^2 / (D dt)).

    A displacement over lag m from position i and one over lag l from j
    covary by 2 D dt times the frames they share plus sigma^2 times the
    noise term (+1 where their ends meet as i+m = j+l or i = j, -1 where
    i+m = j or i = j+l).  The MSD is a mean of squares of these jointly
    Gaussian displacements, and for those Cov(a^2, b^2) = 2 Cov(a, b)^2
    (Isserlis), so the covariance of the MSD at lags m and l is twice the
    sum of the squared covariances over all their pairs, over the numbers of
    pairs, (n - m)(n - l).
    """
    s1, s2, s3 = _msd_sums(int(n), int(p))
    q = 2.0                                       # 2 D dt per axis and frame
    k = np.arange(1, p + 1)
    scale = 2.0 / np.outer(n - k, n - k)
    return scale * (q * q * s1 + 2 * q * x * s2 + x * x * s3)


def _weights(p: int, fitted: bool) -> Optional[np.ndarray]:
    """The weights of the MSD points 1..p in the estimate of 2 D dt (per
    axis): through the known offset, or a straight line's slope."""
    k = np.arange(1, p + 1, dtype=float)
    if not fitted:
        return k / np.sum(k * k)
    if p < 2:
        return None
    c = k - k.mean()
    return c / np.sum(c * c)


def lag_errors(n: int, x: float, fitted: bool = False) -> np.ndarray:
    """The relative standard error of D estimated from the first p MSD
    points, for p = 1, 2, ... (NaN where p cannot be used): the curve whose
    minimum Michalet (2010) showed.  For one axis: d independent axes divide
    it by sqrt(d), which moves no minimum."""
    p_max = max(1, min(int(n) - 1, MAX_LAGS))
    cov = msd_covariance(int(n), p_max, float(x))
    out = np.full(p_max, np.nan)
    for p in range(1, p_max + 1):
        w = _weights(p, fitted)
        if w is None:
            continue
        # the estimate of 2 D dt per axis is w . MSD_axis; D dt = 1 here
        out[p - 1] = np.sqrt(max(w @ cov[:p, :p] @ w, 0.0)) / 2.0
    return out


def _bucket(n: int) -> int:
    """Track lengths on a coarse grid beyond 40: the optimum moves slowly."""
    n = int(min(max(n, 2), LONGEST))
    if n <= 40:
        return n
    return int(round(40 * 1.25 ** round(np.log(n / 40) / np.log(1.25))))


@lru_cache(maxsize=512)
def _optimal_row(n: int, fitted: bool) -> np.ndarray:
    """The optimal number of lags at each x of `X_GRID`, for length n."""
    lowest = 2 if fitted else 1
    rows = []
    for x in X_GRID:
        errors = lag_errors(n, x, fitted)
        rows.append(int(np.nanargmin(errors)) + 1 if np.isfinite(errors).any() else lowest)
    return np.array(rows)


def optimal_lags(n, x, fitted: bool = False) -> np.ndarray:
    """The number of MSD points that gives the most precise D for tracks of
    ``n`` positions and reduced localization errors ``x`` (arrays, or
    numbers), from `lag_errors` tabulated at `X_GRID`."""
    n = np.atleast_1d(np.asarray(n, np.int64))
    x = np.broadcast_to(np.atleast_1d(np.asarray(x, float)), n.shape)
    xi = np.clip(np.round((np.log10(np.clip(np.nan_to_num(x, nan=1e4), 1e-3, 1e4))
                           + 3.0) / 0.125).astype(int), 0, len(X_GRID) - 1)
    out = np.empty(len(n), np.int64)
    buckets = np.array([_bucket(v) for v in n])
    for b in np.unique(buckets):
        here = buckets == b
        out[here] = _optimal_row(int(b), bool(fitted))[xi[here]]
    lowest = 2 if fitted else 1
    return np.clip(out, lowest, np.maximum(n - 1, lowest))


# ------------------------------------------------------------ the MSD fit

@dataclass
class MSDFit:
    """Per track (or one, pooled): D (um^2/s), alpha, the lags used, the
    offset (nm^2) and the reduced localization error."""
    D_um2_per_s: np.ndarray
    alpha: np.ndarray
    lags: np.ndarray
    offset_nm2: np.ndarray
    x: np.ndarray


def _cumulative(msd, offset, fitted):
    """Running sums over lags of what the fits need, per track."""
    k = np.arange(1, msd.shape[1] + 1, dtype=float)
    have = np.isfinite(msd)
    rho = np.where(have, msd, 0.0)
    one = have.astype(float)
    if fitted:
        return (np.cumsum(one, 1), np.cumsum(one * k, 1), np.cumsum(one * k * k, 1),
                np.cumsum(rho, 1), np.cumsum(rho * k, 1))
    y = np.where(have, msd - offset[:, None], 0.0)
    return np.cumsum(y * k, 1), np.cumsum(one * k * k, 1)


def _slope_at(sums, p, fitted):
    """2 d D dt (nm^2 per frame, all axes) and the intercept, with the
    first p lags of each track."""
    at = (np.arange(len(p)), np.asarray(p) - 1)
    if fitted:
        s, sk, skk, sr, skr = (a[at] for a in sums)
        den = s * skk - sk * sk
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = np.where(den > 0, (s * skr - sk * sr) / den, np.nan)
            icept = np.where(s > 0, (sr - slope * sk) / s, np.nan)
        return slope, icept
    num, den = (a[at] for a in sums)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan), None


def _alpha(msd, offset, p):
    """The slope of log(MSD - offset) over log(lag), on lags 1..max(p, 2)."""
    K = msd.shape[1]
    k = np.arange(1, K + 1, dtype=float)
    y = msd - offset[:, None]
    use = np.isfinite(y) & (y > 0) & (k[None, :] <= np.maximum(p, 2)[:, None])
    lx = np.where(use, np.log(k)[None, :], 0.0)
    ly = np.where(use, np.log(np.where(use, y, 1.0)), 0.0)
    n = use.sum(1)
    sx, sy = lx.sum(1), ly.sum(1)
    sxx, sxy = (lx * lx).sum(1), (lx * ly).sum(1)
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n >= 2) & (den > 0), (n * sxy - sx * sy) / den, np.nan)


def fit_msd(msd: np.ndarray, spans: np.ndarray, offset_nm2: np.ndarray, dims: int,
            dt_s: float, fitted: bool = False, lags: Optional[int] = None) -> MSDFit:
    """D and alpha of each row of ``msd`` (tracks x lags 1..K, NaN where a
    lag has no pairs), each track ``spans`` frames long, with Michalet's
    iteration for the number of lags unless ``lags`` fixes it.
    ``offset_nm2`` is 2 sum_axes sigma^2 (ignored when ``fitted``)."""
    msd = np.atleast_2d(np.asarray(msd, float))
    T, K = msd.shape
    spans = np.broadcast_to(np.asarray(spans, np.int64), (T,))
    offset = np.broadcast_to(np.asarray(offset_nm2, float), (T,)).astype(float)
    sums = _cumulative(msd, offset, fitted)
    lowest = 2 if fitted else 1
    top = np.clip(np.minimum(spans - 1, K), lowest, K)
    if lags:
        p = np.clip(np.full(T, int(lags)), lowest, top)
        slope, icept = _slope_at(sums, p, fitted)
    else:
        p = np.clip(np.round(spans / 10.0).astype(np.int64), lowest, top)
        seen = [p.copy()]
        for _ in range(10):
            slope, icept = _slope_at(sums, p, fitted)
            off = icept if fitted else offset
            # x = sigma^2 per axis / (D dt): offset / 2d over slope / 2d
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.where(slope > 0, np.maximum(off, 0) / slope, 1e4)
            new = np.minimum(optimal_lags(spans, x, fitted), top)
            if np.array_equal(new, p) or any(np.array_equal(new, s) for s in seen):
                p = new
                break
            seen.append(new)
            p = new
        slope, icept = _slope_at(sums, p, fitted)
    off = icept if fitted else offset
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(slope > 0, np.maximum(off, 0) / slope, np.inf)
    D = slope / (2 * dims * dt_s) * 1e-6                       # nm^2/s -> um^2/s
    return MSDFit(D, _alpha(msd, off, p), p, off, x)


@dataclass
class MSDResult:
    """The MSD analysis of a set of tracks: the pooled curve and its fit,
    and a fit per track (NaN for a track too short)."""
    lag_s: np.ndarray
    msd_nm2: np.ndarray                 # pooled over the tracks
    pairs: np.ndarray
    pooled: MSDFit
    per_track: MSDFit
    ids: np.ndarray
    n_locs: np.ndarray


def msd_analysis(t: Tracks, fitted: bool = False, lags: Optional[int] = None,
                 min_length: int = 5) -> MSDResult:
    """The MSD of every track and of all pooled, each fitted (`fit_msd`).
    A track of fewer than ``min_length`` localizations gets no D of its
    own; the pooled MSD uses every track."""
    K = MAX_LAGS if not lags else max(int(lags), 2 if fitted else 1)
    sums, counts = lag_sums(t, K)
    with np.errstate(invalid="ignore", divide="ignore"):
        msd = np.where(counts > 0, sums / counts, np.nan)
    n_locs = t.counts()
    spans = t.spans()
    err = np.bincount(t.track, weights=t.err2, minlength=t.n_tracks)
    offset = 2.0 * err / np.maximum(n_locs, 1)                # 2 sum_axes sigma^2
    per = fit_msd(msd, spans, offset, t.dims, t.dt_s, fitted, lags)
    short = n_locs < max(int(min_length), 2)
    for name in ("D_um2_per_s", "alpha", "x"):
        getattr(per, name)[short] = np.nan
    total, pairs = sums.sum(0), counts.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_msd = np.where(pairs > 0, total / pairs, np.nan)
    n_all = max(len(t.track), 1)
    pooled = fit_msd(pooled_msd[None, :], [int(np.median(spans)) if len(spans) else 2],
                     [2.0 * t.err2.sum() / n_all], t.dims, t.dt_s, fitted, lags)
    return MSDResult(np.arange(1, K + 1) * t.dt_s, pooled_msd, pairs, pooled, per,
                     t.ids, n_locs)


# -------------------------------------------------------- jump distances

def _pairs_at(t: Tracks, lag: int):
    """The pairs of a track's localizations exactly ``lag`` frames apart:
    the squared distance, the error variance per axis, the track and the
    first one's row, in the order of the rows (by track, then time)."""
    out = [], [], [], []
    for m in range(1, int(lag) + 1):
        if m >= len(t.track):
            break
        use = (t.track[m:] == t.track[:-m]) & (t.step[m:] - t.step[:-m] == lag)
        a = np.flatnonzero(use)
        out[0].append(((t.xyz[a + m] - t.xyz[a]) ** 2).sum(axis=1))
        out[1].append((t.err2[a + m] + t.err2[a]) / t.dims)
        out[2].append(t.track[a])
        out[3].append(a)
    if not out[0]:
        return np.zeros(0), np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64)
    d2, err, track, row = (np.concatenate(o) for o in out)
    order = np.argsort(row, kind="stable")          # rows are sorted by track, time
    return d2[order], err[order], track[order], row[order]


def jumps(t: Tracks, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Every pair of a track's localizations exactly ``lag`` frames apart:
    the squared distance (nm^2) and the error variance per axis it carries,
    ``sigma_a^2 + sigma_b^2`` (nm^2, the mean over the axes)."""
    d2, err, _, _ = _pairs_at(t, lag)
    return d2, err


def jumps_over_lags(t: Tracks, lags: int = 1
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The jumps at every lag 1..``lags`` (frames): squared distance (nm^2),
    error variance per axis (nm^2) and the lag of each, in frames.

    Every pair, from every localization of every track.  Spot-On takes only
    each track's first four jumps at each lag (*JumpsToConsider*), so that
    long tracks of slow particles weigh less; that is what biases the
    focus correction (`in_focus`), which holds for particles anywhere in
    the focus, while a track's first localizations are often of a particle
    that has just come in at its edge.  On simulated tracks (700 nm focus,
    D = 0.05 and 3 um^2/s, half each) all jumps gave back 49% and 51%, the
    first four 21% and 79%."""
    out = [], [], []
    for lag in range(1, max(int(lags), 1) + 1):
        d2, err, _, _ = _pairs_at(t, lag)
        out[0].append(d2)
        out[1].append(err)
        out[2].append(np.full(len(d2), lag, np.int64))
    return tuple(np.concatenate(o) for o in out)


# how finely the focal slab is cut when the loss from it is computed: the
# fraction kept is right to 1e-4 against the exact series for any D
FOCUS_CELLS = 200


def _cell_mean_cdf(u_hi, u_lo):
    """The mean of Phi over u from u_lo to u_hi, by the antiderivative
    u Phi(u) + phi(u); Phi(u_hi) where the two are equal."""
    from scipy.special import ndtr
    def g(u):
        return u * ndtr(u) + np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    width = u_hi - u_lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (g(u_hi) - g(u_lo)) / width
    return np.where(width > 1e-12, mean, ndtr(u_hi))


def in_focus(d_um2_s, step_s: float, range_nm: float, lags: int) -> np.ndarray:
    """The fraction of particles in focus in one frame that are still in
    focus in each of the next 1..``lags`` frames, for each D (um^2/s): a
    (len(D), lags) array.

    The focus is the slab |z| <= range / 2; a particle in it is equally
    likely anywhere in z (as the particles of a cell in focus at one moment
    are), moves in z by a Gaussian of variance 2 D step each frame, and is
    lost the first frame it is outside -- looked at once a frame, as the
    camera does, not continuously, which is saSPT's correction (Heckert et
    al. 2022) rather than Spot-On's continuous absorbing series (Hansen et
    al. 2018), and needs none of Spot-On's empirical widening of the slab.
    The density inside is carried from frame to frame on `FOCUS_CELLS`
    cells: a matrix of the probability of moving from each cell into each
    other, averaged over the cell it leaves, so that it is right however
    narrow the step is against a cell."""
    D = np.atleast_1d(np.asarray(d_um2_s, float))
    K = max(int(lags), 1)
    out = np.ones((len(D), K))
    if not range_nm or range_nm <= 0:
        return out
    h = float(range_nm) / 2.0
    N = FOCUS_CELLS
    edges = np.linspace(-h, h, N + 1)
    lo, hi = edges[:-1], edges[1:]
    for i, d in enumerate(D):
        sigma = np.sqrt(max(2.0 * d * 1e6 * step_s, 0.0))
        if sigma <= 0:
            continue
        # M[a, b]: from anywhere in cell b into cell a, averaged over b
        # P(z' < e | z) = Phi((e - z) / sigma); its mean over z in cell b
        # runs over u from (e - hi_b) / sigma to (e - lo_b) / sigma
        def below(e, sigma=sigma):
            return _cell_mean_cdf((e[:, None] - lo[None, :]) / sigma,
                                  (e[:, None] - hi[None, :]) / sigma)
        M = below(hi) - below(lo)
        p = np.full(N, 1.0 / N)
        for k in range(K):
            p = M @ p
            out[i, k] = p.sum()
    return np.clip(out, 0.0, 1.0)


@dataclass
class JumpFit:
    """The diffusive states fitted to the jumps: D (um^2/s) and fraction of
    each, slowest first; the log-likelihood and BIC.  The fractions are of
    the particles: with a focal range, corrected for the fast ones leaving
    the focus sooner.  Per lag (``lag_s``), ``seen`` is each state's
    fraction of the jumps (a state per row), ``kept`` the fraction of its
    particles still in focus (1 without a focal range)."""
    D_um2_per_s: np.ndarray
    fraction: np.ndarray
    loglik: float
    bic: float
    n_jumps: int
    n_fitted: int
    lag_s: np.ndarray = None
    seen: np.ndarray = None
    kept: np.ndarray = None
    focal_range_nm: float = 0.0


def _jump_logpdf(d2, s2, dims):
    """log of the density of r^2 for a d-dimensional Gaussian jump of
    variance s2 per axis: chi-square with d degrees of freedom, scaled."""
    h = dims / 2.0
    from scipy.special import gammaln
    return ((h - 1) * np.log(np.maximum(d2, 1e-300)) - d2 / (2 * s2)
            - h * np.log(2 * s2) - gammaln(h))


def fit_jumps(d2: np.ndarray, err: np.ndarray, dims: int, lag_s,
              states: int = 1, tol: float = 1e-8, max_iter: int = 500,
              max_jumps: int = 200_000, focal_range_nm: float = 0.0,
              step_s: Optional[float] = None) -> JumpFit:
    """D and fraction of 1 to 3 diffusive states, by maximum likelihood on
    the squared jumps ``d2`` (nm^2) with their error variance per axis
    ``err`` (nm^2), each over ``lag_s`` seconds (one number, or one per
    jump): state i's jumps have per axis the variance 2 D_i lag + err.

    Expectation maximisation: the fractions are the mean memberships, and a
    state's D solves its weighted likelihood equation, sum over jumps of
    membership lag (d2 - d s^2) / s^4 = 0 with s^2 = 2 D lag + err, by the
    fixed point D = sum u (d2 / d - err) / (2 sum u lag), u = membership
    lag / s^4.  Each round is one pass over the jumps, so a million jumps
    take seconds; a general optimiser on the same likelihood took a minute.
    The states start at quantiles of the jumps' own D, so a mixture is not
    started on one state twice.

    With ``focal_range_nm`` (the thickness of the focus, `in_focus`), a
    state's share of the jumps at lag k is its fraction of the particles
    times the fraction of them still in focus after k frames (of
    ``step_s``, by default the shortest lag), renormalised over the states:
    fast particles leave sooner, so they make fewer jumps than their
    number, and fewer the longer the lag (Hansen et al. 2018).  That
    couples each D to every fraction, so the EM answer is only the start:
    the likelihood itself is then maximised (L-BFGS, the gradient exact but
    for the focus's own slope in D, taken by a difference).

    At most ``max_jumps`` jumps are fitted, a random subset when there are
    more (the same one every run): D is then known to a few tenths of a
    percent anyway, and EM is slow exactly where it does not matter -- two
    states fitted to one population, whose split it refines for thousands
    of rounds.  The log-likelihood and BIC are of the jumps fitted.
    """
    from scipy.special import logsumexp
    d2 = np.asarray(d2, float)
    err = np.maximum(np.asarray(err, float), 0.0)
    n = n_all = len(d2)
    t_s = np.broadcast_to(np.asarray(lag_s, float), d2.shape).astype(float)
    states = int(states)
    if states < 1:
        raise ValueError("at least one state")
    if n < 2 * states + 1:
        raise ValueError(f"{n} jumps: too few for {states} state"
                         f"{'s' if states > 1 else ''}")
    if n > max_jumps:
        pick = np.sort(np.random.default_rng(0).choice(n, int(max_jumps), replace=False))
        d2, err, t_s, n = d2[pick], err[pick], t_s[pick], int(max_jumps)
    floor = 1e-9                                      # nm^2/s: D stays above 0
    raw = np.maximum(d2 / dims - err, 0.0) / (2 * t_s)
    D = np.maximum(np.quantile(raw, (np.arange(states) + 0.5) / states), floor)
    D = D * (1 + 1e-3 * np.arange(states))           # never two equal starts
    f = np.full(states, 1.0 / states)
    last = -np.inf
    for _ in range(max_iter):
        s2 = 2 * D[None, :] * t_s[:, None] + err[:, None]
        terms = _jump_logpdf(d2[:, None], s2, dims) + np.log(np.maximum(f, 1e-300))[None, :]
        total = logsumexp(terms, axis=1)
        loglik = float(total.sum())
        member = np.exp(terms - total[:, None])
        f = member.mean(axis=0)
        u = member * t_s[:, None] / (s2 * s2)
        D = np.maximum((u * (d2[:, None] / dims - err[:, None])).sum(0)
                       / (2 * np.maximum((u * t_s[:, None]).sum(0), 1e-300)), floor)
        if abs(loglik - last) <= tol * abs(loglik):
            break
        last = loglik

    lags_s = np.unique(t_s)
    step = float(step_s) if step_s else float(lags_s.min())
    frames = np.maximum(np.round(lags_s / step).astype(np.int64), 1)
    at = np.searchsorted(lags_s, t_s)               # each jump's lag, an index
    focus = bool(focal_range_nm) and focal_range_nm > 0
    if focus:
        D, f = _fit_in_focus(d2, err, t_s, at, frames, dims, D, f,
                             float(focal_range_nm), step, floor)
    kept = in_focus(D * 1e-6, step, focal_range_nm if focus else 0.0,
                    int(frames.max()))[:, frames - 1]
    seen = f[:, None] * kept
    seen = seen / np.maximum(seen.sum(0, keepdims=True), 1e-300)
    s2 = 2 * D[None, :] * t_s[:, None] + err[:, None]
    loglik = float(logsumexp(_jump_logpdf(d2[:, None], s2, dims)
                             + np.log(np.maximum(seen[:, at].T, 1e-300)), axis=1).sum())
    order = np.argsort(D)
    k = 2 * states - 1
    return JumpFit(D[order] * 1e-6, f[order], loglik, k * np.log(n) - 2 * loglik,
                   n_all, n, lags_s, seen[order], kept[order],
                   float(focal_range_nm) if focus else 0.0)


def _fit_in_focus(d2, err, t_s, at, frames, dims, D, f, range_nm, step, floor):
    """The states' D (nm^2/s) and fractions of the particles that make the
    jumps most likely when each state's share of the jumps at a lag is
    weighed by the fraction of it still in focus (`fit_jumps`).  Started
    from the EM's D and its fractions of the jumps undone for the focus at
    the first lag."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    S = len(D)
    K = int(frames.max())
    h = dims / 2.0
    N_k = np.bincount(at, minlength=len(frames)).astype(float)
    eps = 1e-3

    def kept_and_slope(D):
        F = in_focus(np.concatenate([D, D * np.exp(eps), D * np.exp(-eps)]) * 1e-6,
                     step, range_nm, K)[:, frames - 1]
        F0, Fp, Fm = F[:S], F[S:2 * S], F[2 * S:]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (np.log(np.maximum(Fp, 1e-300)) - np.log(np.maximum(Fm, 1e-300))) / (2 * eps)
        return np.maximum(F0, 1e-300), np.nan_to_num(g)

    def unpack(theta):
        D = np.exp(theta[:S])
        a = np.concatenate([[0.0], theta[S:]])
        f = np.exp(a - logsumexp(a))
        return D, f

    def negative(theta):
        D, f = unpack(theta)
        F, g = kept_and_slope(D)
        Z = (f[:, None] * F).sum(0)
        log_pi = np.log(np.maximum(f, 1e-300))[:, None] + np.log(F) - np.log(Z)[None, :]
        pi = np.exp(log_pi)
        s2 = 2 * D[None, :] * t_s[:, None] + err[:, None]
        terms = _jump_logpdf(d2[:, None], s2, dims) + log_pi[:, at].T
        total = logsumexp(terms, axis=1)
        r = np.exp(terms - total[:, None])
        n_mk = np.stack([np.bincount(at, weights=r[:, m], minlength=len(frames))
                         for m in range(S)])
        dlogp = (d2[:, None] / (2 * s2 * s2) - h / s2) * 2 * t_s[:, None] * D[None, :]
        grad_D = (r * dlogp).sum(0) + (g * (n_mk - N_k[None, :] * pi)).sum(1)
        grad_a = (n_mk - N_k[None, :] * pi).sum(1)[1:]
        return -float(total.sum()), -np.concatenate([grad_D, grad_a])

    F1 = in_focus(D * 1e-6, step, range_nm, 1)[:, 0]
    f0 = f / np.maximum(F1, 1e-6)
    f0 = np.maximum(f0 / f0.sum(), 1e-6)
    theta = np.concatenate([np.log(np.maximum(D, floor)), np.log(f0[1:] / f0[0])])
    bounds = [(np.log(floor), np.log(1e12))] * S + [(-30.0, 30.0)] * (S - 1)
    result = minimize(negative, theta, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500})
    return unpack(result.x)


def jump_density(r_nm: np.ndarray, fit: JumpFit, err: float, dims: int,
                 lag_s: float) -> np.ndarray:
    """The fitted density of the jump distance r (per nm) at ``lag_s``,
    each state's as a row, at a typical error variance per axis ``err``,
    each weighed by its share of the jumps at that lag."""
    r = np.asarray(r_nm, float)
    out = []
    shares = fit.fraction
    if fit.seen is not None and fit.lag_s is not None and len(fit.lag_s):
        # the states' shares of the jumps at this lag, as they were seen
        shares = fit.seen[:, int(np.argmin(np.abs(fit.lag_s - lag_s)))]
    for D, f in zip(fit.D_um2_per_s, shares):
        s2 = 2 * D * 1e6 * lag_s + err
        # density of r = density of r^2 times 2r
        out.append(f * 2 * r * np.exp(_jump_logpdf(r * r, s2, dims)))
    return np.array(out)
