"""Directional correlations: a correlation image, and a correlation along a line.

The directional approaches of Spatial statistics (`plugins/spatial_statistics.py`,
`docs/smap_porting/wave4_design.md`, step 17c), SMAP's ``correlationtools``
and the 2D half of ``correlation``.  The radial approaches average every
direction into one curve g(r); these keep the direction, for what is
periodic along one -- spectrin rings every 190 nm along an axon, the
sarcomere, a z-stack of layers -- or shifted between two channels.

**Axes.**  Positions are taken in a *frame*: with a line ROI (drawn, or an
ROI's named line, Line Profile's `region_line`) x runs along the line and y
across it; otherwise x and y are the table's.  z is ``z_nm``, over a range
taken from the data -- the 0.5th and 99.5th percentiles, clipped to
-500 .. 500 nm -- so a few far-out mislocalizations do not stretch it, and
the localizations outside it are left out.  A 2D image is of x-y, x-z or
y-z; a line correlation along x, y or z.

**Normalised by the window, as the radial ones.**  Binned at *bin* into an
image (or a profile) h, the correlation of two layers at a shift k is

    g(k) = sum_x h_a(x) h_b(x + k) / (n_a n_b / W^2 * sum_x w(x) w(x + k)),

w the window's weight in each bin and W its sum: the window's own
autocorrelation in place of 17a's set covariance, which is the same thing
not averaged over directions (Veatch's normalisation, SMAP's
``paircrosscorrelationfunction`` before its radial sum).  For an x-y image
w is the window's mask; for x-z it is the window's width across x at each
x, times 1 inside the z range; for a line along x, the same width.  So g = 1
for localizations placed at random in the window, in every direction and
at every shift -- where SMAP's ``xcorr(c, c) / mean(c)^2`` falls off as the
overlap of a profile with itself shrinks, and its peaks had to be found
after a quadratic detrend.  An auto-correlation leaves out each
localization's pair with itself (n at k = 0).  Files are counted apart and
their sums added.

**The period** of a line correlation: the local maxima of g at positive
shifts beyond two bins, the one nearest the *period* guess (SMAP's
``selectpeak``), else the first at least half as high above 1 as the
highest (SMAP took the highest, which on a regular structure is as often
two periods as one); refined by a parabola through it and its neighbours.  The
spectrum's (|FFT| of the profile, one-sided, as SMAP's ``fft_one_sided``)
is the same choice among 1 / f at its peaks.  SMAP's multiple
autocorrelation filter (three rounds of detrend and autocorrelate) is not
ported: on a window-normalised g it is not needed to find the peaks, and
its period could not be told from the plain one in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: z is taken between these quantiles of the data, clipped to `Z_LIMIT`
Z_QUANTILES = (0.005, 0.995)
Z_LIMIT = (-500.0, 500.0)
#: at most this many bins in an image (with its padding): 64 MB of floats
MAX_PIXELS = 8_000_000
#: shifts this many bins from 0 are not a period (the blinking, the precision)
SKIP_BINS = 2
#: a window overlap below this fraction of its largest is too thin to divide by
MIN_OVERLAP = 1e-3

PLANES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}


# ----------------------------------------------------------------- the frame

@dataclass
class Frame:
    """Coordinates along and across a line: ``u`` along from ``origin``,
    ``v`` across, positive to the left -- Line Profile's `project`."""
    origin: Tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0                  # radians, of the u axis from x

    @classmethod
    def of_line(cls, p0, p1) -> "Frame":
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        if not np.hypot(*d) > 0:
            raise ValueError("the line ROI has no length")
        return cls((float(p0[0]), float(p0[1])), float(np.arctan2(d[1], d[0])))

    def to(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(self.angle), np.sin(self.angle)
        dx = np.asarray(x, float) - self.origin[0]
        dy = np.asarray(y, float) - self.origin[1]
        return dx * c + dy * s, -dx * s + dy * c

    def back(self, u, v) -> Tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(self.angle), np.sin(self.angle)
        u, v = np.asarray(u, float), np.asarray(v, float)
        return self.origin[0] + u * c - v * s, self.origin[1] + u * s + v * c

    @property
    def turned(self) -> bool:
        return self.angle != 0.0 or self.origin != (0.0, 0.0)


def z_range(z) -> Optional[Tuple[float, float]]:
    """The z range to measure in: robust quantiles of the data, clipped to
    -500 .. 500 nm; None without z."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if not len(z):
        return None
    lo, hi = np.quantile(z, Z_QUANTILES)
    lo, hi = max(float(lo), Z_LIMIT[0]), min(float(hi), Z_LIMIT[1])
    if not hi > lo:
        raise ValueError(f"the z range of the data is empty after clipping to "
                         f"{Z_LIMIT[0]:g} .. {Z_LIMIT[1]:g} nm")
    return lo, hi


@dataclass
class Grid:
    """Bins of ``bin`` from ``start`` for each axis (u, v, z)."""
    start: Dict[str, float]
    count: Dict[str, int]
    bin: float

    def index(self, axis: str, values) -> np.ndarray:
        return np.floor((np.asarray(values, float) - self.start[axis]) / self.bin
                        ).astype(np.int64)


def lateral_mask(shape, frame: Frame, bin_nm: float, inside=None,
                 bounds_uv=None) -> Tuple[np.ndarray, float, float]:
    """The window in the frame, binned: ``(mask[iv, iu], u0, v0)``.

    ``shape`` (a `regions.Region`) decides it when given -- a bin is in when
    its centre is; else ``inside(x, y)``, a test (a `spatial_statistics.Window`'s
    `contains`), over ``bounds_uv`` (u0, v0, u1, v1)."""
    if shape is not None:
        xs, ys = (np.asarray(a) for a in shape.outline())
        u, v = frame.to(xs, ys)
        bounds_uv = (u.min(), v.min(), u.max(), v.max())
        test = shape.mask
    else:
        test = inside
    u0, v0, u1, v1 = (float(b) for b in bounds_uv)
    nu = max(1, int(np.ceil((u1 - u0) / bin_nm)))
    nv = max(1, int(np.ceil((v1 - v0) / bin_nm)))
    if nu * nv > MAX_PIXELS:
        raise ValueError(f"{nu} x {nv} bins of {bin_nm:g} nm: a larger bin or a "
                         "smaller ROI")
    cu = u0 + (np.arange(nu) + 0.5) * bin_nm
    cv = v0 + (np.arange(nv) + 0.5) * bin_nm
    gu, gv = np.meshgrid(cu, cv)
    x, y = frame.back(gu.ravel(), gv.ravel())
    mask = np.asarray(test(x, y), dtype=bool).reshape(nv, nu)
    if not mask.any():
        raise ValueError("the window has no area at this bin")
    return mask, u0, v0


# ------------------------------------------------------------ correlations

@dataclass
class Binned:
    """One file's binned layers and window on the axes measured."""
    images: List[np.ndarray]           # per layer, counts
    counts: List[int]                  # per layer, the localizations binned
    window: np.ndarray                 # weights, the images' shape


def window_weights(mask: np.ndarray, axes: Sequence[str], nz: int = 0) -> np.ndarray:
    """The window on ``axes`` (of u, v, z) from the lateral ``mask[iv, iu]``:
    the mask itself for (u, v); its width along the axis dropped, for one
    lateral axis -- times 1 in each z bin for z; all of its area per z bin
    for z alone."""
    wu = mask.sum(axis=0).astype(float)            # per u bin: bins across
    wv = mask.sum(axis=1).astype(float)
    ones = np.ones(max(nz, 0))
    if tuple(axes) == ("u", "v"):
        return mask.T.astype(float)                # [iu, iv]
    if tuple(axes) == ("u", "z"):
        return np.outer(wu, ones)
    if tuple(axes) == ("v", "z"):
        return np.outer(wv, ones)
    if tuple(axes) == ("u",):
        return wu
    if tuple(axes) == ("v",):
        return wv
    if tuple(axes) == ("z",):
        return ones * float(mask.sum())
    raise ValueError(f"no axes {axes!r}")


def bin_layers(points: Sequence[Dict[str, np.ndarray]], mask: np.ndarray, u0: float,
               v0: float, axes: Sequence[str], bin_nm: float,
               zr: Optional[Tuple[float, float]] = None) -> Binned:
    """Each layer's ``{"u", "v", "z"}`` counted in the bins of ``axes``, the
    window beside them.  A localization outside the window's bins (or the z
    range) is left out."""
    nz = 0
    if "z" in axes:
        if zr is None:
            raise ValueError("no z_nm: a z axis needs 3D data")
        nz = max(1, int(np.ceil((zr[1] - zr[0]) / bin_nm)))
    start = {"u": u0, "v": v0, "z": zr[0] if zr else 0.0}
    size = {"u": mask.shape[1], "v": mask.shape[0], "z": nz}
    weights = window_weights(mask, axes, nz)
    images, counts = [], []
    lateral_ok = None
    for p in points:
        n = len(p["u"])
        keep = np.ones(n, dtype=bool)
        idx = {}
        for a in ("u", "v") + (("z",) if "z" in axes else ()):
            i = np.floor((p[a] - start[a]) / bin_nm).astype(np.int64)
            keep &= (i >= 0) & (i < size[a])
            idx[a] = i
        # in the window laterally, whatever is measured
        lateral_ok = np.zeros(n, dtype=bool)
        lateral_ok[keep] = mask[idx["v"][keep], idx["u"][keep]]
        keep &= lateral_ok
        shape = tuple(size[a] for a in axes)
        image = np.zeros(shape)
        np.add.at(image, tuple(idx[a][keep] for a in axes), 1.0)
        images.append(image)
        counts.append(int(keep.sum()))
    return Binned(images, counts, weights)


def _xcorr(a: np.ndarray, b: np.ndarray, lags: int) -> np.ndarray:
    """``c[k] = sum_x a(x) b(x + k)`` for every k within ``lags`` bins on each
    axis, by FFT with padding (no wrap)."""
    from scipy.fft import irfftn, next_fast_len, rfftn
    shape = tuple(next_fast_len(n + lags + 1) for n in a.shape)
    fa = rfftn(a, shape)
    fb = fa if b is a else rfftn(b, shape)
    c = irfftn(np.conj(fa) * fb, shape)
    index = np.ix_(*[np.arange(-lags, lags + 1) % n for n in shape])
    return c[index]


def correlate(files: Sequence[Binned], a: int, b: int, lags: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """g of layers ``a`` and ``b`` over shifts of up to ``lags`` bins (see the
    module docstring): ``(g, pairs, expected)``, each with the files' sums,
    the shift 0 in the middle.  NaN where the window barely overlaps itself."""
    pairs = expected = None
    for f in files:
        na, nb = f.counts[a], f.counts[b]
        if not na or not nb:
            continue
        c = _xcorr(f.images[a], f.images[b], lags)
        if a == b:
            c[(lags,) * c.ndim] -= na               # each with itself
        w = _xcorr(f.window, f.window, lags)
        total = float(f.window.sum())
        e = (na * (na - 1.0) if a == b else na * float(nb)) / total ** 2 * w
        cut = w < MIN_OVERLAP * w.max()
        e[cut] = 0.0
        c[cut] = 0.0
        pairs = c if pairs is None else pairs + c
        expected = e if expected is None else expected + e
    if pairs is None:
        n = (2 * lags + 1,) * files[0].window.ndim if files else (2 * lags + 1,)
        empty = np.full(n, np.nan)
        return empty, np.zeros(n), np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(expected > 0, pairs / expected, np.nan)
    return g, pairs, expected


# ------------------------------------------------------------- peaks

def refine(x: np.ndarray, y: np.ndarray, i: int) -> Tuple[float, float]:
    """The vertex of the parabola through ``i`` and its neighbours (SMAP's
    ``refinepeaks``), kept between them."""
    if 0 < i < len(y) - 1 and np.all(np.isfinite(y[i - 1:i + 2])):
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            t = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            step = x[i + 1] - x[i]
            return float(x[i] + t * step), float(y1 - 0.25 * (y0 - y2) * t)
    return float(x[i]), float(y[i])


def local_maxima(y: np.ndarray) -> np.ndarray:
    y = np.where(np.isfinite(y), y, -np.inf)
    inner = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    return inner[np.isfinite(y[inner])]


def select_peak(xs: Sequence[float], ys: Sequence[float], guess: float = 0.0
                ) -> Tuple[float, float]:
    """SMAP's ``selectpeak``: the peak nearest ``guess``, else the highest."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if not len(xs):
        return float("nan"), float("nan")
    k = int(np.argmin(np.abs(xs - guess))) if guess else int(np.argmax(ys))
    return float(xs[k]), float(ys[k])


def period_of(lag: np.ndarray, g: np.ndarray, guess: float = 0.0
              ) -> Tuple[float, float]:
    """``(period, g there)`` of a line correlation: among its local maxima at
    shifts beyond `SKIP_BINS` bins, the one nearest ``guess`` (SMAP's
    ``selectpeak``); without a guess the first that rises at least half as
    far above 1 as the highest -- the highest itself is as often the second
    or third period as the first.  Refined by a parabola.  SMAP took out a
    quadratic trend first; on a window-normalised g there is none to take
    out, and on a few periods the fitted one moved the peaks."""
    pos = lag > 0
    x, y = lag[pos], g[pos]
    peaks = [i for i in local_maxima(y) if i >= SKIP_BINS]
    if not peaks:
        return float("nan"), float("nan")
    found = [refine(x, y, i) for i in peaks]
    if guess:
        return select_peak([p for p, _ in found], [h for _, h in found], guess)
    heights = np.array([h for _, h in found]) - 1.0
    top = heights.max()
    if not top > 0:
        return float("nan"), float("nan")
    first = int(np.flatnonzero(heights >= 0.5 * top)[0])
    return found[first]


def spectrum(profile: np.ndarray, bin_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided |FFT| of a profile (SMAP's ``fft_one_sided``): frequencies in
    1/nm and magnitudes, |X| / N, doubled between 0 and Nyquist."""
    x = np.asarray(profile, float)
    n = len(x)
    mag = np.abs(np.fft.rfft(x)) / max(n, 1)
    if n > 2:
        mag[1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, bin_nm), mag


def spectrum_period(f: np.ndarray, mag: np.ndarray, guess: float = 0.0,
                    bin_nm: float = 1.0) -> Tuple[float, float]:
    """``(period, magnitude)`` of the spectrum's peak (1 / f), by
    `select_peak`; f = 0 and periods under `SKIP_BINS` bins left out."""
    peaks = [i for i in local_maxima(mag) if f[i] > 0]
    if not peaks:
        return float("nan"), float("nan")
    found = [refine(f, mag, i) for i in peaks]
    periods = [1.0 / ff if ff > 0 else np.inf for ff, _ in found]
    keep = [k for k, p in enumerate(periods) if p > SKIP_BINS * bin_nm]
    if not keep:
        return float("nan"), float("nan")
    return select_peak([periods[k] for k in keep], [found[k][1] for k in keep], guess)


# ------------------------------------------------------------- the whole

@dataclass
class Directional:
    """What `directional` measured: per layer and pair."""
    names: List[str]
    bin: float
    lags: np.ndarray                                   # nm, -K .. K
    image_axes: Tuple[str, str] = ("x", "y")
    images: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    line_axis: str = "x"
    line: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    profiles: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    spectra: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    periods: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    fft_periods: List[Tuple[float, float]] = field(default_factory=list)
    z: Optional[Tuple[float, float]] = None


AXIS = {"x": "u", "y": "v", "z": "z"}


def directional(layers: Sequence[Sequence[Dict[str, np.ndarray]]],
                masks: Sequence[Tuple[np.ndarray, float, float]], bin_nm: float,
                max_shift_nm: float, image: Optional[str] = "xy",
                line: Optional[str] = "x", cross: bool = True, guess: float = 0.0,
                names: Optional[Sequence[str]] = None) -> Directional:
    """The correlation image on ``image`` (a plane of `PLANES`, or None) and
    the correlation along ``line`` (x, y or z, or None) of every layer and
    pair.  ``layers[i][f]`` is layer i's ``{"u", "v"[, "z"]}`` in file f, in
    the frame; ``masks[f]`` file f's window (`lateral_mask`)."""
    if not bin_nm > 0:
        raise ValueError("the bin must be positive")
    lags = int(max(1, round(max_shift_nm / bin_nm)))
    k = len(layers)
    names = list(names) if names else [f"layer {i + 1}" for i in range(k)]
    jobs = [(i, i) for i in range(k)]
    if cross:
        jobs += [(i, j) for i in range(k) for j in range(i + 1, k)]
    wants_z = (image and "z" in image) or line == "z"
    zr = None
    if wants_z:
        every = [p["z"] for per in layers for p in per if "z" in p]
        zr = z_range(np.concatenate(every)) if every else None
        if zr is None:
            raise ValueError("no z_nm: the z axis needs 3D data")
    out = Directional(names, bin_nm, np.arange(-lags, lags + 1) * bin_nm, z=zr)

    def binned(axes):
        files = []
        for f, (mask, u0, v0) in enumerate(masks):
            files.append(bin_layers([layers[i][f] for i in range(k)], mask, u0, v0,
                                    axes, bin_nm, zr))
        return files

    if image:
        a, b = PLANES[image]
        out.image_axes = (a, b)
        files = binned((AXIS[a], AXIS[b]))
        for i, j in jobs:
            out.images[(i, j)] = correlate(files, i, j, lags)[0]
    if line:
        out.line_axis = line
        files = binned((AXIS[line],))
        for i, j in jobs:
            g = correlate(files, i, j, lags)[0]
            out.line[(i, j)] = g
            if i == j:
                out.periods[(i, j)] = period_of(out.lags, g, guess)
            else:
                ok = np.isfinite(g)
                if ok.any():
                    top = int(np.nanargmax(g))
                    out.periods[(i, j)] = refine(out.lags, np.where(ok, g, -np.inf), top)
                else:
                    out.periods[(i, j)] = (float("nan"), float("nan"))
        for i in range(k):
            # the profile of the first file with this layer, and its spectrum
            prof = next((f.images[i] for f in files if f.counts[i]), None)
            start = 0.0
            if line == "z":
                start = zr[0]
            elif files:
                start = masks[0][1] if line == "x" else masks[0][2]
            if prof is None:
                out.profiles.append((np.zeros(0), np.zeros(0)))
                out.spectra.append((np.zeros(0), np.zeros(0)))
                out.fft_periods.append((float("nan"), float("nan")))
                continue
            centres = start + (np.arange(len(prof)) + 0.5) * bin_nm
            out.profiles.append((centres, prof))
            f, mag = spectrum(prof, bin_nm)
            out.spectra.append((f, mag))
            out.fft_periods.append(spectrum_period(f, mag, guess, bin_nm))
    return out
