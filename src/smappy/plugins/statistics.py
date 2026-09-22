"""What the localizations look like: photons, precision and on-time.

Four distributions, each with the law it is expected to follow drawn over it,
so a glance says whether the data behaves and the numbers say how well.

**Photons.**  A blinking fluorophore emits until it goes dark, and the number
of photons collected before it does is exponential, ``p(N) = exp(-N/N0)/N0``.
The measured histogram is only exponential above its maximum -- below it the
detection threshold has eaten the dim localizations -- so the fit starts at the
maximum, and ``N0`` is the decay constant of what is left.

**Localization precision.**  It follows from the photons.  A precision is
``sigma = S / sqrt(N)`` with S a constant of the PSF and the pixel size, so an
exponential N gives sigma a distribution of its own::

    p(sigma) = (2a / sigma^3) * exp(-a / sigma^2),    a = S^2 / N0 = sigma_c^2

-- already normalized, one parameter, and that parameter is the precision at
the mean photon count, ``sigma_c = S / sqrt(N0)``.  Two numbers fall out of it
in closed form, and both are landmarks a person can read off the histogram:

* the maximum, ``sigma_max = sqrt(2a/3) = 0.8165 * sigma_c``
* the rising edge, where the curve first reaches half its maximum coming up
  from zero: ``sigma_rise = sqrt(a / u_half)`` with ``u_half`` the large root
  of ``u^1.5 * exp(-u) = 0.5 * 1.5^1.5 * exp(-1.5)`` (`HALF_MAX_U`)

Both are reported from the fit *and* read off the histogram itself.  They agree
when the photons are exponential and part company when they are not, which is
the point of printing them side by side.

**On-time.**  How many frames a fluorophore stays on before it blinks off, from
the grouped table's ``n_in_group``.  A constant off-rate makes it geometric,
``P(t) = (1 - q) q^(t-1)``, a straight line on a log axis, and the mean
on-time is ``tau = -1 / ln(q)`` frames.

The photon and on-time fits are maximum likelihood on the localizations, so
the bin width is a drawing choice and changes no number.  The precision is the
exception and is fitted to its histogram by least squares: the likelihood
weighs a localization by ``1/sigma^2``, so the handful of rows a fitter
returns with a precision of a fraction of a nanometre -- failed fits, not good
localizations -- carry the answer, and a few per cent of them pull ``sigma_c``
down by an order of magnitude.  Binned they are a few counts in bins the model
puts near zero and they move nothing.  ``precision fit`` puts the likelihood
back for anyone who wants it.

The functions are module level and take arrays: a script can have the same
numbers without a session, a plugin or a window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations
from . import Context, Plot, Plugin, Result, param, register

# the precision columns, in the order they are looked for
PRECISION_FIELDS = ("loc_precision_nm", "loc_precision_pix")
PRECISION_Z_FIELDS = ("loc_precision_z_nm",)
# the tails dropped before a precision is fitted.  A row with a precision of
# 0.1 nm is not a good localization, it is a bad fit, and the estimator below
# weighs it by 1/sigma^2 -- a hundred such rows in a million would carry the
# answer.  The fit is told about the cut (it is a truncated likelihood), so
# trimming costs no bias, only those rows.
TRIM_PERCENT = 0.5
MIN_FOR_FIT = 20                    # below this a fit says nothing
SMOOTH_BINS = 1.0                   # for the maximum and the rising edge


def _half_max_u() -> float:
    """The large root of ``u^1.5 exp(-u) = 0.5 * peak``, the rising edge in u.

    ``p(sigma)`` in ``u = a / sigma^2`` is proportional to ``u^1.5 exp(-u)``,
    which peaks at ``u = 1.5``.  Small sigma is large u, so the *rising* edge
    of the sigma histogram is the root above the peak.
    """
    def f(u: float) -> float:
        return u ** 1.5 * np.exp(-u) - 0.5 * 1.5 ** 1.5 * np.exp(-1.5)

    lo, hi = 1.5, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


HALF_MAX_U = _half_max_u()          # 3.4367...
MODE_OVER_SIGMA_C = np.sqrt(2 / 3)  # 0.8165
RISE_OVER_SIGMA_C = 1 / np.sqrt(HALF_MAX_U)


# ------------------------------------------------------------------ estimators

def exponential_mean(values, lo: float = 0.0, hi: float = np.inf) -> float:
    """The mean of the exponential that produced these values, seen through a cut.

    The sample is everything the experiment kept, ``lo <= x <= hi``, which is
    not the whole exponential; the maximum likelihood estimate of the *whole*
    one solves

        mean(x) = m + (lo - hi exp(-(hi - lo)/m)) / (1 - exp(-(hi - lo)/m))

    -- the mean of the truncated law, which is monotone in ``m``, so a
    bisection finds it without an optimizer.  With ``hi`` infinite this is the
    familiar ``mean(x) - lo``.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= lo) & (x <= hi)]
    if len(x) < 2:
        return float("nan")
    mean = float(x.mean())
    if not np.isfinite(hi):
        return max(mean - lo, np.finfo(float).tiny)
    if mean <= lo or mean >= hi:
        return float("nan")

    def truncated_mean(m: float) -> float:
        d = (hi - lo) / m
        if d > 700:                 # the upper cut is beyond reach
            return m + lo
        e = np.exp(-d)
        return m + (lo - hi * e) / (1 - e)

    # as m -> 0 the truncated mean -> lo, as m -> inf it -> (lo + hi)/2, and it
    # is monotone between; anything outside that range has no solution
    if mean >= 0.5 * (lo + hi):
        return float("inf")
    low, high = (hi - lo) * 1e-6, (hi - lo) * 1e6
    for _ in range(200):
        mid = np.sqrt(low * high)
        if truncated_mean(mid) < mean:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low * high))


def photon_decay(photons, start: float = 0.0) -> Dict[str, float]:
    """``N0`` of ``p(N) ~ exp(-N/N0)``, fitted to the localizations above ``start``."""
    values = np.asarray(photons, dtype=float)
    values = values[np.isfinite(values)]
    kept = values[values >= start]
    return {"n": len(values), "n_fitted": len(kept), "start": float(start),
            "n0": exponential_mean(kept, lo=start) if len(kept) >= MIN_FOR_FIT
                  else float("nan"),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "median": float(np.median(values)) if len(values) else float("nan")}


def precision_cdf(sigma, a: float) -> np.ndarray:
    """``P(sigma\' <= sigma)`` of the model, in closed form.

    ``y = 1/sigma^2`` is exponential with mean ``1/a``, so a small sigma is a
    large y and ``P(sigma\' <= s) = P(y >= 1/s^2) = exp(-a/s^2)``.  Exact per
    bin, which is what the histogram fit needs near zero where the density
    swings over a bin\'s width.
    """
    s = np.asarray(sigma, dtype=float)
    out = np.zeros_like(s)
    good = s > 0
    out[good] = np.exp(-a / s[good] ** 2)
    return out


def _histogram_residual(counts: np.ndarray, edges: np.ndarray, a: float) -> float:
    """Least squares of the model against the counts, amplitude profiled out."""
    p = np.diff(precision_cdf(edges, a))
    denom = float(p @ p)
    if denom <= 0:
        return float("inf")
    amplitude = float(counts @ p) / denom
    residual = counts - amplitude * p
    return float(residual @ residual)


def precision_from_histogram(counts, edges) -> Tuple[float, float]:
    """``(a, amplitude)`` fitting ``p(sigma)`` to a histogram by least squares.

    One free shape parameter, so the amplitude comes out of a linear solve for
    each ``a`` and the search is one-dimensional: a log grid over ``sigma_c``
    wide enough to hold any histogram, then a golden section on the bracket it
    picks.  There is no derivative and no optimizer to import.

    Least squares on the counts rather than a likelihood on the localizations,
    because that is what makes it robust.  A row whose fit collapsed carries a
    precision of a fraction of a nanometre; the unbinned estimator weighs a row
    by ``1/sigma^2``, so a few per cent of them outweigh the whole sample and
    the answer comes back several times too small.  Binned, those rows are a
    few counts in bins the model puts near zero, and a bounded residual against
    a peak of thousands moves the fit by nothing.
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if counts.sum() <= 0 or len(edges) != len(counts) + 1 or edges[-1] <= 0:
        return float("nan"), float("nan")
    # sigma_c is within a factor of a few of the histogram\'s peak; the grid is
    # far wider than that so no shape of histogram falls off the end of it
    top = float(edges[-1])
    grid = np.geomspace(top * 1e-4, top * 10, 240)
    losses = [_histogram_residual(counts, edges, sc ** 2) for sc in grid]
    i = int(np.argmin(losses))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(60):                    # golden section in log sigma_c
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if _histogram_residual(counts, edges, c ** 2) < \
                _histogram_residual(counts, edges, d ** 2):
            hi = d
        else:
            lo = c
    sigma_c = float(np.sqrt(lo * hi))
    a = sigma_c ** 2
    p = np.diff(precision_cdf(edges, a))
    denom = float(p @ p)
    amplitude = float(counts @ p) / denom if denom > 0 else float("nan")
    return a, amplitude


def precision_model(precision, low: Optional[float] = None,
                    high: Optional[float] = None, bins: int = 100,
                    method: str = "histogram") -> Dict[str, float]:
    """``sigma_c`` of ``p(sigma) = 2a/sigma^3 exp(-a/sigma^2)``, and its landmarks.

    Two estimators, and the default is the binned one:

    ``"histogram"`` fits the model to the precision histogram by least squares
    (`precision_from_histogram`).  It is what a person would do by eye, and it
    survives the rows a fitter produces with a precision of a fraction of a
    nanometre -- rows that are not good localizations but failed fits.

    ``"mle"`` is the unbinned maximum likelihood: ``y = 1/sigma^2`` turns the
    model into an exponential in ``y`` with mean ``1/a`` -- ``y`` *is*
    ``N/S^2`` -- so the photon estimator above fits it, truncated at ``low``
    and ``high`` where the sample was cut (a filter on the precision column,
    say), or at the trimmed percentiles when those are not given.  It is the
    efficient estimator on clean data and it is the one that goes wrong on
    real data: weighing each row by ``1/sigma^2`` is exactly what hands the
    answer to the smallest few.
    """
    values = np.asarray(precision, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    out: Dict[str, float] = {
        "n": len(values),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "method": method}
    if len(values) < MIN_FOR_FIT:
        out.update(sigma_c=float("nan"), max=float("nan"), rising=float("nan"),
                   low=float("nan"), high=float("nan"))
        return out
    trimmed = np.percentile(values, (TRIM_PERCENT, 100 - TRIM_PERCENT))
    low = float(trimmed[0]) if low is None else float(low)
    high = float(trimmed[1]) if high is None else float(high)
    if not high > low > 0:
        low, high = float(trimmed[0]), float(trimmed[1])
    if method == "histogram":
        counts, edges = _binned(values, bins, 0.0, precision_range(values))
        a, amplitude = precision_from_histogram(counts, edges)
        out["amplitude"] = amplitude
    else:
        y = 1.0 / values ** 2
        mean_y = exponential_mean(y, lo=1.0 / high ** 2, hi=1.0 / low ** 2)
        a = 1.0 / mean_y if mean_y > 0 else float("nan")
    sigma_c = float(np.sqrt(a))
    out.update(sigma_c=sigma_c, a=float(a), low=float(low), high=float(high),
               max=sigma_c * MODE_OVER_SIGMA_C, rising=sigma_c * RISE_OVER_SIGMA_C)
    return out


def precision_range(values) -> float:
    """Where the precision histogram stops: the 99.5th percentile.

    The same cut for the picture and for the fit, so the curve drawn is the
    curve fitted and a bin nobody can see cannot move it.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(values, 99.5)) if len(values) else 1.0


def precision_density(sigma, a: float) -> np.ndarray:
    """``p(sigma)`` of the model, normalized over ``(0, inf)``."""
    s = np.asarray(sigma, dtype=float)
    out = np.zeros_like(s)
    good = s > 0
    out[good] = 2 * a / s[good] ** 3 * np.exp(-a / s[good] ** 2)
    return out


def ontime_decay(frames) -> Dict[str, float]:
    """``tau`` of the geometric on-time, in frames.

    ``t >= 1`` and ``P(t) = (1 - q) q^(t-1)`` has ``mean(t) = 1/(1 - q)``, so
    the estimate is the sample mean and nothing else.  ``tau = -1/ln q`` is the
    equivalent exponential lifetime -- what the curve decays with, rather than
    the count, which is what makes it comparable to a bleaching time.
    """
    t = np.asarray(frames, dtype=float)
    t = t[np.isfinite(t) & (t >= 1)]
    out: Dict[str, float] = {"n": len(t),
                             "mean": float(t.mean()) if len(t) else float("nan"),
                             "median": float(np.median(t)) if len(t) else float("nan")}
    if len(t) < MIN_FOR_FIT or out["mean"] <= 1:
        # every molecule seen in exactly one frame: no decay to fit
        out.update(q=float("nan"), tau=float("nan"))
        return out
    q = 1.0 - 1.0 / out["mean"]
    out.update(q=float(q), tau=float(-1.0 / np.log(q)))
    return out


# ------------------------------------------------------- reading the histogram

def histogram_landmarks(counts, edges, smooth: float = SMOOTH_BINS
                        ) -> Dict[str, float]:
    """The maximum and the rising edge, read off the histogram itself.

    The counts are smoothed first (a histogram's noisiest bin is otherwise its
    maximum), the peak is refined by a parabola through its two neighbours, and
    the rising edge is where the smoothed curve last crosses half the peak
    below it, interpolated between bin centres.
    """
    counts = np.asarray(counts, dtype=float)
    centers = 0.5 * (np.asarray(edges[:-1], float) + np.asarray(edges[1:], float))
    out = {"max": float("nan"), "rising": float("nan"), "peak": float("nan")}
    if len(counts) < 3 or not counts.any():
        return out
    if smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(counts, smooth, mode="nearest")
    else:
        smoothed = counts
    peak = int(np.argmax(smoothed))
    width = centers[1] - centers[0]
    position = centers[peak]
    if 0 < peak < len(smoothed) - 1:            # parabolic refinement
        left, middle, right = smoothed[peak - 1:peak + 2]
        denominator = left - 2 * middle + right
        if denominator < 0:
            position += 0.5 * width * (left - right) / denominator
    out["max"] = float(position)
    out["peak"] = float(smoothed[peak])
    half = 0.5 * smoothed[peak]
    below = np.flatnonzero(smoothed[:peak] < half)
    if len(below):
        i = int(below[-1])                      # the last crossing before the peak
        span = smoothed[i + 1] - smoothed[i]
        frac = (half - smoothed[i]) / span if span > 0 else 0.0
        out["rising"] = float(centers[i] + frac * width)
    return out


# --------------------------------------------------------------- distributions

@dataclass
class Distribution:
    """One quantity: what was counted, what it is expected to follow, and both
    sets of numbers."""
    key: str
    label: str
    unit: str = ""
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stats: Dict[str, float] = field(default_factory=dict)
    curve: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (x, counts) of the fit
    log_y: bool = False
    summary: str = ""
    note: str = ""


def _binned(values, bins: int, low=None, high=None) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if low is None:
        low = values.min() if len(values) else 0.0
    if high is None:
        high = values.max() if len(values) else 1.0
    if not high > low:
        high = low + 1.0
    return np.histogram(values, bins=bins, range=(float(low), float(high)))


def photon_distribution(photons, bins: int = 100, start: float = 0.0
                        ) -> Distribution:
    """The photon histogram with its exponential tail fitted."""
    values = np.asarray(photons, dtype=float)
    values = values[np.isfinite(values)]
    high = np.percentile(values, 99.9) if len(values) else 1.0
    counts, edges = _binned(values, bins, 0.0, high)
    landmarks = histogram_landmarks(counts, edges)
    if start <= 0:                       # start where the threshold stops biting
        start = landmarks["max"] if np.isfinite(landmarks["max"]) else 0.0
    stats = photon_decay(values, start)
    stats["histogram_max"] = landmarks["max"]
    curve = None
    n0 = stats["n0"]
    if np.isfinite(n0) and n0 > 0:
        x = np.linspace(start, edges[-1], 200)
        width = edges[1] - edges[0]
        scale = stats["n_fitted"] * width / n0
        curve = (x, scale * np.exp(-(x - start) / n0))
    return Distribution(
        key="photons", label="photons", counts=counts, edges=edges, stats=stats,
        curve=curve,
        summary=(f"N0 = {n0:.0f} photons above {start:.0f} "
                 f"(mean {stats['mean']:.0f}, median {stats['median']:.0f}, "
                 f"{stats['n']} localizations)"))


def precision_distribution(precision, bins: int = 100, key: str = "precision",
                           label: str = "localization precision",
                           unit: str = "nm", method: str = "histogram") -> Distribution:
    """The precision histogram with the model the exponential photons imply."""
    values = np.asarray(precision, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    stats = precision_model(values, bins=bins, method=method)
    counts, edges = _binned(values, bins, 0.0, precision_range(values))
    landmarks = histogram_landmarks(counts, edges)
    stats["histogram_max"] = landmarks["max"]
    stats["histogram_rising"] = landmarks["rising"]
    curve = None
    a = stats.get("a", float("nan"))
    if np.isfinite(a) and a > 0:
        x = np.linspace(edges[0] + 1e-6, edges[-1], 400)
        width = edges[1] - edges[0]
        # the binned fit carries its own amplitude: the counts it describes are
        # the ones in the picture, which is fewer than n when a tail was cut
        scale = stats.get("amplitude", stats["n"])
        if not np.isfinite(scale):
            scale = stats["n"]
        curve = (x, scale * width * precision_density(x, a))
    return Distribution(
        key=key, label=label, unit=unit, counts=counts, edges=edges, stats=stats,
        curve=curve,
        summary=(f"max at {stats['histogram_max']:.1f} {unit} "
                 f"(model {stats['max']:.1f}), rising edge "
                 f"{stats['histogram_rising']:.1f} {unit} "
                 f"(model {stats['rising']:.1f}), median {stats['median']:.1f}, "
                 f"sigma_c = {stats['sigma_c']:.1f}"))


def ontime_distribution(frames, exposure_ms: float = 0.0) -> Distribution:
    """The on-time histogram, one bin per frame, with its geometric fit."""
    t = np.asarray(frames, dtype=float)
    t = t[np.isfinite(t) & (t >= 1)]
    stats = ontime_decay(t)
    high = max(int(np.percentile(t, 99.9)) if len(t) else 1, 2)
    edges = np.arange(0.5, high + 1.5)
    counts, edges = np.histogram(t, bins=edges)
    curve = None
    q, tau = stats["q"], stats["tau"]
    if np.isfinite(q) and 0 < q < 1:
        x = np.arange(1, high + 1, dtype=float)
        curve = (x, stats["n"] * (1 - q) * q ** (x - 1))
    time = f", {tau * exposure_ms:.0f} ms" if exposure_ms > 0 and np.isfinite(tau) else ""
    return Distribution(
        key="on_time", label="on-time", unit="frames", counts=counts, edges=edges,
        stats=stats, curve=curve, log_y=True,
        summary=(f"tau = {tau:.2f} frames{time} "
                 f"(mean {stats['mean']:.2f}, {stats['n']} blinks)"))


def statistics(locs: Localizations, bins: int = 100, photon_start: float = 0.0,
               on_time: Optional[Sequence[float]] = None,
               exposure_ms: float = 0.0,
               precision_fit: str = "histogram") -> List[Distribution]:
    """Every distribution the table can supply, in reading order.

    ``on_time`` is given separately because it lives in the *grouped* table:
    one row per blink, not one per frame.  A table that carries ``n_in_group``
    already -- a grouped file, or a grouped layer -- supplies it itself.
    """
    found: List[Distribution] = []
    if "photons" in locs:
        found.append(photon_distribution(locs["photons"], bins, photon_start))
    lateral = next((n for n in PRECISION_FIELDS if n in locs), None)
    if lateral:
        found.append(precision_distribution(
            locs[lateral], bins, key="precision",
            label="localization precision",
            unit="nm" if lateral.endswith("_nm") else "pixel",
            method=precision_fit))
    axial = next((n for n in PRECISION_Z_FIELDS if n in locs), None)
    if axial:
        found.append(precision_distribution(
            locs[axial], bins, key="precision_z", label="z precision", unit="nm",
            method=precision_fit))
    if on_time is None and "n_in_group" in locs:
        on_time = locs["n_in_group"]
    if on_time is not None and len(on_time):
        found.append(ontime_distribution(on_time, exposure_ms))
    return found


# --------------------------------------------------------------------- drawing

def draw(ax, dist: Distribution) -> None:
    """One panel: the histogram, the fit over it, and the landmarks."""
    width = np.diff(dist.edges)
    ax.bar(dist.edges[:-1], dist.counts, width=width, align="edge",
           color="0.75", edgecolor="0.45", linewidth=0.4)
    if dist.curve is not None:
        x, y = dist.curve
        ax.plot(x, y, color="#d62728", linewidth=1.5)
    if dist.key.startswith("precision"):
        # solid: read off the histogram.  dashed: where the model puts it.
        for name, color, style, label in (
                ("histogram_max", "#1f77b4", "-", "maximum"),
                ("histogram_rising", "#2ca02c", "-", "rising edge"),
                ("max", "#1f77b4", "--", "maximum, model"),
                ("rising", "#2ca02c", "--", "rising edge, model")):
            value = dist.stats.get(name, float("nan"))
            if np.isfinite(value):
                ax.axvline(value, color=color, linestyle=style, linewidth=1.0,
                           label=label)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    if dist.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    ax.set_xlabel(f"{dist.label} ({dist.unit})" if dist.unit else dist.label)
    ax.set_ylabel("localizations" if dist.key != "on_time" else "blinks")
    ax.set_title(dist.summary, fontsize=7.5, color="0.25")
    if dist.note:
        ax.text(0.98, 0.9, dist.note, transform=ax.transAxes, ha="right",
                fontsize=7, color="0.45")


def draw_all(figure, found: Sequence[Distribution]) -> None:
    """Every panel, one under the other, in one figure.

    Neither the size nor the layout engine is set here: the figure may be a
    `SubFigure` of a page of small multiples, which has neither, and the size
    is a hint the window reads (`Plot.size`) rather than something a plugin
    imposes on the window it landed in.
    """
    for axis, dist in zip(figure.subplots(len(found), 1, squeeze=False).ravel(),
                          found):
        draw(axis, dist)


# ---------------------------------------------------------------- the plugin

@dataclass
class StatisticsSettings:
    source: str = param("selection", label="localizations",
                        choices=(("selection", "as plotted (filter and ROI)"),
                                 ("all", "all, unfiltered")),
                        help="the localizations on screen, or the whole table")
    bins: int = param(100, label="bins", min=5,
                      help="the photon and on-time fits are on the "
                           "localizations and ignore this; the precision fit "
                           "is on these bins")
    precision_fit: str = param("histogram", label="precision fit",
                               choices=(("histogram", "least squares on the histogram"),
                                        ("mle", "maximum likelihood (outlier-sensitive)")),
                               advanced=True,
                               help="the likelihood weighs a localization by "
                                    "1/sigma^2, so a few failed fits with a "
                                    "precision near zero carry the answer; "
                                    "the binned fit does not see them")
    photon_start: float = param(0.0, label="photons from", unit="photons", min=0.0,
                                help="where the exponential fit starts; "
                                     "0: at the maximum of the histogram, below "
                                     "which the detection threshold has eaten the "
                                     "dim localizations")
    exposure_ms: float = param(0.0, label="exposure", unit="ms", min=0.0,
                               advanced=True,
                               help="0: the on-time is reported in frames only")


def grouped_on_time(session, layer: int = 0, selected: bool = True
                    ) -> Tuple[Optional[np.ndarray], str]:
    """``n_in_group`` from the session's grouped table, and why not if not.

    The grouped table is a table of its own with a filter of its own, so the
    selection is rebuilt here rather than carried over: the same bounds and the
    same ROI, applied to blinks instead of to frames.  Nothing is linked -- a
    session that has never been switched to grouped has no on-time to show, and
    saying so is better than spending a minute of someone's time on it.
    """
    if session is None:
        return None, "no session: no grouped table to take the on-time from"
    layers = getattr(session, "layers", None)
    if not layers:
        return None, "no layer to take the on-time from"
    index = layer if 0 <= layer < len(layers) else 0
    if getattr(layers[index], "is_image", False):
        index = session.first_locs_layer()
    state = getattr(layers[index], "state", None)
    sets = getattr(state, "sets", {}) or {}
    if "grouped" not in sets:
        return None, ("no on-time: switch the layer to grouped once, so the "
                      "localizations are linked into blinks")
    if getattr(state, "grouped_stale", False):
        return None, ("no on-time: the grouped table is out of date; switch the "
                      "layer to grouped to link it again")
    locset = sets["grouped"]
    locs = locset.locs
    if "n_in_group" not in locs or not len(locs):
        return None, "no on-time: the grouped table has no n_in_group"
    if not selected:
        return np.asarray(locs["n_in_group"]), ""
    from ..render import positions
    mask = np.asarray(locset.filter.mask, dtype=bool)
    roi = getattr(session, "roi", None)
    slab = getattr(session, "slab", None)
    if roi is not None or (getattr(session, "select_in_slab", False) and slab):
        x, y = positions(locs)
        if roi is not None:
            mask = mask & roi.mask(x, y)
        if getattr(session, "select_in_slab", False) and slab is not None:
            z = locs["z_nm"] if "z_nm" in locs else None
            mask = mask & slab.mask(x, y, z)
    return np.asarray(locs["n_in_group"])[mask], ""


@register("Analysis/Measure/Localization Statistics")
class LocalizationStatistics(Plugin):
    """Photons, localization precision and on-time, each with its law fitted."""

    Settings = StatisticsSettings
    version = "1"

    def run(self, ctx: Context, settings: StatisticsSettings) -> Result:
        if settings.source == "all":
            locs = ctx.locs
            where = "all localizations"
        else:
            ctx.selection.require(MIN_FOR_FIT, ctx.report, "a distribution")
            locs = ctx.selection.apply(ctx.locs)
            where = str(ctx.selection)
        if not len(locs):
            raise ValueError("no localizations to describe")
        on_time, why = None, ""
        if "n_in_group" not in locs:
            on_time, why = grouped_on_time(ctx.session, ctx.layer,
                                           settings.source != "all")
        found = statistics(locs, bins=settings.bins,
                           photon_start=settings.photon_start, on_time=on_time,
                           exposure_ms=settings.exposure_ms,
                           precision_fit=settings.precision_fit)
        if not found:
            raise ValueError("the table has no photons, precision or on-time: "
                             f"it has {', '.join(sorted(locs.keys()))}")
        if why:
            ctx.report(why)
        lines = [f"{len(locs)} localizations, {where}"]
        lines += [f"{d.label}: {d.summary}" for d in found]
        if why:
            lines.append(why)
        data = {"distributions": {d.key: d for d in found},
                "stats": {d.key: d.stats for d in found},
                "n": len(locs), "note": why}

        def plot(figure) -> None:
            """One panel per distribution, in whatever it is given to draw in.

            A single axis is right for a plugin with one thing to draw and
            there are up to four here, so this declares its panels and is
            handed the figure -- which on the window's *All* page is one
            subfigure of it, and works the same.
            """
            draw_all(figure, found)

        return Result(text="\n".join(lines), data=data, settings=settings,
                      plot=Plot(draw=plot, panels=len(found),
                                size=(5.5, max(2.1 * len(found), 2.1))))
