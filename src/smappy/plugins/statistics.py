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
``n_in_group``, one value per blink.  A constant off-rate makes it geometric,
``P(t) = (1 - q) q^(t-1)``, a straight line on a log axis, and the mean
on-time is ``tau = -1 / ln(q)`` frames -- and in seconds with the file's
frame time (`metadata.table_frame_time`), or the *frame time* set here.

The photon and on-time fits are maximum likelihood on the localizations, so
the bin width is a drawing choice and changes no number.  The precision is the
exception and is fitted to its histogram by least squares: the likelihood
weighs a localization by ``1/sigma^2``, so the handful of rows a fitter
returns with a precision of a fraction of a nanometre -- failed fits, not good
localizations -- carry the answer, and a few per cent of them pull ``sigma_c``
down by an order of magnitude.  Binned they are a few counts in bins the model
puts near zero and they move nothing.  ``precision fit`` puts the likelihood
back for anyone who wants it.

**Per frame.**  How many localizations each frame holds, over the
acquisition: the dye bleaches and the density thins out, or the activation
laser is turned up and it climbs.  A constant bleaching rate makes the count
decay as ``exp(-t/T)``, so the frames of the localizations are a sample of
an exponential cut to the acquisition, and ``T`` is `exponential_mean` of
them over ``[first, last]`` -- the same truncated estimator as the photons,
needing no bins.  A count that does not fall has no ``T`` (it is reported as
none); the median frame, by which half the localizations are in, is
reported either way.  One line per visible layer, as each shows it, in a tab
of its own: *Time Course* is the place for the rate against seconds, per
file and beside other quantities.

**Which table.**  The one the layer shows: a grouped layer is described blink
by blink (the photons added, the precision of the combined position), an
ungrouped one frame by frame -- the same table the count beside the picture
counts.  The on-time is the exception, because it is a property of the blink:
once a table has been linked, every ungrouped row carries its blink's
``n_in_group``, and taking that column as it stands counted a three-frame blink
three times and a one-frame blink once, which weighs the histogram towards the
long ones.  It is reduced to one value per ``group_id`` first, so both views
report the same on-time.

The functions are module level and take arrays: a script can have the same
numbers without a session, a plugin or a window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations
from ..metadata import table_frame_time
from . import Context, Plot, Plugin, Result, param, register

# the precision columns, in the order they are looked for
PRECISION_FIELDS = ("xy_err_nm",)
PRECISION_Z_FIELDS = ("z_err_nm",)
# the tails dropped before a precision is fitted.  A row with a precision of
# 0.1 nm is not a good localization, it is a bad fit, and the estimator below
# weighs it by 1/sigma^2 -- a hundred such rows in a million would carry the
# answer.  The fit is told about the cut (it is a truncated likelihood), so
# trimming costs no bias, only those rows.
TRIM_PERCENT = 0.5
MIN_FOR_FIT = 20                    # below this a fit says nothing
SMOOTH_BINS = 1.0                   # for the maximum and the rising edge
TITLE_SIZE = 9                      # points: a panel's fit, above it


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
    (`precision_from_histogram`), over ``low`` to ``high`` when they are given:
    a filter's cut, fitted as a truncated histogram.  It is what a person would do by eye, and it
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
    if len(values) < MIN_FOR_FIT or method == "none":
        # "none": the median and the histogram's own landmarks only -- the fit
        # is most of what this costs, ~10 ms whatever the count
        out.update(sigma_c=float("nan"), max=float("nan"), rising=float("nan"),
                   low=float("nan"), high=float("nan"))
        return out
    cut = (low, high)
    trimmed = np.percentile(values, (TRIM_PERCENT, 100 - TRIM_PERCENT))
    low = float(trimmed[0]) if low is None else float(low)
    high = float(trimmed[1]) if high is None else float(high)
    if not high > low > 0:
        low, high = float(trimmed[0]), float(trimmed[1])
    if method == "histogram":
        # a cut that was given is where the histogram starts or stops, so the
        # model is fitted to the part that survived and not to empty bins
        # below a filter -- which read as a distribution shifted up
        start = float(cut[0]) if cut[0] is not None and cut[0] > 0 else 0.0
        stop = precision_range(values)
        if cut[1] is not None and start < cut[1] < stop:
            stop = float(cut[1])
        counts, edges = _binned(values, bins, start, stop)
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
    short: str = ""         # the fit alone, for a legend beside other layers'


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


def photon_distribution(photons, bins: int = 100, start: float = 0.0,
                        high: Optional[float] = None) -> Distribution:
    """The photon histogram with its exponential tail fitted.  ``high`` is
    where the histogram stops, its own 99.9th percentile by default: layers
    drawn on one axis share one (`shared_highs`), and the fit ignores it."""
    values = np.asarray(photons, dtype=float)
    values = values[np.isfinite(values)]
    if high is None:
        high = own_high("photons", values)
    counts, edges = _binned(values, bins, 0.0, high)
    landmarks = histogram_landmarks(counts, edges)
    if start <= 0:                       # start where the threshold stops biting
        start = landmarks["max"] if np.isfinite(landmarks["max"]) else 0.0
    stats = photon_decay(values, start)
    stats["histogram_max"] = landmarks["max"]
    n0 = stats["n0"]
    return Distribution(
        key="photons", label="photons", counts=counts, edges=edges, stats=stats,
        curve=model_curve("photons", stats, edges),
        short=f"N0 = {n0:.0f}",
        summary=(f"N0 = {n0:.0f} photons above {start:.0f} "
                 f"(mean {stats['mean']:.0f}, median {stats['median']:.0f}, "
                 f"{stats['n']} localizations)"))


def precision_distribution(precision, bins: int = 100, key: str = "precision",
                           label: str = "localization precision",
                           unit: str = "nm", method: str = "histogram",
                           high: Optional[float] = None) -> Distribution:
    """The precision histogram with the model the exponential photons imply.
    ``high``: where the histogram is drawn to (`photon_distribution`); the
    fit keeps its own range."""
    values = np.asarray(precision, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    stats = precision_model(values, bins=bins, method=method)
    if high is None:
        high = precision_range(values)
    counts, edges = _binned(values, bins, 0.0, high)
    landmarks = histogram_landmarks(counts, edges)
    stats["histogram_max"] = landmarks["max"]
    stats["histogram_rising"] = landmarks["rising"]
    return Distribution(
        key=key, label=label, unit=unit, counts=counts, edges=edges, stats=stats,
        curve=model_curve(key, stats, edges),
        short=(f"sigma_c = {stats['sigma_c']:.1f} {unit}" if np.isfinite(stats["sigma_c"])
               else f"median {stats['median']:.1f} {unit}"),
        summary=(f"max at {stats['histogram_max']:.1f} {unit} "
                 f"(model {stats['max']:.1f}), rising edge "
                 f"{stats['histogram_rising']:.1f} {unit} "
                 f"(model {stats['rising']:.1f}), median {stats['median']:.1f}, "
                 f"sigma_c = {stats['sigma_c']:.1f}"))


def ontime_distribution(frames, frame_time_ms: Optional[float] = None,
                        high: Optional[float] = None) -> Distribution:
    """The on-time histogram, one bin per frame, with its geometric fit, and
    ``tau`` in ms too (``stats["tau_ms"]``) with a frame time.  ``high``:
    the longest on-time drawn (`photon_distribution`)."""
    t = np.asarray(frames, dtype=float)
    t = t[np.isfinite(t) & (t >= 1)]
    stats = ontime_decay(t)
    high = int(own_high("on_time", t) if high is None else high)
    edges = np.arange(0.5, high + 1.5)
    counts, edges = np.histogram(t, bins=edges)
    tau = stats["tau"]
    curve = model_curve("on_time", stats, edges)
    known = bool(frame_time_ms) and frame_time_ms > 0
    stats["tau_ms"] = tau * frame_time_ms if known else float("nan")
    time = f", {stats['tau_ms']:.0f} ms" if known and np.isfinite(tau) else ""
    return Distribution(
        key="on_time", label="on-time", unit="frames", counts=counts, edges=edges,
        stats=stats, curve=curve, log_y=True,
        short=f"tau = {tau:.2f} frames{time}",
        summary=(f"tau = {tau:.2f} frames{time} "
                 f"(mean {stats['mean']:.2f}, {stats['n']} blinks)"))


def model_curve(key: str, stats: Dict[str, float], edges: np.ndarray
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The fitted law over a histogram, in counts per bin, from the fit's
    numbers alone -- so a figure kept as its histogram and numbers
    (`LocalizationStatistics.keep`) draws its curve again.  None when the
    law was not fitted."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 2:
        return None
    width = edges[1] - edges[0]
    if key == "photons":
        n0, start = stats.get("n0", np.nan), stats.get("start", 0.0)
        if not (np.isfinite(n0) and n0 > 0):
            return None
        x = np.linspace(start, edges[-1], 200)
        return x, stats["n_fitted"] * width / n0 * np.exp(-(x - start) / n0)
    if key.startswith("precision"):
        a = stats.get("a", float("nan"))
        if not (np.isfinite(a) and a > 0):
            return None
        top = stats.get("high", np.nan)
        top = edges[-1] if not np.isfinite(top) else min(top, edges[-1])
        x = np.linspace(edges[0] + 1e-6, top, 400)
        # the binned fit carries its own amplitude: the counts it describes are
        # the ones in the picture, which is fewer than n when a tail was cut
        scale = stats.get("amplitude", stats["n"])
        if not np.isfinite(scale):
            scale = stats["n"]
        return x, scale * width * precision_density(x, a)
    if key == "on_time":
        q = stats.get("q", np.nan)
        if not (np.isfinite(q) and 0 < q < 1):
            return None
        x = np.arange(1, int(round(edges[-1] - 0.5)) + 1, dtype=float)
        return x, stats["n"] * (1 - q) * q ** (x - 1)
    return None


def statistics(locs: Localizations, bins: int = 100, photon_start: float = 0.0,
               on_time: Optional[Sequence[float]] = None,
               frame_time_ms: Optional[float] = None,
               precision_fit: str = "histogram",
               highs: Optional[Dict[str, float]] = None) -> List[Distribution]:
    """Every distribution the table can supply, in reading order.

    ``on_time`` is given separately because it lives in the *grouped* table:
    one row per blink, not one per frame.  A table that carries ``n_in_group``
    already -- a grouped file, or a grouped layer -- supplies it itself.
    ``precision_fit`` is `precision_model`'s method; "none" fits no model,
    and the median and the histogram's landmarks are still there.  ``highs``
    are where each histogram stops, by key (`shared_highs`): for layers
    drawn on one axis; they change the picture, not the fits.
    """
    highs = highs or {}
    found: List[Distribution] = []
    if "photons" in locs:
        found.append(photon_distribution(locs["photons"], bins, photon_start,
                                         high=highs.get("photons")))
    lateral = next((n for n in PRECISION_FIELDS if n in locs), None)
    if lateral:
        found.append(precision_distribution(
            locs[lateral], bins, key="precision",
            label="localization precision",
            unit="nm" if lateral.endswith("_nm") else "pixel",
            method=precision_fit, high=highs.get("precision")))
    axial = next((n for n in PRECISION_Z_FIELDS if n in locs), None)
    if axial:
        found.append(precision_distribution(
            locs[axial], bins, key="precision_z", label="z precision", unit="nm",
            method=precision_fit, high=highs.get("precision_z")))
    if on_time is None and "n_in_group" in locs:
        on_time = locs["n_in_group"]
    if on_time is not None and len(on_time):
        found.append(ontime_distribution(on_time, frame_time_ms,
                                         high=highs.get("on_time")))
    return found


def own_high(key: str, values) -> float:
    """Where one table's histogram of ``key`` stops: the 99.9th percentile of
    the photons and the on-time, the precision's `precision_range`."""
    values = np.asarray(values, dtype=float)
    if key.startswith("precision"):
        return precision_range(values)
    values = values[np.isfinite(values)]
    if key == "on_time":
        values = values[values >= 1]
        return max(int(np.percentile(values, 99.9)) if len(values) else 1, 2)
    return float(np.percentile(values, 99.9)) if len(values) else 1.0


def shared_highs(tables: Sequence[Tuple[Localizations, Optional[Sequence[float]]]]
                 ) -> Dict[str, float]:
    """One upper end per histogram for several layers, ``(locs, on_time)``
    each: the largest of their own.  Layers drawn as steps on one axis need
    the same bins, or a layer with a longer tail has wider bins and, per bin,
    more counts than it has per photon."""
    columns = {"photons": ("photons",), "precision": PRECISION_FIELDS,
               "precision_z": PRECISION_Z_FIELDS}
    out: Dict[str, float] = {}
    for locs, on_time in tables:
        found = {key: locs[next(n for n in names if n in locs)]
                 for key, names in columns.items() if any(n in locs for n in names)}
        if on_time is None and "n_in_group" in locs:
            on_time = locs["n_in_group"]
        if on_time is not None and len(on_time):
            found["on_time"] = on_time
        for key, values in found.items():
            if len(values):
                out[key] = max(out.get(key, -np.inf), own_high(key, values))
    return out


def statistics_values(locs: Localizations, found: Sequence[Distribution]
                      ) -> Dict[str, float]:
    """The numbers of a description, a column each of a region table.

    The count, then per distribution its centre and its fitted parameter:
    the photons' mean, median and ``N0``; the precision's mean, median and
    ``sigma_c`` (NaN when the model was not fitted), in the column's unit;
    the on-time's ``tau``, in frames and in seconds (NaN without a frame
    time), and the blinks it was counted over.
    """
    out: Dict[str, float] = {"n_localizations": len(locs)}
    by_key = {d.key: d for d in found}
    if "photons" in by_key:
        stats = by_key["photons"].stats
        out.update(mean_photons=stats["mean"], median_photons=stats["median"],
                   photons_n0=stats["n0"])
    for key, column_names, suffix in (("precision", PRECISION_FIELDS, ""),
                                      ("precision_z", PRECISION_Z_FIELDS, "z_")):
        if key not in by_key:
            continue
        column = next(n for n in column_names if n in locs)
        unit = "nm"
        values = np.asarray(locs[column], dtype=float)
        values = values[np.isfinite(values)]
        stats = by_key[key].stats
        out[f"mean_{suffix}precision_{unit}"] = (float(values.mean()) if len(values)
                                                 else float("nan"))
        out[f"median_{suffix}precision_{unit}"] = stats["median"]
        out[f"sigma_c_{suffix}{unit}"] = stats["sigma_c"]
    if "on_time" in by_key:
        stats = by_key["on_time"].stats
        out.update(on_time_frames=stats["tau"],
                   on_time_s=stats.get("tau_ms", float("nan")) / 1000.0,
                   n_blinks=stats["n"])
    return {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
            for k, v in out.items()}


# ---------------------------------------------------------------- per frame

#: localizations per frame are counted in about this many windows, whatever
#: the length of the acquisition: 20000 single frames drawn as bars are a
#: grey smear, and the trend is what the panel is for
PER_FRAME_BINS = 200
#: a decay time longer than this many acquisitions is no decay: over the
#: acquisition the count falls by less than 1 - exp(-1/10), 10 %
NO_DECAY_SPANS = 10.0


@dataclass
class FrameCounts:
    """The localizations per frame of one layer, and their decay."""
    label: str
    colour: str
    counts: np.ndarray            # per window
    edges: np.ndarray             # frames, the windows' bounds
    stats: Dict[str, float] = field(default_factory=dict)
    merged: bool = False          # blinks, not localizations

    @property
    def rate(self) -> np.ndarray:
        """Per frame: the counts over each window's width."""
        return self.counts / np.maximum(np.diff(self.edges), 1e-12)

    def curve(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """The fitted decay in localizations per frame, or None without one."""
        t, first, last, n = (self.stats.get(k, np.nan)
                             for k in ("decay_frames", "first", "last", "n"))
        if not (np.isfinite(t) and t > 0 and last > first):
            return None
        x = np.linspace(first, last, 300)
        norm = t * (1.0 - np.exp(-(last - first) / t))
        return x, n * np.exp(-(x - first) / t) / norm


def frame_decay(frames, first: Optional[float] = None,
                last: Optional[float] = None) -> Dict[str, float]:
    """How the count per frame falls over ``[first, last]``.

    ``decay_frames`` is ``T`` of a rate ``exp(-t/T)``, the maximum
    likelihood estimate (`exponential_mean` on the frames, cut to the
    acquisition), NaN with too few localizations and inf when the count does
    not fall; ``median_frame`` is where half of them are in, which needs no
    model; ``first_rate`` and ``last_rate`` are the mean per frame in the
    first and the last tenth, which say how much it fell when it is not
    exponential.
    """
    f = np.asarray(frames, dtype=float)
    f = f[np.isfinite(f)]
    out: Dict[str, float] = {"n": len(f)}
    if not len(f):
        out.update(first=np.nan, last=np.nan, decay_frames=np.nan,
                   median_frame=np.nan, first_rate=np.nan, last_rate=np.nan)
        return out
    first = float(f.min()) if first is None else float(first)
    last = float(f.max()) if last is None else float(last)
    span = max(last - first, 1.0)
    tenth = span / 10.0
    out.update(first=first, last=last, median_frame=float(np.median(f)),
               first_rate=float(np.sum(f < first + tenth) / tenth),
               last_rate=float(np.sum(f > last - tenth) / tenth))
    t = exponential_mean(f, first, last) if len(f) >= MIN_FOR_FIT else float("nan")
    # a decay many times the acquisition is a flat count with noise on it:
    # the 1 % it falls over the acquisition is not something to report a
    # time for
    out["decay_frames"] = float("inf") if t > NO_DECAY_SPANS * span else t
    return out


def frame_counts(frames, label: str = "", colour: str = "0.3",
                 bins: int = PER_FRAME_BINS, frame_time_ms: Optional[float] = None,
                 merged: bool = False) -> FrameCounts:
    """Localizations per frame in windows of whole frames, with the decay;
    ``decay_s`` too with a frame time."""
    f = np.asarray(frames, dtype=float)
    f = f[np.isfinite(f)]
    lo, hi = (float(f.min()), float(f.max())) if len(f) else (0.0, 1.0)
    width = max(1, int(np.ceil((hi - lo + 1) / bins)))
    edges = np.arange(lo, hi + width + 1, width, dtype=float)
    counts, edges = np.histogram(f, bins=edges)
    stats = frame_decay(f, lo, hi)
    known = bool(frame_time_ms) and frame_time_ms > 0
    stats["decay_s"] = (stats["decay_frames"] * frame_time_ms / 1000.0 if known
                        else float("nan"))
    return FrameCounts(label=label, colour=colour, counts=counts, edges=edges,
                       stats=stats, merged=merged)


def frame_summary(c: FrameCounts) -> str:
    """One line: the decay time, or that there is none, and the median."""
    s = c.stats
    what = "blinks" if c.merged else "localizations"
    t = s.get("decay_frames", np.nan)
    if np.isinf(t):
        decay = "no decay"
    elif np.isfinite(t):
        seconds = s.get("decay_s", np.nan)
        decay = f"decay time {t:.0f} frames" + (f" ({seconds:.1f} s)"
                                                 if np.isfinite(seconds) else "")
    else:
        decay = "too few to fit"
    return (f"{decay}; half of the {what} by frame {s['median_frame']:.0f}; "
            f"{s['first_rate']:.2f} per frame at the start, "
            f"{s['last_rate']:.2f} at the end")


def draw_per_frame(ax, found: Sequence[FrameCounts]) -> None:
    """The count per frame of every layer, its decay dashed over it."""
    for c in found:
        centres = 0.5 * (c.edges[:-1] + c.edges[1:])
        ax.plot(centres, c.rate, color=c.colour, linewidth=1.0,
                label=c.label or None)
        curve = c.curve()
        if curve is not None:
            ax.plot(*curve, color=c.colour, linestyle="--", linewidth=1.2)
    merged = all(c.merged for c in found)
    ax.set_xlabel("frame")
    ax.set_ylabel(("blinks" if merged else "localizations") + " per frame")
    ax.set_ylim(bottom=0)
    if any(c.label for c in found) and len(found) > 1:
        ax.legend(fontsize=7, frameon=False)
    if len(found) == 1:
        ax.set_title(frame_summary(found[0]), fontsize=TITLE_SIZE, color="0.25")


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
    ax.set_title(dist.summary, fontsize=TITLE_SIZE, color="0.25")
    if dist.note:
        ax.text(0.98, 0.9, dist.note, transform=ax.transAxes, ha="right",
                fontsize=7, color="0.45")


def draw_layers(ax, entries: Sequence[Tuple[str, str, Distribution]]) -> None:
    """One quantity of several layers, ``(name, colour, distribution)`` each:
    a step line per layer, faint, and its fit over it in the same colour,
    solid and heavier -- dashed, it vanished into its own steps -- with the
    fit's number in the legend.  Bars would hide each other; the landmarks,
    four lines a layer, are in the text."""
    first = entries[0][2]
    for name, colour, dist in entries:
        ax.stairs(dist.counts, dist.edges, color=colour, linewidth=0.8, alpha=0.55,
                  label=f"{name}: {dist.short}" if dist.short else name)
        if dist.curve is not None:
            ax.plot(*dist.curve, color=colour, linewidth=1.8)
    if any(d.log_y for _, _, d in entries):
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel(f"{first.label} ({first.unit})" if first.unit else first.label)
    ax.set_ylabel("localizations" if first.key != "on_time" else "blinks")
    ax.set_title(first.label, fontsize=TITLE_SIZE, color="0.25")
    ax.legend(fontsize=7, frameon=False, loc="upper right")


def draw_all_layers(figure, layers: Sequence[Tuple[str, str, Sequence[Distribution]]]
                    ) -> None:
    """Every quantity any layer has, a panel each, ``(name, colour,
    distributions)`` per layer (`draw_all`'s rules on the figure)."""
    keys: List[str] = []
    for _, _, found in layers:
        keys += [d.key for d in found if d.key not in keys]
    axes = figure.subplots(len(keys), 1, squeeze=False).ravel()
    for axis, key in zip(axes, keys):
        draw_layers(axis, [(name, colour, d) for name, colour, found in layers
                           for d in found if d.key == key])


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
                               choices=(("none", "none: median and landmarks only"),
                                        ("histogram", "least squares on the histogram"),
                                        ("mle", "maximum likelihood (outlier-sensitive)")),
                               help="the model of the precision and its sigma_c; "
                                    "the likelihood is thrown by a few failed "
                                    "fits with a precision near zero")
    photon_start: float = param(0.0, label="photons from", unit="photons", min=0.0,
                                help="where the exponential fit starts; "
                                     "0: at the maximum of the histogram, below "
                                     "which the detection threshold has eaten the "
                                     "dim localizations")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001, advanced=True,
                                           help="for the on-time in seconds; "
                                                "auto: the file's, else its "
                                                "exposure")


def blink_on_times(locs: Localizations) -> np.ndarray:
    """``n_in_group`` once per blink, whichever table it comes from.

    The grouped table has a row per blink already; the ungrouped one repeats
    the value on every frame of it, and ``group_id`` says which rows are one
    blink.  A table without ``group_id`` is taken to be one row per blink.
    """
    on_time = np.asarray(locs["n_in_group"])
    if "group_id" not in locs:
        return on_time
    gid = np.asarray(locs["group_id"])
    _, first = np.unique(gid, return_index=True)
    first = first[gid[first] > 0]       # 0: in no blink (`group.USE`)
    return on_time[first]


def grouped_on_time(session, layer: int = 0, selected: bool = True
                    ) -> Tuple[Optional[np.ndarray], str]:
    """``n_in_group`` from the session's grouped table, and why not if not.

    The grouped table is a table of its own, so the selection is rebuilt here
    rather than carried over: the layer's bounds and the same ROI, applied to
    blinks instead of to frames.  Nothing is linked -- a
    session that has never been switched to grouped has no on-time to show, and
    saying so is better than spending a minute of someone's time on it.
    """
    if session is None:
        return None, "no session: no merged blinks to take the on-time from"
    layers = getattr(session, "layers", None)
    if not layers:
        return None, "no layer to take the on-time from"
    index = layer if 0 <= layer < len(layers) else 0
    if getattr(layers[index], "is_image", False):
        index = session.first_locs_layer()
    state = getattr(layers[index], "state", None)
    tables = getattr(session, "tables", None)
    locset = tables.peek_grouped() if tables is not None else None
    if state is None or locset is None:
        return None, ("no on-time: tick merge blinks on the layer once, so the "
                      "localizations are merged into blinks")
    if not tables.is_current:
        return None, ("no on-time: the merged blinks are out of date; tick merge "
                      "blinks on the layer to merge them again")
    locs = locset.locs
    if "n_in_group" not in locs or not len(locs):
        return None, "no on-time: the merged blinks have no n_in_group"
    if not selected:
        return np.asarray(locs["n_in_group"]), ""
    from ..render import positions
    mask = np.asarray(state.filter_for(locset).mask, dtype=bool)
    roi = getattr(session, "roi", None)
    in_slab = bool(getattr(session, "selects_slab", False))
    if roi is not None or in_slab:
        x, y = positions(locs)
        if roi is not None:
            mask = mask & roi.mask(x, y)
        if in_slab:
            z = locs["z_nm"] if "z_nm" in locs else None
            mask = mask & session.slab.mask(x, y, z)
    return np.asarray(locs["n_in_group"])[mask], ""


@dataclass
class _Group:
    """One table the plugin describes: a layer's, or the only one."""
    locs: Localizations
    where: str
    grouped: bool
    on_time: Optional[np.ndarray] = None
    why: str = ""                       # why there is no on-time
    current: bool = False               # the layer the run was started on
    name: str = ""
    colour: str = "0.3"
    found: List[Distribution] = field(default_factory=list)


@register("Analysis/Measure/Localization Statistics")
class LocalizationStatistics(Plugin):
    """Photons, localization precision and on-time, each with its law fitted."""

    Settings = StatisticsSettings
    changes_locs = False
    per_region = True
    # 2: the table the layer shows rather than always the ungrouped one, and
    # the on-time counted once per blink rather than once per frame
    # 3: finds a SMAPpy 3D fit's z precision, z_err_nm
    # 4: *precision fit* none / histogram / MLE, one choice; a region fits the
    # model as the view does, where it used to start without it
    # 5: the on-time in seconds with the file's frame time, unasked, where it
    # took an exposure typed in; and on_time_s in a region's row
    # 6: every visible layer described, as steps on shared bins with their
    # own fits, where it was the current layer's alone
    version = "6"

    def run(self, ctx: Context, settings: StatisticsSettings) -> Result:
        groups = self._groups(ctx, settings)
        frame_time = settings.frame_time_ms or table_frame_time(groups[0].locs)
        # layers on one axis share their bins; one layer keeps its own
        highs = (shared_highs([(g.locs, g.on_time) for g in groups])
                 if len(groups) > 1 else None)
        for g in groups:
            g.found = statistics(g.locs, bins=settings.bins,
                                 photon_start=settings.photon_start, on_time=g.on_time,
                                 frame_time_ms=frame_time,
                                 precision_fit=settings.precision_fit, highs=highs)
        main = next((g for g in groups if g.current), groups[0])
        found = main.found
        if not any(g.found for g in groups):
            raise ValueError("the table has no photons, precision or on-time: "
                             f"it has {', '.join(sorted(main.locs.keys()))}")
        lines = []
        for g in groups:
            # the selection names its layer: "(layer 2, blinks merged)"
            lines.append(f"{len(g.locs)} localizations, {g.where}")
            lines += [f"{'  ' if len(groups) > 1 else ''}{d.label}: {d.summary}"
                      for d in g.found]
        notes = list(dict.fromkeys(g.why for g in groups if g.why))
        for why in notes:
            ctx.report(why)
        # per frame is over the acquisition, which a region is not a part of
        # in any useful way: the view only
        per_frame = ([] if ctx.per_region or "frame" not in main.locs
                     else self._per_frame(ctx, settings, main.locs, main.grouped))
        for c in per_frame:
            lines.append(f"per frame{', ' + c.label if c.label else ''}: "
                         f"{frame_summary(c)}")
        lines += notes
        data = {"distributions": {d.key: d for d in found},
                "stats": {d.key: d.stats for d in found},
                "layers": {g.name: {d.key: d for d in g.found} for g in groups},
                "per_frame": per_frame,
                "n": len(main.locs), "note": "; ".join(notes)}
        keys = {d.key for g in groups for d in g.found}

        def plot(figure) -> None:
            """One panel per distribution, in whatever it is given to draw in.

            A single axis is right for a plugin with one thing to draw and
            there are up to four here, so this declares its panels and is
            handed the figure -- which on the window's *All* page is one
            subfigure of it, and works the same.
            """
            if len(groups) > 1:
                draw_all_layers(figure, [(g.name, g.colour, g.found) for g in groups])
            else:
                draw_all(figure, found)

        # no *All* page: the main figure is already every distribution, one
        # under the other, and the per-frame panel squeezed beside it read
        # worse than either tab on its own
        return Result(text="\n".join(lines), data=data, settings=settings,
                      values=statistics_values(main.locs, found),
                      plot=Plot(draw=plot, panels=len(keys), summary=False,
                                size=(5.5, max(2.1 * len(keys), 2.1))),
                      plots=({"per frame": Plot(lambda ax: draw_per_frame(ax, per_frame),
                                                size=(5.5, 3.0), summary=False)}
                             if per_frame else {}))

    @staticmethod
    def _groups(ctx: Context, settings: StatisticsSettings) -> List["_Group"]:
        """What is described: every visible layer as it shows it, when the
        localizations are *as plotted* and there is more than one -- the
        comparison is usually between them -- else the current layer's table.
        A region describes its own rows, and *all, unfiltered* the whole
        table, once."""
        from .line_profile import layer_colour
        layers = getattr(ctx.session, "layers", None) or []
        visible = [i for i, layer in enumerate(layers)
                   if not layer.is_image and layer.visible]
        if (ctx.per_region or settings.source == "all" or len(visible) < 2):
            return [LocalizationStatistics._group(ctx, settings, None)]
        groups: List[_Group] = []
        for i in visible:
            taken = [g.colour for g in groups]
            g = LocalizationStatistics._group(ctx, settings, i)
            g.name = layers[i].name
            g.colour = layer_colour(layers[i].get_display(), len(groups), taken)
            g.current = i == ctx.layer
            groups.append(g)
        return groups

    @staticmethod
    def _group(ctx: Context, settings: StatisticsSettings,
               layer: Optional[int]) -> "_Group":
        """One table to describe: layer ``layer``'s, or with None the one
        the context is about, with its on-time."""
        table, selection = ctx.table() if layer is None else ctx.table(layer)
        grouped = table is not ctx.ungrouped()[0]
        if settings.source == "all":
            locs = table
            where = "all localizations" + (" (blinks merged)" if grouped else "")
        else:
            if not ctx.per_region:
                # in a region an empty one is an answer -- no localizations
                # here -- and a row that says so, not a failure
                selection.require(MIN_FOR_FIT, ctx.warn, "a distribution")
            locs = selection.apply(table)
            where = str(selection)          # names the grouping itself
        if not len(locs) and not ctx.per_region:
            raise ValueError("no localizations to describe")
        on_time, why = None, ""
        if "n_in_group" in locs:
            on_time = blink_on_times(locs)
        elif not ctx.per_region:
            on_time, why = grouped_on_time(ctx.session,
                                           ctx.layer if layer is None else layer,
                                           settings.source != "all")
        # in a region the session's grouped table is the whole field's, not
        # the region's: no on-time unless the region's own table has it
        return _Group(locs=locs, where=where, grouped=grouped, on_time=on_time,
                      why=why, current=True)

    @staticmethod
    def _per_frame(ctx: Context, settings: StatisticsSettings, locs: Localizations,
                   grouped: bool) -> List[FrameCounts]:
        """Every visible layer's count per frame, as it shows it: the
        comparison is usually between them.  Without a session, or with
        *all, unfiltered*, the table described."""
        from .time_course import _layers
        ft = settings.frame_time_ms or table_frame_time(locs)
        found = []
        if ctx.session is not None and settings.source != "all":
            found = [frame_counts(table["frame"], name, colour, frame_time_ms=ft,
                                  merged=merged)
                     for name, colour, table, merged in _layers(ctx)
                     if "frame" in table and len(table)]
        return found or [frame_counts(locs["frame"], frame_time_ms=ft, merged=grouped)]

    def keep(self, result: Result):
        """The histograms and their fits, binned: what the figure is drawn
        from, a few kB whatever the number of localizations -- what a region
        keeps so its figure comes back without measuring again
        (`docs/regions.md`, "Figures")."""
        found = (result.data or {}).get("distributions") or {}
        if not found:
            return None
        return {"distributions": [distribution_data(d) for d in found.values()]}

    def restore(self, saved):
        try:
            found = [distribution_from(d) for d in saved["distributions"]]
        except (KeyError, TypeError, ValueError):
            return None
        if not found:
            return None
        return Result(plot=Plot(draw=lambda figure: draw_all(figure, found),
                                panels=len(found),
                                size=(5.5, max(2.1 * len(found), 2.1))))


def distribution_data(d: Distribution) -> Dict:
    """A distribution as plain data (`LocalizationStatistics.keep`): equal
    bins as their range and number rather than every edge, which is most of
    what a region would otherwise keep."""
    edges = np.asarray(d.edges, dtype=float)
    steps = np.diff(edges)
    equal = len(edges) > 1 and np.allclose(steps, steps[0], rtol=1e-9, atol=0)
    return {"key": d.key, "label": d.label, "unit": d.unit,
            "counts": np.asarray(d.counts).astype(int).tolist(),
            **({"bins": [float(edges[0]), float(edges[-1]), len(edges) - 1]} if equal
               else {"edges": edges.tolist()}),
            "stats": {k: (v.item() if isinstance(v, np.generic) else v)
                      for k, v in d.stats.items()},
            "log_y": d.log_y, "summary": d.summary, "note": d.note,
            "short": d.short}


def distribution_from(doc: Dict) -> Distribution:
    """A distribution from `distribution_data`, its curve drawn again from
    the fit's numbers (`model_curve`) rather than kept."""
    if "bins" in doc:
        lo, hi, n = doc["bins"]
        edges = np.linspace(lo, hi, int(n) + 1)
    else:
        edges = np.asarray(doc["edges"], dtype=float)
    stats = dict(doc.get("stats") or {})
    return Distribution(key=doc["key"], label=doc["label"], unit=doc.get("unit", ""),
                        counts=np.asarray(doc["counts"]), edges=edges, stats=stats,
                        curve=model_curve(doc["key"], stats, edges),
                        log_y=bool(doc.get("log_y")), summary=doc.get("summary", ""),
                        note=doc.get("note", ""), short=doc.get("short", ""))
