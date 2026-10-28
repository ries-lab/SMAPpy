"""Binding kinetics of one site: its time trace, its binding events, qPAINT.

A port of Picasso's pick kinetics (`postprocess.pick_kinetics`,
`dark_times`, `pick_properties`; `lib.fit_cum_exp`, `estimate_kinetic_rate`;
jungmannlab/picasso, MIT) and of its qPAINT count (Jungmann et al. 2016),
with the frame analysis of its clusterer (`clusterer._frame_analysis`) as
values to curate sites by.  Plain functions on frames, which a script, a
test and the plugin (`plugins/binding_kinetics.py`) call alike.

**A binding event** is every localization of the region joined across dark
gaps of at most *link gap* frames, whatever their positions -- Picasso links
a pick with ``r_max=999999, max_dark_time=3``.  That is `blinking.bursts`
on the region's blinks (`blinking.blinks_of`), so the region is one
structure and nothing is linked in space: `group.connect` with an infinite
box would give the same events, slower.  An event's **bright time** is
``last - first + 1`` frames, and the **dark time** before it is the frames
from the closest preceding event's end, ``first - last_before - 1``
(`blinking.dark_frames`) -- Picasso's convention since 0.11.3 and
*Blinking*'s.  A region of several files keeps each file's events apart.

**The mean dark time, by maximum likelihood** (the default), where Picasso
fits ``a (1 - exp(-t/tau)) + c`` to the dark times' cumulative distribution
by least squares (kept as the *cumulative fit* estimator, ported as it is).
For dark times exponential with mean tau the likelihood is maximised by the
dark frames observed over the number of dark periods that ended, with two
corrections the fit leaves to its free offset ``c`` or ignores:

* *Linking truncates*: a gap of *link gap* T frames or fewer is not a dark
  time, so each one observed is longer than T and only its excess over
  ``T + 1/2`` (a frame is bright from half covered) is exponential.
* *The acquisition censors*: the time from the start to the first event is
  a complete dark time (memorylessness), the time from the last event to
  the end an incomplete one.  Leaving both out, as a mean of the dark times
  between events does, underestimates tau by a factor ``n / (n + 1)`` for
  n events -- a bias that grows as the sites get fewer, so a count
  calibrated on large sites would come out too high on small ones.

So ``tau = (start + sum(d_i - T - 1/2) + end) / (n_between + 1)``, the
start and end counted only where the acquisition's frames are known (the
table's ``n_frames``, a setting, or the file's first and last frame).  The
**qPAINT index** is the rate of bindings, ``(n - 1) / total`` for n dark
periods -- not ``1 / tau``, which overestimates a rate by ``n / (n - 1)``,
11% at ten events -- with a relative error of ``1 / sqrt(n)``.  It is
proportional to the number of sites, and
**sites** = index / influx with the influx (one site's binding rate per
frame) calibrated on regions of known site number.  Pooled over regions
the estimate is the sum of their dark frames over the sum of their dark
periods (`pooled_index`), which weighs each region by its events as
Picasso's pooled fit does.

**The bright time**, under maximum likelihood, is the mean of the events'
bright times: they are not truncated by linking, and an event cut by the
start or the end of the acquisition is rare when events are short.  The
cumulative fit fits it as it fits the dark time, as Picasso does.

**Curation values** (Picasso's frame analysis, `clusterer._frame_analysis`):
the mean frame of the localizations as a fraction of the acquisition (a
site rejected below 0.2 or above 0.8), the largest fraction of them in one
twentieth of it (rejected above 0.8), the spread of the frames (0.29 for
uniform; G5M's ``std_frame``), the longest dark time and the longest event
(a sticky imager is one event hundreds of frames long), and a Kolmogorov-
Smirnov test of the event starts against a constant rate (``uniform_p``).
A sticky or partial site is a step in the cumulative count; these turn
that step into numbers *Filter ROIs* can use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .plugins.blinking import Blinks, blinks_of, bursts, dark_frames

#: Picasso's link gap: frames without a localization inside one event
LINK_GAP = 3
#: the trace's windows over the acquisition
WINDOWS = 50
#: Picasso's frame analysis: twentieths, and the bounds it rejects at
FRAME_BINS = 20
ESTIMATORS = (("ml", "maximum likelihood"), ("cdf", "cumulative fit (Picasso)"))


@dataclass
class Events:
    """One row per binding event, sorted by file, then by first frame."""
    file: np.ndarray        # which file (a region of one file: all the same)
    first: np.ndarray       # first frame
    last: np.ndarray        # last frame
    n: np.ndarray           # localizations in it

    def __len__(self) -> int:
        return len(self.first)

    @property
    def bright(self) -> np.ndarray:
        """Bright time of each event in frames, first to last, both counted."""
        return self.last - self.first + 1

    def as_blinks(self) -> Blinks:
        return Blinks(self.file, self.first, self.last, self.n)


def binding_events(frame, blink=None, file=None, link_gap: int = LINK_GAP) -> Events:
    """The binding events of one region: its blinks (``blink`` the
    ``group_id`` of each localization, else runs of consecutive frames)
    joined across dark gaps of at most ``link_gap`` frames, per file."""
    found = blinks_of(frame, blink, file)
    if not len(found):
        empty = np.zeros(0, np.int64)
        return Events(empty, empty, empty, empty)
    burst = bursts(found, link_gap)
    starts = np.flatnonzero(np.r_[True, burst[1:] != burst[:-1]])
    first = found.first[starts]
    last = np.maximum.reduceat(found.last, starts)
    n = np.add.reduceat(found.n, starts)
    return Events(found.molecule[starts], first, last, n)


def dark_times(events: Events) -> np.ndarray:
    """The dark time before each event, in frames; NaN for a file's first."""
    return dark_frames(events.as_blinks())


def fit_cumulative(values) -> float:
    """Picasso's estimate of a mean time (`lib.estimate_kinetic_rate`): a
    least-squares fit of ``a (1 - exp(-x/t)) + c`` to the cumulative count
    of the sorted values, ``t`` held between their least and largest; the
    mean for two values or fewer, or for values all alike."""
    from scipy import optimize
    data = np.sort(np.asarray(values, float))
    data = data[np.isfinite(data)]
    n = len(data)
    if n == 0:
        return float("nan")
    if n <= 2 or data[-1] == data[0]:
        return float(np.mean(data))
    y = np.arange(1, n + 1)

    def cumulative(x, a, t, c):
        return a * (1 - np.exp(-x / t)) + c

    try:
        popt, _ = optimize.curve_fit(cumulative, data, y,
                                     p0=[n, np.mean(data), data[0]],
                                     bounds=([0, data[0], 0], [np.inf, data[-1], np.inf]))
    except (RuntimeError, ValueError):
        return float(np.mean(data))
    return float(popt[1])


def dark_ml(gaps, link_gap: int, ends: Optional[Tuple[float, float]] = None,
            n_files: int = 1) -> Tuple[float, int]:
    """``(total, periods)``: the dark frames the likelihood counts and the
    dark periods that ended, so that tau = total / periods (module
    docstring).  ``gaps`` are the dark times between events; ``ends`` the
    dark frames before the first event and after the last, summed over the
    files, None where the acquisition's frames are not known."""
    gaps = np.asarray(gaps, float)
    gaps = gaps[np.isfinite(gaps)]
    total = float(np.sum(gaps - (link_gap + 0.5)))
    periods = len(gaps)
    if ends is not None:
        total += float(ends[0]) + float(ends[1])
        periods += int(n_files)
    return total, periods


def pooled_index(totals, periods) -> Tuple[float, float]:
    """The qPAINT index of regions taken together, from each one's dark
    frames and dark periods (`dark_ml`), with its standard error."""
    t = float(np.nansum(np.asarray(totals, float)))
    p = float(np.nansum(np.asarray(periods, float)))
    if t <= 0 or p <= 1:
        return float("nan"), float("nan")
    return (p - 1) / t, (p - 1) / t / np.sqrt(p)


def window_edges(span: Tuple[int, int], windows: int = WINDOWS) -> np.ndarray:
    """Edges of about ``windows`` windows of whole frames over the frames
    ``span`` (first and last, both included): `time_course.frame_windows`,
    so that a constant rate does not wobble from window to window."""
    from .plugins.time_course import frame_windows
    return frame_windows(span[0], span[1], windows)[0]


def trace(events: Events, frame, edges: np.ndarray) -> Dict[str, np.ndarray]:
    """The time trace on ``edges``: events starting and localizations in
    each window, and the fraction of the events begun by each window's end
    (a site binding at a constant rate climbs the diagonal)."""
    frame = np.asarray(frame, float)
    started = np.histogram(events.first, edges)[0].astype(float)
    locs = np.histogram(frame, edges)[0].astype(float)
    total = started.sum()
    cumulative = np.cumsum(started) / total if total else np.full(len(started), np.nan)
    return {"frame": 0.5 * (edges[1:] + edges[:-1] - 1), "events": started,
            "localizations": locs, "cumulative": cumulative}


def frame_spread(frame, file, spans: Dict[int, Tuple[int, int]]) -> Dict[str, float]:
    """Picasso's frame analysis of one region: where in the acquisition its
    localizations fall, each as a fraction of its own file's frames."""
    frame = np.asarray(frame, float)
    file = np.asarray(file, np.int64)
    rel = np.full(len(frame), np.nan)
    for f, (lo, hi) in spans.items():
        here = file == f
        rel[here] = (frame[here] - lo + 0.5) / (hi - lo + 1)
    rel = rel[np.isfinite(rel)]
    if not len(rel):
        return {"mean_frame_rel": float("nan"), "std_frame_rel": float("nan"),
                "peak_window_fraction": float("nan")}
    counts = np.histogram(np.clip(rel, 0, 1), np.linspace(0, 1, FRAME_BINS + 1))[0]
    return {"mean_frame_rel": float(rel.mean()), "std_frame_rel": float(rel.std()),
            "peak_window_fraction": float(counts.max() / len(rel))}


def uniform_p(events: Events, spans: Dict[int, Tuple[int, int]]) -> float:
    """The Kolmogorov-Smirnov p-value of the events' starts against a
    constant rate over the acquisition: small for a site whose events bunch
    (a sticky period, a site lost or gained half way)."""
    from scipy import stats
    if len(events) < 2:
        return float("nan")
    rel = []
    for f, (lo, hi) in spans.items():
        here = events.file == f
        rel.append((events.first[here] - lo + 0.5) / (hi - lo + 1))
    rel = np.concatenate(rel) if rel else np.zeros(0)
    if len(rel) < 2:
        return float("nan")
    return float(stats.kstest(np.clip(rel, 0, 1), "uniform").pvalue)


def region_kinetics(frame, blink=None, file=None, link_gap: int = LINK_GAP,
                    spans: Optional[Dict[int, Tuple[int, int]]] = None,
                    estimator: str = "ml", influx: Optional[float] = None,
                    frame_time_ms: Optional[float] = None, windows: int = WINDOWS
                    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Events]:
    """Everything measured in one region: ``(values, arrays, events)``.

    ``spans`` is each file's first and last frame of the acquisition; where
    it is None the region's own first and last frame stand in, and the dark
    time before the first event and after the last is not counted.
    ``influx`` (bindings per frame of one site) turns the qPAINT index into
    sites; ``frame_time_ms`` the frames into seconds.
    """
    frame = np.asarray(frame, np.int64)
    file = (np.zeros(len(frame), np.int64) if file is None
            else np.asarray(file, np.int64))
    events = binding_events(frame, blink, file, link_gap)
    known = spans is not None
    if spans is None:
        spans = {int(f): (int(frame[file == f].min()), int(frame[file == f].max()))
                 for f in np.unique(file)}
    gaps = dark_times(events)
    bright = events.bright.astype(float)
    between = gaps[np.isfinite(gaps)]
    nan = float("nan")

    ends = None
    if known and len(events):
        before = after = 0.0
        for f in np.unique(events.file):
            here = events.file == f
            lo, hi = spans.get(int(f), (int(events.first[here].min()),
                                        int(events.last[here].max())))
            before += max(int(events.first[here].min()) - lo, 0)
            after += max(hi - int(events.last[here].max()), 0)
        ends = (before, after)
    total, periods = dark_ml(between, link_gap, ends, len(np.unique(events.file)))
    if estimator == "cdf":
        tau_dark = fit_cumulative(between) if len(between) else nan
        tau_bright = fit_cumulative(bright) if len(bright) else nan
    elif estimator == "ml":
        tau_dark = total / periods if periods and total > 0 else nan
        tau_bright = float(bright.mean()) if len(bright) else nan
    else:
        raise ValueError(f"no estimator {estimator!r}: "
                         + ", ".join(e for e, _ in ESTIMATORS))
    if estimator == "ml":
        # the unbiased rate: periods / total overestimates it by n / (n - 1)
        index = (periods - 1) / total if periods > 1 and total > 0 else nan
    else:
        index = (1.0 / tau_dark if tau_dark and np.isfinite(tau_dark) and tau_dark > 0
                 else nan)
    index_err = index / np.sqrt(periods) if periods else nan
    sites = index / influx if influx else nan
    sites_err = index_err / influx if influx else nan

    seconds = (frame_time_ms / 1000.0 if frame_time_ms and frame_time_ms > 0 else nan)
    first = int(frame.min()) if len(frame) else 0
    last = int(frame.max()) if len(frame) else -1
    values = {
        "n_localizations": int(len(frame)),
        "n_events": int(len(events)),
        "locs_per_event": float(events.n.mean()) if len(events) else nan,
        "first_frame": first if len(frame) else nan,
        "last_frame": last if len(frame) else nan,
        "span_frames": last - first + 1 if len(frame) else nan,
        "longest_dark_frames": float(between.max()) if len(between) else nan,
        "longest_bright_frames": float(bright.max()) if len(bright) else nan,
        "mean_bright_frames": float(bright.mean()) if len(bright) else nan,
        "mean_dark_frames": float(between.mean()) if len(between) else nan,
        "tau_bright_frames": tau_bright,
        "tau_dark_frames": tau_dark,
        "tau_bright_s": tau_bright * seconds,
        "tau_dark_s": tau_dark * seconds,
        "qpaint_index": index,
        "qpaint_index_err": index_err,
        "n_sites": sites,
        "n_sites_err": sites_err,
        "dark_total_frames": total if periods else nan,
        "dark_periods": int(periods),
    }
    values.update(frame_spread(frame, file, spans))
    values["uniform_p"] = uniform_p(events, spans)
    arrays: Dict[str, np.ndarray] = {}
    if len(spans) == 1:
        # one file: the trace on its acquisition, the same grid for every
        # region of it, so *Curves* can average them
        arrays.update(trace(events, frame, window_edges(next(iter(spans.values())),
                                                        windows)))
    arrays["event_first_frame"] = events.first.astype(float)
    arrays["bright_frames"] = bright
    arrays["dark_frames"] = gaps
    return values, arrays, events


def spans_of(frame, file) -> Dict[int, Tuple[int, int]]:
    """Each file's first and last frame among ``frame``: the acquisition as
    its localizations show it."""
    frame = np.asarray(frame, np.int64)
    file = np.zeros(len(frame), np.int64) if file is None else np.asarray(file, np.int64)
    if not len(frame):
        return {}
    order = np.argsort(file, kind="stable")
    f, fr = file[order], frame[order]
    starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    lo = np.minimum.reduceat(fr, starts)
    hi = np.maximum.reduceat(fr, starts)
    return {int(f[s]): (int(a), int(b)) for s, a, b in zip(starts, lo, hi)}


def sites_from(index: Sequence[float], influx: float) -> np.ndarray:
    """Binding sites from qPAINT indices and one site's influx."""
    return np.asarray(index, float) / float(influx)
