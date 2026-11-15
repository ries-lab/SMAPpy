"""The quality of a MINFLUX sequence, from every iteration of every attempt.

What abberior's ParaFlux shows (its wiki page, "Iteration events" and
"Burst/Track statistics"), computed from a table read with *every
iteration* (`io.minflux.table(every=True)`): how many of the attempts the
scouting iteration started pass each iteration and why the rest end, the
background each iteration measured, how long each iteration takes, where
the catching iterations put the molecule against where it ends up, and per
trace how many localizations and photons it gave and why it stopped.

Plain functions of arrays, as the rest of smappy: the plugin
(`plugins/sequence_qc.py`) reads the columns and draws, a script or a test
calls these.

**The iterations.**  A localization attempt starts in the scouting
iteration (0), passes the catching iterations, each with a smaller pattern,
and reaches the final iteration, which repeats for as long as the molecule
stays on: every repetition a localization.  So the iterations before the
final one are counted per attempt -- the fraction of the scouting events
that reach and pass each, as ParaFlux's bars -- and the final iteration per
row.  The final iteration is the highest that holds a localization.

**Why an iteration ended** is ``sta``, coded differently in the two record
layouts (abberior's wiki, "MINFLUX Data Parameters"): m2410 1 valid final, 2
valid, 4 derived (aggregation), 6 failed the cfr check, 8 no signal (dark),
9 DAC out of range, 11 background measurement; m2205 0 valid, 4089 cfr,
4096 no signal, 4100 out of range.  `OUTCOMES` maps both onto passed, cfr,
dark, out of range and other.

**A trace** (``tid``, per thread ``thi``) is a burst: the attempt and the
localizations that followed while the molecule was on.  It ended for the
reason its last row gives, a background measurement (11) aside.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

#: what each ``sta`` means, both layouts (abberior's wiki)
OUTCOMES: Dict[int, str] = {
    1: "passed", 2: "passed", 4: "other", 5: "other", 6: "cfr", 8: "dark",
    9: "out of range", 11: "other",
    0: "passed", 4089: "cfr", 4096: "dark", 4100: "out of range",
}
#: the outcomes in the order the bars stack them
CATEGORIES = ("passed", "cfr", "dark", "out of range", "other")
#: the code of a background measurement: never why a trace ended
BACKGROUND = 11


def outcome(sta, vld=None) -> np.ndarray:
    """Each row's outcome, one of `CATEGORIES`, from ``sta`` (and ``vld``
    where the code is unknown: valid is passed, else other)."""
    sta = np.asarray(sta, np.int64)
    out = np.full(len(sta), "other", dtype=object)
    for code, name in OUTCOMES.items():
        out[sta == code] = name
    if vld is not None:
        unknown = ~np.isin(sta, list(OUTCOMES))
        out[unknown & np.asarray(vld, bool)] = "passed"
    return out


def final_iteration(iteration, final) -> Optional[int]:
    """The iteration that gives the localizations: the highest one with a
    ``final`` row; None if there is none."""
    iteration = np.asarray(iteration)
    final = np.asarray(final, bool)
    return int(iteration[final].max()) if final.any() else None


@dataclass
class IterationEvents:
    """Per iteration (``iterations``, ascending): how many rows reached it
    (``reached``) and how they ended (``counts[category]``), and, for the
    iterations before the final one, each as a fraction of the scouting
    events (``fraction[category]``, NaN for the final iteration and after,
    which repeat and are counted per row instead)."""
    iterations: np.ndarray
    reached: np.ndarray
    counts: Dict[str, np.ndarray]
    fraction: Dict[str, np.ndarray]
    final: Optional[int]

    def passed_fraction(self) -> Dict[int, float]:
        """Of the scouting events, the fraction that passed each iteration
        before the final one."""
        return {int(k): float(f) for k, f in zip(self.iterations, self.fraction["passed"])
                if np.isfinite(f)}


def iteration_events(iteration, sta, final, vld=None) -> IterationEvents:
    """How the attempts fared, iteration by iteration (see `IterationEvents`)."""
    iteration = np.asarray(iteration, np.int64)
    kinds = outcome(sta, vld)
    last = final_iteration(iteration, final)
    its = np.unique(iteration)
    reached = np.array([(iteration == k).sum() for k in its])
    counts = {c: np.array([((iteration == k) & (kinds == c)).sum() for k in its])
              for c in CATEGORIES}
    events = float(reached[0]) if len(its) and its[0] == 0 else float(reached.max(initial=0))
    catching = its < last if last is not None else np.ones(len(its), bool)
    fraction = {c: np.where(catching, counts[c] / max(events, 1.0), np.nan)
                for c in CATEGORIES}
    return IterationEvents(its, reached, counts, fraction, last)


def per_iteration(iteration, values, rows=None) -> Dict[str, np.ndarray]:
    """``values``' finite entries per iteration: ``iterations``, ``median``,
    ``mean``, ``std``, ``q25``, ``q75``, ``n``; ``rows`` a mask of the rows
    taken."""
    iteration = np.asarray(iteration, np.int64)
    values = np.asarray(values, np.float64)
    rows = np.ones(len(values), bool) if rows is None else np.asarray(rows, bool)
    its = np.unique(iteration[rows])
    out = {k: np.full(len(its), np.nan) for k in ("median", "mean", "std", "q25", "q75")}
    out["n"] = np.zeros(len(its), np.int64)
    for j, k in enumerate(its):
        v = values[rows & (iteration == k)]
        v = v[np.isfinite(v)]
        out["n"][j] = len(v)
        if len(v):
            out["median"][j], out["mean"][j] = np.median(v), v.mean()
            out["q25"][j], out["q75"][j] = np.quantile(v, [0.25, 0.75])
            out["std"][j] = v.std(ddof=1) if len(v) > 1 else np.nan
    out["iterations"] = its
    return out


def time_per_iteration(iteration, time_s) -> Dict[str, np.ndarray]:
    """How the acquisition's time divides among the iterations: each row
    lasts until the next row starts (rows sorted by time), its iteration
    takes that.  ``iterations``, ``seconds``, ``fraction``, and the
    scouting rows' own durations (``scouting_s``)."""
    iteration = np.asarray(iteration, np.int64)
    t = np.asarray(time_s, np.float64)
    order = np.argsort(t, kind="stable")
    dt = np.zeros(len(t))
    if len(t) > 1:
        dt[order[:-1]] = np.diff(t[order])
    its = np.unique(iteration)
    seconds = np.array([dt[iteration == k].sum() for k in its])
    total = seconds.sum()
    return {"iterations": its, "seconds": seconds,
            "fraction": seconds / total if total > 0 else np.full(len(its), np.nan),
            "scouting_s": dt[iteration == its[0]] if len(its) else np.zeros(0)}


def catching_offsets(iteration, final, trace, index, positions) -> Dict[str, np.ndarray]:
    """Where each catching iteration put the molecule against its
    localization: for every non-final row with a position, its offset from
    the final row it leads to -- the same ``trace`` and ``index``
    (`io.minflux.leading_index`).  Per iteration ``iterations``, ``mean``
    and ``std`` (iterations x axes, nm), ``n``."""
    iteration = np.asarray(iteration, np.int64)
    final = np.asarray(final, bool)
    p = np.atleast_2d(np.asarray(positions, np.float64))
    p = p.T if p.shape[0] != len(iteration) else p
    key = (np.asarray(trace, np.int64) << np.int64(24)) + np.asarray(index, np.int64)
    fin_key = key[final]
    order = np.argsort(fin_key, kind="stable")
    at = np.searchsorted(fin_key[order], key)
    found = (at < len(order))
    found[found] = fin_key[order][at[found]] == key[found]
    rows = ~final & found & np.isfinite(p).all(axis=1)
    target = np.flatnonzero(final)[order][np.minimum(at, max(len(order) - 1, 0))]
    offset = p - p[target] if len(order) else np.full(p.shape, np.nan)
    its = np.unique(iteration[rows])
    mean = np.array([offset[rows & (iteration == k)].mean(axis=0) for k in its]) \
        if len(its) else np.zeros((0, p.shape[1]))
    std = np.array([offset[rows & (iteration == k)].std(axis=0) for k in its]) \
        if len(its) else np.zeros((0, p.shape[1]))
    n = np.array([(rows & (iteration == k)).sum() for k in its], np.int64)
    return {"iterations": its, "mean": mean, "std": std, "n": n}


@dataclass
class Traces:
    """Per trace (a burst): its ``ids`` (``tid``), ``thread``, rows,
    localizations, photons over every row (``photons``: eco + ecc, the
    budget), first and last time, and why it ended (``sta``, its last row's
    other than a background measurement; ``reason`` the outcome)."""
    ids: np.ndarray
    thread: np.ndarray
    rows: np.ndarray
    localizations: np.ndarray
    photons: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    sta: np.ndarray
    reason: np.ndarray = field(default_factory=lambda: np.zeros(0, object))

    def valid(self) -> np.ndarray:
        """The traces that gave a localization: ParaFlux's valid bursts."""
        return self.localizations > 0


def traces(tid, sta, final, time_s=None, thi=None, eco=None, ecc=None) -> Traces:
    """Each trace's numbers (see `Traces`), one sort."""
    tid = np.asarray(tid, np.int64)
    n = len(tid)
    thi = np.zeros(n, np.int64) if thi is None else np.asarray(thi, np.int64)
    t = np.arange(n, dtype=np.float64) if time_s is None else np.asarray(time_s, np.float64)
    order = np.lexsort((t, tid, thi))
    k_tid, k_thi = tid[order], thi[order]
    start = np.r_[0, np.flatnonzero((k_tid[1:] != k_tid[:-1]) | (k_thi[1:] != k_thi[:-1])) + 1] \
        if n else np.zeros(0, np.int64)
    counts = np.diff(np.r_[start, n])
    fin = np.asarray(final, np.int64)[order]
    photons = np.zeros(n)
    for v in (eco, ecc):
        if v is not None:
            photons += np.nan_to_num(np.asarray(v, np.float64)[order])
    sums = lambda v: np.add.reduceat(v, start) if n else np.zeros(0)    # noqa: E731
    s = np.asarray(sta, np.int64)[order]
    # the last row that is not a background measurement says why it ended
    usable = s != BACKGROUND
    idx = np.where(usable, np.arange(n), -1)
    last = np.maximum.reduceat(idx, start) if n else np.zeros(0, np.int64)
    end_row = np.r_[start[1:], n] - 1 if n else np.zeros(0, np.int64)
    last = np.where(last >= start, last, end_row)
    reason_sta = s[last] if n else np.zeros(0, np.int64)
    out = Traces(k_tid[start] if n else np.zeros(0, np.int64),
                 k_thi[start] if n else np.zeros(0, np.int64),
                 counts, sums(fin), sums(photons),
                 t[order][start] if n else np.zeros(0), t[order][end_row] if n else np.zeros(0),
                 reason_sta)
    out.reason = outcome(reason_sta)
    return out


def exponential_mean(counts) -> float:
    """The mean of a geometric (discrete exponential) number of
    localizations per burst, from bursts of at least one: the maximum
    likelihood estimate, the sample mean."""
    counts = np.asarray(counts, np.float64)
    counts = counts[counts > 0]
    return float(counts.mean()) if len(counts) else float("nan")


def summary(events: IterationEvents, traces_: Traces, background: Dict[str, np.ndarray],
            times: Dict[str, np.ndarray]) -> Dict[str, float]:
    """The numbers a batch compares across measurements: the passed fraction
    per catching iteration, localizations per scouting event, the median
    background per iteration (kHz), localizations and photons per valid
    burst, why the valid bursts ended, the scouting's share of the time."""
    out: Dict[str, float] = {}
    for k, f in events.passed_fraction().items():
        out[f"passed_itr{k}"] = f
    events0 = float(events.reached[0]) if len(events.reached) else float("nan")
    out["scouting_events"] = events0
    out["localizations"] = float(traces_.localizations.sum())
    out["localizations_per_event"] = (out["localizations"] / events0
                                      if events0 else float("nan"))
    for k, m in zip(background["iterations"], background["median"]):
        out[f"fbg_itr{int(k)}_khz"] = float(m) / 1e3
    valid = traces_.valid()
    out["bursts"] = float(valid.sum())
    out["locs_per_burst"] = exponential_mean(traces_.localizations[valid])
    out["photons_per_burst"] = (float(np.median(traces_.photons[valid]))
                                if valid.any() else float("nan"))
    for c in CATEGORIES:
        out[f"ended_{c.replace(' ', '_')}"] = (float((traces_.reason[valid] == c).mean())
                                              if valid.any() else float("nan"))
    if len(times["iterations"]):
        out["scouting_time_fraction"] = float(times["fraction"][0])
    return out


def list_of(values) -> List[float]:
    return [float(v) for v in np.asarray(values, np.float64)]
