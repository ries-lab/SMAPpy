"""How still the sample is: the vibrations of a bead's position over time.

The algorithms of *Stability* (`plugins/stability.py`), on arrays.  SMAP's
``Analyze/measure/Vibrations`` is where it comes from: one bead in the ROI,
x, y and z against time, their SD beside the mean precision, and the
amplitude spectrum 2|FFT|/L with its highest peak.  The decisions are in
`docs/smap_porting/wave5_design.md`, step 5e.  Where this departs from it:

* **Gaps are filled, and counted.**  An FFT wants one sample per time step;
  SMAP took the bead's localizations as if they were, so a missed frame
  shifted every later sample by one step and smeared the spectrum.  Here
  each trace is put on a regular grid (`resample`): a missing frame is
  interpolated linearly between its neighbours, and how many were filled
  is reported, since a filled point carries no information about anything
  faster than the gap.
* **Irregular times** (MINFLUX's ``time_s``) are interpolated onto a grid
  whose step is the 20th percentile of the intervals, as SMAP's -- but not
  halved: SMAP's grid at half that step made the upper half of the
  spectrum interpolation, not data.
* **Welch, not one FFT.**  One FFT of the whole trace gives a spectrum
  whose noise is as large as its mean at every frequency, so a small peak
  is not told from a noisy bin; the average over segments (Hann windowed,
  half overlapping) is smoother by the square root of their number.  The
  amplitude is scaled so that a sinusoid of amplitude A reads A at its
  peak, as SMAP's 2|FFT|/L did.
* **A peak's amplitude is its power, not its tallest bin.**  A sinusoid
  between two bins of a Hann-windowed spectrum reads up to 15 % low at its
  tallest bin; the power summed over the main lobe (five bins) above the
  floor around it does not depend on where the frequency falls.
* **Slow drift is removed** before the SD (a zero-phase high-pass,
  `highpass`): over a long acquisition drift dominates the SD, and the SD
  is meant to say how much the bead jitters.  SD^2 - precision^2 is then
  the vibration's variance (`vibration_rms`).
* **The random scatter** (`scatter`, `pooled_scatter`) is the other
  question: not how much the bead jitters at which frequency, but how much
  of its scatter is noise that averages away -- the localization error, set
  against the precision the fit claims (the CRB).  It is the Allan
  deviation of the position: half the mean square of the difference of
  consecutive means over m points, which at m = 1 is RMS(diff(x))/sqrt(2),
  NeNA's idea on a trace whose points are known to be one bead.  Drift
  cancels in a difference to first order, so nothing is high-passed; white
  noise falls as 1/sqrt(m), drift and slow vibrations rise or flatten it.
  It takes the measured points only: a filled point would pull the
  difference to its neighbours' and read the noise low.
* **Several beads**: each, their mean, or **differential** -- one bead
  minus another, or each minus the mean of the others (`combine`) -- so
  motion common to the sample (the stage, drift) cancels and what is left
  is relative motion and localization noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AXES = ("x", "y", "z")
# a Hann window's main lobe is four bins wide; a peak's power is summed over
# the bins within this many of its tallest
PEAK_HALF_WIDTH = 2
# the floor under a peak: the median of the spectrum this many bins either side
FLOOR_HALF_WIDTH = 20


@dataclass
class Trace:
    """A position against time on a regular grid: ``values`` (n, axes) in nm,
    NaN where the trace has no data (before it starts, after it ends);
    ``filled`` the points interpolated over a gap; ``precision`` the RMS
    localization precision per axis (NaN where unknown)."""
    name: str
    t_s: np.ndarray
    values: np.ndarray
    filled: np.ndarray
    precision: np.ndarray

    @property
    def dt_s(self) -> float:
        return float(self.t_s[1] - self.t_s[0]) if len(self.t_s) > 1 else 1.0

    @property
    def n_filled(self) -> int:
        return int(self.filled.sum())

    def span(self) -> slice:
        """The points from the first with data to the last."""
        have = np.flatnonzero(np.isfinite(self.values).all(axis=1))
        return slice(int(have[0]), int(have[-1]) + 1) if len(have) else slice(0, 0)


def grid_step(t_s: np.ndarray) -> float:
    """The step of the regular grid irregular times are put on: the 20th
    percentile of the intervals between them (SMAP's, not halved), so most
    intervals hold one sample."""
    d = np.diff(np.unique(np.asarray(t_s, float)))
    d = d[d > 0]
    if not len(d):
        raise ValueError("fewer than two distinct times: no trace")
    return float(np.quantile(d, 0.2))


def resample(t_s, values, dt_s: float, start_s: Optional[float] = None,
             n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, values, filled)``: ``values`` (rows, axes) sampled at ``t_s`` put
    on the grid ``start_s + k dt_s`` (default from the first sample to the
    last).  Samples at one grid point are averaged; a grid point with none
    between two samples is interpolated linearly and flagged ``filled``;
    outside the samples it is NaN.  Frames are the regular case: a missed
    frame is one filled point."""
    t_s = np.asarray(t_s, float)
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    good = np.isfinite(t_s) & np.isfinite(values).all(axis=1)
    t_s, values = t_s[good], values[good]
    if not len(t_s):
        raise ValueError("no samples")
    if start_s is None:
        start_s = float(t_s.min())
    k = np.rint((t_s - start_s) / dt_s).astype(np.int64)
    if n is None:
        n = int(k.max()) + 1
    inside = (k >= 0) & (k < n)
    k, values = k[inside], values[inside]
    counts = np.bincount(k, minlength=n).astype(float)
    sums = np.stack([np.bincount(k, values[:, a], minlength=n)
                     for a in range(values.shape[1])], axis=1)
    seen = counts > 0
    out = np.full((n, values.shape[1]), np.nan)
    out[seen] = sums[seen] / counts[seen, None]
    where = np.flatnonzero(seen)
    filled = np.zeros(n, bool)
    if len(where):
        first, last = where[0], where[-1]
        gap = np.arange(first, last + 1)
        gap = gap[~seen[gap]]
        for a in range(out.shape[1]):
            out[gap, a] = np.interp(gap, where, out[where, a])
        filled[gap] = True
    return start_s + dt_s * np.arange(n), out, filled


def bead_traces(ids, t_s, xyz, dt_s: float, err=None,
                regular: bool = True) -> List[Trace]:
    """One trace per id above 0, all on one grid from the first sample of any
    to the last, centred on its own mean.  ``err`` (rows, axes) gives each
    trace's RMS precision.  ``regular``: the times are frames times a step
    and the grid is the frames; else the grid step is ``dt_s`` as given."""
    ids = np.asarray(ids)
    t_s = np.asarray(t_s, float)
    xyz = np.asarray(xyz, float)
    if xyz.ndim == 1:
        xyz = xyz[:, None]
    keep = ids > 0
    if not keep.any():
        return []
    start = float(t_s[keep].min())
    n = int(np.rint((t_s[keep].max() - start) / dt_s)) + 1
    traces = []
    for i in np.unique(ids[keep]):
        rows = np.flatnonzero(ids == i)
        t, v, filled = resample(t_s[rows], xyz[rows], dt_s, start, n)
        v = v - np.nanmean(v, axis=0)
        if err is None:
            p = np.full(xyz.shape[1], np.nan)
        else:
            e = np.asarray(err, float)[rows] ** 2
            have = np.isfinite(e).sum(0)
            # NaN on an axis whose precision the table does not give
            p = np.sqrt(np.nansum(e, 0) / np.where(have, have, 1))
            p[have == 0] = np.nan
        traces.append(Trace(str(int(i)), t, v, filled, np.atleast_1d(p)))
    return traces


def _mean(traces: Sequence[Trace]) -> np.ndarray:
    stack = np.stack([t.values for t in traces])
    with np.errstate(invalid="ignore"):
        count = np.isfinite(stack).sum(0)
        out = np.nansum(stack, 0) / np.where(count, count, 1)
    out[count == 0] = np.nan
    return out


def combine(traces: Sequence[Trace], how: str) -> List[Trace]:
    """The traces analysed: ``each`` bead as it is, the ``mean`` over beads
    (where at least one is seen), or differential -- ``pair``: each bead
    minus the first (two beads give their difference), ``others``: each
    minus the mean of the others.  The precisions are propagated (beads'
    errors independent): a difference of two is sqrt(2) times one's."""
    traces = list(traces)
    if how == "each" or not traces:
        return traces
    if how == "mean":
        n = len(traces)
        p = np.sqrt(np.sum([t.precision ** 2 for t in traces], 0)) / n
        filled = np.all([t.filled | ~np.isfinite(t.values).all(1) for t in traces], 0)
        v = _mean(traces)
        filled &= np.isfinite(v).all(1)
        return [Trace(f"mean of {n}", traces[0].t_s, v, filled, p)]
    if len(traces) < 2:
        raise ValueError(f"a differential trace needs two beads; there is {len(traces)}")
    out = []
    if how == "pair":
        ref = traces[0]
        for t in traces[1:]:
            v = t.values - ref.values
            out.append(Trace(f"{t.name} - {ref.name}", t.t_s, v - np.nanmean(v, 0),
                             (t.filled | ref.filled) & np.isfinite(v).all(1),
                             np.sqrt(t.precision ** 2 + ref.precision ** 2)))
        return out
    if how == "others":
        for i, t in enumerate(traces):
            rest = traces[:i] + traces[i + 1:]
            v = t.values - _mean(rest)
            p_rest = np.sqrt(np.sum([r.precision ** 2 for r in rest], 0)) / len(rest)
            out.append(Trace(f"{t.name} - others", t.t_s, v - np.nanmean(v, 0),
                             t.filled & np.isfinite(v).all(1),
                             np.sqrt(t.precision ** 2 + p_rest ** 2)))
        return out
    raise ValueError(f"unknown combination {how!r}")


def longest_run(values: np.ndarray) -> slice:
    """The longest stretch of finite points: a spectrum needs one."""
    ok = np.isfinite(values) if values.ndim == 1 else np.isfinite(values).all(1)
    if not ok.any():
        return slice(0, 0)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    starts, stops = edges[::2], edges[1::2]
    i = int(np.argmax(stops - starts))
    return slice(int(starts[i]), int(stops[i]))


def highpass(x: np.ndarray, dt_s: float, cutoff_hz: float) -> np.ndarray:
    """``x`` with what is slower than ``cutoff_hz`` removed: a 2nd-order
    Butterworth run forwards and backwards (zero phase, 4th order in all),
    so a vibration is not shifted in time.  At 0 Hz, or at or above the
    Nyquist frequency, only the mean is removed."""
    x = np.asarray(x, float)
    nyquist = 0.5 / dt_s
    if cutoff_hz <= 0 or cutoff_hz >= nyquist or len(x) < 16:
        return x - np.mean(x)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, cutoff_hz / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def smooth(x: np.ndarray, points: int) -> np.ndarray:
    """A running mean over ``points`` (centred; shorter at the ends)."""
    x = np.asarray(x, float)
    if points <= 1 or len(x) < 2:
        return x.copy()
    k = np.ones(min(int(points), len(x)))
    return np.convolve(x, k, "same") / np.convolve(np.ones(len(x)), k, "same")


@dataclass
class Spectrum:
    """A one-sided spectrum: ``psd`` in nm^2/Hz (its sum times ``df`` is the
    variance), ``amplitude`` in nm (a sinusoid of amplitude A reads A at its
    peak), from ``segments`` segments of ``length`` points."""
    f_hz: np.ndarray
    psd: np.ndarray
    amplitude: np.ndarray
    segments: int
    length: int

    @property
    def df(self) -> float:
        return float(self.f_hz[1] - self.f_hz[0]) if len(self.f_hz) > 1 else 1.0


def spectrum(x: np.ndarray, dt_s: float, segment: int = 1024) -> Spectrum:
    """Welch's average over Hann-windowed segments of ``segment`` points (the
    whole trace when shorter), half overlapping, each segment's mean
    removed."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        raise ValueError(f"{n} points: too few for a spectrum")
    length = int(min(max(segment, 4), n))
    step = max(length // 2, 1)
    starts = np.arange(0, n - length + 1, step)
    w = np.hanning(length + 2)[1:-1]          # no zero ends: all points count
    fs = 1.0 / dt_s
    power = np.zeros(length // 2 + 1)
    for s in starts:
        seg = x[s:s + length]
        power += np.abs(np.fft.rfft((seg - seg.mean()) * w)) ** 2
    power /= len(starts)
    psd = power / (fs * np.sum(w ** 2))
    psd[1:] *= 2.0
    if length % 2 == 0:
        psd[-1] /= 2.0                        # the Nyquist bin has no twin
    psd[0] = 0.0
    amplitude = 2.0 * np.sqrt(power) / np.sum(w)
    amplitude[0] = 0.0
    return Spectrum(np.fft.rfftfreq(length, dt_s), psd, amplitude, len(starts), length)


def cumulative_rms(s: Spectrum) -> np.ndarray:
    """The RMS of everything up to each frequency: its square at the top is
    the variance of the trace the spectrum came from."""
    return np.sqrt(np.cumsum(s.psd) * s.df)


def _floor(psd: np.ndarray, i: int) -> float:
    lo, hi = max(i - FLOOR_HALF_WIDTH, 1), min(i + FLOOR_HALF_WIDTH + 1, len(psd))
    around = np.r_[psd[lo:max(i - PEAK_HALF_WIDTH, lo)],
                   psd[min(i + PEAK_HALF_WIDTH + 1, hi):hi]]
    return float(np.median(around)) if len(around) else 0.0


def peak_amplitude(s: Spectrum, i: int) -> float:
    """The amplitude of a sinusoid at bin ``i``: from the power over the
    main lobe above the floor around it, sqrt(2 P), which does not depend
    on where between two bins its frequency falls."""
    lo, hi = max(i - PEAK_HALF_WIDTH, 1), min(i + PEAK_HALF_WIDTH + 1, len(s.psd))
    power = float(np.sum(s.psd[lo:hi] - _floor(s.psd, i)) * s.df)
    return float(np.sqrt(2.0 * max(power, 0.0)))


def peak_frequency(s: Spectrum, i: int) -> float:
    """The peak's frequency, between bins: the power-weighted mean over its
    main lobe."""
    lo, hi = max(i - PEAK_HALF_WIDTH, 1), min(i + PEAK_HALF_WIDTH + 1, len(s.psd))
    w = np.clip(s.psd[lo:hi] - _floor(s.psd, i), 0.0, None)
    return float(np.sum(w * s.f_hz[lo:hi]) / np.sum(w)) if w.sum() > 0 else float(s.f_hz[i])


def peaks(s: Spectrum, n: int = 5, above_hz: float = 0.0,
          min_ratio: float = 4.0) -> List[Tuple[float, float]]:
    """``[(frequency_hz, amplitude_nm)]`` of the ``n`` largest peaks above
    ``above_hz``: local maxima of the spectrum at least ``min_ratio`` times
    the floor around them (in power), at least a main lobe apart, largest
    first."""
    psd = s.psd
    found = []
    for i in range(1, len(psd)):
        if s.f_hz[i] <= above_hz:
            continue
        lo, hi = max(i - PEAK_HALF_WIDTH, 1), min(i + PEAK_HALF_WIDTH + 1, len(psd))
        if psd[i] < psd[lo:hi].max() or psd[i] <= 0:
            continue
        if psd[i] < min_ratio * _floor(psd, i):
            continue
        found.append(i)
    found.sort(key=lambda i: -psd[i])
    out: List[Tuple[float, float]] = []
    taken: List[int] = []
    for i in found:
        if any(abs(i - j) <= 2 * PEAK_HALF_WIDTH for j in taken):
            continue
        taken.append(i)
        out.append((peak_frequency(s, i), peak_amplitude(s, i)))
        if len(out) >= n:
            break
    out.sort(key=lambda p: -p[1])
    return out


@dataclass
class AxisResult:
    """One axis of one trace: the high-passed trace's SD, the precision and
    the vibration they leave, the spectrum and its peaks."""
    axis: str
    sd_nm: float
    precision_nm: float
    vibration_nm: float
    spectrum: Spectrum
    peaks: List[Tuple[float, float]]
    detrended: np.ndarray


def vibration_rms(sd_nm: float, precision_nm: float) -> float:
    """sqrt(SD^2 - precision^2): what of the jitter the localization error
    does not explain; the SD where the precision is unknown, 0 when the
    precision explains all of it."""
    if not np.isfinite(precision_nm):
        return float(sd_nm)
    return float(np.sqrt(max(sd_nm ** 2 - precision_nm ** 2, 0.0)))


def analyse_axis(x: np.ndarray, dt_s: float, precision_nm: float = np.nan, *,
                 axis: str = "x", highpass_hz: float = 0.0, segment: int = 1024,
                 n_peaks: int = 5) -> AxisResult:
    """One axis of a gap-free trace: high-pass, SD, the vibration's RMS, the
    spectrum and its peaks above the high-pass frequency."""
    d = highpass(x, dt_s, highpass_hz)
    sd = float(np.std(d))
    s = spectrum(d, dt_s, segment)
    return AxisResult(axis, sd, float(precision_nm), vibration_rms(sd, precision_nm),
                      s, peaks(s, n_peaks, above_hz=highpass_hz), d)


@dataclass
class TraceResult:
    trace: Trace
    run: slice                       # the stretch analysed
    axes: List[AxisResult] = field(default_factory=list)


def analyse(traces: Sequence[Trace], axes: Sequence[str] = AXES, *,
            highpass_hz: float = 0.0, segment: int = 1024,
            n_peaks: int = 5) -> List[TraceResult]:
    """Every trace, every axis: over the longest stretch it has without a
    hole (a gap inside it is filled; one at the start or end is not data)."""
    out = []
    for t in traces:
        run = longest_run(t.values)
        if run.stop - run.start < 4:
            continue
        r = TraceResult(t, run)
        for a, name in enumerate(axes):
            p = t.precision[a] if a < len(t.precision) else np.nan
            r.axes.append(analyse_axis(t.values[run, a], t.dt_s, p, axis=name,
                                       highpass_hz=highpass_hz, segment=segment,
                                       n_peaks=n_peaks))
        out.append(r)
    return out


def pooled(results: Sequence[TraceResult], a: int, n_peaks: int = 5,
           above_hz: float = 0.0) -> Dict[str, float]:
    """One axis over all traces: variances and the precisions' squares
    averaged, the spectra averaged in power (where their frequencies agree,
    else the first trace's), its largest peak.  With one trace these are its
    own numbers."""
    axes = [r.axes[a] for r in results]
    sd = float(np.sqrt(np.mean([x.sd_nm ** 2 for x in axes])))
    prec = float(np.sqrt(np.mean([x.precision_nm ** 2 for x in axes])))
    s = pooled_spectrum(axes)
    top = peaks(s, 1, above_hz=above_hz)
    f, amp = top[0] if top else (np.nan, np.nan)
    return {"sd_nm": sd, "precision_nm": prec, "vibration_nm": vibration_rms(sd, prec),
            "peak_hz": float(f), "peak_nm": float(amp)}


def pooled_spectrum(axes: Sequence[AxisResult]) -> Spectrum:
    """The power averaged over spectra on the first one's frequencies."""
    first = axes[0].spectrum
    same = [x.spectrum for x in axes if len(x.spectrum.f_hz) == len(first.f_hz)
            and np.allclose(x.spectrum.f_hz, first.f_hz)]
    if len(same) == 1:
        return first
    psd = np.mean([s.psd for s in same], 0)
    amp = np.sqrt(np.mean([s.amplitude ** 2 for s in same], 0))
    return Spectrum(first.f_hz, psd, amp, sum(s.segments for s in same), first.length)


# --------------------------------------------------------- random scatter
#: 1 / Phi^-1(3/4): a median absolute deviation as a Gaussian's SD
MAD_SCALE = 1.4826


def averaging_points(longest: int, n: int = 24, also: Sequence[int] = ()) -> np.ndarray:
    """The m of an Allan curve: 1 and ``longest`` and about ``n`` points
    evenly spaced in log between them, with ``also`` (a chosen time scale)."""
    longest = max(int(longest), 1)
    m = np.rint(np.geomspace(1, longest, max(int(n), 2))).astype(np.int64)
    return np.unique(np.r_[m, [int(a) for a in also if a >= 1]])


def runs_of(t_s: np.ndarray, ok: np.ndarray, max_gap: float) -> Tuple[List[np.ndarray], float]:
    """``(runs, step)``: the points where ``ok`` in stretches without a gap,
    each an index array, and the typical step between consecutive points.

    A gap is a step longer than ``max_gap`` times the median one: a
    difference across it is of two moments far apart, whatever moved in
    between, not of noise.  The median and not the grid step, since on an
    irregular clock (MINFLUX) the measured points sit a few grid steps apart.
    """
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return ([idx] if len(idx) else []), np.nan
    dt = np.diff(np.asarray(t_s, float)[idx])
    step = float(np.median(dt))
    cut = np.flatnonzero(dt > max_gap * step) + 1
    return [r for r in np.split(idx, cut) if len(r)], step


def allan_sums(x: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(sum of squares, count)`` per m of the differences of consecutive
    overlapping means over m points of one gap-free run: what pools over
    runs and traces before the root is taken."""
    x = np.asarray(x, float)
    c = np.r_[0.0, np.cumsum(x - x.mean())]
    sq, count = np.zeros(len(m)), np.zeros(len(m), np.int64)
    for i, mi in enumerate(m):
        if len(x) < 2 * mi:
            continue
        means = (c[mi:] - c[:-mi]) / mi
        d = means[mi:] - means[:-mi]
        sq[i], count[i] = float(np.dot(d, d)), len(d)
    return sq, count


@dataclass
class Scatter:
    """One axis of one trace: the Allan deviation over ``m`` points averaged
    (``tau_s`` = m x the typical step), from ``sq`` and ``count`` (kept to
    pool), and the steps diff(x)/sqrt(2) for the robust value."""
    axis: str
    precision_nm: float
    step_s: float
    m: np.ndarray
    sq: np.ndarray
    count: np.ndarray
    steps_nm: np.ndarray

    @property
    def tau_s(self) -> np.ndarray:
        return self.m * self.step_s

    @property
    def allan_nm(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, np.sqrt(self.sq / (2 * np.maximum(self.count, 1))),
                            np.nan)

    @property
    def scatter_nm(self) -> float:
        """At one point: RMS(diff x) / sqrt(2)."""
        return float(self.allan_nm[0])

    @property
    def robust_nm(self) -> float:
        """The same from the median absolute deviation of the steps: a few
        jumps (a mislinked point, a bead that hops) do not inflate it."""
        s = self.steps_nm
        if not len(s):
            return np.nan
        return float(MAD_SCALE * np.median(np.abs(s - np.median(s))))

    def at(self, points: int) -> float:
        hit = np.flatnonzero(self.m == int(points))
        return float(self.allan_nm[hit[0]]) if len(hit) else np.nan


def scatter(trace: Trace, axes: Sequence[str] = AXES, m: Optional[np.ndarray] = None,
            max_gap: float = 3.0) -> List[Scatter]:
    """Every axis of one trace: its measured points (not the filled ones),
    in runs without a gap (`runs_of`), the Allan sums pooled over the runs."""
    ok = np.isfinite(trace.values).all(1) & ~trace.filled
    runs, step = runs_of(trace.t_s, ok, max_gap)
    if m is None:
        m = averaging_points(max([len(r) for r in runs] or [1]) // 2)
    m = np.asarray(m, np.int64)
    out = []
    for a, name in enumerate(axes):
        sq, count = np.zeros(len(m)), np.zeros(len(m), np.int64)
        steps = []
        for r in runs:
            x = trace.values[r, a]
            s, c = allan_sums(x, m)
            sq += s
            count += c
            steps.append(np.diff(x) / np.sqrt(2))
        p = trace.precision[a] if a < len(trace.precision) else np.nan
        out.append(Scatter(name, float(p), step, m, sq, count,
                           np.concatenate(steps) if steps else np.zeros(0)))
    return out


def pooled_scatter(per_trace: Sequence[Scatter]) -> Scatter:
    """One axis over traces: the sums added (so a long trace counts as much
    as it has differences), the steps joined, the precision the root of the
    mean square and the step the median."""
    first = per_trace[0]
    return Scatter(first.axis,
                   float(np.sqrt(np.nanmean([s.precision_nm ** 2 for s in per_trace])))
                   if any(np.isfinite(s.precision_nm) for s in per_trace) else np.nan,
                   float(np.nanmedian([s.step_s for s in per_trace])), first.m,
                   np.sum([s.sq for s in per_trace], 0),
                   np.sum([s.count for s in per_trace], 0),
                   np.concatenate([s.steps_nm for s in per_trace]))
