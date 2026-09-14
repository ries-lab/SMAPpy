"""Drift correction: estimate with COMET, apply to every localization.

Sample drift is a property of the *acquisition*, not of a subset of it, so the
two halves of drift correction want different data:

* **Estimating** it works best on a clean subset -- bright, well-fitted
  localizations -- because a badly localized point contributes noise to the
  overlap cost and nothing else.  That is exactly what the display filter
  already selects, so `estimate_drift` takes the same `select` a render does.
* **Applying** it is a coordinate correction and must reach *all*
  localizations, including the ones the filter hides.  A filter is a view;
  changing it later must not require re-running the correction.

The estimator is COMET (Cost-function Optimized Maximal Overlap drift
EsTimation, https://github.com/gpufit/Comet), used as a library -- it maximises
the spatiotemporal overlap of localizations across time windows, which needs no
fiducials and no reference structure.  Its result is a per-frame drift table
whose row index *is* the frame number, so applying it is one fancy-index per
coordinate.

Not wrapped: COMET's own file I/O, plotting and molecule-set output.  We hand it
an array and get an array back; reading and writing stays with `io.hdf5`, so the
corrected file is an ordinary smappy file that the viewer opens unchanged.

Units: COMET works in nm.  A table in pixels is converted for the estimate and
the drift is divided back by the pixel size when applied, so the corrected table
keeps whatever units it had.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .locs import Localizations

# the position columns a drift correction touches, and the drift axis each
# follows.  Deliberately not `peak_*`: those are detection positions, kept as
# the record of what was found where, not measurements of the structure.
_POSITION_COLUMNS = {"x_nm": 0, "x_pix": 0,
                     "y_nm": 1, "y_pix": 1,
                     "z_nm": 2}


@dataclass
class DriftSettings:
    """COMET parameters.  The defaults suit a typical SMLM movie.

    ``segmentation_mode`` sets what ``segmentation_var`` counts:
    0 = number of time windows, 1 = localizations per window,
    2 = frames per window (the default).

    A time window must hold enough localizations to define the structure --
    a few thousand is a good target -- and the movie must give at least a few
    tens of windows for the drift curve to have any time resolution.
    """

    segmentation_mode: int = 2
    segmentation_var: int = 500
    max_drift_nm: float = 300.0      # also the neighbour-search radius
    target_sigma_nm: float = 30.0    # refinement stops at this length scale
    initial_sigma_nm: Optional[float] = None   # default: max_drift_nm / 3
    boxcar_width: int = 1            # temporal smoothing between iterations
    interpolation: str = "cubic"     # "cubic" or "catmull-rom"
    max_locs_per_segment: Optional[int] = None   # cap, to bound memory
    # L-BFGS-B relative tolerance.  COMET's own is 1e3*eps (~2e-13), which buys
    # nothing here: loosening it to 1e-7 costs 0.14 nm rms (0.9 nm worst case)
    # on the drift curve -- far below any localization precision -- and cuts the
    # cost evaluations from 423 to 154.  Raise it towards 1e-13 to reproduce
    # upstream COMET exactly.
    optimizer_ftol: float = 1e-7
    # Compute the Gaussian from a table plus a polynomial and skip pairs beyond
    # 6 sigma: 1.3x faster, gradient accurate to 4e-4 relative.  See NOTES.
    approximate_kernel: bool = True
    # A looser tolerance for the coarse sigma steps.  None (the default) means
    # the same as `optimizer_ftol`: measured, loosening the coarse steps is a
    # bad trade -- 2.3x faster for tens of nm on individual windows.  See NOTES.
    optimizer_ftol_coarse: Optional[float] = None
    backend: Optional[str] = None    # "cuda", "torch", "cpu"; None = fastest
    # Estimate from *grouped* localizations: one entry per emitter blink rather
    # than one per frame.  20-35x faster (it removes exactly the dense, very
    # short-range pairs, which is where the pair count grows fastest) and, on
    # the clathrin dataset, recovers 96% of the improvement the full estimate
    # makes to the image.  Off by default: the full estimate is the accurate
    # one, and is now fast enough to be the thing you run.
    # Check each time window's estimate against the overlap it would have with
    # no drift correction at all, and discard the ones that do not beat it --
    # the spline then bridges them.  Costs one extra pass over the pairs.
    quality_control: bool = False   # time-window fit only, not the spline
    # A window is discarded when its fitted drift improves its own overlap by
    # less than this, relative to no correction at all.  0.0 keeps every window
    # whose estimate is better than nothing, which is the honest bar.
    min_lift: float = 0.0
    group: bool = True
    group_dx_nm: float = 50.0        # linking radius
    group_dt: int = 1                # frames a blink may be missing
    # Estimate in two passes: a grouped one to get within a few nm, then an
    # ungrouped one over a small radius to refine.  The pair count is what costs
    # time, and once the coarse pass has taken the drift out, a 30 nm radius
    # keeps a seventh of the pairs -- 18 s instead of 153 s for 99% of the
    # improvement.  The small radius also bounds the fine pass, which is what
    # keeps it from producing the runaway window the single pass has here.
    two_stage: bool = False
    two_stage_radius_nm: float = 30.0
    # Fit the drift as a cubic B-spline in time instead of one free vector per
    # time window.  Drift is smooth -- slow creep, occasionally a small jump --
    # so a curve with a coefficient every `spline_knot_frames` frames has far
    # fewer parameters than a window per 500 frames, and each coefficient is
    # constrained by every localization near it in time rather than by one
    # window's worth.  There is no time window and no interpolation afterwards:
    # every localization is placed at its own frame, and the fitted curve *is*
    # the per-frame drift.  B-splines are variation-diminishing, so the curve
    # cannot overshoot the way an interpolating cubic spline through noisy
    # window estimates does.
    spline: bool = True
    # 2000 frames was the knee on the clathrin dataset (26 coefficients over
    # 46 005 frames); finer knots are noisier without being sharper, and there
    # was no sign of over-smoothing even at 4000.  A dataset with faster drift
    # wants a smaller spacing, which shows up as out-of-sample sharpness getting
    # worse.  Quality control does not apply here -- there are no time windows.
    spline_knot_frames: int = 2000   # spacing of the coefficients, in frames
    spline_penalty: float = 0.0      # extra roughness penalty on 2nd differences
    use_z: Optional[bool] = None     # None: 3D if the table has z_nm


class Drift:
    """Per-frame drift in nm, indexed by frame number.

    ``drift[f]`` is ``(dx, dy, dz)``: how far the sample had moved in frame
    ``f``, so correcting a localization *subtracts* it.
    """

    def __init__(self, drift_nm: np.ndarray, settings: Optional[DriftSettings] = None,
                 n_used: Optional[int] = None,
                 flagged_windows: Optional[np.ndarray] = None):
        drift = np.asarray(drift_nm, dtype=np.float64)
        if drift.ndim != 2 or drift.shape[1] != 3:
            raise ValueError(f"drift must be (n_frames, 3), got {drift.shape}")
        self.drift = drift
        self.settings = settings
        self.n_used = n_used  # localizations the estimate was made from
        # time windows quality control discarded; the curve is interpolated
        # across them, so they are a caveat on the result, not a hole in it
        self.flagged_windows = flagged_windows

    def __len__(self) -> int:
        return self.drift.shape[0]

    @property
    def frames(self) -> np.ndarray:
        return np.arange(len(self))

    def __str__(self) -> str:
        span = self.drift.max(axis=0) - self.drift.min(axis=0)
        text = (f"drift over {len(self)} frames, range "
                f"x {span[0]:.0f} nm, y {span[1]:.0f} nm, z {span[2]:.0f} nm")
        if self.flagged_windows is not None and len(self.flagged_windows):
            text += (f"; {len(self.flagged_windows)} time window(s) flagged and "
                     f"interpolated over: {list(self.flagged_windows)}")
        return text

    def apply(self, locs: Localizations,
              pixelsize_nm: Optional[float] = None) -> Localizations:
        """Return a copy of the whole table with the drift subtracted."""
        frames = np.asarray(locs["frame"], dtype=np.int64)
        if len(frames) and (frames.min() < 0 or frames.max() >= len(self)):
            raise ValueError(
                f"the table spans frames {frames.min()}..{frames.max()} but the "
                f"drift covers 0..{len(self) - 1}; it was estimated from a "
                f"different acquisition")

        columns = dict(locs.columns)
        corrected = []
        for name, axis in _POSITION_COLUMNS.items():
            if name not in columns:
                continue
            shift = self.drift[frames, axis]
            if name.endswith("_pix"):
                shift = shift / _pixelsize_nm(locs, pixelsize_nm)
            values = np.asarray(columns[name], dtype=np.float64) - shift
            columns[name] = values.astype(np.asarray(columns[name]).dtype)
            corrected.append(name)

        metadata = dict(locs.metadata)
        metadata["drift_correction"] = {
            "method": "comet",
            "columns": corrected,
            "n_frames": len(self),
            "n_localizations_used": self.n_used,
            "flagged_windows": (None if self.flagged_windows is None
                                else [int(w) for w in self.flagged_windows]),
            "settings": asdict(self.settings) if self.settings else None,
        }
        return Localizations(columns, metadata)

    def plot(self, ax=None):
        """Drift vs frame, the standard sanity check."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotting the drift needs matplotlib: "
                              "pip install smappy-smlm[drift]") from None

        if ax is None:
            _, ax = plt.subplots()
        for axis, label in enumerate("xyz"):
            if np.any(self.drift[:, axis]):
                ax.plot(self.frames, self.drift[:, axis], label=label)
        ax.set_xlabel("frame")
        ax.set_ylabel("drift (nm)")
        ax.legend()
        return ax


# ------------------------------------------------------------- what it will cost
# Both the time and the memory of a drift estimate are set by one number: how
# many neighbour pairs there are within `max_drift_nm`.  Every cost evaluation
# is one `exp()` per pair, and the KD-tree search has to hold all of them at
# once.  The constants below are measured on an M1 Max (see NOTES.md, "Where
# the time goes"); they are the right order on any CPU and are only ever used
# to decide whether to warn, never to change what is computed.
_SECONDS_PER_PAIR_EVALUATION = 2.1e-9   # the compiled kernel, across the cores
_SECONDS_PER_PAIR_SEARCH = 16e-9        # cKDTree.query_pairs
_SECONDS_PER_LOCALIZATION = 3e-7        # grouping, and applying the drift
# The sigma ladder runs a full L-BFGS-B at each level; 136 evaluations over 6
# levels measured on a 93 k-frame dataset, 154 over 6 on the clathrin one.
_EVALUATIONS_PER_SIGMA_LEVEL = 23
# scipy hands the pairs back as one int64 (N, 2) array and `pair_indices_kdtree`
# copies them into two int32 arrays, all three alive at once.  scipy builds that
# array from an internal vector, which can transiently double the 16 again, so
# this is a floor and not a worst case.
_BYTES_PER_PAIR = 24
# localizations the pair count is estimated from.  The count scales as N^2, so
# a sample answers for the whole: 60 k was within 0.1% of the true pair count
# on a 1.7 M-localization dataset, in under a tenth of a second.
_PAIR_SAMPLE = 60_000


@dataclass
class DriftCost:
    """What a drift estimate with given settings is going to cost.

    `seconds` and `memory_bytes` both follow from `n_pairs`, which is estimated
    rather than counted -- see `estimate_cost`.  They are an order-of-magnitude
    guide meant for deciding whether to start at all, not a promise.
    """

    n_localizations: int            # what the estimate runs on, after grouping
    n_pairs: float                  # neighbour pairs within max_drift_nm
    n_evaluations: int              # cost evaluations the sigma ladder will do
    seconds: float
    memory_bytes: float
    max_drift_nm: float
    machine_bytes: Optional[int] = None   # this computer's RAM, if known

    @property
    def fits_in_memory(self) -> bool:
        """Is there room for the pairs?  Unknown memory counts as room.

        Half the machine is the bar: the table, its index and the rendered
        view are already resident, and the pair estimate is a floor.
        """
        if self.machine_bytes is None:
            return True
        return self.memory_bytes < 0.5 * self.machine_bytes

    def question(self, slow_seconds: float = 300.0) -> Optional[str]:
        """What to put to the user before starting, or None to just start."""
        if not self.fits_in_memory:
            return (f"{self}\n\nThis will not fit in memory and will most "
                    f"likely die after a long time in swap.\n\n"
                    f"{_CHEAPER}\n\nRun it anyway?")
        if self.seconds > slow_seconds:
            return (f"{self}\n\n{_CHEAPER}\n\nRun it?")
        return None

    def __str__(self) -> str:
        return (f"{_count(self.n_localizations)} localizations, about "
                f"{_count(self.n_pairs)} neighbour pairs within "
                f"{self.max_drift_nm:g} nm: {_duration(self.seconds)} and at "
                f"least {_bytes(self.memory_bytes)} of memory"
                + ("" if self.machine_bytes is None
                   else f" ({_bytes(self.machine_bytes)} in this machine)"))


_CHEAPER = ("The pair count grows with the square of the localization count "
            "and steeply with the search radius, so the levers are a smaller "
            "max drift (it is the neighbour radius -- use what the stage "
            "actually drifts, not a safe-looking round number), grouping, and "
            "a tighter filter on what the estimate is made from.")


def estimate_cost(locs: Localizations, settings: Optional[DriftSettings] = None,
                  select=None, pixelsize_nm: Optional[float] = None) -> DriftCost:
    """What a drift estimate on this table will cost, without running it.

    Counting the pairs outright is the expensive part of the run itself, so
    they are counted on a sample and scaled: the count grows as N^2 for a
    fixed field, which holds to a fraction of a percent on real data.  The
    whole thing takes well under a second on millions of localizations, which
    is what lets it run before every drift correction rather than on request.
    """
    settings = settings or DriftSettings()
    index = _selection(locs, select)
    n_selected = int(len(index))
    machine = _machine_memory()
    if n_selected < 2 or "frame" not in locs:
        return DriftCost(n_selected, 0.0, 0, 0.0, 0.0,
                         settings.max_drift_nm, machine)

    x, y, z = _positions_nm(locs, pixelsize_nm, settings.use_z)
    coords = np.column_stack([x[index], y[index], z[index]])
    frames = np.asarray(locs["frame"], dtype=np.int64)[index]

    sample, ratio = _cost_sample(coords, frames, settings, pixelsize_nm)
    n_estimate = max(2, int(round(n_selected * ratio)))

    pairs = _pairs_in_radius(sample, settings.max_drift_nm, n_estimate)
    if settings.two_stage:
        # a grouped pass at the full radius, then an ungrouped one at a small
        # one; both run, so the time adds and the memory is whichever is larger
        fine = _pairs_in_radius(coords, settings.two_stage_radius_nm, n_selected)
    else:
        fine = 0.0

    levels = _sigma_levels(settings)
    evaluations = levels * _EVALUATIONS_PER_SIGMA_LEVEL
    seconds = ((pairs + fine) * (_SECONDS_PER_PAIR_SEARCH
                                 + evaluations * _SECONDS_PER_PAIR_EVALUATION)
               + len(locs) * _SECONDS_PER_LOCALIZATION)
    return DriftCost(n_localizations=n_estimate, n_pairs=pairs,
                     n_evaluations=evaluations, seconds=seconds,
                     memory_bytes=max(pairs, fine) * _BYTES_PER_PAIR,
                     max_drift_nm=settings.max_drift_nm, machine_bytes=machine)


def _pairs_in_radius(sample: np.ndarray, radius: float, n_total: int) -> float:
    """Pairs within `radius` among `n_total` points, from a sample of them."""
    from scipy.spatial import cKDTree

    if len(sample) > _PAIR_SAMPLE:
        # a fixed seed, so the same table gives the same estimate every run
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), _PAIR_SAMPLE, replace=False)]
    if len(sample) < 2:
        return 0.0
    tree = cKDTree(sample)
    pairs = (tree.count_neighbors(tree, radius) - len(sample)) / 2.0
    return float(pairs) * (n_total / len(sample)) ** 2


def _cost_sample(coords: np.ndarray, frames: np.ndarray, settings: DriftSettings,
                 pixelsize_nm: Optional[float]) -> Tuple[np.ndarray, float]:
    """Positions to estimate the pair count from, and what grouping will do.

    With `group` on, the estimate runs on grouped localizations, which are both
    fewer and differently distributed -- grouping is exactly what removes the
    densest, shortest-range pairs.  So the sample is *grouped* too: a stretch
    of frames from the middle of the acquisition, which is a fair sample of the
    field because every frame images the same one.  The ratio it groups by is
    what the whole table will group by.
    """
    if not settings.group:
        return coords, 1.0

    lo, hi = int(frames.min()), int(frames.max())
    n_frames = hi - lo + 1
    span = max(1, int(n_frames * min(1.0, 4.0 * _PAIR_SAMPLE / max(len(frames), 1))))
    start = lo + max(0, (n_frames - span) // 2)
    taken = (frames >= start) & (frames < start + span)
    if taken.sum() < 2:
        return coords, 1.0

    slice_table = Localizations({"x_nm": coords[taken, 0], "y_nm": coords[taken, 1],
                                 "z_nm": coords[taken, 2], "frame": frames[taken]})
    grouped = _grouped(slice_table, settings, pixelsize_nm)
    if len(grouped) < 2:
        return coords, 1.0
    return (np.column_stack([grouped["x_nm"], grouped["y_nm"], grouped["z_nm"]]),
            len(grouped) / float(taken.sum()))


def _report_evaluation(progress: Optional[Progress], state: dict,
                       expected: int, sigma: float) -> None:
    """Say where the optimizer is, at most once a second.

    Once a second rather than every evaluation because each one is a queued
    signal into the GUI thread, and on a small dataset they arrive hundreds of
    times a second.
    """
    if progress is None:
        return
    now = time.time()
    if now - state["reported"] < 1.0:
        return
    state["reported"] = now
    done = state["evaluations"]
    each = (now - state["started"]) / done
    left = max(0.0, (expected - done) * each)
    progress(f"sigma {sigma:.0f} nm: evaluation {done} of about {expected}, "
             f"{each:.2g} s each, ~{_duration(left)} left")


def _sigma_levels(settings: DriftSettings) -> int:
    """How many times the refinement loop runs the optimizer."""
    sigma = (settings.initial_sigma_nm if settings.initial_sigma_nm is not None
             else settings.max_drift_nm / 3.0)
    target = settings.target_sigma_nm
    levels = 1
    while sigma > target and levels < 100:
        sigma = max(sigma / 1.5, target)
        levels += 1
    return levels


def _machine_memory() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def _count(n: float) -> str:
    for limit, suffix in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if n >= limit:
            return f"{n / limit:.1f} {suffix}"
    return f"{n:.0f}"


def _bytes(n: float) -> str:
    for limit, suffix in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
        if n >= limit:
            return f"{n / limit:.1f} {suffix}"
    return f"{n:.0f} B"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


# --------------------------------------------------------------------- running
def estimate_drift(locs: Localizations, settings: Optional[DriftSettings] = None,
                   select=None, pixelsize_nm: Optional[float] = None,
                   display: bool = False,
                   progress: Optional[Progress] = None) -> Drift:
    """Estimate drift from the selected localizations.

    ``select`` is what to estimate *from*: a `LocFilter`, a boolean mask, an
    index array, or None for everything.  The result covers every frame of the
    input table, not only the selected ones.
    """
    from ._comet import best_backend, comet_run_kd   # vendored; see _comet/
    from ._comet.core import drift_optimizer
    from ._comet.core.qc_utils import flag_flawed_segments_by_lift
    from ._comet.core.cpu_wrapper import (cpu_wrapper_chunked_approx,
                                          cpu_wrapper_chunked_fast)

    settings = settings or DriftSettings()
    if settings.quality_control and settings.spline:
        raise ValueError("quality control flags time windows, and the spline "
                         "fit has none; pass spline=False to use it")
    if settings.spline:
        return _estimate_spline(locs, settings, select, pixelsize_nm, display,
                                progress)
    if settings.two_stage:
        return _two_stage(locs, settings, select, pixelsize_nm, display, progress)

    # Opt in to the two changes made to the vendored COMET (see NOTES.md); its
    # own defaults are untouched for anyone importing it directly.
    drift_optimizer.CPU_WRAPPER = (cpu_wrapper_chunked_approx
                                   if settings.approximate_kernel
                                   else cpu_wrapper_chunked_fast)
    drift_optimizer.LBFGSB_OPTIONS = dict(drift_optimizer.LBFGSB_OPTIONS,
                                          ftol=float(settings.optimizer_ftol))
    drift_optimizer.LBFGSB_OPTIONS_COARSE = None if settings.optimizer_ftol_coarse is None else dict(
        drift_optimizer.LBFGSB_OPTIONS, ftol=float(settings.optimizer_ftol_coarse))

    # the drift must cover every frame of the table it will be applied to,
    # whatever subset it is estimated from
    all_frames = np.asarray(locs["frame"], dtype=np.int64)
    n_frames = int(all_frames.max()) + 1 if len(all_frames) else 0

    backend = settings.backend or best_backend()
    if settings.quality_control:
        backend += "_qc"
    drift_optimizer.LAST_QUALITY_CONTROL = None
    drift_optimizer.FLAG_SEGMENTS = lambda q_obs, q_null: flag_flawed_segments_by_lift(
        q_obs, q_null, settings.min_lift)

    source = locs[_selection(locs, select)]
    if settings.group:
        source = _grouped(source, settings, pixelsize_nm, display, progress)
    x, y, z = _positions_nm(source, pixelsize_nm, settings.use_z)
    dataset = np.column_stack([x, y, z,
                               np.asarray(source["frame"], dtype=np.float64)])
    if len(dataset) < 2:
        raise ValueError("need at least two localizations to estimate drift")
    # COMET reports through its own `display`; from here it is one long call,
    # so say what was handed to it before it starts
    _log(display, f"estimating drift from {len(dataset):,} localizations in "
                  f"time windows (neighbours within {settings.max_drift_nm:g} nm)",
         progress)

    # COMET's `display` is both its progress log and a blocking `plt.show()`;
    # we want the log, and to draw the drift ourselves when asked (Drift.plot)
    with _without_blocking_plots(display):
        drift = comet_run_kd(
            dataset,
            segmentation_mode=settings.segmentation_mode,
            segmentation_var=settings.segmentation_var,
            max_drift_nm=settings.max_drift_nm,
            target_sigma_nm=settings.target_sigma_nm,
            initial_sigma_nm=settings.initial_sigma_nm,
            boxcar_width=settings.boxcar_width,
            interpolation_method=settings.interpolation,
            max_locs_per_segment=settings.max_locs_per_segment,
            mode=backend,
            display=display,
            # an upfront pair-count estimate: the neighbour search runs over the
            # whole dataset at once, and is where too large a max_drift_nm turns
            # into minutes and gigabytes.  Better warned before than after.
            pair_indices_safety_check=True,
            # the estimate is made from a subset, but must cover every frame of
            # the table it will be applied to
            min_max_frames=(0, n_frames - 1),
        )
    quality = drift_optimizer.LAST_QUALITY_CONTROL
    return Drift(drift[:, :3], settings, n_used=len(dataset),
                 flagged_windows=None if quality is None else quality["flagged"])


def _spline_basis(x, n_coefficients: int, lo: float, hi: float, degree: int = 3):
    """Cubic B-spline design matrix: row per point, column per coefficient."""
    from scipy.interpolate import BSpline

    interior = np.linspace(lo, hi, n_coefficients - degree + 1)
    knots = np.concatenate([np.full(degree, lo), interior, np.full(degree, hi)])
    return BSpline.design_matrix(np.clip(np.asarray(x, dtype=float), lo, hi),
                                 knots, degree)


def _estimate_spline(locs: Localizations, settings: DriftSettings, select,
                     pixelsize_nm: Optional[float], display: bool,
                     progress: Optional[Progress] = None) -> "Drift":
    """Fit drift as a spline in time, with no segmentation at all.

    The optimizer's variables are the spline's coefficients; the cost and its
    gradient are COMET's, over the same neighbour pairs.  Each localization
    carries its own frame, so `mu` handed to the kernel is the drift *per
    frame*, and the chain rule to the coefficients is one matrix product.
    """
    from ._comet.core.cpu_wrapper import cpu_wrapper_chunked_approx
    from ._comet.core.pair_indices import pair_indices_kdtree
    from scipy.optimize import minimize

    all_frames = np.asarray(locs["frame"], dtype=np.int64)
    n_frames = int(all_frames.max()) + 1 if len(all_frames) else 0

    source = locs[_selection(locs, select)]
    if settings.group:
        source = _grouped(source, settings, pixelsize_nm, display, progress)
    x, y, z = _positions_nm(source, pixelsize_nm, settings.use_z)
    coords = np.ascontiguousarray(np.column_stack([x, y, z]), dtype=np.float32)
    frames = np.ascontiguousarray(source["frame"], dtype=np.int32)

    # One uninterruptible scipy call that has to hold every pair at once, so
    # this is where too large a radius turns into swap and a dead process.
    # `estimate_cost` is what stands in front of it; say what is being asked
    # for, so a run that dies here says so in the log.
    _log(display, f"searching for neighbour pairs within "
                  f"{settings.max_drift_nm:g} nm among {len(coords):,} "
                  f"localizations", progress)
    idx_i, idx_j, ok = pair_indices_kdtree(coords, settings.max_drift_nm)
    if not ok:
        raise MemoryError("the neighbour search ran out of memory; reduce "
                          "max_drift_nm or use group=True")

    n_coefficients = max(4, int(round(n_frames / settings.spline_knot_frames)) + 3)
    basis = _spline_basis(np.arange(n_frames), n_coefficients, 0.0, n_frames - 1.0)
    basis_t = basis.T.tocsr()
    _log(display, f"spline drift: {n_coefficients} coefficients over {n_frames} "
                  f"frames, {len(idx_i):,} pairs", progress)

    penalty = float(settings.spline_penalty)
    if penalty:
        # second differences of the coefficients: what "not oscillating" means
        d2 = np.diff(np.eye(n_coefficients), 2, axis=0)
        curvature = d2.T @ d2

    # Every evaluation costs the same -- one pass over the pairs -- so two of
    # them are enough to say how long the rest will take.  The total is only
    # approximate (L-BFGS-B decides when it is done), which the wording says.
    state = {"evaluations": 0, "started": time.time(), "reported": 0.0}
    expected = _sigma_levels(settings) * _EVALUATIONS_PER_SIGMA_LEVEL

    def objective(flat, sigma, factor):
        c = flat.reshape(n_coefficients, 3)
        mu = np.ascontiguousarray(basis @ c)
        value, gradient = cpu_wrapper_chunked_approx(
            mu, coords, frames, idx_i, idx_j, sigma, factor)
        gradient = basis_t @ gradient.reshape(n_frames, 3)
        if penalty:
            value += penalty * float((c * (curvature @ c)).sum())
            gradient = gradient + 2.0 * penalty * (curvature @ c)
        state["evaluations"] += 1
        _report_evaluation(progress, state, expected, sigma)
        return value, gradient.ravel()

    sigma = (settings.initial_sigma_nm if settings.initial_sigma_nm is not None
             else settings.max_drift_nm / 3.0)
    target = settings.target_sigma_nm
    bound = settings.max_drift_nm * 2.0
    coefficients = np.zeros(n_coefficients * 3)
    while True:
        result = minimize(objective, coefficients, args=(sigma, 1.0), jac=True,
                          method="L-BFGS-B", bounds=[(-bound, bound)] * coefficients.size,
                          options={"ftol": settings.optimizer_ftol, "gtol": 1e-5,
                                   "maxls": 40})
        coefficients = result.x
        _log(display, f"  sigma {sigma:6.1f} nm: {result.nfev} evaluations, "
                      f"cost {result.fun:.6g}", progress)
        state["reported"] = 0.0          # the next evaluation line starts fresh
        if sigma <= target:
            break
        sigma = max(sigma / 1.5, target)

    drift = np.asarray(basis @ coefficients.reshape(n_coefficients, 3))
    return Drift(drift, settings, n_used=len(coords))


# What a caller is told while this runs.  The GUI passes the plugin panel's
# `ctx.report`, the command line passes `print` through `display`.
Progress = Callable[[str], None]


def _log(enabled: bool, message: str, progress: Optional[Progress] = None) -> None:
    if progress is not None:
        progress(message)
    if enabled:
        print(message)


def _two_stage(locs: Localizations, settings: DriftSettings, select,
               pixelsize_nm: Optional[float], display: bool,
               progress: Optional[Progress] = None) -> "Drift":
    """A grouped pass to get close, then an ungrouped one over a small radius."""
    _log(display, "pass 1 of 2: grouped, over the full search radius", progress)
    coarse = estimate_drift(locs, replace(settings, two_stage=False, group=True),
                            select, pixelsize_nm, display, progress)
    radius = settings.two_stage_radius_nm
    _log(display, f"pass 2 of 2: ungrouped, within {radius:g} nm", progress)
    fine = estimate_drift(
        coarse.apply(locs, pixelsize_nm),
        replace(settings, two_stage=False, group=False, max_drift_nm=radius,
                initial_sigma_nm=radius / 3.0, target_sigma_nm=radius / 5.0),
        select, pixelsize_nm, display, progress)
    # the two are drift of the same sample measured in sequence, so they add
    return Drift(coarse.drift + fine.drift, settings, n_used=fine.n_used,
                 flagged_windows=fine.flagged_windows)


def correct_drift(locs: Localizations, settings: Optional[DriftSettings] = None,
                  select=None, pixelsize_nm: Optional[float] = None,
                  display: bool = False, progress: Optional[Progress] = None):
    """Estimate from the selection, correct everything.  Returns (locs, drift)."""
    drift = estimate_drift(locs, settings, select, pixelsize_nm, display, progress)
    _log(display, f"applying the drift to {len(locs):,} localizations", progress)
    return drift.apply(locs, pixelsize_nm), drift


# ----------------------------------------------------------------------- saving
def drift_corrected_path(path) -> Path:
    """``foo.hdf5`` -> ``foo_driftc.hdf5``."""
    path = Path(path)
    return path.with_name(path.stem + "_driftc" + (path.suffix or ".hdf5"))


def save_drift_corrected(path, locs: Localizations, drift: Drift) -> Path:
    """Write the corrected table, with the drift curve alongside it.

    The file is an ordinary smappy localization file -- the viewer and every
    other reader take it as it is -- plus a ``/drift`` group, so the correction
    that was applied can be inspected or undone later.
    """
    import h5py

    from .io.hdf5 import save_localizations

    from . import _comet

    path = save_localizations(path, locs)
    with h5py.File(path, "a") as f:
        group = f.create_group("drift")
        for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
            group.create_dataset(name, data=drift.drift[:, axis].astype(np.float32))
        group.create_dataset("frame", data=drift.frames.astype(np.int64))
        # the estimator is somebody else's published method: say so in the file,
        # so a result can be traced -- and cited -- back to it
        group.attrs["method"] = _comet.UPSTREAM
        group.attrs["method_version"] = _comet.__version__
    return path


def load_drift(path) -> Drift:
    """Read the drift curve stored next to a corrected table."""
    import h5py

    with h5py.File(path, "r") as f:
        group = f["drift"]
        drift = np.column_stack([group[name][()]
                                 for name in ("x_nm", "y_nm", "z_nm")])
    return Drift(drift)


# ---------------------------------------------------------------------- helpers
def _grouped(locs: Localizations, settings: "DriftSettings",
             pixelsize_nm: Optional[float], display: bool = False,
             progress: Optional[Progress] = None) -> Localizations:
    """Collapse each emitter's blink into one localization before estimating."""
    from .group import GroupSettings, group

    dx = settings.group_dx_nm
    if "x_nm" not in locs:  # a table in pixels; the linking radius is in nm
        dx /= _pixelsize_nm(locs, pixelsize_nm)
    _log(display, f"grouping {len(locs):,} localizations", progress)
    grouped, _ = group(locs, GroupSettings(dx=dx, dt=settings.group_dt))
    _log(display, f"grouped {len(locs):,} localizations into {len(grouped):,} "
                  f"blinks", progress)
    return grouped


@contextmanager
def _without_blocking_plots(active: bool):
    """Let COMET log its progress without stopping in its own `plt.show()`.

    Its diagnostic figures are closed again; a drift curve is one line here
    (`Drift.plot`) and belongs where the caller decides to draw it.
    """
    if not active:
        yield
        return
    try:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotting the drift needs matplotlib: "
                              "pip install smappy-smlm[drift]") from None
    except ImportError:
        yield
        return
    existing = set(plt.get_fignums())
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = show
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)


def _selection(locs: Localizations, select) -> np.ndarray:
    """A LocFilter, a boolean mask or an index array -> indices."""
    n = len(locs)
    if select is None:
        return np.arange(n)
    indices = getattr(select, "indices", None)  # a LocFilter
    if indices is not None:
        return np.asarray(indices)
    select = np.asarray(select)
    if select.dtype == bool:
        if select.shape != (n,):
            raise ValueError(f"mask has {select.shape[0]} entries, table has {n}")
        return np.flatnonzero(select)
    return select.astype(np.int64)


def _pixelsize_nm(locs: Localizations, pixelsize_nm: Optional[float]) -> float:
    """The pixel size, given or from the table's own provenance."""
    if pixelsize_nm is not None:
        return float(pixelsize_nm)
    value = locs.metadata.get("pixelsize_nm")
    if value is None:
        camera = locs.metadata.get("camera") or {}
        micron = camera.get("pixelsize_um") if isinstance(camera, dict) else None
        value = None if micron is None else float(micron) * 1000.0
    if value is None:
        raise ValueError("this table is in pixels and its pixel size is not "
                         "recorded; pass pixelsize_nm=...")
    return float(value)


def _positions_nm(locs: Localizations, pixelsize_nm: Optional[float],
                  use_z: Optional[bool]):
    """x, y, z of the whole table in nm; z is zero for a 2D table."""
    if "x_nm" in locs and "y_nm" in locs:
        x = np.asarray(locs["x_nm"], dtype=np.float64)
        y = np.asarray(locs["y_nm"], dtype=np.float64)
    elif "x_pix" in locs and "y_pix" in locs:
        scale = _pixelsize_nm(locs, pixelsize_nm)
        x = np.asarray(locs["x_pix"], dtype=np.float64) * scale
        y = np.asarray(locs["y_pix"], dtype=np.float64) * scale
    else:
        raise ValueError("no position columns (x_nm/y_nm or x_pix/y_pix)")

    has_z = "z_nm" in locs
    if use_z is None:
        use_z = has_z
    elif use_z and not has_z:
        raise ValueError("use_z=True but the table has no z_nm column")
    z = np.asarray(locs["z_nm"], dtype=np.float64) if use_z else np.zeros_like(x)
    return x, y, z
