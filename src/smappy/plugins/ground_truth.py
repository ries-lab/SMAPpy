"""How well a fit found a simulation's molecules: detection and accuracy.

A simulation knows where every spot was, so a fit of it can be scored rather
than eyeballed.  The truth is not stored beside the table but worked out again
from the recipe -- the ``*.sim.yaml`` the table was fitted from (its
``source``), or the settings a simulated table carries -- because the recipe
and the seed determine it exactly and it costs a second to redraw.

**Matching.**  In each frame, localizations and true spots are paired one to
one by the Hungarian algorithm on their lateral distance, and a pair further
apart than the radius is no pair, as in the SMLM challenge (Sage et al., Nat.
Methods 2019).  A localization left over is a false positive, a true spot
left over a false negative, and the Jaccard index ``TP / (TP + FP + FN)`` is
the one number for detection.

**Which true spots count.**  Every spot is matched against, but only some are
*counted*: those above ``min_photons`` (a fluorophore on for a sliver of a
frame is found by no fitter, and counting it measures the simulation, not the
fit), and optionally only the isolated ones.  A localization matched to a
spot that is not counted is neither right nor wrong and is set aside --
otherwise tightening what counts would turn good fits into false positives.

**Accuracy** is read from the matched pairs: the bias and the RMS error per
axis, and the *pull*, the error over the precision the table claims for that
localization.  A pull of spread 1 says the reported precision is honest; the
precision plugins test the same claim without a truth.  With z, fitted
against true z is fitted by a line, whose slope is the scale of the axial
calibration (1 is right; crowded spots pull it below, NOTES.md "Crowding
compresses z").

The table's layer filter and ROI apply to the localizations, and the ROI to
the truth as well, so the score is of what is being looked at: filtering on
fit quality trades false positives for false negatives, and the Jaccard index
says whether that was a good trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..locs import Localizations
from . import Context, Plot, Plugin, Result, param, register

SIM_FILTER = "Simulations (*.sim.yaml);;All files (*)"


# ------------------------------------------------------------------- truth
def truth_for(locs: Localizations, path: str = "") -> Localizations:
    """The true spots behind ``locs``: from ``path`` (a ``*.sim.yaml``), else
    from the recipe the table was fitted from, else from the settings a
    simulated table carries."""
    from ..simulate import camera_truth, ground_truth
    from ..simulate.source import is_simulation, read_recipe
    source = path or str(locs.metadata.get("source") or "")
    if source:
        if not is_simulation(source):
            raise ValueError(f"{source} is not a simulation (*.sim.yaml): there is "
                             f"no truth to compare with")
        if not Path(source).exists():
            raise FileNotFoundError(f"the simulation {source} is not there any more")
        settings = read_recipe(source)
    elif locs.metadata.get("simulation_settings"):
        from ..simulate import SimulationSettings
        from . import settings_from
        settings = settings_from(SimulationSettings, locs.metadata["simulation_settings"])
    else:
        raise ValueError("this table says nothing of a simulation: fit a *.sim.yaml, "
                         "simulate one, or name the simulation file")
    return camera_truth(ground_truth(settings))


# ---------------------------------------------------------------- matching
def match(fit_xy, fit_frame, true_xy, true_frame, radius: float
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Indices ``(fitted, true)`` of the pairs: one to one within each frame,
    the assignment of least total distance, no pair further than ``radius``."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    fit_frame, true_frame = np.asarray(fit_frame), np.asarray(true_frame)
    fo, to = np.argsort(fit_frame, kind="stable"), np.argsort(true_frame, kind="stable")
    frames = np.intersect1d(fit_frame, true_frame)
    fa, fb = (np.searchsorted(fit_frame[fo], frames, s) for s in ("left", "right"))
    ta, tb = (np.searchsorted(true_frame[to], frames, s) for s in ("left", "right"))
    out_f, out_t = [], []
    big = 1e12
    for k in range(len(frames)):
        fi, ti = fo[fa[k]:fb[k]], to[ta[k]:tb[k]]
        if len(fi) == 1 and len(ti) == 1:
            if np.hypot(*(fit_xy[fi[0]] - true_xy[ti[0]])) <= radius:
                out_f.append(fi)
                out_t.append(ti)
            continue
        # only pairs within the radius can match: a sparse matrix, most
        # frames being a handful of spots far apart
        near = cKDTree(true_xy[ti]).sparse_distance_matrix(
            cKDTree(fit_xy[fi]), radius, output_type="ndarray")
        if not len(near):
            continue
        cost = np.full((len(ti), len(fi)), big)
        cost[near["i"], near["j"]] = near["v"]
        rows, cols = linear_sum_assignment(cost)
        ok = cost[rows, cols] < big
        out_f.append(fi[cols[ok]])
        out_t.append(ti[rows[ok]])
    if not out_f:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_f), np.concatenate(out_t)


def _near_any(xy, frame, other_xy, other_frame, radius) -> np.ndarray:
    """Whether each point has one of the others within ``radius`` in its frame."""
    from scipy.spatial import cKDTree
    out = np.zeros(len(frame), bool)
    if not len(frame) or not len(other_frame):
        return out
    # frame as a far third coordinate: no two frames are ever within reach
    scale = 10.0 * radius + 1e6
    tree = cKDTree(np.column_stack([other_xy, np.asarray(other_frame) * scale]))
    d, _ = tree.query(np.column_stack([xy, np.asarray(frame) * scale]),
                      distance_upper_bound=radius)
    return np.isfinite(d)


@dataclass
class Comparison:
    n_fitted: int
    n_counted: int                  # true spots that count
    tp: int
    fp: int
    fn: int
    set_aside: int                  # fits matched to a spot that does not count
    axes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    z_slope: Optional[float] = None
    z_intercept: Optional[float] = None
    # for the figures
    true_photons: np.ndarray = None     # of the counted spots
    found: np.ndarray = None            # whether each was matched
    pairs: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return self.tp / self.n_counted if self.n_counted else float("nan")

    @property
    def correct(self) -> float:
        """The fraction of the scored localizations that are a molecule."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def jaccard(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else float("nan")

    def summary(self) -> Dict[str, float]:
        return {"n_fitted": self.n_fitted, "n_true": self.n_counted, "tp": self.tp,
                "fp": self.fp, "fn": self.fn, "set_aside": self.set_aside,
                "recall": self.recall, "correct": self.correct,
                "jaccard": self.jaccard, "axes": self.axes,
                "z_slope": self.z_slope, "z_intercept": self.z_intercept}


def compare(fitted: Localizations, truth: Localizations, radius: float = 100.0,
            counted: Optional[np.ndarray] = None,
            reach: Optional[float] = None) -> Comparison:
    """Score ``fitted`` against ``truth``; ``counted`` (boolean over the
    truth) says which true spots count, all of them by default.  A
    localization left unmatched within ``reach`` (default ``radius``) of a
    spot that does not count is set aside rather than called false."""
    if "x_nm" not in fitted or "y_nm" not in fitted:
        raise ValueError("the comparison is in nanometres, and the table has "
                         f"{', '.join(sorted(fitted.keys()))}")
    counted = np.ones(len(truth), bool) if counted is None else np.asarray(counted, bool)
    fxy = np.column_stack([fitted["x_nm"], fitted["y_nm"]]).astype(float)
    txy = np.column_stack([truth["x_nm"], truth["y_nm"]]).astype(float)
    fi, ti = match(fxy, fitted["frame"], txy, truth["frame"], radius)
    scored = counted[ti]
    fi_s, ti_s = fi[scored], ti[scored]
    tp = len(fi_s)
    # a localization left over beside a spot that does not count -- the
    # second fit of a crowded pair, say -- is that spot's, not a false one
    left = np.setdiff1d(np.arange(len(fitted)), fi)
    beside = _near_any(fxy[left], np.asarray(fitted["frame"])[left],
                       txy[~counted], np.asarray(truth["frame"])[~counted],
                       radius if reach is None else reach)
    set_aside = int((~scored).sum() + beside.sum())
    fp = len(left) - int(beside.sum())
    n_counted = int(counted.sum())
    found = np.zeros(len(truth), bool)
    found[ti_s] = True
    result = Comparison(n_fitted=len(fitted), n_counted=n_counted, tp=tp, fp=fp,
                        fn=n_counted - tp, set_aside=set_aside,
                        true_photons=np.asarray(truth["photons"])[counted],
                        found=found[counted])
    names = [("x", "x_nm", "xy_err_nm"), ("y", "y_nm", "xy_err_nm")]
    if "z_nm" in fitted and "z_nm" in truth:
        names.append(("z", "z_nm", "z_err_nm"))
    for axis, col, err in names:
        d = np.asarray(fitted[col], float)[fi_s] - np.asarray(truth[col], float)[ti_s]
        stats = {"bias_nm": float(np.mean(d)) if tp else float("nan"),
                 "rmse_nm": float(np.sqrt(np.mean(d ** 2))) if tp else float("nan")}
        result.pairs[f"d{axis}"] = d
        if err in fitted and tp:
            e = np.asarray(fitted[err], float)[fi_s]
            ok = e > 0
            pull = d[ok] / e[ok]
            # robust: a mismatched pair should not decide the spread
            stats["pull"] = float(1.4826 * np.median(np.abs(pull - np.median(pull))))
            result.pairs[f"pull_{axis}"] = pull
            result.pairs[f"err_{axis}"] = e
        result.axes[axis] = stats
    if "photons" in fitted:
        result.pairs["photons"] = np.asarray(fitted["photons"], float)[fi_s]
    if "z" in result.axes and tp > 10:
        tz = np.asarray(truth["z_nm"], float)[ti_s]
        fz = np.asarray(fitted["z_nm"], float)[fi_s]
        if np.ptp(tz) > 0:
            result.z_slope, result.z_intercept = (float(v) for v in np.polyfit(tz, fz, 1))
            result.pairs["true_z"], result.pairs["fitted_z"] = tz, fz
    return result


# ----------------------------------------------------------------- figures
def draw_all(figure, c: Comparison) -> None:
    panels = 3 + (c.z_slope is not None)
    axes = figure.subplots(panels, 1, squeeze=False).ravel()
    _draw_recall(axes[0], c)
    _draw_error(axes[1], c)
    _draw_pull(axes[2], c)
    if c.z_slope is not None:
        _draw_z(axes[3], c)


def _draw_recall(ax, c: Comparison) -> None:
    p = c.true_photons
    if not len(p) or p.max() <= 0:
        ax.set_axis_off()
        return
    edges = np.geomspace(max(p[p > 0].min(), 1.0), p.max() * 1.001, 25)
    total, _ = np.histogram(p, edges)
    hit, _ = np.histogram(p[c.found], edges)
    centre = np.sqrt(edges[:-1] * edges[1:])
    ok = total > 0
    ax.semilogx(centre[ok], hit[ok] / total[ok], "o-", ms=3, lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("true photons in the frame")
    ax.set_ylabel("found")
    ax.set_title(f"recall {c.recall:.2f}, Jaccard {c.jaccard:.2f}", fontsize=9)


def _draw_error(ax, c: Comparison) -> None:
    """Measured lateral error against the claimed, by photons: the two lines
    lie on each other when the reported precision is the real one."""
    ph = c.pairs.get("photons")
    if ph is None or len(ph) < 20 or "err_x" not in c.pairs:
        ax.set_axis_off()
        return
    d = np.hypot(c.pairs["dx"], c.pairs["dy"]) / np.sqrt(2)
    edges = np.quantile(ph, np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(edges, ph, side="right") - 1, 0, 9)
    measured = [np.sqrt(np.mean(d[k == i] ** 2)) for i in range(10)]
    claimed = [np.median(c.pairs["err_x"][k == i]) for i in range(10)]
    centre = [np.median(ph[k == i]) for i in range(10)]
    ax.loglog(centre, measured, "o", ms=4, label="measured (RMS per axis)")
    ax.loglog(centre, claimed, "-", lw=1.2, label="reported precision")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("photons")
    ax.set_ylabel("lateral error (nm)")
    ax.legend(fontsize=7, frameon=False)


def _draw_pull(ax, c: Comparison) -> None:
    keys = [k for k in ("pull_x", "pull_y", "pull_z") if k in c.pairs]
    if not keys:
        ax.set_axis_off()
        return
    bins = np.linspace(-5, 5, 61)
    for key in keys:
        ax.hist(c.pairs[key], bins, histtype="step", density=True,
                label=f"{key[-1]}: {c.axes[key[-1]]['pull']:.2f}")
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8, label="N(0, 1)")
    ax.set_xlabel("error / reported precision")
    ax.legend(fontsize=7, frameon=False)


def _draw_z(ax, c: Comparison) -> None:
    tz, fz = c.pairs["true_z"], c.pairs["fitted_z"]
    ax.hist2d(tz, fz, bins=60, cmap="Greys", cmin=1)
    lo, hi = np.percentile(tz, [0.5, 99.5])
    x = np.array([lo, hi])
    ax.plot(x, x, "k--", lw=0.8)
    ax.plot(x, c.z_slope * x + c.z_intercept, "r-", lw=1)
    ax.set_xlabel("true z (nm)")
    ax.set_ylabel("fitted z (nm)")
    ax.set_title(f"slope {c.z_slope:.3f}", fontsize=9)


# ------------------------------------------------------------------ plugin
@dataclass
class GroundTruthSettings:
    truth: str = param("", label="simulation", kind="open_file", file_filter=SIM_FILTER,
                       help="auto: the simulation the table was fitted from, or the "
                            "one that made it")
    radius_nm: float = param(100.0, label="match within", unit="nm", min=0.1,
                             help="a localization and a true spot in one frame closer "
                                  "than this can be the same molecule; paired one to "
                                  "one")
    min_photons: float = param(0.0, label="count spots from", unit="photons", min=0,
                               help="true spots with fewer photons in the frame are "
                                    "not counted: a fluorophore on for a sliver of a "
                                    "frame is found by no fitter")
    isolated: bool = param(False, label="isolated spots only",
                           help="count only true spots with no other within 1 um in "
                                "their frame, and set aside what was fitted near the "
                                "rest: the accuracy without crowding")
    drift_corrected: bool = param(False, label="drift corrected",
                                  help="the table's drift has been taken out: compare "
                                       "with the truth without it")


@register("Analysis/Measure/Ground Truth")
class GroundTruth(Plugin):
    """A fit of a simulation against where the molecules really were."""

    Settings = GroundTruthSettings
    version = "1"

    def run(self, ctx: Context, settings: GroundTruthSettings) -> Result:
        from ..simulate import ISOLATED_NM
        ctx.report("working out the truth from the simulation...")
        truth = truth_for(ctx.locs, settings.truth)
        if settings.drift_corrected:
            drift = truth.metadata.get("drift_truth")
            if drift:
                shift = np.asarray(drift)[truth["frame"]]
                truth = Localizations({**truth.columns,
                                       "x_nm": truth["x_nm"] - shift[:, 0],
                                       "y_nm": truth["y_nm"] - shift[:, 1],
                                       "z_nm": truth["z_nm"] - shift[:, 2]},
                                      truth.metadata)
        fitted = ctx.selection.apply(ctx.locs)
        if not len(fitted):
            raise ValueError("no localizations selected to compare")
        # the truth where the table could have found it: its frames, its ROI
        frames = ctx.locs["frame"]
        counted = ((truth["frame"] >= frames.min()) & (truth["frame"] <= frames.max())
                   & (truth["photons"] >= settings.min_photons))
        roi = ctx.selection.roi
        if roi is not None and hasattr(roi, "mask"):
            counted &= roi.mask(truth["x_nm"], truth["y_nm"])
        if settings.isolated:
            counted &= truth["neighbour_nm"] > ISOLATED_NM
        ctx.report(f"matching {len(fitted)} localizations to {int(counted.sum())} "
                   f"true spots...")
        # a crowded pair is fitted as one spot between them, or twice: with
        # only the isolated counted, what lies within half the isolation
        # distance of a crowded spot is the crowding's, not a false fit
        reach = max(settings.radius_nm, ISOLATED_NM / 2) if settings.isolated else None
        c = compare(fitted, truth, settings.radius_nm, counted, reach)
        lines = [f"{c.n_fitted} localizations against {c.n_counted} true spots "
                 f"(of {len(truth)} drawn), matched within {settings.radius_nm:g} nm",
                 f"found {c.tp} (recall {c.recall:.3f}), false {c.fp} "
                 f"({1 - c.correct:.3f} of those scored), missed {c.fn}; "
                 f"Jaccard {c.jaccard:.3f}"]
        if c.set_aside:
            lines.append(f"{c.set_aside} localizations of spots that are not counted, set aside")
        for axis, s in c.axes.items():
            line = f"{axis}: bias {s['bias_nm']:+.2f} nm, RMS error {s['rmse_nm']:.2f} nm"
            if "pull" in s:
                line += f", error / reported precision {s['pull']:.2f}"
            lines.append(line)
        if c.z_slope is not None:
            lines.append(f"fitted z against true: slope {c.z_slope:.3f}, "
                         f"offset {c.z_intercept:+.1f} nm")

        def plot(figure) -> None:
            draw_all(figure, c)

        panels = 3 + (c.z_slope is not None)
        return Result(text="\n".join(lines), data={"comparison": c.summary()},
                      settings=settings,
                      plot=Plot(draw=plot, panels=panels, size=(5.5, 2.1 * panels)))
