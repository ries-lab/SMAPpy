"""Nuclear pore complexes: finding them, counting their corners, and the
labelling efficiency that follows.

Three plugins, one for each step of the ROI manager, ported from SMAP's
``segmentNPC`` with ``NPCsegmentCleanup``, ``NPCLabelingQuantify_s`` and
``NPCLabelingEfficiency`` (Thevathasan et al. 2019):

* **Segment regions**' circle template with a ring (`segment.py`, preset
  ``NPC``) filters a density image with a ring, fits a circle to every
  candidate and keeps the ones that look like a pore, by its *keep*
  condition (`smappy.segmentation`; this file only finds its ROIs again,
  `npc_rois`).
* **Evaluate/NPC Corners** counts, per pore, the corners seen with the
  precise localizations and the localizations it has -- and SMAP's corner
  count beside them.
* **Analyze/NPC labelling efficiency** fits the two jointly over all pores
  with the model of `smappy.npc`, or SMAP's corner histogram alone.

*Workflow/NPC Analysis*, which runs the three in one go, is not a plugin but
a chain of them, ``npc_analysis.chain.yaml`` beside this file: a chain runs an
evaluator over every ROI before its next step (`smappy.chain`).

Why it is not SMAP's corner counting, and why the settings are what they are,
is measured in ``studies/npc_le`` (its README has the numbers):

* Counting a corner from any localization in its segment is biased both ways
  -- imprecise localizations spill into empty neighbours and open them, and a
  precision filter loses the corners whose blinks were dim -- by up to 10
  points of efficiency at 5000 photons per blink and far more at 500.  The
  joint model counts with a cutoff (20 nm), models the spill, and takes the
  blinks from the localizations per pore, which is what extrapolates to the
  dim fluorophores the cutoff loses.
* The fit is conditioned on 5 corners or more.  The pores a segmentation
  loses are the sparse ones, and junk rarely shows 5 corners; fitting from 4
  left up to 18 points of bias with junk, from 5 about 11 at worst and from 6
  about 6, but 6 costs where few pores show that many corners (one blink).
* The NPC preset judges a candidate on its localizations better than the
  same cutoff: its fitted radius, and how far each lies from the ring in
  units of its own precision (``far_p``).  Fractions inside and outside
  counted on every localization reject sparse pores at low photon numbers
  (imprecise localizations land inside) and pass pore-sized filled blobs
  (the ring filter puts the site beside a filled blob, not on it, and from
  there it is an arc).  It asks ``n_ring >= 4`` -- a pore showing 4 corners
  has at least 4 -- since more loses the sparse pores that still show 5.
* Corners are counted from 40 nm out: nearer the centre a corner's segment is
  narrower than a localization's spread.

Smaller departures:

* The circle fits use a soft-L1 loss, so the background inside the window
  does not pull the centre; SMAP's ``fitposring`` is plain least squares.
* The rotation of a pore is the weighted circular mean of ``8 theta`` instead
  of a least-squares fit of a sawtooth: closed form, no start value.
* SMAP's analysis is kept as the *corners (SMAP)* method, its histogram
  fitted by maximum likelihood conditioned on the fit range, or by SMAP's
  least squares on the square root of the histogram.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import Context, Plot, Plugin, Result, param, register, settings_from
from .roi import EVALUATION_HELP, SiteAnalysisPlugin, auto_only
# the circle fit and the ring filter are the segmenter's, in `smappy.segmentation`
from ..segmentation import fit_circle, ring_kernel  # noqa: F401

# SMAP's NPCLabelingQuantify counts a localization towards a corner only when
# its precision is below 0.4 of the arc between two corners (15.7 nm at 50 nm)
PRECISION_OF_ARC = 0.4
SMAP_RADIUS_NM, SMAP_RING_WIDTH_NM = 50.0, 20.0
# fixed-radius fits of a pore's centre: the second is on the window around
# the first answer, which an ROI centre a few nm off can need
FIT_PASSES = 2


def precision_column(locs) -> Optional[str]:
    return next((n for n in ("xy_err_nm", "x_err_nm") if n in locs), None)


# --------------------------------------------------------------- segmenting

def npc_rois(project, file_id) -> List[Any]:
    """The ROIs the circle template of *Segment regions* made in this file."""
    from .segment import made_by
    return [roi for roi in project.rois.values()
            if roi.file_id == file_id and made_by(roi, "circle")]


# ---------------------------------------------------------------- corners

@dataclass
class NPCCornersSettings:
    radius_nm: float = param(55.0, label="radius", unit="nm", min=1.0,
                             help="radius of the ring band the corners are counted in")
    ring_width_nm: float = param(15.0, label="ring width", unit="nm", min=1.0,
                                 help="half the width of the ring band")
    precision_nm: float = param(20.0, label="corner precision", unit="nm", min=0.1,
                                help="corners are counted with the localizations "
                                     "more precise than this")
    n_precision_nm: float = param(30.0, label="count precision", unit="nm", min=0.1,
                                  help="the localizations per pore are those more "
                                       "precise than this")
    n_window_nm: float = param(100.0, label="count window", unit="nm", min=1.0,
                               help="... and within this distance of the centre")
    corners: int = param(8, label="corners", min=2,
                         help="symmetry of the pore: segments around the ring")
    min_locs: int = param(1, label="min per corner", min=1, advanced=True,
                          help="localizations a corner needs to count as seen")


def segments(theta, weight, corners: int, min_locs: int):
    """``(seen, rotation, per corner)``: the rotation the weighted circular mean
    of ``corners * theta``, each localization in the segment centred on the
    nearest corner, a segment with `min_locs` a corner seen."""
    if not len(theta):
        return 0, float("nan"), [0] * corners
    step = 2 * np.pi / corners
    phase = np.angle(np.sum(weight * np.exp(1j * corners * theta))) / corners
    segment = np.floor(np.mod(theta - phase + step / 2, 2 * np.pi) / step).astype(int)
    per_corner = np.bincount(segment, minlength=corners)[:corners]
    return int(np.sum(per_corner >= min_locs)), float(phase), per_corner.tolist()


def count_corners(x, y, center, settings: NPCCornersSettings, precision=None
                  ) -> Dict[str, Any]:
    """The corners a pore shows and the localizations it has.

    The centre is fitted with the radius fixed on the localizations better
    than *corner precision* (all of them, without a precision), then twice
    more on those within ``R + 2 dR``.  ``n_corners``: the corners seen with
    those precise localizations in the band ``R +- dR``.  ``n_localizations``:
    the localizations better than *count precision* within *count window*.
    ``n_corners_smap``: SMAP's count, every localization in 50 +- 20 nm better
    than 0.4 of the arc between corners, around the same centre.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    R, dR, n = settings.radius_nm, settings.ring_width_nm, int(settings.corners)
    sigma = None if precision is None else np.asarray(precision, dtype=float)
    sharp = np.ones(len(x), bool) if sigma is None else sigma < settings.precision_nm
    fit_on = sharp if sharp.sum() >= 3 else np.ones(len(x), bool)
    c = np.asarray(fit_circle(x[fit_on], y[fit_on], center, radius=R, scale=dR)[:2])
    for _ in range(FIT_PASSES):
        close = fit_on & (np.hypot(x - c[0], y - c[1]) < R + 2 * dR)
        if close.sum() >= 3:
            c = np.asarray(fit_circle(x[close], y[close], c, radius=R, scale=dR)[:2])
    rho = np.hypot(x - c[0], y - c[1])
    theta = np.arctan2(y - c[1], x - c[0])
    band = sharp & (rho >= R - dR) & (rho <= R + dR)
    weight = (np.ones(len(x)) if sigma is None
              else 1.0 / np.maximum(sigma / np.maximum(rho, 1e-9), 1e-6) ** 2)
    seen, phase, per_corner = segments(theta[band], weight[band], n, settings.min_locs)
    counted = (np.ones(len(x), bool) if sigma is None
               else sigma < settings.n_precision_nm) & (rho < settings.n_window_nm)
    # SMAP's rule, around the same centre
    smap = (rho > SMAP_RADIUS_NM - SMAP_RING_WIDTH_NM) & (rho < SMAP_RADIUS_NM
                                                         + SMAP_RING_WIDTH_NM)
    if sigma is not None:
        smap &= sigma < PRECISION_OF_ARC * 2 * np.pi / n * SMAP_RADIUS_NM
    seen_smap, _, _ = segments(theta[smap], weight[smap], n, settings.min_locs)
    radius = (fit_circle(x[band], y[band], c, scale=dR)[2] if band.sum() >= 3
              else float("nan"))
    return {"n_corners": seen, "n_localizations": int(counted.sum()),
            "n_corners_smap": seen_smap, "n_ring_locs": int(band.sum()),
            "radius_nm": float(radius),
            "ring_radius_nm": float(np.median(rho[band])) if band.any() else float("nan"),
            "rotation_deg": float(np.degrees(phase)),
            "x_nm": float(c[0]), "y_nm": float(c[1]), "per_corner": per_corner}


def draw_corners(ax, x, y, values: Dict[str, Any], settings: NPCCornersSettings,
                 precision=None) -> None:
    """The site, the ring band, and the segment borders; the localizations
    that were counted in colour."""
    from matplotlib.patches import Circle
    R, dR, n = settings.radius_nm, settings.ring_width_nm, int(settings.corners)
    x0, y0 = values["x_nm"], values["y_nm"]
    rho = np.hypot(x - x0, y - y0)
    ring = (rho >= R - dR) & (rho <= R + dR)
    if precision is not None:
        ring &= np.asarray(precision) < settings.precision_nm
    ax.scatter(x[~ring] - x0, y[~ring] - y0, s=5, c="0.7", linewidths=0)
    ax.scatter(x[ring] - x0, y[ring] - y0, s=7, c="#2f7fd0", linewidths=0)
    for r, style in ((R, "-"), (R - dR, ":"), (R + dR, ":")):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=style, color="k", lw=0.7))
    if np.isfinite(values["rotation_deg"]):
        phase = np.radians(values["rotation_deg"])
        seen = np.asarray(values["per_corner"]) >= settings.min_locs
        for k in range(n):
            border = phase + (k + 0.5) * 2 * np.pi / n
            ax.plot([0, 1.5 * R * np.cos(border)], [0, 1.5 * R * np.sin(border)],
                    color="0.5", lw=0.6)
            corner = phase + k * 2 * np.pi / n
            ax.plot(R * np.cos(corner), R * np.sin(corner), "o", ms=7, mew=1.2,
                    mfc="#d62728" if seen[k] else "none", mec="#d62728")
    lim = 1.6 * R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{values['n_corners']} of {n} corners, "
                 f"{values['n_localizations']} localizations")


@register("Analysis/Measure/NPC Corners")
class NPCCorners(Plugin):
    """Counts the corners a nuclear pore shows and the localizations it has."""

    Settings = NPCCornersSettings
    changes_locs = False
    per_region = True
    version = "2"

    def run(self, ctx: Context, settings: NPCCornersSettings) -> Result:
        table, selection = ctx.table()
        locs = selection.apply(table)
        x = np.asarray(locs["x_nm"], dtype=float)
        y = np.asarray(locs["y_nm"], dtype=float)
        if not len(x):
            raise ValueError("no localizations in this ROI")
        column = precision_column(locs)
        precision = None if column is None else np.asarray(locs[column], dtype=float)
        center = (ctx.region.center.tolist() if ctx.region is not None
                  else [float(np.mean(x)), float(np.mean(y))])
        values = count_corners(x, y, center, settings, precision)
        data = {k: v for k, v in values.items() if k != "per_corner"}
        return Result(text=f"{values['n_corners']} corners, "
                           f"{values['n_localizations']} localizations",
                      data=data, values=data,
                      arrays={"per_corner": np.asarray(values["per_corner"])},
                      plot=lambda ax: draw_corners(ax, x, y, values, settings, precision),
                      settings=settings)


# ----------------------------------------------------- labelling efficiency

def corner_probability(efficiency, per_corner: int):
    """The chance that a corner shows: at least one of its proteins labelled."""
    return 1.0 - (1.0 - np.asarray(efficiency, dtype=float)) ** per_corner


def corner_model(efficiency: float, corners: int, per_corner: int) -> np.ndarray:
    """``P(k)``, k = 0..corners: corners seen, each independently."""
    from scipy.stats import binom
    return binom.pmf(np.arange(corners + 1), corners,
                     corner_probability(efficiency, per_corner))


def fit_likelihood(counts: np.ndarray, corners: int, per_corner: int,
                   lo: int, hi: int) -> float:
    """The efficiency that maximises the likelihood of the histogram within
    ``lo..hi``, the binomial renormalised to that range."""
    from scipy.optimize import minimize_scalar
    k = np.arange(lo, hi + 1)
    h = np.asarray(counts, dtype=float)[k]

    def negative(p):
        model = corner_model(p, corners, per_corner)[k]
        model = np.maximum(model / max(model.sum(), 1e-300), 1e-300)
        return -float(np.sum(h * np.log(model)))
    return float(minimize_scalar(negative, bounds=(1e-4, 1 - 1e-4),
                                 method="bounded", options={"xatol": 1e-6}).x)


def fit_sqrt_lsq(counts: np.ndarray, corners: int, per_corner: int,
                 lo: int, hi: int) -> float:
    """SMAP's ``fitNPClabeling``: least squares of ``sqrt(A P(k))`` against the
    square root of the histogram within ``lo..hi``, from ``A`` = the number of
    sites and an efficiency of 0.4."""
    from scipy.optimize import least_squares
    k = np.arange(lo, hi + 1)
    h = np.asarray(counts, dtype=float)

    def residual(p):
        return np.sqrt(p[0] * corner_model(p[1], corners, per_corner)[k]) - np.sqrt(h[k])
    out = least_squares(residual, [h.sum(), 0.4], bounds=([0, 0], [np.inf, 1]))
    return float(out.x[1])


FITS = {"likelihood": fit_likelihood, "sqrt least squares (SMAP)": fit_sqrt_lsq}


METHODS = (("joint", "corners and localizations"), ("smap", "corners (SMAP)"))


@dataclass
class LabelingEfficiencySettings:
    method: str = param("joint", label="method", choices=METHODS,
                        help="the joint model of corners and localizations, or "
                             "SMAP's corner histogram alone")
    fit_min: int = param(5, label="fit from", min=0,
                         help="only pores with at least this many corners")
    evaluation: str = param("", label="results from", choices=auto_only,
                            help=EVALUATION_HELP)
    corners: int = param(8, label="corners", min=2, advanced=True,
                         help="corners of a pore")
    per_corner: int = param(4, label="proteins per corner", min=1, advanced=True,
                            help="copies of the labelled protein in one corner")
    fit_max: int = param(8, label="fit to", min=0, advanced=True,
                         help="most corners in the fit range (SMAP's method)")
    fit: str = param("likelihood", label="SMAP fit", advanced=True,
                     choices=(("likelihood", "maximum likelihood"),
                              ("sqrt least squares (SMAP)", "SMAP's least squares")),
                     help="how SMAP's method fits its histogram")
    bootstrap: int = param(100, label="bootstrap", min=0, advanced=True,
                           help="resamplings of the pores for SMAP's error; 0: none")


def labeling_efficiency(n_corners, settings: LabelingEfficiencySettings,
                        seed: int = 0) -> Dict[str, Any]:
    """The effective labelling efficiency from the corners seen per pore.

    ``{"efficiency", "error", "counts", "n_sites", "corner_probability"}``;
    the error is the standard deviation over ``bootstrap`` resamplings of the
    sites, with a fixed seed so a rerun gives the same number.
    """
    n = np.asarray(n_corners, dtype=float)
    n = n[np.isfinite(n)].astype(int)
    corners, per = int(settings.corners), int(settings.per_corner)
    lo, hi = int(settings.fit_min), min(int(settings.fit_max), corners)
    if lo > hi:
        raise ValueError("the fit range is empty: 'fit from' is above 'fit to'")
    if np.any((n < 0) | (n > corners)):
        raise ValueError(f"corner counts outside 0..{corners}: is 'corners' right?")
    fit = FITS[settings.fit]
    counts = np.bincount(n, minlength=corners + 1)
    in_range = int(counts[lo:hi + 1].sum())
    if in_range < 5:
        raise ValueError(f"{in_range} pores in the fit range: at least 5 are needed")
    efficiency = fit(counts, corners, per, lo, hi)
    error = float("nan")
    if settings.bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = [fit(np.bincount(rng.choice(n, len(n)), minlength=corners + 1),
                     corners, per, lo, hi) for _ in range(int(settings.bootstrap))]
        error = float(np.std(draws))
    return {"efficiency": efficiency, "error": error, "counts": counts,
            "n_sites": int(len(n)), "n_fitted": in_range,
            "corner_probability": float(corner_probability(efficiency, per))}


def joint_efficiency(k, n, sigma_field, corner_settings: NPCCornersSettings,
                     ring_radius: float, fit_min: int = 5, per_corner: int = 4
                     ) -> Dict[str, Any]:
    """The labelled efficiency and the blinks per copy from the corners `k`
    and localizations `n` of each pore (NPC Corners), with the precisions
    `sigma_field` of every localization of the field -- `smappy.npc`."""
    from .. import npc as model
    cs = corner_settings
    a, b, eps = model.localization_classes(
        sigma_field, ring_radius, (cs.radius_nm - cs.ring_width_nm,
                                   cs.radius_nm + cs.ring_width_nm),
        cs.precision_nm, cs.n_precision_nm, cs.n_window_nm, int(cs.corners))
    out = model.fit(k, n, a, b, eps, k_min=fit_min, corners=int(cs.corners),
                    per_corner=per_corner)
    out.update(a=a, b=b, eps=eps, k=np.asarray(k, int), n=np.asarray(n, int))
    return out


def true_corners(locs, corners: int = 8) -> Dict[int, int]:
    """For a simulated table: how many corners of each pore were labelled.

    The labelled molecules are drawn again from the recipe the table carries
    (as Ground Truth does), and each one is put in a corner by its angle
    around its pore's centre, with the pore's rotation found the way
    `count_corners` finds it -- on exact positions, where it cannot miss.
    ``{copy: corners labelled}``.
    """
    from ..simulate import ground_truth
    from ..simulate.scene import settings_of
    recipe = locs.metadata.get("simulation_settings")
    copies = locs.metadata.get("copies")
    if not recipe or not copies:
        raise ValueError("this table is not a simulation")
    if "components" in recipe and "x_nm" not in copies:
        raise ValueError("a scene of several components: label corners need one")
    truth = ground_truth(settings_of(recipe))
    f = truth.fluorophores
    cx = np.asarray(copies["x_nm"], dtype=float)
    cy = np.asarray(copies["y_nm"], dtype=float)
    step = 2 * np.pi / corners
    out = {}
    for copy in range(len(cx)):
        mine = f.copy == copy
        if not mine.any():
            out[copy] = 0
            continue
        theta = np.arctan2(f.xyz[mine, 1] - cy[copy], f.xyz[mine, 0] - cx[copy])
        phase = np.angle(np.sum(np.exp(1j * corners * theta))) / corners
        segment = np.floor(np.mod(theta - phase + step / 2, 2 * np.pi) / step)
        out[copy] = int(len(np.unique(segment)))
    return out


def site_truth(project, rows, corners: int) -> Optional[np.ndarray]:
    """The labelled corners of the pore under each site, or None when the
    data are not simulated.  A site's pore is the ``copy`` most of its
    localizations belong to."""
    tables = {}
    out = []
    for row in rows:
        roi = project.roi_numbered(row["roi_id"])
        if roi is None:
            return None
        locs = project.extract(roi)
        if "copy" not in locs or not len(locs):
            return None
        if roi.file_id not in tables:
            try:
                tables[roi.file_id] = true_corners(project.state(roi.file_id).locs,
                                                   corners)
            except ValueError:
                return None
        copy = np.asarray(locs["copy"]).astype(int)
        copy = copy[copy >= 0]
        if not len(copy):
            out.append(np.nan)
            continue
        out.append(tables[roi.file_id].get(int(np.bincount(copy).argmax()), np.nan))
    return np.asarray(out, dtype=float)


def simulated_efficiency(project, rows) -> Optional[float]:
    """The labelling efficiency the sites' file was simulated with, if it was."""
    roi = project.roi_numbered(rows[0]["roi_id"]) if rows else None
    if roi is None:
        return None
    recipe = project.state(roi.file_id).locs.metadata.get("simulation_settings") or {}
    if recipe.get("components"):            # a scene: its first component's
        recipe = recipe["components"][0]
    value = (recipe.get("labelling") or {}).get("efficiency")
    return None if value is None else float(value)


CORNERS = "Analysis/Measure/NPC Corners"


def column(rows, name: str, label: Optional[str] = None) -> Optional[str]:
    """`name` as the site table spells it: plain while one evaluation writes
    it, ``"<label>.<name>"`` when two do -- NPC Corners and Statistics both
    write `n_localizations`, two NPC Corners under two names write all."""
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    if label and f"{label}.{name}" in keys:
        return f"{label}.{name}"
    if name in keys:
        return name
    if label:
        return None
    return next((k for k in sorted(keys) if k.endswith("." + name)), None)


def corner_column(rows) -> Optional[str]:
    return column(rows, "n_corners")


def corner_settings(ctx: Context, evaluation, rows=()) -> NPCCornersSettings:
    """The settings the NPC Corners evaluation ``evaluation`` (an
    `region_results.Evaluation`, or its name) counted these sites with -- the
    cutoffs the counts mean something with -- as its run recorded them, not
    whatever the evaluation window holds now; on the ROIs or per cluster."""
    from ..region_results import Evaluation, steps
    if not evaluation:
        return NPCCornersSettings()
    ev = evaluation if isinstance(evaluation, Evaluation) else Evaluation.parse(evaluation)
    ids = [r["roi_id"] for r in rows if "roi_id" in r]
    if ev.on_rois:            # a row names its ROI by number, the store by id
        found = [ctx.rois.roi_numbered(i) if ctx.rois is not None else None
                 for i in ids]
        ids = [roi.id for roi in found if roi is not None]
    made = steps(ctx, ev, ids or None)
    if len(made) > 1:
        raise ValueError(f"the pores were counted with different settings under "
                         f"{str(ev)!r}: measure them all again")
    values = (made[0].get("parameters") if made else None) or {}
    return settings_from(NPCCornersSettings, values)


def field_precisions(project, rows) -> np.ndarray:
    """The precision of every localization of the files the sites are in, as
    the layer filters them -- what a cutoff's pass fraction is measured on."""
    out = []
    found = (project.roi_numbered(r["roi_id"]) for r in rows)
    for file_id in {roi.file_id for roi in found if roi is not None}:
        state = project.state(file_id)
        locs = state.locs[state.filter.indices]
        name = precision_column(locs)
        if name is None:
            raise ValueError("the joint model needs the localization precision "
                             "(xy_err_nm); use the corners (SMAP) method")
        out.append(np.asarray(locs[name], float))
    if not out:
        raise ValueError("the pores' files are not in the ROI manager")
    return np.concatenate(out)


def draw_histogram(ax, fitted: Dict[str, Any], settings: LabelingEfficiencySettings,
                   truth: Optional[Dict[str, Any]] = None) -> None:
    """The corners seen and the model at the fitted efficiency, fit range
    marked; the labelled corners of a simulation beside them."""
    corners = int(settings.corners)
    k = np.arange(corners + 1)
    counts = fitted["counts"]
    width = 0.4 if truth else 0.8
    ax.bar(k - (width / 2 if truth else 0), counts, width=width, color="0.7",
           label="measured")
    lo, hi = int(settings.fit_min), min(int(settings.fit_max), corners)
    model = corner_model(fitted["efficiency"], corners, int(settings.per_corner))
    scale = fitted["n_fitted"] / max(model[lo:hi + 1].sum(), 1e-300)
    ax.plot(k, scale * model, "-", color="#d62728", lw=1)
    ax.plot(k[lo:hi + 1], scale * model[lo:hi + 1], "o", color="#d62728", ms=4,
            label=f"fit: {100 * fitted['efficiency']:.1f} %")
    if truth:
        ax.bar(k + width / 2, truth["counts"], width=width, color="#2f7fd0",
               label=f"labelled (truth): {100 * truth['efficiency']:.1f} %")
    ax.set_xlabel("corners seen")
    ax.set_ylabel("pores")
    ax.set_xticks(k)
    ax.legend(fontsize=7, frameon=False)


def draw_joint(figure, fitted: Dict[str, Any], truth=None) -> None:
    """The two histograms the joint fit explains, over the pores it used, and
    the model's at the fitted efficiency and blinks."""
    from .. import npc as model
    k_min = fitted["k_min"]
    use = fitted["k"] >= k_min
    k, n = fitted["k"][use], fitted["n"][use]
    corners_model, n_model = model.marginals(fitted["pmf"], k_min)
    left, right = figure.subplots(1, 2, gridspec_kw={"wspace": 0.35})
    kk = np.arange(len(corners_model))
    left.bar(kk, np.bincount(k, minlength=len(kk)), color="0.7", label="pores")
    left.plot(kk[k_min:], len(k) * corners_model[k_min:], "o-", color="#d62728", ms=4,
              lw=1, label=f"model: {100 * fitted['efficiency']:.1f} %")
    if truth is not None:
        left.set_title(f"labelled (truth): {100 * truth:.1f} %", fontsize=8)
    left.set_xlabel("corners seen")
    left.set_ylabel("pores")
    left.set_xticks(kk)
    left.legend(fontsize=7, frameon=False)
    top = int(max(n.max(), 1))
    width = max(1, int(np.ceil((top + 1) / 40)))
    edges = np.arange(0, top + width + 1, width)
    counts, _ = np.histogram(n, bins=edges)
    right.bar(edges[:-1], counts, width=width, align="edge", color="0.7")
    padded = np.zeros(edges[-1])
    length = min(len(n_model), edges[-1])
    padded[:length] = n_model[:length]
    per_bin = padded.reshape(-1, width).sum(axis=1)
    right.plot(edges[:-1] + width / 2, len(n) * per_bin, "-", color="#d62728", lw=1,
               label=f"{fitted['blinks']:.2f} blinks per copy")
    right.set_xlabel("localizations per pore")
    right.set_ylabel("pores")
    right.legend(fontsize=7, frameon=False)


@register("Analysis/Regions/Summarize/NPC labelling efficiency")
class NPCLabelingEfficiency(SiteAnalysisPlugin):
    """The labelling efficiency of nuclear pores, from the corners each shows
    and the localizations it has (NPC Corners)."""

    Settings = LabelingEfficiencySettings
    changes_locs = False
    version = "2"

    evaluator = CORNERS

    def run(self, ctx: Context, settings: LabelingEfficiencySettings) -> Result:
        rows, ev = self.rows(ctx, settings)
        if not rows:
            raise ValueError("the ROIs have no corner counts: measure them with "
                             "NPC Corners first")
        count = corner_settings(ctx, ev, rows)
        label = None if ev is None else ev.label
        # per cluster there are no ROIs: no files whose field the precisions
        # come from (`field_precisions`), no shapes to find a pore's truth in
        # (`site_truth`) -- said here rather than as a missing ROI further down
        per_cluster = ev is not None and not ev.on_rois
        project = None if per_cluster else ctx.rois
        if per_cluster and settings.method != "smap":
            raise ValueError(
                f"the joint model needs the ROIs' files, for the precisions of the "
                f"whole field, and {str(ev)!r} was counted per region of "
                f"{ev.source}, not in ROIs: use the corners (SMAP) method, or make "
                f"ROIs of the clusters (Segment regions, from an id column) and count "
                f"their corners there")
        result = self.analyse(project, rows, settings, count, label)
        if per_cluster and ctx.locs.metadata.get("simulation_settings"):
            result.text += ("; not compared with the simulation's labelled corners, "
                            "which are found through ROIs")
        if ev is not None and str(ev) != CORNERS.rsplit("/", 1)[-1]:
            result.text += f"; counted by {str(ev)!r}"
        if settings.method != "smap" and getattr(ctx.rois, "grouped", True) is False:
            result.text += ("; the layer does not merge blinks: the model expects one "
                            "localization per blink")
        return result

    def analyse(self, project, rows, settings: LabelingEfficiencySettings,
                count: NPCCornersSettings, label: Optional[str] = None) -> Result:
        """The analysis of these site rows, counted with `count` by the NPC
        Corners evaluation called `label` (None: whichever the rows have)."""
        name = column(rows, "n_corners_smap" if settings.method == "smap" else "n_corners",
                      label)
        if name is None:
            raise ValueError("the region results have no corner counts: add NPC "
                             "Corners to the per-region steps and run them")
        k = np.array([r.get(name, np.nan) for r in rows], dtype=float)
        truth = None
        known = site_truth(project, rows, int(settings.corners)) \
            if project is not None else None
        if known is not None and np.isfinite(known).sum() >= 5:
            try:
                truth = labeling_efficiency(
                    known, LabelingEfficiencySettings(corners=settings.corners,
                                                      per_corner=settings.per_corner,
                                                      fit_min=0, bootstrap=0))
            except ValueError:
                truth = None
        if settings.method == "smap":
            result = self._smap(k, settings, truth)
        else:
            result = self._joint(project, rows, k, settings, truth, count, label)
        simulated = simulated_efficiency(project, rows) if project is not None else None
        if simulated is not None:
            result.data["simulated_efficiency"] = simulated
            result.text += f"; simulated with {100 * simulated:.0f} %"
        if truth is not None:
            result.data["true_efficiency"] = truth["efficiency"]
            result.data["true_counts"] = truth["counts"].tolist()
            result.text += (f"; the same pores' labelled corners give "
                            f"{100 * truth['efficiency']:.1f} %")
        return result

    def _joint(self, project, rows, k, settings, truth, count, label=None) -> Result:
        n_name = column(rows, "n_localizations", label)
        if n_name is None:
            raise ValueError("the region results have no n_localizations: measure again "
                             "with this version of NPC Corners")
        n = np.array([r.get(n_name, np.nan) for r in rows], dtype=float)
        ok = np.isfinite(k) & np.isfinite(n)
        radius_name = column(rows, "ring_radius_nm", label)
        radii = np.array([r.get(radius_name, np.nan) for r in rows], float) \
            if radius_name else np.array([np.nan])
        ring_radius = float(np.nanmedian(radii)) if np.isfinite(radii).any() else 53.7
        if project is None:
            raise ValueError("the joint model needs the ROI manager's files, for the "
                             "precisions of the whole field")
        fitted = joint_efficiency(k[ok].astype(int), n[ok].astype(int),
                                  field_precisions(project, rows), count, ring_radius,
                                  int(settings.fit_min), int(settings.per_corner))
        text = (f"labelling efficiency {100 * fitted['efficiency']:.1f} "
                f"± {100 * fitted['error']:.1f} %, {fitted['blinks']:.2f} "
                f"± {fitted['blinks_error']:.2f} blinks per copy, from "
                f"{fitted['pores']} pores with {settings.fit_min} corners or more "
                f"(of {int(ok.sum())})")
        data = {key: fitted[key] for key in ("efficiency", "error", "blinks",
                                             "blinks_error", "pores", "a", "b", "eps")}
        data["counts"] = np.bincount(k[ok].astype(int),
                                     minlength=int(settings.corners) + 1).tolist()
        true = None if truth is None else truth["efficiency"]
        return Result(text=text, data=data, settings=settings,
                      plot=Plot(lambda f: draw_joint(f, fitted, true), panels=2,
                                size=(8, 3.4)))

    def _smap(self, k, settings, truth) -> Result:
        fitted = labeling_efficiency(k, settings)
        text = (f"labelling efficiency (SMAP) {100 * fitted['efficiency']:.1f} "
                f"± {100 * fitted['error']:.1f} % from {fitted['n_sites']} pores "
                f"({fitted['n_fitted']} in the fit range)")
        data = {"efficiency": fitted["efficiency"], "error": fitted["error"],
                "n_sites": fitted["n_sites"], "counts": fitted["counts"].tolist(),
                "corner_probability": fitted["corner_probability"]}
        return Result(text=text, data=data, settings=settings,
                      plot=lambda ax: draw_histogram(ax, fitted, settings, truth))
