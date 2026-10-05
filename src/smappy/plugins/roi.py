"""The ROI manager's plugins: finding sites, measuring one, summarising many.

Three kinds, all ordinary plugins:

* **Segment** proposes candidate ROIs on a whole file.  It runs once.
* **Evaluate** measures *one* ROI and returns that site's row.  It declares
  ``scope = "site"``, which is the only thing that marks it as an evaluator --
  a property of the plugin, not of its folder, so the evaluation window can
  filter correctly with one shared plugin tree.
* **Analyze** reads the table evaluation produced and summarises it.  It runs
  once, over ``ctx.site_table``.

The algorithms stay module-level functions so a script can call them without a
plugin, a session or a project.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import (Context, Plot, Plugin, PreflightChoice, PreflightQuestion, Result,
               param, register)

MAX_FINDER_PIXELS = 16_000_000


# ------------------------------------------------------------ segmentation

@dataclass
class DensityPeakSettings:
    bin_nm: float = param(20.0, label="bin", unit="nm", min=0.1,
                          help="pixel size of the density image the peaks are found in")
    sigma_nm: float = param(50.0, label="smoothing", unit="nm", min=0.1,
                            help="Gaussian blur applied before looking for maxima")
    separation_nm: float = param(150.0, label="separation", unit="nm", min=0.1,
                                 help="reject a peak this close to one already accepted")
    count_radius_nm: float = param(75.0, label="count radius", unit="nm", min=0.1,
                                   help="localizations within this radius must reach "
                                        "the minimum count")
    min_count: int = param(10, label="minimum count", min=1,
                           help="a peak with fewer localizations is not a site")


def density_peaks(locs, settings: DensityPeakSettings,
                  existing: Sequence = ()) -> List[List[float]]:
    """Candidate centres, brightest first, no two closer than the separation."""
    from scipy.ndimage import gaussian_filter, label, find_objects, maximum_filter
    from scipy.spatial import cKDTree

    for name in ("bin_nm", "sigma_nm", "separation_nm", "count_radius_nm", "min_count"):
        if not getattr(settings, name) > 0:
            raise ValueError(f"{name} must be positive")
    xy = np.column_stack((locs["x_nm"], locs["y_nm"])).astype(float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < settings.min_count:
        return []
    pixel = float(settings.bin_nm)
    # Padding keeps peaks at the data extent from being reflected/omitted.
    pad = 4 * settings.sigma_nm + pixel
    origin = np.floor((xy.min(axis=0) - pad) / pixel) * pixel
    shape = np.ceil((xy.max(axis=0) + pad - origin) / pixel).astype(int) + 1
    if int(shape[0]) * int(shape[1]) > MAX_FINDER_PIXELS:
        raise ValueError("finder image exceeds 16 million pixels; increase the bin size")
    bins = np.floor((xy - origin) / pixel).astype(int)
    density = np.zeros(tuple(shape[::-1]), dtype=float)
    np.add.at(density, (bins[:, 1], bins[:, 0]), 1)
    density = gaussian_filter(density, settings.sigma_nm / pixel, mode="constant")
    # Small local-max neighborhood, followed by radial suppression in nm.
    maxima = (density == maximum_filter(density, size=3, mode="constant")) & (density > 0)
    labels, _ = label(maxima)
    peaks = []
    for index, region in enumerate(find_objects(labels), 1):
        if region is None:
            continue
        yy, xx = np.nonzero(labels[region] == index)
        yy, xx = yy + region[0].start, xx + region[1].start
        peaks.append((float(density[yy[0], xx[0]]), float(xx.mean()), float(yy.mean())))
    peaks.sort(key=lambda v: (-v[0], v[2], v[1]))
    tree = cKDTree(xy)
    accepted = [np.asarray(c, dtype=float) for c in existing]
    found = []
    for _, x, y in peaks:
        center = origin + (np.array([x, y]) + 0.5) * pixel
        neighbors = tree.query_ball_point(center, settings.count_radius_nm)
        if len(neighbors) < settings.min_count:
            continue
        # Recenter using the compact structure, not the full analysis ROI.
        center = xy[neighbors].mean(axis=0)
        if len(tree.query_ball_point(center, settings.count_radius_nm)) < settings.min_count:
            continue
        if any(np.linalg.norm(center - other) < settings.separation_nm
               for other in accepted):
            continue
        accepted.append(center)
        found.append(center.tolist())
    return found


@register("ROIManager/Segment/Density Peaks")
class DensityPeaks(Plugin):
    """Proposes ROIs where localizations pile up, after a Gaussian blur."""

    Settings = DensityPeakSettings
    version = "1"

    def propose(self, locs, settings: DensityPeakSettings,
                existing: Sequence = ()) -> List[List[float]]:
        """The centres alone.  `ROIProject.find` calls this and records the
        provenance, so `run` can go through the project without recursing."""
        return density_peaks(locs, settings, existing)

    def run(self, ctx: Context, settings: DensityPeakSettings) -> Result:
        project = ctx.rois
        if project is None:                 # no session: just report the centres
            centers = density_peaks(ctx.locs, settings)
            return Result(text=f"{len(centers)} candidates",
                          data={"centers": centers}, settings=settings)
        file_id = current_file(project)
        if file_id is None:
            raise ValueError("no file to search: load localizations first")
        found = project.find(file_id, plugin=self, settings=settings)
        return Result(text=f"{len(found)} candidates found", settings=settings,
                      data={"rois": found, "centers": [r.center for r in found]})


def current_file(project) -> Optional[str]:
    """The file the manager is showing, or the only one there is."""
    chosen = project.navigation.get("file")
    if chosen in project.sources:
        return chosen
    return next(iter(project.sources), None)


# -------------------------------------------------------------- evaluation

@dataclass
class StatisticsSettings:
    precision_column: str = param(
        "xy_err_nm", label="precision column",
        help="the column averaged as the site's localization precision")


def precision_name(column: str) -> str:
    """The result's name for a precision column, in that column's unit: a
    pixel table's ``xy_err_pix`` averages to pixels, not nanometres."""
    return "mean_precision_pix" if column.endswith("_pix") else "mean_precision_nm"


def site_statistics(locs, settings: StatisticsSettings) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_localizations": len(locs)}
    for column, name in ((settings.precision_column,
                          precision_name(settings.precision_column)),
                         ("photons", "mean_photons")):
        if column not in locs:
            raise ValueError(f"statistics requires {column}")
        values = np.asarray(locs[column], dtype=float)
        finite = values[np.isfinite(values)]
        out[name] = float(finite.mean()) if len(finite) else float("nan")
        out[name + "_n"] = len(finite)
    return out


@register("ROIManager/Evaluate/Statistics")
class Statistics(Plugin):
    """Counts a site's localizations and averages their precision and photons."""

    Settings = StatisticsSettings
    scope = "site"
    version = "1"

    def run(self, ctx: Context, settings: StatisticsSettings) -> Result:
        values = site_statistics(ctx.locs, settings)

        def plot(ax) -> None:
            """The site itself, as the numbers' sanity check.

            The ROI manager redraws this in the same window for every site, so
            what it is for is the comparison: two hundred localizations in a
            ring and two hundred in a smear give the same three numbers.
            """
            x = np.asarray(ctx.locs["x_nm"], dtype=float)
            y = np.asarray(ctx.locs["y_nm"], dtype=float)
            ax.scatter(x, y, s=6, c="#2f7fd0", alpha=0.6, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            name = precision_name(settings.precision_column)
            unit = "px" if name.endswith("_pix") else "nm"
            ax.set_title(f"{values['n_localizations']} localizations, "
                         f"{values[name]:.1f} {unit} precision")

        return Result(text=f"{values['n_localizations']} localizations",
                      plot=plot if len(ctx.locs) else None,
                      data=values, settings=settings)


# ----------------------------------------------------------------- analysis

def auto_only():
    """*results from* without a session: only *auto*.  The plugin's
    `choices` puts the evaluations on the ROIs beside it."""
    return [("", "auto")]


@dataclass
class HistogramSettings:
    bins: int = param(20, label="bins", min=1,
                      help="number of equal-width bins, spanning each column's "
                           "smallest to largest value")
    fields: str = param("", label="columns",
                        help="comma separated; empty means every numeric "
                             "column but the counts behind the means")
    evaluation: str = param("", label="results from", choices=auto_only,
                            help="the evaluation whose columns are drawn; "
                                 "auto: every evaluation on the ROIs")


def histograms(rows: Sequence[Dict[str, Any]], settings: HistogramSettings
               ) -> Dict[str, Dict[str, Any]]:
    """One histogram per column of the site table."""
    if not rows:
        return {}
    named = [f.strip() for f in settings.fields.split(",") if f.strip()]
    if not named:
        # every numeric column but the ids and the ``_n`` counts, which only
        # say how many values went into a mean -- nearly always the
        # localization count again
        named = [k for k in rows[0]
                 if k not in ("roi_id", "file_id") and not k.endswith("_n")
                 and all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                         for r in rows)]
    out = {}
    for field in named:
        values = np.asarray([r.get(field, np.nan) for r in rows], dtype=float)
        finite = np.isfinite(values)
        counts, edges = np.histogram(values[finite], bins=settings.bins)
        out[field] = {"counts": counts, "edges": edges,
                      "roi_ids": [r["roi_id"] for r, ok in zip(rows, finite) if ok],
                      "values": values[finite], "missing": int((~finite).sum())}
    return out


# ---------------------------------------------------------------- analysis

EVALUATION_HELP = ("the evaluation whose results are analysed; auto: the only "
                   "one there is, or the chain's own -- asked when there are "
                   "several")


class AmbiguousEvaluation(ValueError):
    """More than one evaluation could be meant, and nothing says which."""

    def __init__(self, evaluator: str, labels: Sequence[str]):
        self.labels = list(labels)
        name = evaluator.rsplit("/", 1)[-1] if evaluator else "evaluator"
        super().__init__(f"the ROIs have {len(labels)} {name} evaluations "
                         f"({', '.join(labels)}): name one in *results from*")


def choose_evaluation(project, evaluator: str = "", chosen: str = "",
                      made: Sequence = (), roi_ids=None) -> Optional[str]:
    """The name of the evaluation an ROI analysis reads.

    ``chosen`` when the user named one; else, in a chain that has made an
    evaluation by ``evaluator`` (``made``: its ``(name, path)`` so far), the
    chain's own, newest; else the only one on the ROIs.  None when there is
    none -- the analysis says what is missing.  Several, and nothing saying
    which, is `AmbiguousEvaluation`: a guess between two settings of the same
    evaluator would be a result nobody chose.  ``evaluator`` empty means any.
    """
    found = project.labels(evaluator or None, roi_ids) if project is not None else []
    own = [name for name, path in made if not evaluator or path == evaluator]
    if chosen:
        if chosen not in found and chosen not in own:
            raise ValueError(f"no evaluation on the ROIs is called {chosen!r}"
                             + (f"; there is {', '.join(found)}" if found else ""))
        return chosen
    if own:
        return own[-1]
    if len(found) == 1:
        return found[0]
    if not found:
        return None
    raise AmbiguousEvaluation(evaluator, found)


class SiteAnalysisPlugin(Plugin):
    """An ROI manager analysis: it summarises what evaluators measured.

    ``evaluator`` is the path of the evaluator whose results it reads (empty:
    any).  Its settings have a field ``evaluation`` (*results from*) naming
    which evaluation, by its name: the same evaluator renamed and run with
    other settings is a second evaluation, to compare, and the analysis must
    know which one it is about.  Empty takes the only one, or the chain's
    own; when that leaves several, `preflight` asks, once, with a button per
    evaluation, and the answer is written into the field.
    """
    evaluator: str = ""

    def evaluation(self, ctx: Context, settings) -> Optional[str]:
        """The evaluation this run reads, by name (`choose_evaluation`)."""
        return choose_evaluation(ctx.rois, self.evaluator,
                                 getattr(settings, "evaluation", ""),
                                 ctx.evaluations)

    def rows(self, ctx: Context, settings) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """The site table of that evaluation, and its name.  A context that
        brings its own table (`site_table`) is taken as it is."""
        if ctx.site_table is not None:
            return list(ctx.site_table), getattr(settings, "evaluation", "") or None
        if ctx.rois is None:
            raise ValueError("evaluate some ROIs first: there is no site table")
        label = self.evaluation(ctx, settings)
        if label is None and self.evaluator:
            return [], None
        return ctx.rois.results(labels=None if label is None else [label]), label

    # what *auto* is called when it means every evaluation (Histograms)
    auto_label = ""

    def choices(self, ctx: Context, settings):
        """*results from*: *auto*, saying what it resolves to, then every
        evaluation by `evaluator` on the ROIs -- a chain's own included,
        before it has run."""
        from dataclasses import replace
        project = ctx.rois
        found = project.labels(self.evaluator or None) if project is not None else []
        for name, path in ctx.evaluations:
            if (not self.evaluator or path == self.evaluator) and name not in found:
                found.append(name)
        try:
            meant = self.evaluation(ctx, replace(settings, evaluation=""))
            auto = (f"auto ({meant})" if meant else
                    f"auto ({self.auto_label})" if self.auto_label else "auto")
        except AmbiguousEvaluation:
            auto = "auto (asks which)"
        return {"evaluation": [("", auto)] + [(name, name) for name in sorted(found)]}

    def preflight(self, ctx: Context, settings):
        """Which evaluation, when that is not clear: a choice, not a yes."""
        try:
            self.evaluation(ctx, settings)
        except AmbiguousEvaluation as unclear:
            from dataclasses import replace
            project = ctx.rois
            choices = []
            for label in unclear.labels:
                made = project.evaluation_steps(label) if project is not None else []
                values = (made[0].get("parameters") if made else None) or {}
                choices.append(PreflightChoice(
                    label=label, settings=replace(settings, evaluation=label),
                    help=", ".join(f"{k} {v}" for k, v in values.items())))
            name = self.evaluator.rsplit("/", 1)[-1] or "evaluator"
            return PreflightQuestion(
                text=f"The ROIs have {len(unclear.labels)} {name} evaluations. "
                     f"Which should {self.name or 'the analysis'} read?",
                choices=choices, run_label="")
        return None


@register("ROIManager/Analyze/Histograms")
class Histograms(SiteAnalysisPlugin):
    """Distributions of the evaluation results, one histogram per column."""

    Settings = HistogramSettings
    version = "1"
    auto_label = "all"

    def evaluation(self, ctx: Context, settings) -> Optional[str]:
        # every column of every evaluation unless one is named: a histogram
        # of each is the overview, and nothing here can be ambiguous
        if settings.evaluation:
            return choose_evaluation(ctx.rois, "", settings.evaluation)
        return None

    def run(self, ctx: Context, settings: HistogramSettings) -> Result:
        rows, _ = self.rows(ctx, settings)
        if not rows:
            raise ValueError("evaluate some ROIs first: there is no site table")
        data = histograms(rows, settings)
        if not data:
            raise ValueError("no numeric columns to histogram")

        def one(ax, field, d) -> None:
            ax.bar(d["edges"][:-1], d["counts"], width=np.diff(d["edges"]),
                   align="edge", color="0.6", edgecolor="0.3")
            ax.set_xlabel(field.replace("_", " "))
            ax.set_ylabel("ROIs")
            if d["missing"]:
                ax.set_title(f"{d['missing']} missing", fontsize=8, color="0.4")

        def plot(figure) -> None:
            """A grid, one histogram per column.

            As many panels as there are columns, which is why this declares
            them and is handed the figure: a single axis is right for a plugin
            with one thing to draw and this has as many as the site table has
            columns.
            """
            for axis, (field, d) in zip(figure.subplots(len(data), 1, squeeze=False)
                                        .ravel(), data.items()):
                one(axis, field, d)

        return Result(text=f"{len(rows)} ROIs, {len(data)} columns",
                      plot=Plot(draw=plot, panels=len(data),
                                size=(5, max(2.2 * len(data), 2.2))),
                      data={"histograms": data, "rows": rows}, settings=settings)
