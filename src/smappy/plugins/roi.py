"""The ROI manager's plugins: finding sites, measuring one, summarising many.

Three kinds, all ordinary plugins:

* **Segment** proposes candidate ROIs on a whole file.  It runs once, and
  derives from `SegmenterPlugin`, which gives it a Preview: the same
  proposals, drawn in the ROI manager and not added.  The one shipped is
  *Segment regions* (`segment.py`); the density-peaks finder it offers is
  here, `density_peaks`.
* **Evaluate** measures *one* ROI and returns that site's row in
  `Result.values`.  It declares ``per_region = True``, which is what the
  evaluation window offers -- a property of the plugin, not of its folder, so
  the window can filter correctly with one shared plugin tree -- and runs on
  the view like any other plugin as well (`docs/regions.md`).  None lives
  here any more: the statistics an ROI gets are Localization Statistics'
  (`statistics.py`), the same plugin as on the view.
* **Analyze** reads the table evaluation produced and summarises it.  It runs
  once, over ``ctx.site_table``.

The algorithms stay module-level functions so a script can call them without a
plugin, a session or a project.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import (Context, FormState, Plot, Plugin, PreflightChoice, PreflightQuestion,
               Result, param, register)
from .. import region_results
from ..region_results import Evaluation

MAX_FINDER_PIXELS = 16_000_000


def replace_param(what: str = "ROIs"):
    """The *replace* setting every segmenter's settings end with.

    Running a segmenter again on a file it has segmented before would
    otherwise find nothing: every candidate is within the separation of an
    ROI the last run made, which is how a second NPC run reported "0 pores".
    So it is on by default -- the ROI undo brings the earlier ones back --
    and off only to add a second pass to the first.  `SegmenterPlugin.replaced`
    reads it.
    """
    return param(True, label=f"replace earlier {what}",
                 help="remove the ROIs this segmenter made on the file "
                      "before finding them again")


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
                           help="a peak with fewer localizations is not made an ROI")


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


class SegmenterPlugin(Plugin):
    """A finder: proposes ROIs on the file the manager is showing.

    Its Preview runs the same finder and adds nothing: the proposals go to
    the manager as `ROIProject.segment_preview`, drawn in grey, each preview
    replacing the last, and only Run adds them.  The undo stack cannot do
    this job -- it holds tables, and ROIs are not one -- and removing the
    last try's ROIs by hand after every change of a setting is how a finder
    stops being tuned.

    A subclass implements `propose` (what `ROIProject.find` calls) and
    `segment`, which does the work of `run` and `preview` both and goes
    through `found` for the one place they differ.  Its settings end with
    ``replace: bool = replace_param()``; see `replace_param` for why.

    A run is one step of the project's ROI undo (`ROIProject.edit`), the
    ROIs it replaced included.
    """
    changes_locs = False        # proposes ROIs; the table is untouched

    preview_help = ("propose the ROIs and draw them, grey, in the ROI manager "
                    "without adding them; Run adds them")

    def propose(self, locs, settings, existing: Sequence = ()) -> List[Any]:
        """Centres, or ``{"center", "use", "origin"}`` dicts; see `ROIProject.find`."""
        raise NotImplementedError

    def segment(self, ctx: Context, settings, commit: bool) -> Result:
        raise NotImplementedError

    def replaced(self, project, file_id, settings) -> List[str]:
        """The ROIs a run removes before it adds its own: with *replace*,
        those this segmenter made on the file (their origin names it)."""
        if not getattr(settings, "replace", False):
            return []
        return [roi.id for roi in project.rois.values() if roi.file_id == file_id
                and (roi.origin or {}).get("method") == self.path]

    def run(self, ctx: Context, settings) -> Result:
        return self.segment(ctx, settings, commit=True)

    def preview(self, ctx: Context, settings) -> Result:
        return self.segment(ctx, settings, commit=False)

    def found(self, project, file_id, settings, commit: bool
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """``(sites, effect)``: each site ``{"center", "use"}``, and what the
        result asks of the session, as `Result` keywords -- the ROIs a run
        added (``rois``), or the preview for the manager (``roi_preview``).

        Previewing changes nothing in the project, not even the preview
        itself: this runs on a worker thread, and the session puts the
        preview in place on the GUI thread (`Session.show_preview`).
        """
        replaced = self.replaced(project, file_id, settings)
        if commit:
            with project.edit(self.name) as step:
                for roi_id in replaced:
                    project.rois.pop(roi_id, None)
                rois = project.find(file_id, plugin=self, settings=settings)
                step.label = f"{self.name}: {len(rois)} ROIs" + (
                    f" for {len(replaced)}" if replaced else "")
            return [{"center": r.center, "use": r.use} for r in rois], {"rois": rois}
        _, proposals = project.propose(file_id, self, settings, exclude=replaced)
        sites = [{"center": p["center"], "use": p["use"]} for p in proposals]
        return sites, {"roi_preview": {"file_id": file_id, "method": self.path,
                                       "sites": sites, "replaces": list(replaced)}}


def current_file(project) -> Optional[str]:
    """The file the manager is showing, or the only one there is."""
    chosen = project.navigation.get("file")
    if chosen in project.sources:
        return chosen
    return next(iter(project.sources), None)


# ----------------------------------------------------------------- analysis

def auto_only():
    """*results from* without a session: only *auto*.  The plugin's
    `describe` puts the evaluations on the ROIs beside it."""
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
                            help="the measurement whose columns are drawn, on "
                                 "the ROIs or per region of an id column; "
                                 "auto: everything measured on the ROIs")


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


def grid_shape(n: int, columns: int = 4) -> Tuple[int, int]:
    """``(rows, columns)`` for ``n`` panels: up to four across, so a dozen
    measures read as three rows rather than a column a screen tall."""
    across = max(1, min(columns, n, int(np.ceil(np.sqrt(n)))))
    return int(np.ceil(n / across)), across


def measure_title(field: str) -> str:
    """``radius_nm`` -> ``radius (nm)``: the unit a column carries by name."""
    units = {"nm": "nm", "deg": "\u00b0", "pix": "px", "s": "s"}
    stem, _, last = field.rpartition("_")
    if stem and last in units:
        return f"{stem.replace('_', ' ')} ({units[last]})"
    return field.replace("_", " ")


# ---------------------------------------------------------------- analysis

EVALUATION_HELP = ("the measurement whose results are analysed; auto: the only "
                   "one there is, or the chain's own -- asked when there are "
                   "several")


class AmbiguousEvaluation(ValueError):
    """More than one evaluation could be meant, and nothing says which."""

    def __init__(self, evaluator: str, found: Sequence[Evaluation]):
        self.evaluations = list(found)
        name = evaluator.rsplit("/", 1)[-1] if evaluator else "the steps"
        super().__init__(f"there are {len(found)} sets of results by {name} "
                         f"({', '.join(map(str, found))}): name one in *results from*")


def choose_evaluation(ctx: Context, evaluator: str = "", chosen: str = ""
                      ) -> Optional[Evaluation]:
    """The evaluation an analysis reads, on the ROIs or per region of an id
    column (`region_results`).

    ``chosen`` when the user named one; else, in a chain that has made one
    by ``evaluator`` (``ctx.evaluations``), the chain's own, newest; else the
    only current one there is.  None when there is none -- the analysis says
    what is missing.  Several, and nothing saying which, is
    `AmbiguousEvaluation`: a guess between two settings of the same evaluator
    would be a result nobody chose.  ``evaluator`` empty means any.
    """
    found = region_results.available(ctx, evaluator or None)
    own = [Evaluation.parse(name) for name, path in ctx.evaluations
           if not evaluator or path == evaluator]
    if chosen:
        ev = Evaluation.parse(chosen)
        if ev not in found and ev not in own:
            raise ValueError(region_results.missing(ctx, ev))
        return ev
    if own:
        return own[-1]
    if len(found) == 1:
        return found[0]
    if not found:
        return None
    raise AmbiguousEvaluation(evaluator, found)


class SiteAnalysisPlugin(Plugin):
    """An analysis of regions: it summarises what evaluators measured.

    ``evaluator`` is the path of the evaluator whose results it reads (empty:
    any).  Its settings have a field ``evaluation`` (*results from*) naming
    which evaluation -- on the ROIs, or per region of an id column, in the
    string form of `region_results.Evaluation`: the same evaluator renamed
    and run with other settings is a second evaluation, to compare, and the
    analysis must know which one it is about.  Empty takes the only one, or
    the chain's own; when that leaves several, `preflight` asks, once, with a
    button per evaluation, and the answer is written into the field.
    """
    evaluator: str = ""
    changes_locs = False

    def evaluation(self, ctx: Context, settings) -> Optional[Evaluation]:
        """The evaluation this run reads (`choose_evaluation`)."""
        return choose_evaluation(ctx, self.evaluator, getattr(settings, "evaluation", ""))

    def rows(self, ctx: Context, settings
             ) -> Tuple[List[Dict[str, Any]], Optional[Evaluation]]:
        """The current rows of that evaluation (`region_results.rows`), and
        the evaluation; every evaluation on the ROIs when it is None.  A
        context that brings its own table (`site_table`) is taken as it is."""
        if ctx.site_table is not None:
            chosen = getattr(settings, "evaluation", "")
            return list(ctx.site_table), Evaluation.parse(chosen) if chosen else None
        ev = self.evaluation(ctx, settings)
        if ev is None:
            if self.evaluator or ctx.rois is None:
                return [], None
            return ctx.rois.results(), None
        found = region_results.rows(ctx, ev)
        if not found and not region_results.is_current(ctx, ev):
            raise ValueError(region_results.missing(ctx, ev))
        return found, ev

    # what *auto* is called when it means every evaluation (Histograms)
    auto_label = ""

    def describe(self, ctx, settings) -> FormState:
        """*results from*: *auto*, saying what it resolves to, then every
        current evaluation by `evaluator` -- a chain's own included, before
        it has run -- on the ROIs, then per region of each id column."""
        from dataclasses import replace
        if ctx is None:
            return FormState()
        found = region_results.available(ctx, self.evaluator or None)
        for name, path in ctx.evaluations:
            ev = Evaluation.parse(name)
            if (not self.evaluator or path == self.evaluator) and ev not in found:
                found.append(ev)
        try:
            meant = self.evaluation(ctx, replace(settings, evaluation=""))
            auto = (f"auto ({meant})" if meant else
                    f"auto ({self.auto_label})" if self.auto_label else "auto")
        except AmbiguousEvaluation:
            auto = "auto (asks which)"
        names = sorted(str(ev) for ev in found if ev.on_rois) + \
            [str(ev) for ev in found if not ev.on_rois]
        return FormState(choices={"evaluation": [("", auto)]
                                  + [(name, name) for name in names]})

    def preflight(self, ctx: Context, settings):
        """Which evaluation, when that is not clear: a choice, not a yes.  An
        evaluation that cannot be read is left to the run to say."""
        try:
            self.evaluation(ctx, settings)
        except AmbiguousEvaluation as unclear:
            from dataclasses import replace
            choices = []
            for ev in unclear.evaluations:
                made = region_results.steps(ctx, ev)
                values = (made[0].get("parameters") if made else None) or {}
                choices.append(PreflightChoice(
                    label=str(ev), settings=replace(settings, evaluation=str(ev)),
                    help=", ".join(f"{k} {v}" for k, v in values.items())))
            name = self.evaluator.rsplit("/", 1)[-1] or "the steps"
            return PreflightQuestion(
                text=f"There are {len(unclear.evaluations)} sets of results by {name}. "
                     f"Which should {self.name or 'the analysis'} read?",
                choices=choices, run_label="")
        except ValueError:
            return None
        return None


@register("Analysis/Regions/Summarize/Histograms")
class Histograms(SiteAnalysisPlugin):
    """Distributions of what was measured per region, one histogram per column."""

    Settings = HistogramSettings
    changes_locs = False
    version = "1"
    auto_label = "all"

    def evaluation(self, ctx: Context, settings) -> Optional[Evaluation]:
        # every column of every evaluation on the ROIs unless one is named: a
        # histogram of each is the overview, and nothing here can be ambiguous
        if settings.evaluation:
            return choose_evaluation(ctx, "", settings.evaluation)
        return None

    def run(self, ctx: Context, settings: HistogramSettings) -> Result:
        rows, ev = self.rows(ctx, settings)
        if not rows:
            raise ValueError("nothing measured in the regions yet: run the "
                             "per-region steps first")
        data = histograms(rows, settings)
        # a measure no region has a value for is said, not drawn as an empty
        # panel (NPC's labelling-efficiency columns before it has run, say)
        empty = [field for field, d in data.items() if not len(d["values"])]
        shown = {field: d for field, d in data.items() if len(d["values"])}
        if not shown:
            raise ValueError("no numeric columns with values to histogram"
                             + (f" ({', '.join(empty)} have none)" if empty else ""))
        what = "ROIs" if ev is None or ev.on_rois else "regions"
        rows_n, across = grid_shape(len(shown))

        def one(ax, field, d, first_column) -> None:
            ax.bar(d["edges"][:-1], d["counts"], width=np.diff(d["edges"]),
                   align="edge", color="0.6", edgecolor="0.3")
            ax.set_title(measure_title(field), fontsize=9)
            ax.tick_params(labelsize=7)
            if first_column:
                ax.set_ylabel(what, fontsize=8)
            if d["missing"]:
                ax.text(0.98, 0.95, f"{d['missing']} without a value",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="0.4")

        def plot(figure) -> None:
            """A grid, up to four across, one histogram per measure.

            As many panels as there are measures, which is why this declares
            them and is handed the figure: a single axis is right for a plugin
            with one thing to draw and this has as many as the table has
            columns.
            """
            axes = figure.subplots(rows_n, across, squeeze=False).ravel()
            for i, (axis, (field, d)) in enumerate(zip(axes, shown.items())):
                one(axis, field, d, i % across == 0)
            for axis in axes[len(shown):]:
                axis.set_visible(False)

        text = f"{len(rows)} {what}, {len(shown)} measures"
        if empty:
            text += f"; no values for {', '.join(empty)}"
        return Result(text=text,
                      plot=Plot(draw=plot, panels=len(shown),
                                size=(3.2 * across, 2.6 * rows_n)),
                      data={"histograms": data, "rows": rows}, settings=settings)
