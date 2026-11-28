"""*Find steps*: the steps of every trace, a motor's along its track --
SMAP's StepsMINFLUX for all traces at once, with MINFLUXStepsAnalyze's
pooled step sizes and dwell times, and MotorPAINT's steps as a file.

Per trace (an id column: MINFLUX's ``tid``, a linked ``track_id``) the
position along the track (the trace turned onto its long axis, forward in
time, `traces.rotate`) is fitted with AutoStepFinder
(`smappy.steps`) or BNP-Step (`smappy.bnp_steps`, every trace in one
batched sampler), optionally split and merged to an expected step size.
The per-trace numbers are a region table; the pooled step sizes and dwell
times are drawn and summarised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register, settings_values
from .region_statistics import _id_choices, id_column

LABEL = "Find steps"
METHODS = (("autostepfinder", "AutoStepFinder"), ("bnp", "BNP-Step (Bayesian)"))
#: traces sampled together by BNP-Step: memory is (traces, steps, steps)
BNP_BATCH = 128
OUTPUTS = (("table", "region table"), ("columns", "plateaus on the localizations"),
           ("file", "steps as a file"))
COLOURS = (("auto", "auto"), ("together", "each its own steps"), ("one", "as one trace"))
SMOOTHING = (("median", "running median"), ("mean", "running mean"))
AXES = (("moving", "its moving parts"), ("all", "all its localizations"))
#: the colours of a two-colour molecule's figure: colour 0 (the longer
#: wavelength, 638 nm) magenta, colour 1 green -- as the groups draw them
COLOUR_DOTS = ("#e7a6d3", "#a9dcb0")
COLOUR_JOIN = ("#cc5fa6", "#55b065")      # the localizations joined: thin, clearly seen
COLOUR_LINES = ("#b0177e", "#1f8a3a")


@dataclass
class CoMoveSettings:
    window_s: float = param(0.5, label="window", unit="s", min=0.0,
                            help="the two colours' lines fitted in windows this long; 0: one "
                                 "line over the whole time both are seen (frames where "
                                 "there is no time_s)")
    max_angle_deg: float = param(30.0, label="directions within", unit="°", min=0.0,
                                 max=180.0, advanced=True,
                                 help="the angle between the colours' directions in a window")
    max_ratio: float = param(1.5, label="speeds within", unit="×", min=1.0, advanced=True,
                             help="the faster colour's speed over the slower one's")
    max_distance_nm: float = param(50.0, label="lines within", unit="nm", min=0.0,
                                   advanced=True,
                                   help="the distance between the lines at the window's centre")
    min_travel_nm: float = param(30.0, label="judged from", unit="nm", min=0.0, advanced=True,
                                 help="a window is judged when each colour is seen for as "
                                      "long as the motor takes to travel this: a few steps")
    min_share: float = param(0.5, label="together from", min=0.0, max=1.0,
                             help="a molecule moves together when this share of its moving "
                                  "windows match; else its ROI is proposed 'apart'")


@dataclass
class AutoStepFinderSettings:
    threshold: float = param(0.15, label="threshold", min=0.0, max=10.0,
                             help="a round is accepted when its S-score peaks above this; "
                                  "lower finds more steps")
    overshoot: float = param(1.0, label="overshoot", min=0.1, max=3.0,
                             help="takes this many times the steps the S-score finds "
                                  "(SMAP's coarseness)")


@dataclass
class BnpSettings:
    sweeps: int = param(400, label="sweeps", min=20, max=100000,
                        help="Gibbs sampling sweeps, started from AutoStepFinder's steps; "
                             "more is slower and settles better")
    height_nm: float = param(0.0, label="step prior", unit="nm", min=0.0,
                             help="the prior width of a step; 0: three times each "
                                  "trace's noise")


@dataclass
class FindStepsSettings:
    id_column: str = param("", label="id attribute", choices=_id_choices,
                           help="the traces; auto: tid, else track_id, else cluster_id")
    min_locs: int = param(20, label="at least", unit="localizations", min=4,
                          help="shorter traces are passed over")
    method: str = param("autostepfinder", label="method", choices=METHODS,
                        help="AutoStepFinder: greedy fit and S-score, fast; BNP-Step: the "
                             "number of steps learned by Bayesian sampling, slower")
    # each method's own settings, a part titled by it and shown only while it
    # is the method: under one *more* they read as everyone's (2026-11)
    autostepfinder: AutoStepFinderSettings = param(
        default_factory=AutoStepFinderSettings, label="AutoStepFinder", collapsed=True)
    bnp: BnpSettings = param(default_factory=BnpSettings, label="BNP-Step", collapsed=True)
    stat: str = param("mean", label="levels", choices=(("mean", "mean"),
                                                       ("median", "median")),
                      help="a plateau's level: its points' mean or median")
    step_nm: float = param(0.0, label="split/merge to", unit="nm", min=0.0,
                           help="0: off; else merge steps smaller and split steps larger "
                                "than this expected size")
    colours: str = param("auto", label="two colours", choices=COLOURS,
                         help="auto: in a two-colour table each colour of a molecule its "
                              "own steps, both in colour 0's frame and on one clock")
    comove: CoMoveSettings = param(default_factory=CoMoveSettings,
                                   label="moving together", collapsed=True)
    in_rois: bool = param(False, label="only ROIs' traces",
                          help="only the ids the ROI manager's used ROIs hold (an ROI of "
                               "its id alone): the kinesins Segment regions found")
    # the steps are always along the track: x or y alone measured a motor's
    # steps shortened by its angle, which nobody wanted (2026-11)
    axis: str = param("moving", label="track axis from", choices=AXES, advanced=True,
                      help="the direction it walks in where it moves, or the long axis "
                           "of all its localizations, stuck parts included")
    window_s: float = param(0.1, label="moving window", unit="s", min=1e-4,
                            advanced=True, under="axis",
                            help="where it moves is judged from its mean position over "
                                 "windows this long (frames where there is no time_s)")
    min_speed: float = param(100.0, label="moving from", unit="nm/s", min=0.0,
                             advanced=True, under="axis",
                             help="two windows in a row this much faster than standing "
                                  "still count as moving (nm per frame where there is no "
                                  "time_s)")
    output: str = param("table", label="output", choices=OUTPUTS, advanced=True,
                        help="a region table only; plateau and step_level on the "
                             "localizations; or a file of one localization per plateau")
    line: bool = param(True, label="localizations as a line", advanced=True,
                       help="the traces' localizations joined in time order in the "
                            "figure; off, as dots")
    guide_nm: float = param(16.0, label="guide lines every", unit="nm", min=0.0,
                            advanced=True,
                            help="thin grey lines behind the traces this far apart, from "
                                 "the first plateau: a ruler for the steps; 0: none")
    smooth_stat: str = param("median", label="smoothing", choices=SMOOTHING, advanced=True,
                             help="the figure's smoothed line (its 'smoothed' tick): a "
                                  "running median or mean")
    smooth_n: int = param(9, label="smooth over", unit="localizations", min=2,
                          max=1000, advanced=True,
                          help="the running window of the smoothed line")


def rois_ids(ctx: Context, column: str) -> set:
    """The ids of ``column`` the ROI manager's used ROIs hold, by id
    (`ROI.ids`); the ROIs of other id columns, and drawn ones, left out."""
    project = getattr(ctx.session, "rois", None) if ctx.session is not None else None
    if project is None:
        return set()
    return {int(roi.ids[column]) for roi in project.rois.values()
            if roi.use and column in (roi.ids or {})}


@register("Analysis/Tracking/Find steps")
class FindSteps(Plugin):
    """The steps of every trace (AutoStepFinder or BNP-Step, split and merge): step
    sizes and dwell times per trace and pooled, as a region table, on the
    localizations, or as a file of the plateaus."""

    Settings = FindStepsSettings
    changes_locs = True
    grouping = "ungrouped"
    # in an ROI (the per-region steps) it measures the ROI's molecule and
    # proposes its moving direction as the ROI's angle
    per_region = True
    # 2: pairs -- both threads of a pair in the first one's frame
    # 3: the dwell models fitted by likelihood, compared by AIC
    # 4: BNP-Step as a second method
    # 5: two colours by the laser; only the ROIs' traces
    # 6: the track's axis from its moving parts; two colours together on auto
    # 7: two colours: whether they move together, and a keyword proposed
    version = "8"

    def changes_locs_for(self, settings: FindStepsSettings) -> bool:
        return settings.output != "table"

    def describe(self, ctx: Optional[Context], settings: FindStepsSettings) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            from ..tracking.columns import integer_columns
            from .region_statistics import AUTO_IDS
            found = next((c for c in AUTO_IDS if c in ctx.locs), "")
            state.choices["id_column"] = ([("", f"auto ({found})" if found else "auto")]
                                          + [(c, c) for c in integer_columns(ctx.locs)])
        bnp = settings.method == "bnp"
        state.shown["autostepfinder"] = not bnp
        state.shown["bnp"] = bnp
        for name in ("window_s", "min_speed"):
            state.shown[name] = settings.axis == "moving"
        state.shown["comove"] = settings.colours != "one"
        return state

    def run(self, ctx: Context, settings: FindStepsSettings) -> Result:
        from .. import steps as S
        from .. import traces as T
        from ..region_runs import column_table, layer_mask
        locs = ctx.locs
        # the colour: the laser (channel) where the reader knew it, else the thread
        from ..dual_colour import colour_column, colour_names
        colour = colour_column(locs)
        two = colour in locs and len(np.unique(np.asarray(locs[colour]))) > 1
        if settings.colours not in dict(COLOURS):
            raise ValueError(f"two colours is auto, together or one, not {settings.colours!r}")
        if settings.colours == "together" and not two:
            raise ValueError("each colour its own steps needs two colours: a two-colour "
                             "MINFLUX table (channel or thi)")
        pairs = two and settings.colours != "one"
        names = colour_names(locs, colour) if pairs else {}
        column = ("pair_id" if pairs and not settings.id_column and "pair_id" in locs
                  else id_column(locs, settings.id_column))
        clock = "time_s" in locs
        time = np.asarray(locs["time_s"] if clock else locs["frame"], np.float64)
        r = T.runs(locs[column], time, layer_mask(ctx))
        long_enough = r.n >= settings.min_locs
        if settings.in_rois:
            wanted = rois_ids(ctx, column)
            if not wanted:
                raise ValueError(f"no used ROI holds an id of {column}: make them with "
                                 "Segment regions' 'from an id attribute', or untick "
                                 "'only ROIs' traces'")
            long_enough &= np.isin(r.ids, np.fromiter(wanted, np.int64))
        if not long_enough.any():
            raise ValueError(f"no trace of {column} has {settings.min_locs} localizations "
                             "the layer keeps"
                             + (" among the ROIs' ids" if settings.in_rois else ""))
        x = np.asarray(locs["x_nm"], np.float64)[r.order]
        y = np.asarray(locs["y_nm"], np.float64)[r.order]
        t = time[r.order]
        thread = (np.asarray(locs[colour], np.int64)[r.order] if pairs
                  else np.zeros(len(x), np.int64))
        axis_angle = np.full(len(r.ids), np.nan)
        moving_share = np.full(len(r.ids), np.nan)
        a = T.axes(r, x, y)
        if pairs:
            # a pair in its colour 0's frame, so both colours' steps line up
            a0 = T.axes(r, x, y, thread == 0)
            a = {k: np.where(np.isfinite(a0[k]), a0[k], a[k]) for k in a}
        if settings.axis == "moving":
            # the direction it walks in where it walks, both colours
            # together: the stuck parts pull the long axis, not this
            window = settings.window_s
            for i in np.flatnonzero(long_enough):
                sl = slice(r.start[i], r.start[i] + r.n[i])
                ang, share = T.moving_direction(t[sl], x[sl], y[sl], window,
                                                settings.min_speed)
                moving_share[i] = share
                if np.isfinite(ang):
                    a["angle_deg"][i] = ang
        axis_angle = np.asarray(a["angle_deg"], float).copy()
        coord, _ = T.rotate(r, x, y, a)
        per = {k: np.full(len(r.ids), np.nan) for k in
               ("n_steps", "step_median_nm", "step_mean_nm", "dwell_mean", "velocity_steps",
                "erlang_shape")}
        comove = {k: np.full(len(r.ids), np.nan) for k in
                  ("comove_share", "comove_angle_deg", "comove_speed_ratio",
                   "comove_distance_nm", "comove_windows")}
        if pairs:
            from ..dual_colour import co_movement
            cm = settings.comove
            for i in np.flatnonzero(long_enough):
                sl = slice(r.start[i], r.start[i] + r.n[i])
                seen = np.unique(thread[sl])
                if len(seen) != 2:
                    continue
                c = co_movement(t[sl], x[sl], y[sl], (thread[sl] == seen[1]).astype(np.int8),
                                cm.window_s, settings.min_speed, max_angle=cm.max_angle_deg,
                                max_ratio=cm.max_ratio, max_distance=cm.max_distance_nm,
                                min_travel=cm.min_travel_nm)
                comove["comove_share"][i] = c.share
                comove["comove_angle_deg"][i] = c.angle_deg
                comove["comove_speed_ratio"][i] = c.speed_ratio
                comove["comove_distance_nm"][i] = c.distance_nm
                comove["comove_windows"][i] = c.windows
        sizes, dwells, examples = [], [], []
        plateau = np.full(len(x), -1, np.int64)
        level = np.full(len(x), np.nan)
        rows_file = []
        next_plateau = 1
        by_thread = {}                      # pairs: per thread, the median |step| per pair

        def pieces():
            """Each trace's (and, for pairs, thread's) points: (i, thread, mask)."""
            for i in np.flatnonzero(long_enough):
                sl = slice(r.start[i], r.start[i] + r.n[i])
                for th in (np.unique(thread[sl]) if pairs else np.zeros(1, np.int64)):
                    sub = (thread[sl] == th) & np.isfinite(coord[sl])
                    if sub.sum() >= settings.min_locs // (2 if pairs else 1):
                        yield i, int(th), coord[sl][sub]

        sampled = {}
        if settings.method == "bnp":
            from ..bnp_steps import bnp_steps
            todo = sorted(pieces(), key=lambda p: len(p[2]))   # alike lengths batched: less padding
            for start in range(0, len(todo), BNP_BATCH):
                if ctx.stopped:
                    break
                ctx.report(f"BNP-Step: traces {start + 1}-"
                           f"{min(start + BNP_BATCH, len(todo))} of {len(todo)}")
                chunk = todo[start:start + BNP_BATCH]
                res = bnp_steps([p[2] for p in chunk], sweeps=settings.bnp.sweeps,
                                height_sd=settings.bnp.height_nm or None, seed=start)
                for (i, th, _), bound in zip(chunk, res.boundaries):
                    sampled[(i, th)] = bound
        for i in np.flatnonzero(long_enough):
            if ctx.stopped:
                break
            sl = slice(r.start[i], r.start[i] + r.n[i])
            threads = np.unique(thread[sl]) if pairs else np.zeros(1, np.int64)
            shown = []
            for th in threads:
                sub = (thread[sl] == th) & np.isfinite(coord[sl])
                if sub.sum() < settings.min_locs // (2 if pairs else 1):
                    continue
                xi, ti = coord[sl][sub], t[sl][sub]
                if settings.method == "bnp":
                    b = sampled.get((i, int(th)), np.zeros(0, np.int64))
                else:
                    b = S.autostepfinder(xi, settings.autostepfinder.threshold,
                                         settings.autostepfinder.overshoot,
                                         stat=settings.stat).boundaries
                if settings.step_nm > 0 and len(b):
                    b = S.split_merge(xi, b, settings.step_nm, settings.stat)
                elif len(b) and settings.method != "bnp":
                    # the sampler's times are already where the data put them
                    b = S.refine(xi, b, settings.stat)
                tab = S.step_table(xi, ti, b, settings.stat)
                first = th == threads[0]
                if first:
                    per["n_steps"][i] = len(tab["size"])
                if len(tab["size"]):
                    med = float(np.median(np.abs(tab["size"])))
                    if pairs:
                        by_thread.setdefault(int(th), np.full(len(r.ids), np.nan))[i] = med
                    if first:
                        per["step_median_nm"][i] = med
                        per["step_mean_nm"][i] = float(np.mean(tab["size"]))
                        inner = tab["dwell"][1:-1] if len(tab["dwell"]) > 2 else np.zeros(0)
                        if len(inner):
                            per["dwell_mean"][i] = float(inner.mean())
                            per["erlang_shape"][i] = S.erlang_shape(inner)
                        duration = ti[-1] - ti[0]
                        if duration > 0:
                            per["velocity_steps"][i] = float(np.sum(tab["size"]) / duration)
                    sizes.append(tab["size"])
                    dwells.append(tab["dwell"][1:-1])
                shown.append((ti, xi, S.fit_from(xi, b, settings.stat), int(th)))
                rows = np.arange(r.start[i], r.start[i] + r.n[i])[sub]
                edges = np.r_[0, b, len(xi)]
                for k, (s0, e0) in enumerate(zip(edges[:-1], edges[1:])):
                    plateau[rows[s0:e0]] = next_plateau + k
                    level[rows[s0:e0]] = tab["level"][k]
                    if settings.output == "file":
                        sel = rows[s0:e0]
                        rows_file.append((float(x[sel].mean()), float(y[sel].mean()),
                                          float(t[sel][0]), int(r.ids[i]), tab["level"][k],
                                          float(tab["dwell"][k]), int(e0 - s0)))
                next_plateau += len(edges) - 1
            if shown and (len(examples) < 4 or r.n[i] > min(e[1] for e in examples)):
                examples.append((shown, r.n[i], r.ids[i]))
                examples = sorted(examples, key=lambda e: -e[1])[:4]
        sizes = np.concatenate(sizes) if sizes else np.zeros(0)
        dwells = np.concatenate(dwells) if dwells else np.zeros(0)
        unit = "s" if clock else "frames"
        values = {"n_steps": per["n_steps"], "step_median_nm": per["step_median_nm"],
                  "step_mean_nm": per["step_mean_nm"], f"dwell_mean_{unit}": per["dwell_mean"],
                  ("velocity_nm_per_s" if clock else "velocity_nm_per_frame"): per["velocity_steps"],
                  "erlang_shape": per["erlang_shape"]}
        for th, med in sorted(by_thread.items()):
            values[f"step_median_{colour}{th}_nm"] = med
        if pairs:
            values.update(comove)
        values["angle_deg"] = axis_angle
        if settings.axis == "moving":
            values["moving_share"] = moving_share
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(locs, column, LABEL, r.ids, values, step,
                             layer=ctx.layer, session=ctx.session)
        pooled = pooled_steps(sizes, dwells)
        together = ""
        if pairs:
            share = comove["comove_share"]
            judged = np.isfinite(share)
            if judged.any():
                apart = int((share[judged] < settings.comove.min_share).sum())
                together = (f"; of {int(judged.sum())} two-colour molecules that move, "
                            f"{apart} not together")
        lines = [f"{int(long_enough.sum())} traces of {column} with {settings.min_locs} or more "
                 f"localizations: {len(sizes)} steps{together}",
                 f"step size: {pooled['step_mean_nm']:.2f} ± {pooled['step_sd_nm']:.2f} nm "
                 f"(mean ± sd of |size|); dwell {pooled['dwell_mean']:.4g} {unit}, "
                 f"Erlang shape {pooled['erlang_shape']:.2f}"
                 if len(sizes) else "no steps found",
                 *([model_line(pooled, unit)] if pooled.get("models") else []),
                 f"kept as 'per {column}: {LABEL}'"]
        out_locs, files = None, []
        if settings.output == "columns":
            from ..locs import Localizations
            cols = dict(locs.columns)
            full_p = np.full(len(locs), -1, np.int64)
            full_l = np.full(len(locs), np.nan, np.float32)
            full_p[r.order] = plateau
            full_l[r.order] = level
            cols["plateau"], cols["step_level_nm"] = full_p, full_l
            out_locs = Localizations(cols, dict(locs.metadata))
        elif settings.output == "file" and rows_file:
            files = [steps_file(rows_file, clock, ctx)]

        # in an ROI: its molecule's numbers, and its direction as the ROI's
        # angle, set only by *Apply proposals*
        region_values, proposals = {}, {}
        if ctx.per_region:
            done = np.flatnonzero(long_enough)
            if len(done):
                i = int(done[np.argmax(r.n[done])])
                region_values = {k: float(v[i]) for k, v in values.items()}
                if np.isfinite(axis_angle[i]):
                    proposals["angle"] = float(axis_angle[i])
                    lines.append(f"proposes for the ROI: angle {axis_angle[i]:.0f}°")
                share = comove["comove_share"][i]
                if pairs and np.isfinite(share):
                    from .segment import APART, COLOURS_LIST, DUAL
                    word = DUAL if share >= settings.comove.min_share else APART
                    proposals[f"keyword:{COLOURS_LIST}"] = word
                    lines.append(f"moves together in {share:.0%} of its moving windows: "
                                 f"proposes {COLOURS_LIST} '{word}'")
        label = f"{column} {{}}"
        line = settings.line
        smoothing = (settings.smooth_n, settings.smooth_stat)
        guide = settings.guide_nm

        def draw(figure, switches):
            draw_steps(figure, examples, sizes, dwells, pooled, unit, names=names,
                       label=label, line=line, steps=switches.get("steps", True),
                       guide=guide,
                       smoothed=smoothing if switches.get("smoothed") else None)

        return Result(locs=out_locs, files=files, append=bool(files),
                      text="\n".join(lines), plot=Plot(draw, panels=6, size=(13, 6.4),
                                                     switches={"steps": True, "smoothed": False}),
                      values=region_values, proposals=proposals,
                      id_regions={column: {LABEL: table}},
                      data={"sizes": sizes, "dwells": dwells, "pooled": pooled},
                      settings=settings, title="find steps")


def pooled_steps(sizes, dwells) -> dict:
    """MINFLUXStepsAnalyze's numbers over all traces: the step size (mean and
    sd of |size|, and the forward fraction), the dwell time (mean; the
    exponential's rate is its inverse), the Erlang shape (mean^2 / var of the
    dwells: the hidden sub-steps), and the velocity step / dwell."""
    from ..steps import dwell_models, erlang_shape
    a = np.abs(np.asarray(sizes, np.float64))
    d = np.asarray(dwells, np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    out = {"n_steps": float(len(a)),
           "step_mean_nm": float(a.mean()) if len(a) else np.nan,
           "step_sd_nm": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
           "forward": float((np.asarray(sizes) > 0).mean()) if len(a) else np.nan,
           "dwell_mean": float(d.mean()) if len(d) else np.nan,
           "erlang_shape": erlang_shape(d)}
    out["velocity"] = (out["step_mean_nm"] * (2 * out["forward"] - 1) / out["dwell_mean"]
                       if len(a) and len(d) else np.nan)
    out["models"] = dwell_models(d)
    out["best_model"] = (min(out["models"], key=lambda k: out["models"][k]["AIC"])
                         if out["models"] else "")
    return out


def model_line(pooled, unit: str) -> str:
    """The dwell models, the one the AIC prefers first."""
    m = pooled["models"]
    parts = []
    for name in sorted(m, key=lambda k: m[k]["AIC"]):
        v = m[name]
        if name == "exponential":
            what = f"rate {v['rate']:.4g}/{unit[0]}"
        elif name == "gamma":
            what = f"shape {v['shape']:.2f}, rate {v['rate']:.4g}/{unit[0]}"
        else:
            what = f"rates {v['rate1']:.4g} and {v['rate2']:.4g}/{unit[0]}"
        parts.append(f"{name} ({what}, AIC {v['AIC'] - m[pooled['best_model']]['AIC']:+.1f})")
    return "dwell models, best first: " + "; ".join(parts)


def steps_file(rows, clock: bool, ctx: Context):
    """One localization per plateau, at its points' mean position: MotorPAINT's
    steps as a file of their own."""
    from ..io.formats import FileInfo
    from ..locs import Localizations
    a = np.array(rows, dtype=np.float64)
    n = len(a)
    cols = {"x_nm": a[:, 0].astype(np.float32), "y_nm": a[:, 1].astype(np.float32),
            ("time_s" if clock else "frame_start"): a[:, 2],
            "frame": np.arange(n, dtype=np.int64), "tid": a[:, 3].astype(np.int64),
            "step_level_nm": a[:, 4].astype(np.float32), "dwell": a[:, 5].astype(np.float32),
            "n_locs": a[:, 6].astype(np.int64)}
    name = "steps"
    if ctx.session is not None and ctx.session.files:
        name = f"{ctx.session.files[0].name} (steps)"
    return (Localizations(cols, {"units": "nm"}), FileInfo(name, "", "steps"), None)


def draw_steps(figure, examples, sizes, dwells, pooled, unit, names=None,
               label="{}", line=True, steps=True, smoothed=None,
               guide=0.0) -> None:
    """The longest traces, each a panel of its own above, their steps over
    the localizations; the pooled step sizes, dwells and velocity below.  A
    two-colour molecule's colours share its clock (time from its first
    localization in either) and its frame (the position from colour 0's
    first plateau), each drawn in its colour with its own steps.

    ``line``: the localizations joined in time order, solid, no dots (dots
    on a line hid it at a motor's rates); else dots.  ``steps``: the fitted
    steps over them.  ``smoothed``: ``(n, "median" | "mean")``, the trace
    run through that window drawn as the localizations are, and the
    localizations behind it faint.  ``guide``: thin grey lines behind it
    all this far apart (nm), from the first plateau -- a ruler a step of
    that size is read against.  The steps are drawn thin: a thick line hid
    the localizations it was fitted to (2026-11)."""
    from ..steps import smoothed as smooth
    names = names or {}
    # the traces fill the upper half, side by side; the pooled numbers below
    grid = figure.add_gridspec(2, 2, height_ratios=(1.3, 1.0), hspace=0.32, wspace=0.18,
                               left=0.06, right=0.985, top=0.95, bottom=0.08)
    shown_n = max(1, min(3, len(examples)))
    top = grid[0, :].subgridspec(1, shown_n, wspace=0.22)
    tops = [figure.add_subplot(top[0, k]) for k in range(shown_n)]
    b, c = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])
    for k, ax in enumerate(tops):
        if k >= len(examples):
            ax.set_axis_off()
            continue
        shown, n, tid = examples[k]
        t0 = min(s[0][0] for s in shown)
        first = min(shown, key=lambda s: s[3])            # colour 0 where there is one
        x0 = first[2][0]
        counts = []
        for t, x, fit, th in shown:
            two = bool(names)
            dots = COLOUR_DOTS[th % 2] if two else "0.65"
            join = COLOUR_JOIN[th % 2] if two else "0.45"
            fitc = COLOUR_LINES[th % 2] if two else "tab:red"
            faint = 0.25 if smoothed else 1.0
            if line:
                ax.plot(t - t0, x - x0, "-", lw=0.7, color=join, alpha=faint)
            else:
                ax.plot(t - t0, x - x0, ".", ms=3, color=dots, alpha=faint)
            if smoothed:
                ax.plot(t - t0, smooth(x, *smoothed) - x0, "-", lw=0.9, color=join)
            if steps:
                ax.step(t - t0, fit - x0, where="post", color=fitc, lw=0.9,
                        label=names.get(th) if two else None)
            counts.append(int(np.count_nonzero(np.diff(fit))))
        ax.set_xlabel(f"time ({unit})")
        if guide > 0:
            lo, hi = ax.get_ylim()
            ys = np.arange(np.ceil(lo / guide), np.floor(hi / guide) + 1) * guide
            if len(ys) <= 200:               # a ruler, not a grey fill
                for y in ys:
                    ax.axhline(y, color="0.82", lw=0.5, zorder=0)
            ax.set_ylim(lo, hi)
        ax.set_ylabel("position along the track (nm)")
        ax.set_title(label.format(tid) + ": " + " + ".join(str(c) for c in counts)
                     + " steps", fontsize=8)
        if names and steps:                  # the colours are named on the steps
            ax.legend(fontsize=6, frameon=False)
    if len(sizes):
        a = np.abs(sizes)
        lim = np.quantile(a, 0.98) * 1.2
        # few steps per data set: bins a histogram of them can fill
        nb = int(np.clip(1.5 * np.sqrt(len(sizes)), 6, 25))
        b.hist(sizes, bins=np.linspace(-lim, lim, 2 * nb + 1), color="0.35")
        b.set_title(f"|step| {pooled['step_mean_nm']:.1f} ± {pooled['step_sd_nm']:.1f} nm, "
                    f"{100 * pooled['forward']:.0f} % forward", fontsize=8)
    b.set_xlabel("step size (nm)")
    b.set_ylabel("steps")
    if len(dwells):
        hi = np.quantile(dwells, 0.98)
        nb = int(np.clip(np.sqrt(len(dwells)), 5, 30))
        c.hist(dwells[dwells <= hi], bins=nb, color="0.35", density=True)
        tt = np.linspace(1e-9, hi, 200)
        models = pooled.get("models") or {}
        best = pooled.get("best_model")
        for name, colour in (("exponential", "tab:blue"), ("gamma", "tab:red"),
                             ("hypoexponential", "tab:green")):
            v = models.get(name)
            if v is None:
                continue
            if name == "exponential":
                pdf = v["rate"] * np.exp(-v["rate"] * tt)
            elif name == "gamma":
                from scipy.stats import gamma
                pdf = gamma.pdf(tt, v["shape"], scale=1 / v["rate"])
            else:
                k1, k2 = v["rate1"], v["rate2"]
                pdf = k1 * k2 / max(k2 - k1, 1e-12) * (np.exp(-k1 * tt) - np.exp(-k2 * tt))
            c.plot(tt, pdf, color=colour, lw=1.6 if name == best else 0.8,
                   label=name + (" (best)" if name == best else ""))
        c.legend(fontsize=6, frameon=False)
    c.set_xlabel(f"dwell ({unit})")
    c.set_ylabel("density")
    per = unit[:-1] if unit == "frames" else "s"
    c.set_title(f"dwell {pooled['dwell_mean']:.3g} {unit}; velocity {pooled['velocity']:.3g} "
                f"nm/{per} (step × net direction / dwell)", fontsize=8)
