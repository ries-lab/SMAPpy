"""*Find steps*: the steps of every trace, a motor's along its track --
SMAP's StepsMINFLUX for all traces at once, with MINFLUXStepsAnalyze's
pooled step sizes and dwell times, and MotorPAINT's steps as a file.

Per trace (an id column: MINFLUX's ``tid``, a linked ``track_id``) the
position along the track (the trace turned onto its long axis, forward in
time, `traces.rotate`) or along x or y is fitted with AutoStepFinder,
optionally split and merged to an expected step size (`smappy.steps`).
The per-trace numbers are a region table; the pooled step sizes and dwell
times are drawn and summarised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register, settings_values
from .id_statistics import _id_choices, id_column

LABEL = "Find steps"
ALONG = (("track", "along the track"), ("x", "x"), ("y", "y"))
OUTPUTS = (("table", "region table"), ("columns", "plateaus on the localizations"),
           ("file", "steps as a file"))


@dataclass
class FindStepsSettings:
    id_column: str = param("", label="id attribute", choices=_id_choices,
                           help="the traces; auto: tid, else track_id, else cluster_id")
    along: str = param("track", label="steps along", choices=ALONG,
                       help="the track's long axis (turned forward in time), or x or y")
    min_locs: int = param(20, label="at least", unit="localizations", min=4,
                          help="shorter traces are passed over")
    threshold: float = param(0.15, label="threshold", min=0.0, max=10.0, advanced=True,
                             help="AutoStepFinder accepts a round whose S-score peaks above "
                                  "this")
    overshoot: float = param(1.0, label="overshoot", min=0.1, max=3.0, advanced=True,
                             help="takes this many times the steps the S-score finds "
                                  "(SMAP's coarseness)")
    stat: str = param("mean", label="levels", choices=(("mean", "mean"),
                                                       ("median", "median")),
                      help="a plateau's level: its points' mean or median")
    step_nm: float = param(0.0, label="split/merge to", unit="nm", min=0.0,
                           help="0: off; else merge steps smaller and split steps larger "
                                "than this expected size")
    output: str = param("table", label="output", choices=OUTPUTS,
                        help="a region table only; plateau and step_level on the "
                             "localizations; or a file of one localization per plateau")


@register("Analysis/Tracking/Find steps")
class FindSteps(Plugin):
    """The steps of every trace (AutoStepFinder, split and merge): step
    sizes and dwell times per trace and pooled, as a region table, on the
    localizations, or as a file of the plateaus."""

    Settings = FindStepsSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def changes_locs_for(self, settings: FindStepsSettings) -> bool:
        return settings.output != "table"

    def describe(self, ctx: Optional[Context], settings: FindStepsSettings) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            from ..tracking.columns import integer_columns
            from .id_statistics import AUTO_IDS
            found = next((c for c in AUTO_IDS if c in ctx.locs), "")
            state.choices["id_column"] = ([("", f"auto ({found})" if found else "auto")]
                                          + [(c, c) for c in integer_columns(ctx.locs)])
        return state

    def run(self, ctx: Context, settings: FindStepsSettings) -> Result:
        from .. import steps as S
        from .. import traces as T
        from ..region_runs import column_table, layer_mask
        locs = ctx.locs
        column = id_column(locs, settings.id_column)
        clock = "time_s" in locs
        time = np.asarray(locs["time_s"] if clock else locs["frame"], np.float64)
        r = T.runs(locs[column], time, layer_mask(ctx))
        long_enough = r.n >= settings.min_locs
        if not long_enough.any():
            raise ValueError(f"no trace of {column} has {settings.min_locs} localizations "
                             "the layer keeps")
        x = np.asarray(locs["x_nm"], np.float64)[r.order]
        y = np.asarray(locs["y_nm"], np.float64)[r.order]
        t = time[r.order]
        if settings.along == "track":
            a = T.axes(r, x, y)
            coord, _ = T.rotate(r, x, y, a)
        else:
            coord = x if settings.along == "x" else y
        per = {k: np.full(len(r.ids), np.nan) for k in
               ("n_steps", "step_median_nm", "step_mean_nm", "dwell_mean", "velocity_steps",
                "erlang_shape")}
        sizes, dwells, examples = [], [], []
        plateau = np.full(len(x), -1, np.int64)
        level = np.full(len(x), np.nan)
        rows_file = []
        next_plateau = 1
        for i in np.flatnonzero(long_enough):
            if ctx.stopped:
                break
            sl = slice(r.start[i], r.start[i] + r.n[i])
            xi, ti = coord[sl], t[sl]
            ok = np.isfinite(xi)
            xi, ti = xi[ok], ti[ok]
            found = S.autostepfinder(xi, settings.threshold, settings.overshoot,
                                     stat=settings.stat)
            b = found.boundaries
            if settings.step_nm > 0 and len(b):
                b = S.split_merge(xi, b, settings.step_nm, settings.stat)
            elif len(b):
                b = S.refine(xi, b, settings.stat)
            tab = S.step_table(xi, ti, b, settings.stat)
            per["n_steps"][i] = len(tab["size"])
            if len(tab["size"]):
                per["step_median_nm"][i] = float(np.median(np.abs(tab["size"])))
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
            if len(examples) < 4 or r.n[i] > min(e[3] for e in examples):
                examples.append((ti, xi, S.fit_from(xi, b, settings.stat), r.n[i], r.ids[i]))
                examples = sorted(examples, key=lambda e: -e[3])[:4]
            rows = np.arange(r.start[i], r.start[i] + r.n[i])[ok]
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
        sizes = np.concatenate(sizes) if sizes else np.zeros(0)
        dwells = np.concatenate(dwells) if dwells else np.zeros(0)
        unit = "s" if clock else "frames"
        values = {"n_steps": per["n_steps"], "step_median_nm": per["step_median_nm"],
                  "step_mean_nm": per["step_mean_nm"], f"dwell_mean_{unit}": per["dwell_mean"],
                  ("velocity_nm_per_s" if clock else "velocity_nm_per_frame"): per["velocity_steps"],
                  "erlang_shape": per["erlang_shape"]}
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(locs, column, LABEL, r.ids, values, step,
                             layer=ctx.layer, session=ctx.session)
        pooled = pooled_steps(sizes, dwells)
        lines = [f"{int(long_enough.sum())} traces of {column} with {settings.min_locs} or more "
                 f"localizations: {len(sizes)} steps",
                 f"step size: {pooled['step_mean_nm']:.2f} ± {pooled['step_sd_nm']:.2f} nm "
                 f"(mean ± sd of |size|); dwell {pooled['dwell_mean']:.4g} {unit}, "
                 f"Erlang shape {pooled['erlang_shape']:.2f}"
                 if len(sizes) else "no steps found",
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

        def draw(figure):
            draw_steps(figure, examples, sizes, dwells, pooled, unit)

        return Result(locs=out_locs, files=files, append=bool(files),
                      text="\n".join(lines), plot=Plot(draw, panels=4, size=(13, 3.2)),
                      id_regions={column: {LABEL: table}},
                      data={"sizes": sizes, "dwells": dwells, "pooled": pooled},
                      settings=settings, title="find steps")


def pooled_steps(sizes, dwells) -> dict:
    """MINFLUXStepsAnalyze's numbers over all traces: the step size (mean and
    sd of |size|, and the forward fraction), the dwell time (mean; the
    exponential's rate is its inverse), the Erlang shape (mean^2 / var of the
    dwells: the hidden sub-steps), and the velocity step / dwell."""
    from ..steps import erlang_shape
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
    return out


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


def draw_steps(figure, examples, sizes, dwells, pooled, unit) -> None:
    a, b, c, d = figure.subplots(1, 4)
    for k, (t, x, fit, n, tid) in enumerate(examples[:3]):
        off = 30.0 * k
        a.plot(t - t[0], x - x[0] + off, ".", ms=1.5, color="0.6")
        a.step(t - t[0], fit - x[0] + off, where="post", color="tab:red", lw=1)
    a.set_xlabel(f"time ({unit})")
    a.set_ylabel("position (nm, offset per trace)")
    a.set_title("the longest traces and their steps", fontsize=8)
    if len(sizes):
        lim = np.quantile(np.abs(sizes), 0.98) * 1.2
        b.hist(sizes, bins=np.arange(-lim, lim + 0.5, 0.5), color="0.35")
        b.set_title(f"|step| {pooled['step_mean_nm']:.1f} ± {pooled['step_sd_nm']:.1f} nm, "
                    f"{100 * pooled['forward']:.0f} % forward", fontsize=8)
    b.set_xlabel("step size (nm)")
    b.set_ylabel("steps")
    if len(dwells):
        hi = np.quantile(dwells, 0.98)
        c.hist(dwells[dwells <= hi], bins=40, color="0.35", density=True)
        tt = np.linspace(0, hi, 200)
        m = pooled["dwell_mean"]
        c.plot(tt, np.exp(-tt / m) / m, color="tab:blue", lw=1, label="exponential")
        k = pooled["erlang_shape"]
        if np.isfinite(k) and k > 0:
            from scipy.stats import gamma
            c.plot(tt, gamma.pdf(tt, k, scale=m / k), color="tab:red", lw=1,
                   label=f"gamma, shape {k:.1f}")
        c.legend(fontsize=6, frameon=False)
    c.set_xlabel(f"dwell ({unit})")
    c.set_ylabel("density")
    d.text(0.05, 0.6, f"velocity\n{pooled['velocity']:.3g} nm/{unit[:-1] if unit == 'frames' else 's'}"
           f"\n(step × net direction / dwell)", fontsize=9, transform=d.transAxes)
    d.set_axis_off()
