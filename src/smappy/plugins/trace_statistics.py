"""*Trace statistics*: per trace or track, its geometry and precision --
displacement, velocity, the track's long and short axis, the precision from
consecutive differences, the trend across the track -- and its start, the
tail, by a rule; kept as a region table of the id column, and on request
the track turned onto its axes and the tail labelled on the localizations.

The maths is `smappy.traces`.  *Id statistics* stays the quick count,
centre, spread and statistics of attributes; this is the trace's own
geometry, for MINFLUX traces and linked tracks alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register, settings_values
from .id_statistics import _id_choices, id_column

LABEL = "Trace statistics"
RULES = (("none", "none"), ("still", "still molecule"), ("moving", "moving molecule"))


@dataclass
class TailSettings:
    rule: str = param("none", label="tail rule", choices=RULES,
                      help="still: until the trace comes within reach of its settled "
                           "position; moving: until its steps' rolling std settles")
    without_tail: bool = param(True, label="numbers without the tail",
                               help="measure each trace on its settled part")
    reference_from: int = param(20, label="settled from", min=1, max=10000,
                                help="still: the localization the settled position "
                                     "starts at")
    reference_window: int = param(30, label="settled over", min=3, max=10000,
                                  help="still: localizations whose median is the settled "
                                       "position")
    k: float = param(3.0, label="reach", unit="sigma", min=0.5, max=20.0,
                     help="still: within this many robust sigmas of the settled position")
    rolling_window: int = param(10, label="rolling window", min=3, max=1000,
                                help="moving: steps the rolling std is taken over")
    rolling_limit_nm: float = param(7.0, label="settled below", unit="nm", min=0.1,
                                    help="moving: the rolling std of the steps below "
                                         "which the trace has settled")


@dataclass
class TraceStatisticsSettings:
    id_column: str = param("", label="id attribute", choices=_id_choices,
                           help="the traces; auto: tid, else track_id, else cluster_id")
    tail: TailSettings = param(default_factory=TailSettings, label="trace start (tail)")
    cfr_limit: float = param(0.0, label="cut after cfr above", min=0.0, max=10.0,
                             help="0: off; else a trace ends at its first cfr above this "
                                  "(MINFLUX)")
    write_rotated: bool = param(False, label="write the rotated track",
                                help="x_rot_nm, y_rot_nm on the localizations: the long axis "
                                     "on x, forward in time")
    write_tail: bool = param(False, label="write the tail",
                             help="in_tail (and after_cfr) on the localizations, to filter "
                                  "or colour by")


@register("Analysis/Tracking/Trace statistics")
class TraceStatistics(Plugin):
    """Per trace or track: displacement, velocity, the track's axes, the
    precision from consecutive differences, the trend across the track and
    the start (tail) by a rule; a region table, and on request the rotated
    track and the tail on the localizations."""

    Settings = TraceStatisticsSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def changes_locs_for(self, settings: TraceStatisticsSettings) -> bool:
        return bool(settings.write_rotated or settings.write_tail)

    def describe(self, ctx: Optional[Context], settings: TraceStatisticsSettings) -> FormState:
        state = FormState()
        rule = settings.tail.rule
        for name in ("reference_from", "reference_window", "k"):
            state.active[f"tail.{name}"] = rule == "still"
        for name in ("rolling_window", "rolling_limit_nm"):
            state.active[f"tail.{name}"] = rule == "moving"
        state.active["tail.without_tail"] = rule != "none"
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            from ..tracking.columns import integer_columns
            from .id_statistics import AUTO_IDS
            found = next((c for c in AUTO_IDS if c in ctx.locs), "")
            state.choices["id_column"] = ([("", f"auto ({found})" if found else "auto")]
                                          + [(c, c) for c in integer_columns(ctx.locs)])
            state.active["cfr_limit"] = "cfr" in ctx.locs
        return state

    def run(self, ctx: Context, settings: TraceStatisticsSettings) -> Result:
        from .. import traces as T
        from ..region_runs import column_table, layer_mask
        locs = ctx.locs
        column = id_column(locs, settings.id_column)
        for c in ("x_nm", "y_nm"):
            if c not in locs:
                raise ValueError(f"no attribute {c}: the table has "
                                 + ", ".join(sorted(locs.keys())))
        clock = "time_s" in locs
        time = np.asarray(locs["time_s"] if clock else locs["frame"], np.float64)
        unit = "s" if clock else "frames"
        keep = layer_mask(ctx)
        r = T.runs(locs[column], time, keep)
        if not len(r.ids):
            raise ValueError(f"no traces: no localization the layer keeps has a "
                             f"{column} (-1 is none)")
        x = np.asarray(locs["x_nm"], np.float64)[r.order]
        y = np.asarray(locs["y_nm"], np.float64)[r.order]
        t = time[r.order]
        photons = np.asarray(locs["photons"], np.float64)[r.order] if "photons" in locs else None
        ts = settings.tail
        tail = T.tails(r, x, y, t, ts.rule, ts.reference_from, ts.reference_window, ts.k,
                       ts.rolling_window, ts.rolling_limit_nm)
        cut = (T.cfr_cut(r, np.asarray(locs["cfr"], np.float64)[r.order], settings.cfr_limit)
               if settings.cfr_limit > 0 and "cfr" in locs else np.zeros(len(x), bool))
        use = ~cut
        if ts.rule != "none" and ts.without_tail:
            use &= ~tail.in_tail
        st = T.statistics(r, x, y, t, use, photons)
        values = {
            "n_locs": st["n_locs"], "displacement_nm": st["displacement_nm"],
            f"duration_{unit}": st["duration"],
            ("velocity_nm_per_s" if clock else "velocity_nm_per_frame"): st["velocity"],
            "std_long_nm": st["std_long_nm"], "std_short_nm": st["std_short_nm"],
            "eccentricity": st["eccentricity"], "angle_deg": st["angle_deg"],
            "precision_x_nm": st["precision_x_nm"], "precision_y_nm": st["precision_y_nm"],
            "precision_long_nm": st["precision_long_nm"],
            "precision_short_nm": st["precision_short_nm"], "trend_nm": st["trend_nm"],
            f"dt_median_{unit}": st["dt_median"]}
        if photons is not None:
            values["photons_sum"] = st["photons_sum"]
        if ts.rule != "none":
            values.update({"tail_n": tail.n.astype(float), "tail_reach_nm": tail.reach_nm,
                           f"tail_duration_{unit}": tail.duration})
        if settings.cfr_limit > 0 and "cfr" in locs:
            values["cut_n"] = r.sums(cut.astype(float))
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(locs, column, LABEL, r.ids, values, step,
                             layer=ctx.layer, session=ctx.session)
        out_locs = None
        if settings.write_rotated or settings.write_tail:
            out_locs = self._written(locs, r, x, y, st, tail, cut, settings, T)
        conv = (T.convergence(r, tail.distance, min(60, int(r.n.max())), photons)
                if ts.rule != "none" else None)

        good = st["n_locs"] >= 3
        lines = [f"{len(r.ids)} traces of {column}, {int(st['n_locs'].sum())} localizations"
                 + (" (without their tails)" if ts.rule != "none" and ts.without_tail else ""),
                 f"precision from consecutive differences: median "
                 f"{np.nanmedian(st['precision_x_nm'][good]):.2f} nm (x), "
                 f"{np.nanmedian(st['precision_y_nm'][good]):.2f} nm (y)"
                 if good.any() else "too few localizations a trace for a precision"]
        if np.isfinite(st["velocity"]).any():
            lines.append(f"velocity: median {np.nanmedian(st['velocity']):.3g} nm/{unit[:-1] if unit == 'frames' else 's'}")
        if ts.rule != "none":
            with_tail = tail.n > 0
            lines.append(f"tails ({dict(RULES)[ts.rule]}): {int(with_tail.sum())} of "
                         f"{len(r.ids)} traces, median {np.median(tail.n[with_tail]) if with_tail.any() else 0:.0f} "
                         "localizations")
        lines.append(f"kept as 'per {column}: {LABEL}'")

        def draw(figure):
            panels = figure.subplots(1, 4)
            hist(panels[0], st["velocity"][good], f"velocity (nm/{'s' if clock else 'frame'})")
            hist(panels[1], st["precision_short_nm"][good], "precision across the track (nm)")
            hist(panels[2], st["eccentricity"][good], "eccentricity (long/short)")
            draw_convergence(panels[3], conv)

        return Result(locs=out_locs, text="\n".join(lines),
                      plot=Plot(draw, panels=4, size=(13, 3.0)),
                      id_regions={column: {LABEL: table}},
                      data={"statistics": values, "column": column, "tails": tail},
                      settings=settings, title="trace statistics")

    @staticmethod
    def _written(locs, r, x, y, st, tail, cut, settings, T):
        from ..locs import Localizations
        n = len(locs)
        columns = dict(locs.columns)
        if settings.write_rotated:
            a = {k: st[k] for k in ("centre_x_nm", "centre_y_nm", "angle_deg")}
            xr, yr = T.rotate(r, x, y, a)
            for name, v in (("x_rot_nm", xr), ("y_rot_nm", yr)):
                full = np.full(n, np.nan, np.float32)
                full[r.order] = v
                columns[name] = full
        if settings.write_tail:
            flag = np.zeros(n, np.int8)
            flag[r.order] = tail.in_tail
            columns["in_tail"] = flag
            if cut.any() or settings.cfr_limit > 0:
                after = np.zeros(n, np.int8)
                after[r.order] = cut
                columns["after_cfr"] = after
        return Localizations(columns, dict(locs.metadata))


def hist(ax, values, label: str) -> None:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if len(v):
        hi = np.quantile(v, 0.98) if len(v) > 20 else v.max()
        ax.hist(v[v <= hi], bins=min(50, max(8, len(v) // 10)), color="0.35")
        ax.set_title(f"median {np.median(v):.3g}", fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("traces")


def draw_convergence(ax, conv) -> None:
    """Pooled: the n-th localization's distance from its trace's settled
    position, with 120 nm / sqrt(photons so far) beside it where known."""
    if conv is None:
        ax.text(0.5, 0.5, "no tail rule chosen", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return
    ax.plot(conv["index"], conv["mean"], "o-", ms=3, color="0.2", label="mean")
    ax.plot(conv["index"], conv["median"], "-", color="0.6", label="median")
    if np.isfinite(conv["photons"]).any():
        ref = 120.0 / np.sqrt(np.maximum(conv["photons"], 1.0))
        ax.plot(conv["index"], ref, ":", color="tab:red", label="120 nm / sqrt(N)")
    ax.set_xlabel("localization in its trace")
    ax.set_ylabel("distance from settled (nm)")
    ax.set_title("trace start", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
