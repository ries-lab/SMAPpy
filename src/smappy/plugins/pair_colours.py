"""*Pair colours*: the traces of a two-thread MINFLUX measurement paired --
a trace of thread 0 with the thread-1 trace measured at the same time --
as ``pair_id``, and the offset between the colours measured in the pairs:
per pair, over time, over the field, and as a named transformation that
puts thread 1 onto thread 0 (*Transform*).  SMAP's Segment_cotracks and
DCoffset_Beads_ROIs, stripped down; the maths is `smappy.pairing`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register, settings_values

LABEL = "Pair colours"


@dataclass
class PairColoursSettings:
    min_overlap_s: float = param(0.0, label="overlap at least", unit="s", min=0.0,
                                 help="two traces measured together for less are not a pair")
    max_dt_ms: float = param(5.0, label="match within", unit="ms", min=0.01,
                             help="a thread-1 localization is matched with the nearest "
                                  "thread-0 one in time, if this close")
    name: str = param("", label="transformation",
                      help="the name the offset is kept under for Transform; empty: "
                           "'thi 1 onto thi 0'")


@register("Analysis/MINFLUX/Pair colours")
class PairColours(Plugin):
    """Pairs the traces of a two-thread MINFLUX measurement (pair_id: one
    track in either colour or both) and measures the offset between the
    colours in the pairs, kept as a transformation for Transform."""

    Settings = PairColoursSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: PairColoursSettings) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            thi = np.asarray(ctx.locs["thi"]) if "thi" in ctx.locs else None
            if thi is None or len(np.unique(thi)) < 2:
                state.notice = "needs two threads (thi 0 and 1): this table has one"
        return state

    def run(self, ctx: Context, settings: PairColoursSettings) -> Result:
        from .. import pairing as P
        from ..locs import Localizations
        from ..mathparse import remember
        from ..region_runs import column_table
        from ..transformation import Transformation, shift
        locs = ctx.locs
        for c in ("tid", "thi", "time_s", "x_nm", "y_nm"):
            if c not in locs:
                raise ValueError(f"pairing needs {c} (a MINFLUX table): the table has "
                                 + ", ".join(sorted(locs.keys())))
        thi = np.asarray(locs["thi"], np.int64)
        if len(np.unique(thi)) < 2:
            raise ValueError("pairing needs two threads (thi 0 and 1): this table has "
                             f"only thi {int(thi[0]) if len(thi) else 0}")
        use = np.asarray(ctx.selection.mask, bool)
        tid = np.where(use, np.asarray(locs["tid"], np.int64), -1)
        files = np.asarray(locs["filenumber"]) if "filenumber" in locs else None
        t = np.asarray(locs["time_s"], np.float64)
        pid = P.pair_traces(tid, thi, t, files, settings.min_overlap_s)
        off = P.pair_offsets(pid, thi, t, locs["x_nm"], locs["y_nm"],
                             settings.max_dt_ms * 1e-3)
        dx, dy = off.pooled()
        out = Localizations({**locs.columns, "pair_id": pid}, dict(locs.metadata))
        remember(out, "pair_id", "", grouped="max")
        transformations = {}
        name = settings.name.strip() or "thi 1 onto thi 0"
        if np.isfinite(dx):
            t_ = Transformation("affine", shift(-dx, -dy),
                                description=f"thread 1 onto thread 0: shift by "
                                            f"({-dx:.1f}, {-dy:.1f}) nm, from "
                                            f"{len(off.pair_id)} pairs",
                                origin={"plugin": self.path})
            transformations[name] = t_
        ids = np.unique(pid[pid >= 0])
        n_pairs = len(off.pair_id)
        values = {"paired": np.isin(ids, off.pair_id).astype(float),
                  "dx_nm": np.full(len(ids), np.nan), "dy_nm": np.full(len(ids), np.nan),
                  "n_matched": np.zeros(len(ids))}
        where = np.searchsorted(ids, off.pair_id)
        values["dx_nm"][where], values["dy_nm"][where] = off.dx, off.dy
        values["n_matched"][where] = off.n
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(out, "pair_id", LABEL, ids, values, step,
                             layer=ctx.layer, session=ctx.session)
        lines = [f"{len(ids)} tracks: {n_pairs} in both colours, {len(ids) - n_pairs} in one",
                 (f"offset of thread 1 from thread 0: median ({dx:.1f}, {dy:.1f}) nm over "
                  f"{n_pairs} pairs; kept as {name!r} for Transform" if np.isfinite(dx)
                  else "no pair to measure an offset in")]

        def draw(figure):
            draw_pairs(figure, off)

        return Result(locs=out, text="\n".join(lines),
                      plot=Plot(draw, panels=3, size=(11, 3.2)),
                      id_regions={"pair_id": {LABEL: table}}, transformations=transformations,
                      data={"offsets": off, "pooled_nm": (dx, dy)},
                      settings=settings, title="pair colours")


def draw_pairs(figure, off) -> None:
    """The offset per pair, over time and over the field."""
    a, b, c = figure.subplots(1, 3)
    if not len(off.pair_id):
        for ax in (a, b, c):
            ax.set_axis_off()
        a.text(0.5, 0.5, "no pairs", ha="center", va="center", transform=a.transAxes)
        return
    a.plot(off.dx, off.dy, "o", color="0.3", ms=4)
    a.axhline(0, color="0.8", lw=0.6)
    a.axvline(0, color="0.8", lw=0.6)
    dx, dy = off.pooled()
    a.plot([dx], [dy], "x", color="tab:red", ms=10)
    a.set_xlabel("dx (nm)")
    a.set_ylabel("dy (nm)")
    a.set_aspect("equal", adjustable="datalim")
    a.set_title("offset of thread 1 per pair", fontsize=8)
    b.plot(off.time, off.dx, "o", ms=3, label="dx")
    b.plot(off.time, off.dy, "o", ms=3, label="dy")
    b.set_xlabel("time (s)")
    b.set_ylabel("offset (nm)")
    b.legend(fontsize=6, frameon=False)
    b.set_title("over time", fontsize=8)
    c.quiver(off.x / 1e3, off.y / 1e3, off.dx, off.dy, angles="xy")
    c.set_aspect("equal", adjustable="datalim")
    c.set_xlabel("x (µm)")
    c.set_ylabel("y (µm)")
    c.set_title("over the field", fontsize=8)
