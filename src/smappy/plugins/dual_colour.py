"""*MINFLUX dual colour*: two-colour MINFLUX in one step -- the offset
between the colours, from beads or from the molecules themselves, with the
colours' drift against each other from the MBM beads; colour 1 moved onto
colour 0; and the traces of both colours joined into molecules, ``pair_id``.

It began as *Pair colours* -- SMAP's Segment_cotracks and
DCoffset_Beads_ROIs -- and grew into this.  The maths is
`smappy.dual_colour` (and `smappy.pairing` for the bead measurement).

**What is a bead measurement.**  The groups measure a bead in the sample
in a measurement of its own, before the tracking (and, better, after it
too).  Any open measurement that is not in the selection and has both
colours is offered; on *auto* the last one before the tracking and the
first one after it, by the acquisition dates.

**Moving the colour, again and again.**  The shift is recorded in the
table's metadata (`RECORD`) per file and measurement, and taken off before
a new one is put on: a second run moves from the positions as read, not
from the first run's.  It is put on `x_nc_nm` too where there is one, so
that *MBM drift correction* -- which starts from it -- keeps it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register, settings_values

LABEL = "MINFLUX dual colour"
#: where the shift put on colour 1 is kept, per "file:measurement"
RECORD = "colour_offsets"
SOURCES = (("auto", "auto"), ("beads", "bead measurements"), ("tracks", "the molecules (fit)"),
           ("named", "a named transformation"), ("typed", "typed"), ("none", "none"))
#: the loose distance of the first pass when the offset comes from the pairs
FIRST_PASS_NM = 100.0
#: a measurement whose typical localization is in a trace this long is taken
#: for beads on *auto*: a bead is followed for thousands (3000 per colour in
#: the bead measurement looked at), a motor for tens to hundreds
BEAD_TRACE_LOCS = 1000


def _no_measurements():
    return [("", "auto")]


def _no_after():
    return [("", "auto"), ("-", "none")]


def _no_names():
    return [("", "-")]


@dataclass
class OffsetSettings:
    source: str = param("auto", label="offset from", choices=SOURCES,
                        help="auto: the bead measurements if one is open, else the "
                             "molecules themselves")
    before: str = param("", label="beads before", choices=_no_measurements,
                        help="the bead measurement taken before the tracking; auto: the "
                             "last one")
    after: str = param("", label="beads after", choices=_no_after,
                       help="one taken after it, interpolated between the two; auto: "
                            "the first one after, if any")
    mbm_drift: bool = param(True, label="MBM colour drift",
                            help="the colours' drift against each other over time, from "
                                 "MBM beads recorded in both colours")
    knot_s: float = param(120.0, label="drift knot spacing", unit="s", min=5.0, max=10000.0,
                          advanced=True, help="the colour drift's spline, a coefficient "
                                              "this often")
    still_nm: float = param(30.0, label="moving from", unit="nm", min=0.0, advanced=True,
                            help="a molecule moving less is still: it tells the offset, "
                                 "not the lead along the track")
    dx_nm: float = param(0.0, label="dx", unit="nm",
                         help="colour 1 - colour 0, typed")
    dy_nm: float = param(0.0, label="dy", unit="nm", beside=True,
                         help="colour 1 - colour 0, typed")
    named: str = param("", label="transformation", choices=_no_names,
                       help="a named transformation that puts colour 1 onto colour 0")


@dataclass
class PairingSettings:
    max_dt_ms: float = param(5.0, label="match within", unit="ms", min=0.01,
                             help="a colour-1 localization is matched with the colour-0 "
                                  "one nearest in time, if this close")
    max_distance_nm: float = param(30.0, label="apart at most", unit="nm", min=1.0,
                                   help="how far apart a match may be, the offset taken off")
    min_fraction: float = param(0.5, label="for a share of", min=0.0, max=1.0, beside=True,
                                help="the share of the matches that must be that close")
    min_matched: int = param(10, label="matched at least", unit="locs", min=1,
                             help="two traces with fewer matches are not one molecule")
    min_overlap_s: float = param(0.0, label="overlap at least", unit="s", min=0.0,
                                 advanced=True,
                                 help="two traces measured together for less are not one")


@dataclass
class DualColourSettings:
    offset: OffsetSettings = param(default_factory=OffsetSettings, label="offset")
    pairing: PairingSettings = param(default_factory=PairingSettings, label="pairing")
    apply: bool = param(True, label="move colour 1 onto colour 0",
                        help="off: only measure and pair, the positions as they are")
    name: str = param("", label="keep as",
                      help="the name the constant offset is kept under for Transform; "
                           "empty: by the colours")


# ------------------------------------------------------------ measurements

@dataclass
class _Measurement:
    """One measurement of the open table: its file and number in it, and
    what places it in time."""
    file: int
    number: int
    did: str
    label: str
    title: str                      # what the choice lists
    start_abs: Optional[float]      # its acquisition, s since the epoch
    time_offset: float              # where `msr.merge` put its time_s
    meta: Dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.file}:{self.number}"


def _measurements(ctx: Context) -> List[_Measurement]:
    """Every MINFLUX measurement of the table, in file and measurement order."""
    from ..io.msr import _start
    from ..metadata import file_metadata
    locs = ctx.locs
    files = (np.unique(np.asarray(locs["filenumber"])) if "filenumber" in locs
             else np.zeros(1, np.int64))
    names = list(getattr(ctx.session, "file_names", lambda: [])() or [])
    out = []
    for k in files:
        meta = file_metadata(locs.metadata, int(k))
        if meta.get("mbm_of"):
            continue
        ms = ((meta.get("minflux") or {}).get("measurements")
              if isinstance(meta.get("minflux"), dict) else None) or []
        for j, m in enumerate(ms):
            fname = names[int(k)] if int(k) < len(names) else f"file {int(k) + 1}"
            label = str(m.get("label") or f"measurement {j + 1}")
            out.append(_Measurement(int(k), j, str(m.get("did") or f"{k}:{j}"), label,
                                    fname if len(ms) == 1 else f"{fname}: {label}",
                                    _start(m.get("acquisition_date", "")),
                                    float(m.get("time_offset_s", 0.0) or 0.0), m))
    return out


def _rows_of(locs, file: int, number: int) -> np.ndarray:
    n = len(locs)
    rows = (np.asarray(locs["filenumber"]) == file) if "filenumber" in locs \
        else np.ones(n, bool)
    if "measurement" in locs:
        rows &= np.asarray(locs["measurement"]) == number
    return rows


def _bead_like(locs, m: "_Measurement", colour: np.ndarray) -> bool:
    """Whether measurement ``m`` looks like beads: both colours, and its
    typical localization (the median over localizations) in a trace of
    `BEAD_TRACE_LOCS` or more."""
    rows = _rows_of(locs, m.file, m.number) & (np.asarray(locs["tid"], np.int64) >= 0)
    c = colour[rows]
    if not ((c == 0).any() and (c == 1).any()):
        return False
    keys = np.column_stack([c, np.asarray(locs["tid"], np.int64)[rows]])
    _, counts = np.unique(keys, axis=0, return_counts=True)
    lengths = np.sort(counts)
    half = np.searchsorted(np.cumsum(lengths), lengths.sum() / 2.0)
    return bool(lengths[min(half, len(lengths) - 1)] >= BEAD_TRACE_LOCS)


def _colour_names(m: Optional[_Measurement], column: str) -> Tuple[str, str]:
    if column == "channel" and m is not None:
        chans = {c["channel"]: c["name"] for c in m.meta.get("channels") or []}
        if 0 in chans and 1 in chans:
            return chans[0], chans[1]
    return (f"{column} 0", f"{column} 1")


@register("Analysis/MINFLUX/MINFLUX dual colour")
class DualColour(Plugin):
    """Two-colour MINFLUX: the offset between the colours (beads, or the
    molecules themselves, with the MBM colour drift), colour 1 moved onto
    colour 0, and the traces of both colours joined into molecules
    (pair_id)."""

    name = LABEL
    Settings = DualColourSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "2"

    def changes_locs_for(self, settings: DualColourSettings) -> bool:
        return True

    def describe(self, ctx: Optional[Context], settings: DualColourSettings) -> FormState:
        from ..dual_colour import colour_column
        state = FormState()
        src = settings.offset.source
        for f in ("before", "after"):
            state.shown[f"offset.{f}"] = src in ("auto", "beads")
        state.shown["offset.mbm_drift"] = src in ("auto", "beads", "tracks")
        state.shown["offset.still_nm"] = src in ("auto", "tracks")
        state.shown["offset.dx_nm"] = state.shown["offset.dy_nm"] = src == "typed"
        state.shown["offset.named"] = src == "named"
        if ctx is None or ctx.session is None or not len(ctx.locs):
            return state
        column = colour_column(ctx.locs)
        if column not in ctx.locs or len(np.unique(np.asarray(ctx.locs[column]))) < 2:
            state.notice = Notice("needs two colours: a two-thread MINFLUX table")
            return state
        found = _measurements(ctx)
        listed = [(m.did, m.title) for m in found]
        state.choices["offset.before"] = [("", "auto")] + listed
        state.choices["offset.after"] = [("", "auto"), ("-", "none")] + listed
        names = sorted(getattr(ctx.session, "transformations", {}) or {})
        state.choices["offset.named"] = [("", "-")] + [(n, n) for n in names]
        return state

    def run(self, ctx: Context, settings: DualColourSettings) -> Result:
        from .. import dual_colour as DC
        from ..locs import Localizations
        from ..mathparse import remember
        from ..region_runs import column_table
        from ..transformation import Transformation, shift
        locs = ctx.locs
        column = DC.colour_column(locs)
        for c in ("tid", "time_s", "x_nm", "y_nm", column):
            if c not in locs:
                raise ValueError(f"needs {c} (a MINFLUX table): the table has "
                                 + ", ".join(sorted(locs.keys())))
        colour = np.asarray(locs[column], np.int64)
        if not ((colour == 0).any() and (colour == 1).any()):
            raise ValueError("needs two colours: two threads (thi 0 and 1) or two lasers "
                             "(channel 0 and 1); this table has one")
        o, p = settings.offset, settings.pairing
        mask = np.asarray(ctx.selection.mask, bool)
        found = _measurements(ctx)
        by_did = {m.did: m for m in found}
        named_beads = {d for d in (o.before, o.after) if d and d != "-"}
        for d in named_beads:
            if d not in by_did:
                raise ValueError(f"no open measurement {d!r}: choose the bead measurement "
                                 "among the open ones")
        if o.source in ("auto", "beads"):
            # a bead measurement read with the tracking (one .msr, merged) is
            # in the selection too: it is told by its long traces
            named_beads |= {m.did for m in found if _bead_like(locs, m, colour)}
        # the targets: the measurements the selection holds, beads chosen by name aside
        if found:
            targets = [m for m in found if m.did not in named_beads
                       and (mask & _rows_of(locs, m.file, m.number)).any()]
        else:
            targets = [_Measurement(0, 0, "", "", "", None, 0.0, {})]
        if not targets:
            shown = [m.title for m in found if (mask & _rows_of(locs, m.file, m.number)).any()]
            raise ValueError("the layer shows only bead measurements ("
                             + ", ".join(shown) + "): show the tracking measurement in "
                             "the layer, beside them or alone")
        t_all = np.asarray(locs["time_s"], np.float64)
        x = np.asarray(locs["x_nm"], np.float64).copy()
        y = np.asarray(locs["y_nm"], np.float64).copy()
        cols = dict(locs.columns)
        nc = [c for c in ("x_nc_nm", "y_nc_nm") if c in cols]
        ncx = {c: np.asarray(cols[c], np.float64).copy() for c in nc}
        record = dict((locs.metadata or {}).get(RECORD) or {})
        pair_id = np.full(len(locs), -1, np.int64)
        lines, per_target, values_all, ids_all = [], [], {}, []
        next_id = 0
        n_both = 0
        first_names = None
        for m in targets:
            rows_m = (_rows_of(locs, m.file, m.number) if found else np.ones(len(locs), bool))
            names = _colour_names(m if found else None, column)
            first_names = first_names or names
            ones = rows_m & (colour == 1)
            # what an earlier run put on, taken off first
            old = record.pop(m.key, None)
            if old:
                back = np.column_stack([np.interp(t_all[ones], old["t"], old[a])
                                        for a in ("dx", "dy")])
                x[ones] += back[:, 0]
                y[ones] += back[:, 1]
                for c, a in zip(nc, (0, 1)):
                    ncx[c][ones] += back[:, a]
            drift = None
            if o.mbm_drift and o.source in ("auto", "beads", "tracks") and found:
                drift = self._drift(ctx, m, o.knot_s)
            rows_t = rows_m & mask
            curve, how, fit, bead_info = self._offset(ctx, m, found, targets, rows_t, colour,
                                                      t_all, x, y, drift, settings)
            span = (t_all[rows_m].min(), t_all[rows_m].max()) if rows_m.any() else (0.0, 0.0)
            if settings.apply and (np.any(curve.c) or drift is not None):
                sh = curve(t_all[ones])
                x[ones] -= sh[:, 0]
                y[ones] -= sh[:, 1]
                for c, a in zip(nc, (0, 1)):
                    ncx[c][ones] -= sh[:, a]
                ts = np.linspace(span[0], span[1], 200) if span[1] > span[0] \
                    else np.array([span[0]])
                sampled = curve(ts)
                record[m.key] = {"t": ts.tolist(), "dx": sampled[:, 0].tolist(),
                                 "dy": sampled[:, 1].tolist(), "colours": list(names)}
            links = DC.link_traces(
                np.where(rows_t, np.asarray(locs["tid"], np.int64), -1), colour, t_all, x, y,
                max_dt=p.max_dt_ms * 1e-3, max_distance=p.max_distance_nm,
                min_matched=p.min_matched, min_fraction=p.min_fraction,
                min_overlap=p.min_overlap_s, offset=None if settings.apply else curve)
            has = links.pair_id >= 0
            pair_id[has] = links.pair_id[has] + next_id
            ids_all.append(links.ids + next_id)
            for key, v in links.values().items():
                values_all.setdefault(key, []).append(v)
            next_id += len(links.ids)
            n_both += int(links.both.sum())
            per_target.append((m, names, curve, how, fit, drift, bead_info, links,
                               settings.apply))
            c = curve.c
            lines.append(f"{m.title or 'the table'}: offset {names[1]} - {names[0]} "
                         f"({c[0]:.1f}, {c[1]:.1f}) nm, {how}"
                         + (f"; {curve.note}" if curve.note else ""))
            if drift is not None:
                d_end = drift(np.array([span[1]]))[0]
                lines.append(f"  MBM colour drift over the measurement: ({d_end[0]:.1f}, "
                             f"{d_end[1]:.1f}) nm, from {int(drift.beads.used.sum())} of "
                             f"{len(drift.names)} beads"
                             + (f", recorded over {drift.span[1] - drift.span[0]:.0f} s of "
                                f"{span[1] - span[0]:.0f}, held flat after"
                                if drift.span[1] < span[1] - 0.1 * (span[1] - span[0])
                                else ""))
            if bead_info and bead_info.get("warn"):
                ctx.warn(bead_info["warn"])
        cols["x_nm"] = x.astype(np.asarray(locs["x_nm"]).dtype)
        cols["y_nm"] = y.astype(np.asarray(locs["y_nm"]).dtype)
        for c in nc:
            cols[c] = ncx[c].astype(np.asarray(locs[c]).dtype)
        cols["pair_id"] = pair_id
        meta = dict(locs.metadata)
        meta[RECORD] = record
        out = Localizations(cols, meta)
        remember(out, "pair_id", "", grouped="max")
        ids = np.concatenate(ids_all)
        values = {k: np.concatenate(v) for k, v in values_all.items()}
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(out, "pair_id", LABEL, ids, values, step,
                             layer=ctx.layer, session=ctx.session)
        transformations = {}
        c0 = per_target[0][2].c
        tname = settings.name.strip() or (f"{first_names[1]} onto {first_names[0]}")
        if np.all(np.isfinite(c0)):
            transformations[tname] = Transformation(
                "affine", shift(-c0[0], -c0[1]),
                description=f"{first_names[1]} onto {first_names[0]}: shift by "
                            f"({-c0[0]:.1f}, {-c0[1]:.1f}) nm",
                origin={"plugin": self.path})
        n_ids = len(ids)
        lines.insert(0, f"{n_ids} molecules: {n_both} in both colours, {n_ids - n_both} in one")
        lines.append(f"pair_id written; the constant offset kept as {tname!r} for Transform"
                     + ("" if settings.apply else "; positions not moved"))

        def draw(figure):
            draw_dual(figure, per_target)

        return Result(locs=out, text="\n".join(lines),
                      plot=Plot(draw, panels=4, size=(14, 3.4)),
                      id_regions={"pair_id": {LABEL: table}}, transformations=transformations,
                      data={"targets": [(m.title, curve, how, fit, drift, links)
                                        for m, _, curve, how, fit, drift, _, links, _
                                        in per_target]},
                      settings=settings, title="MINFLUX dual colour")

    # ------------------------------------------------------------ helpers

    def _drift(self, ctx: Context, m: _Measurement, knot: float):
        """The MBM colour drift of measurement ``m``, None where its beads
        were not recorded in both colours."""
        from ..dual_colour import mbm_colour_drift
        from ..mbm import measurements_of
        from ..metadata import file_metadata
        locs = ctx.locs
        rows = ((np.asarray(locs["filenumber"]) == m.file) if "filenumber" in locs
                else np.ones(len(locs), bool))
        try:
            ms = measurements_of(file_metadata(locs.metadata, m.file), locs[rows])
        except Exception as e:                      # noqa: BLE001 -- the drift is optional
            ctx.warn(f"{m.title}: the MBM beads could not be read ({e})")
            return None
        mine = [b for b in ms if b.label == (m.label or b.label)]
        if not mine:
            return None
        b = mine[0]
        return mbm_colour_drift(b.beads, list(b.info), knot=knot)

    def _offset(self, ctx, m, found, targets, rows_t, colour, t_all, x, y, drift, settings):
        """The offset curve for target ``m``, how it was found, the fit (from
        the molecules) and what the bead measurements said."""
        from .. import dual_colour as DC
        o, p = settings.offset, settings.pairing
        src = o.source
        zero_drift_t0 = drift.span[0] if drift is not None else 0.0
        if src == "none":
            return DC.OffsetCurve(np.zeros(2)), "none", None, None
        if src == "typed":
            return (DC.OffsetCurve(np.array([o.dx_nm, o.dy_nm], float)), "typed", None, None)
        if src == "named":
            tr = (getattr(ctx.session, "transformations", {}) or {}).get(o.named)
            if tr is None:
                raise ValueError(f"no transformation {o.named!r}: choose one of the "
                                 "session's, or another source")
            # the transformation moves colour 1 onto colour 0: the offset is
            # where it takes a point, reversed
            px, py, _ = tr.apply(np.array([0.0]), np.array([0.0]))
            return (DC.OffsetCurve(np.array([-px[0], -py[0]], float)),
                    f"the transformation {o.named!r}", None, None)
        if src in ("auto", "beads"):
            before, after = self._bead_measurements(m, found, targets, o)
            if before is None and after is None:
                if src == "beads":
                    raise ValueError("no bead measurement open: open the one taken before "
                                     "the tracking, choose it in 'beads before', or take "
                                     "the offset from the molecules")
            else:
                return self._from_beads(ctx, m, before, after, colour, t_all, x, y, drift, p)
        # from the molecules: a loose pass, the fit, done again by the caller
        locs = ctx.locs
        drift_only = DC.OffsetCurve(np.zeros(2), drift, zero_drift_t0)
        loose = DC.link_traces(
            np.where(rows_t, np.asarray(locs["tid"], np.int64), -1), colour, t_all, x, y,
            max_dt=p.max_dt_ms * 1e-3, max_distance=max(FIRST_PASS_NM, 3 * p.max_distance_nm),
            min_matched=p.min_matched, min_fraction=p.min_fraction,
            min_overlap=p.min_overlap_s, offset=drift_only)
        both = loose.both & np.isfinite(loose.dx)
        if not both.any():
            raise ValueError("no molecule seen in both colours to measure the offset in: "
                             "open a bead measurement, or type the offset")
        # each pair's offset less the colour drift at its time
        dd = (drift_only(loose.time[both]) if drift is not None
              else np.zeros((both.sum(), 2)))
        fit = DC.fit_track_offset(loose.dx[both] - dd[:, 0], loose.dy[both] - dd[:, 1],
                                  loose.ux[both], loose.uy[both],
                                  np.nan_to_num(loose.displacement_nm[both]) > o.still_nm)
        how = (f"fitted over {int(fit.used.sum())} molecules ({fit.n_moving} moving, "
               f"{fit.n_still} still), ± ({fit.c_err[0]:.1f}, {fit.c_err[1]:.1f}) nm"
               + (f"; lead of colour 1 along the motion {fit.a:.1f} ± {fit.a_err:.1f} nm"
                  if np.isfinite(fit.a) else "; the tracks' directions too few to tell a "
                                             "lead"))
        return DC.OffsetCurve(fit.c, drift, zero_drift_t0), how, fit, None

    def _bead_measurements(self, m, found, targets, o):
        """The bead measurements before and after target ``m``: as chosen,
        or the nearest in time of the open ones that are not targets."""
        by_did = {x.did: x for x in found}
        target_dids = {t.did for t in targets}
        others = [x for x in found if x.did not in target_dids]
        before = by_did.get(o.before) if o.before else None
        after = None if o.after == "-" else (by_did.get(o.after) if o.after else None)
        if m.start_abs is None:
            return (before or (others[0] if others and not o.before else None)), after
        if before is None and not o.before:
            earlier = [x for x in others if x.start_abs is not None
                       and x.start_abs <= m.start_abs]
            before = max(earlier, key=lambda x: x.start_abs) if earlier else None
        if after is None and not o.after:
            later = [x for x in others if x.start_abs is not None
                     and x.start_abs > m.start_abs]
            after = min(later, key=lambda x: x.start_abs) if later else None
        if before is None and after is not None:
            before, after = after, None
        return before, after

    def _from_beads(self, ctx, m, before, after, colour, t_all, x, y, drift, p):
        from .. import dual_colour as DC
        locs = ctx.locs
        tid = np.asarray(locs["tid"], np.int64)

        def measured(b: _Measurement):
            rows = _rows_of(locs, b.file, b.number)
            res = DC.bead_offset(np.where(rows, tid, -1), colour, t_all, x, y,
                                 p.max_dt_ms * 1e-3, p.min_matched)
            # its time on the target's clock: through the acquisition dates
            if b.start_abs is not None and m.start_abs is not None:
                t = (b.start_abs + res.time - b.time_offset) - m.start_abs + m.time_offset
            else:
                t = res.time
            return res, t

        rb, tb = measured(before)
        if after is not None:
            ra, ta = measured(after)
            curve = DC.offset_curve((rb.offset, tb), (ra.offset, ta), drift)
            how = (f"beads before ({before.title}) and after ({after.title}), "
                   f"interpolated")
        else:
            curve = DC.offset_curve((rb.offset, tb), None, drift)
            how = f"beads in {before.title} ({rb.n_pairs} emitters, {rb.n_matched} matches)"
        info = {}
        if drift is not None and (tb < drift.span[0] or after is not None):
            gap = drift.span[0] - tb
            if gap > 0:
                info["warn"] = (f"the bead measurement was taken {gap:.0f} s before the MBM "
                                "beads' first point: the colours' drift in between is not "
                                "measured")
        elif drift is None and m.start_abs is not None and before.start_abs is not None:
            gap = m.start_abs - before.start_abs
            if gap > 60:
                info["warn"] = (f"the bead measurement was taken {gap:.0f} s before the "
                                "tracking, with no MBM colour drift to follow the colours "
                                "in between")
        return curve, how, None, info


def draw_dual(figure, per_target) -> None:
    """The pairs' offsets against the curve used, over time, the MBM colour
    drift, and the lead along the track."""
    a, b, c, d = figure.subplots(1, 4)
    _, names, curve, _, fit, drift, _, links, applied = per_target[0]
    both = links.both & np.isfinite(links.dx)
    # the pairs' offsets as measured: with the shift added back where it was applied
    seen = np.column_stack([links.dx[both], links.dy[both]])
    if applied and both.any():
        seen = seen + curve(links.time[both])
    a.plot(seen[:, 0], seen[:, 1], "o", ms=4, color="0.35", label="molecules")
    a.plot([curve.c[0]], [curve.c[1]], "x", ms=11, mew=2, color="tab:red", label="offset")
    a.axhline(0, color="0.85", lw=0.6)
    a.axvline(0, color="0.85", lw=0.6)
    a.set_xlabel(f"dx (nm), {names[1]} - {names[0]}")
    a.set_ylabel("dy (nm)")
    a.set_aspect("equal", adjustable="datalim")
    a.legend(fontsize=6, frameon=False)
    a.set_title("offset per molecule", fontsize=8)
    if both.any():
        order = np.argsort(links.time[both])
        tt = links.time[both][order]
        b.plot(tt, seen[order, 0], "o", ms=3, color="tab:blue", label="dx")
        b.plot(tt, seen[order, 1], "o", ms=3, color="tab:orange", label="dy")
        grid = np.linspace(tt.min(), tt.max(), 200) if len(tt) > 1 else tt
        cv = curve(grid)
        b.plot(grid, cv[:, 0], "-", color="tab:blue", lw=1)
        b.plot(grid, cv[:, 1], "-", color="tab:orange", lw=1)
        b.legend(fontsize=6, frameon=False)
    b.set_xlabel("time (s)")
    b.set_ylabel("offset (nm)")
    b.set_title("over time: molecules and the curve used", fontsize=8)
    if drift is not None:
        s = drift.beads
        for k, name in enumerate(drift.names):
            rows = drift.bead == k
            o = s.offsets_nm[k] if k < len(s.offsets_nm) else np.zeros(2)
            style = "-" if s.used[k] else "--"
            c.plot(drift.t[rows], drift.d[rows, 0] - o[0], style, lw=0.8, color="tab:blue",
                   alpha=0.6 if s.used[k] else 0.3)
            c.plot(drift.t[rows], drift.d[rows, 1] - o[1], style, lw=0.8, color="tab:orange",
                   alpha=0.6 if s.used[k] else 0.3)
        g = np.linspace(*drift.span, 200)
        cv = drift(g)
        c.plot(g, cv[:, 0], color="tab:blue", lw=2, label="dx")
        c.plot(g, cv[:, 1], color="tab:orange", lw=2, label="dy")
        c.legend(fontsize=6, frameon=False)
        c.set_xlabel("time (s)")
        c.set_ylabel("colour drift (nm)")
        c.set_title(f"MBM colour drift, {int(s.used.sum())} of {len(drift.names)} beads",
                    fontsize=8)
    else:
        c.set_axis_off()
        c.text(0.5, 0.5, "no MBM beads in both colours", ha="center", va="center",
               transform=c.transAxes, fontsize=8)
    moving = both & (np.nan_to_num(links.displacement_nm) > 30)
    if moving.any():
        res = seen[moving[both]] - curve.c
        along = res[:, 0] * links.ux[moving] + res[:, 1] * links.uy[moving]
        d.hist(along, bins=max(5, min(30, len(along))), color="0.6")
        if fit is not None and np.isfinite(fit.a):
            d.axvline(fit.a, color="tab:red", lw=1.5, label=f"lead {fit.a:.1f} nm")
            d.legend(fontsize=6, frameon=False)
        d.set_xlabel(f"{names[1]} ahead of {names[0]} along the motion (nm)")
        d.set_ylabel("molecules")
        d.set_title("lead along the track", fontsize=8)
    else:
        d.set_axis_off()
        d.text(0.5, 0.5, "no moving molecule in both colours", ha="center", va="center",
               transform=d.transAxes, fontsize=8)
