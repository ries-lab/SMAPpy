"""MBM: Imspector's beam-line monitoring beads, chosen again.

Imspector corrects a MINFLUX measurement by the plain mean displacement of
the beads it used (NOTES.md, "MINFLUX beam-line monitoring beads"): a bead
that wanders or jumps moves every localization by its share, and the
beads' scatter -- ~10 nm point to point in 3D -- goes into the correction
undamped.  Here the beads' tracks (`io.msr.bead_table`) are judged against
each other and fitted with one spline (`beads.select_beads`), and the
localizations are corrected from scratch: ``x = x_nc - d(t)``, Imspector's
uncorrected position (`x_nc_nm`) minus the new curve at the localization's
time.  Replacing, never adding: the tracks are raw, so the curve already
holds everything Imspector corrected, and on top of `x_nm` it would count
twice.  Run again, it gives the same table.

**Per measurement.**  Each measurement has its own beads -- Imspector
finds them anew, and reuses the names -- so each is fitted on its own and
the curves are laid end to end on the table's clock (a merged file's
measurements follow one another in `time_s`).  A curve covers its
measurement's localizations as well as its beads, held flat beyond the
beads, so no localization is interpolated towards the next measurement.

**Names.**  A bead is named as Imspector names it (R0, R1, ...), and where
the selection holds several measurements, after the measurement's label up
to its first underscore (``261019-133540/R0``), which is unique to the
second; the choices made by hand (*by hand*) are kept by these names.

**Imspector's choice** is where the automatic one starts, by default: a bead
Imspector did not use stays out unless ticked in.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beads import BeadSet, BeadView, MadRule, format_overrides, parse_overrides, select_beads
from .drift import REFERENCES, Drift
from .locs import Localizations
from .params import param

START = ("Imspector's beads", "all beads")


@dataclass
class RejectionSettings:
    k: float = param(3.0, label="robust sigmas", min=0.0,
                     help="a bead is out where its noise, wander or jump is this "
                          "many robust standard deviations above the others'")
    margin_nm: float = param(2.0, label="margin", unit="nm", min=0.0,
                             help="and above the others' median by at least this")
    min_beads: int = param(2, label="min beads", min=1, max=100,
                           help="never fewer beads than this")


@dataclass
class MBMSettings:
    knot_s: float = param(8.0, label="knot spacing", unit="s", min=0.5, max=1000.0,
                          help="the drift is a spline with a coefficient this often")
    start: str = param(START[0], label="start from", choices=START,
                       help="the beads the automatic choice may use")
    rejection: RejectionSettings = param(default_factory=RejectionSettings,
                                         label="rejection")
    beads: str = param("", label="by hand",
                       help="beads in or out whatever the rule says, as "
                            "'R0 out, R3 in'; Select beads... writes it")
    use_z: Optional[bool] = param(None, label="correct z",
                                  help="auto: if the measurement is 3D")
    reference: str = param("as estimated", label="reference", choices=REFERENCES,
                           advanced=True,
                           help="as estimated: zero at each measurement's first "
                                "bead point, as Imspector's")


@dataclass
class Measurement:
    """One measurement's beads, as the fit wants them."""
    label: str
    prefix: str                     # what its bead names carry ("" if alone)
    beads: Localizations            # its rows of the bead table
    info: List[Dict]                # `bead_table`'s per-bead entries
    span: Tuple[float, float]       # the times its localizations cover


def measurements_of(metadata: Dict, locs: Optional[Localizations] = None,
                    several: bool = False) -> List[Measurement]:
    """The measurements of a file's metadata that recorded beads.

    ``locs`` (the file's rows, with `time_s` and `measurement`) gives each
    one's span; ``several`` names beads after their measurement even when
    the file has only one, as when the selection holds several files.
    """
    from .io.msr import bead_table
    metas = ((metadata or {}).get("minflux") or {})
    metas = metas.get("measurements") if isinstance(metas, dict) else None
    if not metas:
        return []
    table = bead_table(metas)
    if table is None:
        return []
    out = []
    named = several or len(metas) > 1
    for k, meta in enumerate(metas):
        info = [b for b in table.metadata["mbm_beads"] if b["measurement"] == k]
        if not info:
            continue
        rows = (np.ones(len(table), bool) if "measurement" not in table
                else np.asarray(table["measurement"]) == k)
        beads = table[rows]
        label = str(meta.get("label") or f"measurement {k + 1}")
        span = (float(beads["time_s"].min()), float(beads["time_s"].max()))
        if locs is not None and "time_s" in locs and len(locs):
            mine = (np.ones(len(locs), bool) if "measurement" not in locs
                    else np.asarray(locs["measurement"]) == k)
            if mine.any():
                t = np.asarray(locs["time_s"])[mine]
                span = (min(span[0], float(t.min())), max(span[1], float(t.max())))
        out.append(Measurement(label, label.split("_")[0] + "/" if named else "",
                               beads, info, span))
    return out


def _axes(m: Measurement, use_z: Optional[bool]) -> Tuple[str, ...]:
    has_z = "z_nm" in m.beads
    if use_z and not has_z:
        raise ValueError(f"{m.label}: correct z is on but its beads have no z (2D)")
    return ("x", "y", "z") if (has_z if use_z is None else use_z) else ("x", "y")


def choose(m: Measurement, settings: MBMSettings) -> Tuple[BeadSet, Tuple[str, ...]]:
    """The beads of one measurement, judged, and the curve they give."""
    axes = _axes(m, settings.use_z)
    names = [m.prefix + str(b["name"] or b["gri"]) for b in m.info]
    order = np.argsort([b["tid"] for b in m.info])
    names = [names[i] for i in order]
    info = [m.info[i] for i in order]
    candidates = None
    if settings.start == START[0]:
        candidates = np.array([b["used"] is not False for b in info])
    r = settings.rejection
    every = parse_overrides(settings.beads)
    chosen = select_beads(np.asarray(m.beads["time_s"]),
                          np.column_stack([np.asarray(m.beads[f"{a}_nm"]) for a in axes]),
                          np.asarray(m.beads["tid"]), names, knot=settings.knot_s,
                          rule=MadRule(k=r.k, margin_nm=r.margin_nm),
                          overrides={k: v for k, v in every.items() if k in names},
                          min_beads=r.min_beads, candidates=candidates)
    return chosen, axes


def drift_of(measurements: Sequence[Measurement], chosen: Sequence[BeadSet],
             settings: MBMSettings) -> Drift:
    """The measurements' curves end to end, each over its span."""
    times, values = [], []
    for m, s in zip(measurements, chosen):
        knot = settings.knot_s
        step = min(knot / 8.0, 0.5)
        lo, hi = m.span
        t = np.arange(lo, hi + 0.5 * step, step) if hi > lo else np.array([lo])
        d = s.drift_nm(t)
        if settings.reference == "first frame":
            d = d - d[0]
        elif settings.reference == "last frame":
            d = d - d[-1]
        if times and t[0] <= times[-1][-1]:          # never back in time
            keep = t > times[-1][-1]
            t, d = t[keep], d[keep]
        times.append(t)
        values.append(d)
    return Drift(np.concatenate(values), settings, method="mbm",
                 n_used=int(sum(s.used.sum() for s in chosen)),
                 reference=settings.reference, times_s=np.concatenate(times))


def replace_correction(locs: Localizations, drift: Drift, rows=None) -> Localizations:
    """``x = x_nc - d(t)`` on ``rows``: Imspector's correction undone and
    ``drift`` applied instead.  Where a file has no `x_nc_nm` -- Imspector
    corrected nothing, or the table is the beads' own -- `x_nm` is raw."""
    columns = dict(locs.columns)
    mask = np.ones(len(locs), bool) if rows is None else np.asarray(rows, bool)
    for a in ("x", "y", "z"):
        raw = columns.get(f"{a}_nc_nm")
        if raw is None or f"{a}_nm" not in columns:
            continue
        values = np.asarray(columns[f"{a}_nm"]).copy()
        values[mask] = np.asarray(raw)[mask]
        columns[f"{a}_nm"] = values
    return drift.apply(Localizations(columns, dict(locs.metadata)), rows=rows)


def describe_choice(chosen: Sequence[BeadSet]) -> str:
    n = sum(len(s) for s in chosen)
    used = sum(int(s.used.sum()) for s in chosen)
    auto = sum(int(s.outlier.sum()) for s in chosen)
    hand = sum(len(s.overrides) for s in chosen)
    parts = [f"{n} beads: {used} used"]
    if auto:
        parts.append(f"{auto} left out automatically")
    if hand:
        parts.append(f"{hand} chosen by hand")
    return ", ".join(parts)


def bead_view(measurements: Sequence[Measurement], settings: MBMSettings,
              source: int = 0, sources: Optional[List[str]] = None) -> BeadView:
    """What the bead window shows of measurement ``source``."""
    m = measurements[source]
    chosen, axes = choose(m, settings)
    order = np.argsort([b["tid"] for b in m.info])
    info = [m.info[i] for i in order]
    where = np.array([[np.median(np.asarray(m.beads["x_nm"])[m.beads["tid"] == b["tid"]]),
                       np.median(np.asarray(m.beads["y_nm"])[m.beads["tid"] == b["tid"]])]
                      for b in info])
    columns = {"Imspector": ["used" if b["used"] else ("not used" if b["used"] is False
                                                        else "?") for b in info]}
    if any(b.get("sigpos_nm") for b in info):
        columns["sigpos (nm)"] = [b.get("sigpos_nm") for b in info]
    return BeadView(chosen, axes, sources or [x.label for x in measurements], source,
                    columns, where, [str(b["name"] or b["gri"]) for b in info],
                    title=m.label)


def with_overrides(settings: MBMSettings, names: Sequence[str],
                   overrides: Dict[str, bool]) -> MBMSettings:
    """``settings`` with the choices by hand for the beads ``names`` (one
    measurement's) replaced by ``overrides``; other measurements' kept."""
    every = {k: v for k, v in parse_overrides(settings.beads).items() if k not in names}
    every.update({k: bool(v) for k, v in overrides.items()})
    return replace(settings, beads=format_overrides(every))
