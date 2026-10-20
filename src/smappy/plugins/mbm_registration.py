"""MBM registration: two MINFLUX measurements put onto each other by their beads.

The algorithm is `smappy.mbm_registration` (its docstring has the
decisions: start positions, matching by shifts rather than names, a
translation or a rigid map); this is the plugin around it.  The beads come
from each file's own metadata (`mbm.measurements_of`), as *MBM drift
correction*'s do, so no bead file need be open.  What it hands back is a
named transformation (`Result.transformations`), which *Transform* applies
to the target's file -- the way *Register layers* hands on what it measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

FITS = (("auto", "auto: rigid from three beads"), ("translation", "translation"),
        ("rigid", "rigid: rotation and translation"))


def measurements(ctx) -> List[Tuple[str, int, object]]:
    """``(label, file number, Measurement)`` of every measurement with MBM
    beads in the session's files."""
    from ..mbm import measurements_of
    from ..metadata import file_metadata
    session = ctx.session
    locs = ctx.locs if session is None else session.locs
    if locs is None or not len(locs):
        return []
    numbers = (np.unique(np.asarray(locs["filenumber"])).astype(int).tolist()
               if "filenumber" in locs else [0])
    names = list(getattr(session, "file_names", lambda: [])() or []) if session else []
    out = []
    for k in numbers:
        meta = file_metadata(locs.metadata, k)
        if "mbm_of" in meta:                    # a bead file, not a measurement
            continue
        for m in measurements_of(meta, several=True):
            name = names[k] if k < len(names) else f"file {k + 1}"
            out.append((f"{name}: {m.label}", k, m))
    return out


def _choices():
    return [("", "choose...")]


@dataclass
class MBMRegistrationSettings:
    reference: str = param("", label="reference", choices=_choices,
                           help="the measurement the other is put onto")
    target: str = param("", label="target", choices=_choices,
                        help="the measurement that moves")
    fit: str = param("auto", label="fit", choices=FITS,
                     help="a translation, or with three beads or more a rotation too")
    tolerance_nm: float = param(200.0, label="tolerance", unit="nm", min=1.0,
                                help="how far two beads' shifts may differ and agree")
    max_shift_nm: float = param(5000.0, label="largest shift", unit="nm", min=10.0,
                                advanced=True,
                                help="no pair of beads further apart than this")
    name: str = param("", label="keep as",
                      help="the transformation's name; empty: target onto reference")


@register("Analysis/Register/MBM registration")
class MBMRegistration(Plugin):
    """Put one MINFLUX measurement onto another by the MBM beads both saw:
    a translation, or a rotation too, kept as a named transformation that
    Transform applies."""

    Settings = MBMRegistrationSettings
    changes_locs = False
    version = "2"

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = FormState()
        if ctx is None or ctx.session is None:
            return state
        found = measurements(ctx)
        choices = [("", "choose..." if found else "no measurement with MBM beads")]
        choices += [(label, label) for label, _, _ in found]
        state.choices["reference"] = choices
        state.choices["target"] = choices
        name = self._name(settings)
        state.hints["name"] = name or "target onto reference"
        kept = ctx.session.transformations.get(name)
        target = next((k for label, k, _ in found if label == settings.target), None)
        if kept is not None and target is not None:
            from .transform import settings_for
            state.notice = Notice(f"{name}: {kept}", action="Apply with Transform",
                                  window="transform",
                                  settings=settings_for(name, file=target), kind="info")
        return state

    @staticmethod
    def _name(settings) -> str:
        if settings.name.strip():
            return settings.name.strip()
        if settings.target and settings.reference:
            return f"{settings.target} onto {settings.reference} (MBM)"
        return ""

    def run(self, ctx: Context, settings: MBMRegistrationSettings) -> Result:
        from ..mbm import MBMSettings, choose
        from ..mbm_registration import register as fit_map, start_positions
        from ..transformation import Transformation
        found = {label: (k, m) for label, k, m in measurements(ctx)}
        if not found:
            raise ValueError("no MINFLUX measurement with MBM beads is open: the "
                             "beads come from an Abberior .msr")
        for role in ("reference", "target"):
            if getattr(settings, role) not in found:
                raise ValueError(f"choose the {role} measurement among: "
                                 + "; ".join(found))
        if settings.reference == settings.target:
            raise ValueError("the reference and the target are the same measurement")

        def positions(m):
            # the beads' drift, as MBM drift correction fits it: a bead's
            # position is its offset about it, zero at the first bead point
            chosen, axes = choose(m, MBMSettings())
            names = {b["tid"]: str(b["name"] or b["gri"]) for b in m.info}
            ids = sorted(names)
            return start_positions(np.asarray(m.beads["time_s"]),
                                     np.column_stack([np.asarray(m.beads[f"{a}_nm"])
                                                      for a in axes]),
                                     np.asarray(m.beads["tid"]),
                                     [names[i] for i in ids], curve=chosen.drift_nm)
        (_, ref_m), (target_file, tgt_m) = found[settings.reference], found[settings.target]
        reference, target = positions(ref_m), positions(tgt_m)
        rigid = None if settings.fit == "auto" else settings.fit == "rigid"
        found_map = fit_map(reference, target, settings.tolerance_nm,
                            settings.max_shift_nm, rigid)
        name = self._name(settings)
        t = Transformation("affine", found_map.matrix, description=str(found_map),
                           origin={"method": "MBM registration",
                                   "reference": settings.reference,
                                   "target": settings.target,
                                   "pairs": [[reference.names[i], target.names[j]]
                                             for i, j in found_map.pairs]})
        lines = [f"{name}: {found_map}", "pairs (reference, target, residual nm):"]
        lines += [f"  {reference.names[i]}\t{target.names[j]}\t{r:.2f}"
                  for (i, j), r in zip(found_map.pairs, found_map.residuals_nm)]
        unpaired = len(target.names) - len(found_map.pairs)
        if unpaired:
            lines.append(f"{unpaired} bead{'s' if unpaired != 1 else ''} of the target "
                         "found no partner")
        lines.append(f"kept as {name!r}; Transform applies it to the target's file")

        def draw(ax):
            draw_registration(ax, found_map)
        return Result(text="\n".join(lines), plot=Plot(draw, size=(5.0, 4.4)),
                      settings=settings, transformations={name: t},
                      values={"pairs": len(found_map.pairs), "rms_nm": found_map.rms_nm,
                              "shift_x_nm": float(found_map.shift_nm[0]),
                              "shift_y_nm": float(found_map.shift_nm[1]),
                              "angle_deg": found_map.angle_deg},
                      data={"registration": found_map, "transformation": t,
                            "target_file": target_file})


def draw_registration(ax, found) -> None:
    """The reference beads, the target's before and after the map, and the
    pairs, in µm."""
    ref = found.reference.xyz[:, :2] / 1000
    tgt = found.target.xyz[:, :2] / 1000
    m = found.matrix
    mapped = ((m[:2, :2] @ found.target.xyz[:, :2].T).T + m[:2, 3]) / 1000
    ax.scatter(ref[:, 0], ref[:, 1], s=60, facecolors="none", edgecolors="k",
               label="reference")
    ax.scatter(tgt[:, 0], tgt[:, 1], s=14, color="0.6", label="target")
    ax.scatter(mapped[:, 0], mapped[:, 1], s=14, color="tab:red", label="target, mapped")
    for (i, j) in found.pairs:
        ax.plot([tgt[j, 0], mapped[j, 0]], [tgt[j, 1], mapped[j, 1]], color="0.7",
                linewidth=0.6)
    for k, name in enumerate(found.reference.names):
        ax.annotate(name, ref[k], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{found.kind}, {len(found.pairs)} beads, "
                 f"residual {found.rms_nm:.1f} nm rms", fontsize=8)
