"""MBM drift correction: Imspector's beads chosen again, the localizations
corrected from scratch.  The work is in `smappy.mbm` and `smappy.beads`
(their docstrings have the decisions); this is the plugin around them.

The beads are read from each file's own metadata, so a chain or a batch
needs no bead file; a bead file opened beside a measurement
(`io.msr.bead_table`) is corrected with it, so that it shows what is left
of each bead's motion.  The choices made by hand live in the settings (*by
hand*), which the bead window (`gui.bead_window`, *Select beads...*) writes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mbm import (MBMSettings, Measurement, bead_view, choose, describe_choice,
                   drift_of, measurements_of, replace_correction, with_overrides)
from ..metadata import file_metadata
from . import Context, FormState, Notice, Plot, Plugin, Result, register
from .drift_result import KeepsDrift, drift_result


def _files(ctx: Context) -> List[Tuple[int, str, np.ndarray, List[Measurement]]]:
    """The files of the selection with MBM beads: number, name, its rows,
    its measurements."""
    locs = ctx.locs
    mask = np.asarray(ctx.selection.mask, bool)
    numbers = (np.asarray(locs["filenumber"]) if "filenumber" in locs
               else np.zeros(len(locs), np.int64))
    present = [int(k) for k in np.unique(numbers[mask])] if len(locs) else []
    names = list(getattr(ctx.session, "file_names", lambda: [])() or [])
    candidates = []
    for k in present:
        meta = file_metadata(locs.metadata, k)
        if "mbm_of" in meta:                     # a bead file: corrected with its own
            continue
        candidates.append((k, meta))
    out = []
    for k, meta in candidates:
        rows = numbers == k
        ms = measurements_of(meta, locs[rows], several=len(candidates) > 1)
        if ms:
            out.append((k, names[k] if k < len(names) else f"file {k + 1}", rows, ms))
    return out


@register("Analysis/Drift/MBM")
class MBMDrift(KeepsDrift, Plugin):
    description = ("Choose a MINFLUX measurement's MBM beads again -- leaving out "
                   "those that wander, jump or are noisy -- and correct the "
                   "localizations with the drift the others agree on, in place "
                   "of Imspector's.")
    name = "MBM drift correction"
    changes_locs = True
    # the correction moves every localization of the file
    grouping = "ungrouped"
    Settings = MBMSettings
    version = "1"

    def _sources(self, ctx) -> List[Tuple[str, Measurement]]:
        out = []
        for _, name, _, ms in _files(ctx):
            for m in ms:
                out.append((f"{name}: {m.label}" if len(ms) > 1 else name, m))
        return out

    def run(self, ctx: Context, settings: MBMSettings) -> Result:
        files = _files(ctx)
        if not files:
            raise ValueError("no MINFLUX measurement with MBM beads in the selection: "
                             "the beads come from an Abberior .msr")
        locs = ctx.locs
        numbers = (np.asarray(locs["filenumber"]) if "filenumber" in locs else None)
        session_files = list(getattr(ctx.session, "files", None) or [])
        out, drifts, chosen_all, views = locs, [], [], []
        for k, name, rows, ms in files:
            chosen = []
            for m in ms:
                ctx.report(f"{name}: {m.label}")
                seen = float(np.ptp(np.asarray(m.beads["time_s"])))
                if seen < 0.5 * (m.span[1] - m.span[0]):
                    # a file that kept the beads from before the start only
                    ctx.warn(f"{m.label}: the beads were recorded over {seen:.0f} s "
                             f"of {m.span[1] - m.span[0]:.0f}; the drift after that "
                             "is held flat")
                s, axes = choose(m, settings)
                chosen.append(s)
                views.append((m.label, s, axes))
            drift = drift_of(ms, chosen, settings)
            whole = numbers is None or numbers.min() == numbers.max()
            out = replace_correction(out, drift, None if whole else rows)
            if session_files and numbers is not None:
                from ..io.msr import mbm_partner
                partner = mbm_partner(session_files, k)
                if partner is not None:
                    out = drift.apply(out, rows=numbers == partner)
            drifts.append((name, drift))
            chosen_all += chosen
        result = drift_result(out, drifts, settings, lambda d: d.plot,
                              beads=describe_choice(chosen_all))
        result.text = describe_choice(chosen_all) + "\n" + result.text
        from ..beads import draw_bead_tracks
        for label, s, axes in views:
            result.plots = dict(result.plots or {})
            result.plots[f"beads {label}" if len(views) > 1 else "beads"] = Plot(
                lambda figure, s=s, axes=axes: draw_bead_tracks(figure, s, axes),
                name="beads", panels=2 * len(axes), size=(9, 2.0 * len(axes) + 0.8))
        return result

    def describe(self, ctx: Optional[Context], settings: MBMSettings) -> FormState:
        """How many beads, how many used, and the button to the bead window."""
        if ctx is None or ctx.locs is None or not len(ctx.locs):
            return FormState()
        sources = self._sources(ctx)
        if not sources:
            return FormState(notice=Notice(
                "no MINFLUX measurement with MBM beads in the selection", kind="info"))
        chosen = [choose(m, settings)[0] for _, m in sources]
        return FormState(notice=Notice(describe_choice(chosen), "Select beads...",
                                       "bead selector", kind="info"))

    # -- the bead window's interface (`gui.bead_window`)
    def bead_view(self, ctx: Context, settings: MBMSettings, source: int = 0):
        sources = self._sources(ctx)
        if not sources:
            raise ValueError("no MINFLUX measurement with MBM beads in the selection")
        source = min(max(source, 0), len(sources) - 1)
        return bead_view([m for _, m in sources], settings, source,
                         [label for label, _ in sources])

    def with_overrides(self, settings: MBMSettings, view, overrides: Dict[str, bool]
                       ) -> MBMSettings:
        return with_overrides(settings, view.beads.names, overrides)
