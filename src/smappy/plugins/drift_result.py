"""What the drift plugins keep in the file, and how they draw it again.

COMET, RCC and the fiducial fit differ in how they find the curve and in nothing else afterwards,
so what is worth saving -- the curve itself -- and how it is drawn belong to
neither of them.  The mixin is what a plugin adds to say "my result is a
drift", and is the model for any other tool with a figure worth keeping.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..drift import Drift
from ..locs import Localizations
from . import Context, Plot, Result, Stopped


def per_file(ctx: Context, estimate: Callable[[np.ndarray], Drift]
             ) -> Tuple[Localizations, List[Tuple[str, Drift]]]:
    """Estimate and correct each file of the selection on its own.

    Drift is of one acquisition: files opened together share frame numbers,
    so one curve over two of them averages frame 100 of the one with frame
    100 of the other, and applying it to the whole table moved files the
    selection never held by another file's curve.  So ``estimate(select)`` is
    called once per file the selection holds, with the selection cut to that
    file, and its curve moves that file's rows only; files outside the
    selection are left as they were.  A table of one file is corrected
    whole, as before, which also writes its ``drift_correction`` record.
    """
    locs = ctx.locs
    mask = np.asarray(ctx.selection.mask, bool)
    files = (np.asarray(locs["filenumber"]) if "filenumber" in locs
             else np.zeros(len(locs), np.int64))
    # bincount and min/max rather than np.unique: a sort of every row
    numbers = [int(k) for k in np.flatnonzero(np.bincount(files[mask]))]
    if not len(files) or files.min() == files.max():
        drift = estimate(mask)
        return drift.apply(locs), [("", drift)]
    names = list(getattr(ctx.session, "file_names", lambda: [])() or [])
    out, drifts = locs, []
    for k in numbers:
        name = names[k] if 0 <= k < len(names) else f"file {k + 1}"
        rows = files == k
        if ctx.stopped:
            raise Stopped("stopped between files: the table is left as it was")
        ctx.report(f"{name}: {int((mask & rows).sum()):,} localizations")
        drift = estimate(mask & rows)
        # the estimate reads the table as it came: other files' rows are all
        # that has changed in `out`, and a file is estimated from its own
        out = drift.apply(out, rows=rows)
        drifts.append((name, drift))
    return out, drifts


def drift_result(corrected: Localizations, drifts: List[Tuple[str, Drift]],
                 settings, figure: Callable[[Drift], Any], **extra) -> Result:
    """The result of `per_file`: one file's as it always was, several side
    by side -- a curve each, under its file's name."""
    if len(drifts) == 1:
        drift = drifts[0][1]
        return Result(locs=corrected, text=str(drift), plot=figure(drift),
                      data={"drift": drift, **extra}, settings=settings)
    return Result(locs=corrected,
                  text="\n".join(f"{name}: {drift}" for name, drift in drifts),
                  plot=_side_by_side(drifts), settings=settings,
                  data={"drifts": dict(drifts), **extra})


def _side_by_side(drifts: List[Tuple[str, Drift]]) -> Plot:
    def draw(figure):
        axes = np.atleast_1d(figure.subplots(len(drifts), 1, sharex=True))
        for ax, (name, drift) in zip(axes, drifts):
            drift.plot(ax)
            ax.set_title(name, fontsize="medium")
    return Plot(draw, panels=len(drifts), size=(6, 2.2 * len(drifts) + 0.6))


class KeepsDrift:
    """`keep` / `restore` for a plugin whose result carries a `Drift`."""

    def keep(self, result: Result) -> Optional[Dict[str, Any]]:
        data = result.data or {}
        if data.get("drifts"):            # several files, a curve each
            return {"drifts": {name: d.to_dict() for name, d in data["drifts"].items()}}
        drift = data.get("drift")
        return None if drift is None else {"drift": drift.to_dict()}

    def restore(self, saved: Dict[str, Any]) -> Optional[Result]:
        # the settings as the plugin's own class: COMET's, or the fiducial
        # fit's; RCC keeps none
        cls = getattr(self, "Settings", None)
        try:
            if saved.get("drifts"):
                drifts = [(name, Drift.from_dict(d, cls))
                          for name, d in saved["drifts"].items()]
            else:
                drifts = [("", Drift.from_dict(saved.get("drift") or {}, cls))]
        except (TypeError, ValueError, AttributeError):
            return None
        drifts = [(name, d) for name, d in drifts if len(d)]
        if not drifts:
            return None
        return drift_result(None, drifts, drifts[0][1].settings,
                            lambda d: d.figure())
