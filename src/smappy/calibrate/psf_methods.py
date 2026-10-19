"""The methods of the PSF calibration window, one entry each: what it reads, runs, draws and saves.

The window (`psf_window`) shows one method at a time and knows none of them:
each entry says which of the window's sections it reads (`sections`), how a
run and a *Recalculate* go, its result pages, its bead table and how it
saves.  A method added later -- in situ learning from blinking molecules, a
field-dependent model -- is one more entry, not a branch through the window
(`docs/uipsf/plan.md` §7.5).  No Qt here: a test or a script uses an entry
as the window does.

* **bead average**: the bead calibration's own (`core.collect_beads`,
  `build_calibration`; `dual.collect_dual_beads`, `build_dual_calibration`),
  its pages (`plots`) and its fit quality (`validation`), called, not copied.
* **pupil model (uiPSF)**: `pupil.run_calibration` and uipsf's figures.

Both save the same way: with *save model*, beside the stacks under the bead
calibration's name, never over an existing file except this run's own on
*Recalculate*.

A bead's *mark* is the user's wish for the next *Recalculate*: "exclude" a
used bead, or, for the pupil model, "include" a rejected one (the bead
average has no way to force a bead in: its rejection is part of the
alignment).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

#: what only averaging reads, shown with the bead average's settings and hidden for the pupil model
AVERAGE_ONLY = ("max_xy_shift_px", "max_z_shift_nm", "alignment_range_nm", "registration_iterations",
                "rejection_mad", "smooth_z_nm", "smooth_xy_px", "min_beads")


@dataclass
class Page:
    """One result page: ``draw(figure, selected, excluded)`` draws it and may return a status line."""
    title: str
    draw: Callable


@dataclass
class Run:
    """What a method's run left: its own result, and what was saved."""
    result: object
    dual: bool
    saved: Optional[Path] = None
    extra: dict = field(default_factory=dict)          # a method's own: fit quality, the start model


def _save_path(paths, dual, previous: Optional[Run]):
    from .pupil import default_save_path
    if previous is not None and previous.saved is not None:
        return previous.saved, True                    # Recalculate replaces its own run's files
    return default_save_path(paths, dual), False


class BeadAverage:
    key, label = "average", "bead average"
    # the window's sections this method reads
    sections = {"start from": False, "microscope": False, "model": False, "spline": False, "averaging": True}
    columns = ("bead", "stack", "x, y (px)", "brightness", "dz (nm)", "correlation", "residual", "state")
    marks_include = False
    has_quality = True
    has_stack = True

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None) -> Run:
        from .core import build_calibration, collect_beads
        from .dual import build_dual_calibration, collect_dual_beads
        report = (lambda text: progress({"message": text})) if progress else None
        excluded = sorted(b for b, wish in (marks or {}).items() if wish == "exclude")
        if previous is not None:
            beads = previous.result.beads
        else:
            beads = (collect_dual_beads if dual else collect_beads)(paths, inputs["detection"], report)
        build = build_dual_calibration if dual else build_calibration
        result = build(beads, excluded=excluded, progress=report)
        run = Run(result, dual)
        if inputs["run"].save_model:
            path, again = _save_path(paths, dual, previous)
            result.save(path, overwrite=again)
            run.saved = path
        return run

    def save_as(self, run: Run, path) -> Path:
        run.result.save(path, overwrite=True)
        run.saved = Path(path)
        return run.saved

    def quality(self, run: Run):
        """The beads fitted back with the fitter the data will meet (`validation`), kept with the run."""
        from .validation import (aligned_midline_profiles, fit_bead_diagnostics, fit_paired_bead_diagnostics,
                                 paired_profiles)
        result = run.result
        if run.dual:
            fitted = fit_paired_bead_diagnostics(result)
            result.registration.refits = {"global": fitted}
            run.extra["quality"] = (fitted, paired_profiles(result))
        else:
            run.extra["quality"] = (fit_bead_diagnostics(result), aligned_midline_profiles(result))
        return run

    def pages(self, run: Run) -> list:
        from . import plots
        result, dual = run.result, run.dual

        def quality(figure, selected, excluded):
            found = run.extra.get("quality")
            if found is None:
                ax = figure.subplots()
                ax.axis("off")
                ax.text(0.5, 0.5, "Calculate fit quality to see it", ha="center", va="center")
                return None
            if dual:
                return plots.draw_paired_fit_quality(figure, result, found[0], found[1], selected)
            return plots.draw_fit_quality(figure, found[0], found[1], selected)

        pages = [Page("Overview", lambda f, s, e: (plots.draw_pairs if dual else plots.draw_overview)(f, result, s, e))]
        if dual:
            pages.append(Page("Transformation", lambda f, s, e: plots.draw_transformation(f, result, s, e)))
        pages.append(Page("Fit quality", quality))
        if dual:
            pages.append(Page("Field diagnostics", lambda f, s, e: plots.draw_field(f, result, s, e)))
        pages.append(Page("Bead diagnostics", lambda f, s, e: plots.draw_beads(f, result, s, e, dual=dual)))
        return pages

    def rows(self, run: Run, marks: dict) -> list:
        """(values, greyed) per bead, as the bead calibration's table shows them."""
        result, out = run.result, []
        excluded = {b for b, wish in marks.items() if wish == "exclude"}
        accepted_t = getattr(result, "transform_accepted", None)
        for i, record in enumerate(result.beads.records):
            state = result.reasons[i]
            if accepted_t is not None and accepted_t[i] and state != "manual exclusion":
                state = "PSF + transformation" if result.accepted[i] else "transformation only: " + state
            wish = marks.get(i)
            if wish is not None and state != "manual exclusion":
                state = f"{state} -> {wish} on Recalculate"
            out.append(((i, record["stack"] + 1, f"{record['x_px']}, {record['y_px']}",
                         f"{record['brightness_adu']:.4g}",
                         f"{result.shifts[i, 0] * result.calibration.dz:.1f}",
                         f"{result.correlations[i]:.3f}", f"{result.residuals[i]:.3f}", state),
                        i in excluded or not result.accepted[i]))
        return out

    def initial_marks(self, run: Run) -> dict:
        """A run's own manual exclusions stay marked, so a second Recalculate keeps them out."""
        return {i: "exclude" for i, reason in enumerate(run.result.reasons) if reason == "manual exclusion"}

    def mark(self, run: Run, bead: int) -> str:
        return "exclude"

    def stacks(self, run: Run):
        """The averaged bead stack(s) to browse, their titles and the mirror axis (dual colour)."""
        result = run.result
        channels = getattr(getattr(result, "registration", None), "channel_raw_psfs", None)
        if channels is not None and len(channels) > 1:
            return list(channels), ["main", "secondary"], result.beads.geometry.get("mirror_axis_xy")
        return [np.asarray(result.raw_psf)], ["average bead stack"], None

    def summary(self, run: Run) -> str:
        result = run.result
        return f"{int(np.sum(result.accepted))} of {len(result.beads.records)} beads used"


class PupilModel:
    key, label = "pupil", "pupil model (uiPSF)"
    sections = {"start from": True, "microscope": True, "model": True, "spline": True, "averaging": False}
    columns = ("bead", "stack", "x, y (px)", "brightness", "state", "reason", "deviance", "photons",
               "xy offset (px)")
    marks_include = True
    has_quality = False
    has_stack = False

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None) -> Run:
        from .pupil import run_calibration
        done = run_calibration(paths, dual, inputs["optics"], inputs["model"], inputs["detection"],
                               inputs["spline"], inputs["run"],
                               previous=None if previous is None else previous.result, decisions=marks or None,
                               progress=progress, cancel=cancel)
        return Run(done, dual, done.saved)

    def save_as(self, run: Run, path) -> Path:
        from .pupil import dual_calibration, save, spline_calibration
        done = run.result
        if done.calibration is None:                  # learnt with *save model* off: sample it now
            rois = done.rois
            dz, z_range = float(rois.batch.z_step_nm), float(rois.batch.z_nm.abs().max())
            roi = int(rois.batch.data.shape[-1])
            done.calibration = (dual_calibration(done.cal, dz, z_range, roi) if run.dual
                                else spline_calibration(done.cal, dz, z_range, roi, camera=rois.camera))
        done.saved, _ = save(path, done.cal, done.calibration, overwrite=True)
        run.saved = done.saved
        return run.saved

    def pages(self, run: Run) -> list:
        from uipsf.diagnostics import figures
        return [Page(title, lambda f, s, e, draw=draw: draw(f))
                for title, draw in figures(run.result.cal, reference=run.result.start).items()]

    def rows(self, run: Run, marks: dict) -> list:
        table, records = run.result.cal.beads, run.result.rois.records
        out = []
        for i, row in enumerate(table.rows):
            record = records[i] if i < len(records) else {}
            final = row.final or {}
            state = row.state
            wish = marks.get(row.bead)
            if wish is not None:
                state = f"{state} -> {wish} on Recalculate"
            out.append(((row.bead, record.get("stack", 0) + 1, f"{record.get('x_px', '')}, {record.get('y_px', '')}",
                         record.get("brightness_reason", ""), state, row.reason,
                         f"{final.get('deviance', row.scores.get('deviance', float('nan'))):.3f}",
                         f"{row.scores.get('photons', float('nan')):.2f}",
                         f"{row.scores.get('xy_offset_px', float('nan')):.2f}"),
                        row.state != "used"))
        return out

    def initial_marks(self, run: Run) -> dict:
        return {}

    def mark(self, run: Run, bead: int) -> str:
        return "exclude" if run.result.cal.beads.rows[bead].state == "used" else "include"

    def summary(self, run: Run) -> str:
        cal = run.result.cal
        if cal.result.stop_reason == "cancelled":
            return "stopped, nothing saved"
        return f"{len(cal.beads.used)} of {len(cal.beads.rows)} beads used"


METHODS = (BeadAverage(), PupilModel())
