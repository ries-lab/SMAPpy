"""The methods of the PSF calibration window, one entry each: what it reads, runs, draws and saves.

The window (`psf_window`) shows one method at a time and knows none of them:
each entry says which of the window's sections it reads (`sections`), how a
run goes, its result pages, its bead table and how it
saves.  A method added later -- in situ learning from blinking molecules, a
field-dependent model -- is one more entry, not a branch through the window
(`docs/uipsf/plan.md` §7.5).  No Qt here: a test or a script uses an entry
as the window does.

* **bead average**: the bead calibration's own (`core.collect_beads`,
  `build_calibration`; `dual.collect_dual_beads`, `build_dual_calibration`),
  its pages (`plots`) and its fit quality (`validation`), called, not copied.
* **pupil model (uiPSF)**: `pupil.run_calibration` and uipsf's figures.
* **in situ (blinking molecules)**: `insitu_run.run_insitu` on movies, uipsf's figures and the in situ
  pages (`insitu_views`); its table counts molecules by state, with no marks.
* **elliptical Gaussian**: no PSF model at all, the curves of an elliptical Gaussian's widths in z
  (`elliptical`), for Astigmatic z rather than a fitter; saved as ``<name>_astigz.h5`` and never listed
  for the fitters' *auto*.

Both save the same way: with *save model*, beside the stacks under the bead
calibration's name, never over an existing file except the last run's own
when *Calibrate* starts from it (the same beads and settings, other marks).

A run goes in two steps, so that an afternoon's learning is never started by
accident: *Detect beads* (`detect`) finds the beads and says how many and,
for the pupil model, about how long learning them will take; *Calibrate*
learns from what was detected.  After a calibration the method's `finish`
adds what is shown with it -- the fit quality and the model against the data
(`model_views`) -- in the same worker, so neither waits for a button.

A bead's *mark* is the user's wish for the next *Calibrate*: "exclude" a
used bead, or, for the pupil model, "include" a rejected one (the bead
average has no way to force a bead in: its rejection is part of the
alignment).
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .model_views import SECTIONS

#: what only averaging reads, shown with the bead average's settings and hidden for the pupil model
AVERAGE_ONLY = ("max_xy_shift_px", "max_z_shift_nm", "alignment_range_nm", "registration_iterations",
                "rejection_mad", "smooth_z_nm", "smooth_xy_px", "min_beads")


@dataclass
class Page:
    """One result page: ``draw(figure, selected, excluded)`` draws it and may return a status line.
    A page that shows no bead of its own (the pupil, shared by all beads) does not follow the table's
    selection, and is not redrawn when it changes.  ``controls`` ({label: choices}) are choices the window
    puts above the page, the first one chosen at first, or a tick for a bool (its value the tick's start);
    a page with any is drawn as ``draw(figure, selected, excluded, chosen)``, ``chosen`` {label: value}."""
    title: str
    draw: Optional[Callable]
    follows_selection: bool = True
    controls: dict = field(default_factory=dict)
    table: Optional[Callable] = None           # instead of a figure: () -> (columns, rows[, row colours])


@dataclass
class Detected:
    """What *Detect beads* found: the method's own (``found``), a line saying so, and where each bead is,
    (stack, x, y, bead) in the pixels of the stacks as `read_stacks` reads them -- a pair twice."""
    found: object
    text: str
    positions: list = field(default_factory=list)
    seconds: Optional[float] = None                    # about how long calibrating will take, if known


class LearningProgress:
    """How far a run is, as a fraction, and about how long it has left, from the entries a method reports.

    An entry may say its ``fraction`` itself (the steps around learning do).  A learning stage does not
    know how many iterations it will take -- it stops when the objective stops falling -- so its fraction
    is how far the fall per iteration has come, on a log scale, from the first iteration's to the
    solver's tolerance (`GaussNewtonLBFGS`, 1e-3): an L-BFGS fit's gain falls roughly geometrically.  Never
    back: an iteration that gains more than the one before does not undo the bar.  The time left is the time so far scaled by what
    is left, from 5 % on: before that it says nothing worth reading."""
    #: (start, end) of each learning stage in the run's fraction; detection is a step of its own
    STAGES = {1: (0.0, 0.45), 2: (0.45, 0.9)}
    TOL = 1e-3

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.fraction: Optional[float] = None
        self._stage = None
        self._last = None
        self._first_gain = None

    def update(self, entry: dict) -> tuple:
        """(fraction or None, seconds left or None, a status line) after ``entry``; ``{"restart": True}``
        starts the bar and the clock again, for a step that follows another."""
        if entry.get("restart"):
            self.started, self.fraction, self._stage = self.clock(), None, None
            return None, None, "calibrating"
        if "fraction" in entry:
            self._advance(float(entry["fraction"]))
        if "message" in entry:
            return self.fraction, self.left(), entry["message"]
        stage = entry.get("stage")
        objective = entry.get("objective")
        if stage in self.STAGES and objective is not None:
            if stage != self._stage:
                self._stage, self._last, self._first_gain = stage, None, None
            objective = float(objective)
            if self._last is not None and objective < self._last:
                gain = self._last - objective
                self._first_gain = self._first_gain or gain
                done = math.log(self._first_gain / gain) / math.log(max(self._first_gain / self.TOL, 1.0 + 1e-9))
                start, end = self.STAGES[stage]
                self._advance(start + (end - start) * min(max(done, 0.0), 1.0))
            elif self._last is None:
                self._advance(self.STAGES[stage][0])
            self._last = objective
        name = {1: "low orders", 2: "all orders"}.get(stage, "learning")
        text = f"{name}: iteration {entry.get('iteration', 0)}" + (
            "" if objective is None else f", -log L {float(objective):.6g}")
        return self.fraction, self.left(), text

    def _advance(self, fraction: float) -> None:
        self.fraction = max(self.fraction or 0.0, min(max(fraction, 0.0), 1.0))

    def left(self) -> Optional[float]:
        if not self.fraction or self.fraction < 0.05:
            return None
        return (self.clock() - self.started) * (1 - self.fraction) / self.fraction


def duration(seconds: float) -> str:
    """"40 s", "3 min", "1.5 h": a duration as one reads it at a glance."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


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
        return previous.saved, True                    # a warm Calibrate replaces its own run's files
    return default_save_path(paths, dual), False


def _not_yet(figure, text):
    ax = figure.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center")


def _positions(records) -> list:
    """(stack, x, y, bead) of every bead, each channel of a pair: `Detected.positions`."""
    out = []
    for i, record in enumerate(records):
        parts = record.get("channel_records") or record.get("channels")
        if parts:
            for part in parts:
                x, y = part.get("full_image_xy") or (part["x_px"], part["y_px"])
                out.append((int(part.get("stack", record.get("stack", 0))), float(x), float(y), i))
        else:
            out.append((int(record.get("stack", 0)), float(record["x_px"]), float(record["y_px"]), i))
    return out


def _detected_rows(columns, records, marks, brightness) -> list:
    """(values, greyed) per bead found, before any calibration: the method's own columns, those only a
    calibration fills left empty; state "found" or "too dim", and a mark "-> exclude on Calibrate".
    ``brightness(record)`` is what the method's *brightness* column shows."""
    out = []
    for i, record in enumerate(records):
        bright = bool(record.get("brightness_accepted", True))
        state = "found" if bright else str(record.get("brightness_reason", "too dim"))
        if marks.get(i):
            state = f"{state} -> {marks[i]} on Calibrate"
        values = {"bead": i, "stack": int(record.get("stack", 0)) + 1,
                  "x, y (px)": f"{record.get('x_px', '')}, {record.get('y_px', '')}",
                  "brightness": brightness(record), "state": state}
        out.append((tuple(values.get(c, "") for c in columns), not bright or i in marks))
    return out


def _model_page(run: Run) -> Page:
    """*Model vs data*: `model_views.draw_model_vs_data` of the run's views, a selected bead alone."""
    from .model_views import draw_model_vs_data

    def draw(figure, selected, excluded, chosen):
        views = run.extra.get("views")
        if views is None:
            _not_yet(figure, "the model against the data is drawn after calibrating")
            return None
        bead = selected[0] if selected else None
        return draw_model_vs_data(figure, views, bead, chosen["section"], average=chosen["average"])
    # average: the learnt average whatever the table's selection, so that a bead can be picked by its row
    # and compared with the average without losing the selection
    return Page("Model vs data", draw, controls={"section": SECTIONS, "average": False})


def _counted(report, start: float, end: float):
    """A report that puts a step's own "i/n" messages into the run's fraction between ``start`` and
    ``end`` (`LearningProgress`), for steps that only count."""
    if report is None:
        return None

    def tell(text):
        counted = re.search(r"(\d+)/(\d+)", text)
        entry = {"message": text}
        if counted and int(counted.group(2)) > 0:
            entry["fraction"] = start + (end - start) * int(counted.group(1)) / int(counted.group(2))
        report(entry)
    return tell


class BeadAverage:
    key, label = "average", "bead average"
    # the window's sections this method reads
    sections = {"start from": False, "microscope": False, "model": False, "spline": False, "averaging": True}
    columns = ("bead", "stack", "x, y (px)", "brightness", "dz (nm)", "correlation", "residual", "state")
    marks_include = False
    has_stack = True
    has_halves = False              # no Zernikes to give an uncertainty of

    def detect(self, paths, dual, inputs, progress=None) -> Detected:
        from .core import collect_beads
        from .dual import collect_dual_beads
        beads = (collect_dual_beads if dual else collect_beads)(paths, inputs["detection"],
                                                                 _counted(progress, 0.0, 1.0))
        bright = sum(bool(r.get("brightness_accepted", True)) for r in beads.records)
        what = "pairs" if dual else "beads"
        return Detected(beads, f"{len(beads.records)} {what} found, {bright} bright enough to average; "
                               f"averaging takes seconds", _positions(beads.records))

    def detected_rows(self, detected: Detected, marks: dict) -> list:
        return _detected_rows(self.columns, detected.found.records, marks,
                              lambda r: f"{r.get('brightness_adu', float('nan')):.4g}")

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None, detected: Optional[Detected] = None) -> Run:
        from .core import build_calibration
        from .dual import build_dual_calibration
        excluded = sorted(b for b, wish in (marks or {}).items() if wish == "exclude")
        if previous is not None:
            beads = previous.result.beads
        elif detected is not None:
            beads = detected.found
        else:
            beads = self.detect(paths, dual, inputs, progress).found
        build = build_dual_calibration if dual else build_calibration
        result = build(beads, excluded=excluded, progress=_counted(progress, 0.0, 0.8))
        run = Run(result, dual)
        if inputs["run"].save_model:
            path, again = _save_path(paths, dual, previous)
            result.save(path, overwrite=again)
            run.saved = path
        return run

    def finish(self, run: Run, progress=None) -> Run:
        """The fit quality and the model against the data, kept with the run (`Run.extra`)."""
        from .model_views import average_views
        tell = progress or (lambda entry: None)
        tell({"message": "fitting the beads back with the " + ("two-channel" if run.dual else "spline") + " fitter",
              "fraction": 0.8})
        self.quality(run)
        tell({"message": "the model against the data", "fraction": 0.97})
        run.extra["views"] = average_views(run.result, run.dual)
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
                _not_yet(figure, "the fit quality is calculated after calibrating")
                return None
            if dual:
                return plots.draw_paired_fit_quality(figure, result, found[0], found[1], selected)
            return plots.draw_fit_quality(figure, found[0], found[1], selected)

        pages = [Page("Overview", lambda f, s, e: (plots.draw_pairs if dual else plots.draw_overview)(f, result, s, e))]
        if dual:
            pages.append(Page("Transformation", lambda f, s, e: plots.draw_transformation(f, result, s, e)))
        pages.append(_model_page(run))
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
                state = f"{state} -> {wish} on Calibrate"
            elif wish is None and state == "manual exclusion":
                state = f"{state} -> include on Calibrate"         # the exclusion withdrawn
            out.append(((i, record["stack"] + 1, f"{record['x_px']}, {record['y_px']}",
                         f"{record['brightness_adu']:.4g}",
                         f"{result.shifts[i, 0] * result.calibration.dz:.1f}",
                         f"{result.correlations[i]:.3f}", f"{result.residuals[i]:.3f}", state),
                        i in excluded or not result.accepted[i]))
        return out

    def initial_marks(self, run: Run) -> dict:
        """A run's own manual exclusions stay marked, so the next Calibrate keeps them out."""
        return {i: "exclude" for i, reason in enumerate(run.result.reasons) if reason == "manual exclusion"}

    def mark(self, run: Run, bead: int) -> str:
        return "exclude"

    def stacks(self, run: Run):
        """What the stack browser shows: the run's `ModelViews`, and anything with ``z_index_to_nm`` for
        its axial axis."""
        from .model_views import average_views
        views = run.extra.get("views") or average_views(run.result, run.dual)
        return views, run.result.calibration

    def summary(self, run: Run) -> str:
        result = run.result
        return f"{int(np.sum(result.accepted))} of {len(result.beads.records)} beads used"


class PupilModel:
    key, label = "pupil", "pupil model (uiPSF)"
    sections = {"start from": True, "microscope": True, "model": True, "spline": True, "averaging": False}
    columns = ("bead", "stack", "x, y (px)", "brightness", "state", "reason", "deviance", "photons",
               "xy offset (px)")
    marks_include = True
    has_stack = True
    has_halves = True

    def halves_needed(self, run: Run) -> Optional[str]:
        """Why *Uncertainty from halves* cannot run on ``run``, or None if it can."""
        from uipsf.diagnostics import MIN_BEADS
        used = len(run.result.cal.beads.used)
        if used < MIN_BEADS:
            return f"{used} used beads: halves need at least {MIN_BEADS}"
        return None

    def halves_seconds(self, run: Run, inputs: dict) -> Optional[float]:
        """About how long one split takes: its two halves are about one calibration of all the beads."""
        from .pupil import estimate_seconds
        try:
            return estimate_seconds(run.result.rois, inputs["optics"], inputs["model"])
        except Exception:                # an estimate, never a reason not to offer the run
            return None

    def halves(self, run: Run, inputs: dict, splits: int, progress=None, cancel=None):
        """`pupil.half_split_uncertainty` of the run, with the optics it was made with; kept on its
        calibration (``cal.halves``), and saved again into its files when it was saved, so the numbers
        are in the model file."""
        from .pupil import half_split_uncertainty
        optics = run.extra.get("inputs", inputs)["optics"]
        halves = half_split_uncertainty(run.result, optics, splits, progress=progress, cancel=cancel)
        if halves is not None and run.saved is not None:
            self.save_as(run, run.saved)
        return halves

    def detect(self, paths, dual, inputs, progress=None) -> Detected:
        from .pupil import bead_rois, estimate_seconds
        tell = progress or (lambda entry: None)
        tell({"message": "reading the stacks and finding the beads", "fraction": 0.0})
        rois = bead_rois(paths, inputs.get("camera"), dual=dual, settings=inputs["detection"])
        tell({"message": "timing the model on a few beads", "fraction": 0.7})
        seconds = estimate_seconds(rois, inputs["optics"], inputs["model"])
        what = "pairs" if dual else "beads"
        return Detected(rois, f"{rois.batch.n_emitters} {what} to learn from ({'; '.join(rois.messages)}); "
                              f"learning takes about {duration(seconds)}", _positions(rois.records), seconds)

    def detected_rows(self, detected: Detected, marks: dict) -> list:
        return _detected_rows(self.columns, detected.found.records, marks,
                              lambda r: r.get("brightness_reason", ""))

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None, detected: Optional[Detected] = None) -> Run:
        from .pupil import run_calibration
        done = run_calibration(paths, dual, inputs["optics"], inputs["model"], inputs["detection"],
                               inputs["spline"], inputs["run"], camera=inputs.get("camera"),
                               previous=None if previous is None else previous.result, decisions=marks or None,
                               progress=progress, cancel=cancel, rois=None if detected is None else detected.found)
        return Run(done, dual, done.saved)

    def finish(self, run: Run, progress=None) -> Run:
        """The fit quality (Spline 3D on every bead) and the model against the data, kept with the run;
        nothing after a stopped run."""
        from .model_views import pupil_views
        if run.result.cal.result.stop_reason == "cancelled":
            return run
        tell = progress or (lambda entry: None)
        tell({"message": "fitting the beads back with the " + ("two-channel" if run.dual else "spline") + " fitter",
              "fraction": 0.9})
        self.quality(run)
        tell({"message": "the model against the data", "fraction": 0.97})
        run.extra["views"] = pupil_views(run.result)
        return run

    def save_as(self, run: Run, path) -> Path:
        from .pupil import calibration_of, save
        done = run.result
        done.saved, _ = save(path, done.cal, calibration_of(done), overwrite=True)  # sampled if not saved yet
        run.saved = done.saved
        return run.saved

    def quality(self, run: Run):
        """Every bead refitted with the calibration by Spline 3D (`pupil.fit_quality`), kept with the run."""
        from .pupil import fit_quality
        run.extra["quality"] = fit_quality(run.result)
        return run

    def pages(self, run: Run) -> list:
        from uipsf.diagnostics import draw_beads, figures
        from uipsf.diagnostics.figures import zernike_table

        from . import plots
        cal = run.result.cal
        from uipsf.diagnostics.figures import unit_choices
        pages = []
        halves = getattr(cal, "halves", None)            # after *Uncertainty from halves*
        for title, draw in figures(cal, reference=run.result.start, halves=halves).items():
            if title == "Beads":
                continue
            if title.startswith("Pupil and Zernikes"):
                # the phase in nm, waves or rad: a choice above the page, `PHASE_UNIT` first (the one shown)
                units = unit_choices()
                pages.append(Page(title, lambda f, s, e, chosen, draw=draw: draw(f, units=chosen["units"]),
                                  follows_selection=False, controls={"units": units}))
            else:
                pages.append(Page(title, lambda f, s, e, draw=draw: draw(f), follows_selection=False))
        pages.append(Page("Zernike table", None, follows_selection=False,
                          table=lambda: zernike_table(cal, reference=run.result.start, halves=halves)))

        def quality(figure, selected, excluded):
            found = run.extra.get("quality")
            if found is None:
                _not_yet(figure, "the fit quality is calculated after calibrating")
                return None
            return plots.draw_pupil_fit_quality(figure, found, selected, excluded)

        pages.append(_model_page(run))
        pages.append(Page("Fit quality", quality))
        if any(r.field_xy is not None for r in cal.beads.rows):
            pages.append(Page("Beads", lambda f, s, e: draw_beads(f, cal, s)))
        return pages

    def rows(self, run: Run, marks: dict) -> list:
        table, records = run.result.cal.beads, run.result.rois.records
        out = []
        for i, row in enumerate(table.rows):
            record = records[i] if i < len(records) else {}
            final = row.final or {}
            state = row.state
            wish = marks.get(row.bead)
            if wish is not None and wish != row.manual:
                state = f"{state} -> {wish} on Calibrate"
            elif row.manual and wish is None:
                state = f"{state} -> automatic on Calibrate"         # a choice by hand withdrawn
            out.append(((row.bead, record.get("stack", 0) + 1, f"{record.get('x_px', '')}, {record.get('y_px', '')}",
                         record.get("brightness_reason", ""), state, row.reason,
                         f"{final.get('deviance', row.scores.get('deviance', float('nan'))):.3f}",
                         f"{row.scores.get('photons', float('nan')):.2f}",
                         f"{row.scores.get('xy_offset_px', float('nan')):.2f}"),
                        row.state != "used"))
        return out

    def initial_marks(self, run: Run) -> dict:
        """A run's own choices by hand stay marked, so the next Calibrate keeps them: a bead excluded by
        hand stays out (and one included stays in) until it is unmarked."""
        return {row.bead: row.manual for row in run.result.cal.beads.rows if row.manual}

    def mark(self, run: Run, bead: int) -> str:
        return "exclude" if run.result.cal.beads.rows[bead].state == "used" else "include"

    def stacks(self, run: Run):
        """What the stack browser shows: the run's `ModelViews` and the spline, for its axial axis."""
        from .model_views import pupil_views
        from .pupil import calibration_of
        views = run.extra.get("views") or pupil_views(run.result)
        calibration = calibration_of(run.result)
        return views, getattr(calibration, "main", calibration)

    def summary(self, run: Run) -> str:
        cal = run.result.cal
        if cal.result.stop_reason == "cancelled":
            return "stopped, nothing saved"
        from uipsf.diagnostics.figures import tilt_text
        tilt = tilt_text(cal)
        return f"{len(cal.beads.used)} of {len(cal.beads.rows)} beads used" + (f", {tilt}" if tilt else "")


class InSitu:
    """In situ: the pupil learnt from the blinking molecules of the data themselves (`insitu_run`).

    Its files are movies, not bead stacks; its sections the microscope, the model, *start from* and the
    spline as the pupil model's, and its own (*in situ*: the focus, the start's astigmatism, the coverslip,
    the rounds; the detection's settings under *more*).  One channel.  There is no table of molecules to
    mark: thousands of them are chosen by rules, not by hand, so the table counts them by state."""
    key, label = "insitu", "in situ (blinking molecules)"
    sections = {"start from": True, "microscope": True, "model": True, "spline": True, "averaging": False,
                "in situ": True}
    columns = ("state", "units", "molecules")
    marks_include = False
    marks = False
    has_stack = False
    files = "movies"
    finds = "molecules"

    def detect(self, paths, dual, inputs, progress=None) -> Detected:
        from .insitu_run import movie_rois
        if dual:
            raise ValueError("in situ learning is one channel only, for now")
        rois = movie_rois(paths, inputs["detection"], inputs.get("camera"), progress=progress)
        used = len(rois.ids("used"))
        return Detected(rois, f"{used} frames of {len(np.unique(rois.units.blink[rois.ids('used')]))} molecules "
                              f"to learn from ({'; '.join(rois.messages)}); learning takes minutes a round",
                        _unit_positions(rois))

    def detected_rows(self, detected: Detected, marks: dict) -> list:
        return _state_rows(detected.found.units, detected.found.state)

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None, detected: Optional[Detected] = None) -> Run:
        from .insitu_run import run_insitu
        rois = detected.found if detected is not None else (previous.result.rois if previous is not None else None)
        done = run_insitu(paths, inputs["optics"], inputs["model"], inputs["insitu"], inputs["detection"],
                          inputs["spline"], inputs["run"], camera=inputs.get("camera"), rois=rois,
                          progress=progress, cancel=cancel,
                          save_to=previous.saved if previous is not None else None)
        return Run(done, False, done.saved)

    def finish(self, run: Run, progress=None) -> Run:
        """The model against the data per height, kept with the run; nothing after a stopped run."""
        from .insitu_views import model_vs_data
        if run.result.cal.result.stop_reason == "cancelled":
            return run
        (progress or (lambda entry: None))({"message": "the model against the data", "fraction": 0.97})
        run.extra["views"] = model_vs_data(run.result)
        return run

    def save_as(self, run: Run, path) -> Path:
        from .insitu_run import calibration_of
        from .pupil import save
        done = run.result
        done.saved, _ = save(path, done.cal, calibration_of(done), overwrite=True)
        run.saved = done.saved
        return run.saved

    def pages(self, run: Run) -> list:
        from uipsf.diagnostics import figures
        from uipsf.diagnostics.figures import unit_choices, zernike_table

        from .insitu_views import draw_model_vs_data, draw_rounds, draw_z_bins
        result = run.result
        cal = result.cal
        pages = [Page("z bins", lambda f, s, e: draw_z_bins(f, result.rois, result.run), follows_selection=False),
                 Page("Rounds and edge", lambda f, s, e: draw_rounds(f, result.run), follows_selection=False)]
        for title, draw in figures(cal, reference=result.start).items():
            if title == "Beads":
                continue
            if title.startswith("Pupil and Zernikes"):
                units = unit_choices()
                pages.append(Page(title, lambda f, s, e, chosen, draw=draw: draw(f, units=chosen["units"]),
                                  follows_selection=False, controls={"units": units}))
            else:
                pages.append(Page(title, lambda f, s, e, draw=draw: draw(f), follows_selection=False))
        pages.append(Page("Zernike table", None, follows_selection=False,
                          table=lambda: zernike_table(cal, reference=result.start)))

        def versus(figure, selected, excluded):
            views = run.extra.get("views")
            if views is None:
                _not_yet(figure, "the model against the data is drawn after calibrating")
                return None
            return draw_model_vs_data(figure, views)
        pages.append(Page("Model vs data", versus, follows_selection=False))
        return pages

    def rows(self, run: Run, marks: dict) -> list:
        last = run.result.run.rounds[-1]
        return _state_rows(run.result.rois.units, last.state)

    def initial_marks(self, run: Run) -> dict:
        return {}

    def mark(self, run: Run, bead: int):
        return None

    def stacks(self, run: Run):
        return None, None

    def summary(self, run: Run) -> str:
        result = run.result
        if result.cal.result.stop_reason == "cancelled":
            return "stopped, nothing saved"
        focus = float(result.cal.shared()["geometry.focus"])
        twin = result.run.final.twin
        from uipsf.diagnostics.figures import tilt_text
        tilt = tilt_text(result.cal)
        text = (f"{len(result.cal.beads.used)} frames used, {len(result.run.rounds)} rounds, focus {focus:.0f} nm"
                + (f", {tilt}" if tilt else "") + ("" if twin is None else f", twin test: {twin.verdict}"))
        return text + "".join(f"; {w}" for w in result.messages)

    def states(self, run: Run) -> dict:
        """Each unit's state, for its ring on the Stacks page: the last round's."""
        return {i: s.split(":")[0] for i, s in enumerate(run.result.run.rounds[-1].state)}

    def detected_states(self, detected: Detected) -> dict:
        return {i: s.split(":")[0] for i, s in enumerate(detected.found.state)}

    def positions(self, run: Run) -> list:
        return _unit_positions(run.result.rois)


def _unit_positions(rois) -> list:
    """(movie, x, y, unit, frame) of every unit: the stack browser rings each on its frame."""
    u = rois.units
    return [(int(u.source[i]), float(u.x_px[i]), float(u.y_px[i]), i, int(u.frames[i][0]))
            for i in range(len(u.reason))]


def _state_rows(units, state) -> list:
    """(values, greyed) per state: how many units (frames) and molecules (blinks) are in it."""
    kinds: dict = {}
    for i, s in enumerate(state):
        kinds.setdefault(s, set()).add(i)
    order = sorted(kinds, key=lambda s: (s.startswith("rejected"), s != "used", s != "held out", s))
    return [((s, len(kinds[s]), len({int(units.blink[i]) for i in kinds[s]})), s.startswith("rejected"))
            for s in order]


class EllipticalGaussian:
    """The widths of an elliptical Gaussian through focus (`elliptical`), for *Astigmatic z*: the beads
    are found as the bead average finds them, each plane fitted with the elliptical Gaussian, the beads
    put onto each other by their round spots and one defocus curve per axis fitted to them.  Seconds,
    one channel, no torch; its own section *elliptical Gaussian*, and the camera for photons and nm."""
    key, label = "elliptical", "elliptical Gaussian"
    sections = {"start from": False, "microscope": False, "model": False, "spline": False, "averaging": False,
                "elliptical": True}
    columns = ("bead", "stack", "x, y (px)", "brightness", "z0 (nm)", "residual (nm)", "state")
    marks_include = False
    has_stack = False
    listed = False                         # not a PSF: the fitters' *auto* must never pick it
    use_label = "Use in Astigmatic z"

    def detect(self, paths, dual, inputs, progress=None) -> Detected:
        from .core import collect_beads
        if dual:
            raise ValueError("the elliptical Gaussian calibration is one channel only")
        beads = collect_beads(paths, inputs["detection"], _counted(progress, 0.0, 1.0))
        bright = sum(bool(r.get("brightness_accepted", True)) for r in beads.records)
        return Detected(beads, f"{len(beads.records)} beads found, {bright} bright enough; calibrating takes "
                               f"seconds", _positions(beads.records))

    def detected_rows(self, detected: Detected, marks: dict) -> list:
        return _detected_rows(self.columns, detected.found.records, marks,
                              lambda r: f"{r.get('brightness_adu', float('nan')):.4g}")

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None, detected: Optional[Detected] = None) -> Run:
        from .elliptical import bead_residuals, bead_widths, calibrate_widths, save_path
        from .pupil import pixel_size_nm
        if previous is not None:
            beads = previous.result["beads"]
        elif detected is not None:
            beads = detected.found
        else:
            beads = self.detect(paths, dual, inputs, progress).found
        camera = inputs.get("camera")
        if camera is None:
            raise ValueError("the elliptical Gaussian calibration needs the camera: its pixel size for nm and "
                             "its conversion for photons -- give them under microscope")
        settings = inputs["elliptical"]
        tell = _counted(progress, 0.0, 1.0) or (lambda text: None)
        widths = bead_widths(beads, camera, pixel_size_nm(camera), settings, tell)
        # a bead too dim for the bead average is too dim here too
        dim = [i for i, r in enumerate(beads.records) if not r.get("brightness_accepted", True)]
        excluded = sorted({b for b, wish in (marks or {}).items() if wish == "exclude"} | set(dim))
        names = ", ".join(sorted({Path(str(s.get("path", ""))).name for s in beads.sources}))
        cal = calibrate_widths(widths, settings.higher_orders, excluded, source=names)
        result = {"beads": beads, "widths": widths, "calibration": cal, "excluded": excluded,
                  "manual": sorted(b for b, wish in (marks or {}).items() if wish == "exclude"),
                  "residuals": bead_residuals(widths, cal)}
        run = Run(result, False)
        if inputs["run"].save_model:
            path = previous.saved if previous is not None and previous.saved is not None else save_path(paths)
            run.saved = cal.save(path)
        return run

    def finish(self, run: Run, progress=None) -> Run:
        return run

    def save_as(self, run: Run, path) -> Path:
        run.saved = run.result["calibration"].save(path)
        return run.saved

    def pages(self, run: Run) -> list:
        from .elliptical import draw_widths, draw_z_back
        r = run.result
        return [Page("Widths in z", lambda f, s, e: draw_widths(f, r["widths"], r["calibration"], s, r["excluded"])),
                Page("z from the widths",
                     lambda f, s, e: draw_z_back(f, r["widths"], r["calibration"], s, r["excluded"]),
                     follows_selection=False)]

    def rows(self, run: Run, marks: dict) -> list:
        r = run.result
        widths, out = r["widths"], []
        for i, record in enumerate(r["beads"].records):
            if not record.get("brightness_accepted", True):
                state = str(record.get("brightness_reason", "too dim"))
            elif not np.isfinite(widths.z0_nm[i]):
                state = "widths never cross"
            elif i in r["manual"]:
                state = "manual exclusion"
            else:
                state = "used"
            wish = marks.get(i)
            if wish is not None and state != "manual exclusion":
                state = f"{state} -> {wish} on Calibrate"
            elif wish is None and state == "manual exclusion":
                state = f"{state} -> include on Calibrate"
            out.append(((i, record["stack"] + 1, f"{record['x_px']}, {record['y_px']}",
                         f"{record['brightness_adu']:.4g}", f"{widths.z0_nm[i]:.0f}",
                         f"{r['residuals'][i]:.2f}", state),
                        state != "used"))
        return out

    def initial_marks(self, run: Run) -> dict:
        return {i: "exclude" for i in run.result["manual"]}

    def mark(self, run: Run, bead: int) -> str:
        return "exclude"

    def stacks(self, run: Run):
        return None, None

    def summary(self, run: Run) -> str:
        cal = run.result["calibration"]
        return (f"{cal.beads} of {len(run.result['beads'].records)} beads used; rms residual "
                f"{cal.residual_nm[0]:.1f} / {cal.residual_nm[1]:.1f} nm in x / y")


class _FourPi:
    """What the 4Pi mode's methods share: bead scans read through the channel layer -- quadrant files, or
    a DCIMG z stack of the four images on one chip -- their layout the mode's own section (*4Pi*), each bead
    found on channel 0 and every channel's box cut where the registration from the same scans puts it
    (`fourpi_beads.scan_beads`); the model saved as a 4Pi model for *Spline 4Pi* with its layout; and after
    it the beads fitted back with the saved model, z and the phase against the stage."""
    columns = ("bead", "stack", "x, y (px)", "brightness", "z slope", "z scatter (nm)", "state")
    marks_include = False
    has_stack = True                   # each bead's channels, frame by frame (`fourpi_beads.stack_views`)
    files = "bead scans"
    use_label = "Use in Spline 4Pi"
    listed = False                     # a 4Pi model: never one for Spline 3D's *auto*
    #: the learning's time, a start and a time per bead frame (slice x mirror step), None where it is not
    #: known.  The pupil model's measured on the Zenodo 4Pi beads, 21 slices x 3 steps, on a Mac's GPU
    #: (2026-10-29): 21, 31, 30, 73, 90 s for 252, 504, 1008, 2016, 2583 frames -- far from the 3.5 min per
    #: 505 frames the window had said, which was from before the learning was made faster
    START_SECONDS: float = 0.0
    SECONDS_PER_FRAME: Optional[float] = None

    def detect(self, paths, dual, inputs, progress=None) -> Detected:
        from .fourpi_beads import scan_beads
        camera = inputs.get("camera")
        if camera is None:
            raise ValueError("4Pi beads need the camera's conversion and offset: give them under microscope")
        tell = _counted(progress, 0.0, 1.0) or (lambda text: None)
        from .fourpi_beads import brightness_outliers
        settings = inputs["fourpi"]
        beads, layout, registration = scan_beads(paths, settings, camera, tell)
        E, C, J, M = beads.data.shape[:4]
        rms = ", ".join(f"{r:.2f}" for r in registration.rms_px[1:])
        outliers = brightness_outliers(beads, settings.brightness_range)
        found = {"beads": beads, "layout": layout, "registration": registration, "outliers": outliers}
        text = (f"{E} beads in {C} channels, {J} slices x {M} phase step{'s' if M > 1 else ''}"
                + (f" ({len(outliers)} marked out by brightness)" if outliers else "")
                + f"; the channels registered from {min(registration.pairs[1:])} pairs ({rms} px)")
        detected = Detected(found, text, [(r["stack"], r["x_px"], r["y_px"], i) for i, r in enumerate(beads.records)])
        detected.seconds = self.seconds(detected, self.detected_marks(detected))
        return detected

    def seconds(self, detected: Detected, marks: dict) -> Optional[float]:
        """About how long learning the beads not marked out takes (`SECONDS_PER_FRAME`)."""
        if not self.SECONDS_PER_FRAME:
            return None
        beads = detected.found["beads"]
        E, C, J, M = beads.data.shape[:4]
        used = E - sum(1 for b, wish in marks.items() if wish == "exclude" and b < E)
        return self.START_SECONDS + self.SECONDS_PER_FRAME * used * J * M

    def detected_marks(self, detected: Detected) -> dict:
        """The beads *outlier factor* leaves out, marked so: in the table, and let in again by hand."""
        return {b: "exclude" for b in detected.found.get("outliers", ())}

    def detected_text(self, detected: Detected, marks: dict) -> str:
        """The detection's line with the time learning the beads as marked takes."""
        seconds = self.seconds(detected, marks)
        return detected.text + ("" if seconds is None else f"; learning takes about {duration(seconds)}")

    def detected_rows(self, detected: Detected, marks: dict) -> list:
        return _detected_rows(self.columns, detected.found["beads"].records, marks,
                              lambda r: f"{r.get('brightness', float('nan')):.4g}")

    def finish(self, run: Run, progress=None) -> Run:
        """The beads fitted back with the model as saved: z and the phase of every bead against the stage.
        A pupil model's fringe period checked against the beads' own (`fourpi_beads.period_mismatch`),
        saved or not: a wrong emission under *microscope* gives a model that looks merely poor."""
        from .fourpi_beads import bead_fits
        tell = progress or (lambda entry: None)
        if run.result.get("fit") is not None:
            tell({"message": "checking the model's fringe period against the beads'", "fraction": 0.93})
            self.check_period(run)
        cal = run.result.get("calibration")
        if cal is None:
            return run
        tell({"message": "fitting the beads back with the 4Pi model", "fraction": 0.95})
        run.extra["fits"] = bead_fits(run.result["found"]["beads"], cal)
        self._images(run)
        return run

    def check_period(self, run: Run) -> Optional[str]:
        """``run.extra["warning"]``: what `fourpi_beads.period_mismatch` says of the model's fringe period
        against the beads' own (`beads_period_nm`), None where they agree."""
        from .fourpi_beads import beads_period_nm, period_mismatch
        run.extra["warning"] = period_mismatch(self._calibration(run).period_nm,
                                               beads_period_nm(run.result["found"]["beads"]),
                                               run.result.get("wavelength_nm"))
        return run.extra["warning"]

    def _calibration(self, run: Run):
        """The pupil model's calibration: the saved one, or, with *save model* off, one written to a
        temporary file -- what the pages draw."""
        cal = run.result.get("calibration")
        if cal is None:
            from .fourpi_beads import save
            import tempfile
            s = run.result["settings"]
            cal = run.result["calibration"] = save(run.result["fit"], Path(tempfile.mkdtemp()) / "model_4pi.h5",
                                                   s.model_z_range_nm, s.model_dz_nm,
                                                   layout=run.result["found"]["layout"])
        return cal

    def _images(self, run: Run):
        """The learnt model's images of its beads (`fourpi_beads.fit_images`), once; None for a model that
        was not fitted to them (the bead average)."""
        fit = run.result.get("fit")
        if fit is None:
            return None
        if "images" not in run.extra:
            from .fourpi_beads import fit_images
            beads = run.result["found"]["beads"]
            used = [i for i in range(len(beads.records)) if i not in set(run.result["excluded"])]
            run.extra["images"] = fit_images(fit, beads, used)
        return run.extra["images"]

    def save_as(self, run: Run, path) -> Path:
        from .fourpi_beads import save
        s = run.result["settings"]
        run.result["calibration"] = save(run.result["fit"], path, s.model_z_range_nm, s.model_dz_nm,
                                         overwrite=True, layout=run.result["found"]["layout"])
        run.saved = Path(path)
        return run.saved

    def pages(self, run: Run) -> list:
        from .fourpi_beads import draw_against_stage, draw_model
        cal = self._calibration(run)
        from .fourpi_beads import (MODEL_PARTS, demodulate, draw_fit, draw_modulation, draw_pupils,
                                   modulation_curves, parameter_rows)
        fits = run.extra.get("fits")
        beads = run.result["found"]["beads"]
        M = beads.data.shape[3]
        pages = [Page("Model", lambda f, s, e, chosen: draw_model(f, cal, chosen["show"]), follows_selection=False,
                      controls={"show": MODEL_PARTS}),
                 Page("z against the stage",
                      lambda f, s, e: (draw_against_stage(f, fits, s, run.result["excluded"]) if fits is not None
                                       else _not_yet(f, "the beads are fitted back after calibrating")))]
        fit = run.result.get("fit")

        def bead_of(selected, images):
            chosen = [b for b in selected if b in images["used"]]
            if chosen:
                return chosen[0]
            light = beads.data[images["used"]].sum(axis=(1, 2, 3, 4, 5))
            return images["used"][int(np.argmax(light))]

        def model_vs_data(figure, selected, excluded, chosen):
            images = self._images(run)
            return draw_fit(figure, beads, images, bead_of(selected, images), int(chosen["mirror step"]) - 1)

        def curves(figure, selected, excluded):
            images = self._images(run)
            data = model = None
            if images is not None and M >= 3:
                used = images["used"]
                data = modulation_curves(*demodulate(beads.data[used].astype(float), images["background"],
                                                     images["psi"]))
                model = modulation_curves(*demodulate(images["model"], images["background"], images["psi"]))
            if model is None:
                # no fit of the beads, or no steps to demodulate them with: the saved model's own
                from ..fourpi import modulation
                model = {key: np.stack([modulation(cal, zz, r) for zz in beads.z_nm], 1)
                         for key, r in (("centre", 0.5), ("integrated", 4.0))}
            return draw_modulation(figure, beads.z_nm, data, model, cal.parameters.get("coherence"))

        pages.append(Page("Modulation", curves, follows_selection=False))
        pages.append(Page("Parameters", None, follows_selection=False,
                          table=lambda: parameter_rows(run.result, cal)))
        if fit is not None:
            pages.insert(1, Page("Model vs data", model_vs_data,
                                 controls={"mirror step": tuple(str(m + 1) for m in range(M))}))
            from uipsf.diagnostics.figures import unit_choices
            pages.append(Page("Pupils", lambda f, s, e, chosen: draw_pupils(f, fit, chosen["units"]),
                              follows_selection=False, controls={"units": unit_choices()}))
        return pages

    def rows(self, run: Run, marks: dict) -> list:
        beads = run.result["found"]["beads"]
        fits = run.extra.get("fits") or {}
        out = []
        for i, record in enumerate(beads.records):
            state = "excluded" if i in run.result["excluded"] else "used"
            wish = marks.get(i)
            if wish is not None and state == "used":
                state = f"used -> {wish} on Calibrate"
            elif wish is None and state == "excluded":
                state = "excluded -> include on Calibrate"
            slope, scatter = fits.get("slope", {}).get(i, np.nan), fits.get("scatter", {}).get(i, np.nan)
            out.append(((i, record["stack"] + 1, f"{record['x_px']}, {record['y_px']}",
                         f"{record['brightness']:.4g}", f"{slope:.2f}", f"{scatter:.1f}", state),
                        state != "used"))
        return out

    def initial_marks(self, run: Run) -> dict:
        return {i: "exclude" for i in run.result["excluded"]}

    def mark(self, run: Run, bead: int) -> str:
        return "exclude"

    def stacks(self, run: Run):
        """Each bead's channels, frame by frame, with the learnt model's images of it where there are any
        (`fourpi_beads.stack_views`); the browser's axial axis is the beads' own stage z."""
        from .fourpi_beads import stack_views
        views = stack_views(run.result["found"]["beads"], self._images(run), run.result["excluded"])
        return views, run.result.get("calibration")

    def summary(self, run: Run) -> str:
        """The beads, the fringe period, and how much of the light interferes: the coherence a pupil model
        learnt (the fraction its optics do not explain away) and each model's modulation at focus -- of a
        bead average the beads' own (`fourpi.modulation`)."""
        from ..fourpi import modulation
        cal = run.result.get("calibration")
        n = len(run.result["found"]["beads"].records) - len(run.result["excluded"])
        if cal is None:
            return f"{n} beads learnt"
        text = f"{n} beads learnt; fringe period {cal.period_nm:.1f} nm"
        coherence = cal.parameters.get("coherence")
        if coherence is not None:
            text += "; coherence " + ", ".join(f"{c:.2f}" for c in coherence)
        return text + "; modulation at focus " + ", ".join(f"{m:.2f}" for m in modulation(cal))


class FourPiPupil(_FourPi):
    """The pupil model in the 4Pi mode: uiPSF's 4Pi pupil model learnt from the beads (`fourpi_beads.learn`),
    rendered at three phases and saved as I, A and B with the period it shows."""
    key, label = "fourpi_pupil", "pupil model (uiPSF), 4Pi"
    START_SECONDS, SECONDS_PER_FRAME = 15.0, 0.03
    sections = {"start from": False, "microscope": True, "model": False, "spline": False, "averaging": False,
                "4Pi": True}
    needs_torch = True

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None, detected: Optional[Detected] = None) -> Run:
        from .fourpi_beads import FourPiBeads as Beads, learn, magnitude_modes_of, save, steps_of
        from .pupil import default_save_path
        found = detected.found if detected is not None else self.detect(paths, dual, inputs, progress).found
        beads = found["beads"]
        excluded = sorted(b for b, wish in (marks or {}).items() if wish == "exclude")
        keep = [i for i in range(len(beads.records)) if i not in excluded]
        if not keep:
            raise ValueError("every bead is excluded")
        used = Beads(beads.data[keep], beads.roi_xy[keep], beads.z_nm, [beads.records[i] for i in keep])
        optics = inputs["optics"].resolved(inputs.get("camera"))
        if optics["pixel"] is None:
            raise ValueError("4Pi beads need the camera's pixel size")
        settings = inputs["fourpi"]
        fit = learn(used, NA=optics["NA"], wavelength_nm=optics["wavelengths"][0], n_imm=optics["n_immersion"],
                    pixel_size_nm=optics["pixel"], steps_rad=steps_of(settings, used.data.shape[3]),
                    n_max=settings.n_max, photons_per_observation=settings.photons_per_observation,
                    bead_radius_nm=settings.bead_diameter_nm / 2, progress=progress, cancel=cancel,
                    kind="vector" if settings.vectorial else "scalar",
                    magnitude_modes=magnitude_modes_of(settings), same_magnitude=settings.same_magnitude)
        run = Run({"fit": fit, "found": found, "excluded": excluded, "settings": settings,
                   "wavelength_nm": optics["wavelengths"][0]}, False)
        if inputs["run"].save_model:
            path = previous.saved if previous is not None and previous.saved is not None else \
                default_save_path(paths, False).with_name(default_save_path(paths, False).stem + "_4pi.h5")
            run.saved = Path(path)
            run.result["calibration"] = save(fit, path, settings.model_z_range_nm, settings.model_dz_nm,
                                             overwrite=True, layout=found["layout"])
        return run



class FourPiAverage(_FourPi):
    """The bead average in the 4Pi mode: I, A and B solved from the beads themselves, a least squares per
    voxel over beads and channels (`fourpi.iab_from_beads`), no physical model.  A scan of one phase per
    slice; a model-independent check of the pupil model, and one with no PyTorch."""
    key, label = "fourpi_average", "bead average, 4Pi"
    sections = {"start from": False, "microscope": False, "model": False, "spline": False, "averaging": False,
                "4Pi": True}

    def run(self, paths, dual, inputs, progress=None, cancel=None, previous: Optional[Run] = None,
            marks: Optional[dict] = None, detected: Optional[Detected] = None) -> Run:
        from ..fourpi import save_fourpi_calibration
        from .fourpi_beads import average_model
        from .pupil import default_save_path
        found = detected.found if detected is not None else self.detect(paths, dual, inputs, progress).found
        excluded = sorted(b for b, wish in (marks or {}).items() if wish == "exclude")
        settings = inputs["fourpi"]
        cal = average_model(found, excluded, settings)
        run = Run({"fit": None, "found": found, "excluded": excluded, "settings": settings,
                   "calibration": cal}, False)
        if inputs["run"].save_model:
            path = previous.saved if previous is not None and previous.saved is not None else \
                default_save_path(paths, False).with_name(default_save_path(paths, False).stem + "_4pi.h5")
            save_fourpi_calibration(path, cal, overwrite=True)
            run.saved = Path(path)
        return run

    def save_as(self, run: Run, path) -> Path:
        from ..fourpi import save_fourpi_calibration
        save_fourpi_calibration(path, run.result["calibration"], overwrite=True)
        run.saved = Path(path)
        return run.saved


#: what the window's *mode* box offers: the channel geometry, beside the method that makes the PSF
MODES = ("Single channel", "Dual colour", "4Pi")

METHODS = (BeadAverage(), PupilModel(), InSitu(), EllipticalGaussian())

#: the methods in the 4Pi mode, by the key of the method they are in the others
FOURPI = {"average": FourPiAverage(), "pupil": FourPiPupil()}


def modes_of(method) -> tuple:
    """The modes a method (as the method box names it) can run in."""
    return tuple(m for m in MODES if for_mode(method, m) is not None)


def for_mode(method, mode: str):
    """The entry that runs ``method`` in ``mode``, or None where there is none: the 4Pi variant in the 4Pi
    mode, the method itself in the others -- except in situ (one channel, for now) and the elliptical
    Gaussian (one channel)."""
    if mode == "4Pi":
        return FOURPI.get(method.key)
    if mode == "Dual colour" and method.key in ("insitu", "elliptical"):
        return None
    return method
