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
        from uipsf.diagnostics.figures import PHASE_UNITS
        pages = []
        halves = getattr(cal, "halves", None)            # after *Uncertainty from halves*
        for title, draw in figures(cal, reference=run.result.start, halves=halves).items():
            if title == "Beads":
                continue
            if title.startswith("Pupil and Zernikes"):
                # the phase in nm, waves or rad: a choice above the page, as the Zernike table has all three;
                # nm first, the unit an optician reads (the first choice is the one shown)
                units = ("nm",) + tuple(u for u in PHASE_UNITS if u != "nm")
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
        return f"{len(cal.beads.used)} of {len(cal.beads.rows)} beads used"


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
        from uipsf.diagnostics.figures import PHASE_UNITS, zernike_table

        from .insitu_views import draw_model_vs_data, draw_rounds, draw_z_bins
        result = run.result
        cal = result.cal
        pages = [Page("z bins", lambda f, s, e: draw_z_bins(f, result.rois, result.run), follows_selection=False),
                 Page("Rounds and edge", lambda f, s, e: draw_rounds(f, result.run), follows_selection=False)]
        for title, draw in figures(cal, reference=result.start).items():
            if title == "Beads":
                continue
            if title.startswith("Pupil and Zernikes"):
                units = ("nm",) + tuple(u for u in PHASE_UNITS if u != "nm")
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
        text = (f"{len(result.cal.beads.used)} frames used, {len(result.run.rounds)} rounds, focus {focus:.0f} nm"
                + ("" if twin is None else f", twin test: {twin.verdict}"))
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


METHODS = (BeadAverage(), PupilModel(), InSitu())
