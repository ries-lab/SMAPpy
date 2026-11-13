"""The assistant's actions inside SMAPpy's own windows: the bead and the PSF calibration.

`open_window` opened these and nothing more, so "calibrate these beads" ended at an empty
window the assistant could neither fill nor press (a user's report, 2026-11-12).  Each action
here is what one of the window's controls does -- a field set, a button pressed -- done by the
method the click calls, on the window the user sees, so the work is watched where it would be
made by hand and a figure, a table or a mark is the same either way.  `actions_of` in
`assistant_actions.py` adds them to the menus' actions, under the window's name.

What a click would ask in a dialog is an argument; what a dialog would tell is the answer or the
window's *State*.  A job runs in the window's worker as a click's does, so an action that starts
one returns at once: *State* says when it is done, and a job the assistant started fails into
the window's status and ``last_error``, never a modal box, which would hold the assistant's queue
(`CalibrationWindow.quiet_job`).  The calculations are the windows' Qt-free methods
(`calibrate.core`, `calibrate.dual`, `calibrate.psf_methods`), which a script calls without a
window.

Setting the inputs and pressing what only computes inside the window is made in suggest mode
too (``shows``), as `open_plugin` filling a form is; what writes a file or hands a calibration
to a fitter is a proposal there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..assistant.tools import Action, ToolError, _plain, _replaced, _within
from ..calibrate.medium import FIELDS as MEDIUM_FIELDS

#: How many rows of a bead table *State* gives: a stack of a few hundred beads in full is
#: thousands of numbers, of which a model reads the counts and the odd bead.
MAX_ROWS = 200

PATHS = {"type": ["string", "array"], "items": {"type": "string"},
         "description": "a stack or a folder of stacks, or several"}
BEADS = {"type": "array", "items": {"type": "integer"},
         "description": "beads by the table's row, from 0, as State lists them"}
VALUES = {"type": "object", "description": "flat dotted names, as State lists them, e.g. "
                                           "{\"roi_size\": 19} or {\"channels.layout\": \"right-left\"}"}


# ------------------------------------------------------------------- shared
def _status(window) -> str:
    return window.status_bar.currentMessage()


def _table(table) -> Dict[str, Any]:
    """A QTableWidget as its columns and rows, at most `MAX_ROWS` of them."""
    columns = [table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else str(c)
               for c in range(table.columnCount())]
    rows = []
    for r in range(min(table.rowCount(), MAX_ROWS)):
        rows.append([table.item(r, c).text() if table.item(r, c) else "" for c in range(len(columns))])
    out: Dict[str, Any] = {"columns": columns, "rows": rows}
    if table.rowCount() > MAX_ROWS:
        out["not shown"] = f"{table.rowCount() - MAX_ROWS} more rows"
    return out


def _values(form) -> Any:
    from ..plugins import settings_values
    try:
        return _plain(settings_values(form.value()))
    except ValueError as error:                   # a field typed that does not read
        return f"does not read: {error}"


def _set_form(form, values: Dict[str, Any], what: str) -> Dict[str, Any]:
    """``values`` (dotted names) into ``form``, as typed: only what changed is set, and each change
    is said as an edit is (`field_changed`), so the window does what it does on an edit -- forget
    the detected beads, read the microscope again.  A name the form lacks is refused, naming its
    fields (`tools._replaced`)."""
    from ..plugins import settings_values
    try:
        old = form.value()
    except ValueError as error:
        raise ToolError(f"{what}: a field does not read ({error}); set it first") from None
    new = _replaced(old, dict(values or {}), what)
    before, after = settings_values(old), settings_values(new)
    changed = {k: v for k, v in after.items() if before.get(k) != v}
    if changed:
        form.set_values(changed)
        for name in changed:
            form.field_changed.emit(name)
    return _plain(settings_values(form.value()))


def _paths(paths) -> List[str]:
    paths = [paths] if isinstance(paths, str) else list(paths or [])
    missing = [p for p in paths if not Path(p).expanduser().exists()]
    if not paths or missing:
        raise ToolError(f"no file {missing[0] if missing else '(none given)'}")
    return [str(Path(p).expanduser()) for p in paths]


def _found(paths, recursive: bool) -> List[str]:
    """What *Add...* or a drop adds: the acquisitions found in each file or folder, or the path
    itself where none is found (the detection then says what is wrong with it)."""
    from ..calibrate.input import discover_acquisitions
    found: List[str] = []
    for path in paths:
        try:
            hits = [str(p) for p in discover_acquisitions([path], recursive=recursive)]
        except Exception as error:
            raise ToolError(f"{path}: {type(error).__name__}: {error}") from None
        found += hits or [path]
    return found


def _save_target(path: Optional[str], default: Path, output: Path, stacks: List[str]) -> Path:
    """Where a *Save* writes: ``default`` (beside the stacks, as the window's dialog starts) when no
    path is given; a relative path under the output folder; an absolute one only under the output
    folder or the stacks' own folder -- the assistant writes nowhere else."""
    if not path:
        return default
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = output / target
    folders = [output] + [Path(s).expanduser().parent for s in stacks]
    if not any(_within(target, folder) for folder in folders):
        raise ToolError(f"{target} is outside the output folder ({output}) and the stacks' folders")
    return target


def _pages(tabs) -> List[str]:
    return [tabs.tabText(i) for i in range(tabs.count())]


def _show_page(tabs, page: str) -> str:
    titles = _pages(tabs)
    found = [i for i, t in enumerate(titles) if t.lower() == str(page).strip().lower()] or \
            [i for i, t in enumerate(titles) if str(page).strip().lower() in t.lower()]
    if not found or not tabs.isTabEnabled(found[0]):
        raise ToolError(f"no page {page!r} to show; there are {', '.join(titles)}")
    tabs.setCurrentIndex(found[0])
    return titles[found[0]]


def _select(table, beads) -> List[int]:
    rows = sorted({int(b) for b in beads})
    bad = [b for b in rows if not 0 <= b < table.rowCount()]
    if bad:
        raise ToolError(f"no bead {bad[0]}; the table has {table.rowCount()} rows (0 to "
                        f"{table.rowCount() - 1})")
    table.clearSelection()
    from PySide6.QtCore import QItemSelectionModel
    model = table.selectionModel()
    for r in rows:
        model.select(table.model().index(r, 0),
                     QItemSelectionModel.Select | QItemSelectionModel.Rows)
    return rows


def _started(window, start: Callable[[], None], what: str) -> Dict[str, Any]:
    """Start a job of the window's as its button does, quietly (`quiet_job`); its end is *State*'s."""
    window.quiet_job = True
    try:
        start()
    finally:
        if not window.busy:                       # it did not start: the flag is not left for a click
            window.quiet_job = False
    if not window.busy:
        raise ToolError(f"{what} did not start: {window.last_error or _status(window)}")
    return {"started": what, "then": "the window's State says busy false when it is done, "
                                     "and its result or the error"}


def _idle(window) -> None:
    if window.busy:
        raise ToolError("the window is busy with a job: State says when it is done")


# ---------------------------------------------------------- bead calibration
def bead_calibration_actions(control) -> List[Action]:
    """The bead calibration window's controls (`calibrate/qt_gui.CalibrationWindow`)."""
    name = "Bead calibration"

    def win():
        control.open_calibration()
        return control.calibration_window

    def state():
        w = win()
        result = w.result
        out: Dict[str, Any] = {
            "files": list(w.paths), "mode": w.mode, "modes": ["Single channel", "Dual colour"],
            "settings": _values(w.form), "busy": w.busy, "status": _status(w),
            "page": w.tabs.tabText(w.tabs.currentIndex()),
            "pages": [t for i, t in enumerate(_pages(w.tabs)) if w.tabs.isTabEnabled(i)]}
        if w.last_error:
            out["error"] = w.last_error
        if result is not None:
            import numpy as np
            out["result"] = {"beads": len(result.beads.records),
                             "used": int(np.sum(result.accepted)),
                             "excluded by hand": sorted(int(b) for b in w.excluded),
                             "fit quality": w.diagnostics is not None,
                             "saved": getattr(w, "saved_path", None)}
            out["beads"] = _table(w.bead_table)
        return out

    def add(path, subdirectories=True):
        w = win()
        _idle(w)
        w.add_paths(_found(_paths(path), bool(subdirectories)))
        return {"files": list(w.paths)}

    def remove(path=None):
        w = win()
        _idle(w)
        if path is None:
            w.paths.clear()
        else:
            gone = {str(Path(p).expanduser()) for p in ([path] if isinstance(path, str) else path)}
            unknown = sorted(gone - set(w.paths))
            if unknown:
                raise ToolError(f"{unknown[0]} is not in the list; it has {', '.join(w.paths) or 'none'}")
            w.paths[:] = [p for p in w.paths if p not in gone]
        w._refresh_files()
        return {"files": list(w.paths)}

    def mode(mode):
        w = win()
        _idle(w)
        chosen = {"single channel": "Single channel", "single": "Single channel",
                  "dual colour": "Dual colour", "dual color": "Dual colour",
                  "dual": "Dual colour"}.get(str(mode).strip().lower())
        if chosen is None:
            raise ToolError(f"mode: no {mode!r}; it is 'Single channel' or 'Dual colour'")
        w.mode_box.setCurrentText(chosen)
        return {"mode": w.mode, "settings": _values(w.form)}

    def settings(values):
        w = win()
        _idle(w)
        return {"settings": _set_form(w.form, values, "settings")}

    def calibrate():
        w = win()
        _idle(w)
        if not w.paths:
            raise ToolError(f"no bead stacks: {name}/Add files first")
        try:
            w.settings()
        except ValueError as error:
            raise ToolError(f"a setting does not read: {error}") from None
        return _started(w, w.run, "detecting and calibrating")

    def exclude(beads, exclude=True):
        w = win()
        _idle(w)
        if w.result is None:
            raise ToolError(f"no calibration yet: {name}/Detect and calibrate first")
        rows = _select(w.bead_table, beads)
        w.excluded = (w.excluded | set(rows)) if exclude else (w.excluded - set(rows))
        w.refresh_table()
        w.redraw()
        return {"excluded": sorted(w.excluded),
                "then": f"{name}/Recalculate builds the calibration without them"}

    def recalculate():
        w = win()
        _idle(w)
        if w.result is None:
            raise ToolError(f"no calibration yet: {name}/Detect and calibrate first")
        return _started(w, w.rebuild, f"recalculating without {len(w.excluded)} bead(s)")

    def quality():
        w = win()
        _idle(w)
        if w.result is None:
            raise ToolError(f"no calibration yet: {name}/Detect and calibrate first")
        return _started(w, w.fit_quality, "fitting the beads back")

    def page(page):
        w = win()
        return {"page": _show_page(w.tabs, page)}

    def save(path=None):
        from ..calibrate.core import calibration_save_defaults
        w = win()
        _idle(w)
        if w.result is None:
            raise ToolError(f"no calibration yet: {name}/Detect and calibrate first")
        folder, stem = calibration_save_defaults(w.paths, dual=w.is_dual)
        output = control._assistant().server.assistant.output
        target = _save_target(path, Path(folder) / (stem + ".h5"), output, w.paths)
        try:
            saved = w.save_to(target)
        except Exception as error:
            raise ToolError(f"not saved: {type(error).__name__}: {error}") from None
        fitter = "Spline 3D 2C" if w.is_dual else "Spline 3D"
        return {"saved": saved, "fitter": f"the {fitter} fitter now uses it"}

    def use():
        w = win()
        if not getattr(w, "saved_path", None):
            raise ToolError(f"nothing saved yet: {name}/Save first")
        w._use_in_fitter()
        return {"fitter": _status(w)}

    return [
        Action(f"{name}/State", "The bead calibration window as it stands: its stacks, mode, "
               "settings (by the dotted names Settings takes), whether a job runs, the status line "
               "or the last error, and after a calibration how many beads it used and the bead "
               "table.  Opens the window.", state, shows=True),
        Action(f"{name}/Add files", "Bead z-stacks into the window's list: each file or folder "
               "searched for acquisitions, into subfolders unless subdirectories is false.",
               add, {"path": PATHS, "subdirectories": {"type": "boolean"}}, ("path",), shows=True),
        Action(f"{name}/Remove files", "Stacks out of the list, by path; without a path, all.",
               remove, {"path": PATHS}, shows=True),
        Action(f"{name}/Mode", "Single channel or Dual colour (a split chip, registered): the "
               "files and the shared settings are kept, a result dropped.", mode,
               {"mode": {"type": "string", "enum": ["Single channel", "Dual colour"]}},
               ("mode",), shows=True),
        Action(f"{name}/Settings", "Set some of the mode's settings, by the dotted names State "
               "lists; the rest stay.  read_doc('Panels/Bead calibration') explains them.",
               settings, {"values": VALUES}, ("values",), shows=True),
        Action(f"{name}/Detect and calibrate", "Detect + calibrate: find the beads, average them "
               "into a spline PSF; the beads marked doubles start excluded.  Runs in the window; "
               "returns at once.  Nothing is saved.", calibrate, shows=True),
        Action(f"{name}/Exclude beads", "Leave beads out (exclude=false lets them in again), by "
               "their row in State's bead table, then Recalculate.", exclude,
               {"beads": BEADS, "exclude": {"type": "boolean"}}, ("beads",), shows=True),
        Action(f"{name}/Recalculate", "Build the calibration again from the same beads without "
               "the excluded ones; returns at once.", recalculate, shows=True),
        Action(f"{name}/Calculate fit quality", "Fit the beads back with the fitter the data will "
               "meet, for the Fit quality page; returns at once.", quality, shows=True),
        Action(f"{name}/Show page", "Bring a page of plots to the front: Overview, "
               "Transformation, Fit quality, Field diagnostics, Bead diagnostics, CRLB.", page,
               {"page": {"type": "string"}}, ("page",), shows=True),
        Action(f"{name}/Save", "Save the calibration (.h5) -- by default beside the stacks, as "
               "the window's dialog offers; a path given is under the output folder or the "
               "stacks' folder -- and put it into the Spline 3D (or 2C) fitter.", save,
               {"path": {"type": "string", "description": "the .h5 to write; default beside the stacks"}}),
        Action(f"{name}/Use in fitter", "Put the saved calibration into the Spline 3D fitter "
               "(dual colour: Spline 3D 2C) again.", use),
    ]


# ----------------------------------------------------------- PSF calibration
def _psf_parts(w) -> Dict[str, Any]:
    """The PSF window's forms by the name its sections give them, those the chosen method reads
    (`PSFCalibrationWindow._show_method`).  *run* is two forms of one `RunSettings`: *start from*
    shown with the model, *save model* with the run."""
    reads = w.method.sections
    insitu, fourpi = reads.get("in situ", False), reads.get("4Pi", False)
    molecules = reads.get("molecules", False)
    parts = {"run": w.run_form, "microscope": w.optics_form}
    if reads["model"]:
        parts["model"] = w.model_form
    if reads["spline"]:
        parts["spline"] = w.spline_form
    if insitu:
        parts["in situ"] = w.insitu_form
        parts["in situ detection"] = w.insitu_detection_form
    if molecules:
        parts["molecules"] = w.molecule_form
    if reads.get("elliptical", False):
        parts["elliptical"] = w.elliptical_form
    if fourpi:
        parts["4Pi"] = w.fourpi_form
    if not insitu and not fourpi and not molecules:
        parts["beads"] = w.detection_form
    return parts


def psf_calibration_actions(control) -> List[Action]:
    """The PSF calibration window's controls (`calibrate/psf_window.PSFCalibrationWindow`)."""
    from ..calibrate.psf_methods import MODES, SAMPLES, methods_of, modes_of
    name = "PSF calibration"

    def win():
        control.open_psf_calibration()
        return control.psf_calibration_window

    def part_values(w) -> Dict[str, Any]:
        out = {}
        for part, form in _psf_parts(w).items():
            values = _values(form)
            if part == "run" and isinstance(values, dict):
                values["start_from"] = w.start_form.value().start_from
                if not w.method.sections["start from"]:
                    values.pop("start_from")
            if part == "microscope" and isinstance(values, dict):
                # the sample's index is the optics', shown in a form of its own (`medium_form`)
                sample = _values(w.medium_form)
                if isinstance(sample, dict):
                    values.update({k: sample[k] for k in MEDIUM_FIELDS})
            out[part] = values
        return out

    def state():
        from ..calibrate.psf_window import torch_missing
        w = win()
        out: Dict[str, Any] = {
            "files": list(w.paths), "sample": w.sample_box.currentText(),
            "method": w.method_box.currentText(), "mode": w.mode_box.currentText(),
            "choices": {s: {m.label: list(modes_of(m)) for m in methods_of(s)} for s in SAMPLES},
            "settings": part_values(w), "busy": w.busy, "status": _status(w),
            "page": w.tabs.tabText(w.tabs.currentIndex()), "pages": _pages(w.tabs)}
        if w.busy and w.progress_bar.isVisible():
            out["progress"] = w.progress_bar.text()
        if w.found_label.text():
            out["found"] = w.found_label.text()
        if w.warning_label.text():
            out["warning"] = w.warning_label.text()
        if w.last_error:
            out["error"] = w.last_error
        missing = torch_missing()
        if missing and (w.method.key in ("pupil", "insitu", "molecules")
                        or getattr(w.method, "needs_torch", False)):
            out["needs"] = missing
        if w.marks:
            out["marks"] = {int(b): wish for b, wish in w.marks.items()}
        if w.run_result is not None:
            out["result"] = {"summary": w.method.summary(w.run_result),
                             "saved": str(w.run_result.saved) if w.run_result.saved else None,
                             "dual": w.run_result.dual}
        if w.run_result is not None or w.detected is not None:
            out["beads"] = _table(w.bead_table)
        return out

    def add(path, subdirectories=True):
        w = win()
        _idle(w)
        said: List[str] = []
        # a movie added to a bead method is asked about in a box (`_check_kind`): here it is
        # kept, and what the box said is the answer, for the model to choose the sample
        w._ask = lambda box: said.append(box.text())
        try:
            w.add_paths(_found(_paths(path), bool(subdirectories)))
        finally:
            del w._ask
        out: Dict[str, Any] = {"files": list(w.paths), "sample": w.sample_box.currentText(),
                               "method": w.method_box.currentText(),
                               "mode": w.mode_box.currentText()}
        if said:
            out["note"] = said[0].replace("\n\n", " ").replace("\n", ", ") + \
                f"  Kept; {name}/Choose sets the sample that reads them."
        return out

    def remove(path=None):
        w = win()
        _idle(w)
        if path is None:
            w.clear_paths()
        else:
            gone = {str(Path(p).expanduser()) for p in ([path] if isinstance(path, str) else path)}
            unknown = sorted(gone - set(w.paths))
            if unknown:
                raise ToolError(f"{unknown[0]} is not in the list; it has {', '.join(w.paths) or 'none'}")
            w.paths[:] = [p for p in w.paths if p not in gone]
            w._refresh_files()
        return {"files": list(w.paths)}

    def choose(sample=None, method=None, mode=None):
        from ..assistant.tools import _pick
        w = win()
        _idle(w)
        sample = _pick("sample", sample, [(s, s) for s in SAMPLES]) if sample else \
            w.sample_box.currentText()
        if method:
            options = [(m.key, m.label) for m in methods_of(sample)]
            key = _pick("method", method, options)
            method = next(m.label for m in methods_of(sample) if m.key == key)
        w.choose(sample, method)
        if mode:
            modes = [w.mode_box.itemText(i) for i in range(w.mode_box.count())]
            w.mode_box.setCurrentText(_pick("mode", mode, [(m, m) for m in modes]))
        return {"sample": w.sample_box.currentText(), "method": w.method_box.currentText(),
                "mode": w.mode_box.currentText(), "settings": part_values(w)}

    def settings(part, values):
        from ..assistant.tools import _pick
        w = win()
        _idle(w)
        parts = _psf_parts(w)
        chosen = _pick("part", part, [(p, p) for p in parts])
        if chosen == "run":
            values = dict(values or {})
            start = {k: values.pop(k) for k in list(values) if k == "start_from"}
            if start:
                _set_form(w.start_form, start, "run")
            if values:
                _set_form(w.run_form, values, "run")
        elif chosen == "microscope":
            values = dict(values or {})
            medium = {k: values.pop(k) for k in list(values) if k in MEDIUM_FIELDS}
            if medium:
                _set_form(w.medium_form, medium, "microscope")
            if values:
                _set_form(w.optics_form, values, "microscope")
        else:
            _set_form(parts[chosen], values, chosen)
        return {"settings": part_values(w)}

    def ready(w) -> None:
        from ..calibrate.psf_window import torch_missing
        _idle(w)
        if not w.paths:
            raise ToolError(f"no files: {name}/Add files first")
        if (w.method.key in ("pupil", "insitu", "molecules") or getattr(w.method, "needs_torch", False)) \
                and torch_missing() is not None:
            raise ToolError(f"{w.method.label} needs PyTorch: {torch_missing()}")
        try:
            w._inputs()                                 # what a click would refuse in a box
        except ValueError as error:
            raise ToolError(str(error)) from None

    def detect():
        w = win()
        ready(w)
        return _started(w, w.detect, "detecting")

    def calibrate():
        w = win()
        ready(w)
        saving = w.run_form.value().save_model
        out = _started(w, w.run, "calibrating")
        out["saves"] = ("the calibration beside the stacks, into the fitter it belongs to"
                        if saving else "nothing: save model is off; Save as writes it")
        return out

    def stop():
        w = win()
        if not w.busy:
            raise ToolError("nothing is running")
        w._cancel.set()
        return {"stopping": "within seconds; nothing is saved"}

    def mark(beads):
        w = win()
        _idle(w)
        if w.run_result is None and w.detected is None:
            raise ToolError(f"no beads yet: {name}/Detect first")
        _select(w.bead_table, beads)
        w.toggle_beads()
        return {"marks": {int(b): wish for b, wish in w.marks.items()},
                "then": f"{name}/Calibrate uses them"}

    def halves(splits=3):
        w = win()
        _idle(w)
        run = w.run_result
        if run is None or not getattr(w.method, "has_halves", False):
            raise ToolError("no pupil model calibrated to split")
        why = w.method.halves_needed(run)
        if why:
            raise ToolError(why)
        try:
            w._inputs()
        except ValueError as error:
            raise ToolError(str(error)) from None
        return _started(w, lambda: w.halves(int(splits)), f"learning {2 * int(splits)} halves")

    def page(page):
        w = win()
        return {"page": _show_page(w.tabs, page)}

    def save_as(path=None):
        from ..calibrate.pupil import default_save_path
        w = win()
        _idle(w)
        if w.run_result is None:
            raise ToolError(f"no calibration yet: {name}/Calibrate first")
        default = Path(w.run_result.saved or default_save_path(w.paths, w.run_result.dual))
        output = control._assistant().server.assistant.output
        target = _save_target(path, default, output, w.paths)
        try:
            saved = w.save_to(target)
        except Exception as error:
            raise ToolError(f"not saved: {type(error).__name__}: {error}") from None
        return {"saved": saved, "fitter": _status(w)}

    def use():
        w = win()
        if w.run_result is None or w.run_result.saved is None:
            raise ToolError(f"nothing saved yet: {name}/Save as first")
        w._use_in_fitter()
        return {"fitter": _status(w)}

    def keep_pupil():
        from PySide6.QtWidgets import QMessageBox
        w = win()
        _idle(w)
        if w._learnt() is None:
            raise ToolError("no pupil learnt: a pupil model or in situ calibration first")
        said: List[str] = []
        # the click asks in a box; the assistant's call is the user's yes (act mode or a proposal
        # carried out), and what the box said is the answer
        w._confirm = lambda text: said.append(text) or QMessageBox.Ok
        try:
            w._keep_pupil()
        finally:
            del w._confirm
        return {"kept": said[0] if said else "", "status": _status(w)}

    return [
        Action(f"{name}/State", "The PSF calibration window as it stands: its files, the sample, "
               "method and mode chosen (and which there are), the settings of each part the "
               "method reads (by the dotted names Settings takes), whether a job runs and its "
               "progress, the status line, what Detect found, a warning or the last error, and "
               "after a run its summary, the saved file and the bead table.  Opens the window.",
               state, shows=True),
        Action(f"{name}/Add files", "Bead z-stacks, z sweeps or movies into the list: each file or "
               "folder searched for acquisitions, into subfolders unless subdirectories is false.  "
               "4Pi scans switch the mode to 4Pi; files that do not suit the method are kept and "
               "said in the answer.", add, {"path": PATHS, "subdirectories": {"type": "boolean"}},
               ("path",), shows=True),
        Action(f"{name}/Remove files", "Files out of the list, by path; without a path, all.",
               remove, {"path": PATHS}, shows=True),
        Action(f"{name}/Choose", "What was imaged and how to calibrate it: the sample ("
               + ", ".join(SAMPLES) + "), the method (by its label or key: pupil, average, "
               "insitu, elliptical, molecules, ...) and the mode (" + ", ".join(MODES) + "); State "
               "lists which go together.  Files and settings are kept, a result dropped.",
               choose, {"sample": {"type": "string"}, "method": {"type": "string"},
                        "mode": {"type": "string"}}, shows=True),
        Action(f"{name}/Settings", "Set some settings of one part -- run, microscope, model, "
               "spline, beads, in situ, in situ detection, molecules, elliptical, 4Pi: those the "
               "method reads, as State lists them -- by dotted names; the rest stay.  "
               "read_doc('Panels/PSF calibration') explains them.", settings,
               {"part": {"type": "string"}, "values": VALUES}, ("part", "values"), shows=True),
        Action(f"{name}/Detect", "Detect beads (or molecules): find them, ring them on the Stacks "
               "page and say how long calibrating will take; nothing is learnt.  Returns at once.",
               detect, shows=True),
        Action(f"{name}/Calibrate", "Calibrate with the settings as they stand, from the detected "
               "beads and their marks (detecting first if needed).  With save model on (the "
               "default) the calibration is saved beside the stacks and put into its fitter.  "
               "Returns at once; a pupil model takes minutes.", calibrate),
        Action(f"{name}/Stop", "Stop the learning within seconds; nothing is saved.", stop,
               shows=True),
        Action(f"{name}/Mark beads", "Exclude / include: mark beads by their row in State's table "
               "for the next Calibrate -- a used one to be left out, for the pupil model a rejected "
               "one to be let in; marked again, unmarked.", mark, {"beads": BEADS}, ("beads",),
               shows=True),
        Action(f"{name}/Uncertainty from halves", "Learn the pupil model again on random halves of "
               "its beads, splits times two fits, for an error bar with the CRLB; saved into the "
               "calibration when it was saved.  Returns at once.", halves,
               {"splits": {"type": "integer", "description": "2 to 10, default 3"}}),
        Action(f"{name}/Show page", "Bring a page to the front: Stacks or one of the run's pages, "
               "as State lists them.", page, {"page": {"type": "string"}}, ("page",), shows=True),
        Action(f"{name}/Save as", "Save the calibration elsewhere, or when save model was off -- "
               "by default where it was saved or beside the stacks; a path given is under the "
               "output folder or the stacks' folder -- and put it into its fitter.", save_as,
               {"path": {"type": "string", "description": "the .h5 to write"}}),
        Action(f"{name}/Use in fitter", "Put the saved calibration into its fitter again: Spline "
               "3D, Spline 3D 2C, Spline 4Pi or Astigmatic z, by the method and mode.", use),
        Action(f"{name}/Add to microscope", "Keep the learnt pupil with the beads' microscope, "
               "which in situ learning starts from: changes the user's microscope file, replacing "
               "an earlier pupil for the same lens state.", keep_pupil),
    ]


def window_actions(control) -> Dict[str, Action]:
    """Every window's actions, by name, for `assistant_actions.actions_of`."""
    out: Dict[str, Action] = {}
    for make in (bead_calibration_actions, psf_calibration_actions):
        out.update({a.name: a for a in make(control)})
    return out


#: The windows `open_window` opens that have no actions inside, and why: each is one or has its
#: reason (`tests/test_assistant_gui.py`), so a window is never opened and left unreachable
#: without anyone deciding it.
WINDOWS_WITHOUT_ACTIONS = {
    "ROI manager": "its work is the ROIs' tools: set_roi, read_values(roi=), run_plugin per region",
    "3D view": "View/3D view shows it; the view is the user's to turn",
    "help": "read_doc reads the same pages",
    "batch window": "not yet: run_chain on the session; the smappy-batch command over files",
    "DECODE training": "not yet: the training is `smappy-decode train FOLDER` without the window",
    "microscope manager": "not yet: its entries are the user's database",
    "camera editor": "not yet: its entries are the user's database",
}

#: The windows that have actions, by the name `open_window` gives them and the prefix of theirs.
WINDOWS_WITH_ACTIONS = {"bead calibration": "Bead calibration",
                        "PSF calibration": "PSF calibration"}
