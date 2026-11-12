"""A PSF calibration without the window: what *Detect beads* and *Calibrate* do, from settings.

The PSF calibration window gathers its inputs from a dozen forms (`PSFCalibrationWindow._inputs`), and
until 2026-11 nothing else could: a script had `pupil.run_calibration` and `insitu_run.run_insitu`, but no
way to the bead average, the elliptical Gaussian, the molecules or 4Pi through one call, and the plugin
*Localize/PSF calibration* only opened the window -- so an assistant with no window, a chain and a batch
had no calibration at all.  So the gathering is here, Qt-free, and the window calls it as the plugin does:

* `method_for` -- the method a sample, a method's label or key and a mode name (`psf_methods.for_mode`);
* `inputs` -- the method's inputs from its settings, each part asked for by its name only when the method
  reads it (`part`), so a part the method ignores is never read -- a field half typed in a hidden form
  stopped no run in the window, and stops none here;
* `calibrate` -- detect, the marks a detection makes (doubles left out), the run and its finish, as one
  *Calibrate* with nothing marked by hand;
* `list_calibration` -- the saved file into the list the fitters' *auto* chooses from.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .psf_methods import Run, for_mode, methods_of


def method_for(sample: Optional[str], method: str, mode: str):
    """The method that runs, from the window's three choices: ``method`` by its label or key, in
    ``sample`` (None: the sample whose methods have it).  A ValueError names what there is."""
    from .psf_methods import METHODS, SAMPLES, modes_of
    found = None
    for m in METHODS:
        if method in (m.label, m.key) and (sample is None or m in methods_of(sample)):
            found = m
            break
    if found is None:
        where = methods_of(sample) if sample else METHODS
        raise ValueError(f"no method {method!r}" + (f" for {sample}" if sample else "") + "; there are "
                         + ", ".join(f"{m.label} ({m.key})" for m in where)
                         + ("" if sample is None or sample in SAMPLES else
                            f"; the samples are {', '.join(SAMPLES)}"))
    if mode not in modes_of(found):
        raise ValueError(f"{found.label} has no mode {mode!r}; it has {', '.join(modes_of(found))}")
    return for_mode(found, mode), mode == "Dual colour"


def stacks_camera(paths, optics, require: bool = False):
    """The first stack's camera, resolved with the microscope and the camera ``optics`` chooses and
    the camera numbers typed in it -- what every run uses (`PSFCalibrationWindow._stacks_camera`)."""
    from ..io.tiff import open_stack
    from ..plugins.fit import CameraSettings
    from . import pupil
    from .input import discover_acquisitions
    # a fitter's camera settings: the same resolution, the last microscope used included
    settings = CameraSettings(microscope=optics.microscope, camera=optics.camera, **optics.camera_overrides())
    # a path nothing is found in is opened as it is, and says what is wrong with it
    path = (discover_acquisitions(list(paths)[:1]) or [list(paths)[0]])[0]
    if require:
        return pupil.camera_for(path, settings)
    return settings.resolve(open_stack(path), require=False)


def main_half(method, dual: bool, part: Callable[[str], Any]):
    """The dual calibration's main half as chosen, or None on the microscope's: the *channels* of the
    part that holds the sample (molecules, else beads)."""
    if not dual:
        return None
    try:
        channels = part("molecules" if method.sections.get("molecules") else "beads").channels
    except (ValueError, AttributeError):
        return None
    return None if channels.main_channel == "microscope" else channels.main_channel


def inputs(method, dual: bool, paths, part: Callable[[str], Any]) -> dict:
    """The method's inputs, from ``part(name)`` -- run, beads, in situ, in situ detection, molecules,
    elliptical, 4Pi, microscope, model, spline -- asked for only where the method reads it.  *beads* is
    the dual form (`DualColorSettings`) in dual colour and the single one (`CalibrationSettings`)
    otherwise.  A ValueError says what is missing: the microscope's optics for a model."""
    insitu = method.sections.get("in situ", False)
    out = dict(detection=part("in situ detection") if insitu else part("beads"), run=part("run"))
    if insitu:
        out["insitu"] = part("in situ")
    if method.sections.get("elliptical", False):
        out["elliptical"] = part("elliptical")
    if method.sections.get("molecules", False):
        out["molecules"] = part("molecules")
    if method.sections.get("4Pi", False):
        out["fourpi"] = part("4Pi")
    optics = part("microscope")
    if paths:
        # the camera every run takes, chosen and typed values included.  Incomplete, the method
        # resolves its own and says what is missing
        try:
            out["camera"] = stacks_camera(paths, optics, require=True)
        except ValueError:
            pass
    if method.sections["microscope"]:
        out.update(optics=optics, model=part("model"), spline=part("spline"))
        try:
            camera = stacks_camera(paths, optics) if paths else None
        except Exception:                            # a file that cannot be read says so when it is run
            camera = None
        missing = optics.missing(camera, 2 if dual else 1, main_half(method, dual, part))
        if missing:
            scope = (getattr(camera, "microscope", None) or {}).get("name")
            raise ValueError(f"give the microscope's {' and '.join(missing)}: "
                             + (f"{scope} does not give it for these stacks" if scope else
                                "no microscope in the database recognises these stacks"))
    return out


def calibrate(method, dual: bool, paths, inputs: dict, progress=None, cancel=None) -> Run:
    """Detect, then calibrate from what was found with the marks a detection makes (the beads marked
    doubles left out), then the method's finish (fit quality, the model against the data): one
    *Calibrate* of the window with nothing marked by hand.  ``progress(entry)`` as the window's
    (`LearningProgress`); ``cancel`` a `threading.Event` that stops a learning."""
    paths = list(paths)
    report = progress or (lambda entry: None)
    found = method.detect(paths, dual, inputs, report)
    report({"message": found.text})
    report({"restart": True})
    marks_of = getattr(method, "detected_marks", None)
    marks = dict(marks_of(found)) if marks_of else {}
    run = method.run(paths, dual, inputs, report, cancel, detected=found, marks=marks or None)
    run.extra["inputs"] = inputs
    run.extra["detected"] = found.text
    return method.finish(run, report)


def list_calibration(path, method, dual: bool, paths, camera=None) -> None:
    """List a saved calibration where the fitters' *auto* chooses from (`calibration_list`), described
    by the stacks' microscope -- a bead average does not record it in its file -- and the day they were
    taken.  Never a failure of the save: the list is a convenience."""
    from .. import calibration_list
    from ..io.tiff import camera_tags, open_stack
    from ..microscope_db import acquisition_date
    if not getattr(method, "listed", True):
        return                                 # not a PSF: the elliptical Gaussian's curves, a 4Pi model
    made = None
    try:
        day = acquisition_date(camera_tags(open_stack(paths[0]))) if paths else None
        made = day.isoformat() if day else None
    except Exception:
        pass
    try:
        calibration_list.add(path, microscope=getattr(camera, "microscope", None) or None,
                             em_on=getattr(camera, "em_on", None), dual=dual, made=made, method=method.label)
    except Exception as error:
        print(f"PSF calibration: could not list {path}: {error}")


def fitter_of(method, dual: bool) -> str:
    """The fitter a calibration of this method is for, by its path."""
    if method.key == "fourpi" or method.sections.get("4Pi", False):
        return "Localize/Spline 4Pi"
    if not getattr(method, "listed", True):
        return "Analysis/After fit/Astigmatic z"
    return "Localize/Spline 3D 2C" if dual else "Localize/Spline 3D"

