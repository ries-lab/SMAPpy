"""The microscope's PSF: a spline drawn from the pupil kept with its entry, where there is no bead calibration.

*PSF calibration*'s *Add to microscope...* keeps the objective's pupil with the microscope's entry, per state of
its 3D lens and colour (`microscope_db.keep_pupil`): what a bead model learnt that belongs to the objective --
the phase in nm of wavefront, γ and the learnt NA, the blur -- described as plain data
(`uipsf.VectorPSF.describe`).  In situ learning starts from it.  Here it becomes what a fitter reads: uipsf
draws it (`uipsf.export.from_description`) with the file's optics -- its emission, the pixel of its 3D lens's
state -- and smappy samples the drawing into a spline, as a pupil model's calibration is sampled
(`pupil.spline_calibration`).

**One file, not a second path through the fitter.**  The spline is written to the configuration directory
(`MICROSCOPE_FOLDER`, named by what it was drawn from, so a file is drawn once) and records the pupil and the
optics in its parameters (``pupil``), where a pupil model's calibration carries its model.  Spline 3D then
reads it as any calibration, and its *focus* draws it for the focus in the sample from the record
(`depth.model_of`) as it draws a bead model from its calibration: nothing downstream knows the difference.

**It is the beads' PSF, not the data's.**  A bead calibration of the day is better: the pupil is the
objective's as it was when the beads were taken, the blur theirs; the coverslip, the sample's index and a
re-aligned 3D lens are not in it.  So it is chosen by hand (*calibration*'s menu, `MICROSCOPE`), never by the
calibration list's *auto*.

*Simulate* draws its frames with it too (`choices`, `simulation_path`): there is no file there to name a
microscope, so every kept pupil is offered, drawn at its own emission and the simulation's pixel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

#: the calibration field's word for the microscope's pupil (Spline 3D's *calibration*)
MICROSCOPE = "microscope"
#: where the drawn splines are kept, under the configuration directory
MICROSCOPE_FOLDER = "microscope_psfs"
#: the spline's sampling: planes 20 nm apart over ±800 nm about focus, on a 27-pixel region -- a bead
#: calibration's usual range, which Spline 3D's *focus* keeps when it draws the PSF at depth
DZ_NM = 20.0
Z_RANGE_NM = 800.0
ROI_PX = 27


def record(summary: Optional[Dict[str, Any]], wavelength_nm: Optional[float] = None) -> Dict[str, Any]:
    """What the PSF of ``summary``'s microscope (a `MicroscopeResolution.summary`) is drawn from: its pupil
    for the state of its 3D lens and colour (`microscope_db.pupil_of`), and the optics -- the objective's NA
    and immersion from the entry, the emission (``wavelength_nm``, else the file's, else the pupil's), the
    pixel of the file's state, and the beads' medium the pupil was learnt in.  A ValueError says what is
    missing."""
    from ..microscope_db import database, emission_of, pupil_of
    if not summary or not summary.get("name"):
        raise ValueError("no microscope recognised for this file: choose one, or a bead calibration")
    name = str(summary["name"])
    pupil = pupil_of(summary)
    states = summary.get("states") or {}
    entry = database().get(name)
    key = entry.pupil_key(states, wavelength_nm or emission_of(summary)) if entry is not None else ""
    if pupil is None:
        raise ValueError(f"{name} keeps no pupil for {key or 'this state'}: PSF calibration's Add to "
                         f"microscope... keeps one from a bead model, or choose a bead calibration")
    values = summary.get("values") or {}
    pixel = values.get("pixel_nm")
    if pixel is None:
        raise ValueError(f"{name} gives no pixel for this file's state: the camera's pixel is missing")
    pixel = [float(pixel), float(pixel)] if np.isscalar(pixel) else [float(pixel[0]), float(pixel[-1])]
    wavelength = wavelength_nm or emission_of(summary) or pupil.get("wavelength_nm")
    if wavelength is None:
        raise ValueError(f"{name} gives no emission for this file, and its pupil none")
    optics = {"NA": float(values.get("NA") or pupil["NA"]),
              "wavelength_nm": float(wavelength),
              "n_imm": float(values.get("n_immersion") or pupil.get("n_immersion") or 1.518),
              # the beads' medium, as a bead model's calibration is drawn in its own; *focus* draws the
              # sample's (`depth.with_medium`)
              "n_med": float(pupil.get("n_medium") or 1.33),
              "pixel_size_nm": pixel}
    return {"pupil": dict(pupil), "optics": optics, "microscope": name, "key": key,
            "version": summary.get("version")}


_LEARNED: dict = {}


def learned(rec: Dict[str, Any]):
    """The uipsf model ``rec`` (`record`) stands for (`uipsf.export.from_description`), kept while the record
    is the same: the form asks with every change, and a focus draws it many times."""
    try:
        from uipsf.export import from_description
        from uipsf.optics import Optics
    except ImportError as error:
        raise ValueError(f"the microscope's PSF is drawn with uipsf, which needs torch ({error}): "
                         f"pip install smappy-smlm[uipsf]") from error
    key = json.dumps(rec, sort_keys=True, default=str)
    if key not in _LEARNED:
        _LEARNED.clear()
        o = dict(rec["optics"])
        o["pixel_size_nm"] = tuple(o["pixel_size_nm"])
        _LEARNED[key] = from_description(rec["pupil"], Optics(**o), roi_px=ROI_PX)
    return _LEARNED[key]


def calibration(rec: Dict[str, Any], em_on: Optional[bool] = None):
    """The `SplineCalibration` of the record's model, sampled as a pupil model's calibration
    (`pupil.spline_calibration`): z the emitter's height above the focus at the coverslip, ``pupil`` the
    record, which *focus* draws again at depth."""
    from types import SimpleNamespace

    from .pupil import spline_calibration
    o = rec["optics"]
    camera = SimpleNamespace(is_em=bool(em_on), pixelsize_um=[o["pixel_size_nm"][0] / 1000,
                                                         o["pixel_size_nm"][1] / 1000])
    cal = spline_calibration(learned(rec), DZ_NM, Z_RANGE_NM, ROI_PX, camera=camera)
    cal.parameters.update(pupil=rec, source=f"{rec['microscope']}'s pupil, {rec['key']}")
    cal.parameters.pop("model_file", None)
    return cal


def path_for(rec: Dict[str, Any], em_on: Optional[bool] = None) -> Path:
    """Where the spline of ``rec`` is kept: the configuration directory, named by the microscope, the key and
    a hash of everything it was drawn from."""
    from ..config import config_dir
    digest = hashlib.sha1(json.dumps([rec, em_on, DZ_NM, Z_RANGE_NM, ROI_PX], sort_keys=True,
                                     default=str).encode()).hexdigest()[:10]
    stem = f"{rec['microscope']} {rec['key']}".replace("/", "-").replace(":", "").replace("·", "-")
    return config_dir() / MICROSCOPE_FOLDER / f"{'_'.join(stem.split())}_{digest}_3dcal.h5"


def _written(rec: Dict[str, Any], em_on: Optional[bool], summary: Optional[Dict[str, Any]] = None) -> Path:
    """The spline of ``rec`` on disk (`path_for`), drawn and written the first time; ``summary`` the
    microscope it is kept with, as a bead calibration keeps its beads' (`io.calibration.keep_microscope`)."""
    from ..io.calibration import keep_microscope, save_spline_calibration
    path = path_for(rec, em_on)
    if not path.is_file():
        cal = calibration(rec, em_on)
        if summary:
            from types import SimpleNamespace
            keep_microscope(cal, SimpleNamespace(microscope=summary))
        path.parent.mkdir(parents=True, exist_ok=True)
        save_spline_calibration(path, cal, overwrite=True)
    return path


def calibration_path(summary: Optional[Dict[str, Any]], em_on: Optional[bool] = None,
                     wavelength_nm: Optional[float] = None) -> Path:
    """The microscope's PSF for a file of ``summary`` as a calibration file, drawn and written the first time;
    what Spline 3D's *calibration* on *microscope* fits with."""
    return _written(record(summary, wavelength_nm), em_on, summary)


# ------------------------------------------------------------------ for a simulation, which has no file

def choices() -> list:
    """``[(value, label)]``, every pupil kept with a microscope: what *Simulate*'s *PSF calibration* offers,
    where no file names a microscope.  The value is ``microscope:<name>:<key>`` (`simulation_path`)."""
    from ..microscope_db import database
    out = []
    for name in database().names():
        entry = database().get(name)
        for key, pupil in entry.pupils.items():
            if pupil.get("kind"):
                out.append((f"{MICROSCOPE}:{name}:{key}",
                            f"{name}'s pupil, {key} (from {pupil.get('source') or 'beads'})"))
    return out


def simulation_path(value: str, pixel_nm: float, em_on: Optional[bool] = False) -> Path:
    """The spline of a pupil `choices` offers, drawn for a simulation: its own emission, the entry's
    objective, the simulation's square pixel -- the frames are drawn and fitted in it.  A ValueError for a
    value that names no kept pupil."""
    from ..microscope_db import database
    parts = str(value).split(":", 2)
    if len(parts) != 3 or parts[0] != MICROSCOPE:
        raise ValueError(f"{value!r} names no microscope's pupil: microscope:<name>:<key>")
    _, name, key = parts
    entry = database().get(name)
    pupil = entry.pupils.get(key) if entry is not None else None
    if not pupil or not pupil.get("kind"):
        raise ValueError(f"{name} keeps no pupil {key!r}")
    objective = entry.objective
    rec = {"pupil": dict(pupil),
           "optics": {"NA": float(objective.get("NA") or pupil["NA"]),
                      "wavelength_nm": float(pupil["wavelength_nm"]),
                      "n_imm": float(objective.get("n_immersion") or pupil.get("n_immersion") or 1.518),
                      "n_med": float(pupil.get("n_medium") or 1.33),
                      "pixel_size_nm": [float(pixel_nm), float(pixel_nm)]},
           "microscope": name, "key": key, "version": None}
    return _written(rec, em_on)
