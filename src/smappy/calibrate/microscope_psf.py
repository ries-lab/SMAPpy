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

**Without a kept pupil: the theory of the optics.**  A microscope that keeps no pupil is drawn from what its
entry says (`theory_pupil`): a minimal pupil -- cos θ^γ, the aperture, no aberration but the 3D lens's
astigmatism (Noll 6: the entry's ``astigmatism_nm``, else any earlier model of the same microscope, camera and
3D lens state, `earlier_astigmatism`) -- at the entry's NA, γ from the entry or the pupils kept under
other states (`microscope_db.gamma_of`), else a nominal one, and a 20 nm blur, the bead model's start.
It is a start, not a calibration: the fit says so (`Notice`, a warning at Run), and *learn* learns the
data's own aberrations from it (docs/psf-model-parts-plan.md: *PSF model* = *microscope*, "its kept pupil,
else the theory of its optics").

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
#: γ where neither the entry nor a kept pupil gives one: uipsf's `model.build.NOMINAL_GAMMA` (M5's beads),
#: copied so that the form, which asks with every change, never loads torch (a test keeps the two equal)
NOMINAL_GAMMA = 0.36
#: the theory's blur, nm: the bead model's start (`pupil.BeadSettings`' *blur σ*)
THEORY_BLUR_NM = 20.0


def _lens_state(entry, states: Dict[str, Any]) -> Optional[str]:
    """The state of the entry's 3D lens in ``states``, or None (no lens, or not said)."""
    device = entry.lens_device() if entry is not None else None
    return None if device is None else (states or {}).get(device)


_ZERNIKES: dict = {}


def _calibration_noll6(path: str) -> Optional[tuple]:
    """(Noll 6 in rad, the emission it was learnt at or None) a pupil calibration records in its parameters
    (`pupil._record`'s ``zernikes``), kept per file and date: the whole spline is read for it, and the form
    asks with every change."""
    p = Path(path)
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    if key not in _ZERNIKES:
        found = None
        try:
            from ..io.calibration import load_spline_calibration
            from ..microscope_db import emission_of
            params = load_spline_calibration(p).parameters or {}
            phase = ((params.get("zernikes") or [{}])[0] or {}).get("phase") or []
            value = next((float(v) for j, v, *_ in phase if int(j) == 6), None)
            if value is not None:
                found = (value, emission_of(params.get("microscope")))
        except Exception:
            found = None
        _ZERNIKES[key] = found
    return _ZERNIKES[key]


def earlier_astigmatism(summary: Dict[str, Any], wavelength_nm: Optional[float]) -> tuple:
    """(the vertical astigmatism in nm of wavefront, where it comes from), or (None, why): what a theory pupil
    starts from where the entry gives no ``astigmatism_nm``.  The 3D lens puts the same astigmatism into
    every model taken with it, whatever the colour or day, so any earlier model is a good start: a pupil kept
    with the microscope for the same state of its 3D lens and camera, the nearest in colour; else the newest
    pupil (Zernike) calibration listed for that microscope, state and camera (`calibration_list`).  Only Noll
    6 (cos 2θ) is taken: the EM register's mirror in x leaves it as it is, but a camera mounted turned flips
    its sign, so the camera must be the same (M2's Evolve against its iXon, `Microscope.pupil_for`)."""
    from ..microscope_db import database, same_camera
    name = str(summary.get("name") or "")
    value = (summary.get("values") or {}).get("astigmatism_nm")
    if value is not None:
        return float(value), "the entry's for its 3D lens"
    entry = database().get(name)
    states = summary.get("states") or {}
    state = _lens_state(entry, states)
    camera = str(summary.get("camera") or (entry.camera if entry is not None else "") or "")
    if entry is not None:
        prefix = entry.pupil_key(states)
        kept = [(k, p) for k, p in entry.pupils.items()
                if k.split(" \u00b7 ")[0] == prefix and "6" in (p.get("phase_nm") or {})
                and (not camera or not p.get("camera") or same_camera(camera, str(p["camera"])))]
        if kept:
            if wavelength_nm is not None:
                kept.sort(key=lambda kp: abs(float(kp[1].get("wavelength_nm") or wavelength_nm) - wavelength_nm))
            key, pupil = kept[0]
            return float(pupil["phase_nm"]["6"]), f"the pupil {name} keeps for {key}"
    from .. import calibration_list
    listed = []
    for item in calibration_list.entries():
        scope = item.microscope or {}
        if item.dual or str(scope.get("name") or "") != name:
            continue
        if _lens_state(entry, scope.get("states") or {}) != state:
            continue
        if camera and scope.get("camera") and not same_camera(camera, str(scope["camera"])):
            continue
        listed.append(item)
    listed.sort(key=lambda e: e.made, reverse=True)
    for item in listed:
        found = _calibration_noll6(item.path)
        if found is None:
            continue
        rad, emission = found
        at = emission or wavelength_nm
        if at is None:
            continue
        return rad * float(at) / (2 * np.pi), f"{item.name} ({item.made[:10] or 'undated'})"
    return None, (f"none in {name}'s entry, its kept pupils or the calibrations listed for "
                  f"{'this state of its 3D lens' if state is not None else 'it'}")


def theory_pupil(summary: Dict[str, Any], wavelength_nm: Optional[float]) -> Dict[str, Any]:
    """A pupil description (`uipsf.VectorPSF.describe`'s form) of what ``summary``'s entry says of its
    optics, for a microscope that keeps no pupil: a minimal pupil at the entry's NA, its 3D lens's
    astigmatism the only aberration, γ the microscope's (`microscope_db.gamma_of`) or `NOMINAL_GAMMA`.  It
    records what it was made of in ``theory``, every value's source, which the fit says.  A ValueError
    where the entry gives no NA."""
    from ..microscope_db import gamma_of
    name = str(summary.get("name"))
    values = summary.get("values") or {}
    if values.get("NA") is None:
        raise ValueError(f"{name} keeps no pupil and its entry gives no NA: add the objective's NA to the "
                         f"microscope, or choose a bead calibration")
    gamma, na, where = gamma_of(summary, wavelength_nm)
    made = []
    if gamma is None:
        gamma, where = NOMINAL_GAMMA, "nominal (M5's beads)"
    made.append(f"γ {gamma:g}, {where}")
    astigmatism, from_where = earlier_astigmatism(summary, wavelength_nm)
    phase = {}
    if astigmatism is not None:
        phase["6"] = round(float(astigmatism), 3)
        made.append(f"astigmatism {float(astigmatism):.0f} nm, from {from_where}")
    else:
        made.append(f"no astigmatism ({from_where}), so z's sign is not known")
    out = {"kind": "minimal", "phase_nm": phase, "gamma": float(gamma), "sigma_nm": [THEORY_BLUR_NM] * 2,
           "pixel_blur": True, "wavelength_nm": float(wavelength_nm) if wavelength_nm else None,
           "NA": float(values["NA"]), "n_immersion": float(values.get("n_immersion") or 1.518),
           "n_medium": 1.33, "source": "theory", "theory": made}
    if na is not None:
        out["na"] = float(na)                  # learnt with that γ: the two trade, so one goes with the other
    return out


def is_theory(cal) -> bool:
    """Whether a calibration (loaded) is a microscope's optics drawn in theory, not beads."""
    rec = (getattr(cal, "parameters", None) or {}).get("pupil") or {}
    return bool(rec.get("theory"))


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
    values = summary.get("values") or {}
    theory = pupil is None
    if theory:
        pupil = theory_pupil(summary, wavelength_nm or emission_of(summary))
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
    out = {"pupil": dict(pupil), "optics": optics, "microscope": name, "key": key,
           "version": summary.get("version")}
    if theory:
        out["theory"] = list(pupil["theory"])
    return out


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
    what = "optics in theory" if rec.get("theory") else "pupil"
    cal.parameters.update(pupil=rec, source=f"{rec['microscope']}'s {what}, {rec['key']}")
    cal.parameters.pop("model_file", None)
    return cal


def path_for(rec: Dict[str, Any], em_on: Optional[bool] = None) -> Path:
    """Where the spline of ``rec`` is kept: the configuration directory, named by the microscope, the key and
    a hash of everything it was drawn from."""
    from ..config import config_dir
    digest = hashlib.sha1(json.dumps([rec, em_on, DZ_NM, Z_RANGE_NM, ROI_PX], sort_keys=True,
                                     default=str).encode()).hexdigest()[:10]
    stem = f"{rec['microscope']} {rec['key']}{' theory' if rec.get('theory') else ''}".replace("/", "-").replace(":", "").replace("·", "-")
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
