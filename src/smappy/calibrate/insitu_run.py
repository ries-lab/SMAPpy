"""Blinking movies to an in situ PSF calibration: the PSF calibration window's *in situ* method, for a script too.

smappy reads, uipsf learns, as for beads (`pupil`): the movies are read and their molecules found and
chosen here (`insitu`), uipsf learns the pupil in rounds (`uipsf.workflows.insitu_rounds`), and the
result is saved as the bead calibration's are -- a spline the fitter reads, the model file beside it.
What differs from beads (`docs/uipsf/plan.md` §9, M5):

* **Frames, not stacks.**  A movie is thousands of frames; ``frames`` of them are read from each, in
  blocks of ``block_frames`` spread over the whole movie, so that the late, sparse frames count as much
  as the dense first ones.  The background's median runs within a block.
* **The focus** -- where the objective put its focal plane above the coverslip -- is given, or on
  *auto* estimated from the molecules on the coverslip before learning (`focus_from_data`); the rounds
  refine it either way.
* **The start gives the sign** of astigmatism and spherical aberration: a bead model of the microscope
  (*start from*), or the vertical astigmatism typed in, with the microscope's sign.  The twin test
  warns when the data clearly want the other.
* **The spline** is the model sampled at the learnt focus through the heights the molecules were at
  (`insitu_spline`), its z either the height above the coverslip or the height above the plane in
  focus (*spline z*).  It is valid for data imaged at that focus.

torch is imported inside the functions, as in `pupil`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..plugins import param
from .insitu import InSituROIs, InSituSettings, choose, concatenate, insitu_units

SPLINE_Z = (("focus", "above the plane in focus"), ("height", "above the coverslip"))
#: the spline's z step on auto, nm: a molecule's PSF changes over ~100 nm
SPLINE_DZ_NM = 20.0
#: the heights the spline reaches beyond the used molecules', nm
SPLINE_MARGIN_NM = 100.0


@dataclass
class InSituModelSettings:
    """What in situ learning needs beyond the microscope and the model's choices."""
    focus_nm: Optional[float] = param(None, label="focus", unit="nm", min=0.0,
                                      help="the focal plane's height above the coverslip, as the objective was "
                                           "moved from it; auto: estimated from the molecules on the coverslip")
    astigmatism_nm: float = param(0.0, label="astigmatism", unit="nm",
                                  help="the start's vertical astigmatism (Noll 6), with the microscope's sign; "
                                       "not used with start from")
    coverslip: bool = param(True, label="on the coverslip",
                            help="the sample touches the coverslip: its lowest molecules pin the focus")
    max_rounds: int = param(4, label="rounds", min=1, max=8,
                            help="at most this many rounds of localising every molecule, choosing and learning")
    spline_z: str = param("focus", label="spline z", choices=SPLINE_Z,
                          help="what the spline's z is: the height above the plane in focus, or above the "
                               "coverslip")


# ------------------------------------------------------------------ reading the movies

def block_starts(n_frames: int, frames: int, block: int) -> list[int]:
    """The first frames of the blocks read from a movie of ``n_frames``: ⌈frames/block⌉ blocks spread
    evenly over it, all of it when it is shorter."""
    if n_frames <= frames:
        return list(range(0, n_frames, block))
    n = max(1, math.ceil(frames / block))
    return sorted({int(round(s)) for s in np.linspace(0, n_frames - block, n)})


def movie_rois(paths, settings: InSituSettings | None = None, camera=None, progress=None) -> InSituROIs:
    """The molecules of the movies at ``paths``, found per block and chosen over all of them at once.
    ``camera`` defaults to the first movie's, as a fit would resolve it (`pupil.camera_for`)."""
    from ..camera import to_photons
    from ..io.tiff import open_stack
    from .pupil import camera_for, pixel_size_nm
    from .input import discover_acquisitions
    s = settings or InSituSettings()
    # folders to the acquisitions in them, as the window's Stacks page lists them: its movie indices are these
    paths = [str(p) for p in discover_acquisitions(paths)] if paths else []
    if not paths:
        raise ValueError("no movies")
    camera = camera if camera is not None else camera_for(paths[0])
    read_noise = float(camera.readout_variance)
    parts, candidates, sources = [], 0, []
    plan = [(m, path, open_stack(path)) for m, path in enumerate(paths)]
    blocks = [(m, path, src, start) for m, path, src in plan for start in block_starts(src.n_frames, s.frames,
                                                                                       s.block_frames)]
    for k, (m, path, src, start) in enumerate(blocks):
        if progress is not None:
            progress({"message": f"finding molecules: block {k + 1}/{len(blocks)}", "fraction": k / len(blocks)})
        stop = min(start + s.block_frames, src.n_frames)
        frames = np.concatenate([b for _, b in src.frames(chunk=s.block_frames, start=start, stop=stop)])
        u, n = insitu_units(to_photons(frames, camera), s, read_noise)
        parts.append((u, start, m))
        candidates += n
    for m, path, src in plan:
        sources.append({"path": path, "frames": src.n_frames,
                        "read": [b for mm, _, _, b in blocks if mm == m], "block": s.block_frames})
    return choose(concatenate(parts), candidates, pixel_size_nm(camera), s, read_noise,
                  float(camera.excess_noise), sources, camera)


# ------------------------------------------------------------------ the model and its start

def start_shared(model, run_start_from: str, astigmatism_nm: float, wavelength_nm: float):
    """(shared start values, the start model or None): a previous model's (*start from*), else the vertical
    astigmatism typed in, in radians of the emission wavelength."""
    import torch
    if run_start_from:
        from uipsf.workflows import start_values

        from .pupil import load_start
        previous = load_start(run_start_from)
        return start_values(model, previous), previous
    pupil = model.psf.pupil
    phase = torch.zeros(len(pupil.phase_modes), dtype=model.psf.grid.rho.dtype)
    if astigmatism_nm and 6 in pupil.phase_modes:
        phase[pupil.phase_modes.index(6)] = 2 * math.pi * float(astigmatism_nm) / float(wavelength_nm)
    return {"pupil.phase": phase}, None


# ------------------------------------------------------------------ the spline

def used_heights(run) -> tuple[float, float]:
    """(lowest, highest) height the last round learnt from, nm."""
    params = run.final.calibration.result.params.physical()
    h = params["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    return float(np.min(h)), float(np.max(h))


def insitu_spline(model, shared: dict, heights_nm: tuple, dz_nm: float, roi_size: int, z: str = "focus",
                  upsample: Optional[int] = None, camera=None):
    """smappy's `SplineCalibration` of an in situ model: its PSF at the learnt focus, sampled every ``dz_nm``
    through ``heights_nm`` (lowest, highest), normalised as the bead calibrations' (the brightest plane
    sums to 1).  ``z`` "focus": the spline's z is the height above the plane in focus (the sharpest
    plane, z = 0 there), as a bead spline's; "height": the height above the coverslip, the grid then
    reaching down to it, since the spline's z0 must lie on it."""
    from uipsf.export import sample_heights

    from ..io.calibration import SplineCalibration
    from .core import spline_coefficients
    from .pupil import UPSAMPLE
    low, high = float(heights_nm[0]), float(heights_nm[1])
    if z == "height":
        low = min(low, 0.0)
    elif z != "focus":
        raise ValueError(f"spline z {z!r}: focus or height")
    up = upsample if upsample is not None else UPSAMPLE if UPSAMPLE is not None else model.psf.upsample
    n = int(np.ceil((high - low) / dz_nm))
    heights = high - np.arange(n + 1) * dz_nm                 # plane 0 the highest, as in a bead stack
    psf = np.maximum(sample_heights(model, shared, heights, roi_size, up).double().numpy(), 0)
    sharpest = float(heights[int(np.argmax(psf.max(axis=(1, 2))))])
    reference = 0.0 if z == "height" else sharpest
    z0 = (high - reference) / dz_nm                            # z = -(index - z0)·dz = height - reference
    psf /= float(psf.sum(axis=(1, 2)).max())
    focus = float(shared["geometry.focus"])
    parameters = {"method": "uipsf2_insitu", "normalization": "maximum plane sum = 1",
                  "z_convention": "emitter_nm = -(index-z0)*dz",
                  "z_reference": ("the coverslip: z is the height above it" if z == "height"
                                  else "the plane in focus: z is the height above it"),
                  "focus_nm": focus, "in_focus_height_nm": sharpest, "z_offset_nm": reference,
                  "sampling": {"dz_nm": float(dz_nm), "heights_nm": [float(heights[-1]), high],
                               "roi_size": int(roi_size), "upsample": int(up)}}
    psf = psf.astype(np.float32)
    return SplineCalibration(spline_coefficients(psf), float(dz_nm), float(z0), x0=(roi_size - 1) / 2, psf=psf,
                             em_on=None if camera is None else bool(camera.is_em),
                             pixelsize_um=None if camera is None or camera.pixelsize_um is None
                             else np.atleast_1d(np.asarray(camera.pixelsize_um, float)),
                             parameters=parameters)


# ------------------------------------------------------------------ a whole run

@dataclass
class InSituRunResult:
    rois: InSituROIs
    run: object                          # uipsf InSituRun
    model: object                        # the last round's model
    start: object = None                 # the LearnedPSF it started from, or None
    focus_estimate: Optional[float] = None   # the focus from the data, when it was on auto
    calibration: object = None           # the SplineCalibration
    saved: Optional[Path] = None
    messages: list = field(default_factory=list)

    @property
    def cal(self):
        """The last round's uipsf `Calibration`."""
        return self.run.final.calibration


def run_insitu(paths, optics, model, insitu: InSituModelSettings, detection: InSituSettings | None = None,
               spline=None, run=None, camera=None, rois: InSituROIs | None = None, save_to=None, progress=None,
               cancel=None, device=None) -> InSituRunResult:
    """Movies to a learnt in situ model and, with *save model*, its spline on disk (module docstring).
    ``optics`` is an `OpticsSettings`, ``model`` a `BeadSettings`; ``rois``, the molecules found already
    (*Detect*), skips finding them; ``device`` is where the model is learnt (default the fastest)."""
    import torch

    from uipsf.workflows import insitu_model
    from uipsf.workflows.insitu_rounds import (
        RoundSettings,
        calibrate_insitu,
        focus_from_data,
    )

    from .pupil import RunSettings, SplineSettings, default_save_path, save
    detection = detection or InSituSettings()
    spline = spline or SplineSettings()
    run = run or RunSettings()
    tell = progress or (lambda entry: None)
    if rois is None:
        rois = movie_rois(paths, detection, camera, progress)
    if not len(rois.ids("used")):
        raise ValueError("no molecules chosen: review the detection's settings")
    o = optics.optics(rois.camera, 1)[0]
    roi = int(rois.units.regions.shape[-1])
    guess = 1000.0 if insitu.focus_nm is None else float(insitu.focus_nm)
    learnt = insitu_model(o, model, (roi, roi), guess, device=device)
    start, previous = start_shared(learnt, run.start_from, insitu.astigmatism_nm, o.wavelength_nm)
    result_focus = None
    if insitu.focus_nm is None:
        tell({"message": "estimating the focus from the molecules on the coverslip"})
        result_focus, _ = focus_from_data(learnt, rois, start, rois.batch()[1], guess)
        learnt = insitu_model(o, model, (roi, roi), result_focus, device=device)
    start = dict(start, **{"geometry.focus": torch.tensor(learnt.geometry.focus_nm)})
    learned = calibrate_insitu(learnt, rois, start, rois.batch()[1],
                               RoundSettings(max_rounds=insitu.max_rounds, coverslip=insitu.coverslip),
                               selection=detection, progress=progress, cancel=cancel)
    result = InSituRunResult(rois, learned, learned.model or learnt, previous, result_focus)
    cal = learned.final
    if cal.twin is not None:
        result.messages += cal.warnings
    stopped = cal.calibration.result.stop_reason == "cancelled"
    if run.save_model and not stopped:
        result.calibration = calibration_of(result, spline, insitu)
        path = Path(save_to) if save_to else default_save_path(paths, False)
        result.saved, _ = save(path, cal.calibration, result.calibration, overwrite=bool(save_to))
    return result


def calibration_of(result: InSituRunResult, spline=None, insitu: InSituModelSettings | None = None):
    """The run's spline: the saved one, else sampled now as saving would."""
    if result.calibration is None:
        insitu = insitu or InSituModelSettings()
        low, high = used_heights(result.run)
        dz = (spline.dz_nm if spline is not None and spline.dz_nm else None) or SPLINE_DZ_NM
        span = spline.z_range_nm if spline is not None and spline.z_range_nm else None
        cal = result.cal
        model = result.model
        if span:
            sharpest = _sharpest(model, cal.shared(), (low, high), dz)
            low, high = sharpest - span, sharpest + span
        else:
            low, high = low - SPLINE_MARGIN_NM, high + SPLINE_MARGIN_NM
        result.calibration = insitu_spline(model, cal.shared(), (low, high), dz,
                                           int(result.rois.units.regions.shape[-1]), insitu.spline_z,
                                           camera=result.rois.camera)
    return result.calibration


def _sharpest(model, shared, heights, dz) -> float:
    from uipsf.export import sample_heights
    h = np.arange(heights[0], heights[1] + dz, dz)
    psf = sample_heights(model, shared, h, 15).double().numpy()
    return float(h[int(np.argmax(psf.max(axis=(1, 2))))])
