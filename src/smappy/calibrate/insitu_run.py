"""Blinking movies to an in situ PSF calibration: the PSF calibration window's *in situ* method, for a script too.

smappy reads, uipsf learns, as for beads (`pupil`): the movies are read and their molecules found and
chosen here (`insitu`), uipsf learns the pupil in rounds (`uipsf.workflows.insitu_rounds`), and the
result is saved as the bead calibration's are -- a spline the fitter reads, the model file beside it.
What differs from beads (`docs/uipsf/plan.md` §9, M5):

* **Frames, not stacks.**  A movie is thousands of frames, read in blocks of ``block_frames``, coarse to
  fine over the whole movie, until ``molecules`` spots pass the selection (`movie_rois`): how many frames
  that takes depends on the density and the field, which a number of frames did not.  The background's
  median runs within a block.
* **The focus** -- where the objective put its focal plane above the coverslip -- is given, or on
  *auto* estimated from the molecules on the coverslip before learning (`focus_from_data`); the rounds
  refine it either way.
* **The start gives the sign** of astigmatism and spherical aberration: a bead model of the microscope
  (*start from*), or the vertical astigmatism -- the microscope entry's for its 3D lens's state
  (`start_astigmatism`), or typed in, with the microscope's sign.  The twin test
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
from .insitu import InSituROIs, InSituSettings, choose, concatenate, insitu_units, select

SPLINE_Z = (("focus", "above the plane in focus"), ("height", "above the coverslip"))
HOLDS = (("start", "the start's"), ("all", "every aberration"), ("nothing", "nothing"))
FOCUS_FITS = (("learnt", "learnt"), ("coverslip", "from the coverslip"), ("typed", "typed"))
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
    # the start's sign is the data's z: from the microscope's entry for its 3D lens's state, so that it is
    # typed only for a microscope the database does not know (shown then, `psf_window`)
    astigmatism_nm: Optional[float] = param(None, label="astigmatism", unit="nm", advanced=True,
                                            help="the start's vertical astigmatism (Noll 6), with the "
                                                 "microscope's sign; auto: the microscope's, for its 3D lens; "
                                                 "not used with start from")
    hold: str = param("start", label="hold", choices=HOLDS,
                      help="the start's: the symmetric magnitude and the phase above the order held from start "
                           "from or the microscope's pupil; every aberration: the start model's whole pupil, "
                           "only the focus and the blur learnt; nothing: the symmetric magnitude learnt, the "
                           "focus then pinned")
    focus_fit: str = param("learnt", label="focus fit", choices=FOCUS_FITS,
                           help="learnt with the pupil; from the coverslip: set by the lowest molecules' edge "
                                "each round, not fitted; typed: held at focus")
    coverslip: bool = param(True, label="on the coverslip",
                            help="the sample touches the coverslip: its lowest molecules pin the focus")
    max_rounds: int = param(4, label="rounds", min=1, max=8,
                            help="at most this many rounds of localising every molecule, choosing and learning")
    spline_z: str = param("focus", label="spline z", choices=SPLINE_Z,
                          help="what the spline's z is: the height above the plane in focus, or above the "
                               "coverslip")


# ------------------------------------------------------------------ reading the movies

#: blocks read from each movie before the selection is asked whether it has enough: spread over the whole
#: movie, so that a movie dense enough to fill every bin from its first block is still sampled through its
#: whole length -- its bleaching, its drift of focus, the late sparse frames
MIN_BLOCKS = 10
#: reading stops, with z bins still short, once one bin has this many times its share passing: the short ones
#: are short in these data (below the coverslip, the outermost heights), not for want of frames read
SPARSE_FACTOR = 20


def block_order(n_frames: int, block: int) -> list[int]:
    """Every block's first frame in the order they are read: coarse to fine over the whole movie (the van der
    Corput sequence: 0, ½, ¼, ¾, ⅛, ...), so that wherever reading stops, what was read is spread over all
    of it.  A movie shorter than a block is one block."""
    n = max(1, n_frames // block)
    order, seen, i = [], set(), 0
    while len(order) < n:
        v, f, k = 0.0, 0.5, i
        while k:
            v, k, f = v + f * (k & 1), k >> 1, f / 2
        b = int(v * n)
        if b not in seen:
            seen.add(b)
            order.append(b * block)
        i += 1
    return order


def enough(units, settings: InSituSettings) -> tuple[bool, int]:
    """(whether reading can stop, how many pass the selection): every z bin has its share passing, or one
    has `SPARSE_FACTOR` times it (`quotas` then fills from the others)."""
    state, bins, _ = select(units, settings)
    passing = np.array([st in ("used", "held out", "not drawn") for st in state], bool)
    if not passing.any():
        return False, 0
    pools = np.bincount(bins[passing], minlength=settings.bins)
    need = settings.per_bin
    return bool((pools >= need).all() or pools.max() >= SPARSE_FACTOR * need), int(passing.sum())


def movie_rois(paths, settings: InSituSettings | None = None, camera=None, progress=None) -> InSituROIs:
    """The molecules of the movies at ``paths``, found per block and chosen over all of them at once.  Blocks
    are read from all movies in turn, each coarse to fine (`block_order`), at least `MIN_BLOCKS` of each, until
    the selection has enough (`enough`) or the movies are read.  ``camera`` defaults to the first movie's, as a
    fit would resolve it (`pupil.camera_for`)."""
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
    plan = [(m, path, open_stack(path)) for m, path in enumerate(paths)]
    orders = [block_order(src.n_frames, s.block_frames) for _, _, src in plan]
    # the movies in turn, block by block: a short movie runs out and the others go on
    blocks = [(m, orders[m][j]) for j in range(max(map(len, orders))) for m in range(len(plan))
              if j < len(orders[m])]
    first = sum(min(MIN_BLOCKS, len(o)) for o in orders)
    parts, candidates, read = [], 0, {m: [] for m in range(len(plan))}
    passing = 0
    for k, (m, start) in enumerate(blocks):
        if progress is not None:
            progress({"message": f"finding molecules: block {k + 1}, {passing} of {s.molecules} so far",
                      "fraction": min(0.99, max(k / len(blocks), passing / s.molecules))})
        src = plan[m][2]
        stop = min(start + s.block_frames, src.n_frames)
        frames = np.concatenate([b for _, b in src.frames(chunk=s.block_frames, start=start, stop=stop)])
        u, n = insitu_units(to_photons(frames, camera), s, read_noise)
        parts.append((u, start, m))
        read[m].append(start)
        candidates += n
        if k + 1 >= first:
            done, passing = enough(concatenate(parts), s)
            if done:
                break
    sources = [{"path": path, "frames": src.n_frames, "read": sorted(read[m]), "block": s.block_frames}
               for m, path, src in plan]
    rois = choose(concatenate(parts), candidates, pixel_size_nm(camera), s, read_noise,
                  float(camera.excess_noise), sources, camera)
    rois.messages.append(f"{len(parts)} of {len(blocks)} blocks of {s.block_frames} frames read")
    return rois


# ------------------------------------------------------------------ the model and its start

def start_astigmatism(insitu: InSituModelSettings, camera=None) -> tuple[Optional[float], str]:
    """(the start's vertical astigmatism in nm, where it comes from): typed, else the microscope entry's for
    the state its 3D lens was in (``astigmatism_nm``, from bead models of it), else (None, why).  It is what
    sets the sign of z: the twin, every even aberration of the other sign, sees the same molecules mirrored."""
    if insitu.astigmatism_nm is not None:
        return float(insitu.astigmatism_nm), "typed"
    scope = getattr(camera, "microscope", None) or {}
    from ..microscope_db import pupil_of
    kept = (pupil_of(scope) or {}).get("phase_nm") or {}
    if "6" in kept:
        return float(kept["6"]), f"{scope.get('name')}'s objective pupil, for its 3D lens"
    value = (scope.get("values") or {}).get("astigmatism_nm")
    if value is not None:
        return float(value), f"{scope.get('name')}'s, for its 3D lens"
    return None, (f"{scope.get('name')}'s entry gives none" if scope.get("name")
                  else "no microscope recognised")


def start_shared(model, run_start_from: str, astigmatism_nm: float, wavelength_nm: float, pupil=None):
    """(shared start values, the start model or None): a previous model's (*start from*), else the
    objective's pupil kept with the microscope (``pupil``, `microscope_db.pupil_of`: every Zernike of
    phase and magnitude the model has) with the vertical astigmatism over it, in radians of the emission
    wavelength."""
    import torch
    if run_start_from:
        from uipsf.workflows import start_values

        from .pupil import load_start
        previous = load_start(run_start_from)
        return start_values(model, previous), previous
    zernikes = model.psf.pupil
    dtype = model.psf.grid.rho.dtype
    phase = torch.zeros(len(zernikes.phase_modes), dtype=dtype)
    out = {}
    if pupil:
        for i, j in enumerate(zernikes.phase_modes):
            nm = (pupil.get("phase_nm") or {}).get(str(j))
            if nm is not None:
                phase[i] = 2 * math.pi * float(nm) / float(wavelength_nm)
        if len(zernikes.magnitude_modes) and pupil.get("magnitude"):
            magnitude = torch.zeros(len(zernikes.magnitude_modes), dtype=dtype)
            for i, j in enumerate(zernikes.magnitude_modes):
                magnitude[i] = float(pupil["magnitude"].get(str(j), 0.0))
            out["pupil.magnitude"] = magnitude
    if astigmatism_nm and 6 in zernikes.phase_modes:
        phase[zernikes.phase_modes.index(6)] = 2 * math.pi * float(astigmatism_nm) / float(wavelength_nm)
    out["pupil.phase"] = phase
    return out, None


def held_pupil(previous, kept, order: int, wavelength_nm: float, every: bool = False) -> tuple[dict, dict, str]:
    """(phase held {Noll: rad}, magnitude held {Noll: value}, where from) for in situ's *hold* the start's:
    the start model's (``previous``, a `LearnedPSF`) or else the microscope's kept pupil (``kept``,
    `microscope_db.pupil_of`) symmetric magnitude and its phase above ``order``; empty with neither.

    Why these (`docs/uipsf/architecture.md`, "In situ on matched M2 pairs"): the symmetric magnitude is the
    objective's transmission, the same over days and colours, and learnt from molecules it trades with the
    focus; the phase above order 6 gains little over the beads' and scatters between molecule sets.  The
    phase goes through nm of wavefront, so that a pupil learnt at another emission holds at this one.

    ``every``: the whole pupil, every mode of phase and magnitude whatever its order -- *hold* every aberration,
    where the beads' pupil is taken as the objective's and only what belongs to the sample is learnt (the focus,
    the blur, the coverslip's tilt): the beads' aberrations are known far better than a movie can tell them,
    and with nothing of the pupil to learn the sign of z needs no twin test."""
    import math

    from uipsf.optics.zernike import noll_to_nm
    if previous is not None:
        model = previous.model
        pupil = model.channels[0].psf.pupil if hasattr(model, "channels") else model.psf.pupil
        shared = previous.shared() if callable(getattr(previous, "shared", None)) else previous.shared
        if not hasattr(pupil, "coefficients"):
            return {}, {}, ""
        own = float(pupil.grid.optics.wavelength_nm)
        phase_nm = {j: v * own / (2 * math.pi) for j, v in pupil.coefficients(shared, "phase").items()}
        magnitude = pupil.coefficients(shared, "magnitude") if "pupil.magnitude" in shared else {}
        source = "the start model"
    elif kept:
        phase_nm = {int(j): float(v) for j, v in (kept.get("phase_nm") or {}).items()}
        magnitude = {int(j): float(v) for j, v in (kept.get("magnitude") or {}).items()}
        source = "the microscope's pupil"
    else:
        return {}, {}, ""
    phase = {j: 2 * math.pi * nm / float(wavelength_nm) for j, nm in phase_nm.items()
             if every or noll_to_nm(j)[0] > order}
    symmetric = {j: v for j, v in magnitude.items() if (every or noll_to_nm(j)[1] == 0) and j > 1}
    return phase, symmetric, source


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
    if getattr(camera, "microscope", None):
        parameters["microscope"] = dict(camera.microscope)    # the movie's: what the model describes
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
    from dataclasses import replace

    from ..microscope_db import pupil_of
    from .pupil import AUTO_ORDER, load_start
    kept = pupil_of(getattr(rois.camera, "microscope", None))
    every = insitu.hold == "all"
    astigmatism, why = start_astigmatism(insitu, rois.camera)
    if every and not (run.start_from or kept):
        raise ValueError("holding every aberration needs a pupil to hold: a bead model to start from, or one "
                         "kept with the microscope")
    if astigmatism is None and not run.start_from and not every:
        raise ValueError(f"give the start's astigmatism (in situ, more) or a model to start from: {why}")
    messages = []
    # what is held: the start's symmetric magnitude and high orders, or nothing; the m != 0 magnitude is
    # never learnt in situ, where it took the astigmatism over from the phase
    held_phase, held_magnitude, source = {}, {}, ""
    if insitu.hold in ("start", "all"):
        order = model.zernike_order or AUTO_ORDER
        held_phase, held_magnitude, source = held_pupil(load_start(run.start_from) if run.start_from else None,
                                                        kept, order, o.wavelength_nm, every=every)
        if every:
            from uipsf.optics.zernike import noll_to_nm, zernike_modes
            # no phase mode left to learn: the order is the held pupil's own, and a mode up to it that the
            # pupil does not have is held at 0
            top = max([noll_to_nm(j)[0] for j in held_phase] + [2])
            held_phase = {j: held_phase.get(j, 0.0) for j in sorted(set(zernike_modes(top)) | set(held_phase))}
            model = replace(model, zernike_order=top)
        if not source:
            messages.append("nothing to hold: no start model and no pupil kept with the microscope; the "
                            "symmetric magnitude is learnt and the focus pinned")
    model = replace(model, magnitude="fixed" if source else "symmetric")
    focus_fit = insitu.focus_fit
    if not source and focus_fit == "learnt":
        # a learnt symmetric magnitude and a learnt focus trade (140 nm between starts on M2's data)
        focus_fit = "coverslip"
        messages.append("the focus is pinned to the coverslip: with the magnitude learnt it cannot be learnt too")
    if focus_fit == "typed" and insitu.focus_nm is None:
        raise ValueError("a typed focus needs the focus: type it in, or choose the focus from the coverslip")
    edge = insitu.coverslip and focus_fit != "typed"
    if focus_fit == "coverslip" and not insitu.coverslip:
        raise ValueError("a focus from the coverslip needs the sample on the coverslip: type the focus instead")

    def make(focus):
        return insitu_model(o, model, (roi, roi), focus, device=device, held_phase=held_phase,
                            held_magnitude=held_magnitude, fit_focus=focus_fit == "learnt", fit_tilt=model.fit_tilt)
    learnt = make(guess)
    start, previous = start_shared(learnt, run.start_from, astigmatism or 0.0, o.wavelength_nm, kept)
    result_focus = None
    if insitu.focus_nm is None:
        tell({"message": "estimating the focus from the molecules on the coverslip"})
        result_focus, _ = focus_from_data(learnt, rois, start, rois.batch()[1], guess)
        learnt = make(result_focus)
    start = dict(start, **{"geometry.focus": torch.tensor(learnt.geometry.focus_nm)})
    learned = calibrate_insitu(learnt, rois, start, rois.batch()[1],
                               RoundSettings(max_rounds=insitu.max_rounds, coverslip=edge, twin_test=not every),
                               selection=detection, progress=progress, cancel=cancel)
    result = InSituRunResult(rois, learned, learned.model or learnt, previous, result_focus)
    if source and every:
        messages.insert(0, f"held from {source}: every aberration; learnt: the focus and the blur")
    elif source:
        messages.insert(0, f"held from {source}: the symmetric magnitude and the phase above order "
                           f"{model.zernike_order or AUTO_ORDER}")
    if edge and not any(r.focus_prior is not None for r in learned.rounds[1:]) and len(learned.rounds) > 1:
        messages.append("no coverslip edge found: the molecules may lie too deep; the focus is not pinned by "
                        "the coverslip -- type it if it is known")
    result.messages += messages
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
        from uipsf.diagnostics import coverslip_tilt
        tilt = coverslip_tilt(cal)                        # with its CRLB, as a bead model's record keeps it
        if tilt is not None:
            result.calibration.parameters["coverslip_tilt_deg"] = [[float(v), float(e)]
                                                                   for v, e in zip(*tilt, strict=True)]
    return result.calibration


def _sharpest(model, shared, heights, dz) -> float:
    from uipsf.export import sample_heights
    h = np.arange(heights[0], heights[1] + dz, dz)
    psf = sample_heights(model, shared, h, 15).double().numpy()
    return float(h[int(np.argmax(psf.max(axis=(1, 2))))])
