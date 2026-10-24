"""Can the pupil magnitude be calibrated on beads and held fixed in situ?  M2, 2026-08-29: beads and a
TIRF clathrin movie (AF647) of the same day, microscope and camera.

    python benchmarks/uipsf/real/insitu_magnitude.py beads BEAD_ACQUISITION --model M.uipsf.h5
    python benchmarks/uipsf/real/insitu_magnitude.py detect MOVIE_FOLDER --rois R.pkl [--molecules 2000]
    python benchmarks/uipsf/real/insitu_magnitude.py learn --rois R.pkl --model M.uipsf.h5 \\
        --magnitude free|bead|bead-symmetric --start bead|cold [--out results/insitu_magnitude.jsonl]
    python benchmarks/uipsf/real/insitu_magnitude.py score --rois R.pkl --model M.uipsf.h5 \
        --runs free bead --runs bead cold ... [--set common|heldout|detected]
    python benchmarks/uipsf/real/insitu_magnitude.py beadfit --rois R.pkl --model M.uipsf.h5 [--focus ...]

The idea (the user's, October 2026): the magnitude is mostly the objective's transmission over the
aperture, so it should not change from day to day, and over the M2 iXon bead days its symmetric modes
barely do (Noll 11, 22, 37 within ±0.01; Noll 4 from −0.04 to −0.14, against σ).  Held at the beads'
value it would take the parameters that trade with the blur σ out of in situ learning.

*magnitude* ``free`` learns every mode in situ, ``bead`` holds the beads' whole magnitude, and
``bead-symmetric`` holds its m = 0 modes and learns the others.  A held magnitude is a fixed map added to
the pupil's (`BasedPupil`), so nothing in the solver changes.  *start* ``bead`` takes the beads' phase
and σ (and the magnitude modes learnt here), ``cold`` only their vertical astigmatism (Noll 6), as a
user would type it, which gives the sign.

The optics are M2's (`lab_beads.py`): NA 1.43, oil 1.516, 127 × 116 nm (the iXon's), 676 nm; the movie's
sample index 1.33.  The beads are learnt at order 8 with magnitude and blur free, 100 nm beads; in situ
at order 8 too, so the beads' magnitude carries over whole.  Prints and appends one JSON line per run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import time

import numpy as np
import torch

NA, N_IMM, N_MED, WAVELENGTH_NM, PIXEL = 1.43, 1.516, 1.33, 676.0, (127.0, 116.0)
#: the background per molecule: "constant", or "planar" (a level and a slope, `uipsf.model.background.Planar`)
BACKGROUND = "constant"
ORDER, BEAD_NM = 8, 100.0


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def optics():
    from smappy.calibrate.pupil import OpticsSettings
    return OpticsSettings(NA=NA, wavelength_nm=WAVELENGTH_NM, n_immersion=N_IMM, n_medium=N_MED,
                          pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])


def _np(t):
    return torch.as_tensor(t).detach().cpu().double().numpy()


def sigma_optics(sigma):
    """σ with the pixel's own width taken out (`lab_beads.py`)."""
    sigma = list(np.ravel(sigma)) * (2 if np.size(sigma) == 1 else 1)        # a round σ is one value
    return [round(math.sqrt(max(s * s - q * q / 12, 0.0)), 2) for s, q in zip(sigma, PIXEL, strict=True)]


# ------------------------------------------------------------------ a magnitude held fixed

def based_pupil(pupil, base, phase_base=0.0):
    """``pupil`` (a `ZernikePupil`) with the fixed maps ``base`` (magnitude) and ``phase_base`` [P, P]
    added: magnitude 1 + base + Σ c_j Z_j and phase phase_base + Σ c_j Z_j over its own (free) modes.
    Its stage-1 truncation keeps both."""
    from uipsf.model.pupil import ZernikePupil

    class BasedPupil(ZernikePupil):
        def truncated(self, n_max):
            t = ZernikePupil.truncated(self, n_max)
            t.__class__, t.base, t.phase_base = BasedPupil, self.base, self.phase_base
            return t

        def __call__(self, p):
            magnitude, phase = ZernikePupil.__call__(self, p)
            return magnitude + self.base, phase + self.phase_base

    pupil.__class__, pupil.base, pupil.phase_base = BasedPupil, base, phase_base
    return pupil


def bead_magnitude(previous):
    """{Noll mode: coefficient} of the bead model's magnitude."""
    pupil = previous.model.channels[0].psf.pupil
    return dict(zip(pupil.magnitude_modes, _np(previous.shared["pupil.magnitude"]).tolist(), strict=True))


KINDS = ("free", "bead", "bead-symmetric", "bead-nonsymmetric", "uniform")


def held_mode(kind: str, j: int) -> bool:
    """Whether ``kind`` holds the beads' magnitude mode ``j``: all, the m = 0 ones, or the m ≠ 0 ones."""
    from uipsf.optics.zernike import noll_to_nm
    m = noll_to_nm(j)[1]
    return {"free": False, "bead": True, "bead-symmetric": m == 0, "bead-nonsymmetric": m != 0,
            "uniform": False}[kind]


def fixed_map(grid, coefficients: dict):
    """Σ c_j Z_j on ``grid`` for {Noll mode: coefficient}; 0 for none."""
    from uipsf.optics.zernike import zernike_basis
    if not coefficients:
        return 0.0
    modes = list(coefficients)
    basis = zernike_basis(grid, modes)
    c = torch.tensor([coefficients[j] for j in modes], dtype=basis.dtype, device=basis.device)
    return torch.tensordot(c, basis, dims=([0], [0]))


def patch_insitu_model(kind: str, coefficients: dict, phase_above: int | None = None, phase: dict | None = None):
    """`uipsf.workflows.insitu_model` replaced for this process by one whose pupil holds the beads'
    magnitude (``kind`` bead, bead-symmetric, bead-nonsymmetric; ``free`` holds none) and, with
    ``phase_above``, the beads' phase modes of radial order above it (``phase``, {Noll: rad})."""
    if kind == "free" and phase_above is None and BACKGROUND == "constant":
        return
    import uipsf.workflows as wf
    from uipsf.optics.zernike import noll_to_nm, zernike_basis
    from uipsf.workflows.calibrate import magnitude_modes
    original = wf.insitu_model

    def model(o, settings, roi_shape, focus_nm, **kw):
        from dataclasses import replace
        held = {j: c for j, c in coefficients.items() if held_mode(kind, j)}
        # uniform: no magnitude modes at all, none held (v1's in situ, const_pupilmag)
        free = [] if kind in ("bead", "uniform") else magnitude_modes(ORDER, "free")
        free = [j for j in free if j not in held]
        m = original(o, replace(settings, magnitude="fixed"), roi_shape, focus_nm, **kw)
        for ch in m.channels:
            pupil = ch.psf.pupil
            grid = pupil.grid
            pupil.magnitude_modes = free
            pupil.magnitude_basis = zernike_basis(grid, free) if free else zernike_basis(grid, [2])[:0]
            phase_base = 0.0
            if phase_above is not None:
                # the beads' high orders as a fixed map; only the low ones are learnt
                fixed = {j: c for j, c in phase.items() if noll_to_nm(j)[0] > phase_above}
                keep = [i for i, j in enumerate(pupil.phase_modes) if noll_to_nm(j)[0] <= phase_above]
                pupil.phase_modes = [pupil.phase_modes[i] for i in keep]
                pupil.phase_basis = pupil.phase_basis[keep]
                phase_base = fixed_map(grid, fixed)
            based_pupil(pupil, fixed_map(grid, held), phase_base)
        if BACKGROUND == "planar":
            from uipsf.model.background import Planar
            m.background = Planar(o.pixel_size_nm)
        return m

    wf.insitu_model = model


# ------------------------------------------------------------------ steps

def camera_of(args):
    """The camera given (--camera conversion offset), for files the database does not know; else None."""
    if args.camera is None:
        return None
    from smappy.metadata import CameraMetadata
    return CameraMetadata(conversion=args.camera[0], offset=args.camera[1], read_noise_e=1.6,
                          pixelsize_um=[PIXEL[0] / 1000, PIXEL[1] / 1000])


def beads(args):
    from smappy.calibrate.pupil import BeadSettings, RunSettings, run_calibration
    files = sorted(glob.glob(os.path.join(args.acquisition, "*", "*.ome.tif")))
    t0 = time.perf_counter()
    model = BeadSettings(zernike_order=ORDER, magnitude=args.magnitude, blur=args.blur,
                         bead_diameter_nm=args.bead_diameter)
    from smappy.calibrate.core import CalibrationSettings
    detection = CalibrationSettings(region=tuple(args.region)) if args.region else None
    # the model file alone: a spline needs a z range every bead covers, and beads at heights more than
    # a micrometre apart (the 665LP stacks) leave too little of one
    run = run_calibration(files, False, optics(), model, detection=detection, run=RunSettings(save_model=False),
                          camera=camera_of(args))
    cal = run.cal
    import uipsf
    uipsf.save_model(cal, args.model)
    p = cal.result.params.physical()
    bead_z = _np(p["emitter.xyz"][:, 2])                    # the used beads only
    pupil = cal.model.psf.pupil
    emit(dict(step="beads", acquisition=os.path.basename(os.path.normpath(args.acquisition)),
              used=len(cal.beads.used), beads=len(run.rois.records), objective=float(cal.result.objective),
              bead_z_nm=np.percentile(bead_z, [0, 25, 50, 75, 100]).round(0),
              minutes=round((time.perf_counter() - t0) / 60, 2), sigma_nm=_np(p["psf.sigma"]).round(2),
              sigma_optics_nm=sigma_optics(_np(p["psf.sigma"])),
              magnitude=dict(zip(pupil.magnitude_modes, _np(p["pupil.magnitude"]).round(4).tolist(), strict=True)),
              phase=dict(zip(pupil.phase_modes, _np(p["pupil.phase"]).round(4).tolist(), strict=True))), args.out)
    if args.halves:
        # the beads' own spread: every in situ difference is judged against it
        from smappy.calibrate.pupil import half_split_uncertainty
        h = half_split_uncertainty(run, optics(), splits=args.halves)
        sd = {k: np.asarray(v, float).reshape(-1).round(4) for k, v in h.std.items()}
        emit(dict(step="halves", acquisition=os.path.basename(os.path.normpath(args.acquisition)),
                  splits=h.n_splits, sigma_sd_nm=sd.get("psf.sigma"),
                  phase_sd=dict(zip(pupil.phase_modes, sd["pupil.phase"].tolist(), strict=True)),
                  magnitude_sd=dict(zip(pupil.magnitude_modes, sd["pupil.magnitude"].tolist(), strict=True))),
             args.out)


def detect(args):
    from smappy.calibrate.insitu import InSituSettings
    from smappy.calibrate.insitu_run import movie_rois
    t0 = time.perf_counter()
    rois = movie_rois([args.movie], InSituSettings(molecules=args.molecules, seed=args.seed, roi_px=args.roi,
                                                     region=tuple(args.region) if args.region else None),
                      camera=camera_of(args), progress=lambda e: print(e.get("message", ""), flush=True))
    with open(args.rois, "wb") as f:
        pickle.dump(rois, f)
    emit(dict(step="detect", movie=os.path.basename(os.path.normpath(args.movie)), molecules=args.molecules,
              candidates=rois.candidates, counts=rois.counts(), minutes=round((time.perf_counter() - t0) / 60, 2)),
         args.out)


def learn(args):
    from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu
    from smappy.calibrate.pupil import BeadSettings, RunSettings, load_start
    previous = load_start(args.model)
    held = bead_magnitude(previous)
    bead_phase = bead_phases(previous)
    patch_insitu_model(args.magnitude, held, args.hold_phase_above, bead_phase)
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    if args.subsets > 1:
        keep_subset(rois, args.subset, args.subsets)
    if args.crop:
        crop(rois, args.crop)
    if args.poisson:
        # plain Poisson, as v1 treats the data: no EM excess noise (`InSituROIs.batch` reads it)
        rois.excess_noise = 1.0
    if args.max_haze is not None:
        drop_hazy(rois, args.max_haze)
    astig_nm = bead_phase[6] * WAVELENGTH_NM / (2 * math.pi)
    # a blur held round at a value (--blur-fixed): does in situ's extra astigmatism trade with σx != σy?
    model = (BeadSettings(zernike_order=ORDER, magnitude="free", blur=args.blur) if args.blur_fixed is None else
             BeadSettings(zernike_order=ORDER, magnitude="free", blur="fixed", blur_nm=args.blur_fixed))
    from dataclasses import replace as replaced
    model = replaced(model, fit_tilt=args.fit_tilt)          # the coverslip's tilt learnt (uipsf 774472e)
    insitu = InSituModelSettings(focus_nm=args.focus, astigmatism_nm=astig_nm, max_rounds=args.rounds,
                                 coverslip=not args.no_coverslip)
    run = RunSettings(start_from=args.model if args.start == "bead" else "", save_model=False)
    t0 = time.perf_counter()
    import smappy.calibrate.insitu_run as insitu_run
    from smappy.calibrate.insitu import InSituSettings
    # this script holds the pupil its own way (`patch_insitu_model`); `run_insitu`'s own holding (its *hold*,
    # the start's) is switched off, with a source named so that the focus stays learnt, as in these runs
    insitu_run.held_pupil = lambda *a, **k: ({}, {}, "this script's held maps")
    if args.height_range is not None:
        # the rounds from the second on choose only molecules at these heights above the coverslip
        import uipsf.workflows.insitu_rounds as rounds
        base = rounds.RoundSettings
        rounds.RoundSettings = lambda **k: base(**{"height_range_nm": tuple(args.height_range), **k})
    if args.z_range is not None:
        # v1's window: only molecules within this of the height in focus are chosen, from round 2
        import uipsf.workflows.insitu_rounds as rounds
        base_rounds = rounds.RoundSettings
        rounds.RoundSettings = lambda **k: base_rounds(**{"z_range_nm": args.z_range, **k})
    if args.reject_k is not None:
        # a stronger outlier rejection at learning (v1 drops the worst 10 % by likelihood, more by MSE)
        import uipsf.workflows.insitu_rounds as rounds
        from uipsf.workflows.rejection import RejectionSettings
        base_learn = rounds.learn_insitu
        rounds.learn_insitu = lambda *a, **k: base_learn(
            *a, **{"rejection": RejectionSettings(k=args.reject_k, max_masked=1.0), **k})
    # the rounds choose by the detection's settings: a stricter shape test after the first round
    selection = InSituSettings() if args.shape_k is None else InSituSettings(shape_k=args.shape_k)
    result = run_insitu(None, optics(), model, insitu, detection=selection, run=run, rois=rois)
    minutes = (time.perf_counter() - t0) / 60
    cal = result.cal
    p, std = cal.result.params.physical(), cal.result.std
    pupil = result.model.channels[0].psf.pupil
    mag = (dict(zip(pupil.magnitude_modes, _np(p["pupil.magnitude"]).round(4).tolist(), strict=True))
           if pupil.magnitude_modes else {})
    sigma = _np(p["psf.sigma"])
    twin = result.run.final.twin
    rounds = [dict(round=r.index, used=len(r.used), objective=float(r.calibration.calibration.result.objective),
                   sigma_nm=_np(r.calibration.calibration.result.params.physical()["psf.sigma"]).round(1),
                   focus_nm=round(float(r.calibration.calibration.shared()["geometry.focus"]), 1),
                   edge_nm=None if r.edge is None else round(r.edge.shift_nm, 1)) for r in result.run.rounds]
    emit(dict(step="learn", magnitude_kind=args.magnitude, start=args.start, rois=os.path.basename(args.rois),
              used=len(result.run.rounds[-1].used), objective=float(cal.result.objective),
              objective_per_unit=float(cal.result.objective) / max(1, len(result.run.rounds[-1].used)),
              minutes=round(minutes, 2), focus_nm=round(float(cal.shared()["geometry.focus"]), 1),
              focus_estimate_nm=result.focus_estimate,
              sigma_nm=sigma.round(2), sigma_crlb_nm=_np(std["psf.sigma"]).round(2) if "psf.sigma" in std else None,
              sigma_optics_nm=sigma_optics(sigma),
              phase=dict(zip(pupil.phase_modes, _np(p["pupil.phase"]).round(4).tolist(), strict=True)),
              phase_crlb=dict(zip(pupil.phase_modes, _np(std["pupil.phase"]).round(4).tolist(), strict=True)),
              magnitude=mag, magnitude_held=None if args.magnitude == "free" else
              {j: round(c, 4) for j, c in held.items() if held_mode(args.magnitude, j)},
              twin_delta_nats=None if twin is None else round(twin.delta_nats, 1), rounds=rounds,
              tilt_deg=tilt_of(cal),
              messages=result.messages), args.out)
    # what `score` needs: the learnt values and the molecules this run ended with
    torch.save({"shared": {k: torch.as_tensor(v).detach().cpu() for k, v in cal.shared().items()},
                "used": np.asarray(result.run.rounds[-1].used), "magnitude_kind": args.magnitude, "start": args.start,
                "model": args.model, "order": ORDER, "hold_phase_above": args.hold_phase_above,
                "subset": [args.subset, args.subsets], "max_haze": args.max_haze, "shape_k": args.shape_k,
                "background": BACKGROUND, "n_medium": N_MED, "crop": args.crop,
                "blur": args.blur if args.blur_fixed is None else "fixed", "blur_nm": args.blur_fixed},
               run_path(args.rois, args.magnitude, args.start, args.model, run_tag(args)))


def tilt_of(cal):
    """The learnt coverslip tilt, [[t_x, CRLB], [t_y, CRLB]] in degrees, or None."""
    from uipsf.diagnostics import coverslip_tilt
    tilt = coverslip_tilt(cal)
    return None if tilt is None else [[round(float(v), 3), round(float(e), 3)] for v, e in zip(*tilt, strict=True)]


def run_tag(args) -> str:
    """The order, held phase, subset and filters of a run, in its file name ('' for the defaults)."""
    tag = "" if ORDER == 8 else f"_o{ORDER}"
    tag += "" if args.max_haze is None else f"_haze{args.max_haze:g}"
    tag += "" if args.shape_k is None else f"_shape{args.shape_k:g}"
    tag += "" if BACKGROUND == "constant" else f"_{BACKGROUND}"
    tag += "" if args.height_range is None else f"_hr{args.height_range[0]:g}-{args.height_range[1]:g}"
    tag += "" if N_MED == 1.33 else f"_n{N_MED:g}"
    tag += "" if args.blur_fixed is None else f"_sigma{args.blur_fixed:g}"
    tag += "_tilt" if args.fit_tilt else ""
    tag += "_poisson" if args.poisson else ""
    tag += "_round" if args.blur == "round" else ""
    tag += f"_crop{args.crop}" if args.crop else ""
    tag += "" if args.z_range is None else f"_zr{args.z_range:g}"
    tag += "" if args.reject_k is None else f"_rk{args.reject_k:g}"
    tag += "" if args.hold_phase_above is None else f"_h{args.hold_phase_above}"
    return tag + ("" if args.subsets <= 1 else f"_s{args.subset}of{args.subsets}")


def bead_phases(previous) -> dict:
    """{Noll mode: rad} of the bead model's phase."""
    modes = previous.model.channels[0].psf.pupil.phase_modes
    return dict(zip(modes, _np(previous.shared["pupil.phase"]).tolist(), strict=True))


def crop(rois, n: int):
    """Each region cut to its central ``n`` px: the same molecules, chosen at the detection's region size,
    fitted in a smaller one -- whether a region size acts through the pixels or through the selection."""
    u = rois.units
    k = (u.regions.shape[-1] - n) // 2
    if k < 0 or n % 2 == 0:
        raise ValueError(f"cannot crop {u.regions.shape[-1]} px regions to {n}")
    u.regions, u.masks = u.regions[:, k:k + n, k:k + n], u.masks[:, k:k + n, k:k + n]


def keep_subset(rois, k: int, n: int):
    """Only the molecules of subset ``k`` of ``n`` (by molecule: a blink's frames stay together) may be
    learnt from: the others are marked out, so no round draws them; the held-out ones stay held out."""
    other = (rois.units.blink % n) != k
    rois.units.reason = [r if r is not None or not o else "other subset"
                         for r, o in zip(rois.units.reason, other, strict=True)]
    rois.state = [("rejected: other subset" if o and st != "held out" else st)
                  for st, o in zip(rois.state, other, strict=True)]


def haze(rois) -> np.ndarray:
    """Each unit's background against its noise: the variance of its region's unmasked edge pixels over
    what the noise allows (EM: twice the mean, plus the read noise).  1 for a flat background; free
    imagers' haze, and neighbours at the edge, raise it (DNA-PAINT: median 1.36, against 1.05-1.10)."""
    u = rois.units
    ring = np.zeros(u.regions.shape[1:], bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    with np.errstate(all="ignore"):
        vals = np.where(u.masks[:, ring], u.regions[:, ring], np.nan)
        mean = np.nanmean(vals, 1)
        return np.nanvar(vals, 1) / (rois.excess_noise * np.maximum(mean, 1e-3) + rois.read_noise_var)


def drop_hazy(rois, limit: float):
    """Units whose background scatters more than ``limit`` times the noise are not learnt from."""
    hazy = ~(haze(rois) <= limit)
    rois.units.reason = [r if r is not None or not h else "hazy background"
                         for r, h in zip(rois.units.reason, hazy, strict=True)]
    rois.state = [("rejected: hazy background" if h and st != "held out" else st)
                  for st, h in zip(rois.state, hazy, strict=True)]


def run_path(rois, magnitude, start, model=None, extra=""):
    """Where a run's values are kept; a bead model other than ``beads.uipsf.h5`` adds its name, ``extra``
    the run's order, held phase and subset (`run_tag`)."""
    stem = "" if model is None else os.path.basename(model).split(".")[0]
    tag = "" if stem in ("", "beads") else f"_{stem}"
    return f"{os.path.splitext(rois)[0]}_{magnitude}_{start}{tag}{extra}.pt"


def score(args):
    """Every saved run's model on one set of molecules (*set*), each molecule localised on it from the
    same heights: the deviance says which model the data prefer, Δnats = Δdeviance / 2."""
    global ORDER, N_MED
    from smappy.calibrate.pupil import BeadSettings, load_start

    import uipsf.workflows as wf
    previous = load_start(args.model)
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    # a run given as MAGNITUDE START [MODEL [TAG]]: the tag is `run_tag`'s, after the model
    runs = [torch.load(run_path(args.rois, r[0], r[1], r[2] if len(r) > 2 else None, r[3] if len(r) > 3 else ""),
                       weights_only=False) for r in args.runs]
    if args.set == "common":                # the molecules every run ended with
        ids = runs[0]["used"]
        for r in runs[1:]:
            ids = np.intersect1d(ids, r["used"])
    else:
        ids = score_ids(rois, args.set)
    if args.crop:
        crop(rois, args.crop)
    if args.max_haze is not None:
        ids = ids[haze(rois)[ids] <= args.max_haze]
    ids = ids[:args.limit] if args.limit else ids
    original = wf.insitu_model
    o = optics().optics(rois.camera)[0]
    roi = int(rois.units.regions.shape[-1])
    settings = BeadSettings(zernike_order=ORDER, magnitude="free", blur="free")
    scored_n = N_MED
    for r in runs:
        wf.insitu_model = original
        ORDER = r.get("order", 8)
        # each run with the sample index it learnt with (the score's own for runs from before it was kept)
        N_MED = r.get("n_medium", scored_n)
        o = optics().optics(rois.camera)[0]
        # each run with its own blur: a round one is one value (`VectorPSF.round_sigma`)
        settings = BeadSettings(zernike_order=ORDER, magnitude="free", blur=r.get("blur", "free"),
                                blur_nm=r.get("blur_nm") or 30.0)
        # a run held its own bead model's magnitude, and its high phase orders
        own = load_start(r["model"]) if "model" in r else previous
        patch_insitu_model(r["magnitude_kind"], bead_magnitude(own), r.get("hold_phase_above"), bead_phases(own))
        focus = float(r["shared"]["geometry.focus"])
        # a run that learnt the coverslip's tilt is scored with it
        model = wf.insitu_model(o, settings, (roi, roi), focus, device=args.device,
                                fit_tilt="geometry.tilt" in r["shared"])
        loc, d, pixels = deviances(model, r["shared"], rois, ids)
        emit(dict(step="score", set=args.set, magnitude_kind=r["magnitude_kind"], start=r["start"], molecules=len(ids),
                  model=os.path.basename(r.get("model", args.model)), order=ORDER,
                  hold_phase_above=r.get("hold_phase_above"), subset=r.get("subset"),
                  learnt_max_haze=r.get("max_haze"), learnt_shape_k=r.get("shape_k"), scored_max_haze=args.max_haze,
                  learnt_background=r.get("background", "constant"), scored_background=BACKGROUND,
                  focus_nm=round(focus, 1), sigma_nm=_np(r["shared"]["psf.sigma"]).round(2), **summary(loc, d, pixels)),
             args.out)


def score_ids(rois, which):
    """The molecules a model is scored on: held out at detection (no run learnt from them) or round 1's."""
    return rois.ids("held out" if which == "heldout" else "used")


def deviances(model, shared, rois, ids):
    """(localisations, each molecule's whole deviance, its pixels) of ``ids`` on ``model`` at ``shared``,
    each localised from the same heights."""
    from uipsf.workflows.insitu import start_heights
    from uipsf.workflows.insitu_rounds import localise_units
    dev = model.psf.grid.rho
    shared = {k: torch.as_tensor(v).to(device=dev.device, dtype=dev.dtype) for k, v in shared.items()}
    batch, noise = rois.batch(ids=ids)
    loc = localise_units(model, shared, rois, ids, noise, heights=start_heights(model, float(shared["geometry.focus"])))
    # `score_beads` gives the mean per unmasked pixel: times the pixels, each molecule's whole deviance
    pixels = (torch.ones(batch.data.shape, dtype=torch.bool) if batch.mask is None
              else batch.mask.cpu().expand(batch.data.shape)).flatten(1).sum(1).double().numpy()
    return loc, np.asarray(loc["deviance"], float) * pixels, pixels


def summary(loc, d, pixels):
    return dict(deviance=round(float(np.nansum(d)), 1), median_deviance=round(float(np.nanmedian(d)), 2),
                per_pixel=round(float(np.nansum(d) / pixels[np.isfinite(d)].sum()), 4),
                heights_nm=np.percentile(loc["h"], [5, 25, 50, 75, 95]).round(1))


def beadfit(args):
    """The bead model itself on the molecules (the user's question, October 2026): its pupil and σ as the
    beads learnt them, the bead's size already out of it (in situ models are point-like), only the focus
    scanned -- it is the one thing beads cannot tell.  Scored as `score` scores the in situ runs."""
    from smappy.calibrate.pupil import BeadSettings, load_start

    from uipsf.workflows import insitu_model, start_values
    previous = load_start(args.model)
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    ids = score_ids(rois, args.set)
    ids = ids[:args.limit] if args.limit else ids
    o = optics().optics(rois.camera)[0]
    roi = int(rois.units.regions.shape[-1])
    settings = BeadSettings(zernike_order=ORDER, magnitude="free", blur="free")
    for focus in args.focus:
        model = insitu_model(o, settings, (roi, roi), focus, device=args.device)
        shared = dict(start_values(model, previous), **{"geometry.focus": torch.tensor(float(focus))})
        if args.sigma is not None:
            shared["psf.sigma"] = torch.tensor(args.sigma, dtype=torch.float64)
        loc, d, pixels = deviances(model, shared, rois, ids)
        emit(dict(step="beadfit", set=args.set, molecules=len(ids), focus_nm=float(focus),
                  sigma_nm=_np(shared["psf.sigma"]).round(2), **summary(loc, d, pixels)), args.out)

def main():
    global WAVELENGTH_NM, PIXEL, ORDER, BACKGROUND, N_MED, NA
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("beads")
    b.add_argument("acquisition")
    b.add_argument("--model", required=True, help="the model file saved (.uipsf.h5)")
    b.add_argument("--halves", type=int, default=0, help="this many half-split pairs: the beads' own spread")
    b.add_argument("--bead-diameter", type=float, default=BEAD_NM, help="nm, taken out of the model")
    b.add_argument("--blur", choices=("free", "round"), default="free", help="σ free in x and y, or one value")
    b.add_argument("--magnitude", choices=("free", "symmetric", "fixed"), default="free",
                   help="the beads' magnitude: symmetric learns only its m = 0 modes")
    d = sub.add_parser("detect")
    d.add_argument("movie")
    d.add_argument("--rois", required=True)
    d.add_argument("--molecules", type=int, default=2000)
    d.add_argument("--roi", type=int, default=15, help="each molecule's region, px (v1: 25)")
    d.add_argument("--seed", type=int, default=0)
    le = sub.add_parser("learn")
    le.add_argument("--rois", required=True)
    le.add_argument("--model", required=True)
    le.add_argument("--magnitude", choices=KINDS, default="free")
    le.add_argument("--start", choices=("bead", "cold"), default="bead")
    le.add_argument("--focus", type=float, default=None, help="nm above the coverslip; default: from the data")
    le.add_argument("--rounds", type=int, default=4)
    le.add_argument("--no-coverslip", action="store_true", help="no molecules on the coverslip: no edge")
    le.add_argument("--hold-phase-above", type=int, default=None,
                    help="the beads' phase modes of radial order above this held; only the lower ones learnt")
    le.add_argument("--subset", type=int, default=0, help="learn only from molecule subset k (of --subsets)")
    le.add_argument("--subsets", type=int, default=1)
    le.add_argument("--fit-tilt", action="store_true", help="learn the coverslip's tilt")
    le.add_argument("--poisson", action="store_true", help="plain Poisson noise (no EM excess), as v1")
    le.add_argument("--z-range", type=float, default=None,
                    help="from round 2 learn only from molecules within this of the height in focus, nm (v1: ~1000)")
    le.add_argument("--reject-k", type=float, default=None,
                    help="the learning's outlier rejection, robust sd above the median (default 3)")
    le.add_argument("--crop", type=int, default=None, help="learn on each region's central this many px")
    le.add_argument("--blur", choices=("free", "round"), default="free", help="σ free in x and y, or one value")
    le.add_argument("--blur-fixed", type=float, default=None, help="the blur held round at this σ, nm")
    le.add_argument("--height-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="from round 2 learn only from molecules this high above the coverslip, nm")
    le.add_argument("--max-haze", type=float, default=None,
                    help="learn only from molecules whose background scatters at most this times the noise")
    le.add_argument("--shape-k", type=float, default=None,
                    help="the rounds' shape test, robust sd (default the detection's 3)")
    sc = sub.add_parser("score")
    sc.add_argument("--rois", required=True)
    sc.add_argument("--model", required=True)
    sc.add_argument("--runs", nargs="+", action="append", metavar="MAGNITUDE START [MODEL [TAG]]", required=True,
                    help="a run by its magnitude, start, bead model and tag (`run_tag`: _o6_h4_s0of3 ...)")
    sc.add_argument("--set", choices=("common", "heldout", "detected"), default="common",
                    help="common: those every run used; heldout: held out at detection; detected: round 1's")
    sc.add_argument("--max-haze", type=float, default=None, help="score only molecules this clean (`haze`)")
    sc.add_argument("--crop", type=int, default=None, help="score on each region's central this many px")
    sc.add_argument("--limit", type=int, default=None, help="only the first this many molecules: a quick check")
    sc.add_argument("--device", default=None)
    bf = sub.add_parser("beadfit")
    bf.add_argument("--rois", required=True)
    bf.add_argument("--model", required=True)
    bf.add_argument("--set", choices=("heldout", "detected"), default="heldout")
    bf.add_argument("--focus", type=float, nargs="+", default=[300, 500, 700, 900, 1100, 1300, 1500])
    bf.add_argument("--sigma", type=float, nargs=2, default=None, help="σ x y, nm, in place of the beads'")
    bf.add_argument("--limit", type=int, default=None)
    bf.add_argument("--device", default=None)
    for p in (b, d):
        p.add_argument("--region", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                       help="only beads / molecules inside, in the file's pixels: one channel of a split chip")
        p.add_argument("--camera", type=float, nargs=2, default=None, metavar=("CONVERSION", "OFFSET"),
                       help="e-/ADU and offset, for a camera the database does not know")
    for p in (b, d, le, sc, bf):
        p.add_argument("--wavelength", type=float, default=WAVELENGTH_NM, help="emission, nm")
        p.add_argument("--order", type=int, default=ORDER, help="the Zernike order of the model learnt")
        p.add_argument("--n-medium", type=float, default=N_MED, help="the sample's refractive index")
        p.add_argument("--na", type=float, default=NA, help="the objective's NA")
        p.add_argument("--background", choices=("constant", "planar"), default=BACKGROUND,
                       help="each molecule's background: a level, or a level and a slope (learning and scoring)")
        p.add_argument("--pixel", type=float, nargs=2, default=PIXEL, metavar=("X", "Y"),
                       help="nm on the sample: 127 116 for M2's iXon, 116 127 for its Evolve")
        p.add_argument("--out", default=None)
        p.add_argument("--memory", type=float, default=0.3, help="MPS memory fraction (the Mac is shared)")
    args = ap.parse_args()
    WAVELENGTH_NM, PIXEL, ORDER, BACKGROUND = args.wavelength, tuple(args.pixel), args.order, args.background
    N_MED, NA = args.n_medium, args.na
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    {"beads": beads, "detect": detect, "learn": learn, "score": score, "beadfit": beadfit}[args.step](args)


if __name__ == "__main__":
    main()
