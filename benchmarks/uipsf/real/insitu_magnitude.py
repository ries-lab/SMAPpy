"""Can the pupil magnitude be calibrated on beads and held fixed in situ?  M2, 2026-08-29: beads and a
TIRF clathrin movie (AF647) of the same day, microscope and camera.

    python benchmarks/uipsf/real/insitu_magnitude.py beads BEAD_ACQUISITION --model M.uipsf.h5
    python benchmarks/uipsf/real/insitu_magnitude.py detect MOVIE_FOLDER --rois R.pkl [--frames 4000]
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
    return [round(math.sqrt(max(s * s - q * q / 12, 0.0)), 2) for s, q in zip(sigma, PIXEL, strict=True)]


# ------------------------------------------------------------------ a magnitude held fixed

def based_pupil(pupil, base):
    """``pupil`` (a `ZernikePupil`) with the fixed magnitude map ``base`` [P, P] added: magnitude
    1 + base + Σ c_j Z_j over its own (free) modes.  Its stage-1 truncation keeps the base."""
    from uipsf.model.pupil import ZernikePupil

    class BasedPupil(ZernikePupil):
        def truncated(self, n_max):
            t = ZernikePupil.truncated(self, n_max)
            t.__class__, t.base = BasedPupil, self.base
            return t

        def __call__(self, p):
            magnitude, phase = ZernikePupil.__call__(self, p)
            return magnitude + self.base, phase

    pupil.__class__, pupil.base = BasedPupil, base
    return pupil


def bead_magnitude(previous):
    """{Noll mode: coefficient} of the bead model's magnitude."""
    pupil = previous.model.channels[0].psf.pupil
    return dict(zip(pupil.magnitude_modes, _np(previous.shared["pupil.magnitude"]).tolist(), strict=True))


KINDS = ("free", "bead", "bead-symmetric", "bead-nonsymmetric")


def held_mode(kind: str, j: int) -> bool:
    """Whether ``kind`` holds the beads' magnitude mode ``j``: all, the m = 0 ones, or the m ≠ 0 ones."""
    from uipsf.optics.zernike import noll_to_nm
    m = noll_to_nm(j)[1]
    return {"free": False, "bead": True, "bead-symmetric": m == 0, "bead-nonsymmetric": m != 0}[kind]


def patch_insitu_model(kind: str, coefficients: dict):
    """`uipsf.workflows.insitu_model` replaced for this process by one whose pupil holds the beads'
    magnitude (``kind`` bead, bead-symmetric); ``free`` leaves it."""
    if kind == "free":
        return
    import uipsf.workflows as wf
    from uipsf.optics.zernike import zernike_basis
    from uipsf.workflows.calibrate import magnitude_modes
    original = wf.insitu_model

    def model(o, settings, roi_shape, focus_nm, **kw):
        from dataclasses import replace
        held = {j: c for j, c in coefficients.items() if held_mode(kind, j)}
        free = [] if kind == "bead" else magnitude_modes(ORDER, "free")
        free = [j for j in free if j not in held]
        m = original(o, replace(settings, magnitude="fixed"), roi_shape, focus_nm, **kw)
        for ch in m.channels:
            pupil = ch.psf.pupil
            grid = pupil.grid
            modes = list(held)
            basis = zernike_basis(grid, modes)
            base = torch.tensordot(torch.tensor([held[j] for j in modes], dtype=basis.dtype, device=basis.device),
                                   basis, dims=([0], [0]))
            pupil.magnitude_modes = free
            pupil.magnitude_basis = zernike_basis(grid, free) if free else basis[:0]
            based_pupil(pupil, base)
        return m

    wf.insitu_model = model


# ------------------------------------------------------------------ steps

def beads(args):
    from smappy.calibrate.pupil import BeadSettings, RunSettings, run_calibration
    files = sorted(glob.glob(os.path.join(args.acquisition, "*", "*.ome.tif")))
    t0 = time.perf_counter()
    model = BeadSettings(zernike_order=ORDER, magnitude=args.magnitude, blur="free", bead_diameter_nm=BEAD_NM)
    from smappy.calibrate.core import CalibrationSettings
    detection = CalibrationSettings(region=tuple(args.region)) if args.region else None
    # the model file alone: a spline needs a z range every bead covers, and beads at heights more than
    # a micrometre apart (the 665LP stacks) leave too little of one
    run = run_calibration(files, False, optics(), model, detection=detection, run=RunSettings(save_model=False))
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
    rois = movie_rois([args.movie], InSituSettings(frames=args.frames, seed=args.seed,
                                                     region=tuple(args.region) if args.region else None),
                      progress=lambda e: print(e.get("message", ""), flush=True))
    with open(args.rois, "wb") as f:
        pickle.dump(rois, f)
    emit(dict(step="detect", movie=os.path.basename(os.path.normpath(args.movie)), frames=args.frames,
              candidates=rois.candidates, counts=rois.counts(), minutes=round((time.perf_counter() - t0) / 60, 2)),
         args.out)


def learn(args):
    from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu
    from smappy.calibrate.pupil import BeadSettings, RunSettings, load_start
    previous = load_start(args.model)
    held = bead_magnitude(previous)
    patch_insitu_model(args.magnitude, held)
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    phase_modes = previous.model.channels[0].psf.pupil.phase_modes
    bead_phase = dict(zip(phase_modes, _np(previous.shared["pupil.phase"]).tolist(), strict=True))
    astig_nm = bead_phase[6] * WAVELENGTH_NM / (2 * math.pi)
    model = BeadSettings(zernike_order=ORDER, magnitude="free", blur="free")
    insitu = InSituModelSettings(focus_nm=args.focus, astigmatism_nm=astig_nm, max_rounds=args.rounds,
                                 coverslip=not args.no_coverslip)
    run = RunSettings(start_from=args.model if args.start == "bead" else "", save_model=False)
    t0 = time.perf_counter()
    result = run_insitu(None, optics(), model, insitu, run=run, rois=rois)
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
              sigma_nm=sigma.round(2), sigma_crlb_nm=_np(std["psf.sigma"]).round(2),
              sigma_optics_nm=sigma_optics(sigma),
              phase=dict(zip(pupil.phase_modes, _np(p["pupil.phase"]).round(4).tolist(), strict=True)),
              phase_crlb=dict(zip(pupil.phase_modes, _np(std["pupil.phase"]).round(4).tolist(), strict=True)),
              magnitude=mag, magnitude_held=None if args.magnitude == "free" else
              {j: round(c, 4) for j, c in held.items() if held_mode(args.magnitude, j)},
              twin_delta_nats=None if twin is None else round(twin.delta_nats, 1), rounds=rounds,
              messages=result.messages), args.out)
    # what `score` needs: the learnt values and the molecules this run ended with
    torch.save({"shared": {k: torch.as_tensor(v).detach().cpu() for k, v in cal.shared().items()},
                "used": np.asarray(result.run.rounds[-1].used), "magnitude_kind": args.magnitude, "start": args.start,
                "model": args.model},
               run_path(args.rois, args.magnitude, args.start, args.model))


def run_path(rois, magnitude, start, model=None):
    """Where a run's values are kept; a bead model other than ``beads.uipsf.h5`` adds its name."""
    stem = "" if model is None else os.path.basename(model).split(".")[0]
    tag = "" if stem in ("", "beads") else f"_{stem}"
    return f"{os.path.splitext(rois)[0]}_{magnitude}_{start}{tag}.pt"


def score(args):
    """Every saved run's model on one set of molecules (*set*), each molecule localised on it from the
    same heights: the deviance says which model the data prefer, Δnats = Δdeviance / 2."""
    from smappy.calibrate.pupil import BeadSettings, load_start

    import uipsf.workflows as wf
    previous = load_start(args.model)
    held = bead_magnitude(previous)
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    runs = [torch.load(run_path(args.rois, *r), weights_only=False) for r in args.runs]
    if args.set == "common":                # the molecules every run ended with
        ids = runs[0]["used"]
        for r in runs[1:]:
            ids = np.intersect1d(ids, r["used"])
    else:
        ids = score_ids(rois, args.set)
    ids = ids[:args.limit] if args.limit else ids
    original = wf.insitu_model
    o = optics().optics(rois.camera)[0]
    roi = int(rois.units.regions.shape[-1])
    settings = BeadSettings(zernike_order=ORDER, magnitude="free", blur="free")
    for r in runs:
        wf.insitu_model = original
        # a run held its own bead model's magnitude
        patch_insitu_model(r["magnitude_kind"], bead_magnitude(load_start(r["model"])) if "model" in r else held)
        focus = float(r["shared"]["geometry.focus"])
        model = wf.insitu_model(o, settings, (roi, roi), focus, device=args.device)
        loc, d, pixels = deviances(model, r["shared"], rois, ids)
        emit(dict(step="score", set=args.set, magnitude_kind=r["magnitude_kind"], start=r["start"], molecules=len(ids),
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
    global WAVELENGTH_NM, PIXEL
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("beads")
    b.add_argument("acquisition")
    b.add_argument("--model", required=True, help="the model file saved (.uipsf.h5)")
    b.add_argument("--halves", type=int, default=0, help="this many half-split pairs: the beads' own spread")
    b.add_argument("--magnitude", choices=("free", "symmetric", "fixed"), default="free",
                   help="the beads' magnitude: symmetric learns only its m = 0 modes")
    d = sub.add_parser("detect")
    d.add_argument("movie")
    d.add_argument("--rois", required=True)
    d.add_argument("--frames", type=int, default=4000)
    d.add_argument("--seed", type=int, default=0)
    le = sub.add_parser("learn")
    le.add_argument("--rois", required=True)
    le.add_argument("--model", required=True)
    le.add_argument("--magnitude", choices=KINDS, default="free")
    le.add_argument("--start", choices=("bead", "cold"), default="bead")
    le.add_argument("--focus", type=float, default=None, help="nm above the coverslip; default: from the data")
    le.add_argument("--rounds", type=int, default=4)
    le.add_argument("--no-coverslip", action="store_true", help="no molecules on the coverslip: no edge")
    sc = sub.add_parser("score")
    sc.add_argument("--rois", required=True)
    sc.add_argument("--model", required=True)
    sc.add_argument("--runs", nargs="+", action="append", metavar="MAGNITUDE START [MODEL]", required=True,
                    help="a run by its magnitude, start and, if not beads.uipsf.h5, its bead model")
    sc.add_argument("--set", choices=("common", "heldout", "detected"), default="common",
                    help="common: those every run used; heldout: held out at detection; detected: round 1's")
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
    for p in (b, d, le, sc, bf):
        p.add_argument("--wavelength", type=float, default=WAVELENGTH_NM, help="emission, nm")
        p.add_argument("--pixel", type=float, nargs=2, default=PIXEL, metavar=("X", "Y"),
                       help="nm on the sample: 127 116 for M2's iXon, 116 127 for its Evolve")
        p.add_argument("--out", default=None)
        p.add_argument("--memory", type=float, default=0.3, help="MPS memory fraction (the Mac is shared)")
    args = ap.parse_args()
    WAVELENGTH_NM, PIXEL = args.wavelength, tuple(args.pixel)
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    {"beads": beads, "detect": detect, "learn": learn, "score": score, "beadfit": beadfit}[args.step](args)


if __name__ == "__main__":
    main()
