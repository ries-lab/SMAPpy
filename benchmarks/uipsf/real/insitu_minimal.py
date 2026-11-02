"""In situ on the minimal model, as the window runs it: γ held from a minimal bead model, no SAF damping, the
phase above the order held, σ round or free (November 2026, roadmap item 9).

    python benchmarks/uipsf/real/insitu_minimal.py beads BEAD_ACQUISITION --model M.uipsf.h5 [--region X Y W H] \\
        [--learn-na] [--blur fixed --blur-nm 0]
    python benchmarks/uipsf/real/insitu_minimal.py learn --rois R.pkl --model M.uipsf.h5 --blur round|free \\
        --start bead|cold [--max-haze 1.5] [--out results/insitu_minimal.jsonl]
    python benchmarks/uipsf/real/insitu_minimal.py score --rois R.pkl --runs TAG [TAG ...]

The question (the user's, November 2026): the PSF model is round in σ everywhere by default; in situ on M2 at
15 px a round σ had forced 0.2–0.3 rad too much astigmatism, at 25 px with σ free it found the beads'
(`docs/uipsf/architecture.md`, "The region size, the blur and the background").  Does a round σ hold up at
25 px on the minimal model, with γ held?

Each *learn* is `smappy.calibrate.insitu_run.run_insitu` on molecules found once (``R.pkl``, written by
`insitu_magnitude.py detect`), as the window would run it: the minimal pupil, order 6 (*auto*), γ and the
phase above order 6 held from the bead model.  *start* ``bead`` starts every learnt mode and σ from the beads
(*start from*); ``cold`` starts from their vertical astigmatism alone, γ and the high orders held the same
(the microscope's kept pupil, ``pupil=``).  *score* localises the molecules held out at detection, which no
run learnt from, on each run's model from the same heights: Δnats per molecule against the best.

Optics are M2's by default (NA 1.43, oil 1.516, sample 1.33); pass the pixel and emission of each pair.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from insitu_magnitude import _np, deviances, drop_hazy, emit, summary  # noqa: E402

NA, N_IMM, N_MED, WAVELENGTH_NM, PIXEL = 1.43, 1.516, 1.33, 680.0, (127.0, 116.0)
BEAD_ORDER, BEAD_NM = 8, 100.0


def optics():
    from smappy.calibrate.pupil import OpticsSettings
    return OpticsSettings(NA=NA, wavelength_nm=WAVELENGTH_NM, n_immersion=N_IMM, n_medium=N_MED,
                          pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])


def describe(path) -> dict:
    """The bead model's hand-over description (`VectorPSF.describe`)."""
    from smappy.calibrate.pupil import load_start
    previous = load_start(path)
    return previous.model.channels[0].psf.describe(previous.shared)


def phase_rad(description, j) -> float:
    return 2 * math.pi * float((description.get("phase_nm") or {}).get(str(j), 0.0)) / WAVELENGTH_NM


def beads(args):
    """A minimal bead model as the bead calibration's defaults learn it, at order 8 so that in situ can hold
    its modes above 6, round σ, linear drift."""
    from smappy.calibrate.core import CalibrationSettings
    from smappy.calibrate.pupil import BeadSettings, PSFModelSettings, RunSettings, run_calibration

    import uipsf
    files = sorted(glob.glob(os.path.join(args.acquisition, "*", "*.ome.tif")))
    t0 = time.perf_counter()
    model = BeadSettings(psf=PSFModelSettings(zernike_order=BEAD_ORDER, blur=args.blur, blur_nm=args.blur_nm,
                                              learn_na=args.learn_na), bead_diameter_nm=BEAD_NM)
    detection = CalibrationSettings(region=tuple(args.region)) if args.region else None
    run = run_calibration(files, False, optics(), model, detection=detection, run=RunSettings(save_model=False))
    cal = run.cal
    uipsf.save_model(cal, args.model)
    d = describe(args.model)
    emit(dict(step="beads", acquisition=os.path.basename(os.path.normpath(args.acquisition)), blur=args.blur,
              used=len(cal.beads.used), learn_na=args.learn_na, na=d.get("na"),
              objective=round(float(cal.result.objective), 1), gamma=d.get("gamma"), sigma_nm=d.get("sigma_nm"),
              phase_rad={j: round(phase_rad(d, j), 4) for j in (5, 6, 11, 12, 22)},
              minutes=round((time.perf_counter() - t0) / 60, 2)), args.out)


def run_tag(args) -> str:
    tag = f"{args.blur}_{args.start}"
    tag += "" if args.max_haze is None else f"_haze{args.max_haze:g}"
    tag += "" if args.gamma is None else f"_g{args.gamma:g}"
    tag += "" if args.astigmatism is None else f"_a{args.astigmatism:g}"
    tag += "" if args.hold == "start" else f"_hold{args.hold}"
    return tag + (f"_{args.tag}" if args.tag else "")


def run_file(rois, tag) -> str:
    return f"{os.path.splitext(rois)[0]}_minimal_{tag}.pt"


def learn(args):
    from smappy.calibrate.insitu import InSituSettings
    from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu
    from smappy.calibrate.pupil import BeadSettings, PSFModelSettings, RunSettings
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    if args.max_haze is not None:
        drop_hazy(rois, args.max_haze)
    order = 6
    if args.model:
        bead = describe(args.model)
        astig_nm = float(bead["phase_nm"]["6"]) if args.astigmatism is None else args.astigmatism
    else:
        # no beads (Zenodo tubulin): the astigmatism typed, γ the nominal one `run_insitu` holds without a pupil
        bead, astig_nm = {"phase_nm": {}}, args.astigmatism
    if not args.model:
        run, pupil = RunSettings(save_model=False), {}
    elif args.start == "bead" and args.gamma is None:
        run, pupil = RunSettings(start_from=args.model, save_model=False), None
    else:
        # a cold start: γ and the phase above the order held as from the beads, the rest from the astigmatism
        from uipsf.optics.zernike import noll_to_nm
        high = {j: v for j, v in bead["phase_nm"].items() if noll_to_nm(int(j))[0] > order}
        pupil = dict(kind="minimal", gamma=bead["gamma"] if args.gamma is None else args.gamma,
                     phase_nm=high if args.start == "cold" else dict(bead["phase_nm"]), wavelength_nm=WAVELENGTH_NM)
        run = RunSettings(save_model=False)
    model = BeadSettings(psf=PSFModelSettings(zernike_order=order, blur=args.blur))
    insitu = InSituModelSettings(focus_nm=args.focus, astigmatism_nm=astig_nm, max_rounds=args.rounds,
                                 coverslip=not args.no_coverslip, hold=args.hold)
    t0 = time.perf_counter()
    result = run_insitu(None, optics(), model, insitu, detection=InSituSettings(), run=run, rois=rois,
                        pupil=pupil, device=args.device)
    cal = result.cal
    p, std = cal.result.params.physical(), cal.result.std
    psf = result.model.psf
    modes = psf.pupil.phase_modes
    phase = dict(zip(modes, _np(p["pupil.phase"]).round(4).tolist(), strict=True))
    crlb = dict(zip(modes, _np(std["pupil.phase"]).round(4).tolist(), strict=True))
    twin = result.run.final.twin
    rounds = [dict(round=r.index, used=len(r.used),
                   focus_nm=round(float(r.calibration.calibration.shared()["geometry.focus"]), 1),
                   sigma_nm=_np(r.calibration.calibration.result.params.physical()["psf.sigma"]).round(1),
                   edge_nm=None if r.edge is None else round(r.edge.shift_nm, 1)) for r in result.run.rounds]
    used = np.asarray(result.run.rounds[-1].used)
    tag = run_tag(args)
    emit(dict(step="learn", rois=os.path.basename(args.rois), tag=tag, blur=args.blur, start=args.start,
              used=len(used), objective_per_unit=round(float(cal.result.objective) / max(1, len(used)), 2),
              minutes=round((time.perf_counter() - t0) / 60, 2),
              focus_nm=round(float(cal.shared()["geometry.focus"]), 1), focus_estimate_nm=result.focus_estimate,
              gamma=psf.pupil.held_gamma if psf.pupil.held_gamma is not None
              else round(float(p["pupil.gamma"]), 4), sigma_nm=_np(p["psf.sigma"]).round(1),
              noll={j: [phase.get(j), crlb.get(j), round(phase_rad(bead, j), 4)] for j in (5, 6, 7, 11, 12)},
              phase=phase, twin_delta_nats=None if twin is None else round(twin.delta_nats, 1), rounds=rounds,
              messages=result.messages), args.out)
    held = dict(phase=dict(psf.pupil.held_phase), gamma=psf.pupil.held_gamma)
    torch.save({"shared": {k: torch.as_tensor(v).detach().cpu() for k, v in cal.shared().items()}, "used": used,
                "blur": args.blur, "order": order, "held": held, "tag": tag}, run_file(args.rois, tag))


def score(args):
    """Every run's model on the molecules held out at detection, each localised from the same heights."""
    from smappy.calibrate.pupil import PSFModelSettings

    from uipsf.model.build import Held
    from uipsf.workflows import insitu_model
    with open(args.rois, "rb") as f:
        rois = pickle.load(f)
    ids = rois.ids("held out")
    ids = ids[:args.limit] if args.limit else ids
    o = optics().optics(rois.camera)[0]
    roi = int(rois.units.regions.shape[-1])
    rows = []
    for tag in args.runs:
        r = torch.load(run_file(args.rois, tag), weights_only=False)
        held = Held(phase={int(j): float(v) for j, v in r["held"]["phase"].items()}, gamma=r["held"]["gamma"])
        spec = PSFModelSettings(zernike_order=r["order"], blur=r["blur"])
        focus = float(r["shared"]["geometry.focus"])
        model = insitu_model(o, spec, (roi, roi), focus, device=args.device, held=held)
        loc, d, pixels = deviances(model, r["shared"], rois, ids)
        rows.append(dict(step="score", rois=os.path.basename(args.rois), tag=tag, molecules=len(ids),
                         focus_nm=round(focus, 1), sigma_nm=_np(r["shared"]["psf.sigma"]).round(1),
                         **summary(loc, d, pixels)))
    best = min(row["deviance"] for row in rows)
    for row in rows:
        row["dnats_per_molecule"] = round((row["deviance"] - best) / 2 / row["molecules"], 3)
        emit(row, args.out)


def main():
    global WAVELENGTH_NM, PIXEL, N_MED, NA, N_IMM
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    b = sub.add_parser("beads")
    b.add_argument("acquisition")
    b.add_argument("--model", required=True)
    b.add_argument("--blur", choices=("round", "free", "fixed"), default="round")
    b.add_argument("--blur-nm", type=float, default=20.0, help="the blur's start, or its value when fixed (0: the "
                   "pixel's blur alone)")
    b.add_argument("--learn-na", action="store_true", help="learn the aperture's NA too")
    b.add_argument("--region", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    le = sub.add_parser("learn")
    le.add_argument("--rois", required=True)
    le.add_argument("--model", default="", help="the minimal bead model: γ, the high orders, the start; "
                    "none: the nominal γ and --astigmatism")
    le.add_argument("--astigmatism", type=float, default=None, help="the start's Noll 6, nm (default the beads')")
    le.add_argument("--no-coverslip", action="store_true", help="no molecules on the coverslip")
    le.add_argument("--blur", choices=("round", "free"), default="round")
    le.add_argument("--start", choices=("bead", "cold"), default="bead")
    le.add_argument("--gamma", type=float, default=None, help="hold this γ instead of the beads'")
    le.add_argument("--hold", choices=("start", "nothing"), default="start",
                    help="nothing: γ learnt from the molecules (the focus then from the coverslip)")
    le.add_argument("--focus", type=float, default=None, help="nm; default: from the data")
    le.add_argument("--rounds", type=int, default=4)
    le.add_argument("--max-haze", type=float, default=None)
    le.add_argument("--tag", default="")
    sc = sub.add_parser("score")
    sc.add_argument("--rois", required=True)
    sc.add_argument("--runs", nargs="+", required=True)
    sc.add_argument("--limit", type=int, default=None)
    for p in (b, le, sc):
        p.add_argument("--wavelength", type=float, default=WAVELENGTH_NM)
        p.add_argument("--pixel", type=float, nargs=2, default=PIXEL, metavar=("X", "Y"))
        p.add_argument("--n-medium", type=float, default=N_MED)
        p.add_argument("--n-immersion", type=float, default=N_IMM)
        p.add_argument("--na", type=float, default=NA)
        p.add_argument("--device", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--memory", type=float, default=0.3, help="MPS memory fraction (the Mac is shared)")
    args = ap.parse_args()
    WAVELENGTH_NM, PIXEL, N_MED = args.wavelength, tuple(args.pixel), args.n_medium
    NA, N_IMM = args.na, args.n_immersion
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    {"beads": beads, "learn": learn, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
