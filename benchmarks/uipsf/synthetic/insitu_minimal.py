"""In situ on the minimal model, on a movie whose PSF is known (November 2026, roadmap item 9).

    python benchmarks/uipsf/synthetic/insitu_minimal.py [--truth-sigma 50 65] [--blur round free] \\
        [--start truth cold] [--gamma held learnt] [--truth-n-med 1.37 --fit-n-med 1.33 1.37] \\
        [--on-coverslip 0.1] [--out benchmarks/uipsf/results/insitu_minimal_sim.jsonl]

Two of the user's questions on the shared minimal model (`model.MinimalPupil`, `model.build`), learnt as the
window learns it -- order 6, no SAF damping, γ held -- and scored against the truth:

- **σ round or free.**  The default blur is round everywhere; on M2's January movie at 15 px a round σ took
  0.3 rad of extra astigmatism, at 25 px σ free found the beads'.  Here the truth's σ is round or not
  (``--truth-sigma``) and each run learns it round or free: the astigmatism, σ, the focus and the held-out
  likelihood against the truth's.
- **Depth, n_med and spherical aberration with γ held.**  The focus and the heights are nearly degenerate;
  the coverslip edge pins their common offset, and γ learnt from molecules traded with the focus.  Here γ is
  held at the truth's or learnt (``--gamma``), and the sample index fitted right or wrong (``--fit-n-med``):
  the focus against the truth, the heights' slope and offset, Noll 11.

The movie: M2's optics (NA 1.43, Evolve pixels 116 × 127 nm, 680 nm, oil 1.516 into ``--truth-n-med``),
January's bead phase (`insitu_region.PHASE`), γ 0.36, molecules 0–1300 nm deep, a tenth on the coverslip,
the focus at 430 nm, ~6000 photons a blink on 55 per pixel, drawn through `VectorPSF.for_emitters` in 41 px.
Molecules are found once (`smappy.calibrate.insitu`) at 25 px and learnt in rounds (`calibrate_insitu`), the
focus started 150 nm high; the twin test runs.  One JSON line per run.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch
from smappy.calibrate import insitu as selection
from smappy.calibrate.pupil import PSFModelSettings
from smappy.simulate import BlinkingSettings

import uipsf
from uipsf.model import Held
from uipsf.movie import MovieSettings, simulate_movie
from uipsf.noise import ShiftedPoisson
from uipsf.workflows import insitu_model
from uipsf.workflows.insitu_rounds import RoundSettings, calibrate_insitu

sys.path.insert(0, os.path.dirname(__file__))
from insitu_nmed import distortion, emit, heights, match  # noqa: E402
from insitu_region import PHASE  # noqa: E402

PIXEL, N_IMM, NA, WAVELENGTH = (116.0, 127.0), 1.516, 1.43, 680.0
ORDER, GAMMA, FOCUS, HEIGHTS = 6, 0.36, 430.0, (0.0, 1300.0)
#: the cold start's astigmatism, rad: the beads' sign, 0.2 rad short
COLD_ASTIGMATISM = -0.5


def optics(n_med: float) -> uipsf.Optics:
    return uipsf.Optics(NA=NA, wavelength_nm=WAVELENGTH, n_imm=N_IMM, n_med=n_med, pixel_size_nm=PIXEL)


def model(n_med, roi, focus, blur, gamma_held=True, device=None):
    spec = PSFModelSettings(zernike_order=ORDER, blur=blur, blur_nm=30.0)
    return insitu_model(optics(n_med), spec, (roi, roi), focus, z_max_nm=1000.0, device=device,
                        held=Held(gamma=GAMMA) if gamma_held else Held())


def truth_shared(m, sigma, focus=FOCUS) -> dict:
    pupil = m.psf.pupil
    dev, dt = pupil.grid.rho.device, pupil.grid.rho.dtype
    return {"pupil.phase": torch.tensor([PHASE.get(j, 0.0) for j in pupil.phase_modes], device=dev, dtype=dt),
            "pupil.magnitude": torch.zeros(0, device=dev, dtype=dt),
            "psf.sigma": torch.tensor(list(sigma), device=dev, dtype=dt),
            "geometry.focus": torch.tensor(focus, device=dev, dtype=dt)}


def start_values(m, start: str, gamma_held: bool) -> dict:
    pupil = m.psf.pupil
    dt = pupil.grid.rho.dtype
    phase = {6: COLD_ASTIGMATISM} if start == "cold" else PHASE
    out = {"pupil.phase": torch.tensor([phase.get(j, 0.0) for j in pupil.phase_modes], dtype=dt)}
    if not gamma_held:
        out["pupil.gamma"] = torch.tensor(0.05 if start == "cold" else GAMMA, dtype=dt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-sigma", type=float, nargs=2, default=[55.0, 55.0], metavar=("X", "Y"))
    ap.add_argument("--truth-n-med", type=float, default=1.33)
    ap.add_argument("--fit-n-med", type=float, nargs="+", default=None, help="default: the truth's")
    ap.add_argument("--blur", nargs="+", default=["round", "free"])
    ap.add_argument("--start", nargs="+", default=["truth", "cold"])
    ap.add_argument("--gamma", nargs="+", default=["held"], choices=("held", "learnt"))
    ap.add_argument("--on-coverslip", type=float, default=0.1)
    ap.add_argument("--roi", type=int, default=25)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--fluorophores", type=int, default=1500)
    ap.add_argument("--molecules", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="benchmarks/uipsf/results/insitu_minimal_sim.jsonl")
    args = ap.parse_args()
    t = time.time()
    truth_m = model(args.truth_n_med, 41, FOCUS, "free", device=args.device)
    truth_sh = truth_shared(truth_m, args.truth_sigma)
    mv = simulate_movie(truth_m, truth_sh, MovieSettings(
        size_px=(128, 128), n_frames=args.frames, fluorophores=args.fluorophores, heights_nm=HEIGHTS,
        on_coverslip=args.on_coverslip, background=55.0, illumination_fwhm_px=200.0, seed=args.seed, spot_px=41,
        blinking=BlinkingSettings(photons=6000.0, photons_std=2500.0, on_time=1.5, blinks=3.0)))
    sel = selection.InSituSettings(molecules=args.molecules, roi_px=args.roi, heldout=0.2)
    rois = selection.insitu_rois(mv.frames, PIXEL, sel)
    ids = np.r_[rois.ids("used"), rois.ids("held out")]
    rows = match(rois, ids, mv.truth)
    ids, rows = ids[rows >= 0], rows[rows >= 0]
    held = np.isin(ids, rois.ids("held out"))
    h_true = mv.truth.h_nm[rows]
    noise = ShiftedPoisson()
    ref_m = model(args.truth_n_med, args.roi, FOCUS, "free", device=args.device)
    ref = heights(ref_m, truth_shared(ref_m, args.truth_sigma), rois, ids, noise)
    base = dict(truth_sigma=args.truth_sigma, truth_n_med=args.truth_n_med, truth_gamma=GAMMA, focus_true_nm=FOCUS,
                on_coverslip=args.on_coverslip, roi=args.roi, seed=args.seed, matched=len(ids),
                held_out=int(held.sum()))
    print(f"movie {time.time() - t:.0f} s, {len(mv.truth)} spots; {rois.counts()}", flush=True)
    emit(dict(base, run="truth model", z=distortion(ref["h"], h_true, ref["sd"], *HEIGHTS)), args.out)
    for n_med, blur, start, gamma in itertools.product(args.fit_n_med or [args.truth_n_med], args.blur, args.start,
                                                       args.gamma):
        t = time.time()
        m = model(n_med, args.roi, FOCUS + 150.0, blur, gamma == "held", args.device)
        run = calibrate_insitu(m, rois, start=start_values(m, start, gamma == "held"), selection=sel,
                               settings=RoundSettings(max_rounds=args.rounds))
        cal = run.final.calibration
        sh = {k: v.detach() for k, v in cal.shared().items()}
        sd = cal.result.std
        got = heights(run.model, sh, rois, ids, noise)
        modes = cal.model.psf.pupil.phase_modes
        phase = sh["pupil.phase"].cpu().double().numpy()
        err = sd["pupil.phase"].cpu().double().numpy()
        noll = {j: [round(float(phase[i] - PHASE.get(j, 0.0)), 3), round(float(err[i]), 3)]
                for i, j in enumerate(modes) if j in (5, 6, 11, 12)}
        twin = run.final.twin
        emit(dict(base, run="in situ", fit_n_med=n_med, blur=blur, start=start, gamma=gamma,
                  gamma_learnt=round(float(sh["pupil.gamma"]), 3) if "pupil.gamma" in sh else None,
                  used=len(run.rounds[-1].used), rounds=len(run.rounds),
                  focus_error_nm=round(float(sh["geometry.focus"]) - FOCUS, 1),
                  focus_sd_nm=round(float(sd["geometry.focus"]), 1) if "geometry.focus" in sd else None,
                  sigma_nm=sh["psf.sigma"].cpu().double().numpy().round(1).tolist(),
                  noll_error=noll, max_phase_error_sd=round(float(np.max(np.abs(
                      [phase[i] - PHASE.get(j, 0.0) for i, j in enumerate(modes)]) / err)), 1),
                  dnats_per_molecule=round(float(np.sum((got["nats"] - ref["nats"])[held])) / int(held.sum()), 3),
                  twin_delta_nats=None if twin is None else round(twin.delta_nats, 1),
                  edges_nm=[None if r.edge is None else round(r.edge.shift_nm, 1) for r in run.rounds],
                  focus_rounds_nm=[round(float(r.calibration.calibration.shared()["geometry.focus"]), 1)
                                   for r in run.rounds],
                  z=distortion(got["h"], h_true, got["sd"], *HEIGHTS), minutes=round((time.time() - t) / 60, 1)),
             args.out)


if __name__ == "__main__":
    main()
