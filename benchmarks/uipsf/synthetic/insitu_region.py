"""Does the molecule's region size bias what in situ learns?  A simulated M2 movie learnt at several sizes.

    python benchmarks/uipsf/synthetic/insitu_region.py [--roi 15 21 25] [--out results/insitu_region.jsonl]

On M2's January dSTORM pair every v2 in situ run at 15 px learnt a primary astigmatism (Noll 6) 0.3-0.4 rad
stronger than the beads' and a secondary one (Noll 12) of -0.2; at 25 px, and in uiPSF v1 (25 px), it
agreed with the beads (docs/uipsf/architecture.md, "In situ on matched M2 pairs").  Here the same on a
movie whose pupil is known: M2's optics (NA 1.43, Evolve pixels 116 x 127 nm, 680 nm, oil into water), its
beads' astigmatism (Noll 6 -0.68 rad) and symmetric magnitude, molecules 0-1300 nm deep with the focus at
430 nm, ~3000 photons on 55 per pixel.  The movie is rendered in 41 px, so no truth is cut off; molecules are
found, chosen and learnt at each region size by the code the window runs (`smappy.calibrate.insitu`,
`calibrate_insitu`), from the truth's own pupil, the symmetric magnitude held as the window holds the start's.
A learnt value that walks away from the truth it started at is a bias.  Prints and appends one JSON line per
region size.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from smappy.calibrate import insitu as selection
from smappy.calibrate.pupil import BeadSettings
from smappy.simulate import BlinkingSettings

import uipsf
from uipsf.movie import MovieSettings, render, simulate_movie
from uipsf.workflows import insitu_model
from uipsf.workflows.insitu_rounds import RoundSettings, calibrate_insitu

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.33, pixel_size_nm=(116, 127))
ORDER = 6
#: rad, Noll: January's bead model, rounded (`~/.cache/uipsf/insitu_jan/beads.uipsf.h5`)
PHASE = {5: -0.05, 6: -0.68, 7: 0.05, 8: 0.08, 11: 0.10, 12: 0.0, 13: 0.03}
MAGNITUDE = {4: -0.08, 11: -0.07, 22: -0.05}    # the symmetric magnitude, held (order 6 has no 22: dropped)
SIGMA, FOCUS = 55.0, 430.0


def truth_model(roi: int, focus: float = FOCUS, device=None, dtype=torch.float32, held_magnitude=None):
    settings = BeadSettings(zernike_order=ORDER, magnitude="fixed", blur="free", blur_nm=SIGMA)
    return insitu_model(OPTICS, settings, (roi, roi), focus, z_max_nm=1300.0, device=device, dtype=dtype,
                        held_magnitude=held_magnitude)


def held_magnitude(model) -> dict:
    """The truth's symmetric magnitude, as many of its modes as the order has."""
    from uipsf.optics.zernike import noll_to_nm
    return {j: v for j, v in MAGNITUDE.items() if noll_to_nm(j)[0] <= ORDER}


def truth_shared(model) -> dict:
    pupil = model.psf.pupil
    dev, dt = pupil.grid.rho.device, pupil.grid.rho.dtype
    phase = torch.tensor([PHASE.get(j, 0.0) for j in pupil.phase_modes], device=dev, dtype=dt)
    return {"pupil.phase": phase, "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes), device=dev, dtype=dt),
            "psf.sigma": torch.tensor([SIGMA, SIGMA], device=dev, dtype=dt),
            "geometry.focus": torch.tensor(FOCUS, device=dev, dtype=dt)}


def movie(frames: int, fluorophores: int, seed: int, device=None, halo=None):
    """The movie; ``halo`` (fraction, σ nm) adds that fraction of each spot's photons again, blurred by a
    Gaussian of σ: light about each molecule that the model has no term for."""
    m = truth_model(41, device=device, held_magnitude=held_magnitude(None))
    shared = truth_shared(m)
    mv = simulate_movie(m, shared, MovieSettings(
        size_px=(128, 128), n_frames=frames, fluorophores=fluorophores, heights_nm=(0.0, 1300.0),
        on_coverslip=0.1, background=55.0, illumination_fwhm_px=200.0, seed=seed, spot_px=41,
        blinking=BlinkingSettings(photons=6000.0, photons_std=2500.0, on_time=1.5, blinks=3.0)))
    if halo:
        from dataclasses import replace
        fraction, sigma = halo
        wide = dict(shared, **{"psf.sigma": torch.full_like(shared["psf.sigma"], float(sigma))})
        extra = np.zeros_like(mv.expected, dtype=np.float64)
        render(m, wide, replace(mv.truth, photons=mv.truth.photons * fraction), extra, 41)
        expected = mv.expected + extra
        frames = np.random.default_rng(seed + 1000).poisson(expected).astype(np.float32)
        mv = replace(mv, frames=frames, expected=expected.astype(np.float32))
    return mv


def learn(frames, roi: int, molecules: int, rounds: int, device=None) -> dict:
    sel = selection.InSituSettings(molecules=molecules, roi_px=roi)
    t = time.time()
    rois = selection.insitu_rois(frames, OPTICS.pixel_size_nm, sel)
    detect_s = time.time() - t
    model = truth_model(roi, FOCUS + 100.0, device=device, held_magnitude=held_magnitude(None))
    start = {k: v for k, v in truth_shared(model).items() if k != "geometry.focus"}
    t = time.time()
    run = calibrate_insitu(model, rois, start=start, selection=sel, settings=RoundSettings(max_rounds=rounds))
    cal = run.final.calibration
    p, sd = cal.shared(), cal.result.std
    modes = cal.model.psf.pupil.phase_modes
    phase = p["pupil.phase"].detach().cpu().double().numpy()
    err = sd["pupil.phase"].detach().cpu().double().numpy()
    counts = rois.counts()
    return dict(roi=roi, molecules=molecules, used=len(run.rounds[-1].used), counts=counts,
                masked=float(np.mean(1 - rois.units.masks[rois.ids("used")].mean((1, 2)))),
                focus_nm=round(float(p["geometry.focus"]), 1),
                sigma_nm=p["psf.sigma"].detach().cpu().double().numpy().round(1).tolist(),
                phase={j: round(float(v), 4) for j, v in zip(modes, phase, strict=True)},
                error={j: round(float(v - PHASE.get(j, 0.0)), 4) for j, v in zip(modes, phase, strict=True)},
                crlb={j: round(float(e), 4) for j, e in zip(modes, err, strict=True)},
                twin_nats=None if run.final.twin is None else round(run.final.twin.delta_nats, 1),
                detect_s=round(detect_s, 1), learn_s=round(time.time() - t, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, nargs="+", default=[15, 21, 25])
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--fluorophores", type=int, default=1500)
    ap.add_argument("--molecules", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--halo", type=float, nargs=2, default=None, metavar=("FRACTION", "SIGMA_NM"),
                    help="add this fraction of each spot's light again, blurred by a Gaussian of this σ")
    ap.add_argument("--device", default=None)
    ap.add_argument("--memory", type=float, default=0.3, help="MPS memory fraction (the Mac is shared)")
    ap.add_argument("--out", default="benchmarks/uipsf/results/insitu_region.jsonl")
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    t = time.time()
    mv = movie(args.frames, args.fluorophores, args.seed, args.device, args.halo)
    print(f"movie: {len(mv.truth)} spots in {time.time() - t:.0f} s", flush=True)
    for roi in args.roi:
        row = dict(seed=args.seed, frames=args.frames, fluorophores=args.fluorophores, blink_photons=6000.0,
                   truth=PHASE, halo=args.halo,
                   **learn(mv.frames, roi, args.molecules, args.rounds, args.device))
        print(json.dumps(row), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
