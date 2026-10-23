"""In situ learning on whole simulated movies: the rounds, the coverslip edge, the focus from the data.

    python benchmarks/uipsf/synthetic/insitu_movie.py [--out benchmarks/uipsf/results/insitu_movie.jsonl]

What the tests cannot afford (each case is minutes, the tests must be seconds): a movie of molecules
0–1400 nm deep (`uipsf.movie`), its molecules found and chosen (`smappy.calibrate.insitu`), learnt in
rounds with the coverslip edge (`uipsf.workflows.insitu_rounds.calibrate_insitu`) from a focus started
250 nm off, and the focus estimated from the data before learning (`focus_from_data`).  The truth: an
astigmatic pupil (Noll 5–9: 0.1, 0.7, −0.2, 0.15, 0.1 rad), focus 700 nm, NA 1.43 oil into water.

Cases: a fifth of the molecules on the coverslip, and none.  Recorded per round: the focus and its error,
the edge, the largest Zernike error in standard errors, the twin test; and the time.  The numbers are in
docs/uipsf/architecture.md, "M5, step 5" and "step 6".
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from smappy.calibrate import insitu as selection
from smappy.simulate import BlinkingSettings

import uipsf
from uipsf.model import Bleaching, InSituGeometry
from uipsf.movie import MovieSettings, simulate_movie
from uipsf.workflows.insitu_rounds import RoundSettings, calibrate_insitu, focus_from_data

f64 = torch.float64
OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
PHASE = (0.1, 0.7, -0.2, 0.15, 0.1)
FOCUS = 700.0


def model(focus_nm: float, P: int = 32):
    grid = uipsf.PupilGrid.build(OPTICS, P, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    return uipsf.ForwardModel(uipsf.VectorPSF(grid, pupil, (15, 15), sigma_nm=(30.0, 30.0)),
                              geometry=InSituGeometry(focus_nm), intensity=Bleaching(order=0))


def truth(m):
    pupil = m.psf.pupil
    phase = torch.zeros(len(pupil.phase_modes), dtype=f64)
    phase[:len(PHASE)] = torch.tensor(PHASE, dtype=f64)
    return {"pupil.phase": phase, "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes), dtype=f64),
            "psf.sigma": torch.tensor([35.0, 35.0], dtype=f64), "geometry.focus": torch.tensor(FOCUS, dtype=f64)}


def case(on_coverslip: float, fluorophores: int = 600, frames: int = 600, per_bin: int = 20, seed: int = 1):
    m = model(FOCUS)
    shared = truth(m)
    movie = simulate_movie(m, shared, MovieSettings(
        size_px=(128, 128), n_frames=frames, fluorophores=fluorophores, heights_nm=(0.0, 1400.0),
        on_coverslip=on_coverslip, illumination_fwhm_px=160.0, seed=seed,
        blinking=BlinkingSettings(photons=6000.0, photons_std=3000.0, on_time=1.5, blinks=3.0)))
    sel = selection.InSituSettings(molecules=per_bin * selection.InSituSettings().bins)
    rois = selection.insitu_rois(movie.frames, OPTICS.pixel_size_nm, sel)
    start = torch.zeros_like(shared["pupil.phase"])
    start[1] = 0.4
    rows = []
    t = time.time()
    run = calibrate_insitu(model(450.0), rois, start={"pupil.phase": start}, selection=sel,
                           settings=RoundSettings(max_rounds=4))
    seconds = time.time() - t
    for r in run.rounds:
        cal = r.calibration.calibration
        sd = cal.result.std
        err = (cal.shared()["pupil.phase"] - shared["pupil.phase"]).abs() / sd["pupil.phase"]
        rows.append(dict(part="rounds", on_coverslip=on_coverslip, round=r.index, used=len(r.used),
                         focus_nm=float(cal.shared()["geometry.focus"]), focus_sd_nm=float(sd["geometry.focus"]),
                         phase_err_max_sd=float(err.max()), moved=float(r.moved),
                         edge=None if r.edge is None else [r.edge.shift_nm, r.edge.sd_nm, r.edge.spike]))
    rows[-1].update(twin_nats=run.final.twin.delta_nats, seconds=seconds)
    t = time.time()
    focus, edge = focus_from_data(model(1000.0), rois, {"pupil.phase": start}, guess_nm=1000.0)
    rows.append(dict(part="focus from data", on_coverslip=on_coverslip, guess_nm=1000.0, focus_nm=focus,
                     edge=[edge.shift_nm, edge.sd_nm, edge.spike], seconds=time.time() - t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/uipsf/results/insitu_movie.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rows = case(0.2) + case(0.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
