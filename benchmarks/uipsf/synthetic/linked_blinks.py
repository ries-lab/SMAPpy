"""Linked blinks against single frames: which defocused emitters each localises, and what each learns.

    python benchmarks/uipsf/synthetic/linked_blinks.py [--out benchmarks/uipsf/results/linked_blinks.jsonl]

Plan §9, M5: a blink seen in K consecutive frames is one emitter with one position and its photons per
frame (`model.PerObservation`), so the frames' information adds up.  The defocused emitters, which carry
most of what the PSF has to learn, are also the dim ones per pixel; whether linking lets them in is
what this measures.

1. *Localisation*, on the true pupil and focus: emitters 0–1500 nm deep (focus 600 nm, in focus at
   ~530 nm), 3 frames each at 500 or 1500 photons per frame, each localised linked (one emitter) and
   frame by frame (`workflows.insitu.localise_from_heights`, then a fit from the best height).  The
   error in height and the fraction within 100 nm, by distance from the focus.
2. *Learning*: `learn_insitu` from the right sign of astigmatism at half its size, on 30 blinks of 3
   frames linked and on the same 90 frames alone.  The phase error against the CRLB, the focus, the twin
   test and the time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

import uipsf
from uipsf.model import Bleaching, InSituGeometry, PerObservation
from uipsf.noise import Poisson
from uipsf.simulate import parameters, simulate, template
from uipsf.workflows.insitu import in_focus_height, learn_insitu, localise_from_heights
from uipsf.workflows.scoring import localise

f64 = torch.float64
OPTICS = dict(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
PHASE = (0.1, 0.7, -0.2, 0.15, 0.1)                 # Noll 5–9: astigmatism, coma, spherical
FOCUS, K, ROI = 600.0, 3, (13, 13)


def model_and_data(n, photons, heights, frames, seed, true_focus=FOCUS):
    grid = uipsf.PupilGrid.build(uipsf.Optics(**OPTICS), 32, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    psf = uipsf.VectorPSF(grid, pupil, ROI, sigma_nm=(30.0, 30.0))
    linked = uipsf.ForwardModel(psf, geometry=InSituGeometry(FOCUS), intensity=PerObservation())
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g, dtype=f64) - 0.5) * 150
    h = heights[0] + torch.rand(n, generator=g, dtype=f64) * (heights[1] - heights[0])
    phase = torch.zeros(len(pupil.phase_modes), dtype=f64)
    phase[:len(PHASE)] = torch.tensor(PHASE, dtype=f64)
    N = torch.as_tensor(photons, dtype=f64).expand(n).clone()
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0]), "geometry.focus": torch.tensor(true_focus),
              "emitter.xyz": torch.cat([xy, h[:, None]], 1), "emitter.background": torch.full((n,), 10.0, dtype=f64),
              "emitter.photons_per_slice": N[:, None].expand(n, frames).clone()}
    b0 = template(n, torch.zeros(frames, dtype=f64), ROI, dtype=f64)
    truth = parameters(linked, b0, values, dtype=f64)
    batch = simulate(linked, truth, b0, seed=seed + 1)
    single = uipsf.ForwardModel(psf, geometry=InSituGeometry(FOCUS), intensity=Bleaching(order=0))
    E, C, Kf, Y, X = batch.data.shape
    frames_alone = uipsf.Batch(data=batch.data.transpose(1, 2).reshape(E * Kf, C, 1, Y, X),
                               z_nm=torch.zeros(1, dtype=f64))
    return linked, single, truth, batch, frames_alone


def located(model, shared, batch, heights):
    found, _ = localise_from_heights(model, shared, batch, Poisson(), heights)
    fit, _ = localise(model, shared, replace(batch, start=found), Poisson())
    return fit.params.physical()["emitter.xyz"][:, 2].detach().numpy(), fit.std["emitter.xyz"][:, 2].numpy()


def localisation(photons, seed=0, n=60):
    linked, single, truth, batch, alone = model_and_data(n, photons, (0.0, 1500.0), K, seed)
    shared = {k: v for k, v in truth.physical().items() if truth.specs[k].scope == "shared"}
    heights = np.arange(0.0, 1650.0, 150.0)
    h = truth.physical()["emitter.xyz"][:, 2].numpy()
    hl, sl = located(linked, shared, batch, heights)
    hs, ss = located(single, shared, alone, heights)
    off = np.abs(h - in_focus_height(linked, FOCUS))
    rows = []
    for lo, hi in ((0, 300), (300, 600), (600, 1000)):
        m = (off >= lo) & (off < hi)
        el, es = np.abs(hl - h)[m], np.abs(hs - np.repeat(h, K)).reshape(n, K)[m].ravel()
        rows.append(dict(part="localisation", photons_per_frame=photons, defocus_nm=[lo, hi], emitters=int(m.sum()),
                         linked_median_err_nm=float(np.median(el)), linked_within_100=float(np.mean(el < 100)),
                         linked_crlb_nm=float(np.median(sl[m])),
                         single_median_err_nm=float(np.median(es)), single_within_100=float(np.mean(es < 100)),
                         single_crlb_nm=float(np.median(ss.reshape(n, K)[m]))))
    return rows


def learning(seed=0, n=30, photons=1500.0):
    linked, single, truth, batch, alone = model_and_data(n, photons, (0.0, 1200.0), K, seed, true_focus=720.0)
    start = torch.zeros(len(linked.psf.pupil.phase_modes), dtype=f64)
    start[1] = 0.4
    rows = []
    for name, model, data in (("linked", linked, batch), ("single frames", single, alone)):
        t = time.time()
        r = learn_insitu(model, data, start={"pupil.phase": start})
        fit, sh = r.calibration.result, r.calibration.shared()
        err = (sh["pupil.phase"] - truth.physical()["pupil.phase"]).numpy()
        sd = fit.std["pupil.phase"].numpy()
        rows.append(dict(part="learning", data=name, emitters=data.n_emitters, used=len(r.calibration.beads.used),
                         phase_err_rms=float(np.sqrt(np.mean(err ** 2))),
                         phase_crlb_rms=float(np.sqrt(np.mean(sd ** 2))),
                         phase_err_over_crlb_max=float(np.max(np.abs(err) / sd)),
                         focus_nm=float(sh["geometry.focus"]), focus_sd_nm=float(fit.std["geometry.focus"]),
                         twin_nats=r.twin.delta_nats, twin=r.twin.verdict, seconds=time.time() - t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/uipsf/results/linked_blinks.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rows = localisation(500.0) + localisation(1500.0) + learning()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
