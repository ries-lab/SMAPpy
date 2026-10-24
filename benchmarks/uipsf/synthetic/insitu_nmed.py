"""Does a wrong sample index distort z in situ, and does the likelihood find the right one?

    python benchmarks/uipsf/synthetic/insitu_nmed.py [--truth 1.37] [--fit 1.31 1.33 ... 1.40] \\
        [--heights 0 2000] [--out benchmarks/uipsf/results/insitu_nmed.jsonl]

The question (the user's, October 2026): in thick samples and focused deep, does an n_med typed as water when
the sample is cytoplasm-like bend z over the depth of field (~2 µm)?  Absolute z matters less than a model
without distortion.  A simulated movie of M2's optics (NA 1.43, Evolve pixels 116 x 127 nm, 680 nm, oil
1.516) whose sample index is ``--truth``, its molecules spread over ``--heights``, the focus imaging the
middle of them sharply; the January beads' pupil (`insitu_region.py`) and symmetric magnitude, held as the
window holds a bead model's.  Molecules are found once and learnt in rounds (`calibrate_insitu`) at each
index of ``--fit``; then every held-out molecule is localised from the same start heights on the learnt
model, and on the truth for reference.

Per index: the likelihood of the held-out molecules (Δnats against the truth model, the profile over
n_med), the heights against the truth -- slope, and what is left after a line (rms, and a median per 200 nm
bin), the distortion a straight z scale cannot take out -- the learnt focus, blur and the phase's spherical
modes, which are what a wrong index would be absorbed by.  One JSON line per index.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from smappy.calibrate import insitu as selection
from smappy.simulate import BlinkingSettings

import uipsf
from uipsf.movie import MovieSettings, simulate_movie
from uipsf.noise import ShiftedPoisson
from uipsf.workflows import insitu_model
from uipsf.workflows.insitu import start_heights
from uipsf.workflows.insitu_rounds import RoundSettings, calibrate_insitu, localise_units

#: as `insitu_region.py`'s: January's bead model, rounded -- phase (rad, Noll), the symmetric magnitude, held
ORDER = 6
PHASE = {5: -0.05, 6: -0.68, 7: 0.05, 8: 0.08, 11: 0.10, 12: 0.0, 13: 0.03}
MAGNITUDE = {4: -0.08, 11: -0.07, 22: -0.05}
SIGMA = 55.0


def held_magnitude(model) -> dict:
    """The truth's symmetric magnitude, as many of its modes as the order has."""
    from uipsf.optics.zernike import noll_to_nm
    return {j: v for j, v in MAGNITUDE.items() if noll_to_nm(j)[0] <= ORDER}

PIXEL = (116.0, 127.0)
N_IMM, NA, WAVELENGTH = 1.516, 1.43, 680.0
BIN_NM = 200.0


def optics(n_med: float) -> uipsf.Optics:
    return uipsf.Optics(NA=NA, wavelength_nm=WAVELENGTH, n_imm=N_IMM, n_med=n_med, pixel_size_nm=PIXEL)


def model(n_med: float, roi: int, focus: float, z_max: float, device=None):
    from smappy.calibrate.pupil import BeadSettings
    settings = BeadSettings(zernike_order=ORDER, magnitude="fixed", blur="free", blur_nm=SIGMA)
    return insitu_model(optics(n_med), settings, (roi, roi), focus, z_max_nm=z_max, device=device,
                        held_magnitude=held_magnitude(None))


def shared(m, focus: float) -> dict:
    pupil = m.psf.pupil
    dev, dt = pupil.grid.rho.device, pupil.grid.rho.dtype
    return {"pupil.phase": torch.tensor([PHASE.get(j, 0.0) for j in pupil.phase_modes], device=dev, dtype=dt),
            "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes), device=dev, dtype=dt),
            "psf.sigma": torch.tensor([SIGMA, SIGMA], device=dev, dtype=dt),
            "geometry.focus": torch.tensor(focus, device=dev, dtype=dt)}


def match(rois, ids, truth, max_px: float = 1.0) -> np.ndarray:
    """Each unit's truth row (-1 where none is alone within ``max_px``): the same frame, the nearest spot."""
    u = rois.units
    tx, ty = truth.x_nm / PIXEL[0], truth.y_nm / PIXEL[1]
    by_frame: dict = {}
    for k, f in enumerate(truth.frame):
        by_frame.setdefault(int(f), []).append(k)
    out = np.full(len(ids), -1)
    for i, uid in enumerate(ids):
        rows = np.asarray(by_frame.get(int(u.frames[uid][0]), []), int)
        if not len(rows):
            continue
        d = np.hypot(tx[rows] - u.x_px[uid], ty[rows] - u.y_px[uid])
        order = np.argsort(d)
        if d[order[0]] <= max_px and (len(order) == 1 or d[order[1]] > 3 * max_px):
            out[i] = rows[order[0]]
    return out


def heights(m, sh: dict, rois, ids, noise) -> dict:
    loc = localise_units(m, sh, rois, ids, noise, heights=start_heights(m, float(sh["geometry.focus"])))
    pixels = rois.units.masks[ids].reshape(len(ids), -1).sum(1)
    return dict(h=loc["h"], sd=loc["sd_h"], nats=loc["deviance"] * pixels / 2)


def distortion(h_fit, h_true, sd, lo: float, hi: float) -> dict:
    """The heights against the truth: a line (slope, offset), the rms about it, the median residual per bin."""
    ok = np.isfinite(h_fit) & (sd < 100.0)
    x, y = h_true[ok], h_fit[ok]
    slope, offset = np.polyfit(x, y, 1)
    r = y - (slope * x + offset)
    edges = np.arange(lo, hi + BIN_NM, BIN_NM)
    med = []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        sel = (x >= a) & (x < b)
        med.append(round(float(np.median(r[sel])), 1) if sel.sum() >= 10 else None)
    bins = [v for v in med if v is not None]
    return dict(n=int(ok.sum()), slope=round(float(slope), 4), offset_nm=round(float(offset), 1),
                rms_nm=round(float(np.sqrt(np.mean(r ** 2))), 1),
                bias_pp_nm=round(float(max(bins) - min(bins)), 1) if bins else None, bins_nm=med,
                mean_error_nm=round(float(np.median(y - x)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=float, default=1.37)
    ap.add_argument("--fit", type=float, nargs="+", default=[1.33, 1.35, 1.37, 1.39])
    ap.add_argument("--heights", type=float, nargs=2, default=[0.0, 2000.0])
    ap.add_argument("--roi", type=int, default=25)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--fluorophores", type=int, default=2000)
    ap.add_argument("--molecules", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--memory", type=float, default=0.3, help="MPS memory fraction (the Mac is shared)")
    ap.add_argument("--out", default="benchmarks/uipsf/results/insitu_nmed.jsonl")
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    lo, hi = args.heights
    middle = (lo + hi) / 2
    # the focus that images the middle sharply, paraxially; z_max covers the molecules' distance from it
    focus = middle * N_IMM / args.truth
    z_max = (hi - lo) / 2 + 300.0
    t = time.time()
    truth_m = model(args.truth, 41, focus, z_max, args.device)
    truth_sh = shared(truth_m, focus)
    mv = simulate_movie(truth_m, truth_sh, MovieSettings(
        size_px=(128, 128), n_frames=args.frames, fluorophores=args.fluorophores, heights_nm=(lo, hi),
        on_coverslip=0.1 if lo == 0 else 0.0, background=55.0, illumination_fwhm_px=200.0, seed=args.seed,
        spot_px=41, blinking=BlinkingSettings(photons=6000.0, photons_std=2500.0, on_time=1.5, blinks=3.0)))
    sel = selection.InSituSettings(molecules=args.molecules, roi_px=args.roi, heldout=0.2)
    rois = selection.insitu_rois(mv.frames, PIXEL, sel)
    # z on every molecule of the first round's choice (its ~50 shared parameters barely fit 2000 molecules'
    # heights); the likelihood only on the held-out ones, which no run learns from
    ids = np.r_[rois.ids("used"), rois.ids("held out")]
    rows = match(rois, ids, mv.truth)
    ids, rows = ids[rows >= 0], rows[rows >= 0]
    held = np.isin(ids, rois.ids("held out"))
    h_true = mv.truth.h_nm[rows]
    noise = ShiftedPoisson()
    print(f"movie {time.time() - t:.0f} s, {len(mv.truth)} spots; {rois.counts()}; {len(ids)} matched, "
          f"{held.sum()} held out", flush=True)
    ref_m = model(args.truth, args.roi, focus, z_max, args.device)
    ref = heights(ref_m, shared(ref_m, focus), rois, ids, noise)
    base = dict(truth_n_med=args.truth, heights_nm=[lo, hi], focus_true_nm=round(focus, 1), roi=args.roi,
                seed=args.seed, matched=len(ids), held_out=int(held.sum()), counts=rois.counts())
    row = dict(base, fit_n_med=args.truth, run="truth model", delta_nats=0.0,
               z=distortion(ref["h"], h_true, ref["sd"], lo, hi))
    emit(row, args.out)
    for n in args.fit:
        t = time.time()
        m = model(n, args.roi, focus + 150.0, z_max, args.device)
        start = {k: v for k, v in shared(m, focus).items() if k != "geometry.focus"}
        run = calibrate_insitu(m, rois, start=start, selection=sel,
                               settings=RoundSettings(max_rounds=args.rounds, twin_test=False))
        cal = run.final.calibration
        sh = {k: v.detach() for k, v in cal.shared().items()}
        got = heights(run.model, sh, rois, ids, noise)
        modes = cal.model.psf.pupil.phase_modes
        phase = sh["pupil.phase"].cpu().double().numpy()
        row = dict(base, fit_n_med=n, run="in situ", rounds=len(run.rounds), used=len(run.rounds[-1].used),
                   delta_nats=round(float(np.sum((got["nats"] - ref["nats"])[held])), 1),
                   focus_nm=round(float(sh["geometry.focus"]), 1),
                   sigma_nm=sh["psf.sigma"].cpu().double().numpy().round(1).tolist(),
                   phase_error={j: round(float(v - PHASE.get(j, 0.0)), 3) for j, v in zip(modes, phase, strict=True)
                                if abs(v - PHASE.get(j, 0.0)) > 0.03},
                   z=distortion(got["h"], h_true, got["sd"], lo, hi), seconds=round(time.time() - t, 1))
        emit(row, args.out)


def emit(row, out):
    print(json.dumps(row), flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
