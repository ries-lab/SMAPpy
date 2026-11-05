"""Is each M5 bead's own lateral slope with z a tilt of its light cone, an off-centre pupil intensity?

    python benchmarks/uipsf/real/m5_beadtilt.py FOLDER [--beads 40] [--window 1000] [--dz 100] [--memory 0.5] \\
        [--out benchmarks/uipsf/results/m5_beadtilt.jsonl]

`m5_beadplace.py` (2 November 2026): every bead moves sideways with z by a slope of its own (±18 nm/µm in x, ±8 in
y about a shared 31 / 22), which it keeps over ten places on the chip; no faint neighbour at focus explains it
(`m5_neighbours.py`).  A phase tilt across the pupil only shifts an image; an image that moves with defocus has
its light cone tilted, which in the pupil is an intensity off its centre -- the magnitude's tilt, Noll 2 and 3.
Something that shadows or reflects part of one bead's emission cone (residue, a speck above it, a wedge of
medium) would tilt that bead's cone alone.

Per bead, on M5's 25-bead stack (676 nm, buffer, 3D lens): the shared minimal model is learnt as
`m5_sidewalk.py` learns it (minimal0, ring as a height, round σ, linear drift per stack), and each bead's slopes
measured on it.  Then ``--beads`` beads evenly over the x slopes are fitted alone, without drift, with the
shared phase, γ, h and σ held, in three ways:

- *held*: nothing of the pupil learnt, only the bead's own position, photons and background;
- *tilt*: the magnitude's tilt (Noll 2, 3) learnt as well;
- *line*: a linear lateral drift of its own learnt instead, the reference for what a slope is worth;
- *coma*: the bead's own coma (Noll 7, 8) learnt instead of the tilt, the rest of the shared phase held and the
  magnitude's tilt at 0.  Coma also moves an image sideways with defocus, so a slope alone cannot tell a tilted
  cone from an aberration of the bead's own; the likelihood of the whole stack can, both ways learning two numbers.

Per bead and way: the slope on its own model (each slice localised alone, `scoring.slice_fits`), the objective
(½ deviance, nats up to a constant, so differences between ways are gains in nats), the tilt or the coma with its
CRLB.  If the cone's tilt is the cause, *tilt* takes the slope to about 0 and gains about what *line* gains, the
tilt follows the slope measured on the shared model, and *coma* gains less than *tilt*.
``--ways`` picks the ways (``held tilt coma`` compares the two on one shared model).
"""
from __future__ import annotations

import argparse
import copy
import functools
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402
from m5_consensus import GRID_NA, PIXEL, minimal  # noqa: E402
from m5_media import N_IMM, N_MED, ORDER, WAVELENGTH, condition, select, subset  # noqa: E402
from m5_sidewalk import lines  # noqa: E402

Z_RANGE = 600.0


def slopes(run, rois_used, scoring, max_dz):
    """Each used bead's dx/dz, dy/dz (nm/µm) and their errors on ``run``'s own model."""
    cal = run.cal
    used = list(cal.beads.used)
    batch = run.rois.batch.subset(torch.as_tensor(np.asarray(used))).thinned(max_dz)
    found, crlb, _ = scoring.slice_fits(cal.model, cal.shared(), batch, run.rois.noise, cal.result.params)
    zk = np.broadcast_to(batch.z_nm.cpu().double().numpy(), np.asarray(found).shape[:2])
    centre = np.array([rois_used.records[i]["focus_nm"] for i in used])
    s, se, _ = lines(np.asarray(found), np.asarray(crlb), zk, centre, Z_RANGE)
    return s, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--beads", type=int, default=40)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.5)
    ap.add_argument("--ways", nargs="+", default=["held", "tilt", "line"], choices=["held", "tilt", "line", "coma"])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_beadtilt.jsonl"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        bead_rois,
        camera_for,
        run_calibration,
    )

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=1, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1))

    files = sorted(glob.glob(os.path.join(args.folder, "*.ome.tif")))
    cond = condition(args.folder)
    lam = WAVELENGTH[cond["colour"]]
    ring = cond["medium"] == "mismatch"
    rois = bead_rois(files, camera_for(files[0]), dual=False)
    keep, focus, cut = select(rois, os.path.basename(os.path.normpath(args.folder)))
    rois = subset(rois, keep, focus, args.window)
    optics = OpticsSettings(NA=GRID_NA, wavelength_nm=lam, n_immersion=N_IMM, n_medium=N_MED[cond["medium"]],
                            pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])

    def learn(r, magnitude, blur, sigma, drift):
        psf = PSFModelSettings(pixel_blur=False, zernike_order=ORDER, magnitude=magnitude, blur=blur,
                               blur_nm=sigma)
        return run_calibration(files, False, optics, BeadSettings(psf=psf, bead_diameter_nm=100.0, drift=drift),
                               run=RunSettings(save_model=False), rois=r,
                               decisions={i: "include" for i in range(len(r.records))})

    t0 = time.perf_counter()
    shared = learn(rois, minimal("height", ring, (), {}, {}, 100.0), "round", 30.0, "linear")
    p = shared.cal.result.params.physical()
    pupil = shared.cal.model.channels[0].psf.pupil
    gamma, h = float(p["pupil.centre_g"]), float(p["pupil.saf_h"]) if ring else None
    phase = {int(j): float(v) for j, v in pupil.coefficients(p, "phase").items()}
    sigma = float(p["psf.sigma"].reshape(-1)[0])
    s0, se0 = slopes(shared, rois, scoring, beads.MAX_DZ_NM)
    used = list(shared.cal.beads.used)
    emit(dict(stage="shared", **cond, used=len(used), gamma=round(gamma, 4), h_nm=h, sigma_nm=round(sigma, 2),
              objective=round(float(shared.cal.result.objective), 1),
              slope_median=np.round(np.nanmedian(s0, 0), 2).tolist(),
              minutes=round((time.perf_counter() - t0) / 60, 2)), args.out)
    hold = {"pupil.centre_g": gamma} | ({"pupil.saf_h": h} if ring else {})
    # coma: every shared phase mode held but 7 and 8, which are learnt per bead (from 0: phase modes start there)
    coma = (7, 8)
    ways = {"held": ((), "none", phase), "tilt": ((2, 3), "none", phase), "line": ((), "linear", phase),
            "coma": ((), "none", {j: v for j, v in phase.items() if j not in coma})}

    # beads evenly over the x slope, the extremes included
    order = np.argsort(np.nan_to_num(s0[:, 0], nan=np.nanmedian(s0[:, 0])))
    pick = order[np.unique(np.round(np.linspace(0, len(order) - 1, min(args.beads, len(order)))).astype(int))]
    for k in pick:
        i = used[k]
        one = copy.copy(rois)
        one.batch = rois.batch.subset(torch.tensor([i]))
        one.records = [dict(rois.records[i], id=0)]
        rec = rois.records[i]
        row = dict(stage="bead", bead=int(i), stack=int(rec["stack"]), x_px=rec["x_px"], y_px=rec["y_px"],
                   shared_slope=np.round(s0[k], 2).tolist(), shared_slope_se=np.round(se0[k], 2).tolist())
        for way in args.ways:
            modes, drift, held_phase = ways[way]
            t1 = time.perf_counter()
            try:
                run = learn(one, minimal("height", ring, modes, hold, {}, 100.0, held_phase=held_phase), "fixed",
                            sigma, drift)
            except Exception as e:                      # a bead that will not fit is reported, not fatal
                row[way] = dict(failed=str(e))
                continue
            q, std = run.cal.result.params.physical(), run.cal.result.std
            s, se = slopes(run, one, scoring, beads.MAX_DZ_NM)
            entry = dict(objective=round(float(run.cal.result.objective), 2), slope=np.round(s[0], 2).tolist(),
                         slope_se=np.round(se[0], 2).tolist(), seconds=round(time.perf_counter() - t1, 1))
            if modes:
                entry["tilt"] = q["pupil.magnitude"].detach().cpu().double().numpy().round(4).tolist()
                entry["tilt_crlb"] = (std["pupil.magnitude"].detach().cpu().double().numpy().round(4).tolist()
                                      if "pupil.magnitude" in std else None)
            if way == "coma":
                # nm of path, as the consensus reports phase; the shared model's coma beside it
                nm = lam / (2 * np.pi)
                bead_pupil = run.cal.model.channels[0].psf.pupil
                learnt = bead_pupil.coefficients(q, "phase")
                entry["coma_nm"] = [round(learnt[j] * nm, 2) for j in coma]
                entry["shared_coma_nm"] = [round(phase.get(j, 0.0) * nm, 2) for j in coma]
                if "pupil.phase" in std:
                    e = dict(zip(bead_pupil.phase_modes, std["pupil.phase"].detach().cpu().double().numpy(),
                                 strict=True))
                    entry["coma_crlb_nm"] = [round(float(e[j]) * nm, 3) for j in coma]
            if drift == "linear":
                entry["rate"] = q["geometry.drift_rate"].detach().cpu().double().numpy().round(2).tolist()
            row[way] = entry
        emit(row, args.out)


if __name__ == "__main__":
    main()
