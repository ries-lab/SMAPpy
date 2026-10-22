"""Goodness of fit on beads beyond the nats: every slice localised alone on the learnt model.

    UIPSF_DEVICE=cpu python benchmarks/uipsf/real/bead_slices.py BEAD_ACQUISITION --wavelength 680 \\
        --pixel 127 116 [--region X Y W H] [--out benchmarks/uipsf/results/bead_slices.jsonl]

The idea (the user's, October 2026): a bead does not move while the stage steps through it, so each slice
localised on its own (`uipsf.workflows.scoring.slice_fits`) should give the same x and y and a z on the
stage's straight line.  A model that is wrong in a way the stack fit can absorb still shows here: x and y
wander with z, z bends away from the line.  Learnt with the magnitude free, symmetric or uniform and the
blur free or held at 0, the same beads, so the choices are compared on what a localisation would see.

Per variant: the stack fit's objective (nats); the *systematic* error -- at each slice the median over
the used beads of (slice fit − stack fit), which the noise averages out of -- as an rms over the slices
within ``--z-range`` and its largest excursion, for x, y and z; and the *scatter* about that curve against
the CRLB (1 when the model is right and the noise is as modelled).  A slice fit is per frame, so it has no
bead kernel (`scoring._single_frames`): a 100 nm bead's own size is missing from it in every variant alike.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

NA, N_IMM, N_MED, BEAD_NM, ORDER = 1.43, 1.516, 1.33, 100.0, 8
VARIANTS = [("free", "free"), ("symmetric", "free"), ("fixed", "free"),
            ("free", "zero"), ("symmetric", "zero"), ("fixed", "zero")]


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def measure(run, z_range: float) -> dict:
    """The slice fits of the used beads against the stack fit: systematic curve and scatter, nm."""
    from uipsf.workflows.beads import MAX_DZ_NM
    from uipsf.workflows.scoring import slice_fits
    cal = run.cal
    used = torch.as_tensor(np.asarray(cal.beads.used))
    # the slices learning saw: its per-slice terms (stage, drift) are theirs
    batch = run.rois.batch.subset(used).thinned(MAX_DZ_NM)
    stack = cal.result.params
    # the stack fit's rows are the used beads already (`learn_beads`)
    found, crlb, where = slice_fits(cal.model, cal.shared(), batch, run.rois.noise, stack)
    d = found - where                                                   # [E, K, 3]
    z = where[..., 2] - np.median(where[..., 2], axis=1, keepdims=True)  # each bead's own z, centred
    near = np.abs(z) <= z_range
    d = np.where(near[..., None] & np.isfinite(d), d, np.nan)
    d = d - np.nanmedian(d, axis=1, keepdims=True)                       # each bead's own offset out
    curve = np.nanmedian(d, axis=0)                                      # [K, 3] the systematic error
    rest = (d - curve[None]) / crlb
    out = {}
    for a, name in enumerate("xyz"):
        c = curve[:, a][np.isfinite(curve[:, a])]
        out[f"{name}_systematic_rms_nm"] = round(float(np.sqrt(np.mean(c ** 2))), 2) if len(c) else None
        out[f"{name}_systematic_max_nm"] = round(float(np.max(np.abs(c))), 2) if len(c) else None
        r = rest[..., a][np.isfinite(rest[..., a])]
        out[f"{name}_scatter_over_crlb"] = round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3)
        out[f"{name}_crlb_median_nm"] = round(float(np.nanmedian(np.where(near, crlb[..., a], np.nan))), 2)
    out["slices"] = int(near.sum())
    out["curve_z_nm"] = np.nanmedian(np.where(near, z, np.nan), axis=0).round(0)
    out["curve_nm"] = curve.round(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisition")
    ap.add_argument("--wavelength", type=float, required=True)
    ap.add_argument("--pixel", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--region", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--z-range", type=float, default=600.0, help="slices within this of each bead's focus, nm")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from smappy.calibrate.core import CalibrationSettings
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        RunSettings,
        bead_rois,
        camera_for,
        run_calibration,
    )
    files = sorted(glob.glob(os.path.join(args.acquisition, "*", "*.ome.tif")))
    optics = OpticsSettings(NA=NA, wavelength_nm=args.wavelength, n_immersion=N_IMM, n_medium=N_MED,
                            pixel_x_nm=args.pixel[0], pixel_y_nm=args.pixel[1])
    detection = CalibrationSettings(region=tuple(args.region)) if args.region else None
    rois = bead_rois(files, camera_for(files[0]), dual=False, settings=detection)
    # every bead included by hand: the same beads in every variant, so the objectives compare
    decisions = {i: "include" for i in range(len(rois.records))}
    for magnitude, blur in VARIANTS:
        t0 = time.perf_counter()
        # a free blur starts at 30 nm as `lab_beads.py`'s: from 0 its transform cannot move it
        model = BeadSettings(zernike_order=ORDER, magnitude=magnitude, blur="free" if blur == "free" else "fixed",
                             blur_nm=30.0 if blur == "free" else 0.0, bead_diameter_nm=BEAD_NM)
        run = run_calibration(files, False, optics, model, detection=detection, run=RunSettings(save_model=False),
                              rois=rois, decisions=decisions)
        p = run.cal.result.params.physical()
        row = dict(acquisition=os.path.basename(os.path.normpath(args.acquisition)), magnitude=magnitude, blur=blur,
                   used=len(run.cal.beads.used), objective=round(float(run.cal.result.objective), 1),
                   sigma_nm=p["psf.sigma"].detach().cpu().double().numpy().round(2),
                   z_range_nm=args.z_range)
        row.update(measure(run, args.z_range))
        row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
        emit(row, args.out)


if __name__ == "__main__":
    main()
