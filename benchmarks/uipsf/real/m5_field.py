"""Do M5's aberrations vary from bead to bead, and does one shared model turn that into blur?

    python benchmarks/uipsf/real/m5_field.py ACQUISITION [...] [--order 6] [--dz 100] [--memory 0.7] \\
        [--out benchmarks/uipsf/results/m5_field.jsonl]

The user's hypothesis (31 October 2026): aberrations that differ across the field, or from bead to bead,
hardly change a PSF in focus (they lower its peak) but make defocused PSFs differ, increasingly with |z|.
A model shared by every bead can only take that up as blur: σ larger than the optics', too large in focus
(the sharp core every model misses, `m5_media_psfs.py`) and too small far out.

The premise is measured here.  Per acquisition (`m5_media.acquisition`, every bright bead in focus, the
whole field):

1. *common*: one model of every bead, free Zernike magnitude and phase to order 8, σx and σy free;
2. *bead*: each bead alone, its phase to ``--order`` and its own σx, σy learnt, the magnitude and the
   phase above ``--order`` held at the common model's -- what a field-dependent aberration changes is the
   low-order phase.

If the hypothesis holds, the beads' phases spread by more than their CRLBs (a gradient across the field,
or scatter), and each bead alone needs less σ than the common model.  The spread is what the simulation
of the hypothesis takes as its input.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from bead_slices import emit  # noqa: E402,I001
from m5_media import GRID_NA, N_IMM, N_MED, WAVELENGTH, acquisition, select, subset  # noqa: E402

PIXEL = (104.7, 106.4)
COMMON_ORDER = 8


def holding(held_phase: dict, held_magnitude: dict):
    """Make `bead_model` build a Zernike pupil that holds these modes (the rest as it asks)."""
    import uipsf.workflows.calibrate as calibrate
    from uipsf.model.pupil import ZernikePupil

    class Build(ZernikePupil):
        # a class, not a function: the module also asks isinstance(pupil, ZernikePupil); __new__ returns a plain
        # ZernikePupil, so its __init__ is not run twice
        def __new__(cls, grid, n_max, **kwargs):
            return ZernikePupil(grid, n_max=n_max, **dict(kwargs, held_phase=held_phase, held_magnitude=held_magnitude))
    calibrate.ZernikePupil = Build


def fitted(run, lam) -> dict:
    """Phase (nm of path) and σ of a fit, with their CRLBs, and every coefficient the pupil holds."""
    cal = run.cal
    p, std = cal.result.params.physical(), cal.result.std
    pupil = cal.model.channels[0].psf.pupil
    path = lam / (2 * math.pi)
    phase = p["pupil.phase"].detach().cpu().double().numpy()
    err = std["pupil.phase"].detach().cpu().double().numpy() if "pupil.phase" in std else np.full(len(phase), np.nan)
    return dict(objective=round(float(cal.result.objective), 1), stop=cal.result.stop_reason,
                sigma_nm=p["psf.sigma"].detach().cpu().double().numpy().round(2).tolist(),
                sigma_crlb_nm=(std["psf.sigma"].detach().cpu().double().numpy().round(2).tolist()
                               if "psf.sigma" in std else None),
                phase_nm_path={int(j): round(float(v) * path, 2)
                               for j, v in zip(pupil.phase_modes, phase, strict=True)},
                phase_crlb_nm={int(j): round(float(v) * path, 3)
                               for j, v in zip(pupil.phase_modes, err, strict=True)},
                all_phase={int(j): float(v) for j, v in pupil.coefficients(p, "phase").items()},
                all_magnitude={int(j): float(v) for j, v in pupil.coefficients(p, "magnitude").items()},
                atlas_mean_deviance=round(float(cal.atlas.mean_deviance), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--order", type=int, default=6, help="the phase each bead learns alone, radial order")
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_field.jsonl"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    from uipsf.model.pupil import ZernikePupil
    from uipsf.optics.zernike import noll_to_nm
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=1, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1))
    done = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    have = {(r["acquisition"], r["stage"], r.get("bead")) for r in done}

    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        lam = WAVELENGTH[cond["colour"]]
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, cut = select(rois)
        rois = subset(rois, keep, focus, args.window)
        optics = OpticsSettings(NA=GRID_NA, wavelength_nm=lam, n_immersion=N_IMM, n_medium=N_MED[cond["medium"]],
                                pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])

        def learn(r, order, decisions, files=files, optics=optics, bead_nm=bead_nm):
            model = BeadSettings(zernike_order=order, magnitude="free", blur="free", blur_nm=30.0,
                                 bead_diameter_nm=bead_nm)
            return run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=r,
                                   decisions=decisions)

        base = dict(acquisition=name, **cond, window_nm=args.window, dz_nm=args.dz, order=args.order)
        common = next((r for r in done if r["acquisition"] == name and r["stage"] == "common"), None)
        if common is None:
            t0 = time.perf_counter()
            calibrate.ZernikePupil = ZernikePupil
            run = learn(rois, COMMON_ORDER, {i: "include" for i in range(len(keep))})
            common = dict(base, stage="common", beads=cut, used=len(run.cal.beads.used), **fitted(run, lam),
                          minutes=round((time.perf_counter() - t0) / 60, 2))
            emit(common, args.out)
            common = json.loads(json.dumps(common))
            print(name, "common", common["sigma_nm"], f"{common['minutes']} min", flush=True)
        # each bead alone: the magnitude and the high-order phase the common model's
        held_phase = {int(j): v for j, v in common["all_phase"].items() if noll_to_nm(int(j))[0] > args.order}
        held_magnitude = {int(j): v for j, v in common["all_magnitude"].items()}
        holding(held_phase, held_magnitude)
        for i in range(len(keep)):
            if (name, "bead", i) in have:
                continue
            t0 = time.perf_counter()
            one = copy.copy(rois)
            one.batch = rois.batch.subset(torch.tensor([i]))
            one.records = [dict(rois.records[i], id=0)]
            try:
                run = learn(one, COMMON_ORDER, {0: "include"})
            except Exception as e:                      # one bead that will not fit is reported, not fatal
                print(name, "bead", i, "failed:", e, flush=True)
                continue
            rec = rois.records[i]
            row = dict(base, stage="bead", bead=i, x_px=rec["x_px"], y_px=rec["y_px"], focus_nm=rec["focus_nm"],
                       **fitted(run, lam), minutes=round((time.perf_counter() - t0) / 60, 2))
            row.pop("all_magnitude")
            emit(row, args.out)
            print(name, "bead", i, row["sigma_nm"], f"{row['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
