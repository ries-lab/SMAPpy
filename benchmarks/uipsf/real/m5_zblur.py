"""Does M5's blur grow with defocus?  σ(z)² = σ₀² + (c·z)² against a constant σ, on the consensus beads.

    python benchmarks/uipsf/real/m5_zblur.py ACQUISITION [...] [--centre 1.0] [--dz 100] [--memory 0.7] \\
        [--out benchmarks/uipsf/results/m5_zblur.jsonl]

The user's hypothesis (31 October 2026, `synthetic/z_blur.py`): aberrations that vary between beads
blur defocused images more than focused ones, so a model shared by the beads wants a blur that grows with
|z|.  The single-bead fits found the premise (`m5_field.py`: the astigmatism changes by 5-7 nm of path per
100 px), and the consensus beads took most of the constant blur away (`results/m5_media_beadset.jsonl`).
Four blurs, round (the user's: σx = σy), each with free Zernike magnitude and phase to order 8 so that only
the blur differs, on the same beads and slices, so the objectives compare in nats:

- *σ*: one constant σ, learnt;
- *σ pixel*: σ held at the pixel's own, p/√12;
- *σ(z), σ₀ pixel*: σ₀ held at the pixel's, c learnt -- the user's expectation;
- *σ(z)*: σ₀ and c learnt.

Per fit: σ₀, c and their CRLBs, the objective, the deviance and the bends (`bead_slices.measure`; the
curve is kept, so the bend over ±200 nm, where most molecules are, comes from it too).
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit, measure  # noqa: E402,I001
from m5_media import GRID_NA, N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402
from z_blur import use_z_blur  # noqa: E402

PIXEL = (104.7, 106.4)
#: name -> (σ(z), blur): the blur as `BeadSettings` takes it
MODELS = {"σ": (False, "round"), "σ pixel": (False, "fixed"), "σ(z), σ₀ pixel": (True, "fixed"),
          "σ(z)": (True, "round")}


def central(rois, keep, fraction: float, shape) -> list:
    H, W = shape
    return [i for i in keep if W * (1 - fraction) / 2 <= rois.records[i]["x_px"] <= W * (1 + fraction) / 2
            and H * (1 - fraction) / 2 <= rois.records[i]["y_px"] <= H * (1 + fraction) / 2]


def _values(d: dict, digits: int, name: str):
    """A parameter's values (or CRLBs) as a list, None where ``d`` has none."""
    return d[name].detach().cpu().double().numpy().reshape(-1).round(digits).tolist() if name in d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--centre", type=float, default=1.0, help="the central fraction of the field; 1: all of it")
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--memory", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_zblur.jsonl"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    import tifffile
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    from uipsf.model.pupil import ZernikePupil
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1) * args.chunk)
    calibrate.ZernikePupil = ZernikePupil
    done = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    have = {(r["acquisition"], r["model"], r["centre"]) for r in done}
    pixel_sigma = float(np.mean(PIXEL)) / math.sqrt(12)

    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        todo = [m for m in args.models if (name, m, args.centre) not in have]
        if not todo:
            continue
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, cut = select(rois, name)
        shape = tifffile.TiffFile(files[0]).pages[0].shape[-2:]
        keep = central(rois, keep, args.centre, shape)
        cut = dict(cut, central=len(keep))
        rois = subset(rois, keep, focus, args.window)
        optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
        for m in todo:
            zblur, blur = MODELS[m]
            t0 = time.perf_counter()
            use_z_blur(zblur)
            model = BeadSettings(zernike_order=ORDER, magnitude="free", blur=blur,
                                 blur_nm=pixel_sigma if blur == "fixed" else 30.0, bead_diameter_nm=bead_nm)
            run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                                  decisions={i: "include" for i in range(len(keep))})
            cal = run.cal
            p, std = cal.result.params.physical(), cal.result.std
            val = functools.partial(_values, p, 4)
            err = functools.partial(_values, std, 5)
            row = dict(acquisition=name, **cond, model=m, centre=args.centre, beads=cut, window_nm=args.window,
                       dz_nm=args.dz, grid_na=GRID_NA, used=len(cal.beads.used),
                       objective=round(float(cal.result.objective), 1), stop=cal.result.stop_reason,
                       sigma0_nm=val("psf.sigma"), sigma0_crlb_nm=err("psf.sigma"),
                       c=val("psf.sigma_c"), c_crlb=err("psf.sigma_c"),
                       atlas_mean_deviance=round(float(cal.atlas.mean_deviance), 3))
            row.update(measure(run, args.z_range, centre_nm=[rois.records[i]["focus_nm"] for i in cal.beads.used]))
            row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
            emit(row, args.out)
            print(name, m, "σ₀", row["sigma0_nm"], "c", row["c"], "z bend", row["z_systematic_rms_nm"],
                  f"{row['minutes']} min", flush=True)
        use_z_blur(False)


if __name__ == "__main__":
    main()
