"""The buffer's index learnt on one objective's beads: M5's 100 nm beads in buffer (docs/4pi.md §24).

    UIPSF_DEVICE=cpu python benchmarks/uipsf/real/m5_nmed.py ACQUISITION [...] \\
        [--out benchmarks/uipsf/results/m5_nmed.jsonl]

The question the 4Pi beads raised: the minimal pupil wants an aperture below the objective's NA (4Pi: 1.31 behind
1.35; one objective 0.97-0.98 of the NA too, `docs/uipsf/architecture.md`).  Is the buffer's index lower than
typed, so that the light beyond it is what is missing?  Each acquisition (`m5_media.acquisition`, its beads as
`m5_consensus.py` cuts them: the bead set, the middle half of the chip, ±1 µm about focus at 100 nm) is learnt
through the window's own path (`run_calibration`) with the shipped defaults -- the minimal pupil to order 8, σ
round over the pixel, linear drift, SAF damping -- at the nominal NA 1.50 and PBS's 1.334, three ways:

- **typed**: n_med and the NA as typed;
- **n_med**: the sample's index learnt from 1.334 (*learn sample index*);
- **NA**: the aperture learnt (*learn NA*).

One JSON line per fit: nats, n_med, the NA, γ, h, σ, each with its CRLB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from m5_consensus import NA, PIXEL, central  # noqa: E402
from m5_media import N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402

VARIANTS = {"typed": {}, "n_med": dict(learn_n_med=True), "NA": dict(learn_na=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--centre", type=float, default=0.5)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_nmed.jsonl"))
    args = ap.parse_args()
    import functools

    import tifffile

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    # m5_consensus.py's cut: slices 100 nm apart, one bead per chunk (a 38 GB chunk otherwise, M5's 25 px ROIs)
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise,
                                         chunk_size=int(2 * args.window // args.dz + 1) * args.chunk)
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        bead_rois,
        camera_for,
        run_calibration,
    )
    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, _ = select(rois, name)
        inner = central(rois, keep, args.centre, tifffile.TiffFile(files[0]).pages[0].shape[-2:])
        rois = subset(rois, inner, focus, args.window)
        optics = OpticsSettings(NA=NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
        for variant in args.variants.split(","):
            t0 = time.perf_counter()
            model = BeadSettings(psf=PSFModelSettings(zernike_order=ORDER, **VARIANTS[variant]),
                                 bead_diameter_nm=bead_nm)
            run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                                  decisions={i: "include" for i in range(len(rois.records))})
            cal = run.cal
            p, std = cal.result.params.physical(), cal.result.std

            def value(name, p=p):
                return None if name not in p else round(float(p[name].reshape(-1)[0]), 4)

            def crlb(name, std=std):
                return None if name not in std else round(float(std[name].reshape(-1)[0]), 4)

            row = dict(acquisition=name, **cond, variant=variant, beads=len(cal.beads.used),
                       objective=round(float(cal.result.objective), 1), stop=cal.result.stop_reason,
                       NA=NA, n_med_typed=N_MED[cond["medium"]],
                       **{k: value(n) for k, n in (("n_med", "optics.n_med"), ("na", "pupil.na"),
                                                   ("gamma", "pupil.gamma"), ("ring_h_nm", "pupil.ring_h"),
                                                   ("sigma_nm", "psf.sigma"))},
                       **{k + "_crlb": crlb(n) for k, n in (("n_med", "optics.n_med"), ("na", "pupil.na"),
                                                            ("gamma", "pupil.gamma"), ("ring_h_nm", "pupil.ring_h"),
                                                            ("sigma_nm", "psf.sigma"))},
                       minutes=round((time.perf_counter() - t0) / 60, 2))
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
