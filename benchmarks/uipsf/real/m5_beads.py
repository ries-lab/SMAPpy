"""The consensus bead set of M5's bead stacks: which beads every study learns from, chosen once, with reasons.

    python benchmarks/uipsf/real/m5_beads.py ACQUISITION [...] [--dz 100] [--memory 0.7]
        writes benchmarks/uipsf/results/m5_beadset.json and m5_beadset_scores.jsonl

The studies so far included every bright bead in focus (`m5_media.select`), overriding uiPSF's rejection
so that models compared on the same beads; the user's worry (31 October 2026) is that outliers -- pairs,
clumps, specks, a dim neighbour in the ROI, a bead that moved -- contaminate what is learnt (600 nm oil 3D:
18 beads among 333 dim objects, σ 51 nm and an asymmetric magnitude of its own).  So the beads are chosen
once, and every script takes them from `m5_media.BEADSET`:

1. *isolated and not too dim*: `bead_rois`, as the calibration finds beads;
2. *in focus*: within `m5_media.FOCUS_NM` of the stack's centre, and not clutter: at least `m5_media.BRIGHT`
   (0.2) of the brightest twenty's median, as before -- the median below is of these;
3. *single*: photons (a `APERTURE_PX` aperture at the focus) within `BRIGHT` times the median of those:
   0.5 … 2 for 100 nm beads, whose pair is about twice as bright and twice the size; 0.33 … 3 for 40 nm beads
   (the user's, 31 October 2026), whose brightness spreads with their volume and whose pair images as one
   bead -- specks and clumps cut, an outlier in shape left to the next step;
4. *like the others*: one common model (free Zernikes to order 8, σ free, ±1 µm) learnt with uiPSF's own
   rejection on (`rejection.decide`: deviance, photons, offset from the ROI's centre, masked pixels, each
   against median + k·1.4826·MAD), so a bead the model fits clearly worse than the rest is dropped -- a dim
   neighbour, clutter, a mover.

Every bead's stage-1 scores and its final fit's go to ``m5_beadset_scores.jsonl``, kept or not, so the
rule can be looked at and other combinations tried.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from m5_media import BEADSET, GRID_NA, N_IMM, N_MED, WAVELENGTH, acquisition, measure_beads, subset  # noqa: E402
from m5_media import BRIGHT as M5_BRIGHT  # noqa: E402

PIXEL = (104.7, 106.4)
#: the photons a single bead may have, times the median, by its diameter: a 40 nm bead's light goes with its
#: volume, so a 10-15 % spread in size spreads it by 30-45 %, and a pair of them (80 nm) is still one image to
#: the PSF -- only specks and clumps of three or more are cut on brightness, the rest left to the deviance;
#: a pair of 100 nm beads (200 nm) is not one image, and is cut at twice the median
BRIGHT = {40.0: (0.33, 3.0), 100.0: (0.5, 2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--bright", type=float, nargs=2, default=None,
                    help="photons, times the median; by default `BRIGHT` for the bead's size")
    ap.add_argument("--k", type=float, default=3.0, help="uiPSF's rejection: robust sds above the median")
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.7)
    ap.add_argument("--scores", default=os.path.join(HERE, "..", "results", "m5_beadset_scores.jsonl"))
    args = ap.parse_args()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    from uipsf.model.pupil import ZernikePupil
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=1, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1))
    calibrate.ZernikePupil = ZernikePupil
    beadset = json.load(open(BEADSET)) if os.path.exists(BEADSET) else {}

    for path in args.acquisitions:
        t0 = time.perf_counter()
        name, cond, files, bead_nm = acquisition(path)
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        focus, photons, near = measure_beads(rois)
        # the median of what may be beads: in the oil samples 740 of 765 objects are dim clutter, whose median
        # made every bead "too bright"; `m5_media.BRIGHT` of the brightest twenty's median cuts the clutter first
        beadlike = near & (photons >= M5_BRIGHT * np.median(np.sort(photons)[-20:]))
        median = float(np.median(photons[beadlike]))
        ratio = photons / median
        bright = args.bright or BRIGHT[bead_nm]
        single = (ratio >= bright[0]) & (ratio <= bright[1])
        candidates = [int(i) for i in np.flatnonzero(near & single)]
        records = [dict(rois.records[i]) for i in range(len(rois.records))]
        rois = subset(rois, candidates, focus, args.window)
        optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
        model = BeadSettings(zernike_order=8, magnitude="free", blur="free", blur_nm=30.0, bead_diameter_nm=bead_nm,
                             rejection_k=args.k)
        run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois)
        cal = run.cal
        table = {row.bead: row for row in cal.beads.rows}

        kept, rejected = [], []
        with open(args.scores, "a") as out:
            for i, r in enumerate(records):
                key = [int(r["stack"]), int(r["x_px"]), int(r["y_px"])]
                row = dict(acquisition=name, stack=key[0], x_px=key[1], y_px=key[2],
                           photons_ratio=round(float(ratio[i]), 3), in_focus=bool(near[i]))
                if not near[i]:
                    why = "out of focus"
                elif not beadlike[i]:
                    why = f"clutter {ratio[i]:.2f} × median"
                elif not single[i]:
                    why = f"photons {ratio[i]:.2f} × median"
                else:
                    b = table[candidates.index(i)]
                    row.update(scores={k: round(v, 4) for k, v in b.scores.items()},
                               final={k: round(float(v), 4) for k, v in b.final.items()}, state=b.state)
                    why = "" if b.state == "used" else (b.reason or b.state)
                row["why"] = why
                (rejected if why else kept).append(dict(row) if why else key)
                out.write(json.dumps(row) + "\n")
        beadset[name] = dict(kept=kept, rejected=[dict(stack=r["stack"], x=r["x_px"], y=r["y_px"], why=r["why"])
                                                  for r in rejected],
                             rule=dict(focus_nm=1000, bright=list(bright), k=args.k,
                                       thresholds={k: round(float(v), 4) for k, v in cal.beads.thresholds.items()}))
        json.dump(beadset, open(BEADSET, "w"), indent=1)
        counts = {}
        for r in rejected:
            reason = r["why"].split(" ")[0]
            counts[reason] = counts.get(reason, 0) + 1
        print(f"{name}: {len(kept)} kept of {len(records)}; out: {counts}  "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
