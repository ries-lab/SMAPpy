"""In situ models scored on ONE fixed set of molecules: each saved model localises the same molecules with x, y,
z (the height), photons and background free, from several heights, and keeps per molecule the best.

    python fair_insitu.py OUTDIR NAME [NAME ...]

The detection is `movie_rois` with the default settings, deterministic, so every run used the same molecules;
the runs' own objectives are summed over each round's own set and cannot be compared.  Heights start at
`STARTS_NM` because a single start falls to one side of the astigmatic ambiguity.
"""
import glob
import os
import sys

import numpy as np
import torch

# the data are not in git: M2_DATA names the folder that holds 261112_new's sets
ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
MOVIE = "261112_MR_Intercalib2_BGAAF647_50ms_30i_nucleus_1/*_MMStack_Pos0.ome.tif"
STARTS_NM = (-200.0, 200.0, 600.0, 1000.0, 1400.0)


def main():
    out, names = sys.argv[1], sys.argv[2:]
    from smappy.calibrate.insitu_run import movie_rois
    from smappy.calibrate.pupil import load_start

    from uipsf.solve import Problem
    from uipsf.workflows.scoring import localise
    rois = movie_rois(sorted(glob.glob(os.path.join(ROOT, MOVIE))))
    batch, noise = rois.batch()
    print(f"{batch.n_emitters} molecules", flush=True)
    per, zs = {}, {}
    for name in names:
        m = load_start(os.path.join(out, f"{name}_3dcal.h5"))
        model, shared = m.model, {k: torch.as_tensor(v) for k, v in m.shared.items()}
        best = np.full(batch.n_emitters, np.inf)
        z_best = np.full(batch.n_emitters, np.nan)
        for z0 in STARTS_NM:
            xyz = torch.zeros(batch.n_emitters, 3, dtype=torch.float64)
            xyz[:, 2] = z0
            fit, problem = localise(model, shared, batch, noise, start={"emitter.xyz": xyz})
            dev = Problem(model, noise, problem.batch).deviance(fit.params)
            e = 0.5 * dev.reshape(dev.shape[0], -1).sum(1).double().numpy()
            z = fit.params.physical()["emitter.xyz"][:, 2].detach().cpu().double().numpy()
            better = e < best
            best[better], z_best[better] = e[better], z[better]
        per[name], zs[name] = best, z_best
        print(f"{name:20s} ½Σdeviance {best.sum():11.1f}  heights p5/50/95 "
              f"{np.percentile(z_best, [5, 50, 95]).round(0).tolist()}", flush=True)
    base = names[0]
    for n in names[1:]:
        print(f"{n:20s} - {base}: {(per[n] - per[base]).sum():9.1f} nats")


if __name__ == "__main__":
    main()
