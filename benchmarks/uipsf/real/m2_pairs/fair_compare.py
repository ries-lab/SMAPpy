"""Learnt molecule-stack models scored on ONE fixed set of molecules: a fair likelihood comparison.

    python fair_compare.py OUTDIR NAME [NAME ...] [--soft] [--set ab|nbx4] [--device cpu]

Each learning run rejects molecules between its stages and rounds, so its own objective sums over its own
molecules and cannot be compared with another run's.  Here the molecules are detected and selected once
(`molecule_rois`, every one that passes, no draw; ``--soft`` with soft_select.py's selection), and every saved
model ``OUTDIR/<NAME>_3dcal.h5`` -- its forward model with the coverslip geometry and its learnt shared values
(pupil, σ, height, each sweep's focus, the focal plane's slope) -- localises them all again with x, y, photons and
background free (z is the stage's and the geometry's), `uipsf.workflows.scoring.localise`.  The score of a model
is ½ Σ deviance over the molecules: lower is better, and differences are log-likelihood ratios (nats) between
models fitted to the same pixels.

Reported per model: the total; the total over the molecules every model fits sensibly (each one's deviance per
pixel below 3 on every model, so that a few blow-ups cannot decide it); and per 200 nm of defocus, the sum and the
difference to the first model -- where in z one model is better.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

# the data are not in git: M2_DATA names the folder that holds 261112_new's sets
ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
SETS = {"ab": "261112_MR_Intercalib2_AB_50ms_coverslip/*/*.ome.tif",
        "nbx4": "261112_MR_Intercalib2_NBX4_50ms_coverslip/*/*.ome.tif"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--set", default="ab")
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.environ.setdefault("UIPSF_DEVICE", args.device)
    if args.soft:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import soft_select  # noqa: F401
    from smappy.calibrate.molecule_stack import MoleculeStackSettings, molecule_rois
    from smappy.calibrate.pupil import load_start

    from uipsf.solve import Problem
    from uipsf.workflows.scoring import localise

    files = sorted(glob.glob(os.path.join(ROOT, SETS[args.set])))
    rois = molecule_rois(files, MoleculeStackSettings(molecules=10 ** 6, significance=3.5 if args.soft else 5.0))
    used = rois.ids("used")
    batch, noise = rois.batch()
    d = np.asarray(rois.defocus_nm)[used]
    print(f"{len(used)} molecules ({'soft' if args.soft else 'standard'} selection), {rois.sweeps} sweeps", flush=True)

    per = {}
    for name in args.names:
        learned = load_start(os.path.join(args.out, f"{name}_3dcal.h5"))
        model = learned.model
        shared = {k: torch.as_tensor(v) for k, v in learned.shared.items()}
        n_focus = shared.get("geometry.focus")
        if n_focus is not None and n_focus.numel() != rois.sweeps:
            raise ValueError(f"{name}: {n_focus.numel()} sweep foci, the data have {rois.sweeps}")
        fit, problem = localise(model, shared, batch, noise)
        dev = Problem(model, noise, problem.batch).deviance(fit.params)              # [E, C, K, Y, X]
        pixels = dev[0].numel()
        e = 0.5 * dev.reshape(dev.shape[0], -1).sum(1).double().numpy()                # ½ deviance per molecule
        per[name] = e
        print(f"{name:24s} ½Σdeviance {e.sum():12.1f}  per molecule {e.mean():7.2f}  "
              f"deviance/pixel {2 * e.mean() / pixels:5.3f}  converged {fit.converged}", flush=True)

    names = args.names
    E = np.stack([per[n] for n in names])                                              # [M, E]
    pixels = rois.units.regions.shape[-1] ** 2
    sane = (2 * E / pixels < 3).all(0)
    base = names[0]
    print(f"\nagainst {base} (nats; negative: the model fits better); sane = {sane.sum()} of {len(sane)} molecules")
    bins = np.arange(-1000, 1001, 200)
    header = "".join(f"{b:+7d}" for b in bins)
    print(f"{'model':24s} {'all':>10s} {'sane':>10s} | per 200 nm of defocus, sane: {header}")
    rows = []
    for k, n in enumerate(names):
        diff = E[k] - E[0]
        by = [float(diff[sane & (np.abs(d - b) <= 100)].sum()) for b in bins]
        print(f"{n:24s} {diff.sum():10.1f} {diff[sane].sum():10.1f} | " + "".join(f"{v:7.0f}" for v in by))
        rows.append(dict(model=n, total=float(E[k].sum()), vs_base=float(diff.sum()),
                         vs_base_sane=float(diff[sane].sum()), by_defocus=dict(zip(bins.tolist(), by, strict=False))))
    counts = [int((sane & (np.abs(d - b) <= 100)).sum()) for b in bins]
    print(f"{'molecules per bin':24s} {'':21s} | " + "".join(f"{c:7d}" for c in counts))
    tag = "soft" if args.soft else "standard"
    with open(os.path.join(args.out, f"fair_{args.set}_{tag}.json"), "w") as f:
        json.dump(dict(molecules=len(used), sane=int(sane.sum()), selection=tag, rows=rows, bins=bins.tolist(),
                       counts=counts), f, indent=1)


if __name__ == "__main__":
    main()
