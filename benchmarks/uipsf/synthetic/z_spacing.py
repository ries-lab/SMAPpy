"""How much a coarser z sampling costs (plan §9 M7, candidate F): beads acquired densely, fitted thinned.

    python benchmarks/uipsf/synthetic/z_spacing.py dim --step 10 --out benchmarks/uipsf/results/z_spacing.jsonl

The beads are simulated at ``--step`` nm over ±1 µm (the review's scenario otherwise) and every fit
uses `Batch.thinned(max_dz)`: the largest multiple of the step not above the maximum.  Each row gives
the time, the phase error and its CRLB, the z spread, and the held-out beads localised on the learned
pupil *with every slice*, so a pupil learned from fewer slices is judged on all of them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import torch
from beads import localise, make, metrics

from uipsf.noise import Poisson
from uipsf.params import Params
from uipsf.solve import GaussNewtonLBFGS, LevenbergMarquardt
from uipsf.workflows import fit_beads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="dim")
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--max-dz", default="10,20,50,100,200")
    ap.add_argument("--beads", type=int, default=6)
    ap.add_argument("--heldout", type=int, default=4)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--solver", default="gnl", choices=["gnl", "lm"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for seed in (int(s) for s in args.seeds.split(",")):
        m, truth_all, batch_all = make(args.scenario, args.beads + args.heldout, seed, z_step_nm=args.step)
        n = args.beads
        batch, truth = batch_all.subset(slice(0, n)), truth_all.subset(slice(0, n))
        held, held_truth = batch_all.subset(slice(n, None)), truth_all.subset(slice(n, None))
        for max_dz in (float(v) for v in args.max_dz.split(",")):
            b = batch.thinned(max_dz)
            solver = GaussNewtonLBFGS() if args.solver == "gnl" else LevenbergMarquardt()
            start = Params(m.specs(b), n_emitters=n, dtype=b.data.dtype)
            t = time.perf_counter()
            result = fit_beads(m, b, Poisson(), start, solver=solver)
            seconds = time.perf_counter() - t
            loc, lprob = localise(m, result.params, held)
            dz = loc.params.physical()["emitter.xyz"][:, 2] - held_truth.physical()["emitter.xyz"][:, 2]
            row = dict(scenario=args.scenario, seed=seed, step_nm=args.step, max_dz_nm=max_dz,
                       z_step_used_nm=round(b.z_step_nm, 2), slices=b.data.shape[2], beads=n, solver=args.solver,
                       stop=result.stop_reason, iterations=result.iterations, seconds=round(seconds, 1),
                       **metrics(result, truth, m),
                       heldout_deviance=float(lprob.deviance(loc.params).mean()),
                       heldout_z_err_nm=[round(float(v), 2) for v in dz],
                       heldout_z_crlb_nm=[round(float(v), 2) for v in loc.std["emitter.xyz"][:, 2]],
                       threads=torch.get_num_threads(), system=platform.system(), torch=torch.__version__)
            row = {k: (round(v, 5) if isinstance(v, float) and math.isfinite(v) else v) for k, v in row.items()}
            print(json.dumps(row), flush=True)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
