"""The bead benchmark with each solver of M7, side by side (plan §9, M7).

    python benchmarks/uipsf/synthetic/solvers.py dim --beads 6 --seeds 0,1,2 \\
        --out benchmarks/uipsf/results/solvers.jsonl
    python benchmarks/uipsf/synthetic/solvers.py dim --solvers lm,gnl --scale pupil.phase=10
    python benchmarks/uipsf/synthetic/solvers.py dim --seeds 0,1,2 --device mps     # default: the CPU

Each row is one fit (`beads.run`, the default two-stage `fit_beads`): time, iterations, Gauss–Newton
rebuilds, the final objective and the review's accuracy metrics.  The solvers:
- ``lm``: `LevenbergMarquardt`, the reference;
- ``gnl``: `GaussNewtonLBFGS` (refresh 10, memory 10), candidate D;
- ``chord``: the same with memory 0, candidate B.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform

import torch
from beads import run

from uipsf.solve import GaussNewtonCG, LevenbergMarquardt
from uipsf.solve.lbfgs import GaussNewtonLBFGS

SOLVERS = {
    "lm": lambda: LevenbergMarquardt(),
    "gnl": lambda: GaussNewtonLBFGS(),
    "gnl20": lambda: GaussNewtonLBFGS(refresh=20, memory=20),
    "chord": lambda: GaussNewtonLBFGS(memory=0),
    "cg": lambda: GaussNewtonCG(),
    "cg-untraced": lambda: GaussNewtonCG(trace=False),
}


def jacobians(result) -> int:
    """Normal-equation builds over all stages: the expensive part (one per stage start, then one per
    accepted LM step or per L-BFGS rebuild)."""
    h = result.history
    return sum(1 for e in h if e["iteration"] == 0) + sum(
        1 for e in h if e.get("rebuild") or (e.get("accepted") and "alpha" not in e and "cg" not in e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=["dim"])
    ap.add_argument("--solvers", default="lm,gnl,chord")
    ap.add_argument("--beads", type=int, default=6)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--scale", action="append", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scale = {k: float(v) for k, v in (s.split("=") for s in args.scale)}
    for name in args.scenarios:
        for seed in (int(s) for s in args.seeds.split(",")):
            for solver in args.solvers.split(","):
                row, result, _ = run(name, scale, args.beads, seed, solver=SOLVERS[solver](), n_max=args.n_max,
                                   device=args.device)
                row.update(solver=solver, objective=result.objective, jacobians=jacobians(result),
                           jacobian="linearised", products=getattr(result, "products", None),
                           n_max=args.n_max, device=args.device, threads=torch.get_num_threads(),
                           system=platform.system(), torch=torch.__version__)
                row = {k: (round(v, 5) if isinstance(v, float) and math.isfinite(v) else v) for k, v in row.items()}
                print(json.dumps(row), flush=True)
                if args.out:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
