"""Where one Levenberg–Marquardt iteration of a bead fit spends its time (plan §9, M7: profile first).

    python benchmarks/uipsf/profile_lm.py                          # dim beads, 6 of them, n_max 6, default device
    python benchmarks/uipsf/profile_lm.py --beads 20 --n-max 8 --device cpu --ops \\
        --out benchmarks/uipsf/results/profile.jsonl

One JSON line per configuration: the time of a forward pass, of a forward + backward (what one L-BFGS
evaluation costs), of the emitter and shared Jacobians (the JVPs), of the normal-equation sums and of
the Schur solve, plus the whole `normal_equations` call.  ``jac_shared`` is by JVPs only,
``jac_shared_linearised`` the pupil's columns from `ForwardModel.pupil_linearisation`, and
``jac_emitter_linearised`` the emitters' from `ForwardModel.emitter_linearisation`, which
`normal_equations` uses.  ``unit`` expresses each in forward + backward
passes, the currency in which LM and L-BFGS can be compared.  ``--ops`` adds torch.profiler's top
operators of one forward pass, by self time.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "synthetic"))

from beads import make  # noqa: E402

import uipsf  # noqa: E402
from uipsf.noise import Poisson  # noqa: E402
from uipsf.params import Params  # noqa: E402
from uipsf.solve import Problem  # noqa: E402
from uipsf.solve.lm import schur_solve  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device, repeats=3):
    fn()                                                  # warm-up: allocation, kernels
    sync(device)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        sync(device)
        best = min(best, time.perf_counter() - t)
    return best


def profile(scenario="dim", n_beads=6, n_max=6, device=None, tangents=None, repeats=3, ops=False):
    device = torch.device(device) if device else uipsf.default_device()
    m, truth, batch = make(scenario, n_beads, n_max=n_max, device=device)
    batch = batch.to(device)
    problem = Problem(m, Poisson(), batch, tangents=tangents)
    p = Params(m.specs(batch), n_emitters=n_beads, dtype=batch.data.dtype, device=device)
    pc = p.subset(slice(0, n_beads))
    us, ue = pc.shared_vector(), pc.emitter_matrix()
    Ps, Pe = us.numel(), ue.shape[1]

    def forward():
        with torch.no_grad():
            m(pc.physical(), batch)

    def forward_backward():
        v = us.detach().clone().requires_grad_(True)
        mu = m(pc.replace(shared=v, emitter=ue).physical(), batch)
        problem.noise.deviance(mu, batch.data, batch.s, batch.mask).sum().backward()

    def mu_of(us_, ue_):
        return m(pc.replace(shared=us_, emitter=ue_).physical(), batch).flatten(1)

    def jac_emitter():
        eye = torch.eye(Pe, dtype=ue.dtype, device=device)[:, None, :].expand(Pe, n_beads, Pe)
        with torch.no_grad():
            problem._columns(lambda t: torch.func.jvp(lambda v: mu_of(us, v), (ue,), (t,))[1],
                             eye, None, None)

    def jac_shared():
        eye = torch.eye(Ps, dtype=us.dtype, device=device)
        with torch.no_grad():
            problem._columns(lambda t: torch.func.jvp(lambda v: mu_of(v, ue), (us,), (t,))[1],
                             eye, None, None)

    def jac_emitter_linearised():
        eye = torch.eye(Pe, dtype=ue.dtype, device=device)[:, None, :].expand(Pe, n_beads, Pe)
        with torch.no_grad():
            problem._emitter_columns(m.emitter_linearisation, pc, batch, us, ue, eye)

    def jac_shared_linearised():
        with torch.no_grad():
            problem._shared_columns(pc, batch, us, ue, mu_of, mu_of(us, ue))

    mu, J_s, J_e = problem._jacobians(pc, batch)
    grad, w = problem.noise.derivatives(mu, batch.data, batch.s, batch.mask)
    grad, w = grad.flatten(1), w.flatten(1)

    def sums():
        wJ_s, wJ_e = J_s * w, J_e * w
        for t in (torch.einsum("iem,jem->eij", wJ_s, J_s), torch.einsum("iem,em->ei", J_s, grad),
                  torch.einsum("iem,jem->eij", wJ_s, J_e), torch.einsum("iem,jem->eij", wJ_e, J_e)):
            uipsf.device.to_cpu64(t)

    ne = problem.normal_equations(p)
    D_s, D_e = ne.H_ss.diagonal().clamp_min(1e-300), ne.H_ee.diagonal(dim1=1, dim2=2).clamp_min(1e-300)

    row = dict(scenario=scenario, beads=n_beads, slices=batch.data.shape[2], n_max=n_max, P_shared=Ps,
               P_emitter=Pe, tangents=problem.tangents, device=device.type, threads=torch.get_num_threads())
    times = dict(forward=timed(forward, device, repeats), forward_backward=timed(forward_backward, device, repeats),
                 jac_emitter=timed(jac_emitter, device, 1), jac_shared=timed(jac_shared, device, 1),
                 jac_shared_linearised=timed(jac_shared_linearised, device, 1),
                 jac_emitter_linearised=timed(jac_emitter_linearised, device, 1),
                 sums=timed(sums, device, repeats), schur=timed(lambda: schur_solve(ne, 1e-3, D_s, D_e), device),
                 normal_equations=timed(lambda: problem.normal_equations(p), device, 1))
    unit = times["forward_backward"]
    row.update({f"{k}_s": round(v, 4) for k, v in times.items()})
    row.update({f"{k}_fb": round(v / unit, 1) for k, v in times.items()})
    row["per_jvp_shared_fb"] = round(times["jac_shared"] / Ps / unit, 2)
    row["per_jvp_emitter_fb"] = round(times["jac_emitter"] / Pe / unit, 2)
    row.update(system=platform.system(), cpu=platform.processor() or platform.machine(), torch=torch.__version__)
    if ops:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprofile
        with tprofile(activities=[ProfilerActivity.CPU]) as prof:
            forward()
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
        print(table, file=sys.stderr)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="dim")
    ap.add_argument("--beads", default="6")
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tangents", type=int, default=None)
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for n in (int(b) for b in args.beads.split(",")):
        row = profile(args.scenario, n, args.n_max, args.device, args.tangents, ops=args.ops)
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
