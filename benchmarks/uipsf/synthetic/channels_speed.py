"""Two channels against one: what a second channel costs (plan §9 M4).

    python benchmarks/uipsf/synthetic/channels_speed.py [--beads 6] [--device cpu] \
        [--out benchmarks/uipsf/results/channels_speed.jsonl]

The same simulated beads (41 slices of 21 × 21 px over ±1 µm, n_max 6, 64-sample pupils), once
in the main channel alone and once in a biplane pair (`tests/uipsf_tests/conftest.two_channel_beads`'s geometry).
Times one `normal_equations` (the Jacobian build, every speed path on), one `value_and_grad`, and the
whole default fit (`learn_beads`), best of ``--repeat`` for the first two.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests"))
import uipsf  # noqa: E402
from uipsf.noise import ShiftedPoisson  # noqa: E402
from uipsf.params import Params  # noqa: E402
from uipsf.solve import Problem  # noqa: E402
from uipsf.workflows import learn_beads  # noqa: E402
from uipsf_tests.conftest import two_channel_beads  # noqa: E402


def best(f, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append(time.perf_counter() - t)
    return min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    two, truth2, batch2 = two_channel_beads(optics, n=args.beads, dtype=torch.float32, P=64, roi=(21, 21),
                                            K=41, z_range=1000.0, n_max=6, photons=5000.0)
    batch2 = batch2.to(args.device)
    one = uipsf.ForwardModel(two.channels[0].psf)
    batch1 = uipsf.Batch(data=batch2.data[:, :1], z_nm=batch2.z_nm, s=batch2.s)
    rows = []
    for name, model, batch in (("one channel", one, batch1), ("two channels", two, batch2)):
        start = Params(model.specs(batch), batch.n_emitters, device=args.device)
        problem = Problem(model, ShiftedPoisson(), batch)
        ne = best(lambda problem=problem, start=start: problem.normal_equations(start), args.repeat)
        vg = best(lambda problem=problem, start=start: problem.value_and_grad(start), args.repeat)
        t = time.perf_counter()
        cal = learn_beads(model, batch)
        fit = time.perf_counter() - t
        rows.append(dict(model=name, beads=args.beads, device=args.device, shared=start.shared_vector().numel(),
                         normal_equations_s=round(ne, 3), value_and_grad_s=round(vg, 3), fit_s=round(fit, 1),
                         iterations=len(cal.result.history), stop=cal.result.stop_reason))
        print(json.dumps(rows[-1]), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
