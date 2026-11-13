"""Does the solver's chunk fit this machine?  Simulated beads of a large pupil, on the CPU or a GPU.

    python benchmarks/uipsf/memory_check.py [--device cuda|cpu|mps] [--beads 24] [--na 1.50]

Prints how this system measures memory (`uipsf.memory`: a resettable peak, a sampling meter, or none), what it
says is available, the chunk a `Problem` takes with its predicted peak, and what one Gauss-Newton matrix of
that chunk then holds: the predicted peak should be above it and below what is available.  For the PC a
calibration runs on, Windows above all, where until November 2026 nothing was measured: a rule chunked NA
1.50 beads up to 25 times larger than a card holds (`solve.problem`, `CUDA_BYTES_PER_VALUE`).
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import torch

import uipsf
from uipsf import memory
from uipsf.device import available_memory_bytes, default_device
from uipsf.noise import Poisson
from uipsf.simulate import parameters, template
from uipsf.solve import Problem
from uipsf.solve import problem as P

GB = 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None, help="default: the fastest (CUDA, Apple's GPU, the CPU)")
    ap.add_argument("--beads", type=int, default=24)
    ap.add_argument("--na", type=float, default=1.50, help="1.50: 128 pupil samples; 1.43: 64")
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else default_device()
    n, roi, K = args.beads, (27, 27), 41
    samples = 128 if args.na >= 1.45 else 64
    optics = uipsf.Optics(NA=args.na, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(106, 106))
    grid = uipsf.PupilGrid.build(optics, samples, device=dev, dtype=torch.float32)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=6), roi, sigma_nm=(30.0, 30.0)))
    xyz = (torch.rand(n, 3, dtype=torch.float64) - 0.5) * torch.tensor([200.0, 200.0, 300.0])
    values = {"emitter.xyz": xyz, "emitter.photons": torch.full((n,), 5000.0),
              "emitter.background": torch.full((n,), 10.0)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-1000, 1000, K)), roi, dtype=torch.float32)
    truth = parameters(model, b0, values, device=dev)
    with torch.no_grad():
        mu = model(truth.physical(), b0.to(dev)).cpu()
    batch = dataclasses.replace(b0, data=mu)

    meter = type(memory._METER).__name__ if memory._METER is not None else "none (a rule instead)"
    print(f"device {dev}; host meter {meter}; system memory available {(available_memory_bytes() or 0) / GB:.1f} GB")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        print(f"card: {free / GB:.1f} of {total / GB:.1f} GB free; a chunk may take {P._cuda_room(dev) / GB:.1f} GB")
    t0 = time.perf_counter()
    p = Problem(model, Poisson(), batch, device=dev)
    peak = f"{p.peak_bytes / GB:.2f} GB" if p.peak_bytes else "not measured"
    print(f"{n} beads of {K} x {roi[0]}^2, NA {args.na}: chunks of {p.chunk_size}, predicted peak {peak} "
          f"(sized in {time.perf_counter() - t0:.1f} s)", flush=True)
    t0 = time.perf_counter()
    _, held = memory.measure(lambda: p.normal_equations(truth), dev)
    held = f"{held / GB:.2f} GB" if held is not None else "unknown"
    print(f"one Gauss-Newton matrix held {held} above the start, {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
