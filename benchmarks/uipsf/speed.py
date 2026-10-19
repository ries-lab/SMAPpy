"""Forward + backward time of the vectorial PSF, the unit of work of a bead fit.

    python benchmarks/uipsf/speed.py                     # default device (CUDA, MPS or CPU)
    python benchmarks/uipsf/speed.py --device cpu --out benchmarks/uipsf/results/speed.jsonl

The workload is the review's (F3): beads × 41 slices × 6 dipole fields, a 64-sample pupil, 21 × 21
pixels, gradient with respect to the Zernike phase.  The review's PyTorch prototype took 0.12 s for
6 beads on 4 CPU cores; one JSON line per size is printed (and appended to --out).
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

import uipsf


def run(device, n_beads, repeats=5, threads=None):
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    grid = uipsf.PupilGrid.build(optics, 64, device=device)
    pupil = uipsf.ZernikePupil(grid, n_max=6)
    psf = uipsf.VectorPSF(grid, pupil, (21, 21))
    p = uipsf.Params(psf.specs(), device=device).physical()
    p["psf.sigma"] = torch.tensor([40.0, 40.0], device=device)
    phase = torch.full_like(p["pupil.phase"], 0.1).requires_grad_(True)
    xyz = torch.zeros(n_beads, 41, 3, device=device)
    xyz[..., 2] = torch.linspace(-1000, 1000, 41, device=device)
    xyz[..., 0] = torch.linspace(-50, 50, n_beads, device=device)[:, None]
    weights = torch.rand(n_beads, 41, 21, 21, device=device)

    def step():
        phase.grad = None
        (psf({**p, "pupil.phase": phase}, xyz) * weights).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    step()                                              # warm-up: allocation, kernels
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        step()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--beads", default="6,20,40")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = uipsf.default_device() if args.device is None else torch.device(args.device)
    for n in (int(b) for b in args.beads.split(",")):
        row = dict(beads=n, slices=41, seconds=round(run(device, n), 4), device=device.type,
                   threads=torch.get_num_threads(), cpu=platform.processor() or platform.machine(),
                   system=platform.system(), torch=torch.__version__, uipsf=uipsf.__version__)
        print(json.dumps(row))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
