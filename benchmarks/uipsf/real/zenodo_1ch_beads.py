"""uipsf on the ROIs uiPSF v1 fitted in the paper's `1ch_40nm_bead` example (docs/uipsf/data.md).

    python scripts/uipsf/fetch_zenodo.py results/1ch_40nm_bead/
    python benchmarks/uipsf/real/zenodo_1ch_beads.py [--beads 19] [--out benchmarks/uipsf/results/zenodo.jsonl]

The ROIs are `rois/psf_data` of v1's result file (19 beads × 39 slices × 25 × 25, already in v1's photons:
(ADU − 400) · 0.22), so ROI finding is out of the comparison and only the model and the solver differ.
The settings are v1's from the file's `params`: NA 1.43, λ 600 nm, immersion 1.516, medium 1.35,
127 × 116 nm pixels, 50 nm steps (slices 1–39 of 41), a 64-sample pupil, Zernikes to n = 8, a 25 nm
bead, no upsampling.  Where v2 differs on purpose, it keeps its defaults: photons by bleaching per bead,
not v1's free count per slice (``--intensity per-observation`` is v1's); Poisson on the photons as
they are.

``--source raw`` fits the same 19 beads cut from the raw stacks instead, in the camera's orientation and
SMAPpy's photons.  v1's ROIs are exactly transpose((ADU − 389) · 0.22) of the camera's pixels, at
`rois/cor` = (x, y) in the stacks sorted by name, slices 1–39: offset 400, then v1's shift of the data
by its 0.1 % quantile (11 ADU), and v1's ``swapxy``.  The camera's own numbers, from SMAPpy's camera
database, are 4.7 e⁻/ADU and an offset of 398.6 (docs/uipsf/data.md).  v1's deviance is then not comparable
and is not printed; v1's Zernikes are in its transposed frame.

It prints one JSON line: the fit's iterations, time and stop reason; the mean deviance of v2's fit and
of v1's own (`rois/psf_fit`) on the same pixels; and both sets of Zernikes (v1's index i is Noll i + 1,
same sign: docs/uipsf/conventions.md).  This is a first look, not the M3 comparison (held-out beads, z-bias).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from zenodo_camera import CAMERA

import uipsf
from uipsf.diagnostics import noise_transfer
from uipsf.model import Bleaching, ForwardModel, PerObservation, PerSliceFactor
from uipsf.noise import Camera, Poisson, ShiftedPoisson
from uipsf.solve import GaussNewtonLBFGS, LevenbergMarquardt, Problem
from uipsf.workflows import fit_beads

RESULT = "results/1ch_40nm_bead/psfmodel_zernike_vector_single.h5"


def load(path, n_beads=None):
    with h5py.File(path, "r") as f:
        data = f["rois/psf_data"][()]
        v1 = dict(fit=f["rois/psf_fit"][()], zernike=f["res/zernike_coeff"][()], sigma=f["res/sigma"][()],
                  params=json.loads(f.attrs["params"]))
    if n_beads:
        data, v1["fit"] = data[:n_beads], v1["fit"][:n_beads]
    return data, v1


def raw_rois(root, n_beads=None, read_noise_e=None):
    """v1's beads, cut from the raw stacks in the camera's orientation, in SMAPpy's photons."""
    import glob

    from uipsf.data.io import read_stack
    from uipsf.data.rois import Beads, extract_rois
    with h5py.File(os.path.join(root, RESULT), "r") as f:
        cor, fid = f["rois/cor"][()], f["rois/fileID"][()]
    if n_beads:
        cor, fid = cor[:n_beads], fid[:n_beads]
    files = sorted(glob.glob(os.path.join(root, "1ch_40nm_bead", "*", "*.ome.tif")))
    used = sorted(set(fid))
    camera = CAMERA if read_noise_e is None else Camera(CAMERA.conversion, CAMERA.offset, read_noise_e=read_noise_e)
    stacks = {i: read_stack(files[i], camera) for i in used}
    order = [used.index(i) for i in fid]
    beads = Beads(cor[:, 0].astype(int), cor[:, 1].astype(int), np.array(order), np.zeros(len(fid)))
    batch = extract_rois([stacks[i] for i in used], beads, 25, slices=slice(1, 40))
    return batch.data[:, 0].numpy(), float(batch.s), camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--beads", type=int, default=None)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--swap-pixels", action="store_true", help="pixel sizes (116, 127): v1 ran with swapxy")
    ap.add_argument("--source", default="v1", choices=["v1", "raw"])
    ap.add_argument("--read-noise-e", type=float, default=None,
                    help="raw: rms read noise in e- (SMAPpy's default is 1; this EMCCD's normal port reads ~21)")
    ap.add_argument("--pixels", default=None, help="x,y in nm (default 127,116)")
    ap.add_argument("--intensity", default="bleaching", choices=["bleaching", "per-slice", "per-observation"])
    ap.add_argument("--solver", default="gnl", choices=["lm", "gnl", "chord"],
                    help="LevenbergMarquardt, GaussNewtonLBFGS, or the latter with memory 0 (plan §9 M7)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    data, v1 = load(os.path.join(args.data, RESULT), args.beads)
    E, K, Y, X = data.shape
    pixels = (116.0, 127.0) if args.swap_pixels else (127.0, 116.0)
    if args.pixels:
        pixels = tuple(float(v) for v in args.pixels.split(","))
    s = 0.0
    if args.source == "raw":
        data, s, camera = raw_rois(args.data, args.beads, args.read_noise_e)
    optics = uipsf.Optics(NA=1.43, wavelength_nm=600, n_imm=1.516, n_med=1.35, pixel_size_nm=pixels)
    grid = uipsf.PupilGrid.build(optics, 64, device="cpu")
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=args.n_max), (Y, X), bead_radius_nm=25.0,
                          sigma_nm=(30.0, 30.0))
    intensity = {"bleaching": Bleaching(), "per-slice": PerSliceFactor(),
                 "per-observation": PerObservation()}[args.intensity]
    model = ForwardModel(psf, intensity=intensity)
    objective = (torch.arange(K, dtype=torch.float32) - (K - 1) / 2) * 50.0     # slices 1–39, focus in the middle
    batch = uipsf.Batch(data=torch.tensor(data)[:, None], z_nm=uipsf.Batch.z_from_objective(objective), s=s)
    noise = ShiftedPoisson() if args.source == "raw" else Poisson()
    t = time.perf_counter()
    progress = (lambda e: print({k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()},
                                flush=True)) if args.verbose else None
    solver = {"lm": LevenbergMarquardt, "gnl": GaussNewtonLBFGS,
              "chord": lambda: GaussNewtonLBFGS(memory=0)}[args.solver]()
    result = fit_beads(model, batch, noise, progress=progress, solver=solver)
    seconds = time.perf_counter() - t
    problem = Problem(model, noise, batch)
    D = problem.deviance(result.params)
    transfer = noise_transfer(batch.data, problem.expected(result.params))
    D1 = Poisson().deviance(torch.tensor(v1["fit"]).double(), torch.tensor(data).double()) \
        if args.source == "v1" else torch.tensor(float("nan"))
    p = result.params.physical()
    pupil = psf.pupil
    v1_phase = {i + 1: float(c) for i, c in enumerate(v1["zernike"][1])}
    v1_mag = {i + 1: float(c / v1["zernike"][0][0]) for i, c in enumerate(v1["zernike"][0])}
    row = dict(source=args.source, read_noise_var=s, beads=E, pixels=pixels, intensity=args.intensity, n_max=args.n_max,
               solver=args.solver, objective=result.objective, stop=result.stop_reason,
               converged=result.converged,
               iterations=result.iterations, seconds=round(seconds, 1), condition=result.condition,
               mean_deviance_v2=float(D.mean()), mean_deviance_v1=float(D1.mean()),
               noise_slope=round(transfer.slope, 3), noise_read_noise_photons=round(transfer.read_noise_photons, 2),
               sigma_nm=[round(float(s), 2) for s in p["psf.sigma"]],
               # The file stores v1's blur as an sd in pixels, in (y, x) order here (v1 ran with swapxy).
               sigma_v1_nm=[round(float(v1["sigma"][1]) * pixels[0], 2), round(float(v1["sigma"][0]) * pixels[1], 2)],
               phase={j: [round(float(c), 4), round(float(sd), 4), round(v1_phase.get(j, 0.0), 4)]
                      for j, c, sd in zip(pupil.phase_modes, p["pupil.phase"], result.std["pupil.phase"],
                                          strict=True)},
               magnitude={j: [round(float(c), 4), round(v1_mag.get(j, 0.0), 4)]
                          for j, c in zip(pupil.magnitude_modes, p["pupil.magnitude"], strict=True)},
               v1_defocus_z4=round(v1_phase[4], 4), torch=torch.__version__, threads=torch.get_num_threads())
    print(json.dumps(row), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
