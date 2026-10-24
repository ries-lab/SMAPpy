"""A rotationally symmetric magnitude, Noll 4 or a monotone centre times an edge, on the lab's real beads.

    python benchmarks/uipsf/real/edge_magnitude.py ACQUISITION [...] --pixel 127 116 \\
        [--variants zernike symmetric logistic ...] [--out benchmarks/uipsf/results/edge_magnitude.jsonl]

`synthetic/radial_magnitude.py` found a logistic edge with Noll 4 as good as or better than free radial
Zernikes on simulated beads (`docs/uipsf/architecture.md`, "A radial magnitude").  Whether it is enough
for a real objective is the question here (the user's, October 2026), with the magnitude rotationally
symmetric and falling monotonically from the centre, and whether one or two parameters more help: a ρ⁴ term
in the centre (*b*) or an asymmetric, Richards edge (*nu*).  The variants (`radial_magnitude.edge_pupil`):

- *zernike*: the default, every magnitude mode to order 8 (the baseline);
- *symmetric*: the m = 0 Zernikes only (Noll 4, 11, 22, 37), what in situ holds;
- *logistic*: Noll 4 times a logistic edge; *mono*: exp(−aρ²) times it, a ≥ 0; *mono b*, *mono nu*,
  *mono b nu*: with the ρ⁴ term, the Richards edge, or both; *logistic asym*: Noll 4, the edge and the
  m ≠ 0 Zernikes, what dropping the asymmetric magnitude costs.

Each acquisition goes through `run_calibration` as `bead_slices.py` runs it (order 8, blur free from 30 nm,
100 nm beads, every bead included, so the objectives compare), with the pupil `bead_model` builds replaced
by the variant's.  Per variant: the objective, σ, the edge's values and profile, and the slices localised
alone (`bead_slices.measure`): x, y and z systematic error within ±600 nm.  The optics are M2's
(`lab_beads.py`): NA 1.43, oil 1.516, water 1.33, 676 nm.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit, measure  # noqa: E402
from radial_magnitude import EdgePupil, edge_pupil  # noqa: E402

NA, N_IMM, N_MED, WAVELENGTH, BEAD_NM, ORDER = 1.43, 1.516, 1.33, 676.0, 100.0, 8
VARIANTS = ("zernike", "symmetric", "logistic", "mono", "mono b", "mono nu", "mono b nu", "logistic asym")


def use_pupil(variant: str):
    """Make `bead_model` build ``variant``'s pupil (the Zernike variants are its own)."""
    import uipsf.workflows.calibrate as calibrate
    from uipsf.model.pupil import ZernikePupil
    if variant in ("zernike", "symmetric"):
        calibrate.ZernikePupil = ZernikePupil
        return "free" if variant == "zernike" else "symmetric"

    class Build(ZernikePupil):
        # a class, not a function: the module also asks isinstance(pupil, ZernikePupil)
        def __new__(cls, grid, n_max, **kwargs):
            return edge_pupil(grid, n_max, variant)
    calibrate.ZernikePupil = Build
    return "free"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--pixel", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--na", type=float, default=NA, help="the nominal NA: the pupil grid's edge, ρ = 1")
    ap.add_argument("--n-med", type=float, default=N_MED)
    ap.add_argument("--order", type=int, default=ORDER, help="the phase's Zernike order (and Zernike magnitude's)")
    ap.add_argument("--blur", default="free", choices=("free", "round"), help="σx and σy, or one σ")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration
    optics = OpticsSettings(NA=args.na, wavelength_nm=WAVELENGTH, n_immersion=N_IMM, n_medium=args.n_med,
                            pixel_x_nm=args.pixel[0], pixel_y_nm=args.pixel[1])
    for acquisition in args.acquisitions:
        files = sorted(glob.glob(os.path.join(acquisition, "*", "*.ome.tif")))
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        decisions = {i: "include" for i in range(len(rois.records))}
        for variant in args.variants:
            t0 = time.perf_counter()
            magnitude = use_pupil(variant)
            model = BeadSettings(zernike_order=args.order, magnitude=magnitude, blur=args.blur, blur_nm=30.0,
                                 bead_diameter_nm=BEAD_NM)
            run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                                  decisions=decisions)
            p = run.cal.result.params.physical()
            pupil = run.cal.model.channels[0].psf.pupil
            row = dict(acquisition=os.path.basename(os.path.normpath(acquisition)), variant=variant, na=args.na,
                       n_med=args.n_med, order=args.order, blur=args.blur,
                       used=len(run.cal.beads.used), objective=round(float(run.cal.result.objective), 1),
                       sigma_nm=p["psf.sigma"].detach().cpu().double().numpy().round(2),
                       parameters=len(pupil.magnitude_modes) + sum(
                           1 for k in p if k.startswith(("pupil.centre", "pupil.edge", "pupil.saf"))))
            if isinstance(pupil, EdgePupil):
                row.update(edge=pupil.edge_values(p), profile=pupil.edge_profile(p))
                if "edge_rho" in row["edge"] and pupil.edge == "hard":
                    row["effective_na"] = round(row["edge"]["edge_rho"] * args.na, 4)
            row["centre"] = {j: round(float(v), 4) for j, v in
                             zip(pupil.magnitude_modes, p["pupil.magnitude"].detach().cpu().double().tolist(),
                                 strict=True) if j in (4, 11, 22, 37)}
            from uipsf.optics.zernike import noll_to_nm
            mag = dict(zip(pupil.magnitude_modes, p["pupil.magnitude"].detach().cpu().double().tolist(), strict=True))
            asym = [v for j, v in mag.items() if noll_to_nm(j)[1] != 0]
            row["asym_modes"] = len(asym)
            row["asym_rms"] = round(float(sum(v * v for v in asym) ** 0.5), 4) if asym else 0.0
            row["asym_by_order"] = {n: round(float(sum(v * v for j, v in mag.items() if noll_to_nm(j)[0] == n
                                                       and noll_to_nm(j)[1] != 0) ** 0.5), 4)
                                    for n in sorted({noll_to_nm(j)[0] for j in mag})}
            row.update(measure(run, args.z_range))
            row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
            emit(row, args.out)


if __name__ == "__main__":
    main()
