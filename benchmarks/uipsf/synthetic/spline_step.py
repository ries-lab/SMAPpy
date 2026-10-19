"""How finely must SMAPpy's spline sample a uipsf model?  The z step, against the CRLB.

    python benchmarks/uipsf/synthetic/spline_step.py [--steps 10 20 50 100] \\
        [--out benchmarks/uipsf/results/spline_step.jsonl]

Plan §6: SMAPpy samples the learned PSF on a grid of its choosing and makes its cubic spline.  The
spline's knots are a pixel apart laterally (the format's) and dz apart in z (SMAPpy's choice).  Each
knot holds the PSF *integrated over a pixel* for an emitter at the ROI's centre (`cal.psf` integrates,
upsample 8 here).  Between knots the spline interpolates that smooth function of the emitter's position,
(PSF ∗ pixel box)(pixel − x₀): the right quantity, so a sub-pixel shift is not "wrong", only
interpolated, with an error set by the knot spacing.

Here the truth is the uipsf model itself, at random positions (±½ px, z ±600 nm).  Its expected images,
without noise, are fitted with SMAPpy's Spline 3D (``fit_cspline``) for splines of several z steps:
the fit's error on a noise-free image is the spline's systematic error alone, and it is compared with
the CRLB of the same fit at 1000 and 10 000 photons.  Poisson images give the precision too.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import uipsf
from uipsf.export import sample_psf

OPTICS = dict(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.335, pixel_size_nm=(110, 110))
PHASE = {6: 0.8, 7: 0.15, 11: 0.2, 12: -0.1}        # astigmatic 3D PSF with coma and spherical
ROI, KNOTS, Z_RANGE, N, UP = 13, 27, 800.0, 400, 8


def model():
    grid = uipsf.PupilGrid.build(uipsf.Optics(**OPTICS), 64, device="cpu")
    pupil = uipsf.ZernikePupil(grid, n_max=6)
    psf = uipsf.VectorPSF(grid, pupil, (ROI, ROI))
    phase = torch.zeros(len(pupil.phase_modes))
    for j, v in PHASE.items():
        phase[pupil.phase_modes.index(j)] = v
    shared = {"pupil.phase": phase, "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes)),
              "psf.sigma": torch.tensor([30.0, 30.0])}
    return uipsf.ForwardModel(psf), shared


def spline(fm, shared, dz):
    """SMAPpy's single-channel calibration of the model: knots at pixel centres, z0 at emitter z = 0,
    clipped and normalised to a maximum plane sum of 1 (`_single_model`, without its smoothing)."""
    from smappy.calibrate.core import spline_coefficients
    from smappy.io.calibration import SplineCalibration
    nz = 2 * int(round(Z_RANGE / dz)) + 1
    z0 = (nz - 1) / 2
    z = -(np.arange(nz) - z0) * dz                        # SMAPpy: emitter_nm = -(index - z0)·dz
    xyz = torch.zeros(nz, 3)
    xyz[:, 2] = torch.tensor(z, dtype=torch.float32)
    psf = sample_psf(fm, shared, xyz, roi_px=KNOTS, upsample=UP).double().numpy()
    psf = np.maximum(psf, 0)
    norm = float(psf.sum((1, 2)).max())
    psf /= norm
    return SplineCalibration(spline_coefficients(psf), dz, z0, x0=(KNOTS - 1) / 2, psf=psf), norm


def truth(fm, shared, seed=0):
    rng = np.random.default_rng(seed)
    px = OPTICS["pixel_size_nm"][0]
    xyz = np.column_stack([rng.uniform(-0.5, 0.5, N) * px, rng.uniform(-0.5, 0.5, N) * px,
                           rng.uniform(-600, 600, N)])
    img = sample_psf(fm, shared, torch.tensor(xyz, dtype=torch.float32), upsample=UP).double().numpy()
    return xyz, img


def fit(cal, rois):
    from smappy.psf import SplinePSF
    m = SplinePSF(cal)
    out = m.unpack(m.fit(np.ascontiguousarray(rois, np.float32), iterations=100))
    px = OPTICS["pixel_size_nm"][0]
    c = (ROI - 1) / 2
    return np.column_stack([(out["x_roi"] - c) * px, (out["y_roi"] - c) * px, out["z_nm"]]), out


def run(steps, photons=(1000.0, 10000.0), bg=10.0, seed=0):
    fm, shared = model()
    xyz, img = truth(fm, shared, seed)
    rows = []
    for dz in steps:
        cal, norm = spline(fm, shared, dz)
        row = dict(dz_nm=dz, knots=list(cal.coeff.shape[1:]))
        for N0 in photons:
            mu = N0 * img + bg
            found, out = fit(cal, mu)                      # noise-free: the spline's error alone
            err = found - xyz
            crlb = np.column_stack([out["x_err_pix"] * OPTICS["pixel_size_nm"][0],
                                    out["y_err_pix"] * OPTICS["pixel_size_nm"][0], out["z_err_nm"]])
            noisy = np.random.default_rng(seed + 1).poisson(mu)
            fn, _ = fit(cal, noisy)
            spread = np.sqrt((((fn - xyz) / crlb) ** 2).mean(0))      # ≈ 1 at the CRLB
            row[f"{int(N0)}"] = dict(
                bias_rms_nm=[round(float(v), 2) for v in np.sqrt((err ** 2).mean(0))],
                bias_max_nm=[round(float(v), 2) for v in np.abs(err).max(0)],
                bias_over_crlb_rms=[round(float(v), 3) for v in np.sqrt(((err / crlb) ** 2).mean(0))],
                crlb_median_nm=[round(float(v), 2) for v in np.median(crlb, 0)],
                noisy_rmse_over_crlb=[round(float(v), 3) for v in spread],
                photons_ratio=round(float(np.median(out["photons"] / (N0 * norm))), 4))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = run(args.steps)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
