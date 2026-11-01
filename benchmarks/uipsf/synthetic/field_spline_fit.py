"""Field-dependent splines, end to end: spots simulated with the exact pupil model, fitted by SMAPpy's
Spline 3D with one spline, the nearest tile, or splines blended bilinearly from a grid.

    python benchmarks/uipsf/synthetic/field_spline_fit.py [--scales 1 2] [--spots 3000] \\
        [--out benchmarks/uipsf/results/field_spline_fit.jsonl]

`docs/uipsf/field-dependence.md`.  The field is `field_interpolation.py`'s.  A spline is linear in the PSF it
is made from, so blending the coefficients of the four neighbouring grid splines bilinearly *is* the spline of
the bilinearly blended PSFs (checked here to float precision).  Every grid spline keeps the model's own
normalisation (photons per photon), never one of its own, so that the blend is one.

Six field positions near the edges and corners, ``spots`` spots each: x, y uniform within ±½ px, z uniform
in ±500 nm, 5000 photons, a background of 20 per pixel, 13 × 13 px.  Splines of 21 × 21 px, 20 nm planes over
±800 nm.  Per position and spline: the bias (the median error in each 100 nm z bin, RMS over the bins) in x,
y and z, and the precision (SD within the bins).  Fits beyond 300 nm of the truth in z count as lost.  With
3000 spots the bias's noise floor is about 0.5 nm laterally and 1 nm in z, the "exact" row.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from field_interpolation import bilinear_weights, field_phase, make_model, shared
from smappy.calibrate.core import spline_coefficients
from smappy.io.calibration import SplineCalibration
from smappy.psf import SplinePSF

from uipsf.export import sample_psf

BOX, ROI, DZ, Z_RANGE = 21, 13, 20.0, 800.0
PIXEL_NM, PHOTONS, BACKGROUND = 110.0, 5000.0, 20.0
POSITIONS = [(0.9, 0.8), (-0.85, 0.6), (0.7, -0.95), (0.45, 0.35), (-0.3, -0.75), (0.95, 0.05)]


def run(scale: float, spots: int, seed: int = 3) -> list[dict]:
    model = make_model(BOX)
    half = int(Z_RANGE / DZ)
    zg = -(np.arange(2 * half + 1) - half) * DZ                 # the spline's convention: plane 0 highest
    cache = {}

    def coeff(u, v):
        key = (round(u, 6), round(v, 6))
        if key not in cache:
            xyz = np.c_[np.zeros((len(zg), 2)), zg]
            psf = sample_psf(model, shared(model, field_phase(model, u, v, scale)), xyz, BOX).double().numpy()
            cache[key] = spline_coefficients(np.maximum(psf, 0))
        return cache[key]

    def blended(u, v, n):
        g = np.linspace(-1, 1, n)
        return sum(w * coeff(g[i], g[j]) for i, j, w in bilinear_weights(u, v, n)).astype(np.float32)

    def nearest(u, v, n=20):
        c = (np.arange(n) + 0.5) / n * 2 - 1
        return coeff(float(c[np.abs(c - u).argmin()]), float(c[np.abs(c - v).argmin()]))

    a, b = (np.maximum(sample_psf(model, shared(model, field_phase(model, u, v, scale)),
                                  np.c_[np.zeros((len(zg), 2)), zg], BOX).double().numpy(), 0)
            for u, v in ((0.3, 0.2), (0.9, -0.7)))
    linearity = float(np.abs(spline_coefficients(0.4 * a + 0.6 * b)
                             - (0.4 * spline_coefficients(a) + 0.6 * spline_coefficients(b))).max())

    small = make_model(ROI)
    rng = np.random.default_rng(seed)
    rows = []
    for u, v in POSITIONS:
        xyz = np.c_[rng.uniform(-PIXEL_NM / 2, PIXEL_NM / 2, (spots, 2)), rng.uniform(-500, 500, spots)]
        mu = sample_psf(small, shared(small, field_phase(model, u, v, scale)), xyz, ROI).double().numpy()
        rois = rng.poisson(mu * PHOTONS + BACKGROUND).astype(np.float32)
        splines = {"exact": coeff(u, v), "central": coeff(0.0, 0.0), "tiles 20x20, nearest": nearest(u, v),
                   **{f"grid {n}x{n}, blended": blended(u, v, n) for n in (3, 5, 7)}}
        bins = np.digitize(xyz[:, 2], np.arange(-500, 501, 100))
        for name, c in splines.items():
            cal = SplineCalibration(np.ascontiguousarray(c, dtype=np.float32), DZ, float(half), x0=(BOX - 1) / 2)
            fit = SplinePSF(cal)
            out = fit.unpack(fit.fit(rois, 50, 0))
            err = np.c_[(out["x_roi"] - (ROI - 1) / 2) * PIXEL_NM - xyz[:, 0],
                        (out["y_roi"] - (ROI - 1) / 2) * PIXEL_NM - xyz[:, 1], out["z_nm"] - xyz[:, 2]]
            ok = np.abs(err[:, 2]) < 300
            med, sd = [], []
            for q in np.unique(bins):
                s = ok & (bins == q)
                if s.sum() >= 10:
                    med.append(np.median(err[s], 0))
                    sd.append(err[s].std(0))
            bias = np.sqrt(np.mean(np.square(med), 0))
            rows.append(dict(scale=scale, u=u, v=v, spline=name, bias_x=bias[0], bias_y=bias[1], bias_z=bias[2],
                             max_bin_bias_z=float(np.abs(np.array(med)[:, 2]).max()),
                             sd_x=float(np.mean(sd, 0)[0]), sd_z=float(np.mean(sd, 0)[2]),
                             lost=float(1 - ok.mean()), spots=spots, linearity=linearity))
    return [{k: (float(x) if isinstance(x, np.floating) else x) for k, x in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--spots", type=int, default=3000)
    ap.add_argument("--out", default="benchmarks/uipsf/results/field_spline_fit.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for scale in args.scales:
            rows = run(scale, args.spots)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            print(f"field x{scale:g} (linearity of blending: {rows[0]['linearity']:.1e}); nm, mean over positions")
            print(f"  {'spline':22s} {'bias x':>7s} {'bias y':>7s} {'bias z':>7s} {'worst z':>8s} "
                  f"{'sd x':>6s} {'sd z':>6s} {'lost':>5s}")
            for name in dict.fromkeys(r["spline"] for r in rows):
                s = [r for r in rows if r["spline"] == name]
                m = {k: np.mean([r[k] for r in s]) for k in ("bias_x", "bias_y", "bias_z", "sd_x", "sd_z", "lost")}
                print(f"  {name:22s} {m['bias_x']:7.1f} {m['bias_y']:7.1f} {m['bias_z']:7.1f} "
                      f"{max(r['bias_z'] for r in s):8.1f} {m['sd_x']:6.1f} {m['sd_z']:6.1f} {m['lost']:5.1%}")


if __name__ == "__main__":
    main()
