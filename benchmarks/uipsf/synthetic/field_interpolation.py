"""How should a field-dependent PSF be stored between field positions?  Tiles, interpolated PSFs, a PCA
basis, or interpolated Zernikes, against the exact pupil model.

    python benchmarks/uipsf/synthetic/field_interpolation.py [--scales 1 2] \\
        [--out benchmarks/uipsf/results/field_interpolation.jsonl]

`docs/uipsf/field-dependence.md`.  The truth is a vector pupil (NA 1.43, 660 nm, 110 nm pixels, σ 30 nm,
0.8 rad of Noll 6) with a nodal field over the normalised field coordinates u, v ∈ [−1, 1]: astigmatism
A2·(u² − v²) + A1·u on Noll 6 and A2·2uv + A1·v on Noll 5 (the binodal quadratic plus a linear, misalignment
term), coma C1·u + 0.02 on Noll 7 and C1·v on Noll 8, with (A2, A1, C1) = (0.15, 0.03, 0.10) rad times
``scale``.  Tilt and defocus over the field are left out: they are coordinate maps, not PSF shapes.

Each representation's PSF at 120 random field positions (13 × 13 px, z −600…600 nm in 30 nm steps) is
compared with the exact one by the expected excess χ² of a spot with the parameters held at the truth,
Σ (μ_true − μ_model)² / μ_true at 5000 photons and a background of 20 per pixel: about 1 is what one spot
just notices; it grows in proportion to the photons.  It is an upper bound on what a fit sees, since a fit
moves x, y, z to absorb part of it (`field_spline_fit.py` measures that).

Representations: one central PSF; the nearest of n × n tiles (tile centres); n × n grid PSFs blended
bilinearly (what blending spline coefficients is, the spline being linear in its samples); a PCA basis of
the 9 × 9 grid's whole stacks with its scores a degree-4 polynomial in (u, v); n × n grid Zernikes blended
bilinearly and the PSF computed from them.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import uipsf
from uipsf.export import sample_psf
from uipsf.workflows import BeadSettings, bead_model

OPTICS = dict(NA=1.43, wavelength_nm=660, n_imm=1.518, n_med=1.33, pixel_size_nm=(110.0, 110.0))
ROI, PHOTONS, BACKGROUND = 13, 5000.0, 20.0
Z = np.arange(-600, 601, 30.0)


def make_model(roi: int = ROI):
    return bead_model(uipsf.Optics(**OPTICS), BeadSettings(zernike_order=6, magnitude="fixed"), (roi, roi),
                      z_max_nm=1500, device="cpu", dtype=torch.float32)


def field_phase(model, u: float, v: float, scale: float) -> torch.Tensor:
    """The pupil's phase at field position (u, v); phase modes start at Noll 5."""
    a2, a1, c1 = 0.15 * scale, 0.03 * scale, 0.10 * scale
    p = torch.zeros(len(model.psf.pupil.phase_modes))
    p[1] = 0.8 + a2 * (u * u - v * v) + a1 * u          # Noll 6
    p[0] = a2 * 2 * u * v + a1 * v                      # Noll 5
    p[2] = c1 * u + 0.02                                # Noll 7
    p[3] = c1 * v                                       # Noll 8
    return p


def shared(model, phase: torch.Tensor) -> dict:
    return {"pupil.phase": phase, "psf.sigma": torch.tensor([30.0, 30.0]),
            "pupil.magnitude": torch.zeros(len(model.psf.pupil.magnitude_modes))}


def stack(model, phase, z=Z, roi=ROI) -> np.ndarray:
    """[nz, roi, roi] photons per photon for an emitter on the central pixel."""
    xyz = np.c_[np.zeros((len(z), 2)), z]
    return sample_psf(model, shared(model, phase), xyz, roi).double().numpy()


def excess(true, model):
    mt, mm = PHOTONS * true + BACKGROUND, PHOTONS * model + BACKGROUND
    return ((mt - mm) ** 2 / mt).sum((-1, -2))


def bilinear_weights(u, v, n):
    """The four grid nodes (i along u, j along v) of an n × n grid over [−1, 1]² and their weights."""
    fu, fv = (u + 1) * (n - 1) / 2, (v + 1) * (n - 1) / 2
    i, j = min(int(fu), n - 2), min(int(fv), n - 2)
    a, b = fu - i, fv - j
    return [(i, j, (1 - a) * (1 - b)), (i + 1, j, a * (1 - b)), (i, j + 1, (1 - a) * b), (i + 1, j + 1, a * b)]


def summary(e) -> dict:
    return dict(mean=float(e.mean()), p95=float(np.percentile(e, 95)), max=float(e.max()))


def run(scale: float, n_test: int = 120, seed: int = 1) -> list[dict]:
    model = make_model()
    test = np.random.default_rng(seed).uniform(-1, 1, (n_test, 2))
    phase = lambda u, v: field_phase(model, u, v, scale)                       # noqa: E731
    truth = np.stack([stack(model, phase(u, v)) for u, v in test])
    rows = [dict(method="one central PSF", **summary(excess(truth, stack(model, phase(0, 0)))))]
    for n in (9, 20):
        c = (np.arange(n) + 0.5) / n * 2 - 1
        e = np.array([excess(t, stack(model, phase(float(c[np.abs(c - u).argmin()]), float(c[np.abs(c - v).argmin()]))))
                      for t, (u, v) in zip(truth, test, strict=True)])
        rows.append(dict(method=f"tiles {n}x{n}, nearest", **summary(e)))
    for n in (3, 5, 7, 9):
        g = np.linspace(-1, 1, n)
        zern = {(i, j): phase(g[i], g[j]) for i in range(n) for j in range(n)}
        psfs = {k: stack(model, p) for k, p in zern.items()}
        e_psf = np.array([excess(t, sum(w * psfs[i, j] for i, j, w in bilinear_weights(u, v, n)))
                          for t, (u, v) in zip(truth, test, strict=True)])
        e_zern = np.array([excess(t, stack(model, sum(w * zern[i, j] for i, j, w in bilinear_weights(u, v, n))))
                           for t, (u, v) in zip(truth, test, strict=True)])
        worst = max(float((sum(w * zern[i, j] for i, j, w in bilinear_weights(u, v, n)) - phase(u, v)).abs().max())
                    for u, v in test)
        rows.append(dict(method=f"grid {n}x{n}, PSFs bilinear", **summary(e_psf)))
        rows.append(dict(method=f"grid {n}x{n}, Zernikes bilinear", worst_zernike_rad=worst, **summary(e_zern)))
        if n == 9:
            X = np.stack([psfs[i, j] for j in range(n) for i in range(n)]).reshape(n * n, -1)
            mu = X.mean(0)
            _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            uu, vv = np.meshgrid(g, g)
            poly = lambda u, v: np.stack([u ** a * v ** b for a in range(5) for b in range(5 - a)], -1)  # noqa: E731
            for K in (2, 4, 6, 8):
                coef, *_ = np.linalg.lstsq(poly(uu.ravel(), vv.ravel()), (X - mu) @ Vt[:K].T, rcond=None)
                fit = mu + (poly(test[:, 0], test[:, 1]) @ coef) @ Vt[:K]
                e = np.array([excess(t, f.reshape(t.shape)) for t, f in zip(truth, fit, strict=True)])
                rows.append(dict(method=f"PCA of the 9x9 grid, K={K}", **summary(e)))
            rows.append(dict(method="PCA variance shares", shares=[float(s) for s in S[:8] ** 2 / (S ** 2).sum()]))
    for r in rows:
        r.update(scale=scale, photons=PHOTONS, background=BACKGROUND, roi=ROI)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--out", default="benchmarks/uipsf/results/field_interpolation.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for scale in args.scales:
            print(f"field x{scale:g}: excess chi2 per spot, mean / 95th / max")
            for r in run(scale):
                f.write(json.dumps(r) + "\n")
                if "mean" in r:
                    worst = r.get("worst_zernike_rad")
                    extra = "" if worst is None else f"   worst Zernike {1000 * worst:.1f} mrad"
                    print(f"  {r['method']:34s} {r['mean']:8.3f} {r['p95']:8.3f} {r['max']:8.3f}{extra}")


if __name__ == "__main__":
    main()
