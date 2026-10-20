"""Where does the learnt blur come from?  The camera pixel: sampled at its centre or integrated.

    python benchmarks/uipsf/synthetic/pixels.py [--models u1 u2 u4 exact] [--photons 1e5] [--seed 0]
                                               [--out benchmarks/uipsf/results/pixels.jsonl]

Plan §10, "To investigate", 7.  The truth is a point source (no bead) with **no blur**: the vectorial
PSF sampled 16 × 16 per pixel and summed, i.e. the camera's integral over each pixel to within
a²/(12·256) of variance (2.3 nm of σ at 127 nm).  It is learnt with σ free by:
- *u1*: pixel centres (`VectorPSF(upsample=1)`, the bead calibration's default);
- *u2*, *u4*: u × u centred sub-samples, summed (the midpoint rule; misses a²/(12u²) of variance);
- *exact*: u = 2 with the sub-pixel box's sinc in the blur's transfer function before the sum, which is
  the full box exactly for a band-limited intensity (sinc(2t) = sinc(t)·cos t).
The prediction: σ² = a²/(12u²) for the u-sums, a²/12 for u1, 0 for exact.  Each learnt model is then
sampled as a spline at its own sampling (as `smappy.calibrate.pupil` does since M6) and SMAPpy's Spline 3D
fits noise-free pixelated images of the truth at random positions: does a shifted, pixelated spline fit
pixelated data without bias?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

import uipsf
from uipsf.model import ForwardModel
from uipsf.noise import Poisson
from uipsf.optics.zernike import noll_to_nm
from uipsf.simulate import parameters, template
from uipsf.workflows import fit_beads

PHASE = {6: 0.5, 7: 0.15, 11: 0.2}
OPTICS = dict(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
ROI, K, E, P, N_MAX = (17, 17), 31, 12, 64, 6
TRUTH_UP = 16
KNOTS, SPLINE_DZ, Z_RANGE, N_LOC, LOC_ROI = 27, 20.0, 700.0, 400, 13


def box_blur_bin_matrices(shape, spacing_nm, sigma_nm, factor):
    """`kernels.blur_bin_matrices` with the sub-sample's box, sinc(q·Δ), in the transfer function: after
    the ``factor``-sum, the whole pixel's box exactly (for samples that do not alias)."""
    out = []
    for L, d, s in ((shape[0], spacing_nm[1], sigma_nm[1]), (shape[1], spacing_nm[0], sigma_nm[0])):
        dev, dt = sigma_nm.device, sigma_nm.dtype
        q = torch.fft.fftfreq(L, d=d, device=dev, dtype=dt)
        h = torch.exp((s * q) ** 2 * (-2 * math.pi ** 2)) * torch.sinc(q * d)
        k = torch.arange(L, device=dev)
        diff = (k[:, None] - k[None, :]) % L
        cos = torch.cos(2 * math.pi * (diff[..., None] * k).remainder(L).to(dt) / L)
        C = cos @ h / L
        out.append(C.reshape(L // factor, factor, L).sum(1))
    return out[0], out[1]


class ExactPixelPSF(uipsf.VectorPSF):
    """The pixel integrated exactly in Fourier space (no bead kernel here)."""

    def _image(self, I, sigma, pad, dz):
        assert self.bead_radius_nm <= 0
        My, Mx = box_blur_bin_matrices(I.shape[-2:], self.spacing, sigma, self.upsample)
        return My @ I @ Mx.transpose(-1, -2)


MODELS = {"u1": (uipsf.VectorPSF, 1), "u2": (uipsf.VectorPSF, 2), "u4": (uipsf.VectorPSF, 4),
          "exact": (ExactPixelPSF, 2), "exact1": (ExactPixelPSF, 1)}


def grid():
    return uipsf.PupilGrid.build(uipsf.Optics(**OPTICS), P, device="cpu")


def make_psf(name, roi=ROI, g=None, sigma=30.0):
    cls, u = MODELS[name] if name in MODELS else (uipsf.VectorPSF, TRUTH_UP)
    g = g or grid()
    return cls(g, uipsf.ZernikePupil(g, n_max=N_MAX), roi, upsample=u, sigma_nm=(sigma, sigma))


def truth_values(psf):
    coeffs = torch.zeros(len(psf.pupil.phase_modes))
    for j, v in PHASE.items():
        coeffs[psf.pupil.phase_modes.index(j)] = v
    return {"pupil.phase": coeffs, "pupil.magnitude": torch.zeros(len(psf.pupil.magnitude_modes)),
            "psf.sigma": torch.tensor([0.0, 0.0])}


def simulate(photons, seed, field=None):
    """``field`` (Noll j, sd rad): that mode varies from bead to bead, normal with that sd (field
    dependence, as beads across a field of view see it); returns (batch, the per-bead offsets)."""
    tpsf = make_psf("truth")
    model = ForwardModel(tpsf)
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(E, 3, generator=g) - 0.5) * torch.tensor([150.0, 150.0, 200.0])
    values = dict(truth_values(tpsf), **{"emitter.xyz": xyz, "emitter.photons": torch.full((E,), photons),
                                          "emitter.background": torch.full((E,), 20.0)})
    b0 = template(E, uipsf.Batch.z_from_objective(torch.linspace(-750, 750, K)), ROI)
    p = parameters(model, b0, values).physical()
    offsets = None
    if field is not None:
        j, sd = field
        offsets = torch.randn(E, generator=torch.Generator().manual_seed(seed + 7)) * sd
        offsets -= offsets.mean()                   # the shared model's truth stays PHASE
        per = p["pupil.phase"].expand(E, -1).clone()
        per[:, tpsf.pupil.phase_modes.index(j)] += offsets
        p["pupil.phase"] = per
    with torch.no_grad():
        mu = model(p, b0).double()
    data = torch.poisson(mu, generator=torch.Generator().manual_seed(seed + 1)).float()
    return uipsf.Batch(data=data, z_nm=b0.z_nm), offsets


def describe(res, psf):
    p = res.params.physical()
    pupil = psf.pupil
    tr = truth_values(psf)["pupil.phase"]
    z = ((p["pupil.phase"] - tr).double() / res.std["pupil.phase"]).numpy()
    c = p["pupil.magnitude"].double().numpy()
    radial = sum(v * v for j, v in zip(pupil.magnitude_modes, c, strict=True) if noll_to_nm(j)[1] == 0)
    top = np.argsort(-np.abs(c))[:3]
    return dict(objective=round(float(res.objective), 1),
                sigma_nm=[round(float(s), 2) for s in p["psf.sigma"]],
                sigma_crlb_nm=[round(float(s), 2) for s in res.std["psf.sigma"]],
                phase_bias_max_crlb=round(float(np.abs(z).max()), 1),
                phase_rms_err_rad=round(float((p["pupil.phase"] - tr).pow(2).mean().sqrt()), 4),
                magnitude_rms=round(float(np.sqrt(np.mean(c ** 2))), 4),
                magnitude_radial_share=round(float(radial / max((c ** 2).sum(), 1e-30)), 2),
                magnitude_top={int(pupil.magnitude_modes[i]): round(float(c[i]), 4) for i in top},
                converged=bool(res.converged)), p


def predicted_sigma(name):
    u = MODELS[name][1]
    a = OPTICS["pixel_size_nm"]
    if name.startswith("exact"):
        return [0.0, 0.0]
    return [round(x / math.sqrt(12) / u, 1) for x in a]


# ------------------------------------------------------------------ SMAPpy's Spline 3D on pixelated data

def spline(name, p):
    """SMAPpy's single-channel spline of a learnt model at its own sampling, as `calibrate.pupil` makes it."""
    from smappy.calibrate.core import spline_coefficients
    from smappy.io.calibration import SplineCalibration
    nz = 2 * int(round(Z_RANGE / SPLINE_DZ)) + 1
    z0 = (nz - 1) / 2
    z = -(np.arange(nz) - z0) * SPLINE_DZ
    psf = make_psf(name, (KNOTS, KNOTS))
    xyz = torch.zeros(1, nz, 3)
    xyz[0, :, 2] = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        img = psf({k: p[k] for k in ("pupil.phase", "pupil.magnitude", "psf.sigma")}, xyz)[0].double().numpy()
    img = np.maximum(img, 0)
    img /= img.sum((1, 2)).max()
    return SplineCalibration(spline_coefficients(img), SPLINE_DZ, z0, x0=(KNOTS - 1) / 2, psf=img)


def localisation_truth(seed):
    """Noise-free pixelated images of the truth at random positions (±½ px, ±600 nm), 10⁴ photons, bg 20."""
    rng = np.random.default_rng(seed + 100)
    px, py = OPTICS["pixel_size_nm"]
    xyz = np.column_stack([rng.uniform(-.5, .5, N_LOC) * px, rng.uniform(-.5, .5, N_LOC) * py,
                           rng.uniform(-600, 600, N_LOC)])
    tpsf = make_psf("truth", (LOC_ROI, LOC_ROI))
    out = []
    with torch.no_grad():
        for chunk in np.array_split(np.arange(N_LOC), 8):
            out.append(tpsf(truth_values(tpsf), torch.tensor(xyz[chunk], dtype=torch.float32)[None])[0])
    return xyz, (torch.cat(out).double().numpy() * 1e4 + 20.0)


def localise(cal, rois):
    from smappy.psf import SplinePSF
    m = SplinePSF(cal)
    out = m.unpack(m.fit(np.ascontiguousarray(rois, np.float32), iterations=100))
    px, py = OPTICS["pixel_size_nm"]
    c = (LOC_ROI - 1) / 2
    return np.column_stack([(out["x_roi"] - c) * px, (out["y_roi"] - c) * py, out["z_nm"]])


def localisation_report(fit, xyz):
    d = fit - xyz
    ok = np.isfinite(d).all(1) & (np.abs(d) < 300).all(1)
    d, t = d[ok], xyz[ok]
    slope = float(np.polyfit(t[:, 2], fit[ok, 2], 1)[0])
    return dict(n=int(ok.sum()), bias_median_nm=[round(float(v), 2) for v in np.median(d, 0)],
                rms_nm=[round(float(v), 2) for v in np.sqrt((d ** 2).mean(0))],
                max_nm=[round(float(v), 2) for v in np.abs(d).max(0)], z_slope=round(slope, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["u1", "u2", "u4", "exact"])
    ap.add_argument("--photons", type=float, nargs="+", default=[1e5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spline", action="store_true")
    ap.add_argument("--field", nargs=2, type=float, action="append", default=[],
                    metavar=("NOLL", "SD"), help="a mode varying over the beads (rad sd); repeatable runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    loc = None if args.no_spline else localisation_truth(args.seed)
    if loc is not None:     # the floor: the truth's own spline, its knots integrated as the data are
        tv = truth_values(make_psf("truth"))
        print(json.dumps(dict(model="truth spline", spline3d=localisation_report(
            localise(spline("truth", tv), loc[1]), loc[0]))), flush=True)
    fields = [None] + [(int(j), sd) for j, sd in args.field] if args.field else [None]
    for photons, field in ((ph, f) for ph in args.photons for f in fields):
        batch, _ = simulate(photons, args.seed, field)
        for name in args.models:
            psf = make_psf(name)
            t = time.perf_counter()
            res = fit_beads(ForwardModel(psf), batch, Poisson())
            row, p = describe(res, psf)
            row = dict(model=name, photons=photons, seed=args.seed, field=field,
                       sigma_predicted_nm=predicted_sigma(name),
                       seconds=round(time.perf_counter() - t, 1), **row)
            if loc is not None:
                row["spline3d"] = localisation_report(localise(spline(name, p), loc[1]), loc[0])
            print(json.dumps(row), flush=True)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
