"""What does a free pupil magnitude absorb?  A dictionary of model errors and the magnitudes they make.

    python benchmarks/uipsf/synthetic/magnitude.py [errors…] [--seeds 0 1 2] [--n-max 6]
                                             [--out benchmarks/uipsf/results/magnitude.jsonl]

Plan §10, "To investigate", 2: fitted magnitudes in v1 were strong and often not rotationally symmetric,
which little in a microscope explains, and a flat magnitude never fitted well.  Here each candidate
error is simulated as the truth, and the data are fitted with uipsf's default model twice: with the
magnitude free (Zernikes Noll 2 … n_max) and flat (apodisation only).  Per error and fit it records the
magnitude the fit makes (the largest modes, the share of rotationally symmetric ones), the phase's
bias against its CRLB, the blur, and what the free magnitude gains over the flat one in likelihood
(nats; twice that is χ² with as many degrees of freedom as magnitude modes).

The errors (only the truth changes; the fitted model is the same in every row):
- *matched*: nothing wrong (the control);
- *NA 1.40*: the objective's effective NA below the nominal 1.43 the fit assumes;
- *soft edge*: the transmission falls smoothly over the outer 10 % of the pupil radius;
- *dichroic*: Ex and Ey transmitted differently (|t_y|² = 0.8), as by a dichroic at 45°: a real cos 2φ;
- *apodisation*: an extra factor √cos θ_imm on the pupil, a different aplanatic factor;
- *missing modes*: phase beyond the fit's n_max (coma of order n_max + 1, spherical of the next even
  order; for n_max = 6 Noll 29, 30 and 37);
- *n_med 1.40*, *n_med 1.00*: beads in a mounting medium, or dried, fitted as if in water (1.335);
- *field dependence*: vertical astigmatism varying linearly across the beads' field, ±0.15 rad.

Scenario: 12 beads, 31 slices over ±750 nm, 17 × 17 px of 127 × 116 nm, NA 1.43, λ 680 nm, oil into
water, a 64-sample pupil, Zernikes to n = 6 (``--n-max``), 5000 photons per slice and a background of 20, blur 35 nm;
the phase truth Noll 6, 7, 11 = 0.5, 0.15, 0.2 rad.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

import uipsf
from uipsf.model import ForwardModel
from uipsf.noise import Poisson
from uipsf.optics.zernike import nm_to_noll, noll_to_nm
from uipsf.simulate import parameters, template
from uipsf.workflows import fit_beads

PHASE = {6: 0.5, 7: 0.15, 11: 0.2}
ERRORS = ("matched", "NA 1.40", "soft edge", "dichroic", "apodisation", "missing modes", "field dependence",
          "n_med 1.40", "n_med 1.00")
OPTICS = dict(wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
ROI, K, E, P = (17, 17), 31, 12, 64


def _psf(NA=1.43, n_max=6, magnitude_n_max=None, n_med=None):
    optics = dict(OPTICS) if n_med is None else dict(OPTICS, n_med=n_med)
    grid = uipsf.PupilGrid.build(uipsf.Optics(NA=NA, **optics), P, device="cpu")
    pupil = uipsf.ZernikePupil(grid, n_max=n_max, magnitude_n_max=magnitude_n_max)
    return uipsf.VectorPSF(grid, pupil, ROI, sigma_nm=(30.0, 30.0))


def with_fields(psf, factor):
    """A copy of ``psf`` whose pupil fields are multiplied by ``factor`` ([F, P, P] or [P, P]): the truth
    of an amplitude error the fitted model does not have."""
    new = copy.copy(psf)
    new.fields = psf.fields * factor
    new.weight = (new.fields.abs() ** 2).sum(0)
    new.energy = new.weight.sum()
    Ft = new.fields.permute(1, 0, 2)
    new._fields_re = torch.cat([Ft.real, Ft.imag], 0).contiguous()
    new._fields_im = torch.cat([-Ft.imag, Ft.real], 0).contiguous()
    new.norm = new.spacing[0] * new.spacing[1] / (new.energy * new.dft.scale)
    return new


def missing(n_max):
    """The phase the fit of order ``n_max`` lacks: coma of order n_max + 1, spherical of the next even one."""
    n_coma = n_max + 1 if (n_max + 1) % 2 else n_max + 2
    n_sph = n_max + 2 if n_max % 2 == 0 else n_max + 1
    return {nm_to_noll(n_coma, 1): 0.1, nm_to_noll(n_coma, -1): -0.08, nm_to_noll(n_sph, 0): 0.1}


def truth(error, n_max=6):
    """(truth PSF, truth phase coefficients by Noll index, per-emitter Noll-6 offsets or None)."""
    phase = dict(PHASE)
    if error == "NA 1.40":
        return _psf(NA=1.40, n_max=n_max), phase, None
    if error.startswith("n_med"):
        return _psf(n_max=n_max, n_med=float(error.split()[1])), phase, None
    if error == "missing modes":
        extra = missing(n_max)
        phase.update(extra)
        return _psf(n_max=max(noll_to_nm(j)[0] for j in extra)), phase, None
    psf = _psf(n_max=n_max)
    rho = psf.grid.rho.real if psf.grid.rho.is_complex() else psf.grid.rho
    if error == "soft edge":
        psf = with_fields(psf, torch.clamp((1 - rho) / 0.1, 0, 1).to(psf.fields.dtype) ** 0.5)
    elif error == "dichroic":
        t = torch.tensor([1.0, 1.0, 1.0, 0.8 ** 0.5, 0.8 ** 0.5, 0.8 ** 0.5], dtype=psf.fields.real.dtype)
        psf = with_fields(psf, t[:, None, None].to(psf.fields.dtype))
    elif error == "apodisation":
        psf = with_fields(psf, torch.sqrt(psf.grid.cos_imm))
    field = None
    if error == "field dependence":
        field = torch.linspace(-0.15, 0.15, E)
    return psf, phase, field


def simulate(error, seed=0, n_max=6):
    tpsf, phase, field = truth(error, n_max)
    model = ForwardModel(tpsf)
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(E, 3, generator=g) - 0.5) * torch.tensor([150.0, 150.0, 200.0])
    coeffs = torch.zeros(len(tpsf.pupil.phase_modes))
    for j, v in phase.items():
        coeffs[tpsf.pupil.phase_modes.index(j)] = v
    values = {"pupil.phase": coeffs, "psf.sigma": torch.tensor([35.0, 35.0]), "emitter.xyz": xyz,
              "emitter.photons": torch.full((E,), 5000.0), "emitter.background": torch.full((E,), 20.0)}
    b0 = template(E, uipsf.Batch.z_from_objective(torch.linspace(-750, 750, K)), ROI)
    p = parameters(model, b0, values).physical()
    if field is not None:
        per = p["pupil.phase"].expand(E, -1).clone()
        per[:, tpsf.pupil.phase_modes.index(6)] += field
        p["pupil.phase"] = per
    with torch.no_grad():
        mu = model(p, b0).double()
    data = torch.poisson(mu, generator=torch.Generator().manual_seed(seed + 1)).float()
    return uipsf.Batch(data=data, z_nm=b0.z_nm)


def describe(result, psf):
    pupil = psf.pupil
    p = result.params.physical()
    tr = torch.zeros(len(pupil.phase_modes))
    for j, v in PHASE.items():
        tr[pupil.phase_modes.index(j)] = v
    z = ((p["pupil.phase"] - tr).double() / result.std["pupil.phase"]).numpy()
    out = dict(objective=result.objective, sigma_nm=[round(float(s), 1) for s in p["psf.sigma"]],
               phase_bias_max_crlb=round(float(np.abs(z).max()), 1),
               phase_biased={int(j): round(float(v), 1) for j, v in zip(pupil.phase_modes, z, strict=True)
                             if abs(v) > 3},
               phase_rms_err_rad=round(float((p["pupil.phase"] - tr).pow(2).mean().sqrt()), 4))
    if pupil.magnitude_modes:
        c = p["pupil.magnitude"].double().numpy()
        radial = sum(v * v for j, v in zip(pupil.magnitude_modes, c, strict=True) if noll_to_nm(j)[1] == 0)
        top = np.argsort(-np.abs(c))[:5]
        out.update(magnitude_rms=round(float(np.sqrt(np.mean(c ** 2))), 4),
                   magnitude_radial_share=round(float(radial / max((c ** 2).sum(), 1e-30)), 3),
                   magnitude_top={f"{pupil.magnitude_modes[i]} {noll_to_nm(pupil.magnitude_modes[i])}":
                                  round(float(c[i]), 4) for i in top})
    return out


def run(error, seed=0, n_max=6):
    batch = simulate(error, seed, n_max)
    row = dict(error=error, seed=seed, n_max=n_max)
    t = time.perf_counter()
    for name, mn in (("free", None), ("flat", 0)):
        psf = _psf(n_max=n_max, magnitude_n_max=mn)
        res = fit_beads(ForwardModel(psf), batch, Poisson())
        row[name] = describe(res, psf)
        row[name]["converged"] = res.converged
    row["free_gain_nats"] = round(row["flat"]["objective"] - row["free"]["objective"], 1)
    n_mag = len(_psf(n_max=n_max).pupil.magnitude_modes)
    from scipy.stats import chi2
    row["free_gain_threshold_nats"] = round(0.5 * float(chi2.isf(1e-6, n_mag)), 1)
    row["seconds"] = round(time.perf_counter() - t, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("errors", nargs="*", default=list(ERRORS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for error, seed in ((e, s) for s in args.seeds for e in args.errors):
        row = run(error, seed, args.n_max)
        row = json.loads(json.dumps(row, default=float))
        for k in ("free", "flat"):
            row[k]["objective"] = round(row[k]["objective"], 1)
        print(json.dumps(row), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
