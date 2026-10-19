"""The review's synthetic bead benchmark (plan §9 M2), fitted with uipsf's default model and solver.

    python benchmarks/uipsf/synthetic/beads.py                       # bright, dim, very dim
    python benchmarks/uipsf/synthetic/beads.py dim --scale photons=1e3 --out benchmarks/uipsf/results/beads.jsonl

The scenario is the review's (`docs/review_2026/scripts/bead_fit_scaling*.py` in ries-lab/uiPSF):
NA 1.43, λ 680 nm, oil into water, 127 × 116 nm pixels, 41 slices 50 nm apart, 21 × 21 pixel ROIs
sampled twice per pixel (v1's bin 2), a 64-sample pupil and Zernikes up to n = 6.  The truth:
- phase (Noll 5, 6, 7, 8, 11, 12) = 0.9, −0.3, 0.25, −0.2, 0.4, 0.15 rad; magnitude Noll 4 = −0.15;
- blur σ = 0.6 upsampled pixels = (38.1, 34.8) nm (v1's 0.6π);
- beads within ±1.5 pixels laterally and ±3 slices (±150 nm) in z, constant photons per slice;
- *bright*: 10 000–30 000 photons per slice, background 20–60; *dim*: 1 500–3 000 and 80–150
  (the review's low-SNR case); *very dim*: 500–1 000 and 80–150.
Photons are uipsf's physical totals; v1's are the in-focus ideal PSF summed over the ROI, about 5 %
less here, so the scenarios are as bright as the review's to within that.

The fit uses no settings beyond the scenario's optics: the default `ForwardModel` (bleaching per bead,
constant background), starting values from the data, the pupil at zero and the blur at 30 nm, and the
default procedure `uipsf.workflows.fit_beads` (Zernikes up to n = 4 first, then all).
`--scale name=factor` writes a parameter in other units (its unconstrained value multiplied by the
factor), which is what v1's ``self.weight`` does; the iterates must not change.

The metrics are the review's: the RMS error of the phase Zernikes (Noll 5–28), and the spread of the
beads' z errors (their standard deviation; the common offset is reported separately).  The review's
best tuned v1 run on the dim beads reached 0.0146 rad and 5.3 nm (8 beads); its default run 0.059 rad
and 61 nm.  Each row also gives the CRLB of both, which is what an unbiased fit can reach: for six dim
beads about 7 nm in z, so the review's 5 nm is luck of the draw, not a bound.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

import uipsf
from uipsf.model import ForwardModel
from uipsf.noise import Poisson
from uipsf.params import Params, Transform
from uipsf.simulate import parameters, simulate, template
from uipsf.solve import LevenbergMarquardt, Problem
from uipsf.workflows import fit_beads

SCENARIOS = {
    "bright": dict(photons=(1e4, 3e4), background=(20, 60)),
    "dim": dict(photons=(1.5e3, 3e3), background=(80, 150)),
    "very dim": dict(photons=(500, 1000), background=(80, 150)),
}
PHASE_TRUTH = {5: 0.9, 6: -0.3, 7: 0.25, 8: -0.2, 11: 0.4, 12: 0.15}
MAGNITUDE_TRUTH = {4: -0.15}


class Rescaled(Transform):
    """The same physical parameter in other units: physical = inner(factor · u)."""

    def __init__(self, inner, factor):
        self.inner, self.factor = inner, float(factor)

    def forward(self, x):
        return self.inner.forward(self.factor * x)

    def inverse(self, y):
        return self.inner.inverse(y) / self.factor


def rescale(params: Params, factors: dict[str, float]) -> Params:
    """``params`` with some parameters written in other units: the same physical values."""
    from uipsf.params import ParamSpec
    phys = params.physical()
    specs = []
    for s in params.specs.values():
        t = Rescaled(s.transform, factors[s.name]) if s.name in factors else s.transform
        specs.append(ParamSpec(s.name, s.shape, s.scope, s.init, t, s.prior, s.fixed))
    dtype = next(iter(params.values.values())).dtype
    return Params(specs, params.n_emitters, dtype=dtype, values=phys)


def model(optics=None, P=64, roi=(21, 21), upsample=2, n_max=6, dtype=torch.float32, device="cpu"):
    optics = optics or uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335,
                                    pixel_size_nm=(127, 116))
    grid = uipsf.PupilGrid.build(optics, P, device=device, dtype=dtype)
    pupil = uipsf.ZernikePupil(grid, n_max=n_max)
    return ForwardModel(uipsf.VectorPSF(grid, pupil, roi, upsample=upsample, sigma_nm=(30.0, 30.0)))


def make(name: str, n_beads: int = 6, seed: int = 0, dtype=torch.float32, z_step_nm: float = 50.0, **kw):
    """(model, truth Params, simulated Batch) for one scenario; slices ``z_step_nm`` apart over ±1 µm."""
    sc = SCENARIOS[name]
    m = model(dtype=dtype, **kw)
    rng = np.random.default_rng(seed)
    z = uipsf.Batch.z_from_objective(torch.linspace(-1000, 1000, round(2000 / z_step_nm) + 1))   # 41 at 50 nm
    px, py = m.pixel_size_nm
    xyz = np.c_[rng.uniform(-1.5, 1.5, n_beads) * px, rng.uniform(-1.5, 1.5, n_beads) * py,
                rng.uniform(-3, 3, n_beads) * 50.0]
    pupil = m.psf.pupil
    phase = torch.zeros(len(pupil.phase_modes))
    for j, v in PHASE_TRUTH.items():
        phase[pupil.phase_modes.index(j)] = v
    magnitude = torch.zeros(len(pupil.magnitude_modes))
    for j, v in MAGNITUDE_TRUTH.items():
        magnitude[pupil.magnitude_modes.index(j)] = v
    values = {"pupil.phase": phase, "pupil.magnitude": magnitude,
              "psf.sigma": torch.tensor([0.6 * px / 2, 0.6 * py / 2]),
              "emitter.xyz": torch.tensor(xyz, dtype=torch.float32),
              "emitter.photons": torch.tensor(rng.uniform(*sc["photons"], n_beads), dtype=torch.float32),
              "emitter.background": torch.tensor(rng.uniform(*sc["background"], n_beads), dtype=torch.float32)}
    device = m.psf.grid.device
    b0 = template(n_beads, z, m.psf.roi_shape, dtype=dtype, device=device)
    truth = parameters(m, b0, {k: v.to(device) for k, v in values.items()}, dtype=dtype, device=device)
    return m, truth, simulate(m, truth, b0, seed=seed + 1)


def metrics(result, truth, m) -> dict:
    fit, tru = result.params.physical(), truth.physical()
    dz = (fit["emitter.xyz"][:, 2] - tru["emitter.xyz"][:, 2]).cpu().double()
    dphase = (fit["pupil.phase"] - tru["pupil.phase"]).cpu().double()
    return dict(rms_phase_err_rad=float(dphase.pow(2).mean().sqrt()),
                rms_phase_crlb_rad=float(result.std["pupil.phase"].pow(2).mean().sqrt()),
                z_err_nm=float(dz.std(unbiased=False)), z_offset_nm=float(dz.mean()),
                z_crlb_nm=float(result.std["emitter.xyz"][:, 2].pow(2).mean().sqrt()),
                sigma_nm=[round(float(v), 2) for v in fit["psf.sigma"]],
                photons_rel_err=float((fit["emitter.photons"] / tru["emitter.photons"] - 1).abs().max()))


def localise(m, fitted: Params, batch):
    """Beads the fit has not seen, localised on its pupil: every shared parameter fixed."""
    # The pupil and blur only: per-slice terms (drift, fixed) belong to the fit's own slices.
    shared = {n: v for n, v in fitted.physical().items()
              if fitted.specs[n].scope == "shared" and n.startswith(("pupil.", "psf."))}
    p = Params(m.specs(batch), batch.n_emitters, dtype=batch.data.dtype, device=batch.data.device, values=shared)
    p = p.fix(*shared)
    problem = Problem(m, Poisson(), batch)
    return LevenbergMarquardt().solve(problem, p), problem


def run(name, scale=None, n_beads=6, seed=0, chunk_size=None, progress=None, heldout=2, solver=None, **kw):
    """One fit; ``heldout`` more beads are simulated alongside, not fitted, and localised afterwards."""
    m, truth_all, batch_all = make(name, n_beads + heldout, seed, **kw)
    batch, truth = batch_all.subset(slice(0, n_beads)), truth_all.subset(slice(0, n_beads))
    problem = Problem(m, Poisson(), batch, chunk_size=chunk_size)
    start = Params(m.specs(batch), n_emitters=n_beads, dtype=batch.data.dtype, device=batch.data.device)
    if scale:
        start = rescale(start, scale)
    t = time.perf_counter()
    result = fit_beads(m, batch, Poisson(), start, chunk_size=chunk_size, progress=progress, solver=solver)
    seconds = time.perf_counter() - t
    out = dict(scenario=name, scale=scale or {}, beads=n_beads, seed=seed, stop=result.stop_reason,
               converged=result.converged, iterations=result.iterations, seconds=round(seconds, 1),
               mean_deviance=float(problem.deviance(result.params).mean()), condition=result.condition,
               **metrics(result, truth, m))
    if heldout:
        held, hp = batch_all.subset(slice(n_beads, None)), truth_all.subset(slice(n_beads, None))
        loc, lprob = localise(m, result.params, held)
        dz = loc.params.physical()["emitter.xyz"][:, 2] - hp.physical()["emitter.xyz"][:, 2]
        out.update(heldout_deviance=float(lprob.deviance(loc.params).mean()),
                   heldout_z_err_nm=[round(float(v), 2) for v in dz],
                   heldout_z_crlb_nm=[round(float(v), 2) for v in loc.std["emitter.xyz"][:, 2]])
    return out, result, (m, truth, batch, problem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--scale", action="append", default=[], help="name=factor, e.g. emitter.photons=1e3")
    ap.add_argument("--beads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    scale = {k: float(v) for k, v in (s.split("=") for s in args.scale)}
    for name in args.scenarios:
        prog = (lambda e: print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                            for k, v in e.items()}), file=sys.stderr)) if args.verbose else None
        row, _, _ = run(name, scale, args.beads, args.seed, progress=prog)
        row.update(threads=torch.get_num_threads(), system=platform.system(), torch=torch.__version__,
                   uipsf=uipsf.__version__)
        row = {k: (round(v, 5) if isinstance(v, float) and math.isfinite(v) else v) for k, v in row.items()}
        print(json.dumps(row), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
