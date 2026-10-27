"""Learning a 4Pi model from bead stacks, and its export as SMAPpy's IAB calibration (docs/4pi.md, D7, M5).

**The fit** (`learn_fourpi`) follows `beads.fit_beads`: the data's start (`model.fourpi_start`), then
each bead's phase from a circle of twelve on the starting model -- the fringe's phase at the focus is
the pupils', which the data alone do not give, and a bead started half a turn off can stop there --
then the Zernikes up to n = 4 of both objectives with everything else but the arms' misalignment and
balance (`SYMMETRIC`), then all of them.

**The export** (`calibration`) samples the model as SMAPpy's fitter will evaluate it: each channel's
image of one photon on the centre of its ROI, at three phases 2π/3 apart.  The model is
I + Re(Q e^{iφ}) in the bead's phase φ, so the mean of the three is I and (2/3) Σ μ_j e^{−iφ_j} is Q,
exactly.  The fringe's period is **measured** from Q: Q turns by k_z per nm of z (the model's own
phase slope, `fringe_slope`), never typed in as v1's was.  With the fitter's phase Φ = k z + φ, the
demodulated D = Q e^{−ikz} varies as slowly as the envelope, and A = Re D, B = −Im D
(`smappy.fourpi.load_uipsf_v1`'s convention).  `smappy.fourpi.from_volumes` normalises and splines them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..device import to_cpu64
from ..model.fourpi import LOWER, FourPiModel
from ..noise import ShiftedPoisson
from ..params import Params
from ..solve import FitResult, GaussNewtonLBFGS, Problem
from .beads import _like, _loosened, _tagged

PHASE_TRIALS = 12
#: held at their start in stage 1: at the start the two pupils are alike, the balance even and the arms
#: aligned, the symmetric point of the arms' gauge (`model.fourpi.swapped`), where these have no curvature.
#: With one phase per stack (no mirror steps) both solvers' steps there were all rejected: 2 iterations,
#: 116 393 nats against the truth's 42 886 on simulated beads; held for stage 1, the fit reached 43 086.
SYMMETRIC = ("fourpi.offset", "fourpi.balance")


@dataclass
class FourPiFit:
    """A learned 4Pi model: the fit on the beads, and what the start found."""
    model: FourPiModel
    result: FitResult
    start: dict
    first: FitResult | None = None

    def shared(self) -> dict:
        p = self.result.params
        return {n: v for n, v in p.physical().items() if p.specs[n].scope == "shared"}


def _device_batch(batch, model):
    device = model.grid.rho.device
    return batch if batch.data.device == device else batch.to(device)


def best_phases(model, batch, params: Params, noise, trials: int = PHASE_TRIALS) -> Params:
    """``params`` with each bead's phase the best of ``trials`` around the circle, the rest held."""
    problem = Problem(model, noise, batch)
    phys = params.physical()
    best, best_phase = None, None
    for phi in np.arange(trials) * 2 * math.pi / trials:
        trial = Params(list(params.specs.values()), params.n_emitters, dtype=batch.data.dtype,
                       device=batch.data.device,
                       values={**phys, "emitter.phase": torch.full_like(phys["emitter.phase"], float(phi))})
        f = problem.objectives(trial)
        if best is None:
            best, best_phase = f, torch.full_like(f, float(phi))
        else:
            better = f < best
            best, best_phase = torch.where(better, f, best), torch.where(better, float(phi), best_phase)
    phase = best_phase.to(device=batch.data.device, dtype=batch.data.dtype)
    return Params(list(params.specs.values()), params.n_emitters, dtype=batch.data.dtype,
                  device=batch.data.device, values={**phys, "emitter.phase": phase})


def _pupil_values(values, model, low, embed: bool):
    """The pupils' coefficients moved between the full model and the truncated one."""
    out = dict(values)
    for pre, full, cut in (("", model.pupils[0], low.pupils[0]), (LOWER, model.pupils[1], low.pupils[1])):
        for which in ("phase", "magnitude"):
            name = f"{pre}pupil.{which}"
            own = full.phase_modes if which == "phase" else full.magnitude_modes
            lo_modes = cut.phase_modes if which == "phase" else cut.magnitude_modes
            if embed:
                out[name] = full.embed(values[name], lo_modes, which)
            else:
                out[name] = values[name][[own.index(j) for j in lo_modes]]
    return out


def learn_fourpi(model: FourPiModel, batch, noise=None, first_order: int | None = 4, solver=None,
                 start: dict | None = None, progress=None, cancel=None) -> FourPiFit:
    """Learn ``model`` from 4Pi bead stacks ``batch`` (module docstring): [E, C, J·M, Y, X] photons,
    z-major, ``Batch.z_nm`` each slice's z repeated M times and ``Batch.roi_xy`` each channel's ROI."""
    from ..model.fourpi_start import rough_start
    noise = noise if noise is not None else ShiftedPoisson()
    solver = solver if solver is not None else GaussNewtonLBFGS()
    batch = _device_batch(batch, model)
    dtype, device = batch.data.dtype, batch.data.device
    start = start if start is not None else rough_start(batch, model)
    params = Params(model.specs(batch, start), batch.n_emitters, dtype=dtype, device=device)
    params = best_phases(model, batch, params, noise)
    low = model if first_order is None else model.with_pupils([pu.truncated(first_order) for pu in model.pupils])
    values = _pupil_values(params.physical(), model, low, embed=False)
    specs = [_like(s, params.specs[s.name]) for s in low.specs(batch, start)]
    p1 = Params(specs, batch.n_emitters, dtype=dtype, device=device, values=values).fix(*SYMMETRIC)
    first = _loosened(solver).solve(Problem(low, noise, batch), p1, progress=_tagged(progress, 1), cancel=cancel)
    if first.stop_reason == "cancelled":
        return FourPiFit(model, first, start, first)
    values = _pupil_values(first.params.physical(), model, low, embed=True)
    params = Params(list(params.specs.values()), batch.n_emitters, dtype=dtype, device=device, values=values)
    result = solver.solve(Problem(model, noise, batch), params, progress=_tagged(progress, 2), cancel=cancel)
    result.history = [dict(h, stage=1) for h in first.history] + [dict(h, stage=2) for h in result.history]
    return FourPiFit(model, result, start, first)


# ---------------------------------------------------------------------------------------------- export

def sample_iab(model: FourPiModel, shared: dict, z_nm, roi_px: int | None = None):
    """(I, Q) of every channel at heights ``z_nm`` [J]: I [C, J, Y, X] real and Q [C, J, Y, X] complex, one
    photon of the emitter on each channel's ROI centre, so that the image at the bead's phase φ is
    I + Re(Q e^{iφ}).  Sampled at three phases and demodulated (module docstring)."""
    if roi_px is not None:
        import copy
        model = copy.copy(model)
        for name in ("psf_up", "psf_lo"):
            base = getattr(model, name)
            setattr(model, name, type(base)(base.grid, base.pupil, (roi_px, roi_px), upsample=base.upsample,
                                            kind=base.kind, sigma_nm=base.sigma_init))
    dev, dt = model.grid.rho.device, model.grid.rho.dtype
    p = {k: torch.as_tensor(v, device=dev) for k, v in shared.items()}
    z = torch.as_tensor(np.asarray(z_nm, float), dtype=dt, device=dev)
    xyz = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1)
    phis = 2 * math.pi * np.arange(3) / 3
    with torch.no_grad():
        mus = [to_cpu64(model.channel_images(p, xyz, torch.full_like(z, float(phi)))).numpy() for phi in phis]
    mus = np.stack(mus)                                                          # [3, C, J, Y, X]
    I = mus.mean(0)
    Q = (2 / 3) * np.tensordot(np.exp(-1j * phis), mus, axes=(0, 0))
    return I, Q


def fringe_slope(Q, z_nm, core: float = 0.3) -> float:
    """dΦ/dz, rad per nm: how fast the model's interference turns with z, measured from Q itself.  The
    phase advance between neighbouring planes, weighted by the interference's energy in both, summed
    over channels and pixels where the interference is at least ``core`` of its strongest plane."""
    z = np.asarray(z_nm, float)
    energy = (np.abs(Q) ** 2).sum((0, 2, 3))
    keep = energy >= core * energy.max()
    adv = (np.conj(Q[:, :-1]) * Q[:, 1:]).sum((0, 2, 3))
    use = keep[:-1] & keep[1:]
    return float(np.sum(np.angle(adv[use]) * np.abs(adv[use])) / np.sum(np.abs(adv[use]) * np.diff(z)[use]))


def calibration(fit, z_range_nm: float = 600.0, dz_nm: float = 20.0, roi_px: int | None = None, **kwargs):
    """The learned model as SMAPpy's `FourPiCalibration` (`smappy.fourpi`), with its period measured.

    ``fit`` is a `FourPiFit` (or (model, shared)); knots every ``dz_nm`` over ±``z_range_nm``, on the
    pixel centres of an ROI of ``roi_px`` (default the model's).  The knots run with the index and
    z_nm against it, `FourPiCalibration`'s convention.  The channels' transforms onto channel 0 are
    kept, as are the model's per-channel numbers in ``parameters``."""
    from smappy import fourpi
    model, shared = (fit.model, fit.shared()) if isinstance(fit, FourPiFit) else fit
    n = int(round(z_range_nm / dz_nm))
    z = -np.arange(-n, n + 1) * float(dz_nm)                       # knot j at z_nm = −(j − z0)·dz
    I, Q = sample_iab(model, shared, z, roi_px)
    k = fringe_slope(Q, z)
    D = Q * np.exp(-1j * k * z)[None, :, None, None]
    C = model.n_channels
    transforms = np.stack([np.eye(3)] + [_transform(shared[f"ch{c}.transform"]) for c in range(1, C)])
    a, b, m = (to_cpu64(v).numpy() for v in model.light({kk: torch.as_tensor(v) for kk, v in shared.items()}))
    params = {"method": "uiPSF v2 4Pi bead model",
              "steps_rad": [0.0] + to_cpu64(shared["fourpi.steps"]).tolist(),
              "deltas_rad": [0.0] + to_cpu64(shared["fourpi.delta"]).tolist(),
              "coherence": to_cpu64(shared["fourpi.coherence"]).tolist(),
              "arms": np.stack([a, b], 1).tolist(), "modulation": (m / (a + b)).tolist(),
              "offset_nm": to_cpu64(shared["fourpi.offset"]).tolist(),
              "sigma_nm": to_cpu64(shared["psf.sigma"]).tolist()}
    params.update(kwargs.pop("parameters", {}))
    return fourpi.from_volumes(I, D.real, -D.imag, dz=float(dz_nm), z0=float(n), period_nm=2 * math.pi / abs(k),
                               phase_per_nm=k, transforms=transforms, pixelsize_nm=model.pixel_size_nm,
                               parameters=params, **kwargs)


def _transform(h) -> np.ndarray:
    return np.r_[to_cpu64(torch.as_tensor(h)).numpy(), 1.0].reshape(3, 3)
