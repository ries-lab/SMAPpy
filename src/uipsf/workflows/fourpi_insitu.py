"""Learning a 4Pi model in situ, from a movie's own molecules, started from a bead model (docs/4pi.md §27, M6).

A bead model is the instrument on the day of the beads, through a bead sample; the movie is another day,
another sample, another alignment.  Here the molecules of the movie itself teach the model, each seen once:
four boxes, one per channel, from one frame.

**The unit** is one molecule in one frame: `FourPiModel` with one observation and no mirror steps
(``n_phases = 1``), no bead kernel (`VectorPSF.for_emitters`), one photon count (`Bleaching(order=0)`) and a
background per channel (`molecule_model`).  Each molecule has its own x, y, z and cavity phase φ, so the
cavity's drift over the movie is absorbed molecule by molecule and needs no model while learning.

**What is learnt** (docs/4pi.md §27's table, D9's in-situ column for a matched index): both pupils' phase,
the coherence, the arms' balance, the channels' shares and phase offsets, the channels' maps, the blur and
the misalignment's Δz.  Held: the pupils' magnitude (the objectives' apodisation: molecules are too dim to
tell it, and it is the objectives' own), the buffer's index, and Δx, Δy -- held by a prior of
`OFFSET_SIGMA_NM`, which leaves Δz loose (D9: realigning moves Δz; the lateral direction was not stable even
from beads).

**A molecule's z has optima a fringe period apart**: φ takes up the interference's phase wherever z puts it,
and only the envelope tells the fringes apart.  So every molecule is localised from its start z and from a
period above and below it (`Z_STARTS`), each from the best of twelve phases (`fourpi.best_phases`), the best
optimum kept (`localise`).

**The rounds** (`learn_insitu`): every molecule localised on the current model, the shared parameters
held; those whose fit is poor -- a neighbour in a box, a bad cut, a fit on the wrong fringe that no start
fixed -- left out by a robust z-score of their deviance per pixel (`outliers`); then the shared parameters
and the kept molecules learnt together (the beads' solver); again on the new model (`ROUNDS`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..data import Batch
from ..model.background import Constant
from ..model.fourpi import LOWER, MAGNITUDE, FourPiModel
from ..model.intensity import Bleaching
from ..noise import ShiftedPoisson
from ..params import Gaussian, Params
from ..solve import BlockLM, FitResult, GaussNewtonLBFGS, Problem
from .fourpi import _device_batch, best_phases

#: where each molecule's z starts, in fringe periods from its start: the fringe's ambiguity (module docstring)
Z_STARTS = (0.0, -1.0, 1.0)
#: the prior's σ on the misalignment Δ (x, y, z) about the bead model's, nm: x and y held, z loose (D9)
OFFSET_SIGMA_NM = (1.0, 1.0, 100.0)
#: a molecule whose deviance per pixel is this many robust σ above the median is left out (`outliers`)
OUTLIER_Z = 4.0
ROUNDS = 2
#: the shared parameters held at the bead model's: the magnitude (both arms) and the buffer's index
HELD = MAGNITUDE + tuple(LOWER + n for n in MAGNITUDE) + ("optics.n_med", "fourpi.steps")


@dataclass
class InSituSettings:
    rounds: int = ROUNDS
    z_starts: tuple = Z_STARTS
    outlier_z: float = OUTLIER_Z
    offset_sigma_nm: tuple = OFFSET_SIGMA_NM
    #: also held, by name: e.g. ``ch1.transform`` where the channels' registration is trusted
    held: tuple = ()


@dataclass
class FourPiInSitu:
    """A 4Pi model learnt in situ: the molecule model, its shared values, the last joint fit, which molecules
    it kept, and every molecule's localisation on the learnt model (``found``: xyz, phase, photons,
    background, and ``deviance`` per pixel)."""
    model: FourPiModel
    shared: dict
    result: FitResult | None
    kept: np.ndarray
    found: dict
    rounds: list = field(default_factory=list)


def molecule_model(model: FourPiModel, roi_shape=None) -> FourPiModel:
    """``model`` (a bead model) for single molecules in single frames: point emitters on ``roi_shape`` boxes,
    one observation, no mirror steps, one photon count and a background per channel (module docstring)."""
    new = copy.copy(model)
    new.psf_up = model.psf_up.for_emitters(roi_shape)
    new.psf_lo = model.psf_lo.for_emitters(roi_shape)
    new.n_phases = 1
    new.steps_init = torch.zeros(0)
    new.intensity = Bleaching(order=0)
    new.background = Constant()
    return new


def molecule_batch(data, roi_xy, read_noise_var: float = 1.0, frame=None) -> Batch:
    """The uiPSF batch of molecules: ``data`` [E, C, Y, X] photons, ``roi_xy`` [E, C, 2] each channel's box
    centre in that channel's own pixels (x, y); one observation each at the nominal z 0."""
    data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
    return Batch(data=data[:, :, None], z_nm=torch.zeros(1), s=float(read_noise_var),
                 roi_xy=torch.as_tensor(np.asarray(roi_xy), dtype=torch.float32),
                 frame=None if frame is None else torch.as_tensor(np.asarray(frame), dtype=torch.long)[:, None])


def subset(batch: Batch, index) -> Batch:
    index = torch.as_tensor(np.asarray(index), dtype=torch.long)
    return replace(batch, data=batch.data[index], roi_xy=batch.roi_xy[index],
                   frame=None if batch.frame is None else batch.frame[index], start=None)


def _start_dict(start: dict, dtype, device) -> dict:
    """``start`` (xyz [E, 3], photons [E], background [E, C], phase [E]) as tensors."""
    return {k: torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v, dtype=dtype, device=device)
            for k, v in start.items() if k in ("xyz", "photons", "background", "phase")}


def params_for(model: FourPiModel, batch: Batch, shared: dict, start: dict) -> Params:
    """The molecules' parameters: ``shared`` (by name, those the molecule model has, of its shape) and each
    molecule's ``start``."""
    dt, dev = batch.data.dtype, batch.data.device
    s = _start_dict(start, dt, dev)
    specs = model.specs(batch, s)
    shapes = {sp.name: sp.shape for sp in specs if sp.scope == "shared"}
    values = {k: torch.as_tensor(v, dtype=dt, device=dev) for k, v in shared.items()
              if k in shapes and tuple(torch.as_tensor(v).shape) == tuple(shapes[k])}
    values.update({"emitter.xyz": s["xyz"], "emitter.phase": s.get("phase", torch.zeros(batch.n_emitters)),
                   "emitter.photons": s["photons"], "emitter.background": s["background"]})
    return Params(specs, batch.n_emitters, dtype=dt, device=dev, values=values)


def _held_shared(params: Params) -> Params:
    return params.fix(*[n for n, s in params.specs.items() if s.scope == "shared"])


def localise(model: FourPiModel, shared: dict, batch: Batch, start: dict, period_nm: float, noise=None,
             z_starts=Z_STARTS, chunk_size=None, cancel=None) -> dict:
    """Every molecule localised on the fixed ``shared`` parameters from each of ``z_starts`` (fringe periods
    from its start z), each from its best of twelve phases, the best optimum kept: xyz, phase, photons,
    background [numpy], ``objective`` and ``deviance`` (mean per pixel)."""
    noise = noise if noise is not None else ShiftedPoisson()
    batch = _device_batch(batch, model)
    best = None
    for k in z_starts:
        s = dict(start)
        xyz = np.array(start["xyz"], float, copy=True)
        xyz[:, 2] += float(k) * period_nm
        s["xyz"] = xyz
        p = _held_shared(params_for(model, batch, shared, s))
        problem = Problem(model, noise, batch, chunk_size)
        p = best_phases(model, batch, p, noise)
        fit = BlockLM().solve(problem, p, cancel=cancel)
        obj = problem.objectives(fit.params).numpy()
        phys = {n: v.detach().cpu().numpy() for n, v in fit.params.physical().items()}
        here = {"xyz": phys["emitter.xyz"], "phase": np.angle(np.exp(1j * phys["emitter.phase"])),
                "photons": phys["emitter.photons"], "background": phys["emitter.background"], "objective": obj}
        if best is None:
            best = here
            continue
        better = obj < best["objective"]
        best = {n: np.where(better.reshape(-1, *[1] * (v.ndim - 1)), here[n], v) for n, v in best.items()}
    p = _held_shared(params_for(model, batch, shared, best))
    D = Problem(model, noise, batch, chunk_size).deviance(p)
    best["deviance"] = D.flatten(1).mean(1).numpy()
    return best


def outliers(deviance, z: float = OUTLIER_Z) -> np.ndarray:
    """Which molecules fit poorly: deviance per pixel more than ``z`` robust σ (1.4826 MAD) above the median.
    One-sided: a molecule that fits better than most is no problem."""
    d = np.asarray(deviance, float)
    med = np.median(d)
    sd = 1.4826 * np.median(np.abs(d - med)) or 1e-12
    return (d - med) / sd > z


def learning_params(params: Params, settings: InSituSettings, offset=None) -> Params:
    """``params`` with the shared parameters that in situ learns free, the rest held (`HELD`, the settings'
    own), and the misalignment's prior (`OFFSET_SIGMA_NM`) about ``offset``, the bead model's -- never about
    a round's own result, or the rounds walk away from it."""
    specs = dict(params.specs)
    phys = params.physical()
    if "fourpi.offset" in specs:
        sigma = torch.as_tensor(settings.offset_sigma_nm, dtype=phys["fourpi.offset"].dtype,
                                device=phys["fourpi.offset"].device)
        specs["fourpi.offset"] = replace(specs["fourpi.offset"],
                                         prior=Gaussian(phys["fourpi.offset"].detach().clone() if offset is None else
                                                        torch.as_tensor(offset, dtype=sigma.dtype,
                                                                        device=sigma.device), sigma))
    out = Params(list(specs.values()), params.n_emitters, dtype=phys["emitter.xyz"].dtype,
                 device=phys["emitter.xyz"].device, values=phys)
    return out.free(*[n for n, s in out.specs.items() if s.scope == "shared"]).fix(
        *[n for n in HELD + tuple(settings.held) if n in out.specs])


def learn_insitu(bead_model: FourPiModel, bead_shared: dict, batch: Batch, start: dict, period_nm: float,
                 noise=None, settings: InSituSettings | None = None, solver=None, chunk_size=None,
                 progress=None, cancel=None) -> FourPiInSitu:
    """A 4Pi model learnt from the molecules in ``batch`` (`molecule_batch`), started from ``bead_model`` and
    its learnt ``bead_shared`` values; ``start`` each molecule's xyz (nm, from its channel-0 box centre: the
    fitter's), photons, backgrounds [E, C] and phase (any: it is searched); ``period_nm`` the fringe's period
    (module docstring).  ``progress`` gets the solver's entries with ``round`` and ``stage`` added."""
    settings = settings if settings is not None else InSituSettings()
    noise = noise if noise is not None else ShiftedPoisson()
    solver = solver if solver is not None else GaussNewtonLBFGS()
    model = molecule_model(bead_model, tuple(batch.data.shape[-2:]))
    batch = _device_batch(batch, model)
    shared = {k: torch.as_tensor(v).detach() for k, v in bead_shared.items()}
    start = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in start.items()}
    E = batch.n_emitters
    rounds, result, kept, found = [], None, np.arange(E), None

    def tell(stage, r):
        def inner(entry):
            if progress:
                progress(dict(entry, round=r, stage=stage))
        return inner

    for r in range(settings.rounds):
        tell("localise", r)({"iteration": 0})
        found = localise(model, shared, batch, start, period_nm, noise, settings.z_starts, chunk_size, cancel)
        bad = outliers(found["deviance"], settings.outlier_z)
        kept = np.flatnonzero(~bad)
        sub = subset(batch, kept)
        p = learning_params(params_for(model, sub, shared, {k: v[kept] for k, v in found.items()}), settings,
                            bead_shared.get("fourpi.offset"))
        result = solver.solve(Problem(model, noise, sub, chunk_size), p, progress=tell("learn", r), cancel=cancel)
        specs = result.params.specs
        shared = {n: v.detach() for n, v in result.params.physical().items() if specs[n].scope == "shared"}
        rounds.append({"round": r, "kept": len(kept), "left_out": int(bad.sum()),
                       "median_deviance": float(np.median(found["deviance"])),
                       "objective": float(result.objective), "stop": result.stop_reason})
        if result.stop_reason == "cancelled":
            break
    found = localise(model, shared, batch, start, period_nm, noise, settings.z_starts, chunk_size, cancel)
    return FourPiInSitu(model, shared, result, kept, found, rounds)


def calibration(fit: FourPiInSitu, **kwargs):
    """The in-situ model as SMAPpy's `FourPiCalibration`, as the beads' (`workflows.fourpi.calibration`): the
    same export, so the same phase convention."""
    from .fourpi import calibration as export
    params = {"method": "uiPSF v2 4Pi in-situ model", "molecules": int(len(fit.kept)), "rounds": fit.rounds}
    params.update(kwargs.pop("parameters", {}))
    return export((fit.model, fit.shared), parameters=params, **kwargs)
