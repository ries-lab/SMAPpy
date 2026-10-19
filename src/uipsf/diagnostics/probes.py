"""Likelihood-ratio probes: does the data ask for a piece of physics the fit left out? (plan §8, M8)

Some missing physics leaves no pattern in the residuals: the rest of the
model absorbs it.  A 5 % piezo z scale on twelve beads over ±1 µm, fitted with
the z scale fixed, left the mean deviance at 1.014 (matched: 1.009) and no residual group above the
noise, yet freeing the z scale recovered 1.052 ± 0.002 and lowered the
objective by 355 nats, while the learned spherical aberration moved by
0.012 rad (tests/uipsf_tests/test_atlas.py).  The review found exactly this term among
the most damaging (review §B).

So each optional term is probed: the model with that term free, started
from the fit, run to convergence, and the drop in the objective (½
deviance, nats) compared with chance.  Twice the drop is χ² with as many
degrees of freedom as the term has parameters when the term is absent; a
probe is flagged beyond the χ² quantile of ``p`` (10⁻⁶ by default: one
false alarm in a million fits per probe).  Probes refit the whole model,
so each costs a few iterations of the full fit.

**One at a time.**  With one term missing, a wrong term can absorb a little
of it and pass its threshold too (a background ramp lent the z scale 44
nats; its own probe gained 6876).  `select_terms` therefore adds only the
largest significant term, refits, and probes again, until no term is
significant: forward selection by likelihood ratio.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from ..device import on_model_device
from ..model import BeadStackGeometry, PerSliceFactor, Planar
from ..params import Params
from ..solve import LevenbergMarquardt, Problem


@dataclass
class Probe:
    term: str
    gain: float                 # drop in the objective, nats
    dof: int                    # parameters the term adds
    threshold: float            # gain beyond which the term is flagged
    flagged: bool
    values: dict                # the term's fitted values
    suggests: str
    model: object = None        # the model with the term, and its fit
    params: Params | None = None


def _variants(model):
    out = {}
    if isinstance(model.geometry, BeadStackGeometry) and not model.geometry.z_scale:
        m = copy.copy(model)
        g = copy.copy(model.geometry)
        g.z_scale = True
        m.geometry = g
        out["z scale"] = (m, ["geometry.z_scale"], "a piezo z scale: BeadStackGeometry(z_scale=True)")
    if type(model.background).__name__ == "Constant":
        m = copy.copy(model)
        m.background = Planar(model.pixel_size_nm)
        out["planar background"] = (m, ["emitter.background_slope"], "a background ramp: background.Planar")
    if type(model.intensity).__name__ == "Bleaching":
        m = copy.copy(model)
        m.intensity = PerSliceFactor(order=model.intensity.order)
        out["per-slice photons"] = (m, ["intensity.slice_factor"], "a fluctuating laser: intensity.PerSliceFactor")
    return out


def probe_terms(model, params: Params, batch, noise, p: float = 1e-6, terms=None, solver=None) -> list[Probe]:
    """Each optional term the model lacks, probed from the fit ``params`` (every emitter of ``batch``);
    the largest gain first."""
    solver = solver or LevenbergMarquardt()
    batch = on_model_device(batch, model)
    base = Problem(model, noise, batch).objective(params)
    out = [_probe(term, variant, names, suggests, params, batch, noise, base, p, solver)
           for term, (variant, names, suggests) in _variants(model).items() if terms is None or term in terms]
    return sorted(out, key=lambda q: -q.gain)


def _probe(term, variant, names, suggests, params, batch, noise, base, p, solver) -> Probe:
    from scipy.stats import chi2
    specs = variant.specs(batch)
    have = {s.name for s in specs}
    start = Params(specs, batch.n_emitters, dtype=batch.data.dtype, device=batch.data.device,
                   values={n: v for n, v in params.physical().items() if n in have})
    keep_fixed = [n for n, s in params.specs.items() if s.fixed and n in have and n not in names]
    if keep_fixed:
        start = start.fix(*keep_fixed)
    res = solver.solve(Problem(variant, noise, batch), start)
    dof = sum(start.specs[n].size * (1 if start.specs[n].scope == "shared" else batch.n_emitters) for n in names)
    gain = base - res.objective
    threshold = 0.5 * float(chi2.isf(p, dof))
    phys = res.params.physical()
    return Probe(term, gain, dof, threshold, gain > threshold, {n: phys[n].detach() for n in names},
                 suggests, variant, res.params)


def select_terms(model, params: Params, batch, noise, p: float = 1e-6, max_terms: int = 3, solver=None):
    """Forward selection: (model, params, [Probe] of the terms added, in order)."""
    added = []
    for _ in range(max_terms):
        probes = probe_terms(model, params, batch, noise, p, solver=solver)
        if not probes or not probes[0].flagged:
            break
        best = probes[0]
        added.append(best)
        model, params = best.model, best.params
    return model, params, added
