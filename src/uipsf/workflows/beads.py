"""Fitting a pupil model to bead stacks: low orders first, then all of them.

**Why two stages.**  With phase modes up to n = 6 and every bead's
position free, a fit from the data's rough start can stop in a local
minimum: on the review's dim beads (benchmarks/uipsf/synthetic/beads.py) it
moved every bead by +50 nm in z and +15 nm laterally and bent coma,
trefoil and the n = 6 radial modes to match, 50 nats above the true
minimum and ten standard errors from the truth.  The common position of
all beads is nearly degenerate with high-order modes: the exact high-NA
defocus and tilt are not Zernike polynomials, so modes of n ≥ 4 carry part
of them.  Fitting the modes up to n = 4 first (astigmatism, coma, trefoil,
spherical) fixes the beads' positions with few modes to trade against;
the second stage starts there and frees the rest.  On the dim beads that
reaches the minimum the truth itself converges to, in 17 iterations in
all; the first stage is cheap, with 25 shared parameters instead of 53.

v1 avoids nothing of this; it is why its results depend on the starting
weights (review §E).  ``first_order=None`` skips the first stage.

`learn_beads` adds outlier rejection between the two stages and returns a
`Calibration` with a table of every bead (`rejection.py`, plan §5.7).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from ..device import on_model_device
from ..model.channels import prefix
from ..model.pupil import ZernikePupil
from ..noise import ShiftedPoisson
from ..params import Params, ParamSpec
from ..solve import FitResult, GaussNewtonLBFGS, LevenbergMarquardt, Problem
from .rejection import BeadTable, RejectionSettings, decide, held_out
from .scoring import BeadScores, merged, score_beads, shared_values


def _truncated(model, n_max):
    """``model`` with every channel's pupil cut to the modes up to ``n_max``."""
    low = copy.copy(model)
    low.channels = []
    for ch in model.channels:
        ch = copy.copy(ch)
        ch.psf = copy.copy(ch.psf)
        ch.psf.pupil = ch.psf.pupil.truncated(n_max)
        low.channels.append(ch)
    return low


# Defaults decided in M7 (plan §9, docs/uipsf/architecture.md):
# - the solver: GaussNewtonLBFGS reached the best optimum in every benchmark, 1.7–12× faster than LM,
#   which stopped in a worse minimum on 2 of 9 seeded synthetic fits; Zenodo 19 beads in 88 s;
# - slices at most 50 nm apart: dim beads acquired at 10 nm lose phase precision only as the CRLB says
#   (0.0047 rad at 10 nm, 0.0077 at 50 nm, no bias) for 6× less time (143 s against 24 s).
MAX_DZ_NM = 50.0
# Stage 1 is a start for stage 2, on a model without the higher modes: converging it to 10⁻³ nats buys
# nothing.  Stopping when a step gains < 1 nat or moves no parameter by 0.1 of its standard error saved
# the ~26 s (two rebuilds and failing line searches at an unchanged objective) that ended stage 1 on
# Zenodo (plan §9 M7).
STAGE1_TOL, STAGE1_XTOL = 1.0, 0.1
LOCALISE_ITERATIONS = 10


def first_stage(model, batch, noise, start: Params, first_order: int | None, solver, chunk_size=None,
                progress=None, cancel=None, localise_first: bool = False):
    """Stage 1: the modes up to ``first_order``.  (Params of the full model with those coefficients
    and zeros above, the stage's FitResult); (None, None) when the pupil has no higher modes.

    ``localise_first``: before stage 1, every bead is fitted alone on the starting pupil (the shared
    parameters fixed), which needs only the emitter block of the Jacobian, so that stage 1 starts
    from beads that already sit where that pupil puts them (plan §9 M7)."""
    pupils = [ch.psf.pupil for ch in model.channels]
    staged = (first_order is not None and all(isinstance(pu, ZernikePupil) for pu in pupils)
              and any(len(pu.truncated(first_order).phase_modes) < len(pu.phase_modes) for pu in pupils))
    if not staged:
        return None, None
    dtype, device = batch.data.dtype, batch.data.device
    low = _truncated(model, first_order)
    phys = start.physical()
    values = dict(phys)
    for c, (pupil, lp) in enumerate(zip(pupils, [ch.psf.pupil for ch in low.channels], strict=True)):
        pre = prefix(c)
        values[pre + "pupil.phase"] = phys[pre + "pupil.phase"][[pupil.phase_modes.index(j) for j in lp.phase_modes]]
        values[pre + "pupil.magnitude"] = phys[pre + "pupil.magnitude"][[pupil.magnitude_modes.index(j)
                                                                         for j in lp.magnitude_modes]]
    specs = [_like(s, start.specs[s.name]) for s in low.specs(batch) if s.name in start.specs]
    p1 = Params(specs, batch.n_emitters, dtype=dtype, device=device, values=values)
    if localise_first:
        shared = [n for n in p1.specs if p1.specs[n].scope == "shared"]
        tagged = None if progress is None else (lambda e: progress(dict(e, localise=True)))
        loc = LevenbergMarquardt(max_iter=LOCALISE_ITERATIONS).solve(
            Problem(low, noise, batch, chunk_size), p1.fix(*shared), progress=tagged, cancel=cancel)
        p1 = p1.replace(emitter=loc.params.emitter_matrix())
    first = _loosened(solver).solve(Problem(low, noise, batch, chunk_size), p1, progress=progress, cancel=cancel)
    phys = first.params.physical()
    values = dict(phys)
    for c, (pupil, lp) in enumerate(zip(pupils, [ch.psf.pupil for ch in low.channels], strict=True)):
        pre = prefix(c)
        values[pre + "pupil.phase"] = pupil.embed(phys[pre + "pupil.phase"], lp.phase_modes, "phase")
        values[pre + "pupil.magnitude"] = pupil.embed(phys[pre + "pupil.magnitude"], lp.magnitude_modes,
                                                      "magnitude")
    full = Params(list(start.specs.values()), batch.n_emitters, dtype=dtype, device=device, values=values)
    return full, first


def fit_beads(model, batch, noise=None, start: Params | None = None, first_order: int | None = 4,
              solver=None, chunk_size: int | None = None, max_dz_nm: float | None = MAX_DZ_NM,
              progress=None, cancel=None, localise_first: bool = False):
    """The fitted `FitResult` for ``model`` on ``batch``; ``start`` defaults to the data's rough start.

    ``noise`` defaults to the shifted Poisson, which with the batch's s = 0
    is plain Poisson.  ``progress(entry)`` receives each iteration's entry,
    with ``"stage"`` added.  Every bead is used; `learn_beads` adds the
    rejection of outliers between the stages.  ``solver`` defaults to
    `GaussNewtonLBFGS`; slices closer than ``max_dz_nm`` are thinned
    (`Batch.thinned`; None keeps every one).
    """
    noise = noise if noise is not None else ShiftedPoisson()
    solver = solver if solver is not None else GaussNewtonLBFGS()
    batch = on_model_device(batch, model).thinned(max_dz_nm)
    if start is None:
        start = Params(model.specs(batch), batch.n_emitters, dtype=batch.data.dtype, device=batch.data.device)
    full, first = first_stage(model, batch, noise, start, first_order, solver, chunk_size,
                              _tagged(progress, 1), cancel, localise_first)
    history = [] if first is None else [dict(h, stage=1) for h in first.history]
    if first is not None and first.stop_reason == "cancelled":
        return first
    stage = 1 if first is None else 2
    result = solver.solve(Problem(model, noise, batch, chunk_size), full if full is not None else start,
                          progress=_tagged(progress, stage), cancel=cancel)
    result.history = history + [dict(h, stage=stage) for h in result.history]
    return result


@dataclass
class Calibration:
    """A learned bead model: the fit on the used beads, and the table of every bead (plan §5.7, §7.1).

    ``result.params`` holds the used beads only, in the order of ``beads.used``.
    """
    model: object
    result: FitResult
    beads: BeadTable
    scores: BeadScores                     # what the automatic decisions were made on
    heldout: list[int]
    first: FitResult | None = None         # stage 1, None after a warm start
    settings: RejectionSettings = field(default_factory=RejectionSettings)
    atlas: object = None                   # diagnostics.Atlas of the used beads
    heldout_report: object = None          # diagnostics.HeldOutReport, when beads were held out
    geometry: dict | None = None           # a split chip as SMAPpy records it (`data.pairing.Pairs.geometry`)

    def shared(self) -> dict:
        return shared_values(self.result.params)

    def psf(self, xyz_nm, roi_px=None, depth_nm=None, include_bead=False, upsample=None, channel: int = 0):
        """The learned PSF of ``channel`` at ``xyz_nm`` [N, 3]: [N, Y, X] photons per photon of the emitter
        (`export.sample_psf`)."""
        from ..export import sample_psf  # export builds on workflows, not the other way round
        return sample_psf(self.model, self.shared(), xyz_nm, roi_px, depth_nm, include_bead, upsample, channel)


def learn_beads(model, batch, noise=None, decisions: dict | None = None, warm_start: Calibration | None = None,
                rejection: RejectionSettings | None = None, heldout: float = 0.0, first_order: int | None = 4,
                solver=None, chunk_size: int | None = None, max_dz_nm: float | None = MAX_DZ_NM,
                localiser=None, progress=None, cancel=None, geometry: dict | None = None) -> Calibration:
    """Learn a pupil from bead stacks with outliers rejected, every decision explained (plan §5.7).

    1. Stage 1 (modes up to ``first_order``) on every bead not held out or
       excluded by hand.  After a ``warm_start`` (the window's
       *Recalculate*) it is skipped: the previous pupil and scores serve.
    2. Every bead of stage 1 is scored from that fit (`scoring`): no fit
       of its own.  Beads in no fit are not scored and never judged.
    3. `rejection.decide` gives each bead its state; ``decisions``
       ({bead: "include" | "exclude"}) always win.
    4. Stage 2, all modes, on the used beads, from stage 1 (a bead stage 1
       did not see starts from the data).
    5. The used beads are scored from stage 2, for the table's final values.
    6. The residual atlas of the used beads; with beads held out, their
       report, which localises them (`diagnostics.heldout_report`,
       ``localiser`` for its slice fits).  The likelihood-ratio probes cost
       a refit each and are run on request (`diagnostics.probe_terms`).

    ``heldout`` is the fraction of beads kept out of learning for that
    report, spread over the field (`rejection.held_out`); none by default,
    since a calibration should use every good bead (decided October 2026).
    ``solver`` and ``max_dz_nm`` as in `fit_beads`.  ``geometry``, for channels split on one chip,
    is kept with the calibration and its model file (`data.pairing.Pairs.geometry`).
    """
    noise = noise if noise is not None else ShiftedPoisson()
    solver = solver if solver is not None else GaussNewtonLBFGS()
    batch = on_model_device(batch, model).thinned(max_dz_nm)
    rejection = rejection if rejection is not None else RejectionSettings()
    decisions = dict(decisions or {})
    E, dtype, device = batch.n_emitters, batch.data.dtype, batch.data.device
    first = None
    if warm_start is not None:
        if len(warm_start.beads.rows) != E:
            raise ValueError("a warm start needs the same beads as the calibration it came from")
        held = list(warm_start.heldout)
        shared = warm_start.shared()
        scores = warm_start.scores
        fitted, fit = warm_start.beads.used, warm_start.result.params
    else:
        held = held_out(batch.field_xy, E, heldout)
        pool = [e for e in range(E) if e not in held and decisions.get(e) != "exclude"]
        sub = batch.subset(torch.tensor(pool))
        start = Params(model.specs(sub), len(pool), dtype=dtype, device=device)
        full, first = first_stage(model, sub, noise, start, first_order, solver, chunk_size,
                                  _tagged(progress, 1), cancel)
        if full is None:            # no higher modes: stage 1 is the whole model
            first = solver.solve(Problem(model, noise, sub, chunk_size), start, progress=_tagged(progress, 1),
                                 cancel=cancel)
            full = first.params
        shared = shared_values(full)
        scores = merged(E, [(pool, score_beads(model, full, sub, noise))])
        fitted, fit = pool, full
    table = decide(scores, rejection, held, decisions,
                   stack=None if batch.stack_id is None else batch.stack_id.tolist(),
                   field_xy=None if batch.field_xy is None else batch.field_xy.tolist())
    used = table.used
    if len(used) == 0:
        raise ValueError("no bead left to learn from")
    sub = batch.subset(torch.tensor(used))
    start = _stage2_start(model, sub, used, shared, fitted, fit, dtype, device)
    result = solver.solve(Problem(model, noise, sub, chunk_size), start, progress=_tagged(progress, 2),
                          cancel=cancel)
    result.history = ([] if first is None else [dict(h, stage=1) for h in first.history]) + \
        [dict(h, stage=2) for h in result.history]
    final = score_beads(model, result.params, sub, noise)
    for j, b in enumerate(used):
        table.rows[b].final = {n: float(getattr(final, n)[j]) for n in ("deviance", "photons")}
    from ..diagnostics import heldout_report, residual_atlas  # diagnostics build on workflows.scoring
    atlas = residual_atlas(model, result.params, sub, noise)
    report = None
    if table.ids("held out"):
        ids = table.ids("held out")
        report = heldout_report(model, shared_values(result.params), batch.subset(torch.tensor(ids)), noise, ids,
                                localiser=localiser)
    return Calibration(model, result, table, scores, held, first, rejection, atlas, report, geometry)


def _stage2_start(model, sub, used, shared, fitted, fit, dtype, device) -> Params:
    """Stage 2's start: the shared values, and each used bead's parameters from ``fit`` (whose beads are
    ``fitted``) where it was in it, else the data's rough start."""
    start = Params(model.specs(sub), len(used), dtype=dtype, device=device, values=shared)
    where = {int(b): i for i, b in enumerate(fitted)}
    rows = [j for j, b in enumerate(used) if int(b) in where]
    if not rows:
        return start
    fp, sp = fit.physical(), start.physical()
    values = dict(shared)
    for n, spec in start.specs.items():
        if spec.scope == "emitter":
            v = sp[n].clone()
            v[rows] = fp[n][[where[int(used[j])] for j in rows]].to(v.dtype)
            values[n] = v
    return Params(model.specs(sub), len(used), dtype=dtype, device=device, values=values)


def _like(spec, original):
    """``spec``'s shape with the original's transform, prior and fixed flag (the caller's choices)."""
    return ParamSpec(spec.name, spec.shape, spec.scope, spec.init, original.transform,
                     original.prior if spec.shape == original.shape else None, original.fixed)


def _loosened(solver):
    """A copy of ``solver`` with stage 1's tolerances (`STAGE1_TOL`), if it is looser than them."""
    s = copy.copy(solver)
    if hasattr(s, "tol"):
        s.tol = max(s.tol, STAGE1_TOL)
    if hasattr(s, "xtol"):
        s.xtol = max(s.xtol, STAGE1_XTOL)
    return s


def _tagged(progress, stage):
    if progress is None:
        return None
    return lambda entry: progress(dict(entry, stage=stage))
