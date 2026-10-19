"""Levenberg–Marquardt per emitter, for problems whose shared parameters are all fixed (localisation).

**Why.**  With the pupil fixed, the normal equations fall apart into one small
block per emitter, and the emitters are independent problems.  The block LM
of `lm.py` still treats them as one: one damping λ, one accept-or-reject,
one stopping test over the sum.  One hard emitter then raises λ for all of
them and keeps every one iterating; bead scoring localises every slice of
every bead alone (72 beads × 40 slices on Zenodo) and spent most of the
calibration's time there (plan §9 M7, docs/uipsf/architecture.md).

**What.**  The same LM -- Marquardt's D = running max of diag H, Nielsen's
gain-ratio update, only decreases accepted, the "step" and "decrease" tests
-- with every quantity per emitter.  An emitter that has stopped leaves the
active set, so later iterations compute only the emitters still moving.
The answer is the same fit; the CRLBs come from one final build.
"""
from __future__ import annotations

import time

import torch

from .lm import _cancelled, _clamped, _diagonals
from .problem import Problem
from .result import FitResult, covariance


class BlockLM:
    def __init__(self, max_iter: int = 100, tol: float = 1e-3, xtol: float = 1e-3, damping: float = 1e-3):
        self.max_iter, self.tol, self.xtol, self.damping = int(max_iter), float(tol), float(xtol), float(damping)

    def solve(self, problem, params, progress=None, cancel=None) -> FitResult:
        if params.shared_vector().numel():
            raise ValueError("BlockLM needs every shared parameter fixed (a localisation problem)")
        t0 = time.perf_counter()
        E, f64 = params.n_emitters, torch.float64
        ne = problem.normal_equations(params)
        H, g = ne.H_ee.clone(), ne.g_e.clone()                       # [E, Pe, Pe], [E, Pe]
        F = problem.objectives(params)                                # [E]
        D = _diagonals(ne)[1].clone()
        lam = torch.full((E,), self.damping, dtype=f64)
        nu = torch.full((E,), 2.0, dtype=f64)
        active = torch.ones(E, dtype=torch.bool)
        stops = ["max_iter"] * E
        history = [dict(iteration=0, objective=float(F.sum()), active=E, seconds=time.perf_counter() - t0)]
        if progress:
            progress(history[-1])
        reason = "max_iter"
        for it in range(1, self.max_iter + 1):
            if _cancelled(cancel):
                reason = "cancelled"
                break
            idx = active.nonzero()[:, 0]
            if idx.numel() == 0:
                reason = "converged"
                break
            h = torch.diagonal(H[idx], dim1=1, dim2=2)
            D[idx] = torch.maximum(D[idx], h)
            d = torch.where(D[idx] > 0, D[idx], torch.ones_like(D[idx])).sqrt()     # Jacobi scale
            A = H[idx] / (d[:, :, None] * d[:, None, :]) + lam[idx, None, None] * torch.eye(H.shape[-1], dtype=f64)
            delta = -torch.linalg.solve(A, (g[idx] / d)[..., None])[..., 0] / d
            predicted = -((g[idx] * delta).sum(1) + 0.5 * torch.einsum("ei,eij,ej->e", delta, H[idx], delta))
            sub = _sub(problem, idx)
            pa = params.subset(idx)
            ue = pa.emitter_matrix()
            trial = pa.replace(emitter=ue + delta.to(ue.device, ue.dtype))
            Ft = sub.objectives(trial)
            gain = F[idx] - Ft
            rho = torch.where(predicted > 0, gain / predicted, torch.full_like(gain, -1.0))
            ok = torch.isfinite(Ft) & (gain > 0) & (rho > 0)
            step = (delta.abs() * h.clamp(min=0).sqrt()).amax(1)
            acc = idx[ok]
            if acc.numel():
                um = params.emitter_matrix().clone()
                um[acc] = trial.emitter_matrix()[ok].to(um.dtype)
                params = params.replace(emitter=um)
                ne_a = _sub(problem, acc).normal_equations(params.subset(acc))
                H[acc], g[acc], F[acc] = ne_a.H_ee, ne_a.g_e, Ft[ok]
                r = rho[ok]
                lam[acc] = lam[acc] * torch.clamp(1 - (2 * r - 1) ** 3, min=1 / 3)
                nu[acc] = 2.0
            rej = idx[~ok]
            lam[rej], nu[rej] = lam[rej] * nu[rej], nu[rej] * 2
            # Per-emitter stops, as LM's.
            for j, e in enumerate(idx.tolist()):
                if ok[j] and step[j] < self.xtol:
                    stops[e], active[e] = "step", False
                elif ok[j] and gain[j] < self.tol and predicted[j] < self.tol:
                    stops[e], active[e] = "decrease", False
                elif not ok[j] and lam[e] > 1e10:
                    stops[e], active[e] = "no decrease", False
            entry = dict(iteration=it, objective=float(F.sum()), active=int(active.sum()), accepted=int(ok.sum()),
                         seconds=time.perf_counter() - t0)
            history.append(entry)
            if progress:
                progress(entry)
        ne = problem.normal_equations(params)                         # for the CRLBs, at the answer
        cov_s, cov_e, cond = covariance(ne)
        converged = all(s in ("step", "decrease", "no decrease") for s in stops)
        if reason == "max_iter" and converged:
            reason = "converged"
        result = FitResult.build(params, history, reason, converged, ne, cov_s, cov_e, cond,
                                 remaining=0.0, clamped=_clamped(problem))
        result.emitter_stops = stops
        return result


def _sub(problem, idx) -> Problem:
    """The same problem on some emitters only."""
    return Problem(problem.model, problem.noise, problem.batch.subset(idx), chunk_size=problem.chunk_size,
                   tangents=problem.tangents, device=problem.device, linearise=problem.linearise)

