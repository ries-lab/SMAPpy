"""L-BFGS on a Gauss–Newton matrix that is rebuilt only now and then (plan §9 M7, candidates B and D).

**Why.**  An LM iteration rebuilds the whole Jacobian: one JVP per parameter,
for six beads as much as ~60 forward + backward passes (`benchmarks/uipsf/
profile_lm.py`), while the gradient alone costs one.  v1's L-BFGS-B used
only gradients, so its steps were cheap -- but it started each run from an
identity matrix, so its progress depended on the units of the parameters,
which is what ``self.weight`` patched (review, `bead_fit_scaling_lowsnr.py`).

**What.**  Here the Gauss–Newton matrix H of `Problem.normal_equations`,
damped as LM damps it (H + λ·D, D = diag H), is L-BFGS's *initial* matrix.
It is applied by the same Schur solve, so the step is unit-free from the
first iteration (the rescaling test of LM holds unchanged), and the
quasi-Newton pairs correct it as the point moves.  Every iteration takes
one `Problem.value_and_grad` (more when the line search backtracks); H is
rebuilt after ``refresh`` iterations, or sooner when progress stalls.
With ``memory=0`` the step is a Gauss–Newton step on a stale matrix and
an exact gradient (a "chord" method, candidate B).

**Stopping** is decided at a rebuild, so the answer carries a fresh H and
the CRLBs and the "remaining" gain are LM's: when the Gauss–Newton step
from there predicts less than ``tol`` (as LM's "decrease"), or on LM's
gradient test.  LM's "step" test (no parameter moves by ``xtol`` of its
standard error) stops it between rebuilds, at the float32 floor of the
objective; that point is within ``xtol`` standard errors of the last H's.
"""
from __future__ import annotations

import dataclasses
import math
import time

from .lm import _cancelled, _clamped, _diagonals, _floor, _gain, _largest, schur_solve
from .result import FitResult, covariance


class _Vec:
    """A point or a direction in the solver's layout: (shared [P_s], emitter [E, P_e]), float64."""

    def __init__(self, s, e):
        self.s, self.e = s, e

    def __add__(self, o):
        return _Vec(self.s + o.s, self.e + o.e)

    def __sub__(self, o):
        return _Vec(self.s - o.s, self.e - o.e)

    def __mul__(self, a: float):
        return _Vec(self.s * a, self.e * a)

    __rmul__ = __mul__

    def dot(self, o) -> float:
        return float(self.s @ o.s + (self.e * o.e).sum())


class GaussNewtonLBFGS:
    def __init__(self, max_iter: int = 300, refresh: int = 10, memory: int = 10, tol: float = 1e-3,
                 xtol: float = 1e-3, damping: float = 1e-3, max_backtracks: int = 8):
        self.max_iter, self.refresh, self.memory = int(max_iter), int(refresh), int(memory)
        self.tol, self.xtol, self.damping = float(tol), float(xtol), float(damping)
        self.max_backtracks = int(max_backtracks)

    def solve(self, problem, params, progress=None, cancel=None) -> FitResult:
        t0 = time.perf_counter()
        history, it, reason, lam = [], 0, "max_iter", self.damping
        ne = problem.normal_equations(params)
        rebuilds = 1
        F, g = ne.objective, _Vec(ne.g_s, ne.g_e)       # always those of ``params``
        history.append(dict(iteration=0, objective=ne.objective, seconds=time.perf_counter() - t0))
        if progress:
            progress(history[-1])
        while True:
            # -- at a fresh Gauss–Newton matrix: the stopping tests, as LM's
            h_s, h_e = _diagonals(ne)
            if _largest(ne.g_s.abs() / _floor(h_s).sqrt(), ne.g_e.abs() / _floor(h_e).sqrt()) < self.xtol:
                reason = "gradient"
                break
            if _gain(ne, self.damping, h_s, h_e)[2] < self.tol:
                reason = "decrease"
                break
            if it >= self.max_iter or _cancelled(cancel):
                reason = "max_iter" if it >= self.max_iter else "cancelled"
                break
            D_s, D_e = _floor(h_s), _floor(h_e)

            def H0_solve(g: _Vec, ne=ne, lam=lam, D_s=D_s, D_e=D_e) -> _Vec:     # (H + λD)⁻¹ g
                ds, de = schur_solve(dataclasses.replace(ne, g_s=g.s, g_e=g.e), lam, D_s, D_e)
                return _Vec(-ds, -de)

            pairs: list[tuple[_Vec, _Vec, float]] = []
            F_start, stalled = F, False
            for _ in range(self.refresh):
                if it >= self.max_iter or _cancelled(cancel):
                    break
                it += 1
                d = _two_loop(g, pairs, H0_solve) * -1.0
                slope = g.dot(d)
                if not slope < 0:                              # not a descent direction: drop the memory
                    pairs.clear()
                    d = H0_solve(g) * -1.0
                    slope = g.dot(d)
                alpha, accepted, trial_params, evals, F_new = 1.0, False, None, 0, math.nan
                for _ in range(self.max_backtracks + 1):
                    trial_params = _moved(params, d * alpha)
                    if _saturates(params, trial_params):
                        alpha *= 0.25
                        continue
                    F_new, gs_new, ge_new = problem.value_and_grad(trial_params)
                    evals += 1
                    if math.isfinite(F_new) and F_new <= F + 1e-4 * alpha * slope:
                        accepted = True
                        break
                    alpha *= 0.25 if not math.isfinite(F_new) else max(0.1, _quadratic_min(F, slope, F_new, alpha))
                entry = dict(iteration=it, objective=F_new if accepted else F, trial=F_new, alpha=alpha,
                             evaluations=evals, accepted=accepted, damping=lam)
                if not accepted:
                    lam *= 10.0                                # the stale matrix is far off: trust it less
                    stalled = True
                    entry["seconds"] = time.perf_counter() - t0
                    history.append(entry)
                    break
                g_new = _Vec(gs_new, ge_new)
                s, y = d * alpha, g_new - g
                sy = s.dot(y)
                if sy > 1e-12 * math.sqrt(max(s.dot(s) * y.dot(y), 0.0)):
                    pairs.append((s, y, sy))
                    if len(pairs) > self.memory:
                        pairs.pop(0)
                gain = F - F_new
                # LM's step test: no parameter moved by xtol of its standard error (at the float32 floor of
                # the objective the line search still "accepts" steps of α ≈ 0 that gain nothing).
                step = _largest((d.s * alpha).abs() * h_s.clamp(min=0).sqrt(),
                                (d.e * alpha).abs() * h_e.clamp(min=0).sqrt())
                params, F, g = trial_params, F_new, g_new
                entry.update(seconds=time.perf_counter() - t0, step=step)
                history.append(entry)
                if progress:
                    progress(entry)
                if step < self.xtol:
                    reason = "step"
                    break
                if gain < self.tol:
                    break
            if reason == "step":
                break
            if F_start - F < self.tol:                       # nothing gained since the rebuild: a stall
                lam *= 10.0
                stalled = True
                if lam > 1e10:
                    reason = "no decrease"
                    break
            elif F < F_start:
                if not stalled:
                    lam = max(self.damping, lam / 3.0)
                ne = problem.normal_equations(params)
                F, g = ne.objective, _Vec(ne.g_s, ne.g_e)
                rebuilds += 1
                history[-1]["rebuild"] = True
        remaining = _gain(ne, self.damping, *_diagonals(ne))[2]
        converged = reason in ("decrease", "gradient", "step") or (
            reason == "no decrease" and remaining < 10 * self.tol)
        cov_s, cov_e, cond = covariance(ne)
        result = FitResult.build(params, history, reason, converged, ne, cov_s, cov_e, cond,
                                 remaining=remaining, clamped=_clamped(problem))
        result.objective = F                    # at ``params``; ne's may be from the last rebuild ("step")
        result.rebuilds = rebuilds
        return result


def _moved(params, d: _Vec):
    us, ue = params.shared_vector(), params.emitter_matrix()
    return params.replace(shared=us + d.s.to(us.device, us.dtype), emitter=ue + d.e.to(ue.device, ue.dtype))


def _saturates(params, trial, factor: float = 100.0) -> bool:
    """Whether a step drives a parameter into the flat tail of its transform.

    A softplus background stepped far below zero has d(physical)/du ≈ 0
    there: no gradient brings it back, and its emitter block turns singular.
    One L-BFGS step did this to two Zenodo beads' backgrounds (11 and 8
    photons to 0; 3800 nats lost, docs/uipsf/architecture.md M7).  A step may
    shrink a derivative by at most ``factor``, so a value can still approach
    its bound, over several steps.
    """
    for scope in ("shared", "emitter"):
        before, after = params.derivative(scope).abs(), trial.derivative(scope).abs()
        if before.numel() and bool((after * factor < before).any()):
            return True
    return False


def _two_loop(g: _Vec, pairs, H0_solve) -> _Vec:
    """H⁻¹g by the L-BFGS two-loop recursion (Nocedal & Wright, Alg. 7.4), H0⁻¹ applied by ``H0_solve``."""
    q, alphas = g, []
    for s, y, sy in reversed(pairs):
        a = s.dot(q) / sy
        alphas.append(a)
        q = q - y * a
    r = H0_solve(q)
    for (s, y, sy), a in zip(pairs, reversed(alphas), strict=True):
        b = y.dot(r) / sy
        r = r + s * (a - b)
    return r


def _quadratic_min(F0, slope, F1, alpha) -> float:
    """The minimiser of the parabola through (0, F0) with ``slope`` and (alpha, F1), as a fraction of alpha."""
    curvature = F1 - F0 - slope * alpha
    if curvature <= 0:
        return 0.5
    return min(0.5, -slope * alpha / (2 * curvature))
