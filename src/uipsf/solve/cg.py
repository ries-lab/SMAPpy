"""Gauss–Newton steps solved by preconditioned conjugate gradients, without rebuilding H (plan §9 M7, C).

**Why.**  Building the Gauss–Newton matrix costs one Jacobian column per
parameter: for six beads at n_max 8, 8.4 s on 4 CPU cores, 45 gradients'
worth (`benchmarks/uipsf/profile_lm.py`).  `Problem.gauss_newton_operator` gives
H·v for 0.8 of a gradient once the model is traced at a point.  So each
step here solves (H + λD) δ = −g by conjugate gradients on the *current* H,
preconditioned by the *last built* one, applied by LM's Schur solve.  While
H changes little, a few products solve the step; when the preconditioner
has gone stale (CG needs ``max_cg`` products), it is rebuilt.  The cost of
a step does not grow with the number of shared parameters, which field
dependence and multi-channel models will multiply.

**What stays LM's.**  The damping λ·D with D = diag H (Moré), so the steps
are unit-free; the gain-ratio update of λ (Nielsen); accepting only
decreases; the stopping tests ("decrease" and "gradient" at a built H,
"step" at any accepted step); the saturation guard of `lbfgs`; and a final
build, so the CRLBs are LM's.

**CG** stops when the preconditioned residual falls below ``forcing``
times that of the start (an inexact Newton step: early steps need little
accuracy), after ``max_cg`` products, or on negative curvature, which a
Gauss–Newton matrix cannot have beyond round-off.
"""
from __future__ import annotations

import dataclasses
import math
import time

from .lbfgs import _moved, _saturates, _Vec
from .lm import _cancelled, _clamped, _diagonals, _floor, _gain, _largest, schur_solve
from .result import FitResult, covariance


class GaussNewtonCG:
    def __init__(self, max_iter: int = 100, max_cg: int = 12, forcing: float = 0.1, tol: float = 1e-3,
                 xtol: float = 1e-3, damping: float = 1e-3, trace: bool = True):
        self.max_iter, self.max_cg, self.forcing = int(max_iter), int(max_cg), float(forcing)
        self.tol, self.xtol, self.damping = float(tol), float(xtol), float(damping)
        self.trace = bool(trace)                           # `Problem.gauss_newton_operator`

    def solve(self, problem, params, progress=None, cancel=None) -> FitResult:
        t0 = time.perf_counter()
        ne = problem.normal_equations(params)
        fresh, builds, products = True, 1, 0          # fresh: ne is at ``params``
        F, g = ne.objective, _Vec(ne.g_s, ne.g_e)
        lam, nu, reason = self.damping, 2.0, "max_iter"
        history = [dict(iteration=0, objective=F, seconds=time.perf_counter() - t0)]
        if progress:
            progress(history[-1])
        D_s, D_e = _diagonals(ne)
        product = None                                     # traced at ``params``; kept while they stay
        for it in range(1, self.max_iter + 1):
            if _cancelled(cancel):
                reason = "cancelled"
                break
            h_s, h_e = _diagonals(ne)
            if fresh:
                if _largest(g.s.abs() / _floor(h_s).sqrt(), g.e.abs() / _floor(h_e).sqrt()) < self.xtol:
                    reason = "gradient"
                    break
                if _gain(ne, self.damping, h_s, h_e)[2] < self.tol:
                    reason = "decrease"
                    break
            D_s, D_e = _floor(_max(D_s, h_s)), _floor(_max(D_e, h_e))
            if product is None:
                product = problem.gauss_newton_operator(params, self.trace)
            d, Hd, used = self._pcg(product, ne, g, lam, D_s, D_e)
            products += used
            predicted = -(g.dot(d) + 0.5 * d.dot(Hd))
            trial = _moved(params, d)
            F_new = math.nan if _saturates(params, trial) else problem.objective(trial)
            gain = F - F_new
            rho = gain / predicted if predicted > 0 and math.isfinite(F_new) else -1.0
            accepted = math.isfinite(F_new) and gain > 0 and rho > 0
            step = _largest(d.s.abs() * h_s.clamp(min=0).sqrt(), d.e.abs() * h_e.clamp(min=0).sqrt())
            entry = dict(iteration=it, objective=F_new if accepted else F, trial=F_new, damping=lam, rho=rho,
                         predicted=predicted, cg=used, accepted=accepted, step=step)
            if accepted:
                params, product = trial, None
                lam, nu = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-12), 2.0
                if used >= self.max_cg:                    # the preconditioner has gone stale
                    ne, fresh, builds = problem.normal_equations(params), True, builds + 1
                    F, g = ne.objective, _Vec(ne.g_s, ne.g_e)
                    entry["rebuild"] = True
                else:
                    F, gs, ge = problem.value_and_grad(params)
                    g, fresh = _Vec(gs, ge), False
            else:
                lam, nu = lam * nu, nu * 2.0
            entry["seconds"] = time.perf_counter() - t0
            history.append(entry)
            if progress:
                progress(entry)
            if accepted and step < self.xtol:
                reason = "step"
                break
            if accepted and gain < self.tol and predicted < self.tol:
                reason = "decrease"
                break
            if not accepted and lam > 1e10:
                reason = "no decrease"
                break
        if not fresh:                                      # the CRLBs need H at the answer
            ne, builds = problem.normal_equations(params), builds + 1
        remaining = _gain(ne, self.damping, *_diagonals(ne))[2]
        converged = reason in ("decrease", "gradient", "step") or (
            reason == "no decrease" and remaining < 10 * self.tol)
        cov_s, cov_e, cond = covariance(ne)
        result = FitResult.build(params, history, reason, converged, ne, cov_s, cov_e, cond,
                                 remaining=remaining, clamped=_clamped(problem))
        result.rebuilds, result.products = builds, products
        return result

    def _pcg(self, product, ne, g: _Vec, lam: float, D_s, D_e):
        """δ ≈ −(H + λD)⁻¹ g by CG, preconditioned by the last built H (+ λD): (δ, Hδ, products used)."""
        def apply(v: _Vec) -> _Vec:                        # (H + λD) v, H the current one
            hs, he = product(v.s, v.e)
            return _Vec(hs + lam * D_s * v.s, he + lam * D_e * v.e)

        def precondition(r: _Vec) -> _Vec:                 # (H_built + λD)⁻¹ r
            ds, de = schur_solve(dataclasses.replace(ne, g_s=r.s, g_e=r.e), lam, D_s, D_e)
            return _Vec(-ds, -de)

        x = _Vec(g.s * 0, g.e * 0)
        Hx = _Vec(g.s * 0, g.e * 0)
        r = g * -1.0
        z = precondition(r)
        p, rz = z, r.dot(z)
        start = math.sqrt(max(rz, 0.0))
        used = 0
        for _ in range(self.max_cg):
            Ap = apply(p)
            used += 1
            curvature = p.dot(Ap)
            if curvature <= 0:                             # round-off: keep what there is
                if used == 1:
                    x, Hx = p, Ap
                break
            a = rz / curvature
            x, Hx, r = x + p * a, Hx + Ap * a, r - Ap * a
            z = precondition(r)
            rz_new = r.dot(z)
            if math.sqrt(max(rz_new, 0.0)) <= self.forcing * start:
                break
            p, rz = z + p * (rz_new / rz), rz_new
        # Hx above includes λD x; the model's prediction uses H alone.
        return x, Hx - _Vec(lam * D_s * x.s, lam * D_e * x.e), used


def _max(a, b):
    return a.maximum(b)
