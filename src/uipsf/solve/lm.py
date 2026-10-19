"""Block Levenberg–Marquardt with a Schur complement (plan §5.6; brief decision 4).

**The step** solves (H + λ·D) δ = −g, with H the Gauss–Newton matrix of
`Problem.normal_equations` and D Marquardt's diagonal: the largest diag(H)
seen so far for each parameter (Moré 1978).  Measuring the damping in each
parameter's own curvature makes the iterates independent of the units the
parameters are written in: rescale a parameter by a, and its column of J
scales by a, D by a², δ by 1/a.  That is what removes v1's hand-set
variable scaling (``self.weight``), which on dim beads moved the learned z
error from 61 nm to 5 nm (review, `bead_fit_scaling_lowsnr.py`); the
regression test is `test_solve.py::test_rescaling_a_parameter_leaves_the_iterates_unchanged`.

The system is solved Jacobi-scaled (D^{-½} H D^{-½} has a unit diagonal
where D = diag H), which also keeps float64 well conditioned whatever the
units.  With S = A − Σ_e B_e C_e⁻¹ B_eᵀ, the Schur complement of the
per-emitter blocks C_e, it costs one P_s × P_s solve and E small ones, so a
thousand beads cost little more than six.

**Accepting a step.**  Only steps that decrease the objective are taken.
λ follows the gain ratio ρ = actual / predicted decrease (Nielsen 1999):
λ·max(⅓, 1 − (2ρ − 1)³) after a success, doubling factors after a failure.
Bounds come from the parameters' transforms, so there are no penalty
ramps and the objective never changes during a fit (v1's ``mu`` ramp did).

**Stopping**, every reason reported:
- "decrease": an accepted step and the one predicted next both gain less
  than ``tol`` in the objective (nats; a one-σ move of one parameter is ½);
- "step": no parameter moved by more than ``xtol`` of its standard error
  (|δ_i|·√H_ii, the error with the others held fixed);
- "gradient": every |g_i|/√H_ii is below ``xtol``: no direction gains more
  than that per standard error;
- "no decrease": λ grew past 10¹⁰ without a successful step, which happens
  at the float32 noise floor of the objective;
- "max_iter", "cancelled".

**Uncertainty.**  At the solution, S⁻¹ is the covariance of the shared
parameters and the block inverse gives each emitter's; with the Fisher
weight this is the CRLB (`result.py`).
"""
from __future__ import annotations

import math
import time

import torch

from .result import FitResult, covariance


def schur_solve(ne, lam: float, D_s: torch.Tensor, D_e: torch.Tensor):
    """δ_s [P_s], δ_e [E, P_e] solving (H + λ·diag(D)) δ = −g blockwise, Jacobi-scaled by D."""
    d_s, d_e = D_s.sqrt(), D_e.sqrt()
    Ps, Pe = d_s.numel(), d_e.shape[1]
    A = ne.H_ss / (d_s[:, None] * d_s[None, :]) + lam * torch.eye(Ps, dtype=torch.float64)
    B = ne.H_se / (d_s[None, :, None] * d_e[:, None, :])
    C = ne.H_ee / (d_e[:, :, None] * d_e[:, None, :]) + lam * torch.eye(Pe, dtype=torch.float64)
    gs, ge = ne.g_s / d_s, ne.g_e / d_e
    Ci_Bt = torch.linalg.solve(C, B.transpose(1, 2))                   # [E, P_e, P_s]
    Ci_ge = torch.linalg.solve(C, ge[..., None])[..., 0]               # [E, P_e]
    S = A - torch.einsum("eij,ejk->ik", B, Ci_Bt)
    rhs = -gs + torch.einsum("eij,ej->i", B, Ci_ge)
    ds = torch.linalg.solve(S, rhs) if Ps else rhs
    de = -Ci_ge - torch.einsum("eij,j->ei", Ci_Bt, ds)
    return ds / d_s, de / d_e


def quadratic(ne, ds, de) -> tuple[float, float]:
    """(gᵀδ, δᵀHδ) from the blocks."""
    g = float(ne.g_s @ ds + (ne.g_e * de).sum())
    h = float(ds @ ne.H_ss @ ds + 2 * torch.einsum("i,eij,ej->", ds, ne.H_se, de)
              + torch.einsum("ei,eij,ej->", de, ne.H_ee, de))
    return g, h


def _floor(D):
    """A parameter that has never had any effect gets a unit scale instead of a division by zero."""
    return torch.where(D > 0, D, torch.ones_like(D))


def _diagonals(ne):
    return torch.diagonal(ne.H_ss), torch.diagonal(ne.H_ee, dim1=1, dim2=2)


def _largest(*parts) -> float:
    return max((float(p.max()) for p in parts if p.numel()), default=0.0)


def _gain(ne, lam, D_s, D_e) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The damped step and the decrease the Gauss–Newton model predicts for it."""
    ds, de = schur_solve(ne, lam, _floor(D_s), _floor(D_e))
    g, h = quadratic(ne, ds, de)
    return ds, de, -(g + 0.5 * h)


def _cancelled(cancel) -> bool:
    return cancel is not None and (cancel.is_set() if hasattr(cancel, "is_set") else bool(cancel()))


class LevenbergMarquardt:
    def __init__(self, max_iter: int = 100, tol: float = 1e-3, xtol: float = 1e-3, damping: float = 1e-3):
        self.max_iter, self.tol, self.xtol, self.damping = int(max_iter), float(tol), float(xtol), float(damping)

    def solve(self, problem, params, progress=None, cancel=None) -> FitResult:
        """Fit ``params`` (a `Params`) to ``problem``; ``progress(entry)`` after every iteration,
        ``cancel`` (a callable or an Event) checked between iterations."""
        t0 = time.perf_counter()
        ne = problem.normal_equations(params)
        D_s, D_e = _diagonals(ne)
        lam, nu = self.damping, 2.0
        history = [dict(iteration=0, objective=ne.objective, damping=lam, seconds=time.perf_counter() - t0)]
        if progress:
            progress(history[-1])
        reason, fresh = "max_iter", True                # fresh: ne belongs to a point not yet tested
        for it in range(1, self.max_iter + 1):
            if _cancelled(cancel):
                reason = "cancelled"
                break
            h_s, h_e = _diagonals(ne)
            if fresh:
                # The objective's slope per standard error of each parameter.
                if _largest(ne.g_s.abs() / _floor(h_s).sqrt(), ne.g_e.abs() / _floor(h_e).sqrt()) < self.xtol:
                    reason = "gradient"
                    break
            D_s, D_e = torch.maximum(D_s, h_s), torch.maximum(D_e, h_e)
            ds, de, predicted = _gain(ne, lam, D_s, D_e)
            step = _largest(ds.abs() * h_s.clamp(min=0).sqrt(), de.abs() * h_e.clamp(min=0).sqrt())
            us, ue = params.shared_vector(), params.emitter_matrix()
            trial = params.replace(shared=us + ds.to(us.device, us.dtype), emitter=ue + de.to(ue.device, ue.dtype))
            F = problem.objective(trial)
            gain = ne.objective - F
            rho = gain / predicted if predicted > 0 else -1.0
            accepted = math.isfinite(F) and gain > 0 and rho > 0
            entry = dict(iteration=it, objective=F if accepted else ne.objective, trial=F, damping=lam,
                         rho=rho, step=step, predicted=predicted, accepted=accepted)
            if accepted:
                params, ne, fresh = trial, problem.normal_equations(trial), True
                lam, nu = lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 2.0
            else:
                lam, nu, fresh = lam * nu, nu * 2.0, False
            entry["seconds"] = time.perf_counter() - t0
            history.append(entry)
            if progress:
                progress(entry)
            if accepted and step < self.xtol:
                reason = "step"
                break
            if accepted and gain < self.tol and _gain(ne, lam, *_diagonals(ne))[2] < self.tol:
                reason = "decrease"
                break
            if not accepted and lam > 1e10:
                reason = "no decrease"
                break
        # What a Gauss–Newton step from the end point would still gain: ~0 at a minimum.
        remaining = _gain(ne, self.damping, *_diagonals(ne))[2]
        converged = reason in ("decrease", "step", "gradient") or (
            reason == "no decrease" and remaining < 10 * self.tol)
        cov_s, cov_e, cond = covariance(ne)
        return FitResult.build(params, history, reason, converged, ne, cov_s, cov_e, cond,
                               remaining=remaining, clamped=_clamped(problem))


def _clamped(problem) -> int:
    b = problem.batch
    return problem.noise.clamped(b.data, b.s, b.mask) if hasattr(problem.noise, "clamped") else 0
