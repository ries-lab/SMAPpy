"""What a fit returns: the parameters, how it went, and their uncertainties.

**Uncertainty is the CRLB.**  The Gauss–Newton matrix with the noise
model's Fisher weight *is* the Fisher information, so its inverse at the
solution is the Cramér–Rao bound, the covariance an unbiased estimator
reaches (`test_solve.py` checks it against the spread of 200 simulated
localisations).  It is formed blockwise: S⁻¹ for the shared parameters,
C_e⁻¹ + C_e⁻¹B_eᵀ S⁻¹ B_e C_e⁻¹ for emitter e, all Jacobi-scaled and in
float64.  The solver works with unconstrained values; the standard errors
are reported for the physical ones, |d(physical)/du|·σ_u.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..device import to_cpu64


def covariance(ne):
    """(cov_s [P_s, P_s], cov_e [E, P_e, P_e], condition number of the scaled S), unconstrained."""
    d_s = torch.where(torch.diagonal(ne.H_ss) > 0, torch.diagonal(ne.H_ss), 1.0).sqrt()
    He = torch.diagonal(ne.H_ee, dim1=1, dim2=2)
    d_e = torch.where(He > 0, He, 1.0).sqrt()
    A = ne.H_ss / (d_s[:, None] * d_s[None, :])
    B = ne.H_se / (d_s[None, :, None] * d_e[:, None, :])
    C = ne.H_ee / (d_e[:, :, None] * d_e[:, None, :])
    # A parameter the data say nothing about (a softplus at zero, an empty slice) has a zero row; a ridge
    # of 1e-12 on the unit-diagonal scaled blocks gives it a CRLB of ~10⁶ times its scale instead of a
    # singular matrix, and changes nothing measurable elsewhere.
    ridge = 1e-12
    Ci = torch.linalg.inv(C + ridge * torch.eye(C.shape[-1], dtype=C.dtype))
    Ci_Bt = Ci @ B.transpose(1, 2)
    S = A - torch.einsum("eij,ejk->ik", B, Ci_Bt)
    Ps = S.shape[0]
    Si = torch.linalg.inv(S + ridge * torch.eye(Ps, dtype=S.dtype)) if Ps else S
    cond = float(torch.linalg.cond(S)) if Ps else 1.0
    cov_e = Ci + Ci_Bt @ Si @ Ci_Bt.transpose(1, 2)
    cov_s = Si / (d_s[:, None] * d_s[None, :])
    cov_e = cov_e / (d_e[:, :, None] * d_e[:, None, :])
    return cov_s, cov_e, cond


@dataclass
class FitResult:
    params: object                          # Params at the solution
    history: list[dict]                     # one entry per iteration (objective, damping, rho, step, …)
    stop_reason: str
    converged: bool
    objective: float                        # ½ deviance + priors at the solution
    covariance_shared: torch.Tensor         # [P_s, P_s], physical units, packed as Params.layout("shared")
    std: dict[str, torch.Tensor] = field(default_factory=dict)   # CRLB of every free parameter, physical
    condition: float = 1.0                  # of the Jacobi-scaled Schur complement: near-degeneracies
    remaining: float = 0.0                  # what a further Gauss–Newton step would still gain (nats)
    clamped: int = 0                        # shifted data values clamped at 0

    @property
    def iterations(self) -> int:
        """Solver iterations, accepted or not, over all stages (each history starts at iteration 0)."""
        return sum(1 for h in self.history if h["iteration"] > 0)

    @classmethod
    def build(cls, params, history, reason, converged, ne, cov_s, cov_e, cond, remaining=0.0, clamped=0):
        js = to_cpu64(params.derivative("shared"))
        je = to_cpu64(params.derivative("emitter"))
        cov_phys = cov_s * js[:, None] * js[None, :]
        sd_s = torch.diagonal(cov_s).clamp(min=0).sqrt() * js.abs()
        sd_e = torch.diagonal(cov_e, dim1=1, dim2=2).clamp(min=0).sqrt() * je.abs()
        std = {}
        for n, a, b in params.layout("shared"):
            std[n] = sd_s[a:b].reshape(params.specs[n].shape)
        for n, a, b in params.layout("emitter"):
            std[n] = sd_e[:, a:b].reshape(params.n_emitters, *params.specs[n].shape)
        return cls(params, history, reason, converged, ne.objective, cov_phys, std, cond, remaining, clamped)
