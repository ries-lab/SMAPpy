"""Photons per emitter and observation, N[e, k] (plan §3, "Photons per slice").

- `Bleaching` (default): a smooth curve per bead, N_e·exp(Σ_j b_ej t_k^j),
  with t_k from −1 at the first slice to +1 at the last.  N_e is then the
  photon count at the middle of the stack, nearly uncorrelated with the
  rates.  Order 1 is an exponential; order 0 a constant (a single frame).
  It keeps the z dependence of the collected photons, which carries
  information about the pupil; v1's free count per slice throws it away.
- `PerSliceFactor`: the same, times a factor f_k shared by all beads in a
  frame, for an unstable laser.  A prior f ~ N(1, 0.2) pins the factors'
  mean, otherwise exactly degenerate with the beads' photons.
- `PerObservation`: a free count per bead and slice (v1's ``var_photon``),
  for excitation that changes per bead and frame, such as speckle.

Photons are physical totals (pupil energy, `docs/uipsf/conventions.md`), kept
positive by a softplus.
"""
from __future__ import annotations

import torch

from ..params import Exp, Gaussian, ParamSpec, Softplus


def _t(K, like):
    if K == 1:
        return torch.zeros(1, dtype=like.dtype, device=like.device)
    return torch.linspace(-1.0, 1.0, K, dtype=like.dtype, device=like.device)


class Bleaching:
    def __init__(self, order: int = 1):
        self.order = int(order)

    def specs(self, batch, start) -> list[ParamSpec]:
        specs = [ParamSpec("emitter.photons", (), "emitter", init=start["photons"], transform=Softplus())]
        if self.order > 0:
            specs.append(ParamSpec("emitter.bleaching", (self.order,), "emitter", init=0.0))
        return specs

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, K]."""
        N = p["emitter.photons"]
        if self.order == 0:
            return N[:, None].expand(-1, batch.data.shape[2])
        t = _t(batch.data.shape[2], N)
        powers = t[None, :] ** torch.arange(1, self.order + 1, device=N.device, dtype=N.dtype)[:, None]
        return N[:, None] * torch.exp(p["emitter.bleaching"] @ powers)


class PerSliceFactor(Bleaching):
    def __init__(self, order: int = 1, prior_sigma: float = 0.2):
        super().__init__(order)
        self.prior_sigma = float(prior_sigma)

    def specs(self, batch, start) -> list[ParamSpec]:
        K = batch.data.shape[2]
        return super().specs(batch, start) + [
            ParamSpec("intensity.slice_factor", (K,), init=1.0, transform=Exp(),
                      prior=Gaussian(1.0, self.prior_sigma))]

    def __call__(self, p, batch) -> torch.Tensor:
        return super().__call__(p, batch) * p["intensity.slice_factor"]


class PerObservation:
    def specs(self, batch, start) -> list[ParamSpec]:
        """The start's photons are one count per emitter, or already one per observation ([E, K])."""
        K = batch.data.shape[2]
        init = start["photons"]
        init = init if init.dim() == 2 else init[:, None].expand(-1, K)
        return [ParamSpec("emitter.photons_per_slice", (K,), "emitter", init=init, transform=Softplus())]

    def __call__(self, p, batch) -> torch.Tensor:
        return p["emitter.photons_per_slice"]
