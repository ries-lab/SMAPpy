"""Where each emitter is in each observation.

**Bead stacks** (plan §3): bead e at slice k is at

    xyz[e, k] = (x_e + δx_k,  y_e + δy_k,  z_e + s·z_k + δz_k),

with z_k the slice's nominal emitter z (`Batch.z_nm`, SMAPpy's sign) and
(x_e, y_e, z_e) the bead's position, z_e its height at the nominal focus.
The stage terms are the cheap physics the review found most damaging when
missing (review §B): a drift δ per slice shared by all beads in a frame, and
a z scale s of the piezo.  Both are fixed by default.  The z scale is the
open question: a wrong one leaves no pattern in the residuals (the pupil
absorbs it), yet the Zenodo `1ch_40nm_bead` stacks ask for s = 0.968, a
gain of 2193 nats on 16 beads (docs/uipsf/architecture.md, M3).  A piezo is the
least likely cause; an effective NA 0.013 below nominal or an emission
20 nm redder than assumed act as the same scale.  Until that is settled
(plan §9, M8) it is fitted only when asked, with a prior s ~ N(1, 0.1)
that keeps it sane where the data say nothing (a single frame).
The drift has a prior of 5 nm per slice and per axis, which also pins its
mean, otherwise exactly degenerate with the beads' positions.
"""
from __future__ import annotations

import torch

from ..params import Gaussian, ParamSpec


class BeadStackGeometry:
    def __init__(self, drift: bool = False, drift_prior_nm: float = 5.0, z_scale: bool = False,
                 z_scale_prior: float = 0.1):
        self.drift, self.drift_prior_nm, self.z_scale = drift, float(drift_prior_nm), z_scale
        self.z_scale_prior = float(z_scale_prior)

    def specs(self, batch, start) -> list[ParamSpec]:
        K = batch.data.shape[2]
        return [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"]),
                ParamSpec("geometry.drift", (K, 3), init=0.0, prior=Gaussian(0.0, self.drift_prior_nm),
                          fixed=not self.drift),
                ParamSpec("geometry.z_scale", (), init=1.0, prior=Gaussian(1.0, self.z_scale_prior),
                          fixed=not self.z_scale)]

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, K, 3] nm."""
        xyz = p["emitter.xyz"][:, None, :]
        z_k = batch.z_nm.to(xyz.dtype)
        z_k = z_k[None] if z_k.dim() == 1 else z_k                        # [1|E, K]
        stage = torch.stack([torch.zeros_like(z_k), torch.zeros_like(z_k), p["geometry.z_scale"] * z_k], -1)
        return xyz + stage + p["geometry.drift"]
