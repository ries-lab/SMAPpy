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


class InSituGeometry:
    """Emitters in the sample, seen through a focus set somewhere above the coverslip (plan §9, M5).

    An emitter's z is its height h above the coverslip (`docs/uipsf/conventions.md`), and the focus f
    is where the objective puts its nominal focal plane: how far it was moved from the coverslip's
    focus, v1's ``stage_pos``.  The defocus wave is exp(2πi (k_z,med h − k_z,imm f)) (uiPSF's in situ
    model): h travels through the sample's index, f through the immersion's, and their difference is the
    index mismatch's spherical aberration growing with depth.  The output carries the two apart, four
    coordinates per observation, (x, y, z_k − f, h): the third is the bead stack's z (through k_z,imm,
    with ``Batch.z_nm`` a nominal focus offset per observation, zero for one frame), the fourth the
    PSF's ``depth_nm``.  Both enter only as factors of the pupil, so the emitter's h and the shared f
    keep the solver's linearised Jacobian columns (`ForwardModel.emitter_linearisation`).

    The focus has a loose prior, N(f₀, 500 nm) by default: the value the user enters is a stage
    reading, often off by a few hundred nanometres (decided October 2026).  An emitter seen in K
    frames (a linked blink) has one position in all of them.
    """

    def __init__(self, focus_nm: float, focus_prior_nm: float | None = 500.0, fit_focus: bool = True):
        self.focus_nm, self.fit_focus = float(focus_nm), fit_focus
        self.focus_prior_nm = None if focus_prior_nm is None else float(focus_prior_nm)

    def specs(self, batch, start) -> list[ParamSpec]:
        prior = None if self.focus_prior_nm is None else Gaussian(self.focus_nm, self.focus_prior_nm)
        return [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"]),
                ParamSpec("geometry.focus", (), init=self.focus_nm, prior=prior, fixed=not self.fit_focus)]

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, K, 4] nm: x, y, the defocus through the immersion, the height in the sample."""
        xyz = p["emitter.xyz"]
        z_k = batch.z_nm.to(xyz.dtype)
        z_k = z_k[None] if z_k.dim() == 1 else z_k                        # [1|E, K]
        E, K = xyz.shape[0], z_k.shape[-1]
        h = xyz[:, None, 2].expand(E, K)
        return torch.stack([xyz[:, None, 0].expand(E, K), xyz[:, None, 1].expand(E, K),
                            (z_k - p["geometry.focus"]).expand(E, K), h], -1)
