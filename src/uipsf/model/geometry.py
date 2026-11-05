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

**The drift's forms** (``drift``): ``"free"``, the δ per slice above; ``"linear"``, a lateral drift
growing linearly through each stack, (δx, δy) = r_s·z_k with a rate r_s (nm per µm of the stage's
travel) of the stack s the bead came from (`Batch.stack_id`) -- a stage drifting steadily while it
steps, which on M5's beads took the lateral bias from 2.4-7.2 nm to 1.3-2.5 nm with two numbers per
stack, against the dozens of a drift per slice (November 2026).  ``drift_axes`` "xy" leaves z out:
a z drift absorbs the z bend shared by the beads by construction.

**The tilt** of the coverslip, ``geometry.tilt`` = (t_x, t_y) in radians, is shared by every emitter
and read by the PSF, not by the positions: it bends the layers' phase (`optics.layers`).  Both
geometries declare it only when asked -- ``fit_tilt``, or a held value ``tilt_rad`` -- so that a model
without it is today's model exactly.  Its prior is weak, N(0, 0.05 rad), 3°: a coverslip in a holder
is rarely tilted by more than a degree, and the prior only keeps the tilt at zero where the data say
nothing.  For beads it is off by default: with an oil objective beads on the coverslip do not see it
at all (nothing but immersion between them and the objective), and it is identifiable only with a
water or silicone objective, through the plate.  In situ, molecules above the glass see it through the
glass–sample interface, the more the deeper they are.
"""
from __future__ import annotations

import torch

from ..params import Gaussian, ParamSpec

#: the tilt's prior, radians per axis (module docstring)
TILT_PRIOR_RAD = 0.05


def tilt_spec(fit_tilt: bool, tilt_rad, prior_rad: float | None) -> list[ParamSpec]:
    """``geometry.tilt``'s spec, or none when it is neither fitted nor held at a value."""
    if not fit_tilt and tilt_rad is None:
        return []
    init = torch.zeros(2) if tilt_rad is None else torch.as_tensor(tilt_rad, dtype=torch.float32).reshape(2)
    prior = None if prior_rad is None else Gaussian(0.0, float(prior_rad))
    return [ParamSpec("geometry.tilt", (2,), init=init, prior=prior, fixed=not fit_tilt)]


class _Tilted:
    """The tilt's settings, shared by both geometries."""

    def _tilt_settings(self, fit_tilt, tilt_rad, tilt_prior_rad):
        self.fit_tilt = bool(fit_tilt)
        self.tilt_rad = None if tilt_rad is None else tuple(float(v) for v in tilt_rad)
        self.tilt_prior_rad = None if tilt_prior_rad is None else float(tilt_prior_rad)

    def _tilt_specs(self):
        return tilt_spec(self.fit_tilt, self.tilt_rad, self.tilt_prior_rad)


#: the drift's forms (module docstring); "none" learns none
DRIFTS = ("none", "free", "linear")
#: the linear drift's prior, nm per µm of the stage's travel: loose, M5's stacks drifted 5-54
DRIFT_RATE_PRIOR = 100.0


class BeadStackGeometry(_Tilted):
    def __init__(self, drift: str = "none", drift_prior_nm: float = 5.0, z_scale: bool = False,
                 z_scale_prior: float = 0.1, fit_tilt: bool = False, tilt_rad=None,
                 tilt_prior_rad: float | None = TILT_PRIOR_RAD, drift_axes: str = "xy"):
        if drift not in DRIFTS:
            raise ValueError(f"drift must be one of {DRIFTS}, got {drift!r}")
        self.drift, self.drift_axes = drift, drift_axes
        if self.drift == "linear" and "z" in self.drift_axes:
            raise ValueError("the linear drift is lateral: drift_axes 'xy'")
        self.drift_prior_nm, self.z_scale = float(drift_prior_nm), z_scale
        self.z_scale_prior = float(z_scale_prior)
        self._tilt_settings(fit_tilt, tilt_rad, tilt_prior_rad)

    def specs(self, batch, start) -> list[ParamSpec]:
        K = batch.data.shape[2]
        out = [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"]),
               ParamSpec("geometry.drift", (K, 3), init=0.0, prior=Gaussian(0.0, self.drift_prior_nm),
                         fixed=self.drift != "free"),
               ParamSpec("geometry.z_scale", (), init=1.0, prior=Gaussian(1.0, self.z_scale_prior),
                         fixed=not self.z_scale)]
        if self.drift == "linear":
            stacks = batch.n_stacks                 # the acquisition's, whichever beads are at hand
            out.append(ParamSpec("geometry.drift_rate", (stacks, 2), init=0.0, prior=Gaussian(0.0, DRIFT_RATE_PRIOR)))
        return out + self._tilt_specs()

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, K, 3] nm."""
        xyz = p["emitter.xyz"][:, None, :]
        z_k = batch.z_nm.to(xyz.dtype)
        z_k = z_k[None] if z_k.dim() == 1 else z_k                        # [1|E, K]
        stage = torch.stack([torch.zeros_like(z_k), torch.zeros_like(z_k), p["geometry.z_scale"] * z_k], -1)
        drift = p["geometry.drift"]
        if self.drift == "free" and self.drift_axes != "xyz":
            drift = drift * drift.new_tensor([float(a in self.drift_axes) for a in "xyz"])
        out = xyz + stage + drift
        if self.drift == "linear":
            rate = p["geometry.drift_rate"]                                # [S, 2] nm per µm
            stack = (torch.zeros(xyz.shape[0], dtype=torch.long, device=rate.device) if batch.stack_id is None
                     else batch.stack_id.to(rate.device))
            lateral = rate[stack][:, None, :] * (z_k / 1000.0)[..., None]  # [E, K, 2]
            out = out + torch.nn.functional.pad(lateral, (0, 1))
        return out


class InSituGeometry(_Tilted):
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
    frames (a linked blink) has one position in all of them.  ``fit_tilt`` learns the coverslip's tilt
    (module docstring), ``tilt_rad`` holds it at a value.
    """

    def __init__(self, focus_nm: float, focus_prior_nm: float | None = 500.0, fit_focus: bool = True,
                 fit_tilt: bool = False, tilt_rad=None, tilt_prior_rad: float | None = TILT_PRIOR_RAD):
        self.focus_nm, self.fit_focus = float(focus_nm), fit_focus
        self.focus_prior_nm = None if focus_prior_nm is None else float(focus_prior_nm)
        self._tilt_settings(fit_tilt, tilt_rad, tilt_prior_rad)

    def with_focus(self, focus_nm: float, focus_prior_nm: float | None) -> InSituGeometry:
        """The same geometry with another focus and prior: its tilt's settings kept."""
        import copy
        new = copy.copy(self)
        new.focus_nm = float(focus_nm)
        new.focus_prior_nm = None if focus_prior_nm is None else float(focus_prior_nm)
        return new

    def specs(self, batch, start) -> list[ParamSpec]:
        prior = None if self.focus_prior_nm is None else Gaussian(self.focus_nm, self.focus_prior_nm)
        return [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"]),
                ParamSpec("geometry.focus", (), init=self.focus_nm, prior=prior, fixed=not self.fit_focus)
                ] + self._tilt_specs()

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, K, 4] nm: x, y, the defocus through the immersion, the height in the sample."""
        xyz = p["emitter.xyz"]
        z_k = batch.z_nm.to(xyz.dtype)
        z_k = z_k[None] if z_k.dim() == 1 else z_k                        # [1|E, K]
        E, K = xyz.shape[0], z_k.shape[-1]
        h = xyz[:, None, 2].expand(E, K)
        return torch.stack([xyz[:, None, 0].expand(E, K), xyz[:, None, 1].expand(E, K),
                            (z_k - p["geometry.focus"]).expand(E, K), h], -1)


#: each sweep's focus about its start (nm), and the focal plane's slope (nm of focus per µm of field): the
#: start is the width curve's focus of the sweep's file, within a few hundred nm; 50 nm/µm is 0.05 rad, as
#: loose as the coverslip's tilt prior
SWEEP_FOCUS_PRIOR_NM, PLANE_PRIOR_NM_PER_UM = 300.0, 50.0
#: the molecules' common height above the glass: started at the glass (0.5 nm: a logistic cannot start on its
#: bound, and its slope there is the height itself, so not much lower), bounded to (0, 200) nm
HEIGHT_START_NM, HEIGHT_MAX_NM = 0.5, 200.0
#: how firmly each molecule's own z offset is held at the plane while learning (nm)
OFFSET_HOLD_NM = 0.5


class CoverslipStackGeometry:
    """Single molecules on the coverslip, imaged through repeated z sweeps (`docs/uipsf/molecule-stacks.md`).

    A detection in frame k of sweep s, at chip position (u, v) µm from the field's centre, is at defocus

        z_k − f_s − (a·u + b·v) + δz_e,

    z_k the stage's nominal z in SMAPpy's sign (`Batch.z_nm`, [E, 1]), f_s the focus of its sweep
    (`Batch.stack_id`: drift between sweeps; within one, about ten seconds, the stage holds), (a, b) the
    focal plane's slope over the field -- the coverslip's tilt, shared by every sweep of one stage position
    -- and δz_e the molecule's own offset from that plane (``emitter.xyz``'s z).  While learning δz_e is
    held at 0 by a tight prior: z is known, and a free z per molecule would take up what the pupil should
    learn.  The z check (``free_z``) frees it, and how far each molecule then moves is the z bend.

    Every molecule is at one height h above the glass (``geometry.height``), its z in the sample, the fourth
    coordinate: the dye's linker and whatever holds it.  It is the sample's, never the optics', and it is the
    only thing the supercritical (SAF) light sees: exp(−2πκh), the same for every molecule, whatever the
    focus, the plane's slope or a molecule's δz, which are all the defocus's.  The PSF reads it through the SAF
    alone (`VectorPSF.saf_only_depth`): the propagating phase a molecule at h has is every molecule's, and the
    pupil's phase and f_s take it up; with it in, it held the sharp feature at the critical angle the Zernikes
    cannot, and h came to 27 nm on M5's AF647 where the SAF alone gives 10 (3 November 2026).  The pupil saved
    is the objective's, that phase taken out (`VectorPSF.for_emitters`): an emitter at any height is then drawn
    with the whole physics of its own z.  Learnt from the glass, h = 0; ``height_nm`` holds it at a value
    instead (the window's *more*: when it runs off).  One h for all molecules is what the data can tell: a
    single frame cannot tell a molecule's own height from its δz.

    Each molecule otherwise learns x, y, photons and background.  ``centre_px`` is the field's centre in
    chip pixels (`Batch.field_xy`'s), ``pixel_nm`` the pixel.  No coverslip tilt of the layers: with an oil
    objective a point at the glass sees none (the module docstring), and the focal plane's slope is the
    tilt as these data see it."""

    fit_tilt, tilt_rad, tilt_prior_rad = False, None, None

    def __init__(self, sweeps: int, focus_nm, centre_px, pixel_nm, height_nm: float | None = None,
                 fit_plane: bool = True, free_z: bool = False, focus_prior_nm: float = SWEEP_FOCUS_PRIOR_NM,
                 plane_prior: float = PLANE_PRIOR_NM_PER_UM):
        self.sweeps = int(sweeps)
        self.focus_nm = torch.as_tensor(focus_nm, dtype=torch.float32).reshape(self.sweeps)
        self.centre_px = tuple(float(v) for v in centre_px)
        self.pixel_nm = tuple(float(v) for v in pixel_nm)
        self.height_nm = None if height_nm is None else float(height_nm)
        self.fit_plane, self.free_z = bool(fit_plane), bool(free_z)
        self.focus_prior_nm, self.plane_prior = float(focus_prior_nm), float(plane_prior)

    def checking(self) -> CoverslipStackGeometry:
        """The same geometry with every molecule's z free: the z check's."""
        import copy
        new = copy.copy(self)
        new.free_z = True
        return new

    def specs(self, batch, start) -> list[ParamSpec]:
        from ..params import Bounded
        dt, dev = batch.data.dtype, batch.data.device
        hold = Gaussian(torch.zeros(3, dtype=dt, device=dev),
                        torch.tensor([1e9, 1e9, OFFSET_HOLD_NM], dtype=dt, device=dev))
        f0 = self.focus_nm.to(dtype=dt, device=dev)
        h = HEIGHT_START_NM if self.height_nm is None else self.height_nm
        return [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"], prior=None if self.free_z else hold),
                ParamSpec("geometry.focus", (self.sweeps,), init=f0, prior=Gaussian(f0, self.focus_prior_nm)),
                ParamSpec("geometry.plane", (2,), init=0.0, prior=Gaussian(0.0, self.plane_prior),
                          fixed=not self.fit_plane),
                ParamSpec("geometry.height", (), init=h, transform=Bounded(0.0, HEIGHT_MAX_NM),
                          fixed=self.height_nm is not None)]

    def plane_focus(self, p, batch) -> torch.Tensor:
        """[E] nm: where the focal plane is for each molecule, its sweep's focus and the slope at its place."""
        xyz = p["emitter.xyz"]
        uv = (batch.field_xy.to(xyz) - xyz.new_tensor(self.centre_px)) * xyz.new_tensor(self.pixel_nm) / 1000
        return p["geometry.focus"][batch.stack_id.to(xyz.device)] + (uv * p["geometry.plane"]).sum(-1)

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, K, 4] nm: x, y, the defocus through the immersion, the height in the sample."""
        xyz = p["emitter.xyz"]
        E = xyz.shape[0]
        z_k = batch.z_nm.to(xyz.dtype).reshape(E, -1)                                   # [E, K]
        K = z_k.shape[1]
        defocus = z_k - self.plane_focus(p, batch)[:, None] + xyz[:, None, 2]
        h = p["geometry.height"].to(xyz.dtype).expand(E, K)
        return torch.stack([xyz[:, None, 0].expand(E, K), xyz[:, None, 1].expand(E, K), defocus, h], -1)
