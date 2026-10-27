"""4Pi: two opposing objectives, their fields summed coherently, several phase channels (docs/4pi.md, D7).

**The field.**  Each objective has its own pupil, P_up and P_lo, shared by every channel.  An emitter at
height z above the common focus is at z for the upper objective and at −z for the lower one, which looks
at the sample from the other side.  Channel c receives both arms,

    E_c = t_c,up · F[P_up e^{2πi k_z z}] + t_c,lo · F[P_lo e^{−2πi k_z z}] · e^{iψ_c},
    ψ_c = φ_e + s_m + δ_c,

F the image (the matrix DFT that also places the emitter laterally), φ_e the bead's cavity phase,
s_m the deformable mirrors' phase step m of the stack and δ_c the channel's phase offset.  With h_up and
h_lo the two arms' image fields (each normalised to one photon, as `VectorPSF`), the intensity is

    I_c = a_c |h_up|² + b_c |h_lo|² + 2 α_c √(a_c b_c) Re(h_up* h_lo e^{iψ_c}),

a_c = |t_c,up|² and b_c = |t_c,lo|² the light each arm sends into channel c, α_c the channel's
coherence: the fraction of the light that interferes, α|E_c|² + (1 − α)(a|h_up|² + b|h_lo|²) written
out.  The cross term is linear in e^{iψ}, so the four images |h_up|², |h_lo|², Re and Im of h_up* h_lo
are blurred, convolved with the bead and binned once, and every channel and phase step is a weighted sum
of them: the bead kernel and the blur are v2's (`VectorPSF._image`), applied to each z stack of one
mirror step.

**The parameters** are chosen so that none trades exactly with another (v1's did, docs/4pi.md §2):
- the pupils: Zernikes of each objective, Noll 5 and up, shared by all channels;
- the misalignment of the objectives, ``fourpi.offset`` = (Δx, Δy, Δz) nm, one set for the instrument,
  not one per channel as in v1: the upper arm sees the emitter at +Δ/2, the lower at −Δ/2.  A relative
  piston would be the same as the beads' phases and is not a parameter; Δz moves the two arms' foci
  apart and leaves the interference phase alone (the waves' Δz terms cancel in h_up* h_lo);
- per channel: its share of the light f_c (Σ = 1, so N is the photons of all channels together, the
  calibration's convention, `smappy.fourpi.normalise`), the upper arm's fraction r_c of it
  (a_c = f_c r_c, b_c = f_c (1 − r_c)), the coherence α_c and the phase offset δ_c, δ_0 = 0 the gauge;
- the mirror steps ``fourpi.steps``, s_0 = 0 the gauge (each bead's φ_e absorbs a common step);
- per bead: x, y, z, its photons (a bleaching curve over the acquisition), φ_e and a background per
  channel;
- each channel c ≥ 1's projective map onto channel 0 (``ch<c>.transform``), as `Channel`'s.

**The observations** of a bead are its z slices × mirror steps, z-major (k = j·M + m), the order the
microscope acquires them in, so that the bleaching curve runs along time: ``Batch.z_nm`` is [K] with
each slice's z repeated M times.

**Scalar or vectorial** (``kind``).  The scalar model is v1's: one field per arm.  The vectorial one
(M7, docs/4pi.md §17) follows each arm's field through the 4Pi-SMS's polarisation optics (Wang et al.
2021, Nat Protoc 16:677): a quarter-wave plate right after each objective, its axes at 45° to the setup
plane; a retarder with its axes along s and p in one arm (the quartz wedges, part of δ_c); the 50:50
cube; and a polarising cube that sends s and p to their own images.  So each channel sees one
polarisation (``polarisations``, s or p), and in it the component the plates turn s or p into: behind
plates aligned with each other (``plates="aligned"``, the protocol's) the circular components
(E_x ∓ i E_y)/√2, the same in both arms; ``"crossed"``, the lower plate at −45°, the other handedness in
the lower arm; ``"none"``, E_x and E_y themselves.  In the sample's frame the lower objective sees a
dipole's field with its axial part reversed -- the wave travels down, and the field's projection of μ_z
changes sign -- so its z dipole's field is negated, beside z → −z.  A freely rotating emitter is the
incoherent sum of three orthogonal dipoles, so the cross term is the sum over dipoles of
u_d* l_d, u and l the two arms' fields in the channel's polarisation: where the dipoles' fields differ
between the arms -- the axial one's sign, and the circular components' winding with the pupil's azimuth
-- they interfere less, and the modulation is set by the optics, not only by the coherence α_c.

**The arms' labels are a gauge.**  Swapping the objectives' roles -- the upper pupil the twin of the
lower one, A'(k) = A*(−k), and the other way round, Δ → −Δ and r_c → 1 − r_c -- gives the same images
exactly: the twin's field is the conjugate of the other arm's at the mirrored height
(`docs/uipsf/conventions.md`, "The twin"), and h_up* h_lo is unchanged.  So which objective is "upper" is
whatever the fit starts nearer; `swapped` gives the other labelling, to compare a fit with a truth or with
v1.  The export does not depend on it.

The sign of the fringe: h_up* h_lo carries e^{−4πi k_z z}, so the interference phase falls as z grows:
the model's `phase_per_nm` is negative, as v1's export is (`smappy.fourpi.load_uipsf_v1`).
"""
from __future__ import annotations

import torch

from ..optics.zernike import noll_to_nm
from ..params import Bounded, ParamSpec
from .background import Constant
from .channels import inverse_map, normalised, unpack_transform
from .intensity import Bleaching
from .psf import VectorPSF
from .pupil import ZernikePupil

LOWER = "lo."


def lower_view(p: dict) -> dict:
    """``p`` as the lower objective's PSF reads it: its pupil under the plain names."""
    out = dict(p)
    out.update({k[len(LOWER):]: v for k, v in p.items() if k.startswith(LOWER)})
    return out


class FourPiModel:
    def __init__(self, grid, roi_shape, n_channels: int = 4, n_phases: int = 3, *, n_max: int = 6,
                 magnitude_n_max: int | None = None, upsample: int = 1, bead_radius_nm: float = 0.0,
                 sigma_nm=(60.0, 60.0), steps_rad=None, deltas_rad=None, coherence: float = 0.8,
                 transforms=None, fit_transform: bool = True, intensity=None, background=None,
                 pupils=None, kind: str = "scalar", plates: str = "aligned", polarisations=None):
        """``grid`` the pupil grid (both objectives share the optics); ``roi_shape`` (Y, X) camera pixels;
        ``steps_rad`` the mirror steps as set (only their differences from the first matter), the start of
        ``fourpi.steps``; ``deltas_rad`` the channels' phase offsets to start from (default 0, π/2, π,
        3π/2); ``transforms`` each channel c ≥ 1's 3×3 map onto channel 0's chip pixels, the identity by
        default.  ``pupils`` (upper, lower) replaces the two `ZernikePupil`s, for a first stage.  ``kind``
        "scalar" or "vector", and for a vectorial model the ``plates`` ("aligned", "crossed" or "none") and
        each channel's polarisation, s or p (default s, p, s, p: the 4Pi-SMS's channels in pairs half a
        turn apart, the pairs a quarter turn), as the module docstring says."""
        if kind not in ("scalar", "vector"):
            raise ValueError(f"kind must be 'scalar' or 'vector', got {kind!r}")
        if plates not in PLATES:
            raise ValueError(f"plates must be one of {', '.join(PLATES)}, got {plates!r}")
        self.grid = grid
        self.kind, self.plates = kind, plates
        #: the lower arm's axial dipole field against the upper's: reversed, as the module docstring
        #: derives; +1 only to test that against data
        self.axial_sign = -1.0
        self.n_channels, self.n_phases = int(n_channels), int(n_phases)
        self.polarisations = tuple(polarisations or ("s", "p") * ((self.n_channels + 1) // 2))[:self.n_channels]
        if len(self.polarisations) != self.n_channels or set(self.polarisations) - {"s", "p"}:
            raise ValueError(f"one polarisation, s or p, per channel: got {self.polarisations}")
        if pupils is None:
            pupils = [ZernikePupil(grid, n_max=n_max, magnitude_n_max=magnitude_n_max) for _ in range(2)]
        self.psf_up = VectorPSF(grid, pupils[0], roi_shape, upsample=upsample, kind=kind,
                                bead_radius_nm=bead_radius_nm, sigma_nm=sigma_nm)
        self.psf_lo = VectorPSF(grid, pupils[1], roi_shape, upsample=upsample, kind=kind,
                                bead_radius_nm=bead_radius_nm, sigma_nm=sigma_nm)
        C, M = self.n_channels, self.n_phases
        steps = torch.zeros(M) if steps_rad is None else torch.as_tensor(steps_rad, dtype=torch.float32)
        self.steps_init = (steps - steps[0])[1:]
        deltas = (torch.arange(C) * torch.pi / 2 if deltas_rad is None
                  else torch.as_tensor(deltas_rad, dtype=torch.float32))
        self.deltas_init = (deltas - deltas[0])[1:]
        self.coherence_init = float(coherence)
        self.transforms = [normalised(torch.eye(3) if transforms is None else transforms[c - 1])
                           for c in range(1, C)]
        self.fit_transform = bool(fit_transform)
        self.intensity = intensity if intensity is not None else Bleaching()
        self.background = background if background is not None else Constant()

    @property
    def pixel_size_nm(self) -> tuple[float, float]:
        return self.grid.optics.pixel_size_nm

    @property
    def pupils(self):
        return self.psf_up.pupil, self.psf_lo.pupil

    def with_pupils(self, pupils) -> FourPiModel:
        """The same model with other pupils (a first stage's truncated ones)."""
        import copy
        new = copy.copy(self)
        new.psf_up, new.psf_lo = copy.copy(self.psf_up), copy.copy(self.psf_lo)
        new.psf_up.pupil, new.psf_lo.pupil = pupils
        return new

    def specs(self, batch, start: dict | None = None) -> list[ParamSpec]:
        if start is None:
            start = getattr(batch, "start", None)
        if start is None:
            from .fourpi_start import rough_start
            start = rough_start(batch, self)
        C = self.n_channels
        specs = self.psf_up.specs() + [ParamSpec(LOWER + s.name, s.shape) for s in self.psf_lo.pupil.specs()]
        share = float(1.0 / C)
        specs += [
            ParamSpec("fourpi.offset", (3,), init=0.0),
            ParamSpec("fourpi.steps", (self.n_phases - 1,), init=start.get("steps", self.steps_init)),
            # Bounded, so each share stays a fraction; channel 0's is 1 − the others'.
            ParamSpec("fourpi.share", (C - 1,), init=share, transform=Bounded(0.0, 1.0)),
            ParamSpec("fourpi.balance", (C,), init=0.5, transform=Bounded(0.0, 1.0)),
            ParamSpec("fourpi.coherence", (C,), init=start.get("coherence", self.coherence_init),
                      transform=Bounded(0.0, 1.0)),
            ParamSpec("fourpi.delta", (C - 1,), init=start.get("deltas", self.deltas_init))]
        transforms = start.get("transforms")
        for c in range(1, C):
            H = self.transforms[c - 1] if transforms is None else normalised(transforms[c - 1])
            specs.append(ParamSpec(f"ch{c}.transform", (8,), init=torch.as_tensor(H.reshape(-1)[:8]),
                                   fixed=not self.fit_transform))
        specs += [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"]),
                  ParamSpec("emitter.phase", (), "emitter", init=start.get("phase", 0.0))]
        return specs + self.intensity.specs(batch, start) + self.background.specs(batch, start)

    # ------------------------------------------------------------------ the parts

    def light(self, p) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(a, b, m) per channel, [C] each: the upper and the lower arm's light in the channel and the
        interference's amplitude 2 α √(a b)."""
        rest = p["fourpi.share"]
        f = torch.cat([(1 - rest.sum()).reshape(1), rest])
        r = p["fourpi.balance"]
        a, b = f * r, f * (1 - r)
        return a, b, 2 * p["fourpi.coherence"] * torch.sqrt(a * b)

    def phases(self, p, phase_e: torch.Tensor) -> torch.Tensor:
        """ψ [E, M, C]: each bead's phase, plus each mirror step and each channel's offset."""
        zero = phase_e.new_zeros(1)
        steps = torch.cat([zero, p["fourpi.steps"].to(phase_e.dtype)])
        deltas = torch.cat([zero, p["fourpi.delta"].to(phase_e.dtype)])
        return phase_e[:, None, None] + steps[None, :, None] + deltas[None, None, :]

    def positions(self, p, batch, xyz: torch.Tensor, c: int) -> torch.Tensor:
        """The emitter's position in channel ``c``'s own ROI, [E, K, 3] nm (`ForwardModel.positions`)."""
        if c == 0:
            return xyz
        if batch.roi_xy is None:
            raise ValueError("a 4Pi model needs each channel's ROI position (Batch.roi_xy)")
        roi = batch.roi_xy.to(xyz.dtype)
        px = xyz.new_tensor(self.pixel_size_nm)
        main = roi[:, None, 0] + xyz[..., :2] / px
        local = (inverse_map(unpack_transform(p[f"ch{c}.transform"]), main) - roi[:, None, c]) * px
        return torch.cat([local, xyz[..., 2:]], -1)

    def images(self, p, xyz: torch.Tensor, z_step_nm: float | None = None,
               polarisation: str = "s") -> torch.Tensor:
        """[E, 4, K, Y, X]: |h_up|², |h_lo|², Re and Im of h_up* h_lo for emitters at ``xyz`` [E, K, 3]
        (one z stack each when there is a bead kernel), blurred, convolved and binned; vectorial, in the
        ``polarisation`` a channel sees, summed over the dipoles (module docstring)."""
        up, lo = self.psf_up, self.psf_lo
        xyz, pad, dz = up._stack(xyz, z_step_nm)
        half = p["fourpi.offset"].to(xyz.dtype) / 2
        at_up = xyz + half
        at_lo = torch.cat([xyz[..., :2] - half[:2], -(xyz[..., 2:3] - half[2])], -1)
        h_up, _, _ = up._image_field(p, at_up, None)               # [E, K, F, uY, 2, uX], (Re, Im)
        h_lo, _, _ = lo._image_field(lower_view(p), at_lo, None)
        if self.kind == "scalar":
            ur, ui = h_up[..., 0, :], h_up[..., 1, :]
            lr, li = h_lo[..., 0, :], h_lo[..., 1, :]
            parts = torch.stack([ur * ur + ui * ui, lr * lr + li * li, ur * lr + ui * li, ur * li - ui * lr], 1)
            parts = parts.sum(3) * up.norm                         # [E, 4, K, uY, uX]
            return up._image(parts, up.blur(p), pad, dz)
        u = analysed(h_up, *PLATES[self.plates][0][polarisation])  # [E, K, 3, uY, uX] complex, per dipole
        lo_field = analysed(h_lo, *PLATES[self.plates][1][polarisation])
        lo_field = lo_field * lo_field.new_tensor([1.0, 1.0, self.axial_sign])[:, None, None]   # μ_z seen from below
        cross = (u.conj() * lo_field).sum(2)
        # ×2: one polarisation of an unpolarised emitter is half its light, and the shares f_c count a
        # channel's light as if it had it all, as the scalar model does
        parts = torch.stack([(u.abs() ** 2).sum(2), (lo_field.abs() ** 2).sum(2), cross.real, cross.imag], 1)
        return up._image(parts * (2 * up.norm), up.blur(p), pad, dz)

    # ------------------------------------------------------------------ the model

    def __call__(self, p: dict[str, torch.Tensor], batch) -> torch.Tensor:
        """μ, [E, C, K, Y, X] photons."""
        C, M = self.n_channels, self.n_phases
        E, Cb, K = batch.data.shape[:3]
        if Cb != C or K % M:
            raise ValueError(f"the batch has {Cb} channels and {K} observations; the model {C} channels "
                             f"and {M} phase steps per slice")
        z = batch.z_nm.to(p["emitter.xyz"].dtype)
        z = z.reshape(K // M, M)[None, :, 0] if z.dim() == 1 else z.reshape(E, K // M, M)[..., 0]   # [1|E, J]
        xyz = p["emitter.xyz"][:, None, :] + torch.stack([torch.zeros_like(z)] * 2 + [z], -1)    # [E, J, 3]
        a, b, m = self.light(p)
        psi = self.phases(p, p["emitter.phase"])                   # [E, M, C]
        cos, sin = torch.cos(psi), torch.sin(psi)
        N = self.intensity(p, batch).reshape(E, K // M, M)          # [E, J, M]
        step = batch.z_step_nm if self.psf_up.bead_radius_nm > 0 else None
        mu = []
        for c in range(C):
            parts = self.images(p, self.positions(p, batch, xyz, c), step, self.polarisations[c])   # [E, 4, J, Y, X]
            incoherent = a[c] * parts[:, 0] + b[c] * parts[:, 1]                    # [E, J, Y, X]
            cross = (cos[:, None, :, c, None, None] * parts[:, 2, :, None]
                     - sin[:, None, :, c, None, None] * parts[:, 3, :, None])        # [E, J, M, Y, X]
            mu.append(N[..., None, None] * (incoherent[:, :, None] + m[c] * cross))
        mu = torch.stack(mu, 1).flatten(2, 3)                       # [E, C, K, Y, X], k = j·M + m
        return mu + self.background(p, batch)

    def channel_images(self, p, xyz: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """[C, N, Y, X]: each channel's image of one photon at ``xyz`` [N, 3], with ψ_c = ``phase`` + δ_c
        (no mirror step, no bead), each channel's emitter on its own ROI's centre: what the IAB export
        samples (`workflows.fourpi.calibration`)."""
        a, b, m = self.light(p)
        deltas = torch.cat([phase.new_zeros(1), p["fourpi.delta"].to(phase.dtype)])
        parts = {pol: self.images(p, xyz[None], polarisation=pol)[0] for pol in set(self.polarisations)}
        psi = phase[None, :] + deltas[:, None]                       # [C, N]
        out = []
        for c, pol in enumerate(self.polarisations):
            q = parts[pol]                                           # [4, N, Y, X]
            out.append(a[c] * q[0] + b[c] * q[1]
                       + m[c] * (torch.cos(psi[c])[:, None, None] * q[2] - torch.sin(psi[c])[:, None, None] * q[3]))
        return torch.stack(out)


#: each plate setting's Jones row for s and p, (on E_x, on E_y), in the upper arm and the lower one: a
#: quarter-wave plate at ±45° (a common phase dropped) turns s and p into the circular components
_R = 2 ** -0.5
_ALIGNED = {"s": (_R, -1j * _R), "p": (-1j * _R, _R)}
_CROSSED = {"s": (_R, 1j * _R), "p": (1j * _R, _R)}
PLATES = {"aligned": (_ALIGNED, _ALIGNED), "crossed": (_ALIGNED, _CROSSED),
          "none": ({"s": (1.0, 0.0), "p": (0.0, 1.0)},) * 2}


def analysed(h: torch.Tensor, on_x: complex, on_y: complex) -> torch.Tensor:
    """One polarisation's field per dipole, [E, K, 3, uY, uX] complex, from a vectorial image field
    [E, K, 6, uY, 2, uX] (x then y component of the x, y and z dipoles, as (Re, Im))."""
    field = torch.complex(h[..., 0, :], h[..., 1, :])
    return on_x * field[:, :, :3] + on_y * field[:, :, 3:]


def swapped(model: FourPiModel, shared: dict) -> dict:
    """The same 4Pi model with the arms' labels swapped (module docstring): each pupil the other's twin
    (phase c_j → −(−1)^m c_j, magnitude c_j → (−1)^m c_j), the misalignment and the balance reversed."""
    out = dict(shared)
    up, lo = model.pupils
    if up.phase_modes != lo.phase_modes or up.magnitude_modes != lo.magnitude_modes:
        raise ValueError("swapping the arms needs the same modes in both pupils")
    for which, modes, sign in (("phase", up.phase_modes, -1.0), ("magnitude", up.magnitude_modes, 1.0)):
        a, b = torch.as_tensor(shared[f"pupil.{which}"]), torch.as_tensor(shared[f"{LOWER}pupil.{which}"])
        parity = a.new_tensor([(-1.0) ** abs(noll_to_nm(j)[1]) for j in modes])
        out[f"pupil.{which}"], out[f"{LOWER}pupil.{which}"] = sign * parity * b, sign * parity * a
    out["fourpi.offset"] = -torch.as_tensor(shared["fourpi.offset"])
    out["fourpi.balance"] = 1 - torch.as_tensor(shared["fourpi.balance"])
    return out
