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

**Vectorial or scalar** (``kind``, vectorial by default).  The scalar model is v1's: one field per arm, kept
for comparison with v1 and for optics the vectorial model does not describe.  The vectorial one
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

from ..optics.zernike import noll_to_nm, symmetric_modes
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
                 magnitude_n_max: int | None = None, magnitude_modes=None, upsample: int = 1,
                 bead_radius_nm: float = 0.0, sigma_nm=(60.0, 60.0), steps_rad=None, deltas_rad=None,
                 coherence: float = 0.8, transforms=None, fit_transform: bool = True, intensity=None, background=None,
                 pupils=None, kind: str = "vector", plates: str = "aligned", polarisations=None,
                 round_sigma: bool = False, same_magnitude: bool = False):
        """``grid`` the pupil grid (both objectives share the optics); ``roi_shape`` (Y, X) camera pixels;
        ``steps_rad`` the mirror steps as set (only their differences from the first matter), the start of
        ``fourpi.steps``; ``deltas_rad`` the channels' phase offsets to start from (default 0, π/2, π,
        3π/2); ``transforms`` each channel c ≥ 1's 3×3 map onto channel 0's chip pixels, the identity by
        default.  ``magnitude_modes`` the pupils' magnitude modes, if not those up to ``magnitude_n_max``;
        with neither, a vectorial model's are the rotationally symmetric ones (`optics.zernike.symmetric_modes`,
        an objective's apodisation: free in every mode the learning had three optima, docs/4pi.md §20) and a
        scalar model's every mode to ``n_max``, which it needs.  ``pupils`` (upper, lower)
        replaces the two `ZernikePupil`s, for a first stage.  ``kind`` "scalar" or "vector", and for a
        vectorial model the ``plates`` ("aligned", "crossed" or "none") and each channel's polarisation, s
        or p (default s, p, s, p: the 4Pi-SMS's channels in pairs half a turn apart, the pairs a quarter
        turn), as the module docstring says.  ``round_sigma`` makes the blur one σ for x and y
        (`VectorPSF`'s): the window's choice, an x and a y σ apart being astigmatism the pupils should hold.
        ``same_magnitude`` gives both objectives one magnitude, each in its own frame: twin objectives have
        the same apodisation, and how much light each arm brings is the balance's (``fourpi.balance``), the
        magnitude's scale being the reference energy's.  The lower pupil then has no ``lo.pupil.magnitude``
        and reads the upper one's through `lower_view`."""
        if kind not in ("scalar", "vector"):
            raise ValueError(f"kind must be 'scalar' or 'vector', got {kind!r}")
        if plates not in PLATES:
            raise ValueError(f"plates must be one of {', '.join(PLATES)}, got {plates!r}")
        self.grid = grid
        self.kind, self.plates = kind, plates
        self.same_magnitude = bool(same_magnitude)
        #: the lower arm's axial dipole field against the upper's: reversed, as the module docstring
        #: derives; +1 only to test that against data
        self.axial_sign = -1.0
        self.n_channels, self.n_phases = int(n_channels), int(n_phases)
        self.polarisations = tuple(polarisations or ("s", "p") * ((self.n_channels + 1) // 2))[:self.n_channels]
        if len(self.polarisations) != self.n_channels or set(self.polarisations) - {"s", "p"}:
            raise ValueError(f"one polarisation, s or p, per channel: got {self.polarisations}")
        if pupils is None:
            if magnitude_modes is None and magnitude_n_max is None and kind == "vector":
                magnitude_modes = symmetric_modes(n_max)
            pupils = [ZernikePupil(grid, n_max=n_max, magnitude_n_max=magnitude_n_max,
                                   magnitude_modes=magnitude_modes) for _ in range(2)]
        self.psf_up = VectorPSF(grid, pupils[0], roi_shape, upsample=upsample, kind=kind,
                                bead_radius_nm=bead_radius_nm, sigma_nm=sigma_nm, round_sigma=round_sigma)
        self.psf_lo = VectorPSF(grid, pupils[1], roi_shape, upsample=upsample, kind=kind,
                                bead_radius_nm=bead_radius_nm, sigma_nm=sigma_nm, round_sigma=round_sigma)
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
        specs = self.psf_up.specs() + [ParamSpec(LOWER + s.name, s.shape) for s in self._lower_specs()]
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
        # the slices' step, not `Batch.z_step_nm`'s: that averages over the observations, where each slice's z
        # is repeated once per mirror step, and gave a third of the step for three -- the bead kernel then
        # damped the fringe by 28 % instead of 3.5 % and the coherence rose to make up for it
        step = float((z[..., 1:] - z[..., :-1]).abs().mean()) if self.psf_up.bead_radius_nm > 0 else None
        mu = []
        for c in range(C):
            parts = self.images(p, self.positions(p, batch, xyz, c), step, self.polarisations[c])   # [E, 4, J, Y, X]
            incoherent = a[c] * parts[:, 0] + b[c] * parts[:, 1]                    # [E, J, Y, X]
            cross = (cos[:, None, :, c, None, None] * parts[:, 2, :, None]
                     - sin[:, None, :, c, None, None] * parts[:, 3, :, None])        # [E, J, M, Y, X]
            mu.append(N[..., None, None] * (incoherent[:, :, None] + m[c] * cross))
        mu = torch.stack(mu, 1).flatten(2, 3)                       # [E, C, K, Y, X], k = j·M + m
        return mu + self.background(p, batch)

    # ------------------------------------------------------------------ the linearisation
    #
    # `Problem` builds the Jacobian from these instead of a JVP through the whole model per column.  A JVP
    # cost 2.9 forward passes, and the 4Pi model has ~220 columns (two pupils, 63 photon counts per bead):
    # 49 s per normal equations for 12 beads on Apple's GPU.  Here a pupil column is one matrix DFT per
    # channel, and the emitter's columns -- and the shares, channel offsets and maps -- products of four
    # PSF tangents (x, y, z and the interference phase) with the derivatives of small components.

    #: tangents per batch for `Problem`: with every column but the blur's linearised, the JVPs' memory
    #: budget no longer applies (it gave 1).  29 beads × 123 observations, vectorial, Apple's GPU: 1 tangent
    #: 17.1 s per normal equations, 4: 13.4 s, 8: 12.4 s, 14 GB each
    tangents = 8

    def psf_names(self) -> set[str]:
        """The parameters that change the images otherwise than through the coordinates of
        `emitter_linearisation` (position, phase, the arms' balance, the coherence, the misalignment) and the
        photons: the pupils, whose columns come from `pupil_linearisation`, and the blur, by JVPs."""
        return {s.name for s in self.psf_up.specs()} | {LOWER + s.name for s in self._lower_specs()}

    def _lower_specs(self):
        """The lower pupil's own parameters: all of its pupil's, but its magnitude when it is the upper's."""
        return [s for s in self.psf_lo.pupil.specs() if not (self.same_magnitude and s.name == "pupil.magnitude")]

    def lower_name(self, name: str) -> str:
        """The parameter the lower pupil's ``name`` is: its own, or with ``same_magnitude`` the upper's."""
        return name if self.same_magnitude and name == "pupil.magnitude" else LOWER + name

    def _components(self, h: torch.Tensor, arm: int, polarisation: str) -> torch.Tensor:
        """The complex field each channel's light is made of, [..., D, uY, uX], from an image field
        [..., F, uY, 2, uX]: the scalar field (D = 1), or one polarisation's field per dipole (D = 3), as
        `images` forms them."""
        field = torch.complex(h[..., 0, :], h[..., 1, :])
        if self.kind == "scalar":
            return field
        on_x, on_y = PLATES[self.plates][arm][polarisation]
        out = on_x * field[..., :3, :, :] + on_y * field[..., 3:, :, :]
        if arm == 1:
            out = out * out.new_tensor([1.0, 1.0, self.axial_sign])[:, None, None]
        return out

    @property
    def _norm(self) -> float:
        return self.psf_up.norm * (1.0 if self.kind == "scalar" else 2.0)

    def _arms(self, p, batch, xyz: torch.Tensor, step):
        """Every channel's two arms: (pad, Δz, (u, l), tangent_up, tangent_lo), u and l the arms' field
        components (`_components`), [C, E, J', D, uY, uX], and ``tangent_*(dA)`` their changes for tangent
        pupils dA [T, P, P] of that arm, [T, C, E, J', D, uY, uX] (`VectorPSF.field_linearisation`: the
        channels share z, so one pupil-plane product per arm), each as a pair (Re, Im).  The channels
        stacked, so that each later step is one launch, not four: on Apple's GPU the launches, not the
        arithmetic, were the cost; and real, not complex: complex products ran 1.5-2× slower there."""
        up, lo = self.psf_up, self.psf_lo
        at_up, at_lo = [], []
        for c in range(self.n_channels):
            xyz_c, pad, dz = up._stack(self.positions(p, batch, xyz, c), step)
            half = p["fourpi.offset"].to(xyz_c.dtype) / 2
            at_up.append(xyz_c + half)
            at_lo.append(torch.cat([xyz_c[..., :2] - half[:2], -(xyz_c[..., 2:3] - half[2])], -1))
        h_up, t_up = up.field_linearisation(p, at_up)
        h_lo, t_lo = lo.field_linearisation(lower_view(p), at_lo)
        pols = self.polarisations

        def stacked(hs, arm):                                           # (Re, Im), each [C, E, J', D, uY, uX]
            z = torch.stack([self._components(h, arm, pols[c]) for c, h in enumerate(hs)], -6)
            return z.real.contiguous(), z.imag.contiguous()

        fields = stacked(h_up, 0), stacked(h_lo, 1)

        def tangent_up(dA):
            return stacked(t_up(dA), 0)                                 # [T, C, E, J', D, uY, uX] each

        def tangent_lo(dA):
            return stacked(t_lo(dA), 1)

        return pad, dz, fields, tangent_up, tangent_lo

    def _parts(self, u, l):
        """The four images before blur, [..., 4, J', uY, uX]: |u|², |l|², Re and Im of u*·l summed over the
        components, from the fields as (Re, Im) pairs [..., J', D, uY, uX]."""
        (ur, ui), (lr, li) = u, l
        parts = torch.stack([_over_d(ur * ur + ui * ui), _over_d(lr * lr + li * li),
                             _over_d(ur * lr + ui * li), _over_d(ur * li - ui * lr)], -3)
        return parts.movedim(-3, -4) * self._norm

    def _d_parts(self, u, l, du, dl):
        """The change of `_parts` for changes du, dl of the fields (None: unchanged), all (Re, Im) pairs: the
        product rule, d|u|² = 2 Re(u*·du), d(u*·l) = du*·l + u*·dl."""
        (ur, ui), (lr, li) = u, l
        d0 = d1 = cr = ci = 0
        if du is not None:
            dr, di = du
            d0 = 2 * _over_d(ur * dr + ui * di)
            cr = cr + _over_d(dr * lr + di * li)
            ci = ci + _over_d(dr * li - di * lr)
        if dl is not None:
            dr, di = dl
            d1 = 2 * _over_d(lr * dr + li * di)
            cr = cr + _over_d(ur * dr + ui * di)
            ci = ci + _over_d(ur * di - ui * dr)
        like = cr
        d0 = d0 if torch.is_tensor(d0) else torch.zeros_like(like)
        d1 = d1 if torch.is_tensor(d1) else torch.zeros_like(like)
        return torch.stack([d0, d1, cr, ci], -3).movedim(-3, -4) * self._norm

    def _step(self, z) -> float | None:
        return float((z[..., 1:] - z[..., :-1]).abs().mean()) if self.psf_up.bead_radius_nm > 0 else None

    def _slices(self, p, batch):
        """(xyz [E, J, 3], the slices' z, J, M): each bead's position per slice, as `__call__` forms it."""
        M, K = self.n_phases, batch.data.shape[2]
        E = p["emitter.xyz"].shape[0]
        z = batch.z_nm.to(p["emitter.xyz"].dtype)
        z = z.reshape(K // M, M)[None, :, 0] if z.dim() == 1 else z.reshape(E, K // M, M)[..., 0]
        xyz = p["emitter.xyz"][:, None, :] + torch.stack([torch.zeros_like(z)] * 2 + [z], -1)
        return xyz, z, K // M, M

    def pupil_linearisation(self, p: dict[str, torch.Tensor], batch):
        """[(None, names, field, apply)], one per objective: its pupil's parameters, ``field(p)`` its pupil
        field and ``apply(dA)`` the change of μ in every channel, [T, E, C, K, Y, X], for tangent pupils dA
        [T, P, P] (`ForwardModel.pupil_linearisation`; None: all channels, since both pupils reach each)."""
        with torch.no_grad():
            xyz, z, J, M = self._slices(p, batch)
            step = self._step(z)
            a, b, m = self.light(p)
            psi = self.phases(p, p["emitter.phase"])
            cos, sin = torch.cos(psi), torch.sin(psi)
            N = self.intensity(p, batch).reshape(xyz.shape[0], J, M)
            sigma = self.psf_up.blur(p)
            pad, dz, fields, t_up, t_lo = self._arms(p, batch, xyz, step)

        def apply_for(arm):
            def apply(dA):
                u, l = fields
                d = (self._d_parts(u, l, t_up(dA), None) if arm == 0 else self._d_parts(u, l, None, t_lo(dA)))
                # mixed before the bead kernel and the blur, which act along the slices and pixels only: one
                # image per mirror step instead of four, so M transforms rather than 4
                d = self.psf_up._image(self._premix(d, a, b, m, cos, sin), sigma, pad, dz)   # [T, C, E, M, J, Y, X]
                d = d * N.permute(0, 2, 1)[..., None, None]
                return d.transpose(-4, -3).flatten(-4, -3).movedim(-5, -4)
            return apply

        # a shared magnitude is in both groups: the solver adds the two arms' columns (`Problem`)
        up_names = [s.name for s in self.psf_up.pupil.specs()]
        return [(None, up_names, lambda q: self.psf_up.pupil_field(q), apply_for(0)),
                (None, [self.lower_name(n) for n in up_names], lambda q: self.psf_lo.pupil_field(lower_view(q)),
                 apply_for(1))]

    @staticmethod
    def _premix(parts, a, b, m, cos, sin):
        """`_mix` without the photons, which change along the slices: [..., C, E, 4, J', uY, uX] ->
        [..., C, E, M, J', uY, uX], linear along the slices and pixels, so it commutes with `_image`."""
        def per_channel(v):
            return v.reshape(-1, 1, 1, 1, 1, 1)                                  # [C, 1, 1, 1, 1, 1]
        cos, sin = (v.permute(2, 0, 1)[..., None, None, None] for v in (cos, sin))       # [C, E, M, 1, 1, 1]
        P = parts[..., None, :, :, :]                                            # [..., C, E, 4, 1, J', Y, X]
        return (per_channel(a) * P[..., 0, :, :, :, :] + per_channel(b) * P[..., 1, :, :, :, :]
                + per_channel(m) * (cos * P[..., 2, :, :, :, :] - sin * P[..., 3, :, :, :, :]))

    @staticmethod
    def _mix(parts, a, b, m, cos, sin, N):
        """μ from every channel's four images [..., C, E, 4, J, Y, X], as `__call__` mixes them, with a, b, m
        [C], cos and sin of ψ [E, M, C] and the photons N [E, J, M]: [..., E, C, K, Y, X]."""
        def per_channel(v):
            return v.reshape(-1, 1, 1, 1, 1, 1)                                  # [C, 1, 1, 1, 1, 1]
        cos, sin = (v.permute(2, 0, 1)[:, :, None, :, None, None] for v in (cos, sin))   # [C, E, 1, M, 1, 1]
        P = parts[..., None, :, :]                                               # [..., C, E, 4, J, 1, Y, X]
        incoherent = per_channel(a) * P[..., 0, :, :, :, :] + per_channel(b) * P[..., 1, :, :, :, :]
        cross = cos * P[..., 2, :, :, :, :] - sin * P[..., 3, :, :, :, :]
        mu = N[..., None, None] * (incoherent + per_channel(m) * cross)          # [..., C, E, J, M, Y, X]
        return mu.flatten(-4, -3).movedim(-5, -4)

    def emitter_linearisation(self, p: dict[str, torch.Tensor], batch, shared_free=None):
        """(I, dI, parts) as `ForwardModel.emitter_linearisation`: μ = f_c N·I + b, I [E, C, K, Y, X] each
        channel's image per photon in it, and dI [9, E, C, K, Y, X] its derivatives with respect to nine
        coordinates of each observation in each channel: its x, y and z in the channel, its interference
        phase ψ, the channel's balance r_c and coherence α_c, and the objectives' misalignment Δ (x, y, z).
        ``parts(p)`` gives those coordinates [E, C, K, 9], the photons f_c N [E, C, K] and the background.
        So the solver builds the columns of the emitter's parameters, the shares, steps, channel offsets and
        maps, the balance, the coherence and the misalignment from these, and only the pupils (their own
        linearisation) and the blur need more.

        A position moves both arms: the upper sees it at +z, the lower at −z, so z's tangent pupils are
        2πi k_z A_up and −2πi k_z A_lo, x's and y's −2πi k A in both.  The misalignment moves the upper arm
        by +Δ/2 and the lower one by −Δ/2, so its tangents are half the position's, the lower arm's negated.
        With a bead kernel these are the derivatives for a shift of the whole stack, as `ForwardModel`'s.
        ψ, r and α enter after the bead and the blur, I = r P0 + (1 − r) P1 + 2α√(r(1 − r)) (cos ψ P2 −
        sin ψ P3), so their derivatives are exact whatever moves them."""
        import math
        with torch.no_grad():
            xyz, z, J, M = self._slices(p, batch)
            step = self._step(z)
            a, b, m = self.light(p)
            f = a + b
            r, alpha = p["fourpi.balance"], p["fourpi.coherence"]
            root = torch.sqrt((r * (1 - r)).clamp_min(1e-12))
            psi = self.phases(p, p["emitter.phase"])
            cos, sin = torch.cos(psi), torch.sin(psi)
            ones = torch.ones(xyz.shape[0], J, M, dtype=xyz.dtype, device=xyz.device)
            one, zero = torch.ones_like(r), torch.zeros_like(r)
            sigma = self.psf_up.blur(p)
            g = self.grid
            two_pi_i = 2j * math.pi
            kx = (-two_pi_i * g.kx)[None, :].expand_as(g.kz_imm)
            ky = (-two_pi_i * g.ky)[:, None].expand_as(g.kz_imm)
            kz = two_pi_i * g.kz_imm
            A_up, A_lo = self.psf_up.pupil_field(p), self.psf_lo.pupil_field(lower_view(p))
            dA_up = torch.stack([kx, ky, kz]).to(A_up.dtype) * A_up
            dA_lo = torch.stack([kx, ky, -kz]).to(A_lo.dtype) * A_lo
            pad, dz, (u, l), t_up, t_lo = self._arms(p, batch, xyz, step)
            du, dl = t_up(dA_up), t_lo(dA_lo)                                        # [3, C, E, J', D, uY, uX]
            parts = self.psf_up._image(self._parts(u, l), sigma, pad, dz)            # [C, E, 4, J, Y, X]
            minus = (-dl[0], -dl[1])
            d = self.psf_up._image(torch.cat([self._d_parts(u, l, du, dl), 0.5 * self._d_parts(u, l, du, minus)]),
                                   sigma, pad, dz)                                   # [6, C, E, 4, J, Y, X]
            per_photon = f[:, None, None, None]                                      # μ = f_c N·I
            I = self._mix(parts, a, b, m, cos, sin, ones) / per_photon
            d_xyz = self._mix(d, a, b, m, cos, sin, ones) / per_photon               # x, y, z, then Δ
            # ψ: the cross term turned a quarter, −sin ψ P2 − cos ψ P3
            d_psi = self._mix(parts, zero, zero, 2 * alpha * root, -sin, cos, ones)
            d_r = self._mix(parts, one, -one, alpha * (1 - 2 * r) / root, cos, sin, ones)
            d_alpha = self._mix(parts, zero, zero, 2 * root, cos, sin, ones)
            dI = torch.cat([d_xyz[:3], torch.stack([d_psi, d_r, d_alpha]), d_xyz[3:]])

        def parts(q):
            xyz, _, J, M = self._slices(q, batch)
            psi = self.phases(q, q["emitter.phase"])                                # [E, M, C]
            a, b, _ = self.light(q)
            f = a + b
            N = self.intensity(q, batch)                                            # [E, K]
            E, K = N.shape
            pos = []
            for c in range(self.n_channels):
                at = self.positions(q, batch, xyz, c).repeat_interleave(M, 1)       # [E, K, 3]
                ps = psi[:, :, c].repeat(1, J)                                      # [E, K], k = j·M + m
                light = torch.stack([q["fourpi.balance"][c], q["fourpi.coherence"][c]]).to(at.dtype)
                offset = q["fourpi.offset"].to(at.dtype)
                pos.append(torch.cat([at, ps[..., None], light.expand(E, K, 2), offset.expand(E, K, 3)], -1))
            Nc = f[None, :, None] * N[:, None, :]
            return torch.stack(pos, 1), Nc, self.background(q, batch)

        return I, dI, parts

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


def _over_d(x: torch.Tensor) -> torch.Tensor:
    """The sum over the field's components, [..., D, uY, uX] -> [..., uY, uX]; the scalar model's one
    component taken, not summed: a sum over a middle axis of length 1 took Apple's GPU 0.3 s for eight
    pupil tangents of 29 beads, the vectorial model's three 0.03 s."""
    return x[..., 0, :, :] if x.shape[-3] == 1 else x.sum(-3)


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
        a = torch.as_tensor(shared[f"pupil.{which}"])
        b = torch.as_tensor(shared.get(f"{LOWER}pupil.{which}", a))          # a shared magnitude: the same
        parity = a.new_tensor([(-1.0) ** abs(noll_to_nm(j)[1]) for j in modes])
        out[f"pupil.{which}"] = sign * parity * b
        if f"{LOWER}pupil.{which}" in shared:
            out[f"{LOWER}pupil.{which}"] = sign * parity * a
    out["fourpi.offset"] = -torch.as_tensor(shared["fourpi.offset"])
    out["fourpi.balance"] = 1 - torch.as_tensor(shared["fourpi.balance"])
    return out
