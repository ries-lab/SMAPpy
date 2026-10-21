"""The forward model: the expected photons μ in every pixel of every observation (plan §3).

    μ[e, c, k, j] = f_c · N[e, k] · I_c(j; xyz_c[e, k]) + b[e, c, k, j]

The PSF I_c of each channel integrates to one photon over the image plane,
so N is a physical total, split over the channels by their photon shares
f_c (Σ f_c = 1).  xyz_c is the emitter's position in channel c: the main
channel's through the channel's transform and defocus offset
(`channels.py`).  Each part is a component that declares its parameters;
the model only composes them, and the solver reads the block structure
from the specs.

**Channels.**  The layout is [E, C, K, Y, X] for every data type.  A model
with one channel is built from its PSF, ``ForwardModel(psf)``; one with
several from a list of `Channel`s, the first the main one (SMAPpy's
"main").  Every channel's ROI has the same shape.

**Emitter independence is a contract** (plan §5.4): an emitter's pixels
depend only on its own parameters and the shared ones, in every channel.
The solver's block structure relies on it, and `tests/uipsf_tests/test_forward.py`
checks it.
"""
from __future__ import annotations

import math

import torch

from ..params import ParamSpec
from .background import Constant
from .channels import Channel, inverse_map, prefix, unpack_transform, view
from .geometry import BeadStackGeometry
from .intensity import Bleaching
from .start import rough_start


class ForwardModel:
    def __init__(self, channels, geometry=None, intensity=None, background=None):
        """``channels``: a PSF (one channel) or a list of `Channel`s, the main one first."""
        self.channels = list(channels) if isinstance(channels, (list, tuple)) else [Channel(channels)]
        self.geometry = geometry if geometry is not None else BeadStackGeometry()
        self.intensity = intensity if intensity is not None else Bleaching()
        self.background = background if background is not None else Constant()

    @property
    def psf(self):
        """The main channel's PSF (the only one of a single-channel model)."""
        return self.channels[0].psf

    @psf.setter
    def psf(self, psf):
        self.channels[0] = Channel(psf) if len(self.channels) == 1 else _with_psf(self.channels[0], psf)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def pixel_size_nm(self) -> tuple[float, float]:
        return self.psf.grid.optics.pixel_size_nm

    def specs(self, batch, start: dict | None = None) -> list[ParamSpec]:
        """Every parameter, emitter-scope ones started from ``start``, else the batch's own (`Batch.start`),
        else the data's rough start (`rough_start`)."""
        if start is None:
            given = getattr(batch, "start", None)
            start = given if given is not None else rough_start(batch, self.pixel_size_nm)
        specs = [s for c, ch in enumerate(self.channels) for s in ch.specs(c, start)]
        return (specs + self.geometry.specs(batch, start) + self.intensity.specs(batch, start)
                + self.background.specs(batch, start))

    def psf_names(self) -> set[str]:
        """The parameters that enter the PSFs themselves (pupils, blur), as opposed to where the emitter
        is and how bright: the solver differentiates the latter through `emitter_linearisation`."""
        return {prefix(c) + s.name for c, ch in enumerate(self.channels) for s in ch.psf.specs()}

    def share(self, p, c: int) -> torch.Tensor | float:
        """f_c: the fraction of the emitter's photons in channel ``c``."""
        if self.n_channels == 1:
            return 1.0
        if c > 0:
            return p[prefix(c) + "share"]
        return 1 - sum(p[prefix(k) + "share"] for k in range(1, self.n_channels))

    def positions(self, p, batch, xyz: torch.Tensor, c: int) -> torch.Tensor:
        """The emitter's position in channel ``c``, [E, K, 3] nm, from the main channel's ``xyz``."""
        if c == 0:
            return xyz
        pre = prefix(c)
        if batch.roi_xy is None:
            raise ValueError("a model with several channels needs each ROI's chip position (Batch.roi_xy)")
        roi = batch.roi_xy.to(xyz.dtype)                                      # [E, C, 2] chip pixels
        px0 = xyz.new_tensor(self.pixel_size_nm)
        pxc = xyz.new_tensor(self.channels[c].pixel_size_nm)
        main = roi[:, None, 0] + xyz[..., :2] / px0
        local = (inverse_map(unpack_transform(p[pre + "transform"]), main) - roi[:, None, c]) * pxc
        return torch.cat([local, (xyz[..., 2] + p[pre + "dz"])[..., None], xyz[..., 3:]], -1)

    def _z_step(self, psf, batch):
        return batch.z_step_nm if psf.bead_radius_nm > 0 else None

    def __call__(self, p: dict[str, torch.Tensor], batch) -> torch.Tensor:
        """μ, [E, C, K, Y, X] photons."""
        C = self.n_channels
        if batch.data.shape[1] != C:
            raise ValueError(f"the batch has {batch.data.shape[1]} channels, the model {C}")
        xyz = self.geometry(p, batch)                                  # [E, K, 3], the main channel's
        N = self.intensity(p, batch)[..., None, None]                  # [E, K, 1, 1]
        mu = []
        for c, ch in enumerate(self.channels):
            pos, depth = _split(self.positions(p, batch, xyz, c))
            I = ch.psf(view(p, c), pos, depth, z_step_nm=self._z_step(ch.psf, batch))
            mu.append(self.share(p, c) * N * I)
        return torch.stack(mu, 1) + self.background(p, batch)

    def pupil_linearisation(self, p: dict[str, torch.Tensor], batch):
        """[(c, names, field, apply)], one per channel: the parameters that act only through channel c's
        pupil field, ``field(p)`` that field, and ``apply(dA)`` the change of μ in channel c alone,
        [T, E, K, Y, X], for tangent fields dA (μ in the other channels does not change).  The solver
        builds the pupil's Jacobian columns from it instead of JVPs through the whole model (plan §9 M7)."""
        xyz = self.geometry(p, batch)
        N = self.intensity(p, batch)[..., None, None]
        groups = []
        for c, ch in enumerate(self.channels):
            pc = view(p, c)
            pos, depth = _split(self.positions(p, batch, xyz, c))
            f = ch.psf.pupil_linearisation(pc, pos, depth, z_step_nm=self._z_step(ch.psf, batch))
            Nc = self.share(p, c) * N
            names = [prefix(c) + s.name for s in ch.psf.pupil.specs()]
            groups.append((c, names, lambda q, c=c, ch=ch: ch.psf.pupil_field(view(q, c)),
                           lambda dA, f=f, Nc=Nc: Nc * f(dA)))
        return groups

    def emitter_linearisation(self, p: dict[str, torch.Tensor], batch):
        """(I, dI, parts): the PSFs I [E, C, K, Y, X], their derivatives dI [3, E, C, K, Y, X] with respect
        to each observation's own x, y and z in each channel, and ``parts(p) -> (xyz, N, b)``, the cheap
        components: positions [E, C, K, 3], photons f_c·N [E, C, K] and background.  In situ the positions
        have a fourth coordinate, the height in the sample, and dI a fourth derivative, its factor
        2πi k_z,med (`InSituGeometry`).

        μ = N·I(xyz) + b, and an emitter's parameters act through xyz, N and b only, so a column of the
        emitter Jacobian is N·Σ_d (∂xyz_d/∂u)·dI_d + (∂N/∂u)·I + ∂b/∂u: three PSF tangents per channel for
        every emitter parameter together, the rest JVPs of the small components.  The position enters the
        pupil only as a factor: x and y through the DFT's exp(2πi k (x − x₀)), z through the defocus
        wave exp(2πi k_z z), so ∂/∂x₀ = −2πi k_x, ∂/∂y₀ = −2πi k_y, ∂/∂z = 2πi k_z times the pupil, and
        `VectorPSF.pupil_linearisation` takes them as tangent pupils (plan §9 M7).  The same holds for any
        shared parameter that acts through xyz, N and b alone (the transform, shares, offsets, a per-slice
        factor): `Problem` builds those columns from it too.
        """
        with torch.no_grad():
            xyz = self.geometry(p, batch)
            Is, dIs = [], []
            for c, ch in enumerate(self.channels):
                pc, z_step = view(p, c), self._z_step(ch.psf, batch)
                xyz_c, depth = _split(self.positions(p, batch, xyz, c))
                g = ch.psf.grid
                Is.append(ch.psf(pc, xyz_c, depth, z_step_nm=z_step))
                A = ch.psf.pupil_field(pc)                                   # [P, P] or [E, P, P]
                two_pi_i = 2j * math.pi
                factors = [(-two_pi_i * g.kx)[None, :].expand_as(g.kz_imm),
                           (-two_pi_i * g.ky)[:, None].expand_as(g.kz_imm), two_pi_i * g.kz_imm]
                if depth is not None:                     # in situ: the height in the sample, through k_z,med
                    factors.append(two_pi_i * g.kz_med)
                factors = torch.stack(factors).to(A.dtype)                   # [3 | 4, P, P]
                dA = factors[:, None] * A if A.dim() == 3 else factors * A
                dIs.append(ch.psf.pupil_linearisation(pc, xyz_c, depth, z_step_nm=z_step)(dA))
            I, dI = torch.stack(Is, 1), torch.stack(dIs, 2)

        def parts(q):
            xyz = self.geometry(q, batch)
            N = self.intensity(q, batch)
            pos = torch.stack([self.positions(q, batch, xyz, c) for c in range(self.n_channels)], 1)
            Nc = torch.stack([self.share(q, c) * N for c in range(self.n_channels)], 1)
            return pos, Nc, self.background(q, batch)

        return I, dI, parts


def _split(positions):
    """(xyz [E, K, 3], depth [E, K] or None): a geometry's positions, with an in situ geometry's fourth
    coordinate, the height in the sample, apart (`InSituGeometry`)."""
    if positions.shape[-1] == 4:
        return positions[..., :3], positions[..., 3]
    return positions, None


def _with_psf(channel, psf):
    import copy
    new = copy.copy(channel)
    new.psf = psf
    return new
