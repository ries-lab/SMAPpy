"""The PSF: a pupil and emitter positions in, expected photons per pixel per photon out.

For each emitter and observation (slice or frame) the pupil is multiplied
by the defocus wave exp(2πi k_z z), its six dipole fields are taken to the
image with the matrix DFT -- which also places the emitter laterally -- and
their intensities are summed.  The result is blurred by a Gaussian, by the
bead's size if it has one, and integrated over pixels by summing an
upsampled grid (uiPSF's ``bin``).  This is uiPSF's model (Suppl. Note 3),
checked against uiPSF v1 and the TU Delft vectorial model in the tests.

**Normalisation.**  The image of an emitter in focus at the coverslip,
through an aberration-free pupil, integrates to 1 over the whole image
plane; the pupil's magnitude is rescaled to keep that energy.  So photons
are physical counts and do not depend on the region's size (brief,
decision 5; vecfit does the same).  Supercritical light lost to depth, and
light falling outside the region, make the sum over a region less than 1.

**Positions** (``xyz``, nm, `docs/uipsf/conventions.md`): x and y relative to the
region's centre, z the emitter's height above the focal plane.  In a bead
stack z = −(objective position), and z enters through the immersion's k_z;
``depth_nm`` (in-situ) adds the emitter's depth in the sample through the
medium's k_z.  With a tilted coverslip (``geometry.tilt``) or one off its design
thickness, the depth and the coverslip enter through the layered model's
phase instead (`optics.layers`).
"""
from __future__ import annotations

import math

import torch

from ..device import compiled
from ..optics import layers
from ..optics.grid import PupilGrid
from ..optics.kernels import bin_pixels, blur_bin_matrices, gaussian_otf, sphere_otf
from ..optics.mdft import MatrixDFT, wide_product
from ..optics.vectorial import dipole_fields, scalar_field
from ..params import Magnitude, ParamSpec


class VectorPSF:
    def __init__(self, grid: PupilGrid, pupil, roi_shape: tuple[int, int], *, upsample: int = 1,
                 kind: str = "vector", analyser: str | None = "unpolarised", bead_radius_nm: float = 0.0,
                 sigma_nm: tuple[float, float] = (0.0, 0.0), fit_sigma: bool = True, round_sigma: bool = False):
        """``roi_shape`` is (Y, X) camera pixels; ``kind`` "vector" or "scalar"; ``analyser``
        "unpolarised" (the default, also None), "x" or "y" (a channel behind a polariser); ``sigma_nm``
        the blur's starting value, or its value with ``fit_sigma=False``; ``round_sigma`` makes it one value
        for x and y (``psf.sigma`` [1]): free, the blur took up 0.1-0.4 rad of in situ astigmatism (M2, October
        2026, docs/uipsf/architecture.md).

        Unpolarised is the default and is said so: v1 put every two-channel vector fit behind an x and
        a y polariser unless told otherwise, so biplane and ratiometric data were fitted with the
        wrong PSFs (review F1)."""
        if kind not in ("vector", "scalar"):
            raise ValueError("kind must be 'vector' or 'scalar'")
        analyser = "unpolarised" if analyser is None else analyser
        if analyser not in ("unpolarised", "x", "y"):
            raise ValueError(f"analyser must be 'unpolarised', 'x' or 'y', got {analyser!r}")
        self.grid, self.pupil = grid, pupil
        self.kind, self.analyser = kind, analyser
        self.roi_shape = (int(roi_shape[0]), int(roi_shape[1]))
        self.upsample = int(upsample)
        self.bead_radius_nm = float(bead_radius_nm)
        self.sigma_init = tuple(float(s) for s in sigma_nm)
        # Fixable while it is unclear whether σ is physical or a sink for mismatch (plan §10, 3).
        self.fit_sigma = bool(fit_sigma)
        self.round_sigma = bool(round_sigma)
        px, py = grid.optics.pixel_size_nm
        self.spacing = (px / self.upsample, py / self.upsample)
        Y, X = self.roi_shape
        self.dft = MatrixDFT(grid, (Y * self.upsample, X * self.upsample), self.spacing)
        fields = (dipole_fields(grid, None if analyser == "unpolarised" else analyser) if kind == "vector"
                  else scalar_field(grid))
        self.fields = grid.aperture * grid.apodisation * fields          # [F, P, P]
        self.weight = (self.fields.abs() ** 2).sum(0)                    # energy of each pupil sample
        self.energy = self.weight.sum()
        # The fields as [Re; Im] along P_y, [2P, F, P]: a pupil a multiplies them as
        # Re a·[Re F; Im F] + Im a·[−Im F; Re F] = [Re; Im] of a·F (`_image_field`).
        Ft = self.fields.permute(1, 0, 2)
        self._fields_re = torch.cat([Ft.real, Ft.imag], 0).contiguous()
        self._fields_im = torch.cat([-Ft.imag, Ft.real], 0).contiguous()
        # |E|² · ΔxΔy integrates the image; E carries Δk² from the DFT, the pupil energy Δk² once.
        self.norm = self.spacing[0] * self.spacing[1] / (self.energy * self.dft.scale)

    def specs(self) -> list[ParamSpec]:
        init = torch.tensor(self.sigma_init)
        init = init.mean().reshape(1) if self.round_sigma else init
        return self.pupil.specs() + [ParamSpec("psf.sigma", tuple(init.shape), init=init,
                                                   transform=Magnitude(), fixed=not self.fit_sigma)]

    @staticmethod
    def blur(p) -> torch.Tensor:
        """The blur's σ (x, y) nm: a round one's single value twice."""
        s = p["psf.sigma"]
        return s.expand(*s.shape[:-1], 2) if s.shape[-1] == 1 else s

    def pupil_field(self, p: dict[str, torch.Tensor]) -> torch.Tensor:
        """The complex pupil without the fields, magnitude rescaled to the reference energy; [..., P, P]."""
        magnitude, phase = self.pupil(p)
        scale = torch.sqrt(self.energy / (self.weight * magnitude ** 2).sum((-2, -1), keepdim=True))
        # Not torch.polar(magnitude, phase): the magnitude 1 + Σ c_j Z_j dips below zero at the rim once its
        # coefficients grow, and polar's backward (torch 2.14) flips the gradient's sign there while its
        # forward mode does not, so value_and_grad disagreed with the Jacobian and L-BFGS's steps failed.
        return torch.polar(torch.ones_like(phase), phase) * (magnitude * scale)

    def intensity(self, p, xyz: torch.Tensor, depth_nm: torch.Tensor | None = None) -> torch.Tensor:
        """[E, K, uY, uX]: the sampled intensity before blur and pixel integration."""
        h, _, _ = self._image_field(p, xyz, depth_nm)
        return (h * h).sum(-2).sum(-3) * self.norm             # one axis at a time (fast on MPS)

    def _image_field(self, p, xyz, depth_nm):
        """(h, By, Bx): the image field of each dipole, [E, K, F, uY, 2, uX] as (Re, Im), and the real DFT
        matrices of each emitter and observation.  Real and laid out wide (`MatrixDFT.apply_wide`): 1.5×
        faster than complex products per field on Apple's GPU, the same sums."""
        A = self.pupil_field(p)
        if A.dim() == 3:                                      # one pupil per emitter (field dependence)
            A = A[:, None]
        a = (A * self._waves(p, xyz, depth_nm))[..., :, None, :]                       # [E, K, P_y, 1, P_x]
        a = torch.cat([a, a], -3)
        # Contiguous parts: strided .real/.imag views were 3.5× slower on MPS.
        f = torch.addcmul(a.real.contiguous() * self._fields_re, a.imag.contiguous(), self._fields_im)
        By, Bx = self.dft.real_matrices(xyz[..., 0], xyz[..., 1])        # [E, K, 2uY, 2P], [E, K, 2P, 2uX]
        return self.dft.apply_wide(f, By, Bx), By, Bx

    def __call__(self, p, xyz: torch.Tensor, depth_nm: torch.Tensor | None = None,
                 z_step_nm: float | None = None) -> torch.Tensor:
        """[E, K, Y, X]: expected photons per pixel for one photon, at positions ``xyz`` [E, K, 3] nm.

        With a bead radius, each emitter's observations are planes of a z stack
        with step ``z_step_nm`` (the acquisition's, so it is given, not
        inferred: the bead is convolved along the stack in Fourier space).  The
        stack is padded as uiPSF pads it, R // Δz + 2 planes on each side,
        continuing from the end planes.
        """
        xyz, pad, dz = self._stack(xyz, z_step_nm, depth_nm)
        return self._image(self.intensity(p, xyz, depth_nm), self.blur(p), pad, dz)

    def pupil_linearisation(self, p, xyz: torch.Tensor, depth_nm: torch.Tensor | None = None,
                            z_step_nm: float | None = None):
        """``apply(dA)``: the change of `__call__` for tangent pupils ``dA`` [T, P, P] (or [T, E, P, P] with
        one pupil per emitter), as [T, E, K, Y, X]; the PSF is linear in the pupil field up to |·|².

        This is the shared Jacobian's bulk.  A JVP through the whole model cost 2.7 forward passes per
        pupil coefficient (`benchmarks/uipsf/profile_lm.py`); here the defocus waves, dipole fields and the
        image field E are computed once and each tangent costs one matrix DFT, 2·Re(E*·dE) and the blur:
        2.5× less for six beads (plan §9, M7).

        The DFT is laid out for few, wide matrix products: real and imaginary parts stacked ([Re; Im],
        one real product instead of four), and every tangent and dipole field of an emitter and slice
        side by side as columns of one product with that slice's A_y, then as rows of one with its A_x.
        As 1 476 small products per tangent (6 beads) the GPU ran at a few per cent of its peak; laid
        out wide, a tangent costs a fifth of a forward pass on Apple's GPU instead of one.  Sums over a
        middle axis are taken one axis at a time: over two axes at once MPS was 150× slower.  On MPS the
        per-tangent part goes through `torch.compile` (`device.compiled`), the tangents and emitters padded
        to powers of two so that few shapes are compiled: 12 % off the normal equations at 20 beads.
        """
        xyz, pad_z, dz = self._stack(xyz, z_step_nm, depth_nm)
        sigma = self.blur(p)
        with torch.no_grad():
            per_emitter = self.pupil_field(p).dim() == 3
            h, By, Bx = self._image_field(p, xyz, depth_nm)                    # [E, K, F, Y, 2, X]
            E2 = (2 * self.norm) * h[:, :, None]                               # [E, K, 1, F, Y, 2, X]
            # W·F as [Re; Im] along P_y, [E, K, 2P, 1, F, P]: Re dA multiplies [Re WF; Im WF], Im dA
            # multiplies [−Im WF; Re WF].  Multiplied out once: 15 % faster on MPS than per tangent.
            WF = self._waves(p, xyz, depth_nm)[..., :, None, :] * self.fields.permute(1, 0, 2)
            WF_re = torch.cat([WF.real, WF.imag], 2)[:, :, :, None].contiguous()
            WF_im = torch.cat([-WF.imag, WF.real], 2)[:, :, :, None].contiguous()
            E2 = E2 * self.dft.scale
        tangent_intensity = compiled(_tangent_intensity, WF_re.device)
        E = WF_re.shape[0]
        fixed = (WF_re, WF_im, By, Bx, E2)
        if tangent_intensity is not _tangent_intensity:
            # Compiled for fixed shapes: the emitters padded with zeros to a power of two, so that a run whose
            # batches come in many sizes (in situ: every localisation, round and twin) compiles a few shapes,
            # not one per size -- after 8, torch gave up compiling and ran everything uncompiled.
            fixed = tuple(_pad_dim0(t, _bucket(E)) for t in fixed)

        def apply(dA: torch.Tensor) -> torch.Tensor:
            # Built in place as [E, K, 2P, T, F, P], the layout `apply_wide` reads: no copy of the tangents.
            d = dA.permute(1, 2, 0, 3)[:, None, :, :, None] if per_emitter else dA.permute(1, 0, 2)[:, :, None]
            d = torch.cat([d, d], -4)                                          # [(E, 1,) 2P, T, 1, P]
            # Contiguous parts first: strided .real/.imag views made this product 3.5× slower on MPS.
            dr, di = d.real.contiguous(), d.imag.contiguous()
            T = dA.shape[0]
            if tangent_intensity is not _tangent_intensity:     # compiled: the tangents to a power of two too
                pad = _bucket(T) - T
                dr, di = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)) for t in (dr, di))
                if per_emitter:
                    dr, di = (_pad_dim0(t, _bucket(E)) for t in (dr, di))
            dI = tangent_intensity(dr, di, *fixed)[:E, :, :T]                   # [E, K, T, Y, X]
            return self._image(dI.movedim(2, 0), sigma, pad_z, dz)

        return apply

    def _waves(self, p, xyz, depth_nm):
        """exp(2πi k_z z) per emitter and observation, [E, K, P, P], with the depth term in-situ: one
        depth per emitter ([E]) or per emitter and observation ([E, K], `InSituGeometry`).  With a tilt of
        the interfaces (``geometry.tilt``) or a coverslip off its design thickness the depth term and the
        coverslip's are the layered model's (`optics.layers`); without, the wave is the untilted one."""
        z = xyz[..., 2][..., None, None].to(self.grid.kz_imm.dtype)
        if not layers.layered(self.grid, p):
            wave = torch.exp(2j * math.pi * self.grid.kz_imm * z)
            if depth_nm is not None:
                wave = wave * torch.exp(2j * math.pi * self.grid.kz_med * _depth(depth_nm, self.grid))
            return wave
        c_h, c_0 = layers.coefficients(self.grid, p.get("geometry.tilt"))
        phase = self.grid.kz_imm * z + c_0
        if depth_nm is not None:
            phase = phase + c_h * _depth(depth_nm, self.grid)
        return torch.exp(2j * math.pi * phase)

    def _stack(self, xyz, z_step_nm, depth_nm=None):
        """The positions, padded along the stack for a bead kernel: (xyz, pad, Δz)."""
        if self.bead_radius_nm <= 0:
            return xyz, 0, None
        if depth_nm is not None and depth_nm.dim() == 2:
            raise ValueError("a bead kernel needs a z stack, not emitters in the sample (InSituGeometry)")
        if z_step_nm is None or xyz.shape[1] < 2:
            raise ValueError("a bead kernel needs a z stack: at least two planes and z_step_nm")
        dz = abs(float(z_step_nm))
        pad = int(self.bead_radius_nm // dz) + 2
        return self._padded_stack(xyz, dz, pad), pad, dz

    def _image(self, I, sigma, pad, dz):
        """Blur, bead kernel and pixel integration of a sampled intensity [..., K, uY, uX]: linear in I."""
        if self.bead_radius_nm <= 0:                    # separable: two small real products, no FFT
            My, Mx = blur_bin_matrices(I.shape[-2:], self.spacing, sigma, self.upsample)
            return My @ I @ Mx.transpose(-1, -2)
        H = gaussian_otf(I.shape[-2:], self.spacing, sigma) * sphere_otf(
            I.shape[-3:], (*self.spacing, dz), self.bead_radius_nm, device=I.device, dtype=I.dtype)
        I = torch.fft.ifftn(torch.fft.fftn(I, dim=(-3, -2, -1)) * H, dim=(-3, -2, -1)).real
        return bin_pixels(I[..., pad:I.shape[-3] - pad, :, :], self.upsample)

    @staticmethod
    def _padded_stack(xyz, dz, pad):
        """``pad`` more planes at each end, z continuing in the stack's direction, x and y the end plane's."""
        z = xyz[..., 2]
        sign = 1.0 if float((z[:, -1] - z[:, 0]).mean()) >= 0 else -1.0
        steps = sign * dz * torch.arange(1, pad + 1, device=xyz.device, dtype=xyz.dtype)
        before = xyz[:, :1].repeat(1, pad, 1)
        after = xyz[:, -1:].repeat(1, pad, 1)
        before = torch.cat([before[..., :2], (z[:, :1] - steps.flip(0))[..., None]], -1)
        after = torch.cat([after[..., :2], (z[:, -1:] + steps)[..., None]], -1)
        return torch.cat([before, xyz, after], 1)


def _depth(depth_nm, grid):
    """The emitters' heights in the sample as [E, K, 1, 1] or [E, 1, 1, 1], complex."""
    d = depth_nm.to(grid.kz_med.dtype)
    return d[..., None, None] if d.dim() == 2 else d.reshape(-1, 1, 1, 1)


def _bucket(n: int) -> int:
    """The size a compiled kernel is padded to: the power of two at or above ``n`` up to 512, then the
    multiple of 512 -- a few shapes, and above 512 at most 40 % more work rather than up to twice it (1284
    in situ molecules were padded to 2048)."""
    n = int(n)
    if n <= 512:
        return 1 << max(0, n - 1).bit_length()
    return -(-n // 512) * 512


def _pad_dim0(t: torch.Tensor, n: int) -> torch.Tensor:
    """``t`` with zeros appended along its first axis up to ``n``."""
    if t.shape[0] >= n:
        return t
    return torch.cat([t, t.new_zeros((n - t.shape[0], *t.shape[1:]))])


def _tangent_intensity(dr, di, WF_re, WF_im, By, Bx, E2):
    """2·Re(E*·dE) for tangent pupils (Re, Im) dr, di [(E, 1,) 2P, T, 1, P]: [E, K, T, Y, X].  Real
    tensors only, so that `torch.compile` can fuse it (`device.compiled`)."""
    T = dr.shape[-3]
    f = torch.addcmul(dr * WF_re, di, WF_im).flatten(-3, -2)               # [E, K, 2P, T·F, P]
    dh = wide_product(f, By, Bx).unflatten(-4, (T, -1))                   # [E, K, T, F, Y, 2, X]
    return (dh * E2).sum(-2).sum(-3)                                      # one axis at a time (MPS)
