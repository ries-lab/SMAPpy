"""A blur that grows with defocus: σ(z)² = σ₀² + (c·z)², for `VectorPSF`.

The user's hypothesis (31 October 2026): aberrations that differ between the beads (across the field, or
from bead to bead) hardly change a PSF in focus but make defocused PSFs differ more and more; a model shared
by all beads takes that up as blur, and a constant σ is then too large in focus -- the sharp core every
model missed on M5 (`real/m5_media_psfs.py`) -- and too small far out.  A spread δ of the beads' focal
positions (an astigmatism varying by δa nm of path moves the x and y foci by about ±2.1·δa at NA 1.5)
widens a defocused image roughly in proportion to |z|, so the blur's variance gets a term (c·z)², c
dimensionless, of the order of the spread over the depth of focus: 0.01-0.04 for 5-10 nm of path.

`ZBlurPSF` is `VectorPSF` with one more shared parameter, ``psf.sigma_c`` (c), the existing ``psf.sigma``
being σ₀ (round or not, learnt or held, as before).  The blur acts per slice after the bead's convolution
along z: σ(z) hardly changes over a bead's 40-100 nm.  Only σ² enters (the Gaussian's transfer function is
exp(−2π²σ²q²)), so σ₀ = c = 0 is smooth.

A benchmark, not uipsf: the slices' z reach the blur through the last `_stack`, which `__call__` and
`pupil_linearisation` both call just before they ask for the blur.  In the forward model proper, z would be
passed to the blur instead (the user: if it works, it becomes part of the forward model).
"""
from __future__ import annotations

import math

import torch

from uipsf.model import VectorPSF
from uipsf.optics.kernels import bin_pixels, sphere_otf
from uipsf.params import Bounded, ParamSpec


class ZBlurPSF(VectorPSF):
    def __init__(self, *args, c_init: float = 0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_init = float(c_init)
        self._z = None

    def specs(self) -> list[ParamSpec]:
        return super().specs() + [ParamSpec("psf.sigma_c", (), init=self.c_init, transform=Bounded(0.0, 0.3))]

    def _stack(self, xyz, z_step_nm, depth_nm=None):
        xyz, pad, dz = super()._stack(xyz, z_step_nm, depth_nm)
        self._z = xyz[..., 2]                                            # [E, K'], the padded stack's
        return xyz, pad, dz

    def blur(self, p) -> torch.Tensor:
        """The blur's variance per slice and axis, [E, K', 2] nm² -- not σ, unlike `VectorPSF.blur`."""
        s0 = VectorPSF.blur(p)                                           # [2]
        c = p["psf.sigma_c"]
        return s0 ** 2 + ((c * self._z) ** 2)[..., None]

    def _image(self, I, variance, pad, dz):
        """The bead's convolution along the stack, then each slice's own Gaussian, then the pixels."""
        if self.bead_radius_nm > 0:
            H = sphere_otf(I.shape[-3:], (*self.spacing, dz), self.bead_radius_nm, device=I.device, dtype=I.dtype)
            X = I.shape[-1]
            I = torch.fft.irfftn(torch.fft.rfftn(I, dim=(-3, -2, -1)) * H[..., :X // 2 + 1], s=I.shape[-3:],
                                 dim=(-3, -2, -1))
            K = I.shape[-3]
            I, variance = I[..., pad:K - pad, :, :], variance[:, pad:K - pad]
        Y, X = I.shape[-2:]
        qx = torch.fft.rfftfreq(X, d=self.spacing[0], device=I.device, dtype=I.dtype)
        qy = torch.fft.fftfreq(Y, d=self.spacing[1], device=I.device, dtype=I.dtype)
        G = torch.exp(-2 * math.pi ** 2 * (variance[..., 0, None, None] * qx ** 2
                                           + variance[..., 1, None, None] * qy[:, None] ** 2))   # [E, K, Y, X/2+1]
        I = torch.fft.irfft2(torch.fft.rfft2(I) * G, s=(Y, X))
        return bin_pixels(I, self.upsample)


def use_z_blur(on: bool, c_init: float = 0.02):
    """Make `bead_model` build `ZBlurPSF` (``on``) or `VectorPSF`."""
    import uipsf.workflows.calibrate as calibrate
    if not on:
        calibrate.VectorPSF = VectorPSF
        return

    class Build(VectorPSF):
        def __new__(cls, *args, **kwargs):
            return ZBlurPSF(*args, c_init=c_init, **kwargs)
    calibrate.VectorPSF = Build
