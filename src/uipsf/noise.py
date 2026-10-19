"""The camera and the likelihood: how far the data are from the expected photons.

**The likelihood is SMAPpy's, written as a deviance.**  A camera adds
Gaussian read noise of variance s to Poisson counts.  Adding s to both the
data and the model, d′ = d + s and μ′ = μ + s, makes a Poisson likelihood a
good approximation of the sum (Huang et al. 2013), and floors every pixel's
weight at 1/(μ + s); SMAPpy's fitters do exactly this (`with_readout`).  The
deviance per pixel is

    D = 2·[d′·log(d′/μ′) − (d′ − μ′)],

which is 0 where the model equals the data and ≈ 1 on average where the
model is right (plan §5.5, §8).  The solver minimises ½·ΣD, the negative
log-likelihood up to a constant, *summed* over pixels: a prior's strength
then does not depend on the number of beads (review F2).

**What v1 did and v2 does not.**  No masking of non-finite pixels (excluded
pixels are an explicit mask, counted), no clipping of the data, no quantile
shift.  A shifted data value below 0, possible only far below −s, is
clamped at 0 and counted (`clamped`); the exact Poisson–Gaussian
likelihood would cost far more for those few pixels (plan §10).

**EM gain** doubles the variance (excess noise F = 2).  As in SMAPpy, data
and model are divided by F before the Poisson likelihood; uipsf keeps the
photons physical and does the division inside the noise model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


class ShiftedPoisson:
    """Poisson on d/F + s against μ/F + s; s is the read-noise variance in those units (SMAPpy's)."""

    def __init__(self, excess: float = 1.0):
        self.excess = float(excess)

    def _shifted(self, mu, data, s):
        d = data / self.excess + s
        return d.clamp(min=0), mu / self.excess + s

    def deviance(self, mu, data, s=0.0, mask=None) -> torch.Tensor:
        """Per pixel, ≥ 0, ≈ 1 on average when the model is right; 0 where masked out."""
        d, m = self._shifted(mu, data, s)
        # d·log(d/m) as a ratio keeps float32 precise near d = m (d·log d − d·log m would cancel);
        # the ratio is set to 1 where d = 0 so that its gradient is 0 there, not 0/0.
        ratio = torch.where(d > 0, d / m, torch.ones_like(m))
        D = 2 * (d * torch.log(ratio) - (d - m))
        return D if mask is None else torch.where(mask, D, torch.zeros_like(D))

    def derivatives(self, mu, data, s=0.0, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(∂(½D)/∂μ, Fisher weight E[∂²(½D)/∂μ²]) per pixel: the gradient's and Gauss–Newton's.

        Fisher scoring rather than the observed curvature d′/μ′², so the
        Hessian is positive definite away from the optimum too.
        """
        d, m = self._shifted(mu, data, s)
        grad = (1 - d / m) / self.excess
        weight = 1 / (self.excess ** 2 * m)
        if mask is not None:
            grad, weight = torch.where(mask, grad, 0.0), torch.where(mask, weight, 0.0)
        return grad, weight

    def variance(self, mu, s=0.0) -> torch.Tensor:
        """The data's variance the model expects, photons²: F·μ + F²·s."""
        return self.excess * mu + self.excess ** 2 * s

    def clamped(self, data, s=0.0, mask=None) -> int:
        """How many shifted data values were below 0 and clamped (reported with the fit)."""
        low = (data / self.excess + s) < 0
        return int((low if mask is None else low & mask).sum())


class Poisson(ShiftedPoisson):
    """Pure Poisson: an ideal photon counter.  Any read-noise variance in the batch is ignored."""

    def __init__(self):
        super().__init__(1.0)

    def _shifted(self, mu, data, s):
        return data.clamp(min=0), mu

    def clamped(self, data, s=0.0, mask=None) -> int:
        return super().clamped(data, 0.0, mask)

    def variance(self, mu, s=0.0) -> torch.Tensor:
        return mu


class EMCCD(ShiftedPoisson):
    """Shifted Poisson with the excess-noise factor 2 of electron multiplication."""

    def __init__(self):
        super().__init__(2.0)


@dataclass(frozen=True)
class Camera:
    """ADU to photons, with SMAPpy's names and defaults (`smappy.metadata.CameraMetadata`).

    ``conversion`` is electrons per ADU, ``offset`` the baseline in ADU,
    ``em_gain`` None without EM.  ``read_noise_e`` is the rms read noise in
    electrons; unset it is 1 e⁻ without EM gain (a modern sCMOS reads 0.7 to
    1.4 e⁻) and 0 with it, as in SMAPpy.  Per-pixel maps come with M3.
    """
    conversion: float
    offset: float
    em_gain: float | None = None
    read_noise_e: float | None = None

    @property
    def excess(self) -> float:
        return 2.0 if self.em_gain else 1.0

    @property
    def read_noise_var(self) -> float:
        """s, in the noise model's units: electrons divided by the excess factor, squared."""
        if self.em_gain:
            return 0.0
        noise = 1.0 if self.read_noise_e is None else float(self.read_noise_e)
        return (noise / self.excess) ** 2

    def to_photons(self, adu) -> torch.Tensor:
        adu = torch.as_tensor(adu, dtype=torch.float32) if not isinstance(adu, torch.Tensor) else adu
        return (adu.to(torch.float32) - self.offset) * (self.conversion / (self.em_gain or 1.0))

    def noise(self) -> ShiftedPoisson:
        return ShiftedPoisson(self.excess)
