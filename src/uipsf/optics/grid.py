"""The pupil plane: sample positions and everything that depends only on the optics.

The pupil is sampled on P × P points at the centres of a grid spanning the
unit disc of normalised radius ρ = sin θ_imm · n_imm / NA, as uiPSF does:
u = (i − (P−1)/2)·2/P along x (columns), v likewise along y (rows).

Signs follow `docs/uipsf/conventions.md`: a plane wave in the pupil carries
exp(+2πi(k_x x + k_y y + k_z z)).  The axial wave numbers are complex above
the critical angle of the sample medium; the square root takes the branch
+i·|·|, so k_z,med·depth decays into the sample (supercritical-angle light).

Fresnel transmission (medium → coverslip → immersion) and the aplanatic
apodisation √cos θ_imm / cos θ_med are uiPSF's (Suppl. Eq. 3.3), checked
against the TU Delft vectorial model in the review (§D).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..device import complex_dtype, resolve
from .system import Optics


def _sqrt(x):
    """The complex square root with the +i branch for negative arguments."""
    return np.lib.scimath.sqrt(x)


@dataclass(frozen=True, eq=False)
class PupilGrid:
    optics: Optics
    size: int
    u: torch.Tensor            # [P, P] normalised pupil coordinate along x
    v: torch.Tensor            # [P, P] along y
    rho: torch.Tensor          # [P, P]
    phi: torch.Tensor          # [P, P] azimuth, atan2(v, u)
    aperture: torch.Tensor     # [P, P] 1 inside the NA, 0 outside
    kx: torch.Tensor           # [P] spatial frequency of each column, cycles/nm
    ky: torch.Tensor           # [P] of each row
    kz_imm: torch.Tensor       # [P, P] complex, cycles/nm
    kz_med: torch.Tensor       # [P, P] complex
    kz_cov: torch.Tensor       # [P, P] complex, in the coverslip (`optics.layers`)
    cos_imm: torch.Tensor      # [P, P] complex
    cos_med: torch.Tensor
    cos_cov: torch.Tensor
    fresnel_p: torch.Tensor    # [P, P] complex, medium -> coverslip -> immersion
    fresnel_s: torch.Tensor
    apodisation: torch.Tensor  # [P, P] complex, √cos θ_imm / cos θ_med
    dk: float                  # pupil sample spacing, cycles/nm

    @classmethod
    def build(cls, optics: Optics, size: int = 64, device=None, dtype=torch.float32) -> PupilGrid:
        device, dtype = resolve(device, dtype)
        cdtype = complex_dtype(dtype)
        P = int(size)
        c = (np.arange(P) - (P - 1) / 2) * 2 / P
        u, v = np.meshgrid(c, c)                     # u varies along columns (x), v along rows (y)
        rho = np.hypot(u, v)
        phi = np.arctan2(v, u)
        s = rho * optics.NA                          # n·sin θ, the same in every layer
        lam = optics.wavelength_nm
        cos_imm = _sqrt(1 - (s / optics.n_imm) ** 2)
        cos_med = _sqrt(1 - (s / optics.n_med) ** 2)
        cos_cov = _sqrt(1 - (s / optics.n_cov) ** 2)
        n_med, n_cov, n_imm = optics.n_med, optics.n_cov, optics.n_imm
        tp = (2 * n_med * cos_med / (n_med * cos_cov + n_cov * cos_med)
              * 2 * n_cov * cos_cov / (n_cov * cos_imm + n_imm * cos_cov))
        ts = (2 * n_med * cos_med / (n_med * cos_med + n_cov * cos_cov)
              * 2 * n_cov * cos_cov / (n_cov * cos_cov + n_imm * cos_imm))
        apod = np.sqrt(cos_imm) / cos_med

        def real(a):
            return torch.as_tensor(np.real(a), dtype=dtype, device=device)

        def cplx(a):
            return torch.as_tensor(np.asarray(a, complex), dtype=cdtype, device=device)

        return cls(optics=optics, size=P, u=real(u), v=real(v), rho=real(rho), phi=real(phi),
                   aperture=real(rho < 1), kx=real(c * optics.NA / lam), ky=real(c * optics.NA / lam),
                   kz_imm=cplx(n_imm / lam * cos_imm), kz_med=cplx(n_med / lam * cos_med),
                   kz_cov=cplx(n_cov / lam * cos_cov),
                   cos_imm=cplx(cos_imm), cos_med=cplx(cos_med), cos_cov=cplx(cos_cov),
                   fresnel_p=cplx(tp), fresnel_s=cplx(ts), apodisation=cplx(apod),
                   dk=2 * optics.NA / lam / P)

    @property
    def device(self) -> torch.device:
        return self.u.device

    @property
    def dtype(self) -> torch.dtype:
        return self.u.dtype
