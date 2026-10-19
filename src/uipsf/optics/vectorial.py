"""The pupil fields of a freely rotating dipole.

Each of the three dipole orientations (x, y, z) contributes a field with an
x and a y component in the image; a freely rotating emitter is the
incoherent sum of all six intensities.  The p and s unit vectors carry the
Fresnel transmission; the basis is uiPSF's (`calpupilfield`): rows 0–2 are
the x-analyser response of the x, y and z dipole, rows 3–5 the y-analyser
response (review §A: the Ex/Ey channels pick the right components).
"""
from __future__ import annotations

import torch

from .grid import PupilGrid


def dipole_fields(grid: PupilGrid, analyser: str | None = None) -> torch.Tensor:
    """[F, P, P] complex: F = 6 without an analyser, 3 behind an x or y polariser."""
    cos_phi = torch.cos(grid.phi).to(grid.kz_med.dtype)
    sin_phi = torch.sin(grid.phi).to(grid.kz_med.dtype)
    sin_med = (grid.rho * grid.optics.NA / grid.optics.n_med).to(grid.kz_med.dtype)
    zero = torch.zeros_like(cos_phi)
    p = grid.fresnel_p * torch.stack([grid.cos_med * cos_phi, grid.cos_med * sin_phi, -sin_med])
    s = grid.fresnel_s * torch.stack([-sin_phi, cos_phi, zero])
    hx = cos_phi * p - sin_phi * s
    hy = sin_phi * p + cos_phi * s
    if analyser is None:
        return torch.cat([hx, hy])
    if analyser == "x":
        return hx
    if analyser == "y":
        return hy
    raise ValueError(f"analyser must be None, 'x' or 'y', got {analyser!r}")


def scalar_field(grid: PupilGrid) -> torch.Tensor:
    """[1, P, P]: the scalar model keeps the mean of the two Fresnel coefficients, as uiPSF does."""
    return ((grid.fresnel_p + grid.fresnel_s) / 2)[None]
