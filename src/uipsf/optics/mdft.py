"""From the pupil to the image: a matrix DFT.

The field at image point (x, y) is a sum over pupil samples,
E(x, y) = Σ F(k) exp(2πi[k_x(x − x0) + k_y(y − y0)]) Δk², which for a
P × P pupil and an X × Y region is two complex matrix products,
A_y · F · A_xᵀ.  For uiPSF's sizes (64-sample pupil, 20–50 px regions) this
is faster than a chirp-z transform built from three padded FFTs: 3–4× on a
CPU and 1.7× on Apple's GPU (review F3).  The emitter's lateral position
enters the matrices, A_x = exp(2πi k_x (x − x0)), so no phase ramp over the
whole pupil is needed.

Image samples are centred on the region: x_m = (m − (X−1)/2)·Δx, so an odd
region has a sample, and a pixel centre, on the emitter at x0 = 0.
"""
from __future__ import annotations

import math

import torch

from ..device import complex_dtype
from .grid import PupilGrid


class MatrixDFT:
    def __init__(self, grid: PupilGrid, shape: tuple[int, int], spacing_nm: tuple[float, float]):
        """``shape`` is (Y, X) samples, ``spacing_nm`` the (x, y) distance between samples."""
        self.grid = grid
        self.shape = (int(shape[0]), int(shape[1]))
        self.spacing_nm = (float(spacing_nm[0]), float(spacing_nm[1]))
        Y, X = self.shape
        dev, dt = grid.device, grid.dtype
        self.x = (torch.arange(X, device=dev, dtype=dt) - (X - 1) / 2) * self.spacing_nm[0]
        self.y = (torch.arange(Y, device=dev, dtype=dt) - (Y - 1) / 2) * self.spacing_nm[1]
        self.scale = grid.dk ** 2                      # Δk_x Δk_y: the sum approximates the integral

    def matrix(self, coords: torch.Tensor, k: torch.Tensor, shift: torch.Tensor | None) -> torch.Tensor:
        """exp(2πi k (coords − shift)), [..., n_coords, P]; ``shift`` broadcasts over leading dims."""
        r = coords[:, None] if shift is None else coords[:, None] - shift[..., None, None]
        phase = 2 * math.pi * r * k
        return torch.polar(torch.ones_like(phase), phase).to(complex_dtype(self.grid.dtype))

    def __call__(self, field: torch.Tensor, x0: torch.Tensor | None = None,
                 y0: torch.Tensor | None = None) -> torch.Tensor:
        """[..., P, P] (rows: k_y, columns: k_x) -> [..., Y, X] complex field.

        ``x0`` and ``y0`` (nm) must broadcast against ``field.shape[:-2]``.
        """
        return self.apply(field, *self.matrices(x0, y0))

    def matrices(self, x0=None, y0=None):
        """(A_x, A_y) for these shifts, to apply to many fields at the same positions."""
        return self.matrix(self.x, self.grid.kx, x0), self.matrix(self.y, self.grid.ky, y0)

    def apply(self, field, Ax, Ay):
        return (Ay @ field @ Ax.transpose(-1, -2)) * self.scale

    def real_matrices(self, x0, y0):
        """(B_y, B_x) for these shifts, real block matrices [..., 2Y, 2P] and [..., 2P, 2X]: for a field
        stacked as [Re F; Im F], B_y·[Re; Im] = [Re; Im] of A_y·F, and [Re G, Im G]·B_x = [Re, Im] of
        G·A_xᵀ.  One real product instead of four; real products are the faster path on CPUs and on
        Apple's GPU.  Built from cos and sin directly: 20 % faster than from `matrices` on MPS."""
        def cos_sin(coords, k, shift):
            phase = 2 * math.pi * (coords[:, None] - shift[..., None, None]) * k
            return torch.cos(phase), torch.sin(phase)
        yr, yi = cos_sin(self.y, self.grid.ky, y0)                       # [..., Y, P]
        xr, xi = (t.transpose(-1, -2) for t in cos_sin(self.x, self.grid.kx, x0))   # [..., P, X]
        By = torch.cat([torch.cat([yr, -yi], -1), torch.cat([yi, yr], -1)], -2)
        Bx = torch.cat([torch.cat([xr, xi], -1), torch.cat([-xi, xr], -1)], -2)
        return By, Bx

    def apply_wide(self, f, By, Bx):
        """`apply` for many fields per position, real: ``f`` [..., 2P, N, P] holds N fields as [Re; Im]
        along the rows, By [..., 2Y, 2P] and Bx [..., 2P, 2X] from `real_matrices`; -> [..., N, Y, 2, X],
        the (Re, Im) image of each field."""
        return wide_product(f, By, Bx) * self.scale


def wide_product(f, By, Bx):
    """`MatrixDFT.apply_wide` without the Δk² scale: a pure function of real tensors, which
    `torch.compile` can take.

    The N fields sit side by side as columns of one product with A_y, then as rows of one with A_x:
    a few wide products instead of N·(batch) small ones.  As small products Apple's GPU ran at a few
    per cent of its peak (plan §9 M7, `benchmarks/uipsf/profile_lm.py`)."""
    *lead, P2, N, P = f.shape
    Y, X = By.shape[-2] // 2, Bx.shape[-1] // 2
    g = By @ f.reshape(*lead, P2, N * P)                                      # [..., 2Y, N·P]
    g = g.reshape(*lead, 2, Y, N, P).movedim(-4, -2).movedim(-4, -3)          # [..., N, Y, 2, P]
    h = g.reshape(*lead, N * Y, P2) @ Bx                                      # [..., N·Y, 2X]
    return h.reshape(*lead, N, Y, 2, X)
