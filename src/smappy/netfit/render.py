"""Cubic-spline PSFs drawn into frames in torch, on any device: what training simulates with.

The spline is SMAPpy's (`io.calibration.SplineCalibration`: 64 coefficients per voxel, ``(64, nz,
ny, nx)``, the monomials ``dz^pz dy^py dx^px`` with px fastest), evaluated exactly as SMAPpy's own
simulation draws it (`simulate.camera._draw_spline` on `evaluate_spline`), so that a network learns
the PSF the fitters fit:

* each spot in a square of ``size = max(min(nx, ny) - 3, 5)`` pixels around its nearest pixel;
* frame pixel ``i`` at spline coordinate ``u = i - x + c`` with ``c = nx / 2`` (for an odd ``size``:
  ``evaluate_spline``'s ``size/2 - 0.5`` plus its offset of the square in the grid), the voxel
  ``floor(u)`` clamped to the grid, the fraction ``u - floor(u)`` not;
* z as ``z_nm_to_index`` makes it, its voxel clamped likewise;
* negative values set to 0, then scaled by the photons.

Spots are drawn in chunks: one spot is ``size^2`` gathers of 64 coefficients.
"""
from __future__ import annotations

import numpy as np
import torch

CHUNK = 4096                      # spots per gather: 4096 x 23^2 x 64 floats is 0.55 GB


class SplineRenderer:
    """A calibration's spline on ``device``, drawing spots into frames."""

    def __init__(self, cal, device="cpu", dtype=torch.float32):
        _, nz, ny, nx = cal.coeff.shape
        self.device, self.dtype = torch.device(device), dtype
        # (nz, ny, nx, 64): a voxel's coefficients contiguous for the gather
        self.coeff = torch.as_tensor(np.ascontiguousarray(np.moveaxis(cal.coeff, 0, -1)), dtype=dtype, device=self.device)
        self.shape = (nz, ny, nx)
        self.size = max(min(nx, ny) - 3, 5)
        if self.size % 2 == 0:
            raise ValueError(f"the spline's square is {self.size} pixels; an odd one is expected (nx, ny even)")
        # where evaluate_spline puts pixel i for a spot at x, for its odd square
        self.cx = self.size / 2 - 0.5 + np.floor((nx + 1 - self.size) / 2)
        self.cy = self.size / 2 - 0.5 + np.floor((ny + 1 - self.size) / 2)
        self.z0, self.dz = float(cal.z0), float(cal.dz)
        powers = torch.arange(4, device=self.device, dtype=dtype)
        self._powers = powers

    def _monomials(self, d):
        return d[..., None] ** self._powers                                  # (..., 4)

    def spots(self, x, y, z_nm):
        """Unit-photon spots ``(m, size, size)`` and the frame pixel of their corner (columns, rows)."""
        half = self.size // 2
        a = torch.arange(self.size, device=self.device)
        ox = torch.round(x).long() - half                                   # rint, as numpy's
        oy = torch.round(y).long() - half
        u = (ox[:, None] + a[None]).to(self.dtype) - x[:, None] + self.cx    # (m, size)
        v = (oy[:, None] + a[None]).to(self.dtype) - y[:, None] + self.cy
        w = -z_nm / self.dz + self.z0
        nz, ny, nx = self.shape
        iu, iv, iw = torch.floor(u), torch.floor(v), torch.floor(w)
        mu, mv, mw = self._monomials(u - iu), self._monomials(v - iv), self._monomials(w - iw)
        iu = iu.long().clamp(0, nx - 1)
        iv = iv.long().clamp(0, ny - 1)
        iw = iw.long().clamp(0, nz - 1)
        c = self.coeff[iw[:, None, None], iv[:, :, None], iu[:, None, :]]  # (m, size, size, 64)
        c = c.view(*c.shape[:3], 4, 4, 4)                                   # pz, py, px
        # sum over pz, py, px of c * dz^pz * dy^py * dx^px
        out = torch.einsum("mabzyx,mz,may,mbx->mab", c, mw, mv, mu)
        return out.clamp_min(0), ox, oy

    def draw(self, frames, index, x, y, z_nm, photons):
        """Add each spot into ``frames`` (n, rows, columns) in place: frame ``index``, at column ``x``
        and row ``y`` (pixel centres at whole numbers), ``z_nm``, ``photons``."""
        _, h, w = frames.shape
        a = torch.arange(self.size, device=self.device)
        for s in range(0, len(x), CHUNK):
            sl = slice(s, s + CHUNK)
            spot, ox, oy = self.spots(x[sl], y[sl], z_nm[sl])
            spot = spot * photons[sl, None, None]
            cols = ox[:, None, None] + a[None, None, :]
            rows = oy[:, None, None] + a[None, :, None]
            inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
            f = index[sl, None, None].expand_as(inside)
            flat = (f * h + rows.clamp(0, h - 1)) * w + cols.clamp(0, w - 1)
            frames.view(-1).index_add_(0, flat[inside], spot[inside].to(frames.dtype))
        return frames
