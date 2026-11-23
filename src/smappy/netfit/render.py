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

With ``whole`` (`SimulationSettings.spots` "whole") each spot covers the spline's whole grid instead, as
DECODE-Plex draws it and the C = 2 sets do (`benchmarks/decode/suite/sets_c2.draw_spline_whole`): a
square of ``nx`` pixels from ``rint(x) - nx // 2``, pixel ``i`` at ``u = i - x + nx / 2``, and a pixel
whose ``u`` falls outside the grid drawn as nothing, not as the edge cell's cubic carried on.  The
square cut 4 -- 8 % of the light far from focus (plex-plan §15).

With ``field`` (a field-dependent calibration, `smappy.field_splines`) each spot is drawn with the
spline blended where it is on the chip (``chip``, x and y pixels): the four nodes around it and their
bilinear weights as `field_splines.bilinear` takes them, each voxel's coefficients the weighted sum of
the four nodes' -- what Spline 3D's `CSplineField` evaluates (`csrc/models.hpp`).  A spline is linear in
its coefficients, so this is the blended spline itself, not a blend of four spots.  Without a ``chip``,
or for a calibration without node splines, the spot is ``cal.coeff``'s, the field's centre
(`docs/decode/fd-plan.md`).

Spots are drawn in chunks: one spot is ``size^2`` gathers of 64 coefficients (four times that with the
field).
"""
from __future__ import annotations

import numpy as np
import torch

CHUNK = 4096                      # spots per gather: 4096 x 23^2 x 64 floats is 0.55 GB


class SplineRenderer:
    """A calibration's spline on ``device``, drawing spots into frames."""

    def __init__(self, cal, device="cpu", dtype=torch.float32, whole: bool = False, field: bool = False):
        _, nz, ny, nx = cal.coeff.shape
        self.device, self.dtype = torch.device(device), dtype
        # (nz, ny, nx, 64): a voxel's coefficients contiguous for the gather
        self.coeff = torch.as_tensor(np.ascontiguousarray(np.moveaxis(cal.coeff, 0, -1)), dtype=dtype, device=self.device)
        # the node splines, (G, nz, ny, nx, 64), and the nodes' chip x and y
        self.nodes = self.node_x = self.node_y = None
        fs = getattr(cal, "field_splines", None) if field else None
        if field and fs is None:
            raise ValueError("a field-dependent drawing needs a calibration with node splines")
        if fs is not None:
            if fs.coeff.shape[2:] != cal.coeff.shape[1:]:
                raise ValueError(f"node splines {fs.coeff.shape[2:]} on another grid than the centre's {cal.coeff.shape[1:]}")
            self.nodes = torch.as_tensor(np.ascontiguousarray(np.moveaxis(fs.coeff, 1, -1)), dtype=dtype,
                                         device=self.device)
            self.node_x = torch.as_tensor(fs.xs, dtype=torch.float32, device=self.device)
            self.node_y = torch.as_tensor(fs.ys, dtype=torch.float32, device=self.device)
        self.shape = (nz, ny, nx)
        self.whole = bool(whole)
        if self.whole:
            if nx != ny:
                raise ValueError(f"a whole-grid spot needs a square grid, not {nx} x {ny}")
            # the square from rint(x) - nx // 2, pixel i at u = i - x + nx / 2 (DECODE-Plex's)
            self.size, self.cx, self.cy = nx, nx / 2, ny / 2
        else:
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

    @property
    def field(self) -> bool:
        """Whether spots are drawn with the node splines where they are on the chip."""
        return self.nodes is not None

    def bilinear(self, chip):
        """(nodes (m, 4) long, weights (m, 4)): the four nodes around each chip position ``chip`` (m, 2)
        and their weights -- `field_splines.bilinear` in torch, positions beyond the grid its edge's."""
        def axis(v, nodes):
            if len(nodes) == 1:
                return torch.zeros(len(v), dtype=torch.long, device=v.device), torch.zeros_like(v)
            v = v.clamp(float(nodes[0]), float(nodes[-1]))
            i = (torch.searchsorted(nodes, v.contiguous(), right=True) - 1).clamp(0, len(nodes) - 2)
            return i, (v - nodes[i]) / (nodes[i + 1] - nodes[i])
        chip = chip.to(torch.float32)
        i, a = axis(chip[:, 0], self.node_x)
        j, b = axis(chip[:, 1], self.node_y)
        gx = len(self.node_x)
        di, dj = int(gx > 1), int(len(self.node_y) > 1)
        nodes = torch.stack([j * gx + i, j * gx + i + di, (j + dj) * gx + i, (j + dj) * gx + i + di], 1)
        weights = torch.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1)
        return nodes, weights.to(self.dtype)

    def spots(self, x, y, z_nm, corner=None, chip=None):
        """Unit-photon spots ``(m, size, size)`` and the frame pixel of their corner (columns, rows).
        ``corner`` (ox, oy) holds the squares where given: a derivative by a small step in x must not
        move the square when x crosses a half pixel (`bounds.information`).  ``chip`` (m, 2): each spot's
        place on the chip, for a field-dependent drawing (ignored without one); held as given, so a
        derivative by x does not move the PSF's shape, as the fitter holds a ROI's."""
        half = self.size // 2
        a = torch.arange(self.size, device=self.device)
        if corner is None:
            ox = torch.round(x).long() - half                               # rint, as numpy's
            oy = torch.round(y).long() - half
        else:
            ox, oy = corner
        u = (ox[:, None] + a[None]).to(self.dtype) - x[:, None] + self.cx    # (m, size)
        v = (oy[:, None] + a[None]).to(self.dtype) - y[:, None] + self.cy
        w = -z_nm / self.dz + self.z0
        nz, ny, nx = self.shape
        iu, iv, iw = torch.floor(u), torch.floor(v), torch.floor(w)
        mu, mv, mw = self._monomials(u - iu), self._monomials(v - iv), self._monomials(w - iw)
        if self.whole:                       # beyond the grid: nothing
            mu = mu * ((iu >= 0) & (iu < nx))[..., None]
            mv = mv * ((iv >= 0) & (iv < ny))[..., None]
        iu = iu.long().clamp(0, nx - 1)
        iv = iv.long().clamp(0, ny - 1)
        iw = iw.long().clamp(0, nz - 1)
        if self.nodes is not None and chip is not None:
            nodes, weights = self.bilinear(chip)
            c = 0
            for k in range(4):              # each voxel's coefficients blended, as CSplineField::value
                c = c + weights[:, k, None, None, None] * self.nodes[nodes[:, k, None, None], iw[:, None, None],
                                                                     iv[:, :, None], iu[:, None, :]]
        else:
            c = self.coeff[iw[:, None, None], iv[:, :, None], iu[:, None, :]]  # (m, size, size, 64)
        c = c.view(*c.shape[:3], 4, 4, 4)                                   # pz, py, px
        # sum over pz, py, px of c * dz^pz * dy^py * dx^px
        out = torch.einsum("mabzyx,mz,may,mbx->mab", c, mw, mv, mu)
        return out.clamp_min(0), ox, oy

    def draw(self, frames, index, x, y, z_nm, photons, chip=None):
        """Add each spot into ``frames`` (n, rows, columns) in place: frame ``index``, at column ``x``
        and row ``y`` (pixel centres at whole numbers), ``z_nm``, ``photons``; ``chip`` (m, 2) where each
        is on the chip, for a field-dependent drawing."""
        _, h, w = frames.shape
        a = torch.arange(self.size, device=self.device)
        for s in range(0, len(x), CHUNK):
            sl = slice(s, s + CHUNK)
            spot, ox, oy = self.spots(x[sl], y[sl], z_nm[sl], chip=None if chip is None else chip[sl])
            spot = spot * photons[sl, None, None]
            cols = ox[:, None, None] + a[None, None, :]
            rows = oy[:, None, None] + a[None, :, None]
            inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
            f = index[sl, None, None].expand_as(inside)
            flat = (f * h + rows.clamp(0, h - 1)) * w + cols.clamp(0, w - 1)
            frames.view(-1).index_add_(0, flat[inside], spot[inside].to(frames.dtype))
        return frames
