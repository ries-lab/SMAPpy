"""The cubic-spline PSF in torch, in place of DECODE's compiled ``spline`` package.

``CubicSplinePSF`` (psf_kernel.py) is DECODE's: it turns frame coordinates into the spline's
(``frame2roi_coord``, ``coord2impl``) and cuts the work into chunks.  What it handed to the C/CUDA
kernel of SplinePSF (``spline_psf.c``, ``spline_psf_gpu.cu``) it hands to `SplineTorch`, which
draws the same numbers on any device torch has: the CPU, CUDA and Apple's GPU.

**What the kernel computes** (spline_psf.c, read 2026-11-10, v0.10.0 and 1f85d10 alike).  The
coefficients are ``coeff[x, y, z, k]`` (SplinePSF reads the ``(X, Y, Z, 64)`` array in Fortran
order).  An emitter at implementation coordinates ``(xc, yc, zc)`` has ``x0 = floor(xc)`` and
``dx = xc - x0`` (likewise y, z), and ROI pixel ``(i, j)`` is::

    phot * sum_k m_k(dx, dy, dz) * coeff[x0 + i, y0 + j, clamp(z0, 0, Z - 1), k]

with the monomials ``m[16 pz + 4 py + px] = dz^pz dy^py dx^px`` (px fastest).  A pixel whose
voxel is outside the grid in x or y is ``1e-10`` -- *not* times the photons; in z only the index
is clamped, the delta is not.  The ROIs are added into the frames at the emitter's pixel
``(x_ix, y_ix)``; what falls outside the frame, or has a frame index outside ``[0, n_frames)``, is
dropped.  The derivatives (``forward_drv_rois``) are the kernel's too, in its order
``(x, y, phot, bg, z)``; outside the grid they are 0 and the model ``1e-10 * phot + bg``.

**How it is computed here.**  Gathering 64 coefficients for every pixel of every ROI reads
about 280 GB for one of DECODE's epochs (1.6 M emitters, 26 x 26 ROIs).  But the emitters come in
few kinds: ``frame2roi_coord`` gives every ROI the same voxel offset in x and y up to the
fractional part, so ``(x0, y0)`` takes one or two values and ``z0`` one of ``Z``.  The emitters
are grouped by ``(x0, y0, z0)``; each group's coefficient slab, ``P x 64`` for ``P`` pixels a ROI,
is gathered once, and a group is one batched matrix product of its monomials with its slab
(`torch.bmm` over the groups, padded to the largest).  Where the emitters are of too many kinds
for that to pay (more groups than an eighth of the emitters), they are drawn by gathering, in
chunks.

**One departure:** SplinePSF's CPU ``kernel_roi`` loops its second axis to ``npx``, not ``npy``,
so a non-square ROI is drawn wrongly on its CPU path; here, as on its GPU path, both axes are
the ROI's own.
"""
from __future__ import annotations

import torch

ROI_OUT_EPS = 1e-10       # spline_psf.c: sp->roi_out_eps, a pixel outside the grid in x or y
_MAX_ELEMENTS = 2 ** 27   # the largest padded output a batched product makes at once (512 MB of float32)


def monomials(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """``(N, 64)``: ``dz^pz dy^py dx^px`` at ``16 pz + 4 py + px``, as ``kernel_computeDelta3D``.

    The powers by repeated multiplication and the product ``(cz cy) cx``, the kernel's order, so
    the float32 values are its own.
    """
    def powers(d):
        p = [torch.ones_like(d), d]
        p.append(p[1] * d)
        p.append(p[2] * d)
        return torch.stack(p, -1)                                     # (N, 4)

    cz, cy, cx = powers(dz), powers(dy), powers(dx)
    return ((cz[:, :, None, None] * cy[:, None, :, None]) * cx[:, None, None, :]).reshape(-1, 64)


def monomial_derivatives(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """``(N, 3, 64)``: the monomials' derivatives in dx, dy and dz, as ``kernel_computeDelta3D``."""
    n = dx.shape[0]
    zero = torch.zeros_like(dx)

    def powers(d):
        p = [torch.ones_like(d), d]
        p.append(p[1] * d)
        p.append(p[2] * d)
        return torch.stack(p, -1)

    def dpowers(d):  # d/dd of d^p is p d^(p-1); the kernel writes (k+1) d^k into slot k+1
        p = powers(d)
        return torch.stack([zero, p[:, 0], 2 * p[:, 1], 3 * p[:, 2]], -1)

    cz, cy, cx = powers(dz), powers(dy), powers(dx)
    dcz, dcy, dcx = dpowers(dz), dpowers(dy), dpowers(dx)
    ddx = (cz[:, :, None, None] * cy[:, None, :, None]) * dcx[:, None, None, :]
    ddy = (cz[:, :, None, None] * dcy[:, None, :, None]) * cx[:, None, None, :]
    ddz = (dcz[:, :, None, None] * cy[:, None, :, None]) * cx[:, None, None, :]
    return torch.stack([ddx, ddy, ddz], 1).reshape(n, 3, 64)


class SplineTorch:
    """SplinePSF's ``PSFWrapperCPU`` / ``PSFWrapperCUDA`` interface, in torch, on `device`.

    Takes and returns torch tensors on the CPU, as DECODE's wrapper took and returned numpy
    arrays, and computes on `device`.
    """

    def __init__(self, coeff: torch.Tensor, roi_size: tuple[int, int], device: str | torch.device = "cpu"):
        if coeff.dim() != 4 or coeff.shape[-1] != 64:
            raise ValueError(f"coefficients must be (X, Y, Z, 64), not {tuple(coeff.shape)}")
        self.device = torch.device(device)
        self.shape = tuple(int(s) for s in coeff.shape[:3])            # X, Y, Z
        self.roi_size = (int(roi_size[0]), int(roi_size[1]))
        # one row of 64 per voxel, the voxel (x, y, z) at (x * Y + y) * Z + z
        self.coeff = coeff.detach().to(self.device, torch.float32).reshape(-1, 64).contiguous()
        ri = torch.arange(self.roi_size[0], device=self.device)
        rj = torch.arange(self.roi_size[1], device=self.device)
        self._di = ri[:, None].expand(self.roi_size).reshape(-1)        # (P,) the pixel's offset in x
        self._dj = rj[None, :].expand(self.roi_size).reshape(-1)        # (P,) and in y

    # --- the pieces -------------------------------------------------------------------------------

    def _split(self, x, y, z):
        x, y, z = (t.to(self.device, torch.float32) for t in (x, y, z))
        x0, y0, z0 = x.floor(), y.floor(), z.floor()
        dx, dy, dz = x - x0, y - y0, z - z0
        z0c = z0.long().clamp(0, self.shape[2] - 1)                   # z: the index clamped, not the delta
        return x0.long(), y0.long(), z0c, dx, dy, dz

    def _slabs(self, x0, y0, z0):
        """For voxel offsets ``(K,)`` each: the coefficients ``(K, P, 64)`` and inside ``(K, P)``."""
        X, Y, Z = self.shape
        xi = x0[:, None] + self._di[None, :]
        yi = y0[:, None] + self._dj[None, :]
        inside = (xi >= 0) & (xi < X) & (yi >= 0) & (yi < Y)
        lin = (xi.clamp(0, X - 1) * Y + yi.clamp(0, Y - 1)) * Z + z0[:, None]
        slab = self.coeff[lin]                                          # (K, P, 64)
        return slab * inside[..., None], inside

    def _contract(self, m: torch.Tensor, x0, y0, z0) -> tuple[torch.Tensor, torch.Tensor]:
        """``sum_k m[n, c, k] coeff[voxel of n's pixel p, k]``: ``(N, C, P)``, and inside ``(N, P)``.

        `m` is ``(N, C, 64)``: C sets of monomials an emitter (1 for the PSF, 4 with derivatives).
        """
        n, c = m.shape[0], m.shape[1]
        p = self.roi_size[0] * self.roi_size[1]
        if n == 0:
            return m.new_zeros((0, c, p)), torch.zeros((0, p), dtype=torch.bool, device=self.device)
        X, Y, Z = self.shape
        # the kind of an emitter: its voxel offset; few kinds unless the coordinates are arbitrary
        key = ((x0 - x0.min()) * (y0.max() - y0.min() + 1) + (y0 - y0.min())) * Z + z0
        kinds, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
        k = kinds.numel()
        if k > max(64, n // 8):
            return self._contract_gathered(m, x0, y0, z0)
        order = torch.argsort(inverse, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        first = order[starts]                                           # an emitter of each kind
        slab, inside = self._slabs(x0[first], y0[first], z0[first])     # (K, P, 64), (K, P)
        rank = torch.empty(n, dtype=torch.long, device=self.device)
        rank[order] = torch.arange(n, device=self.device) - starts[inverse[order]]
        width = int(counts.max())
        if k * width * c * p > _MAX_ELEMENTS:                           # too big at once: by halves
            half = n // 2
            a = self._contract(m[:half], x0[:half], y0[:half], z0[:half])
            b = self._contract(m[half:], x0[half:], y0[half:], z0[half:])
            return torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]])
        padded = m.new_zeros((k, width, c, 64))
        padded[inverse, rank] = m
        out = torch.bmm(padded.reshape(k, width * c, 64), slab.transpose(1, 2))   # (K, W*C, P)
        out = out.reshape(k, width, c, p)[inverse, rank]                # (N, C, P)
        return out, inside[inverse]

    def _contract_gathered(self, m, x0, y0, z0):
        """The same, gathering every emitter's own slab: for coordinates of many kinds, in chunks."""
        p = self.roi_size[0] * self.roi_size[1]
        step = max(1, _MAX_ELEMENTS // (p * 64))
        outs, insides = [], []
        for s in range(0, m.shape[0], step):
            sl = slice(s, s + step)
            slab, inside = self._slabs(x0[sl], y0[sl], z0[sl])          # (n, P, 64)
            outs.append(torch.einsum("nck,npk->ncp", m[sl], slab))
            insides.append(inside)
        return torch.cat(outs), torch.cat(insides)

    # --- SplinePSF's interface --------------------------------------------------------------------

    def rois_on_device(self, x, y, z, phot) -> torch.Tensor:
        """``(N, rx, ry)`` on the device: the ROIs of emitters at implementation coordinates."""
        x0, y0, z0, dx, dy, dz = self._split(x, y, z)
        f, inside = self._contract(monomials(dx, dy, dz)[:, None], x0, y0, z0)
        phot = phot.to(self.device, torch.float32)
        rois = torch.where(inside, f[:, 0] * phot[:, None], torch.full_like(f[:, 0], ROI_OUT_EPS))
        return rois.reshape(-1, *self.roi_size)

    def forward_rois(self, x, y, z, phot) -> torch.Tensor:
        return self.rois_on_device(x, y, z, phot).cpu()

    def forward_drv_rois(self, x, y, z, phot, bg, add_bg: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, 5, rx, ry)`` derivatives in the kernel's order ``(x, y, phot, bg, z)``, and the ROIs."""
        x0, y0, z0, dx, dy, dz = self._split(x, y, z)
        m = torch.cat([monomials(dx, dy, dz)[:, None], monomial_derivatives(dx, dy, dz)], 1)   # (N, 4, 64)
        f, inside = self._contract(m, x0, y0, z0)                       # (N, 4, P)
        phot = phot.to(self.device, torch.float32)[:, None]
        bg = bg.to(self.device, torch.float32)[:, None]
        add = bg if add_bg else torch.zeros_like(bg)
        drv = torch.stack([-phot * f[:, 1], -phot * f[:, 2], f[:, 0], torch.ones_like(f[:, 0]), phot * f[:, 3]], 1)
        drv = drv * inside[:, None]                                     # roi_out_deriv_eps = 0
        model = torch.where(inside, add + phot * f[:, 0], ROI_OUT_EPS * phot + add)
        n = model.shape[0]
        return drv.reshape(n, 5, *self.roi_size).cpu(), model.reshape(n, *self.roi_size).cpu()

    def frames_on_device(self, fx, fy, frame_ix, n_frames, xr, yr, z, x_ix, y_ix, phot) -> torch.Tensor:
        """``(n_frames, fx, fy)`` on the device: the ROIs added into the frames, as roi_accumulator.

        Into a canvas padded by a ROI on every side, so that every pixel of every ROI has a place:
        no mask of what falls outside, no compaction, 32-bit indices; the frames are then cut out
        of it, which drops what fell outside, as roi_accumulator does.  Building the masked 64-bit
        indices of 500 000 ROIs of 26 x 26 had taken four times as long as drawing them (2.7 s
        against 0.6 s on an M1 Max).  A ROI further off the frame than its own size is clamped
        onto the canvas's margin, which is cut away.
        """
        rx, ry = self.roi_size
        px, py = fx + 2 * rx, fy + 2 * ry
        if n_frames * px * py >= 2 ** 31:
            raise ValueError(f"{n_frames} frames of {fx} x {fy} are too many for one call; chunk the frames")
        keep = (frame_ix >= 0) & (frame_ix < n_frames)                  # the base class has done this already
        if not bool(keep.all()):
            xr, yr, z, phot, frame_ix, x_ix, y_ix = (t[keep] for t in (xr, yr, z, phot, frame_ix, x_ix, y_ix))
        ii = (x_ix.to(self.device, torch.int32) + rx).clamp(0, px - rx)
        jj = (y_ix.to(self.device, torch.int32) + ry).clamp(0, py - ry)
        base = (frame_ix.to(self.device, torch.int32) * px + ii) * py + jj                # (N,)
        offset = (self._di.to(torch.int32) * py + self._dj.to(torch.int32))               # (P,)
        canvas = torch.zeros(n_frames * px * py, device=self.device, dtype=torch.float32)
        # in batches whose grouped product fits at once, each added into the canvas as it comes:
        # no ROIs of the whole call held, nothing concatenated
        step = max(1, _MAX_ELEMENTS // (rx * ry))
        for s in range(0, xr.shape[0], step):
            sl = slice(s, s + step)
            rois = self.rois_on_device(xr[sl], yr[sl], z[sl], phot[sl]).reshape(-1)
            canvas.index_add_(0, (base[sl, None] + offset[None, :]).reshape(-1), rois)
        return canvas.reshape(n_frames, px, py)[:, rx:rx + fx, ry:ry + fy].contiguous()

    def forward_frames(self, fx, fy, frame_ix, n_frames, xr, yr, z, x_ix, y_ix, phot) -> torch.Tensor:
        return self.frames_on_device(fx, fy, frame_ix, n_frames, xr, yr, z, x_ix, y_ix, phot).cpu()
