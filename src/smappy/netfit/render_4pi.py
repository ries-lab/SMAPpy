"""The 4Pi IAB model drawn in torch, on any device: what a 4Pi network's training simulates with.

One channel ``c`` of a 4Pi microscope is ``N (I_c + A_c cos phi + B_c sin phi)`` (`smappy.fourpi`):
three cubic splines on one grid, the channel's phase offset and share of the light inside its own
``A_c`` and ``B_c``, one phase ``phi`` for every channel.  This draws it exactly as Spline 4Pi fits it
(`CSpline4Pi` in `csrc/models.hpp`, through `psf.Global4PiPSF`) and as `fourpi.spots` and
`fourpi.add_frames` draw it in numpy, so that a network learns the PSF the fitter fits.  The
geometry is `render.SplineRenderer`'s, which is the same as the C++'s:

* each spot in a square of ``size = max(min(nx, ny) - 3, 5)`` pixels around its nearest pixel
  (``rint``, half to even, as numpy's and torch's ``round``);
* pixel ``i`` of the channel at spline coordinate ``u = i - x + c``, ``c`` the square's centre plus its
  offset in the grid; the voxel ``floor(u)`` clamped to the grid, the fraction ``u - floor(u)`` not --
  `CSpline4Pi::value` clamps the voxel and keeps the fraction, and so does `fourpi.spots`;
* z as `FourPiCalibration.z_nm_to_index` makes it, its voxel clamped likewise;
* the phase as given: ``phase_per_nm * z_nm`` plus the cavity's phase is the caller's (`phase_of`),
  because the network is told the phase it should find, as the fitter reports it;
* negative values set to 0 after I, A and B are summed, as `fourpi.add_frames` does (``clamp``);
  the fitter's model is not clamped, and a test compares with it unclamped.

Each spot is drawn in every channel at that channel's own position (``x``, ``y`` of shape (m, C)):
where a layout's registration puts the molecule (`channels.ChannelLayout.to_channel`), or, on
netfit's grid, channel 0's position plus each channel's whole-pixel shift and remainder
(`netfit.channels`, two channels so far).  The coefficients are read as the fitter reads them, voxel
by voxel (`FourPiCalibration.fit_coeff`, (C, nz, ny, nx, 3, 64)): one gather of a pixel's I, A and B
together.  The phase convention -- no offset per channel beyond what A and B hold -- was settled on
uiPSF v1's beads against DECODE_Internal's 4Pi branch, which added one (docs/4pi.md §30,
`benchmarks/fourpi/phase_convention.py`).
"""
from __future__ import annotations

import numpy as np
import torch

#: spots per gather: 256 x 4 x 23^2 x 192 floats is 0.4 GB.  On the CPU a smaller block stays in the
#: caches (3x the spots a second of 1024's, measured on the Mac); a GPU wants the larger one.
CHUNK = {"cpu": 256}
GPU_CHUNK = 1024


class IABRenderer:
    """A `fourpi.FourPiCalibration` on ``device``, drawing 4Pi spots into each channel's image."""

    def __init__(self, cal, device="cpu", dtype=torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        # (C, nz, ny, nx, 3, 64): a voxel's I, A and B contiguous, the fitter's own layout
        self.coeff = torch.as_tensor(cal.fit_coeff, device=self.device).to(dtype)
        C, nz, ny, nx = self.coeff.shape[:4]
        self.n_channels, self.shape = C, (nz, ny, nx)
        self.size = max(min(nx, ny) - 3, 5)
        if self.size % 2 == 0:
            raise ValueError(f"the spline's square is {self.size} pixels; an odd one is expected (nx, ny even)")
        # where the fitter puts pixel i of a square of `size` for a spot at x (`CSpline4Pi::prepare`)
        self.cx = self.size / 2 - 0.5 + np.floor((nx + 1 - self.size) / 2)
        self.cy = self.size / 2 - 0.5 + np.floor((ny + 1 - self.size) / 2)
        self.z0, self.dz = float(cal.z0), float(cal.dz)
        self.phase_per_nm = float(cal.phase_per_nm)
        self._powers = torch.arange(4, device=self.device, dtype=dtype)
        self.chunk = CHUNK.get(self.device.type, GPU_CHUNK)

    def phase_of(self, z_nm, phase0):
        """The phase a molecule at ``z_nm`` has in a cavity at ``phase0``: the fitter's ``phase_rad``
        before it is wrapped (`fourpi.simulate_rois`)."""
        return self.phase_per_nm * z_nm + phase0

    def _monomials(self, d):
        return d[..., None] ** self._powers                                  # (..., 4)

    def iab(self, x, y, z_nm, corner=None):
        """Each spot's I, A and B per channel, ``(m, C, 3, size, size)``, for one photon, and the
        squares' corners ``(ox, oy)``, each (m, C).  ``x``, ``y`` (m, C): the spot in each channel's
        pixels; ``z_nm`` (m,).  ``corner`` holds the squares where given (a derivative by a step in x
        must not move one across a half pixel)."""
        m, C = x.shape
        if C != self.n_channels:
            raise ValueError(f"positions for {C} channels, the model has {self.n_channels}")
        half = self.size // 2
        a = torch.arange(self.size, device=self.device)
        if corner is None:
            ox = torch.round(x).long() - half
            oy = torch.round(y).long() - half
        else:
            ox, oy = corner
        u = (ox[..., None] + a).to(self.dtype) - x[..., None] + self.cx    # (m, C, size)
        v = (oy[..., None] + a).to(self.dtype) - y[..., None] + self.cy
        w = -z_nm / self.dz + self.z0                                       # (m,)
        nz, ny, nx = self.shape
        iu, iv, iw = torch.floor(u), torch.floor(v), torch.floor(w)
        # the fractions are the same for every pixel of a square (u moves by whole pixels), so a spot's
        # 64 monomials dz^pz dy^py dx^px (px fastest) are one vector per spot and channel
        mu, mv, mw = (self._monomials(d) for d in (u[..., 0] - iu[..., 0], v[..., 0] - iv[..., 0], w - iw))
        delta = (mw[:, None, :, None, None] * mv[:, :, None, :, None] * mu[:, :, None, None, :]).reshape(m, C, 64)
        iu = iu.long().clamp(0, nx - 1)
        iv = iv.long().clamp(0, ny - 1)
        iw = iw.long().clamp(0, nz - 1)
        ch = torch.arange(C, device=self.device)
        c = self.coeff[ch[None, :, None, None], iw[:, None, None, None],
                       iv[:, :, :, None], iu[:, :, None, :]]               # (m, C, size, size, 3, 64)
        out = torch.matmul(c, delta[:, :, None, None, :, None])[..., 0]    # (m, C, size, size, 3)
        return out.permute(0, 1, 4, 2, 3), ox, oy

    def spots(self, x, y, z_nm, phase, corner=None, clamp: bool = True):
        """Unit-photon spots ``(m, C, size, size)`` and their corners (ox, oy), each (m, C): I + A cos
        phase + B sin phase in every channel, set to 0 where negative unless ``clamp`` is False."""
        parts, ox, oy = self.iab(x, y, z_nm, corner)
        cos, sin = torch.cos(phase)[:, None, None, None], torch.sin(phase)[:, None, None, None]
        out = parts[:, :, 0] + parts[:, :, 1] * cos + parts[:, :, 2] * sin
        return (out.clamp_min(0) if clamp else out), ox, oy

    def draw(self, frames, index, x, y, z_nm, phase, photons):
        """Add each spot into ``frames`` (n, C, rows, columns) in place: frame ``index`` (m,), in
        channel c at column ``x[:, c]`` and row ``y[:, c]`` (pixel centres at whole numbers),
        ``z_nm``, ``phase`` and ``photons`` (m,) -- the photons of all channels together, as the
        fitter's N."""
        _, C, h, w = frames.shape
        a = torch.arange(self.size, device=self.device)
        ch = torch.arange(C, device=self.device)[None, :, None, None]
        for s in range(0, len(index), self.chunk):
            sl = slice(s, s + self.chunk)
            spot, ox, oy = self.spots(x[sl], y[sl], z_nm[sl], phase[sl])
            spot = spot * photons[sl, None, None, None]
            cols = ox[..., None, None] + a[None, None, None, :]
            rows = oy[..., None, None] + a[None, None, :, None]
            inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
            f = index[sl, None, None, None]
            flat = ((f * C + ch) * h + rows.clamp(0, h - 1)) * w + cols.clamp(0, w - 1)
            frames.view(-1).index_add_(0, flat[inside], spot[inside].to(frames.dtype))
        return frames
