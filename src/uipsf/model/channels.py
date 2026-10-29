"""Channels: one emitter seen through several detection paths (plan §3, §4; M4).

A `Channel` is a PSF of its own (optics, pupil, blur, analyser) and the
three things that relate it to the main channel:

- **the transform**, a 3×3 projective map from this channel's chip pixels
  to the main channel's, SMAPpy's `DualColorCalibration.transformation`:
  the same object, the same direction (secondary → main), zero-based pixel
  centres on the camera chip.  A mirrored split is in it (a negative
  scale), so the PSF of a channel is always in the camera's own
  orientation, as SMAPpy's secondary spline is;
- **the photon share** f_c: the fraction of the emitter's photons that
  reaches the channel.  Shares sum to 1 over the channels, so the emitter's
  photon count N is its total, SMAPpy's convention for a shared photon
  number (`positive_pair_models`, ``photon_normalization``): μ_c = f_c·N·I_c
  + b_c;
- **the defocus offset** δz_c: the emitter's height above channel c's
  focal plane is z + δz_c, z its height above the main channel's.
  Biplane is a δz of hundreds of nm; for two colours it is the chromatic
  focal shift.  Both channels then share one z, so SMAPpy samples both on
  one z grid and the offset is in the PSFs, not the grid (plan §6.4).

The main channel (index 0) has none of them as parameters: its transform
is the identity, its share is 1 − Σ the others', its offset 0.

**Positions.**  The emitter's (x, y) are the main channel's, in nm from the
centre of its ROI.  In channel c they are where the inverse transform puts
them, relative to that channel's ROI centre: with ``roi_xy[e, c]`` the
chip pixel of ROI c's centre (`Batch.roi_xy`),

    P = roi_xy[e, 0] + (x / px₀, y / py₀),     Q = H⁻¹ P,     (x_c, y_c) = (Q − roi_xy[e, c]) · (px_c, py_c).

The transform is applied exactly, not linearised about the ROI centre as
SMAPpy's fitter does (`multichannel.build_link`): it is cheap, and the fit is
where the transform is refined.

**Parameter names.**  The main channel keeps the single-channel names
(``pupil.phase``, ``psf.sigma``), so a one-channel model is the M3 model;
channel c ≥ 1 prefixes its own with ``ch<c>.`` (``ch1.pupil.phase``) and
has ``ch<c>.transform`` (the 8 entries of H with H₂₂ = 1, row by row),
``ch<c>.share`` and ``ch<c>.dz``.  The PSF of channel c reads its
parameters through `view`, under the unprefixed names.

**Polarisation.**  A channel is unpolarised unless it says otherwise
(``analyser`` of its PSF): uiPSF v1 modelled every two-channel vector fit
as an x- and a y-polarised channel by default, so biplane and ratiometric
data were fitted with the wrong PSFs (review F1).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..params import Bounded, ParamSpec


def prefix(c: int) -> str:
    """The prefix of channel ``c``'s own parameter names: none for the main channel."""
    return "" if c == 0 else f"ch{c}."


def view(p: dict, c: int) -> dict:
    """``p`` as channel ``c``'s PSF reads it: its prefixed parameters under their plain names."""
    if c == 0:
        return p
    pre = prefix(c)
    out = dict(p)
    out.update({k[len(pre):]: v for k, v in p.items() if k.startswith(pre)})
    return out


def unpack_transform(h: torch.Tensor) -> torch.Tensor:
    """The 3×3 matrix from its 8 free entries (H₂₂ = 1)."""
    return torch.cat([h.reshape(-1), h.new_ones(1)]).reshape(3, 3)


def normalised(H) -> np.ndarray:
    """A 3×3 projective matrix scaled so that H₂₂ = 1 (the same map; SMAPpy scales to unit norm)."""
    H = np.asarray(H, float)
    if H.shape != (3, 3) or abs(H[2, 2]) < 1e-12:
        raise ValueError("a channel transform is a 3×3 projective matrix with H[2, 2] ≠ 0")
    return H / H[2, 2]


def map_points(H: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """H applied to points ``xy`` [..., 2] (homogeneous coordinates, divided out)."""
    q = xy[..., 0:1] * H[:, 0] + xy[..., 1:2] * H[:, 1] + H[:, 2]
    return q[..., :2] / q[..., 2:]


def inverse_map(H: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """H⁻¹ applied to ``xy``, through the adjugate (H⁻¹ up to a scale, which the division removes):
    no matrix inverse, so the same operations on every device."""
    a, b, c = H[0], H[1], H[2]
    adj = torch.stack([torch.stack([b[1] * c[2] - b[2] * c[1], a[2] * c[1] - a[1] * c[2], a[1] * b[2] - a[2] * b[1]]),
                       torch.stack([b[2] * c[0] - b[0] * c[2], a[0] * c[2] - a[2] * c[0], a[2] * b[0] - a[0] * b[2]]),
                       torch.stack([b[0] * c[1] - b[1] * c[0], a[1] * c[0] - a[0] * c[1], a[0] * b[1] - a[1] * b[0]])])
    return map_points(adj, xy)


class Channel:
    def __init__(self, psf, transform=None, share: float | None = None, dz_nm: float | None = None,
                 name: str | None = None, fit_transform: bool = True):
        """``psf`` the channel's `VectorPSF`; ``transform`` its 3×3 map to the main channel's chip pixels
        (ignored for the main channel; default the identity); ``share`` and ``dz_nm`` the starting photon
        share and defocus offset, from the data when None (`model.start.rough_start`).
        ``fit_transform=False`` keeps the transform fixed (fewer than four pairs cannot fix eight entries)."""
        self.psf = psf
        self.transform = normalised(np.eye(3) if transform is None else transform)
        self.share = None if share is None else float(share)
        self.dz_nm = None if dz_nm is None else float(dz_nm)
        self.name = name
        self.fit_transform = bool(fit_transform)

    @property
    def pixel_size_nm(self) -> tuple[float, float]:
        return self.psf.grid.optics.pixel_size_nm

    def specs(self, c: int, start: dict | None) -> list[ParamSpec]:
        """Channel ``c``'s parameters: its PSF's (prefixed) and, for c ≥ 1, transform, share and offset."""
        pre = prefix(c)
        specs = [replace(s, name=pre + s.name) for s in self.psf.specs()]
        if c == 0:
            return specs
        start = start or {}
        share = self.share if self.share is not None else float(start.get("shares", {}).get(c, 0.5))
        dz = self.dz_nm if self.dz_nm is not None else float(start.get("dz", {}).get(c, 0.0))
        return specs + [
            ParamSpec(pre + "transform", (8,), init=torch.as_tensor(self.transform.reshape(-1)[:8]),
                      fixed=not self.fit_transform),
            # Bounded, so the share stays a fraction; shares of several secondary channels are not
            # kept below 1 together (two channels need only this).
            ParamSpec(pre + "share", (), init=min(max(share, 1e-3), 1 - 1e-3), transform=Bounded(0.0, 1.0)),
            ParamSpec(pre + "dz", (), init=dz)]
