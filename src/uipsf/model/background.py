"""The background per pixel, b[e, k, y, x] in photons (plan §3).

- `Constant`: one level per emitter, the same in every slice.  v1's model.
- `Planar`: a level and a slope in x and y per emitter, for a ramp across
  the ROI, one of the mismatch effects the review found most damaging
  (review §B).  The level is the value at the ROI's centre; the slopes are
  photons per pixel per nm, so the parameters keep physical units.  The
  plane may go below zero at a corner if the slope is steep; the noise model
  then refuses the step rather than clipping it.

Levels are kept positive by a softplus, so μ > 0 by construction.  With several
channels each has its own level (and slopes) per emitter: ``emitter.background``
is then [E, C], one per channel; with one channel it stays [E].
"""
from __future__ import annotations

import torch

from ..params import ParamSpec, Softplus


def _per_channel(batch) -> tuple[int, ...]:
    C = batch.data.shape[1]
    return () if C == 1 else (C,)


class Constant:
    def specs(self, batch, start) -> list[ParamSpec]:
        return [ParamSpec("emitter.background", _per_channel(batch), "emitter", init=start["background"],
                          transform=Softplus())]

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, C, 1, 1, 1], broadcast over slices and pixels."""
        b = p["emitter.background"]
        return b.reshape(b.shape[0], -1)[..., None, None, None]


class Planar(Constant):
    def __init__(self, pixel_size_nm: tuple[float, float]):
        self.pixel_size_nm = tuple(float(v) for v in pixel_size_nm)

    def specs(self, batch, start) -> list[ParamSpec]:
        return super().specs(batch, start) + [ParamSpec("emitter.background_slope", (*_per_channel(batch), 2),
                                                        "emitter")]

    def __call__(self, p, batch) -> torch.Tensor:
        """[E, C, 1, Y, X], every channel in its own ROI's coordinates."""
        Y, X = batch.data.shape[-2:]
        b = p["emitter.background"]
        E, dt, dev = b.shape[0], b.dtype, b.device
        x = (torch.arange(X, dtype=dt, device=dev) - (X - 1) / 2) * self.pixel_size_nm[0]
        y = (torch.arange(Y, dtype=dt, device=dev) - (Y - 1) / 2) * self.pixel_size_nm[1]
        g = p["emitter.background_slope"].reshape(E, -1, 2)
        plane = g[..., 0, None, None] * x + g[..., 1, None, None] * y[:, None]
        return (b.reshape(E, -1)[..., None, None] + plane)[:, :, None]
