"""Starting values from the data alone, so that a fit needs no settings for them.

Rough on purpose: the solver does the rest, and Levenberg–Marquardt
converges from these on the review's beads (tests/uipsf_tests/test_benchmark.py).
- background: the lowest, over slices, of the median of each slice's
  border pixels (a defocused spot reaches the border, so the quietest
  slice is the best estimate; the median of the border is precise enough
  that its minimum over slices is biased by less than its noise), and at
  least one photon.  A dim channel's border median moves in steps of the
  camera's gain: Zenodo 2ch's channel 0 in 4.7-photon steps around 11
  photons, whose minimum over 40 slices was 0 or below for 50 of 88 beads.
  They started at softplus' old 10⁻³ floor, where d(value)/du is 10⁻³ --
  the stall of roadmap item 4 (docs/uipsf/architecture.md, "Why Zenodo 2ch
  stalled");
- photons: the upper quartile, over slices, of the background-free sum;
- position: z from the slice with the highest peak (the bead is in focus
  there, so its z is minus that slice's), x and y from the centroid there.

With several channels the position comes from the main channel; the
photons are the sum over channels, each channel's share the median over
emitters of its fraction, and each channel's defocus offset the median of
how far its sharpest slice lies from the main channel's (`channels.py`).
"""
from __future__ import annotations

import torch

# Where softplus' slope is 0.63, so a background below it can still go anywhere in a few steps.
BACKGROUND_FLOOR = 1.0


def _one_channel(d, z_nm, pixel_size_nm):
    E, K, Y, X = d.shape
    border = torch.cat([d[..., 0, :], d[..., -1, :], d[..., 1:-1, 0], d[..., 1:-1, -1]], -1)
    bg = border.median(-1).values.amin(1)                        # [E]
    signal = d - bg[:, None, None, None]
    photons = signal.sum((-2, -1)).quantile(0.75, dim=1).clamp(min=1.0)
    k = signal.amax((-2, -1)).argmax(1)                          # [E]
    z_nm = z_nm if z_nm.dim() == 2 else z_nm.expand(E, K)
    z = -z_nm[torch.arange(E), k]
    w = signal[torch.arange(E), k].clamp(min=0)                  # [E, Y, X]
    w = w / w.sum((-2, -1), keepdim=True).clamp(min=1e-12)
    xs = (torch.arange(X, dtype=d.dtype, device=d.device) - (X - 1) / 2) * pixel_size_nm[0]
    ys = (torch.arange(Y, dtype=d.dtype, device=d.device) - (Y - 1) / 2) * pixel_size_nm[1]
    x = (w.sum(-2) * xs).sum(-1)
    y = (w.sum(-1) * ys).sum(-1)
    return torch.stack([x, y, z.to(d.dtype)], -1), photons, bg.clamp(min=BACKGROUND_FLOOR)


def rough_start(batch, pixel_size_nm: tuple[float, float]) -> dict[str, torch.Tensor]:
    C = batch.data.shape[1]
    xyz, photons, bg = _one_channel(batch.data[:, 0], batch.z_nm, pixel_size_nm)
    if C == 1:
        return {"xyz": xyz, "photons": photons, "background": bg}
    found = [(xyz, photons, bg)] + [_one_channel(batch.data[:, c], batch.z_nm, pixel_size_nm)
                                    for c in range(1, C)]
    total = sum(f[1] for f in found)
    return {"xyz": xyz, "photons": total, "background": torch.stack([f[2] for f in found], 1),
            "shares": {c: float((found[c][1] / total).median()) for c in range(C)},
            "dz": {c: float((found[c][0][:, 2] - xyz[:, 2]).median()) for c in range(C)}}
