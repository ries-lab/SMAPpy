"""The loss: DECODE's three terms (Speiser et al. 2021, Methods), in network units (`Scaling`).

For one frame with molecules ``m`` (photons, x, y, z) and the network's maps:

* **mixture** -- the molecules' likelihood under a mixture over the pixels: component k has the
  weight ``p_k / sum p``, the mean (its photons, its pixel's centre plus its offsets, its z) and the
  diagonal Gaussian of its four sigmas; ``-sum_m log sum_k w_k N(m; mu_k, sigma_k)``.  A molecule
  may be explained by any pixel, which is what lets the network place it within +-1 pixel of a pixel;
* **count** -- the number of molecules against the detection map: ``sum p`` has the mean and
  ``sum p (1 - p)`` the variance of a sum of independent Bernoullis, approximated as a Gaussian;
  ``-n log N(n; sum p, sum p (1 - p))``, weighted by n as in DECODE so it keeps pace with the
  mixture term, which grows with n;
* **background** -- the background map's squared error, summed over the pixels.

Each frame's loss is the sum; the batch's the mean over frames.  The part ``loss`` names the terms
(``terms``) and their weights (``weights``), so that step 4 can change one.
"""
from __future__ import annotations

import math

import torch

LOG_2PI = math.log(2 * math.pi)


def targets_scaled(batch, scaling):
    """The batch's molecules in network units: (b, k, 4) photons, x, y, z."""
    xyz = batch.xyz
    return torch.stack([batch.photons / scaling.phot_max, xyz[..., 0], xyz[..., 1], xyz[..., 2] / scaling.z_max], -1)


def terms(out, batch, scaling, chunk: int = 64) -> dict:
    """Each term per frame, (b,) tensors."""
    from .network import grid
    b, _, h, w = out.shape
    p = out[:, 0].reshape(b, -1)                                                   # (b, hw)
    # a finer output grid's cells a pixel (`network.TwoStageUNet.out_scale`), from the frames it saw
    scale = w // batch.frames.shape[-1] if batch.frames is not None else 1
    rows, cols = grid(h, w, scale, out.device, out.dtype)
    mu = torch.stack([out[:, 1], out[:, 2] + cols, out[:, 3] + rows, out[:, 4]], -1).reshape(b, -1, 4)
    sig = out[:, 5:9].permute(0, 2, 3, 1).reshape(b, -1, 4)
    tgt = targets_scaled(batch, scaling)                                          # (b, k, 4)
    mask = batch.mask
    n = mask.sum(1).to(out.dtype)
    # the mixture: log sum_k w_k N(t; mu_k, sig_k), chunked over the molecules for memory
    log_w = torch.log(p.clamp_min(1e-8)) - torch.log(p.sum(1, keepdim=True).clamp_min(1e-8))
    norm = -0.5 * 4 * LOG_2PI - torch.log(sig).sum(-1)                           # (b, hw)
    mix = torch.zeros(b, device=out.device, dtype=out.dtype)
    for s in range(0, tgt.shape[1], chunk):
        t = tgt[:, s:s + chunk]                                                   # (b, c, 4)
        z = (t[:, :, None] - mu[:, None]) / sig[:, None]                         # (b, c, hw, 4)
        lp = torch.logsumexp(log_w[:, None] + norm[:, None] - 0.5 * (z ** 2).sum(-1), -1)   # (b, c)
        mix = mix - (lp * mask[:, s:s + chunk]).sum(1)
    # the count
    mean = p.sum(1)
    var = (p * (1 - p)).sum(1).clamp_min(1e-6)
    count = n * (0.5 * (n - mean) ** 2 / var + 0.5 * torch.log(2 * math.pi * var))
    # the background
    bg = ((out[:, 9] - (batch.background / scaling.bg_max)[:, None, None]) ** 2).sum((1, 2))
    return {"mixture": mix, "count": count, "background": bg}


def loss(out, batch, scaling, part=None) -> tuple:
    """The batch's loss and its terms' means (0-d tensors), for the part ``part`` (`settings.Part`)."""
    settings = part.settings if part is not None else {}
    names = settings.get("terms", ["count", "mixture", "background"])
    weights = settings.get("weights", {})
    t = terms(out, batch, scaling)
    total = sum(weights.get(k, 1.0) * t[k] for k in names).mean()
    # the terms as tensors: a float() here would wait for the GPU every step (`train`)
    return total, {k: v.detach().mean() for k, v in t.items()}
