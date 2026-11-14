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


def match_nearest(t_xyz, t_mask, f_frame, f_xyz, pixel_nm, lateral_nm: float, axial_nm: float):
    """(detection, truth) index pairs, on the device: within the benchmark's gates (`metrics.match`),
    each detection with its nearest truth in its frame where that truth's nearest detection is it --
    mutual nearest neighbours, not the benchmark's assignment, which would wait on the CPU every step.
    ``t_xyz`` (b, k, 3) columns, rows, nm; ``f_xyz`` (m, 3)."""
    if not len(f_frame):
        z = torch.zeros(0, dtype=torch.long, device=f_xyz.device)
        return z, z
    t = t_xyz[f_frame]                                                    # (m, k, 3)
    lat = torch.hypot((t[..., 0] - f_xyz[:, None, 0]) * pixel_nm[0], (t[..., 1] - f_xyz[:, None, 1]) * pixel_nm[1])
    ok = t_mask[f_frame] & (lat <= lateral_nm) & ((t[..., 2] - f_xyz[:, None, 2]).abs() <= axial_nm)
    lat = torch.where(ok, lat, torch.full_like(lat, float("inf")))
    best, k = lat.min(1)                                                  # each detection's nearest truth
    have = torch.isfinite(best)
    # the truth's nearest detection: a scatter of the minimum over detections sharing (frame, truth)
    key = f_frame * t_xyz.shape[1] + k
    nearest = torch.full((t_xyz.shape[0] * t_xyz.shape[1],), float("inf"), device=f_xyz.device)
    nearest = nearest.scatter_reduce(0, key[have], best[have], reduce="amin")
    mine = have & (best <= nearest[key])
    d = torch.nonzero(mine).squeeze(1)
    return d, k[d]


def reported_term(out, batch, scaling, pixel_nm, threshold: float, how: str, scale: int = 1, head: str = "rep",
                  lateral_nm: float = 250.0, axial_nm: float = 500.0):
    """The Gaussian negative log-likelihood of what the extraction reports (step 4, an honest sigma):
    each detection matched to its truth (`match_nearest`), per axis ``log sigma + (sg(mu) - x)^2 /
    (2 sigma^2)`` at its pixel, the position ``mu`` detached so that the term trains the sigma to the
    reported position's error and never the position.  ``head`` "rep" (channels 10 - 12, a head that
    only reports) or "mixture" (the mixture's own sigma, channels 6 - 8).  Per frame, summed over its
    matches; x and y in pixels, z in units of ``z_max``."""
    from . import extract
    with torch.no_grad():
        e = extract.extract(out.detach(), scaling, threshold, how, scale)
        f_xyz = torch.stack([e["x"], e["y"], e["z_nm"]], 1)
        d, k = match_nearest(batch.xyz, batch.mask, e["frame"], f_xyz, pixel_nm, lateral_nm, axial_nm)
    if not len(d):
        return out.sum() * 0.0
    b, i, j = e["frame"][d], e["row"][d], e["column"][d]
    first = 10 if head == "rep" else 6
    sig = out[b, first:first + 3, i, j]                                   # (n, 3) with gradient
    truth = batch.xyz[b, k]
    err = f_xyz[d] - truth
    err = torch.stack([err[:, 0], err[:, 1], err[:, 2] / scaling.z_max], 1)
    nll = torch.log(sig) + err ** 2 / (2 * sig ** 2)
    return nll.sum() / out.shape[0]
