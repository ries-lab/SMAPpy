"""Extraction: from the network's per-pixel maps to a list of molecules (the part ``extraction``).

The maps are first put into physical units (`physical`): p, photons, x and y as absolute columns
and rows (each pixel's centre plus its offset), z in nm, their uncertainties, the background.

* ``lookahead`` -- DECODE's (Speiser et al. 2021): a pixel is a detection where its p is above
  ``RAW_P`` (0.1) and the largest in its 3 x 3; its probability is p summed over itself and its four
  adjacent pixels (a molecule near a pixel's edge splits its p); a pixel next to a detection whose
  own p exceeds ``SPLIT_P`` is a second detection, two molecules in adjacent pixels.  The summed
  probability is capped at 1 and thresholded; each detection takes its pixel's values.
* ``mixture_peak`` -- the same detections, each position moved to the peak of the p-weighted
  mixture of its 3 x 3 pixels' components (the benchmark's finding: halves the pixelation far from
  focus, no change near it; `benchmarks/decode/suite/extraction.py`).

On a finer output grid (``scale`` 2, `network.centres`) a molecule spreads its p over up to 2 x 2
cells, the diagonal one included, so a detection's probability is summed over its 3 x 3 (``summed``
"square"; "plus" is DECODE's) -- a cell is half a pixel, so the square spans 1.5 pixels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SPLIT_P = 0.6
RAW_P = 0.1                     # a pixel's own p a peak needs; the summed probability is thresholded
MEAN_SHIFT_ITERATIONS = 10


def physical(out, scaling, scale: int = 1):
    """(b, 10, h, w) network units -> the same in photons, columns / rows, nm; ``scale`` the output
    grid's cells a pixel."""
    from .network import grid
    _, _, h, w = out.shape
    rows, cols = grid(h, w, scale, out.device, out.dtype)
    y = out.clone()
    y[:, 1] *= scaling.phot_max
    y[:, 2] += cols
    y[:, 3] += rows
    y[:, 4] *= scaling.z_max
    y[:, 5] *= scaling.phot_max
    y[:, 8] *= scaling.z_max
    y[:, 9] *= scaling.bg_max
    return y


_KERNELS = {"plus": [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], "square": [[1.0] * 3] * 3}


def detections(p, threshold, raw: float = RAW_P, summed_over: str = "plus"):
    """(frame, row, column) of the detections and their probability: a peak needs its own p above
    ``raw`` only, a detection its summed probability at ``threshold`` -- a molecule near a pixel's
    edge splits its p, each half below the threshold (DECODE: 0.1 and 0.5)."""
    peak = (p > raw) & (p == F.max_pool2d(p[:, None], 3, 1, padding=1)[:, 0])
    kernel = torch.tensor(_KERNELS[summed_over], device=p.device, dtype=p.dtype)
    summed = F.conv2d(p[:, None], kernel[None, None], padding=1)[:, 0]
    second = (~peak) & (p > SPLIT_P)
    prob = ((peak | second).to(p.dtype) * summed).clamp(max=1.0)
    b, i, j = (prob >= threshold).nonzero(as_tuple=True)
    return b, i, j, prob[b, i, j]


def mixture_peak(y, b, i, j):
    """Each detection's position at the peak of its 3 x 3 neighbourhood's p-weighted mixture (mean
    shift with diagonal Gaussians), in x, y and z; (m, 3) columns, rows, nm."""
    _, _, h, w = y.shape
    d = torch.tensor([-1, 0, 1], device=y.device)
    ii = (i[:, None, None] + d[None, :, None]).expand(-1, 3, 3).reshape(len(i), 9)
    jj = (j[:, None, None] + d[None, None, :]).expand(-1, 3, 3).reshape(len(j), 9)
    inside = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    ii, jj = ii.clamp(0, h - 1), jj.clamp(0, w - 1)
    bb = b[:, None].expand_as(ii)
    p = y[bb, 0, ii, jj] * inside
    mu = torch.stack([y[bb, c, ii, jj] for c in (2, 3, 4)], -1)
    sig = torch.stack([y[bb, c, ii, jj] for c in (6, 7, 8)], -1).clamp_min(1e-3)
    x = mu[:, 4].clone()
    inv = 1 / sig ** 2
    for _ in range(MEAN_SHIFT_ITERATIONS):
        logw = torch.log(p.clamp_min(1e-12)) - 0.5 * (((x[:, None] - mu) ** 2) * inv).sum(-1) - torch.log(sig).sum(-1)
        wgt = (torch.softmax(logw, 1) * (p > 0))[..., None]
        x = (wgt * inv * mu).sum(1) / (wgt * inv).sum(1).clamp_min(1e-12)
    return x


def extract(out, scaling, threshold: float, how: str = "lookahead", scale: int = 1,
            summed_over: str | None = None) -> dict:
    """The molecules of every frame of ``out`` (b, 10, h, w), as tensors: frame (in the batch),
    x, y (columns, rows), z_nm, photons, prob, their uncertainties, background.  ``scale``: the
    output grid's cells a pixel (`network.TwoStageUNet.out_scale`); ``summed_over`` "plus" or
    "square", by default plus at scale 1 and square on a finer grid."""
    y = physical(out, scaling, scale)
    b, i, j, prob = detections(y[:, 0], threshold, summed_over=summed_over or ("plus" if scale == 1 else "square"))
    v = y[b, :, i, j]                                                             # (m, 10)
    xyz = v[:, 2:5]
    if how == "mixture_peak" and len(b):
        xyz = mixture_peak(y, b, i, j)
    elif how not in ("lookahead", "mixture_peak"):
        raise ValueError(f"extraction {how!r}: lookahead or mixture_peak")
    return {"frame": b, "x": xyz[:, 0], "y": xyz[:, 1], "z_nm": xyz[:, 2], "photons": v[:, 1], "prob": prob,
            "photons_err": v[:, 5], "x_err": v[:, 6], "y_err": v[:, 7], "z_err_nm": v[:, 8], "background": v[:, 9]}
