"""Extractions of a DECODE-type network's output: the part a run names ``extraction`` (`benchmark.md`).

DECODE's network gives each pixel k a probability p_k that a molecule is near it, a position mu_k
(the pixel's centre plus an offset within +-1 pixel, and z) and its uncertainty sigma_k.  Its loss is
a mixture over the pixels: a molecule's likelihood is ``sum_k p_k N(x; mu_k, sigma_k) / sum_k p_k``.
DECODE's extraction (``decode_lookahead``, `SpatialIntegration`) then forgets the mixture: it finds
the pixels whose p is the largest in their 3 x 3, sums p over the four adjacent pixels, and takes the
position of that one pixel alone.  Where a molecule sits near a pixel's edge the network splits its
probability between the two pixels, each with a position of its own; one is taken, and the positions
pull towards pixel centres -- pixelation.

**The mixture's peak** (the idea is LUNAR's extraction, Fu et al. 2026, written here from its
description; LUNAR's code is GPL and not read) keeps DECODE's detections and their probability, and
moves each one's position to the peak of the mixture of the components around it, in x, y and z:

* ``mixture_peak`` -- the 3 x 3 pixels around the detection;
* ``mixture_pair`` -- the detection's pixel and the one neighbour its position points to (the
  neighbour whose centre is nearest to mu_k).

The peak is found by the mean shift of a Gaussian mixture with diagonal covariances, from the
detection pixel's own mu: ``x <- sum_i w_i(x) mu_i / sigma_i^2 / sum_i w_i(x) / sigma_i^2`` per
coordinate, ``w_i(x) = p_i N(x; mu_i, sigma_i)``, ``ITERATIONS`` times; it never leaves a peak, and
a neighbour belonging to another molecule (its mu far, in x, y or z) has a weight near 0 there.  The
uncertainties, photons and background stay the detection pixel's.
"""
from __future__ import annotations

import contextlib

import torch

ITERATIONS = 10
EXTRACTIONS = ("decode_lookahead", "mixture_peak", "mixture_pair")
_OFFSETS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def mixture_positions(y: torch.Tensor, b, i, j, mode: str = "mixture_peak", iterations: int = ITERATIONS):
    """The mixture peak of each detection (frame ``b``, row ``i``, column ``j``) of ``y``, a network's
    output with its offsets made absolute and its scales undone: (N, 10, H, W) -- p, photons, rows,
    columns, z, the sigmas of photons, rows, columns, z, background.  Returns (M, 3): rows, columns
    (pixels), z (nm)."""
    n, _, h, w = y.shape
    di = torch.tensor([o[0] for o in _OFFSETS], device=y.device)
    dj = torch.tensor([o[1] for o in _OFFSETS], device=y.device)
    ii, jj = i[:, None] + di[None], j[:, None] + dj[None]                       # (M, 9)
    inside = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    ii, jj = ii.clamp(0, h - 1), jj.clamp(0, w - 1)
    bb = b[:, None].expand_as(ii)
    p = y[bb, 0, ii, jj] * inside
    mu = torch.stack([y[bb, c, ii, jj] for c in (2, 3, 4)], -1)                # (M, 9, 3)
    sig = torch.stack([y[bb, c, ii, jj] for c in (6, 7, 8)], -1).clamp_min(1e-3)
    centre = mu[:, 4]
    if mode == "mixture_pair":
        # the neighbour whose centre is nearest the detection's own position, and the detection
        d = (ii.to(y.dtype) - centre[:, :1]) ** 2 + (jj.to(y.dtype) - centre[:, 1:2]) ** 2
        d[:, 4] = float("inf")
        pick = torch.zeros_like(p, dtype=torch.bool)
        pick[torch.arange(len(p)), d.argmin(1)] = True
        pick[:, 4] = True
        p = p * pick
    elif mode != "mixture_peak":
        raise ValueError(f"extraction {mode!r}: one of {EXTRACTIONS}")
    x = centre.clone()
    inv = 1.0 / sig ** 2
    for _ in range(iterations):
        logw = torch.log(p.clamp_min(1e-12)) - 0.5 * (((x[:, None] - mu) ** 2) * inv).sum(-1) - torch.log(sig).sum(-1)
        wgt = torch.softmax(logw, 1)[..., None] * (p > 0)[..., None]
        x = (wgt * inv * mu).sum(1) / (wgt * inv).sum(1).clamp_min(1e-12)
    return x


class MixturePeak:
    """A drop-in for the port's `SpatialIntegration` step: DECODE's detections and probabilities, the
    positions moved to the mixture's peak (`mixture_positions`)."""

    def __init__(self, param, h, w, raw_th, mode):
        from decode_smlm.neuralfitter.coord_transform import Offset2Coordinate
        from decode_smlm.neuralfitter.post_processing import SpatialIntegration
        from decode_smlm.neuralfitter.scale_transform import InverseParamListRescale
        self.rescale = InverseParamListRescale.parse(param)
        self.coords = Offset2Coordinate(xextent=(-0.5, h - 0.5), yextent=(-0.5, w - 0.5), img_shape=(h, w))
        self.si = SpatialIntegration(raw_th=raw_th, xy_unit="px")
        self.raw_th, self.mode = raw_th, mode

    def forward(self, out):
        y = self.coords.forward(self.rescale.forward(out))
        p_ps = self.si._nms(y[:, 0].clone(), self.si.p_aggregation, self.raw_th, self.si._split_th)
        b, i, j = (p_ps >= self.raw_th).nonzero(as_tuple=True)
        em = self.si.forward(y.clone())          # DECODE's own: the same pixels, in the same order
        if len(b):
            em.xyz = mixture_positions(y, b, i, j, self.mode).cpu().to(em.xyz.dtype)
        return em


@contextlib.contextmanager
def extraction(mode: str):
    """Within it, the port's predictor extracts with ``mode`` -- a switch for the benchmark until the
    new fitter has its own extraction part (step 3)."""
    from decode_smlm import predict
    if mode in (None, "decode_lookahead"):
        yield
        return
    original = predict.Predictor._post

    def _post(self, h, w):
        return MixturePeak(self.param, h, w, self.raw_th, mode)

    predict.Predictor._post = _post
    try:
        yield
    finally:
        predict.Predictor._post = original
