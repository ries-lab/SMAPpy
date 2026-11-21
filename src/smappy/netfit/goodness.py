"""How well each localization's neighbourhood is explained by the fit: a classical fit's likelihood
test for a network's output (off by default; ``NETFIT_LL`` in the C = 2 suite's `run_c2.fit_netfit`),
cheap beside the network itself: drawn and summed in torch.

**Why.**  A network's reported sigma says how sure it is; it cannot say that the frame holds something
it did not explain -- a neighbour it missed or merged, a spot put a few pixels off.  The data can: draw
every localization of the frame with the spline it was trained on, add each one's background, and
compare with the frames read.  Where the fit is right, the difference is noise; where it is not, the
patch around the localization keeps a residual.  It is SMAP's ``LLrel`` for a network (Ries 2020):
nothing is fitted again, only the fit's own numbers rendered.

**The statistic.**  Poisson deviance over a square of ``2 radius + 1`` pixels around the localization,
in each channel (channel 1's square where the registration puts its partner):
``D = 2 sum(m - d + d log(d / m))`` (``2 m`` where ``d <= 0``), divided by the camera's excess noise
(2 for an EM register, whose photons scatter twice as much), and put on a common scale,
``g = (D - n) / sqrt(2 n)`` for ``n`` pixels: about 0 where the model explains the data, a few
standard deviations where it does not.  The model ``m`` is every spot of the frame drawn on the grid
plus a background level per frame and channel; ``d`` the frame in photons.
"""
from __future__ import annotations

import numpy as np


def _spots(spline, x, y, z_nm, photons):
    """Each spot's patch (n, size, size) and its corner (row, column), as `training.joint_crlb.spline_spots`
    draws it, without the derivatives: one at a time in numpy, for tests to draw scenes with
    independently of `render.SplineRenderer`."""
    from ..io.calibration import evaluate_spline
    _, _, ny, nx = spline.coeff.shape
    size = max(min(nx, ny) - 3, 5)
    half = size // 2
    zi = np.asarray(spline.z_nm_to_index(np.asarray(z_nm, float)), float)
    patches = np.empty((len(x), size, size))
    corner = np.empty((len(x), 2), int)
    for k in range(len(x)):
        ox, oy = int(np.rint(x[k])) - half, int(np.rint(y[k])) - half
        patches[k] = photons[k] * np.clip(np.asarray(evaluate_spline(spline, x[k] - ox, y[k] - oy, zi[k], size)), 0, None)
        corner[k] = (oy, ox)
    return patches, corner


def _paint(image, patches, corner):
    h, w = image.shape
    s = patches.shape[-1]
    for p, (r, c) in zip(patches, corner):
        r0, c0 = max(r, 0), max(c, 0)
        r1, c1 = min(r + s, h), min(c + s, w)
        if r1 > r0 and c1 > c0:
            image[r0:r1, c0:c1] += p[r0 - r:r1 - r, c0 - c:c1 - c]


def deviance(data, model, excess: float = 1.0):
    """Poisson deviance of ``data`` against ``model`` (photons, same shape), summed, over ``excess``."""
    m = np.maximum(model, 1e-6)
    d = np.asarray(data, float)
    pos = d > 0
    terms = np.where(pos, m - d + d * np.log(np.where(pos, d, 1.0) / m), m)
    return 2 * terms.sum() / float(excess)


def goodness(splines, stack, frame, xy0, xy1, z_nm, photons, background, excess: float = 1.0,
             radius: int = 4, levels: np.ndarray | None = None, device=None, chunk: int = 32) -> np.ndarray:
    """(n,) ``g`` of each localization (the module's docstring).  ``splines`` the two channels' as the
    network draws them (`channels.splines`); ``stack`` (T, 2, H, W) the frames read, in photons, on
    the grid (`TwoChannels.cut`); ``frame`` (n,) the stack's frame of each; ``xy0`` / ``xy1`` (n, 2)
    its position in each channel's image of the stack (pixels, x along columns); ``photons`` (n,) one
    count, each spline holding its channel's share (biplane), or (n, 2).

    The background is one level a frame and channel: ``background`` (n, 2) per localization -- the
    network's own -- gives each frame the median of its localizations'; None takes each frame's median
    of the data less the drawn spots, from the fit itself and the same rule for any fitter (the frame's
    plain median counts the spots twice where they crowd: at density 50 it put the median ``g`` at
    200).  The levels go into ``levels`` (n, 2) when given, for a bound drawn with the same background
    (`bounds.plugin_crlb`).

    In torch, ``chunk`` frames at a time on ``device`` (CUDA where there is one): every spot drawn with
    `render.SplineRenderer`, the deviance an image, each localization's square summed by a box
    filter -- about 0.1 s for 10 000 localizations on a GPU (an H100 shared with a training, about
    what the network takes), 0.5 -- 1 s on a CPU, where the loop it replaced took 2 s (2026-11-21)."""
    import torch
    import torch.nn.functional as F

    from .render import SplineRenderer
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frame = np.asarray(frame, int)
    photons = np.asarray(photons, float)
    ph = [photons, photons] if photons.ndim == 1 else [photons[:, 0], photons[:, 1]]
    T, _, H, W = stack.shape
    rend = [SplineRenderer(s, dev) for s in splines]
    xys = [np.asarray(xy0, float), np.asarray(xy1, float)]
    k = 2 * radius + 1
    out = np.full(len(frame), np.nan)
    t = lambda v: torch.as_tensor(np.asarray(v, np.float32), device=dev)
    for f0 in range(0, T, chunk):
        f1 = min(f0 + chunk, T)
        rows = np.flatnonzero((frame >= f0) & (frame < f1))
        data = t(stack[f0:f1])                                               # (t, 2, H, W)
        model = torch.zeros_like(data)
        if len(rows):
            idx = torch.as_tensor(frame[rows] - f0, device=dev)
            for c in range(2):
                view = model[:, c].contiguous()
                rend[c].draw(view, idx, t(xys[c][rows, 0]), t(xys[c][rows, 1]), t(z_nm[rows]), t(ph[c][rows]))
                model[:, c] = view
        level = torch.median((data - model).reshape(f1 - f0, 2, -1), dim=2).values             # (t, 2)
        if background is not None and len(rows):
            bg = np.asarray(background, float)[rows]
            for fr in np.unique(frame[rows]):
                level[fr - f0] = t(np.median(bg[frame[rows] == fr], axis=0))
        if levels is not None and len(rows):
            levels[rows] = level.cpu().numpy()[frame[rows] - f0]
        if not len(rows):
            continue
        mu = (model + level[..., None, None]).clamp_min(1e-6)
        pos = data > 0
        dv = torch.where(pos, mu - data + data * torch.log(torch.where(pos, data, torch.ones_like(data)) / mu), mu)
        dv = 2 * dv / float(excess)
        # each pixel's square of k x k summed, the image padded by the radius so that a square reaching
        # beyond it counts only the pixels inside, as a patch clipped to the frame does
        pad = lambda v: F.pad(v.reshape(-1, 1, H, W), (radius,) * 4)
        box = lambda v: F.avg_pool2d(v, k, stride=1, padding=radius, count_include_pad=True) * (k * k)
        sums = box(pad(dv)).reshape(f1 - f0, 2, H + 2 * radius, W + 2 * radius)
        counts = box(pad(torch.ones_like(dv[:1, :1]).expand(1, 1, H, W))).reshape(H + 2 * radius, W + 2 * radius)
        D = np.zeros(len(rows))
        n = np.zeros(len(rows))
        sums, counts = sums.cpu().numpy(), counts.cpu().numpy()
        for c in range(2):
            cx = np.rint(xys[c][rows, 0]).astype(int) + radius
            cy = np.rint(xys[c][rows, 1]).astype(int) + radius
            ok = (cx >= 0) & (cx < W + 2 * radius) & (cy >= 0) & (cy < H + 2 * radius)
            D[ok] += sums[frame[rows][ok] - f0, c, cy[ok], cx[ok]]
            n[ok] += counts[cy[ok], cx[ok]]
        good = n > 0
        out[rows[good]] = (D[good] - n[good]) / np.sqrt(2 * n[good])
    return out


def fit_goodness(two, cal, kind: str, stack, frame, xy_box, z_nm, photons, background, em: bool,
                 radius: int = 4, levels: np.ndarray | None = None, device=None) -> np.ndarray:
    """`goodness` of a network's output on its grid: ``xy_box`` (n, 2) channel 0's positions in the
    stack's pixels (the predictor's ``x_pix``, ``y_pix``), channel 1's where the registration puts
    them (`TwoChannels.remainder`); the rest in the grid's order, as the predictor reports it."""
    from .channels import splines
    r0, _, c0, _ = two.box
    xy_box = np.asarray(xy_box, float)
    xy1 = xy_box + two.remainder(xy_box + [c0, r0])
    return goodness(splines(cal, kind, two.main), stack, frame, xy_box, xy1, z_nm, photons, background,
                    2.0 if em else 1.0, radius, levels, device)
