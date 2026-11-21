"""How well each localization's neighbourhood is explained by the fit: a classical fit's likelihood
test for a network's output (off by default -- some seconds for 20 000 localizations; ``NETFIT_LL``
in the C = 2 suite's `run_c2.fit_netfit`).

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
plus the localization's own background per channel; ``d`` the frame in photons.
"""
from __future__ import annotations

import numpy as np


def _spots(spline, x, y, z_nm, photons):
    """Each spot's patch (n, size, size) and its corner (row, column), as `training.joint_crlb.spline_spots`
    draws it, without the derivatives."""
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
             radius: int = 4, levels: np.ndarray | None = None) -> np.ndarray:
    """(n,) ``g`` of each localization (the module's docstring).  ``splines`` the two channels' as the
    network draws them (`channels.splines`); ``stack`` (T, 2, H, W) the frames read, in photons, on
    the grid (`TwoChannels.cut`); ``frame`` (n,) the stack's frame of each; ``xy0`` / ``xy1`` (n, 2)
    its position in each channel's image of the stack (pixels, x along columns); ``photons`` (n,) one
    count, each spline holding its channel's share (biplane), or (n, 2); ``background`` (n, 2), or
    None: each frame's median per channel of the data less the drawn spots -- one level a frame, from
    the fit itself, the same rule for any fitter (the frame's plain median counts the spots twice
    where they crowd: at density 50 it put the median ``g`` at 200), written into ``levels`` (n, 2)
    when given, for a bound drawn with the same background (`bounds.plugin_crlb`)."""
    frame = np.asarray(frame, int)
    photons = np.asarray(photons, float)
    ph = [photons, photons] if photons.ndim == 1 else [photons[:, 0], photons[:, 1]]
    background = None if background is None else np.asarray(background, float)
    H, W = stack.shape[-2:]
    drawn = [_spots(splines[c], xy[:, 0], xy[:, 1], z_nm, ph[c]) for c, xy in enumerate((xy0, xy1))]
    out = np.full(len(frame), np.nan)
    for f in np.unique(frame):
        rows = np.flatnonzero(frame == f)
        models = []
        for c in range(2):
            img = np.zeros((H, W))
            _paint(img, drawn[c][0][rows], drawn[c][1][rows])
            models.append(img)
        level = None if background is not None else [float(np.median(stack[f, c] - models[c])) for c in range(2)]
        if level is not None and levels is not None:
            levels[rows] = level
        for k in rows:
            D, n = 0.0, 0
            for c, xy in enumerate((xy0, xy1)):
                cx, cy = int(np.rint(xy[k, 0])), int(np.rint(xy[k, 1]))
                r0, r1 = max(cy - radius, 0), min(cy + radius + 1, H)
                c0, c1 = max(cx - radius, 0), min(cx + radius + 1, W)
                if r1 <= r0 or c1 <= c0:
                    continue
                m = models[c][r0:r1, c0:c1] + (background[k, c] if level is None else level[c])
                D += deviance(stack[f, c, r0:r1, c0:c1], m, excess)
                n += m.size
            if n:
                out[k] = (D - n) / np.sqrt(2 * n)
    return out


def fit_goodness(two, cal, kind: str, stack, frame, xy_box, z_nm, photons, background, em: bool,
                 radius: int = 4, levels: np.ndarray | None = None) -> np.ndarray:
    """`goodness` of a network's output on its grid: ``xy_box`` (n, 2) channel 0's positions in the
    stack's pixels (the predictor's ``x_pix``, ``y_pix``), channel 1's where the registration puts
    them (`TwoChannels.remainder`); the rest in the grid's order, as the predictor reports it."""
    from .channels import splines
    r0, _, c0, _ = two.box
    xy_box = np.asarray(xy_box, float)
    xy1 = xy_box + two.remainder(xy_box + [c0, r0])
    return goodness(splines(cal, kind, two.main), stack, frame, xy_box, xy1, z_nm, photons, background,
                    2.0 if em else 1.0, radius, levels)
