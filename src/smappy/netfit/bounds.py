"""The Cramer-Rao bound of each localization from the fit's own estimates, no truth needed: a reference
beside the network's reported sigma (off by default -- `plugin_crlb` evaluates the spline seven times a
spot and channel, some 20 s for 20 000 localizations).

**Plug-in.**  The bound depends on the molecule's photons, background and z and on where it sits on
the pixels, all of which the fit estimates; put in for the truth, they give the bound to a few percent
(median 1.00 -- 1.01, 80 % within 5 -- 10 % at density 1, within 20 % at 1 000 photons; colour and
biplane, early fusion, 2026-11-21).  It is the single-emitter bound: a neighbour's photons are not in
it, which a fit in a crowded frame pays for.

**Two channels** on the network's grid (`channels.TwoChannels`): channel 0's spot at the localization,
channel 1's where the registration puts it (``layout.to_channel``), each drawn with the spline the
network was trained on (`channels.splines`); channel 1's x and y derivatives through the map's
Jacobian onto channel 0's, x, y and z shared, the photons one (biplane) or one per channel (colour), a
background per channel, doubled in variance for an EM register.

**Several frames** (``window`` 3): a fit that reads frames ``t - 1 .. t + 1`` sees a molecule still on
there.  The detection nearest to it in each of those frames, within ``link_px``, is taken as the same
molecule, and their information on x, y and z -- each frame's photons and backgrounds taken out --
adds.  A molecule the fit missed in a neighbouring frame is left out, so the bound is the window's
where the fit found the molecule in it, the frame's where not.
"""
from __future__ import annotations

import numpy as np


def information(spline0, spline1, xy0, xy1, jacobian, z_nm, photons, background, excess: float = 1.0):
    """(n, 3, 3): each spot's Fisher information on x, y (channel 0's pixels) and z (nm), its photons
    and the two backgrounds taken out (the Schur complement): what adds over frames.

    ``xy0``, ``xy1`` (n, 2) the spot's positions in each channel's pixels; ``jacobian`` (n, 2, 2)
    d(xy1) / d(xy0), ``[k, a, b]`` = d xy1_a / d xy0_b; ``photons`` (n,) one count (each spline holds
    its channel's share) or (n, 2) one per channel; ``background`` (n, 2) photons a pixel."""
    from ..training.joint_crlb import spline_spots
    photons = np.asarray(photons, float)
    P = 1 if photons.ndim == 1 else photons.shape[1]
    n = len(z_nm)
    big = (1 << 15, 1 << 15)                                   # a spot alone: never clipped
    fisher = np.zeros((n, 3 + P + 2, 3 + P + 2))
    for c, (spline, xy) in enumerate(((spline0, xy0), (spline1, xy1))):
        ph = photons if P == 1 else photons[:, c]
        s = spline_spots(spline, np.zeros(n, int), xy[:, 0] + 4096, xy[:, 1] + 4096, z_nm, ph,
                         np.zeros(1), big)
        g = s.grads.reshape(n, 4, -1)
        if c == 1:
            j = np.asarray(jacobian, float)
            gx = j[:, 0, 0, None] * g[:, 0] + j[:, 1, 0, None] * g[:, 1]
            gy = j[:, 0, 1, None] * g[:, 0] + j[:, 1, 1, None] * g[:, 1]
            g = np.stack([gx, gy, g[:, 2], g[:, 3]], 1)
        mu = np.maximum(s.model.reshape(n, -1) + background[:, c, None], 1e-9)
        d = np.zeros((n, 3 + P + 2, g.shape[-1]))
        d[:, :3] = g[:, :3]
        d[:, 3 + (c if P == 2 else 0)] = g[:, 3]
        d[:, 3 + P + c] = 1.0
        fisher += np.einsum("nip,njp->nij", d / mu[:, None, :], d) / float(excess)
    a, b, cc = fisher[:, :3, :3], fisher[:, :3, 3:], fisher[:, 3:, 3:]
    return a - b @ np.linalg.solve(cc, np.transpose(b, (0, 2, 1)))


def links(frame, xy, half: int = 1, link_px: float = 1.0):
    """For each localization, the nearest localization in each frame ``frame +- 1 .. half`` within
    ``link_px``: a list of (rows, partners) pairs, one per frame offset."""
    from scipy.spatial import cKDTree
    frame = np.asarray(frame, int)
    out = []
    by_frame = {int(f): np.flatnonzero(frame == f) for f in np.unique(frame)}
    trees = {f: cKDTree(xy[i]) for f, i in by_frame.items()}
    for d in [k for k in range(-half, half + 1) if k]:
        rows, partners = [], []
        for f, i in by_frame.items():
            if f + d not in trees:
                continue
            dist, j = trees[f + d].query(xy[i], distance_upper_bound=link_px)
            ok = np.isfinite(dist)
            rows.append(i[ok])
            partners.append(by_frame[f + d][j[ok]])
        if rows:
            out.append((np.concatenate(rows), np.concatenate(partners)))
    return out


def plugin_crlb(two, cal, kind: str, frame, xy, z_nm, photons, background, pixel_nm, em: bool,
                window: int = 1, link_px: float = 1.0):
    """(n, 3) the bound in x, y, z (nm) of localizations on the network's grid: ``xy`` (n, 2) channel
    0's positions in the layout's channel-0 pixels (the grid's, its box's corner added), ``photons``
    (n,) or (n, 2) and ``background`` (n, 2) in the grid's order, as the network reports them;
    ``window`` the frames the fit reads (3: the molecule's neighbouring detections added)."""
    from .channels import splines
    s0, s1 = splines(cal, kind, two.main)
    xy = np.asarray(xy, float)
    to1 = lambda p: two.layout.to_channel(1, p)
    step = 1e-3
    jac = np.stack([(to1(xy + [step, 0]) - to1(xy - [step, 0])) / (2 * step),
                    (to1(xy + [0, step]) - to1(xy - [0, step])) / (2 * step)], -1)
    info = information(s0, s1, xy, to1(xy), jac, np.asarray(z_nm, float), photons,
                       np.asarray(background, float), 2.0 if em else 1.0)
    if window > 1:
        total = info.copy()
        for rows, partners in links(frame, xy, window // 2, link_px):
            total[rows] += info[partners]
        info = total
    out = np.full((len(info), 3), np.nan)
    good = np.all(np.isfinite(info.reshape(len(info), -1)), 1)
    try:
        out[good] = np.sqrt(np.clip(np.diagonal(np.linalg.inv(info[good]), axis1=1, axis2=2), 0, None))
    except np.linalg.LinAlgError:
        for k in np.flatnonzero(good):
            try:
                out[k] = np.sqrt(np.clip(np.diag(np.linalg.inv(info[k])), 0, None))
            except np.linalg.LinAlgError:
                pass
    return out * [pixel_nm[0], pixel_nm[1], 1.0]
