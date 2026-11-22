"""The Cramer-Rao bound of each localization from the fit's own estimates, no truth needed: a reference
beside the network's reported sigma (off by default; the spline evaluated seven times a spot and
channel, in torch: about 0.4 s for 10 000 localizations on a GPU shared with a training, some 9 s on
a laptop's CPU, 2026-11-21).

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


def _derivatives(r, x, y, z, N, h=0.01):
    """Spots of the renderer ``r`` with ``N`` photons and their derivatives by x, y (pixels), z (nm)
    and the photons, (m, 4, pixels), and the spots (m, pixels): central differences with the square
    held where the spot is, steps of 0.01 pixel and 0.01 of a z plane."""
    import torch
    half = r.size // 2
    corner = (torch.round(x).long() - half, torch.round(y).long() - half)
    at = lambda dx=0.0, dy=0.0, dz=0.0: r.spots(x + dx, y + dy, z + dz, corner)[0].flatten(1)
    hz = h * r.dz
    p = at()
    n = N[:, None]
    g = torch.stack([n * (at(dx=h) - at(dx=-h)) / (2 * h), n * (at(dy=h) - at(dy=-h)) / (2 * h),
                     n * (at(dz=hz) - at(dz=-hz)) / (2 * hz), p], 1)
    return n * p, g


def information(spline0, spline1, xy0, xy1, jacobian, z_nm, photons, background, excess: float = 1.0,
                device=None, chunk: int = 4096):
    """(n, 3, 3): each spot's Fisher information on x, y (channel 0's pixels) and z (nm), its photons
    and the two backgrounds taken out (the Schur complement): what adds over frames.

    ``xy0``, ``xy1`` (n, 2) the spot's positions in each channel's pixels; ``jacobian`` (n, 2, 2)
    d(xy1) / d(xy0), ``[k, a, b]`` = d xy1_a / d xy0_b; ``photons`` (n,) one count (each spline holds
    its channel's share) or (n, 2) one per channel; ``background`` (n, 2) photons a pixel.

    Each spot drawn alone as `training.joint_crlb.spline_spots` draws it (the spline's size less 3
    around its nearest pixel), in torch with `render.SplineRenderer`, ``chunk`` spots at a time on
    ``device`` (CUDA where there is one), only the (n, 3, 3) coming back."""
    import torch

    from .render import SplineRenderer
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    f64 = torch.float64
    rend = [SplineRenderer(s, dev, dtype=f64) for s in (spline0, spline1)]
    photons = np.asarray(photons, float)
    P = 1 if photons.ndim == 1 else photons.shape[1]
    n = len(z_nm)
    xys = [np.asarray(xy0, float), np.asarray(xy1, float)]
    jac = np.asarray(jacobian, float)
    background = np.asarray(background, float)
    out = np.empty((n, 3, 3))
    t = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=f64, device=dev)
    for a in range(0, n, chunk):
        s = slice(a, a + chunk)
        m = len(z_nm[s])
        z = t(np.asarray(z_nm, float)[s])
        fisher = torch.zeros((m, 3 + P + 2, 3 + P + 2), dtype=f64, device=dev)
        for c in range(2):
            N = t(photons[s] if P == 1 else photons[s, c])
            model, g = _derivatives(rend[c], t(xys[c][s, 0]), t(xys[c][s, 1]), z, N)
            if c == 1:
                j = t(jac[s])
                gx = j[:, 0, 0, None] * g[:, 0] + j[:, 1, 0, None] * g[:, 1]
                gy = j[:, 0, 1, None] * g[:, 0] + j[:, 1, 1, None] * g[:, 1]
                g = torch.stack([gx, gy, g[:, 2], g[:, 3]], 1)
            mu = (model + t(background[s, c])[:, None]).clamp_min(1e-9)
            d = torch.zeros((m, 3 + P + 2, g.shape[-1]), dtype=f64, device=dev)
            d[:, :3] = g[:, :3]
            d[:, 3 + (c if P == 2 else 0)] = g[:, 3]
            d[:, 3 + P + c] = 1.0
            fisher += torch.einsum("nip,njp->nij", d / mu[:, None, :], d) / float(excess)
        A, B, C = fisher[:, :3, :3], fisher[:, :3, 3:], fisher[:, 3:, 3:]
        out[s] = (A - B @ torch.linalg.solve(C, B.transpose(1, 2))).cpu().numpy()
    return out


def drawn_crlb(cal, channels, xy0, shift, z_nm, photons, background, pixel_nm, em: bool,
               device=None) -> np.ndarray:
    """(n, 3) the bound in x, y, z (nm) of two-channel molecules as the training draws them
    (`simulate.draw_batch_2c`): channel 0's spot at ``xy0`` (n, 2), channel 1's at ``xy0 + shift``, each
    from its spline (`channels.splines`), one count (``photons`` (n,), linked biplane) or one per channel
    ((n, 2)), ``background`` (n, 2) each channel's at the molecule.  The training's test set has its
    truth, so this is the bound a two-channel *Follow* scores the inliers against."""
    from .channels import splines
    s0, s1 = splines(cal, channels.kind, channels.main)
    xy0 = np.asarray(xy0, float)
    jac = np.broadcast_to(np.eye(2), (len(xy0), 2, 2))     # the drawn shift is the registration's remainder
    info = information(s0, s1, xy0, xy0 + np.asarray(shift, float), jac, np.asarray(z_nm, float), photons,
                       np.asarray(background, float), 2.0 if em else 1.0, device=device)
    return bound(info, pixel_nm)


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
    return bound(info, pixel_nm)


def bound(info, pixel_nm) -> np.ndarray:
    """(n, 3) the bound in x, y, z (nm) from (n, 3, 3) informations on x, y (pixels) and z (nm); NaN
    where one cannot be inverted."""
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
