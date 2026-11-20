"""Extraction: from the network's per-pixel maps to a list of molecules (the part ``extraction``).

The maps are first put into physical units (`physical`): p, photons, x and y as absolute columns
and rows (each pixel's centre plus its offset), z in nm, their uncertainties, the background.

* ``lookahead`` -- DECODE's (Speiser et al. 2021): a pixel is a detection where its p is above
  ``RAW_P`` (0.1) and the largest in its 3 x 3; its probability is p summed over itself and its four
  adjacent pixels (a molecule near a pixel's edge splits its p); a pixel next to a detection whose
  own p exceeds ``SPLIT_P`` is a second detection, two molecules in adjacent pixels.  The summed
  probability is capped at 1 and thresholded; each detection takes its pixel's values.
* ``threshold`` -- every slot whose own p reaches the threshold, at its own values (`by_threshold`):
  for a network trained with the matched loss, which teaches one slot per molecule.
* ``mixture_peak`` -- the same detections as ``lookahead``, each position moved to the peak of the p-weighted
  mixture of its 3 x 3 pixels' components (the benchmark's finding: halves the pixelation far from
  focus, no change near it; `benchmarks/decode/suite/extraction.py`).

Either can be asked to **split** (``split``, a cut on a summed probability; off by default): each
cell with p above ``GATHER_P`` goes to the nearest detection within ``GATHER_PX`` pixels, and a
detection whose gathered p reaches the cut becomes two molecules, at a p-weighted 2-means of its
cells' positions (z scaled by ``Z_PER_PX``).  The map's p summed is the expected number of molecules
there: around a single molecule it comes to about 1, around two 100 - 200 nm apart to about 1.6, and
the detection step, one peak to a 3 x 3, reports two only when the second cell's own p passes
``SPLIT_P`` (13 % of such pairs, 2026-11-14).  A cut of 1.4 - 1.5 found both of a pair half as
often again and split 0.6 -> 0.75 % of isolated molecules; below 1.3 single molecules split fast.

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
GATHER_P = 0.05                 # a cell's p that counts toward its nearest detection's sum (``split``)
GATHER_PX = 1.5                 # how far, in pixels, a cell's position may be from that detection
Z_PER_PX = 200.0                # nm of z that weigh as one pixel when a detection's cells are split in two
KMEANS_ITERATIONS = 10


def physical(out, scaling, scale: int = 1, layout=None):
    """(b, C, h, w) network units -> the same in photons, columns / rows, nm; ``scale`` the output
    grid's cells a pixel.  Every slot's (`network.Layout`)."""
    from .network import Layout, grid
    lay = Layout.of(out, layout)
    _, _, h, w = out.shape
    rows, cols = grid(h, w, scale, out.device, out.dtype)
    y = out.clone()
    P = lay.photons
    for s in range(lay.slots):
        x0, s0 = lay.xyz(s), lay.sig(s)
        # with ``share`` the second photon output is a fraction, left as it is
        y[:, lay.phot(s):lay.phot(s) + (1 if lay.share else P)] *= scaling.phot_max
        y[:, x0] += cols
        y[:, x0 + 1] += rows
        y[:, x0 + 2] *= scaling.z_max
        y[:, s0:s0 + (1 if lay.share else P)] *= scaling.phot_max
        y[:, s0 + P + 2] *= scaling.z_max
    y[:, lay.bg:lay.bg + lay.backgrounds] *= scaling.bg_max
    if lay.reported:                          # the reported sigmas (`network`, ``sigma_rep``)
        y[:, lay.rep + 2] *= scaling.z_max
    return y


def _values(v, lay, k: int = 0) -> dict:
    """A detection's values from its cell's physical channels ``v`` (m, channels), slot ``k``: photons the
    total, and with a count per channel ``photons_c`` (m, P) and their sigmas beside it."""
    P, x0, s0 = lay.photons, lay.xyz(k), lay.sig(k)
    phot, sig_p = v[:, lay.phot(k):x0], v[:, s0:s0 + P]
    if lay.share:                             # the total and the share in channel 1, as counts
        total, f, st, sf = phot[:, 0], phot[:, 1], sig_p[:, 0], sig_p[:, 1]
        phot = torch.stack([total * (1 - f), total * f], 1)
        sig_p = torch.stack([torch.hypot(st * (1 - f), total * sf), torch.hypot(st * f, total * sf)], 1)
        out_total, out_err = total, st
    else:
        out_total, out_err = phot.sum(1), torch.sqrt((sig_p ** 2).sum(1))
    out = {"photons": out_total, "photons_err": out_err,
           "x_err": v[:, s0 + P], "y_err": v[:, s0 + P + 1], "z_err_nm": v[:, s0 + P + 2]}
    if lay.backgrounds == 1:
        out["background"] = v[:, lay.bg]
    else:
        out["background_c"] = v[:, lay.bg:lay.bg + lay.backgrounds]
        out["background"] = out["background_c"][:, 0]
    if P > 1:
        out.update(photons_c=phot, photons_c_err=sig_p)
    return out


def by_threshold(y, threshold: float, layout=None) -> dict:
    """``threshold`` (step 4, the matched loss): every slot of every cell whose own p reaches the
    threshold is a molecule, at its own values -- no peaks, no sums, no merging, the matched loss having
    taught one slot per molecule (`loss.matched_terms`).  ``y`` physical (`physical`)."""
    from .network import Layout
    lay = Layout.of(y, layout)
    k, ps = lay.slots, lay.per_slot
    b, h, w = y.shape[0], y.shape[2], y.shape[3]
    v = y[:, :lay.bg].reshape(b, k, ps, h, w)
    f, s, i, j = (v[:, :, 0] >= threshold).nonzero(as_tuple=True)
    u = v[f, s, :, i, j]                                                     # (m, per slot)
    one = Layout(photons=lay.photons, backgrounds=lay.backgrounds)          # a slot's values as slot 0's
    cell = torch.cat([u, y[f, lay.bg:lay.bg + lay.backgrounds, i, j]], 1)
    x0 = one.xyz()
    return {"frame": f, "x": u[:, x0], "y": u[:, x0 + 1], "z_nm": u[:, x0 + 2], "prob": u[:, 0],
            **_values(cell, one), "row": i, "column": j, "slot": s, "split": torch.zeros_like(i)}


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


def mixture_peak(y, b, i, j, layout=None):
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
    from .network import Layout
    lay = Layout.of(y, layout)
    x0, s0 = lay.xyz(), lay.sig_xyz()
    mu = torch.stack([y[bb, c, ii, jj] for c in (x0, x0 + 1, x0 + 2)], -1)
    sig = torch.stack([y[bb, c, ii, jj] for c in (s0, s0 + 1, s0 + 2)], -1).clamp_min(1e-3)
    x = mu[:, 4].clone()
    inv = 1 / sig ** 2
    for _ in range(MEAN_SHIFT_ITERATIONS):
        logw = torch.log(p.clamp_min(1e-12)) - 0.5 * (((x[:, None] - mu) ** 2) * inv).sum(-1) - torch.log(sig).sum(-1)
        wgt = (torch.softmax(logw, 1) * (p > 0))[..., None]
        x = (wgt * inv * mu).sum(1) / (wgt * inv).sum(1).clamp_min(1e-12)
    return x


def gathered(y, b, xyz, scale: int = 1, layout=None):
    """Each detection's gathered p and the cells it gathered: ``(mass (m,), window cells' index into
    the map's flattened (b, h, w), their ownership mask (m, k))``, every cell with p above
    `GATHER_P` given to the nearest detection of its frame whose position is within `GATHER_PX`."""
    nb, _, h, w = y.shape
    r = round(GATHER_PX * scale) + 1
    dev = y.device
    m = len(b)
    # each detection on the map at its position's cell
    ci = torch.floor((xyz[:, 1] + 0.5) * scale).long().clamp(0, h - 1)
    cj = torch.floor((xyz[:, 0] + 0.5) * scale).long().clamp(0, w - 1)
    owner_map = torch.full((nb, h, w), -1, dtype=torch.long, device=dev)
    owner_map[b, ci, cj] = torch.arange(m, device=dev)
    # the cells that count, and the detections within r cells of each
    cb, ri, rj = torch.nonzero(y[:, 0] > GATHER_P, as_tuple=True)
    d = torch.arange(-r, r + 1, device=dev)
    ii = (ri[:, None, None] + d[None, :, None]).expand(-1, len(d), len(d)).reshape(len(ri), -1)
    jj = (rj[:, None, None] + d[None, None, :]).expand(-1, len(d), len(d)).reshape(len(rj), -1)
    inside = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    cand = owner_map[cb[:, None].expand_as(ii), ii.clamp(0, h - 1), jj.clamp(0, w - 1)]
    cand = torch.where(inside, cand, torch.full_like(cand, -1))
    from .network import Layout
    x0 = Layout.of(y, layout).xyz()
    pos = torch.stack([y[cb, x0, ri, rj], y[cb, x0 + 1, ri, rj]], -1)            # the cells' positions
    dist = torch.linalg.norm(xyz[cand.clamp_min(0), :2] - pos[:, None], dim=-1)
    dist = torch.where(cand >= 0, dist, torch.full_like(dist, float("inf")))
    best, k = dist.min(1)
    own = best <= GATHER_PX
    owner = torch.where(own, cand.gather(1, k[:, None])[:, 0], torch.full_like(best, -1, dtype=torch.long))
    flat_owner = torch.full((nb * h * w,), -1, dtype=torch.long, device=dev)
    flat_owner[(cb * h + ri) * w + rj] = owner
    mass = torch.zeros(m, device=dev, dtype=y.dtype)
    ok = owner >= 0
    mass.index_add_(0, owner[ok], y[cb[ok], 0, ri[ok], rj[ok]])
    # each detection's window of cells, and which of them are its own
    wi = (ci[:, None, None] + d[None, :, None]).expand(-1, len(d), len(d)).reshape(m, -1)
    wj = (cj[:, None, None] + d[None, None, :]).expand(-1, len(d), len(d)).reshape(m, -1)
    win_in = (wi >= 0) & (wi < h) & (wj >= 0) & (wj < w)
    flat = (b[:, None] * h + wi.clamp(0, h - 1)) * w + wj.clamp(0, w - 1)
    mine = win_in & (flat_owner[flat] == torch.arange(m, device=dev)[:, None])
    return mass, flat, mine


def split_two(y, flat, mine, layout=None):
    """Two positions (m, 2, 3) for each detection from its own cells (`gathered`): a p-weighted
    2-means in x, y and z / `Z_PER_PX`, started along the cells' principal axis."""
    nb, c, h, w = y.shape
    yf = y.permute(0, 2, 3, 1).reshape(nb * h * w, c)
    v = yf[flat]                                                                  # (m, k, c)
    p = v[..., 0] * mine
    from .network import Layout
    x0 = Layout.of(y, layout).xyz()
    pts = torch.stack([v[..., x0], v[..., x0 + 1], v[..., x0 + 2] / Z_PER_PX], -1)   # (m, k, 3)
    tot = p.sum(1, keepdim=True).clamp_min(1e-12)
    mu = (p[..., None] * pts).sum(1) / tot
    dev = pts - mu[:, None]
    cov = torch.einsum("mk,mki,mkj->mij", p, dev, dev) / tot[..., None] + 1e-6 * torch.eye(3, device=y.device)
    axis = torch.linalg.eigh(cov)[1][..., -1]                                     # (m, 3)
    spread = torch.sqrt(torch.linalg.eigvalsh(cov)[..., -1]).clamp_min(0.1)
    c1, c2 = mu - spread[:, None] * axis, mu + spread[:, None] * axis
    for _ in range(KMEANS_ITERATIONS):
        g = (torch.linalg.norm(pts - c1[:, None], dim=-1) <= torch.linalg.norm(pts - c2[:, None], dim=-1)).to(p.dtype)
        w1, w2 = p * g, p * (1 - g)
        n1, n2 = w1.sum(1, keepdim=True), w2.sum(1, keepdim=True)
        c1 = torch.where(n1 > 0, (w1[..., None] * pts).sum(1) / n1.clamp_min(1e-12), c1)
        c2 = torch.where(n2 > 0, (w2[..., None] * pts).sum(1) / n2.clamp_min(1e-12), c2)
    out = torch.stack([c1, c2], 1)
    out[..., 2] = out[..., 2] * Z_PER_PX
    return out


def extract(out, scaling, threshold: float, how: str = "lookahead", scale: int = 1,
            summed_over: str | None = None, split: float | None = None, layout=None) -> dict:
    """The molecules of every frame of ``out`` (b, 10, h, w), as tensors: frame (in the batch),
    x, y (columns, rows), z_nm, photons, prob, their uncertainties, background.  ``scale``: the
    output grid's cells a pixel (`network.TwoStageUNet.out_scale`); ``summed_over`` "plus" or
    "square", by default plus at scale 1 and square on a finer grid; ``split`` a cut on the
    gathered p above which a detection is two molecules (`gathered`, `split_two`; a "split" 0 / 1 /
    2 column says which: 1 and 2 the two halves, the values but the position the detection's)."""
    from .network import Layout
    lay = Layout.of(out, layout)
    y = physical(out, scaling, scale, lay)
    if how == "threshold":
        return by_threshold(y, threshold, lay)
    b, i, j, prob = detections(y[:, 0], threshold, summed_over=summed_over or ("plus" if scale == 1 else "square"))
    v = y[b, :, i, j]                                                             # (m, channels)
    x0 = lay.xyz()
    xyz = v[:, x0:x0 + 3]
    if how == "mixture_peak" and len(b):
        xyz = mixture_peak(y, b, i, j, lay)
    elif how not in ("lookahead", "mixture_peak"):
        raise ValueError(f"extraction {how!r}: lookahead, mixture_peak or threshold")
    half = torch.zeros(len(b), dtype=torch.long, device=out.device)
    if split is not None and len(b):
        mass, flat, mine = gathered(y, b, xyz, scale, lay)
        two = torch.nonzero(mass >= split).squeeze(1)
        if len(two):
            pos = split_two(y, flat[two], mine[two], lay)
            xyz = xyz.clone()
            xyz[two] = pos[:, 0]
            idx = torch.cat([torch.arange(len(b), device=out.device), two])
            b, i, j, prob, v = b[idx], i[idx], j[idx], prob[idx], v[idx]
            xyz = torch.cat([xyz, pos[:, 1]])
            half = torch.cat([half, torch.full_like(two, 2)])
            half[two] = 1
    out = {"frame": b, "x": xyz[:, 0], "y": xyz[:, 1], "z_nm": xyz[:, 2], "prob": prob, **_values(v, lay),
           "row": i, "column": j, "split": half}
    if lay.reported:                          # a head that only reports: its sigma is the one reported
        r = lay.rep
        out.update(x_err_mix=out["x_err"], y_err_mix=out["y_err"], z_err_mix_nm=out["z_err_nm"],
                   x_err=v[:, r], y_err=v[:, r + 1], z_err_nm=v[:, r + 2])
    return out
