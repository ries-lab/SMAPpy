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
* **background** -- the background map's squared error, summed over the pixels;
* **local_count** (not DECODE's; step 4, close molecules, not on by default) -- the count again, but
  in every ``window`` x ``window`` neighbourhood of cells rather than over the frame: ``-log P(n_w)``
  of the window's true count under its cells' independent Bernoullis, the Poisson-binomial computed
  exactly (`window_counts`), summed over the windows and divided by ``window^2``, the windows each cell
  is in.  The frame's count term cannot tell one molecule too few in one place from one too many in
  another, so nothing teaches the map to hold two where two are close together -- the network's
  probability summed around a pair 100 - 200 nm apart came to 1.6, around a single molecule 1.04
  (2026-11-14).  It is a proper scoring rule of the count, so on molecules placed at random it asks
  for a calibrated count and adds no reason to split a single molecule.  It separated no close pairs
  and pixelated (pixel_mod 0.19 against 0.05, 2026-11-15): not used.
* **matched** (not DECODE's; step 4, close molecules; the loss part ``matched``) -- each true molecule
  given one slot of its own (`network`'s ``slots``) by a one-to-one assignment, as SHOT does (Seailles,
  Masson, Ponce, Mairal 2025, arXiv:2512.10683, implemented here from the paper): a slot matched to
  molecule m costs ``-log N(m; mu, sigma) - log p``, every other slot ``-log (1 - p)``; the assignment
  minimises the frame's total (`matched_terms`).  A molecule may take only a slot whose cell's centre
  is within ``gate_px`` of it on each axis.  Unlike the mixture, two slots on one molecule cost (one is
  unmatched with p high) and one slot on two molecules costs (one molecule takes a slot with p low or
  far away), so p becomes "this slot is a molecule" and the extraction a threshold (`extract.by_threshold`).
  SHOT solves the assignment with Sinkhorn on the GPU; here it is exact (Hungarian, scipy) on the CPU,
  on the few slots within reach of each molecule -- SHOT found the exact assignment no better, Jaccard
  0.873 against 0.883.  The mixture, count and background terms still apply, over all slots, if named.
  With ``assign`` "sinkhorn" the assignment is SHOT's instead (`sinkhorn_terms`): an entropic optimal
  transport, soft, differentiated through, whose coupling spreads the gradient over the slots that
  could take a molecule early in training rather than handing all of it to one -- tried for whether
  it learns faster than the exact assignment (2026-11-15).

With more than one channel (`network.Layout`) a molecule is the photons a slot holds -- one count,
or one per channel -- and x, y, z, so the mixture and the matched loss are over ``P + 3`` dimensions,
and the background term is summed over the channels' maps.

Each frame's loss is the sum; the batch's the mean over frames.  The part ``loss`` names the terms
(``terms``) and their weights (``weights``), so that step 4 can change one.
"""
from __future__ import annotations

import math

import torch

LOG_2PI = math.log(2 * math.pi)


def targets_scaled(batch, scaling, layout=None):
    """The batch's molecules in network units: (b, k, P + 3) photons, x, y, z -- ``batch.photons``
    (b, k) one count, or (b, k, P) one per channel; with the layout's ``share`` the total and the share
    in channel 1 instead of the two counts."""
    xyz = batch.xyz
    phot = batch.photons if batch.photons.dim() == 3 else batch.photons[..., None]
    if layout is not None and layout.share:
        total = phot.sum(-1, keepdim=True)
        phot = torch.cat([total / scaling.phot_max, phot[..., 1:2] / total.clamp_min(1e-6)], -1)
    else:
        phot = phot / scaling.phot_max
    return torch.cat([phot, xyz[..., 0:2], xyz[..., 2:3] / scaling.z_max], -1)


def backgrounds(batch):
    """(b, C): each channel's background of the middle frame, photons a pixel."""
    return batch.background if batch.background.dim() == 2 else batch.background[:, None]


def cell_counts(batch, h: int, w: int, scale: int = 1):
    """(b, h, w): the true molecules in each output cell -- the cell whose centre is nearest
    (`network.centres`: ``(i + 0.5) / scale - 0.5``), molecules outside the frame left out."""
    b = batch.xyz.shape[0]
    col = torch.floor((batch.xyz[..., 0] + 0.5) * scale).long()
    row = torch.floor((batch.xyz[..., 1] + 0.5) * scale).long()
    ok = batch.mask & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    frame = torch.arange(b, device=batch.xyz.device)[:, None].expand_as(col)
    counts = torch.zeros(b * h * w, device=batch.xyz.device)
    counts.index_add_(0, ((frame * h + row) * w + col)[ok], torch.ones(int(ok.sum()), device=batch.xyz.device))
    return counts.view(b, h, w)


def window_counts(p, counts, window: int = 3):
    """(b,): ``-sum_w log P(n_w)`` over the ``window`` x ``window`` neighbourhoods (stride 1, zero
    padding) of ``p`` (b, h, w), ``n_w`` the true molecules in the window (``counts``, `cell_counts`),
    ``P`` the Poisson-binomial of the window's cells: the distribution of a sum of independent
    Bernoullis, built one cell at a time.  More molecules than cells is impossible under it; such a
    window counts as full."""
    import torch.nn.functional as F
    b, h, w = p.shape
    k = window * window
    cells = F.unfold(p[:, None], window, padding=window // 2).clamp(1e-6, 1 - 1e-6)    # (b, k, hw)
    n = F.unfold(counts[:, None], window, padding=window // 2).sum(1).round().long().clamp(max=k)  # (b, hw)
    dist = torch.zeros(b, h * w, k + 1, device=p.device, dtype=p.dtype)
    dist[..., 0] = 1.0
    for c in range(k):
        q = cells[:, c, :, None]
        dist = torch.cat([dist[..., :1] * (1 - q), dist[..., 1:] * (1 - q) + dist[..., :-1] * q], -1)
    pn = dist.gather(-1, n[..., None])[..., 0]
    return -torch.log(pn.clamp_min(1e-30)).sum(1) / k


def _slot_maps(out, h, w, scale, layout=None):
    """p (b, K, h, w), mu and sig (b, K, h, w, P + 3) over every slot of every cell (`network.Layout`)."""
    from .network import Layout, grid
    lay = Layout.of(out, layout)
    b, k, P = out.shape[0], lay.slots, lay.photons
    v = out[:, :lay.bg].reshape(b, k, lay.per_slot, h, w)
    rows, cols = grid(h, w, scale, out.device, out.dtype)
    x0 = 1 + P
    mu = torch.cat([v[:, :, 1:x0], (v[:, :, x0] + cols)[:, :, None], (v[:, :, x0 + 1] + rows)[:, :, None],
                    v[:, :, x0 + 2:x0 + 3]], 2).permute(0, 1, 3, 4, 2)                 # (b, K, h, w, P + 3)
    sig = v[:, :, x0 + 3:].permute(0, 1, 3, 4, 2)
    return v[:, :, 0], mu, sig, k


def terms(out, batch, scaling, chunk: int = 64, names=("count", "mixture", "background"), window: int = 3,
          gate_px: float = 1.5, assign: dict | None = None, layout=None) -> dict:
    """Each term of ``names`` per frame, (b,) tensors: DECODE's three by default, ``local_count`` (its
    ``window`` in cells) and ``matched`` (its ``gate_px``, and ``assign``: {"how": "hungarian" or
    "sinkhorn", "eps", "iterations"})."""
    b, _, h, w = out.shape
    # a finer output grid's cells a pixel (`network.TwoStageUNet.out_scale`), from the frames it saw
    scale = w // batch.frames.shape[-1] if batch.frames is not None else 1
    from .network import Layout
    lay = Layout.of(out, layout)
    pk, mu5, sig5, _ = _slot_maps(out, h, w, scale, lay)
    D = mu5.shape[-1]
    p = pk.reshape(b, -1)                                                         # (b, K hw)
    mu, sig = mu5.reshape(b, -1, D), sig5.reshape(b, -1, D)
    tgt = targets_scaled(batch, scaling, lay)                                     # (b, n, D)
    mask = batch.mask
    n = mask.sum(1).to(out.dtype)
    out_terms = {}
    if "mixture" in names:
        # the mixture: log sum_k w_k N(t; mu_k, sig_k), chunked over the molecules for memory
        log_w = torch.log(p.clamp_min(1e-8)) - torch.log(p.sum(1, keepdim=True).clamp_min(1e-8))
        norm = -0.5 * D * LOG_2PI - torch.log(sig).sum(-1)                       # (b, Khw)
        mix = torch.zeros(b, device=out.device, dtype=out.dtype)
        for s in range(0, tgt.shape[1], chunk):
            t = tgt[:, s:s + chunk]                                               # (b, c, D)
            z = (t[:, :, None] - mu[:, None]) / sig[:, None]                     # (b, c, Khw, D)
            lp = torch.logsumexp(log_w[:, None] + norm[:, None] - 0.5 * (z ** 2).sum(-1), -1)   # (b, c)
            mix = mix - (lp * mask[:, s:s + chunk]).sum(1)
        out_terms["mixture"] = mix
    if "count" in names:
        mean = p.sum(1)
        var = (p * (1 - p)).sum(1).clamp_min(1e-6)
        out_terms["count"] = n * (0.5 * (n - mean) ** 2 / var + 0.5 * torch.log(2 * math.pi * var))
    if "background" in names:
        pred = out[:, lay.bg:lay.bg + lay.backgrounds]
        if getattr(batch, "background_map", None) is not None:       # uneven: the map, on the output's grid
            bg = batch.background_map / scaling.bg_max                               # (b, C, s, s)
            if bg.shape[-2:] != pred.shape[-2:]:
                bg = torch.nn.functional.interpolate(bg, size=pred.shape[-2:], mode="nearest")
        else:
            bg = (backgrounds(batch) / scaling.bg_max)[:, :, None, None]            # (b, C, 1, 1)
        out_terms["background"] = ((pred - bg) ** 2).sum((1, 2, 3))
    if "local_count" in names:
        out_terms["local_count"] = window_counts(pk.sum(1).clamp(max=1 - 1e-6), cell_counts(batch, h, w, scale), window)
    if "matched" in names:
        if (assign or {}).get("how", "hungarian") == "sinkhorn":
            out_terms.update(sinkhorn_terms(pk, mu5, sig5, tgt, mask, batch.xyz, scale, gate_px,
                                            assign.get("eps", 1e-2), assign.get("iterations", 100)))
        else:
            out_terms.update(matched_terms(pk, mu5, sig5, tgt, mask, batch.xyz, scale, gate_px))
    return out_terms


def sinkhorn_terms(p, mu, sig, tgt, mask, xyz, scale: int, gate_px: float = 1.5, eps: float = 1e-2,
                   iterations: int = 100) -> dict:
    """The matched loss with SHOT's assignment (Seailles et al. 2025, arXiv:2512.10683, from the paper):
    per frame a transport from the molecules (mass 1 each) and one "nothing" row (the rest of the slots'
    mass, at no cost) to the slots (mass 1 each), the cost of molecule m in slot s the exact
    assignment's, ``-log N(m; mu, sigma) - log p + log (1 - p)``, a pair outside the gate barred.
    Entropic, regularised by ``eps`` times the median of the allowed costs' magnitudes (SHOT: 1e-4),
    solved by ``iterations`` log-domain Sinkhorn steps (SHOT: 20 -- see below why more); the loss is
    ``sum -log (1 - p) + <coupling, cost>``.  Dense on the device: every molecule against every slot.
    The last step normalises the molecules' rows, so each molecule is placed whole."""
    b, k, h, w = p.shape
    pc = p.clamp(1e-6, 1 - 1e-6).reshape(b, -1)                                 # (b, S), S = K h w
    neg = -torch.log1p(-pc)
    D = mu.shape[-1]
    mu, sig = mu.reshape(b, -1, D), sig.reshape(b, -1, D)
    rows, cols = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing="ij")
    cx = ((cols.reshape(-1) + 0.5) / scale - 0.5).repeat(k)                       # (S,) slot cells' centres
    cy = ((rows.reshape(-1) + 0.5) / scale - 0.5).repeat(k)
    gate = (((cx[None, None] - xyz[..., 0:1]).abs() <= gate_px) & ((cy[None, None] - xyz[..., 1:2]).abs() <= gate_px)
            & mask[..., None])                                                    # (b, n, S)
    z = (tgt[:, :, None] - mu[:, None]) / sig[:, None]                           # (b, n, S, D)
    nll = (torch.log(sig)[:, None] + 0.5 * z ** 2).sum(-1) + 0.5 * D * LOG_2PI
    cost = nll - torch.log(pc)[:, None] + torch.log1p(-pc)[:, None]               # (b, n, S)
    have = gate.any(-1)                                                           # molecules with a slot in reach
    scale_c = cost.detach()[gate].abs().median().clamp_min(1e-3) if gate.any() else cost.new_tensor(1.0)
    e = eps * scale_c
    # a barred pair's kernel finite (-1e30), not -inf: a row of -inf has a NaN gradient through logsumexp
    kern_c = torch.where(gate, -cost.detach(), torch.full_like(cost, -1e30))
    kern_c = torch.cat([kern_c, torch.zeros_like(kern_c[:, :1])], 1)             # the "nothing" row, cost 0
    s_n = pc.shape[1]
    mass = torch.cat([have.to(cost.dtype), (s_n - have.sum(1, keepdim=True)).to(cost.dtype)], 1)   # (b, n + 1)
    log_a = torch.where(mass > 0, torch.log(mass.clamp_min(1e-30)), torch.full_like(mass, -math.inf))
    f = torch.zeros_like(mass)
    g = torch.zeros(b, s_n, device=p.device, dtype=cost.dtype)                    # log b = 0: each slot mass 1
    # Without a gradient, and the regularisation lowered from the costs' scale to ``eps`` over the first
    # half of the steps: at eps 1e-4, 20 steps from cold left two molecules sharing one slot (a toy
    # frame's total 7.3 against the exact 10.7; 2000 steps to reach it).  The coupling is then a
    # constant in the loss, which at the optimum is the transport cost's own gradient (the envelope
    # theorem) -- SHOT differentiates through the steps instead.
    with torch.no_grad():
        for i in range(iterations):
            ei = e * eps ** -max(0.0, 1 - i / max(1, iterations // 2)) if eps < 1 else e
            kern = kern_c / ei
            f = torch.where(mass > 0, log_a - torch.logsumexp(kern + g[:, None], 2), torch.full_like(f, -math.inf))
            g = -torch.logsumexp(kern + f[:, :, None], 1)
        f = torch.where(mass > 0, log_a - torch.logsumexp(kern + g[:, None], 2), torch.full_like(f, -math.inf))
    coupling = torch.exp(kern + f[:, :, None] + g[:, None])[:, :-1]               # (b, n, S), the molecules' rows
    total = neg.sum(1) + (coupling * torch.where(gate, cost, torch.zeros_like(cost))).sum((1, 2))
    return {"matched": total, "unmatched": (mask & ~have).sum(1).to(p.dtype)}


def matched_terms(p, mu, sig, tgt, mask, xyz, scale: int, gate_px: float = 1.5) -> dict:
    """The matched loss (the module's docstring), per frame: "matched" its total, and for the log
    "unmatched" the molecules left without a slot.  ``p`` (b, K, h, w), ``mu``, ``sig`` (b, K, h, w, 4)
    and ``tgt`` (b, n, 4) in network units, ``xyz`` the molecules' columns and rows."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    b, k, h, w = p.shape
    pc = p.clamp(1e-6, 1 - 1e-6)
    neg = -torch.log1p(-pc)                                                       # (b, K, h, w)
    # each molecule's candidate cells: a square of cells around its own, those within the gate
    r = math.ceil(gate_px * scale + 0.5)       # a cell's centre within the gate is at most this many cells off
    d = torch.arange(-r, r + 1, device=p.device)
    col0 = torch.floor((xyz[..., 0] + 0.5) * scale).long()
    row0 = torch.floor((xyz[..., 1] + 0.5) * scale).long()
    cc = col0[..., None, None] + d[None, None, None, :]                          # (b, n, 1, W)
    rr = row0[..., None, None] + d[None, None, :, None]                          # (b, n, W, 1)
    cc, rr = torch.broadcast_tensors(cc, rr)
    cc, rr = cc.reshape(*col0.shape, -1), rr.reshape(*col0.shape, -1)           # (b, n, W2)
    centre_c, centre_r = (cc + 0.5) / scale - 0.5, (rr + 0.5) / scale - 0.5
    ok = ((cc >= 0) & (cc < w) & (rr >= 0) & (rr < h) & mask[..., None]
          & ((centre_c - xyz[..., 0:1]).abs() <= gate_px) & ((centre_r - xyz[..., 1:2]).abs() <= gate_px))
    cc, rr = cc.clamp(0, w - 1), rr.clamp(0, h - 1)
    frame = torch.arange(b, device=p.device)[:, None, None].expand_as(cc)
    # (b, n, W2, K): matching molecule m to slot s of cell c costs -log N - log p + log (1 - p), over all
    # slots' -log (1 - p)
    m_mu = mu[frame, :, rr, cc]                                                   # (b, n, W2, K, 4)
    m_sig = sig[frame, :, rr, cc]
    m_p = pc[frame, :, rr, cc]                                                    # (b, n, W2, K)
    zz = (tgt[:, :, None, None] - m_mu) / m_sig
    nll = (torch.log(m_sig) + 0.5 * zz ** 2).sum(-1) + 0.5 * zz.shape[-1] * LOG_2PI
    cost = nll - torch.log(m_p) + torch.log1p(-m_p)
    big = torch.finfo(cost.dtype).max / 4
    cost_cpu = torch.where(ok[..., None], cost, torch.full_like(cost, big)).detach().cpu().numpy()
    ids = (((torch.arange(k, device=p.device) * h)[None, None, None, :] + rr[..., None]) * w + cc[..., None]).cpu().numpy()
    okn = ok.cpu().numpy()
    fi, mi, ci, si = [], [], [], []
    unmatched = np.zeros(b)
    for f in range(b):
        ms = np.flatnonzero(okn[f].any(1))
        if not len(ms):
            continue
        # only the cells within the gate and the frame: clamped ones would repeat a cell
        sel = np.repeat(okn[f, ms][..., None], k, -1).reshape(len(ms), -1)
        rows = np.repeat(np.arange(len(ms))[:, None], sel.shape[1], 1)[sel]
        places = np.repeat(np.arange(sel.shape[1])[None, :], len(ms), 0)[sel]
        uniq, cols = np.unique(ids[f, ms].reshape(len(ms), -1)[sel], return_inverse=True)
        mat = np.full((len(ms), len(uniq)), big, dtype=np.float64)
        mat[rows, cols] = cost_cpu[f, ms].reshape(len(ms), -1)[sel]
        where = np.zeros((len(ms), len(uniq)), dtype=np.int64)  # the column's place in its molecule's list
        where[rows, cols] = places
        a, c = linear_sum_assignment(mat)
        good = mat[a, c] < big / 2
        a, c = a[good], c[good]
        pos = where[a, c]
        fi.append(np.full(len(a), f)); mi.append(ms[a]); ci.append(pos // k); si.append(pos % k)
        unmatched[f] = len(ms) - len(a)
    total = neg.sum((1, 2, 3))
    if fi:
        fi, mi, ci, si = (torch.as_tensor(np.concatenate(v), device=p.device) for v in (fi, mi, ci, si))
        gain = cost[fi, mi, ci, si]                                               # with gradient
        total = total.index_add(0, fi, gain)
    return {"matched": total, "unmatched": torch.as_tensor(unmatched, device=p.device, dtype=p.dtype)}


DEFAULT_TERMS = {"mixture": ["count", "mixture", "background"], "matched": ["matched", "background"]}


def loss(out, batch, scaling, part=None, layout=None) -> tuple:
    """The batch's loss and its terms' means (0-d tensors), for the part ``part`` (`settings.Part`):
    ``mixture`` (DECODE's) or ``matched``, each with its default terms unless ``terms`` names others."""
    name = part.name if part is not None else "mixture"
    settings = part.settings if part is not None else {}
    names = settings.get("terms", DEFAULT_TERMS[name])
    weights = settings.get("weights", {})
    t = terms(out, batch, scaling, names=names, window=settings.get("window", 3), gate_px=settings.get("gate_px", 1.5),
              assign=settings.get("assign"), layout=layout)
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
                  lateral_nm: float = 250.0, axial_nm: float = 500.0, layout=None):
    """The Gaussian negative log-likelihood of what the extraction reports (step 4, an honest sigma):
    each detection matched to its truth (`match_nearest`), per axis ``log sigma + (sg(mu) - x)^2 /
    (2 sigma^2)`` at its pixel, the position ``mu`` detached so that the term trains the sigma to the
    reported position's error and never the position.  ``head`` "rep" (channels 10 - 12, a head that
    only reports) or "mixture" (the mixture's own sigma, channels 6 - 8).  Per frame, summed over its
    matches; x and y in pixels, z in units of ``z_max``."""
    from . import extract
    from .network import Layout
    lay = Layout.of(out, layout)
    with torch.no_grad():
        e = extract.extract(out.detach(), scaling, threshold, how, scale, layout=lay)
        f_xyz = torch.stack([e["x"], e["y"], e["z_nm"]], 1)
        d, k = match_nearest(batch.xyz, batch.mask, e["frame"], f_xyz, pixel_nm, lateral_nm, axial_nm)
    if not len(d):
        return out.sum() * 0.0
    b, i, j = e["frame"][d], e["row"][d], e["column"][d]
    first = lay.rep if head == "rep" else lay.sig_xyz(0)
    sig = out[b, first:first + 3, i, j]                                   # (n, 3) with gradient
    truth = batch.xyz[b, k]
    err = f_xyz[d] - truth
    err = torch.stack([err[:, 0], err[:, 1], err[:, 2] / scaling.z_max], 1)
    nll = torch.log(sig) + err ** 2 / (2 * sig ** 2)
    return nll.sum() / out.shape[0]
