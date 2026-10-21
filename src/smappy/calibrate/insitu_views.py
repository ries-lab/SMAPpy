"""What the PSF calibration window shows of an in situ calibration: the z bins, the rounds, the model per height.

Qt-free drawing on a matplotlib figure, as `plots` is for the bead calibration, so that a test or a
notebook draws the same pages (`psf_methods.InSitu.pages`).

* **z bins** -- what the molecules were chosen by: the first round's z proxy (a Gaussian's elongation or
  size), then, after calibrating, the last round's heights from the model; each bin's units as used,
  held out, not drawn and rejected.  A bin with nothing used is a height the model learnt nothing at.
* **Rounds and edge** -- the focus each round, with its error and the edge's prior; the rms change that
  stops the rounds; the lowest localised heights with the coverslip edge fitted to them; the twin test.
* **Model vs data** (`model_vs_data`, `draw_model_vs_data`) -- the used molecules in bins of their height:
  the mean of their regions beside the mean of the model's expected images for the same molecules (each
  at its own sub-pixel position, so nothing is aligned or interpolated), and the difference, on one
  scale per bin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: the states' colours, as the stack browser rings them
COLOURS = {"used": "#1f77b4", "held out": "#2ca02c", "not drawn": "#bbbbbb", "rejected": "#d62728"}


def _kind(state: str) -> str:
    return "rejected" if state.startswith("rejected") else state


def draw_z_bins(figure, rois, run=None) -> str:
    """The units by the quantity they were binned by, stacked by state; after a run the last round's."""
    if run is not None and len(run.rounds) > 1 and run.rounds[-1].heights is not None:
        last = run.rounds[-1]
        pool = np.flatnonzero(np.array([r is None for r in rois.units.reason]))
        value = np.full(len(rois.state), np.nan)
        value[pool] = last.heights - (last.edge.shift_nm if last.edge is not None and last.focus_prior else 0.0)
        state, label = last.state, "height above the coverslip (nm), round %d" % last.index
    else:
        from .insitu import z_proxy
        value = z_proxy(rois.units.gauss, "astigmatism")
        state, label = rois.state, "z proxy: σx² − σy² (px²), the Gaussian's"
    ax = figure.subplots()
    finite = np.isfinite(value)
    if not finite.any():
        ax.axis("off")
        return "nothing to bin"
    edges = np.linspace(*np.nanquantile(value[finite], [0.005, 0.995]), 31)
    kinds = ("used", "held out", "not drawn", "rejected")
    stacks = [value[finite & np.array([_kind(s) == k for s in state])] for k in kinds]
    ax.hist(stacks, bins=edges, stacked=True, color=[COLOURS[k] for k in kinds], label=list(kinds))
    ax.set_xlabel(label)
    ax.set_ylabel("units")
    ax.legend(fontsize="small")
    n = {k: int(np.sum([_kind(s) == k for s in state])) for k in kinds}
    return ", ".join(f"{v} {k}" for k, v in n.items())


def draw_rounds(figure, run) -> str:
    """The focus per round, the rms change, the edge of the last round's heights, and the twin test."""
    axes = figure.subplots(1, 3)
    rounds = run.rounds
    idx = [r.index for r in rounds]
    focus = [float(r.calibration.calibration.shared()["geometry.focus"]) for r in rounds]
    sd = [float(r.calibration.calibration.result.std["geometry.focus"]) for r in rounds]
    ax = axes[0]
    ax.errorbar(idx, focus, yerr=sd, fmt="o-", capsize=3, label="learnt")
    priors = [(r.index, r.focus_prior) for r in rounds if r.focus_prior is not None]
    if priors:
        ax.errorbar([i for i, _ in priors], [p[0] for _, p in priors], yerr=[p[1] for _, p in priors],
                    fmt="s", mfc="none", capsize=3, label="the edge's prior")
    ax.set_xlabel("round")
    ax.set_ylabel("focus (nm)")
    ax.set_xticks(idx)
    ax.legend(fontsize="small")
    ax = axes[1]
    moved = [(r.index, r.moved) for r in rounds if np.isfinite(r.moved)]
    if moved:
        ax.plot([i for i, _ in moved], [m for _, m in moved], "o-")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("rms change (standard errors)")
    ax.set_xticks(idx)
    ax = axes[2]
    with_edge = [r for r in rounds if r.heights is not None]
    if with_edge:
        r = with_edge[-1]
        h = r.heights[np.isfinite(r.heights)]
        ax.hist(h, bins=np.linspace(np.quantile(h, 0.0), np.quantile(h, 0.5), 40), color="#888888")
        if r.edge is not None:
            ax.axvline(r.edge.shift_nm, color="#d62728",
                       label=f"edge {r.edge.shift_nm:.0f} ± {r.edge.sd_nm:.0f} nm" +
                             (f", {100 * r.edge.spike:.0f} % on it" if r.edge.spike else ""))
            ax.legend(fontsize="small")
        ax.set_xlabel(f"localised height, round {r.index} (nm)")
        ax.set_ylabel("units")
    else:
        ax.axis("off")
    twin = run.final.twin
    if twin is None:
        return "no twin test (stopped)"
    return f"twin test: {twin.verdict} ({twin.delta_nats:+.0f} nats for the learnt sign)"


@dataclass
class ModelVsData:
    edges: np.ndarray            # [B+1] height bin edges, nm
    data: np.ndarray             # [B, Y, X] mean photons of the used units in each bin, masked pixels out
    model: np.ndarray            # [B, Y, X] the model's mean expected photons for the same units
    count: np.ndarray            # [B]


def model_vs_data(result, bins: int = 6) -> ModelVsData:
    """The used units of the last round in ``bins`` bins of their learnt height (equal counts): their mean
    region and the model's mean expected image, masked pixels left out of both."""
    import torch
    from uipsf.device import on_model_device
    from uipsf.solve import Problem
    cal, rois = result.cal, result.rois
    used = result.run.rounds[-1].used[np.asarray(cal.beads.used, int)]
    batch, noise = rois.batch(ids=used)
    batch = on_model_device(batch.to(dtype=cal.model.psf.grid.rho.dtype), cal.model)
    with torch.no_grad():
        mu = Problem(cal.model, noise, batch).expected(cal.result.params)[:, 0, 0].double().numpy()
    data = batch.data[:, 0, 0].cpu().double().numpy()
    mask = batch.mask[:, 0, 0].cpu().numpy() if batch.mask is not None else np.ones_like(data, bool)
    h = cal.result.params.physical()["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    edges = np.unique(np.quantile(h, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(h, edges) - 1, 0, len(edges) - 2)
    D, M, n = [], [], []
    for b in range(len(edges) - 1):
        k = which == b
        w = mask[k].sum(0).clip(min=1)
        D.append((data[k] * mask[k]).sum(0) / w)
        M.append((mu[k] * mask[k]).sum(0) / w)
        n.append(int(k.sum()))
    return ModelVsData(edges, np.array(D), np.array(M), np.array(n))


def draw_model_vs_data(figure, views: ModelVsData) -> str:
    """Data, model and difference per height bin, a row each, one scale per row."""
    B = len(views.count)
    axes = figure.subplots(B, 3, squeeze=False)
    for b in range(B):
        d, m = views.data[b], views.model[b]
        top = max(float(d.max()), float(m.max()))
        r = d - m
        lim = max(float(np.abs(r).max()), 1e-9)
        for a, (img, kw) in enumerate(((d, dict(vmin=0, vmax=top, cmap="gray")), (m, dict(vmin=0, vmax=top, cmap="gray")),
                                       (r, dict(vmin=-lim, vmax=lim, cmap="RdBu_r")))):
            ax = axes[b, a]
            ax.imshow(img, **kw)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[b, 0].set_ylabel(f"{views.edges[b]:.0f}–{views.edges[b + 1]:.0f} nm\n{views.count[b]} units",
                              fontsize="small")
    for a, title in enumerate(("data (mean)", "model (mean)", "data − model")):
        axes[0, a].set_title(title, fontsize="small")
    rel = [float(np.abs(views.data[b] - views.model[b]).max() / max(views.data[b].max(), 1e-9)) for b in range(B)]
    return f"largest difference {100 * max(rel):.1f} % of a bin's peak"
