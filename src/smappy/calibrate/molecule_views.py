"""What the PSF calibration window shows of a calibration from molecules on the coverslip in z sweeps.

Qt-free drawing on a matplotlib figure, as `insitu_views` is for in situ, so a test or a notebook draws the
same pages (`psf_methods.MoleculeStack.pages`):

* **Selection** (`draw_selection`) -- the molecules by their defocus, stacked by state: used, not drawn,
  and each rule that rejected them.  A bin with nothing used is a defocus the model learnt nothing at.
* **z check** (`draw_z_check`) -- every used molecule's z found again on the model with its z free,
  against its defocus: the bend (the median per 100 nm), the CRLB about it, and the molecules off the plane.
* **Foci and plane** (`draw_foci`) -- each sweep's focus, the focal plane's slope, the molecules' height.
* **Model vs data** (`model_vs_data`) -- the used molecules in bins of defocus, the mean of their regions
  beside the model's mean expected image for the same molecules, in situ's drawing
  (`insitu_views.draw_model_vs_data`).
"""
from __future__ import annotations

import numpy as np

from .insitu_views import COLOURS, ModelVsData

#: the rejection rules' colours, after the states'
_RULES = ("shape", "off centre", "neighbour", "brightness", "fit failed")
_RULE_COLOURS = ("#d62728", "#ff9896", "#9467bd", "#e377c2", "#7f7f7f")


def draw_selection(figure, rois, result=None) -> str:
    """The molecules by defocus, stacked by state; after a run the used ones are the last round's."""
    state = list(rois.state)
    if result is not None:
        used = set(np.asarray(result.used()).tolist())
        state = ["used" if i in used else ("rejected: in learning" if s == "used" else s)
                 for i, s in enumerate(state)]
    d = np.asarray(rois.defocus_nm, float)
    ax = figure.subplots()
    edges = np.arange(np.floor(d.min() / 50) * 50 - 25, d.max() + 75, 50.0) if len(d) else np.arange(2)
    kinds = [("used", COLOURS["used"]), ("not drawn", COLOURS["not drawn"])]
    kinds += [(f"rejected: {r}", c) for r, c in zip(_RULES, _RULE_COLOURS, strict=True)]
    kinds += [("rejected: in learning", "#8c564b")]
    stacks, labels, colours = [], [], []
    for kind, colour in kinds:
        k = np.array([s == kind for s in state], bool)
        if k.any():
            stacks.append(d[k])
            labels.append(kind.removeprefix("rejected: ") if kind.startswith("rejected") else kind)
            colours.append(colour)
    if stacks:
        ax.hist(stacks, bins=edges, stacked=True, color=colours, label=labels)
        ax.legend(fontsize="small", ncol=2)
    ax.set_xlabel("defocus by the stage (nm)")
    ax.set_ylabel("molecules")
    counts = {lab: len(v) for lab, v in zip(labels, stacks, strict=True)}
    return ", ".join(f"{v} {k}" for k, v in counts.items())


def draw_z_check(figure, result) -> str:
    """Each used molecule's z offset from the plane against its defocus, the bend and its CRLB band."""
    check = result.check
    ax = figure.subplots()
    if check is None:
        ax.axis("off")
        return "no z check (stopped)"
    d, dz, crlb, bad = check.defocus_nm, check.dz_nm, check.crlb_nm, check.bad
    ax.plot(d[~bad], dz[~bad], ".", ms=2, color="#888888", alpha=0.4, label="molecules")
    if bad.any():
        ax.plot(d[bad], np.clip(dz[bad], -300, 300), "x", ms=3, color="#d62728", label="off the plane")
    ok = np.isfinite(check.bend_nm)
    ax.plot(check.grid_nm[ok], check.bend_nm[ok], "o-", color="C0", label="bend (median)")
    band = np.array([np.median(crlb[~bad & (np.abs(d - g) <= 50)]) if np.any(~bad & (np.abs(d - g) <= 50))
                     else np.nan for g in check.grid_nm])
    fine = ok & np.isfinite(band)
    ax.fill_between(check.grid_nm[fine], check.bend_nm[fine] - band[fine], check.bend_nm[fine] + band[fine],
                    color="C0", alpha=0.15, label="± CRLB")
    ax.axhline(0, color="black", lw=0.6)
    top = 3 * float(np.nanmedian(crlb)) if len(crlb) else 100.0
    ax.set_ylim(-top, top)
    ax.set_xlabel("defocus by the stage (nm)")
    ax.set_ylabel("z found less z by the stage (nm)")
    ax.legend(fontsize="small")
    return (f"bend {check.bend_rms(600):.1f} nm rms over ±600 nm, {check.bend_rms(200):.1f} over ±200; scatter "
            f"{check.scatter:.2f} × CRLB (median {np.median(crlb):.0f} nm); {int(bad.sum())} off the plane")


def draw_foci(figure, result) -> str:
    """Each sweep's focus with its error, the plane's slope and the height."""
    shared = result.cal.shared()
    std = result.cal.result.std
    f = shared["geometry.focus"].detach().cpu().double().numpy()
    e = std.get("geometry.focus")
    e = np.zeros_like(f) if e is None else e.detach().cpu().double().numpy().reshape(f.shape)
    files = np.array([int(result.rois.units.source[np.flatnonzero(result.rois.sweep == s)[0]])
                      if np.any(result.rois.sweep == s) else -1 for s in range(len(f))])
    ax = figure.subplots()
    for k in np.unique(files):
        m = files == k
        ax.errorbar(np.flatnonzero(m), f[m], yerr=e[m], fmt="o", ms=3, capsize=2,
                    label=None if k < 0 else f"file {k + 1}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("sweep")
    ax.set_ylabel("focus less its file's width-curve focus (nm)")
    if len(np.unique(files)) <= 8:
        ax.legend(fontsize="small")
    plane = shared["geometry.plane"].detach().cpu().double().numpy()
    tilt = np.degrees(np.arctan(np.hypot(*plane) / 1000))
    h = float(shared["geometry.height"])
    return (f"focal plane {plane[0]:.2f}, {plane[1]:.2f} nm/µm in x, y ({tilt:.2f}°); height {h:.1f} nm; "
            f"foci spread {np.std(f):.0f} nm rms")


def model_vs_data(result, bins: int = 6) -> ModelVsData:
    """The used molecules of the last round in ``bins`` bins of their defocus (equal counts): their mean
    region and the model's mean expected image, masked pixels left out of both."""
    import torch
    from uipsf.device import on_model_device
    from uipsf.solve import Problem
    cal, rois = result.cal, result.rois
    used = result.used()
    batch, noise = rois.batch(ids=used)
    batch = on_model_device(batch.to(dtype=cal.model.psf.grid.rho.dtype), cal.model)
    with torch.no_grad():         # the stage-2 fit holds the used molecules only, in their order
        mu = Problem(cal.model, noise, batch, 64).expected(cal.result.params)[:, 0, 0].double().numpy()
    data = batch.data[:, 0, 0].cpu().double().numpy()
    mask = batch.mask[:, 0, 0].cpu().numpy() if batch.mask is not None else np.ones_like(data, bool)
    d = rois.defocus_nm[used]
    edges = np.unique(np.quantile(d, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    D, M, n = [], [], []
    for b in range(len(edges) - 1):
        k = which == b
        w = mask[k].sum(0).clip(min=1)
        D.append((data[k] * mask[k]).sum(0) / w)
        M.append((mu[k] * mask[k]).sum(0) / w)
        n.append(int(k.sum()))
    return ModelVsData(edges, np.array(D), np.array(M), np.array(n))
