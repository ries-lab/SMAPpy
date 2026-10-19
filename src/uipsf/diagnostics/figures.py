"""The figures of a calibration: the pupil, its Zernikes, the residual atlas, the beads (plan §7.5).

Every function draws on a matplotlib figure it is given and returns
nothing: SMAPpy's window passes a tab's figure (a `SubFigure` on its *All*
page), a notebook passes `plt.figure()`, and the same pictures come out.
Nothing here imports pyplot, sets a figure's size or its layout engine.

They read a `Calibration` (the fit just made) or a `LearnedPSF` (a model
file) through `zernikes` and `pupil_maps`, so a previous model is drawn the
same way, and `draw_zernikes` draws one against the other: the comparison
a diagnosis is about.  The atlas and the bead table belong to a fit and are
drawn from a `Calibration` only.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.channels import prefix, view
from ..optics.zernike import noll_to_nm

STATES = {"used": "#1f77b4", "held out": "#2ca02c", "rejected": "#d62728", "excluded": "#7f7f7f"}


def _shared(source) -> dict:
    return source.shared() if callable(getattr(source, "shared", None)) else source.shared


def zernikes(source, which: str = "phase", channel: int = 0) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(Noll modes, coefficients, CRLBs) of ``which`` ("phase", rad, or "magnitude") of ``source``, a
    `Calibration` or a `LearnedPSF`; NaN CRLBs where a coefficient was not fitted."""
    pupil = source.model.channels[channel].psf.pupil
    modes = list(getattr(pupil, f"{which}_modes", []))
    name = prefix(channel) + f"pupil.{which}"
    shared = _shared(source)
    if not modes or name not in shared:
        return [], np.zeros(0), np.zeros(0)
    values = torch.as_tensor(shared[name]).detach().cpu().double().numpy()
    std = source.result.std if hasattr(source, "result") else source.crlb
    crlb = (torch.as_tensor(std[name]).detach().cpu().double().numpy() if name in std
            else np.full(len(modes), np.nan))
    return modes, values, crlb


def pupil_maps(source, channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude, phase) of channel ``channel``'s pupil, [P, P], NaN outside the aperture."""
    psf = source.model.channels[channel].psf
    with torch.no_grad():
        p = view({k: torch.as_tensor(v) for k, v in _shared(source).items()}, channel)
        magnitude, phase = psf.pupil({k: v.to(psf.grid.rho.device) for k, v in p.items()})
    inside = psf.grid.aperture.detach().cpu().numpy() > 0
    out = []
    for m in (magnitude, phase):
        m = m.detach().cpu().double().numpy()
        out.append(np.where(inside, m, np.nan))
    return out[0], out[1]


def _label(j: int) -> str:
    n, m = noll_to_nm(j)
    return f"{j}\n({n},{m})"


def draw_pupil(fig, source, channel: int = 0) -> None:
    """The pupil's phase (rad) and magnitude over the aperture."""
    magnitude, phase = pupil_maps(source, channel)
    a, b = fig.subplots(1, 2)
    lim = np.nanmax(np.abs(phase)) or 1.0
    im = a.imshow(phase, cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set_title("phase (rad)")
    fig.colorbar(im, ax=a, shrink=0.8)
    im = b.imshow(magnitude, cmap="viridis")
    b.set_title("magnitude")
    fig.colorbar(im, ax=b, shrink=0.8)
    for ax in (a, b):
        ax.set_xticks([])
        ax.set_yticks([])


def draw_zernikes(fig, source, channel: int = 0, reference=None, reference_label: str = "start") -> None:
    """The phase and, where there are any, the magnitude Zernikes with their CRLBs as error bars;
    ``reference`` (another `Calibration` or `LearnedPSF`) drawn beside them, mode by mode."""
    which = [w for w in ("phase", "magnitude") if zernikes(source, w, channel)[0]]
    axes = np.atleast_1d(fig.subplots(len(which), 1, squeeze=False)[:, 0])
    for ax, w in zip(axes, which, strict=True):
        modes, values, crlb = zernikes(source, w, channel)
        x = np.arange(len(modes))
        width = 0.4 if reference is not None else 0.8
        ax.bar(x, values, width, yerr=crlb, capsize=2, label="this fit")
        if reference is not None:
            rmodes, rvalues, rcrlb = zernikes(reference, w, channel)
            r = {j: (v, e) for j, v, e in zip(rmodes, rvalues, rcrlb, strict=True)}
            rv = np.array([r.get(j, (np.nan, np.nan))[0] for j in modes])
            re = np.array([r.get(j, (np.nan, np.nan))[1] for j in modes])
            ax.bar(x + width, rv, width, yerr=re, capsize=2, color="#999999", label=reference_label)
            ax.legend(fontsize="small")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x + (width / 2 if reference is not None else 0))
        ax.set_xticklabels([_label(j) for j in modes], fontsize="x-small")
        ax.set_ylabel("rad" if w == "phase" else "relative")
        ax.set_title(f"{w}: Noll index (n, m)")


def draw_atlas(fig, atlas, groups=("slice", "|z|", "radius", "field")) -> None:
    """The residual atlas: the mean deviance per group (≈ 1 where the model is right), each panel
    titled with its heterogeneity chi²/dof and in red when it is flagged; the findings as the figure's
    title."""
    shown = [g for g in groups if g in atlas.groups]
    axes = np.atleast_1d(fig.subplots(1, len(shown), squeeze=False)[0])
    for ax, name in zip(axes, shown, strict=True):
        g = atlas.groups[name]
        ax.plot(np.asarray(g.keys, float), g.deviance, "o-", ms=3)
        ax.axhline(1, color="k", lw=0.5)
        threshold = atlas.checks.get(name, (None, None))[1]
        flagged = name in atlas.flagged()
        ax.set_title(f"{name}: χ²/dof {g.chi2:.2g}" + ("" if threshold is None else f" (≤ {threshold:.2g})"),
                     color="#d62728" if flagged else "k", fontsize="small")
        ax.set_xlabel(name)
    axes[0].set_ylabel("deviance per pixel")
    text = "; ".join(f.suggests for f in atlas.findings) or f"nothing flagged (mean deviance {atlas.mean_deviance:.3g})"
    fig.suptitle(text, fontsize="small")


def draw_beads(fig, cal, selected=()) -> None:
    """Every bead where it is on the chip, by its state in the bead table; the ``selected`` beads ringed.
    Each point carries its bead (``pair_ids``, with the pick's index), so a window can select it."""
    ax = fig.subplots()
    selected = set(selected)
    for state, colour in STATES.items():
        rows = [r for r in cal.beads.rows if r.state == state and r.field_xy is not None]
        if rows:
            xy = np.array([r.field_xy for r in rows], float)
            points = ax.scatter(xy[:, 0], xy[:, 1], s=20, color=colour, label=f"{state} ({len(rows)})", picker=5)
            points.pair_ids = np.array([r.bead for r in rows])
            for r, (x, y) in zip(rows, xy, strict=True):
                ax.annotate(str(r.bead), (x, y), fontsize="xx-small", xytext=(2, 2), textcoords="offset points")
    ring = [r.field_xy for r in cal.beads.rows if r.bead in selected and r.field_xy is not None]
    if ring:
        ring = np.array(ring, float)
        ax.scatter(ring[:, 0], ring[:, 1], s=140, facecolors="none", edgecolors="k", linewidths=1.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(fontsize="small")


def figures(cal, reference=None) -> dict:
    """{title: draw(fig)} for every figure of ``cal``, each a closure that draws when asked, so tabs
    draw lazily; ``reference`` is the model it started from, if any."""
    out = {}
    for c in range(cal.model.n_channels):
        tag = "" if cal.model.n_channels == 1 else f" {c + 1}"
        out[f"Pupil{tag}"] = lambda fig, c=c: draw_pupil(fig, cal, c)
        out[f"Zernikes{tag}"] = lambda fig, c=c: draw_zernikes(fig, cal, c, reference)
    if cal.atlas is not None:
        out["Residual atlas"] = lambda fig: draw_atlas(fig, cal.atlas)
    if any(r.field_xy is not None for r in cal.beads.rows):
        out["Beads"] = lambda fig: draw_beads(fig, cal)
    return out
