"""The residual atlas: where the model and the data disagree, and what that points to (plan §8).

A fit's deviance per pixel is ≈ 1 where the model is right.  The atlas
groups it, and the signed residuals, in the ways the missing physics of
review §B shows up, and gives each grouping a statistic that is ≈ 1 on
matched data, so "flagged" means "beyond the noise", not a tuned number:

- **groups** (slice, |z|, radius from the emitter, bead, field radius, and
  with several channels the channel; the radius is each channel's own): the
  mean deviance per group, and how much the groups differ beyond what
  their pixel counts allow, chi²/dof with the per-pixel deviance variance
  taken from the data themselves.  A whole stack that fits worse in
  defocus flags |z| (a piezo scale, a wrong pupil); a bad bead flags bead.
- **photons per slice**, from each bead's and slice's photon residual
  r = Σ(d − μ)/√Σvar, ~ N(0, 1) on matched data:
  - *shared*: E·mean_k(m_k²), with m_k the mean of r over beads in slice
    k, divided by the per-observation scatter below (when that exceeds 1):
    a pattern common to all beads in a frame, an unstable laser:
    `intensity.PerSliceFactor`.  Without the division, speckle on twelve
    beads raised it to 5 (tests/uipsf_tests/test_atlas.py);
  - *per observation*: the variance of r − m_k: brightness that changes
    per bead and slice beyond the noise, speckle:
    `intensity.PerObservation`.
- **background ramp**: a plane fitted to each bead's mean signed residual
  over its ROI, the slopes' squared z-scores per bead averaged:
  `background.Planar`.

A statistic is flagged when it exceeds 1 by four of its standard errors
under the null (√(2/dof)), eight for the generic groups, whose pixels
share fitted parameters (measured on matched fits: up to 3.8 where four
would allow 3.3); the thresholds are reported with it.  The fitted
parameters take some freedom from the residuals, so matched fits sit a
little below 1, not above.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..device import on_model_device


@dataclass
class Group:
    name: str
    keys: np.ndarray            # the group's label (slice index, z in nm, radius in pixels, bead, …)
    deviance: np.ndarray        # mean deviance per pixel in each group
    signed: np.ndarray          # mean Pearson residual (d − μ)/√var in each group: the sign of the pattern
    pixels: np.ndarray
    chi2: float                 # heterogeneity, chi²/dof; ≈ 1 when the groups differ only by noise


@dataclass
class Finding:
    check: str
    value: float
    threshold: float
    suggests: str


@dataclass
class Atlas:
    mean_deviance: float
    groups: dict[str, Group]
    checks: dict[str, tuple[float, float]]          # name -> (statistic, threshold)
    findings: list[Finding] = field(default_factory=list)
    photon_residual: np.ndarray | None = None       # r[e, k]

    def flagged(self) -> list[str]:
        return [f.check for f in self.findings]


SUGGESTS = {
    "slice": "something changes from slice to slice: see the photon checks, drift, the stage",
    "|z|": "the fit worsens with defocus: a piezo z scale, the pupil's order, index mismatch",
    "radius": "the spot's shape or wings: the blur, the bead size, the pupil magnitude",
    "bead": "some beads fit worse than others: bead rejection, field dependence",
    "field": "the fit changes over the field: field dependence",
    "channel": "one channel fits worse than the other: its pupil's order, its optics, the pairing",
    "photons per slice, shared": "a laser that fluctuates: intensity.PerSliceFactor",
    "photons per observation": "brightness that changes per bead and slice (speckle): intensity.PerObservation",
    "background ramp": "a background that is not flat: background.Planar",
}


def _threshold(dof, se=4.0):
    return 1 + se * np.sqrt(2 / max(dof, 1))


def _group(name, labels, D, R, keep, var_D, keys=None):
    labels = labels[keep]
    uniq = np.unique(labels) if keys is None else keys
    dev, sig, n = [], [], []
    for u in uniq:
        sel = labels == u
        n.append(sel.sum())
        dev.append(D[keep][sel].mean() if sel.any() else np.nan)
        sig.append(R[keep][sel].mean() if sel.any() else np.nan)
    dev, sig, n = np.array(dev), np.array(sig), np.array(n)
    ok = n > 0
    overall = (dev[ok] * n[ok]).sum() / n[ok].sum()
    chi2 = float(((dev[ok] - overall) ** 2 * n[ok] / var_D).sum() / max(ok.sum() - 1, 1))
    return Group(name, np.asarray(uniq), dev, sig, n, chi2)


def residual_atlas(model, params, batch, noise, z_bin_nm: float = 100.0) -> Atlas:
    """The atlas of a fit: ``params`` for every emitter of ``batch`` (shared ones fitted or fixed)."""
    batch = on_model_device(batch, model)
    with torch.no_grad():
        p = params.physical()
        # On the CPU in float64; copy, then cast (MPS has no float64, and one-call casts from it are wrong).
        mu = model(p, batch).cpu().double()
        main = model.geometry(p, batch)
        xyz = torch.stack([model.positions(p, batch, main, c) for c in range(model.n_channels)],
                          1).cpu().double()                          # [E, C, K, 3], each channel's own
    d = batch.data.cpu().double()
    s = batch.s.cpu().double() if isinstance(batch.s, torch.Tensor) else float(batch.s)
    mask = torch.ones_like(d, dtype=torch.bool) if batch.mask is None else batch.mask.cpu().expand_as(d)
    D = noise.deviance(mu, d, s, mask).numpy()
    var = noise.variance(mu, s)
    R = ((d - mu) / var.sqrt()).numpy()
    keep = mask.numpy()
    E, C, K, Y, X = d.shape
    var_D = float(D[keep].var())
    # labels per pixel
    k_lab = np.broadcast_to(np.arange(K)[None, None, :, None, None], D.shape)
    e_lab = np.broadcast_to(np.arange(E)[:, None, None, None, None], D.shape)
    z = np.abs(xyz[:, 0, :, 2].numpy())                                # [E, K], the emitter's z (main focus)
    z_lab = np.broadcast_to((np.round(z / z_bin_nm) * z_bin_nm)[:, None, :, None, None], D.shape)
    r = np.empty(D.shape)
    for c, ch in enumerate(model.channels):
        px, py = ch.pixel_size_nm
        xs = (np.arange(X) - (X - 1) / 2) * px
        ys = (np.arange(Y) - (Y - 1) / 2) * py
        dx = xs[None, None, :] - xyz[:, c, :, 0].numpy()[:, :, None]   # [E, K, X]
        dy = ys[None, None, :] - xyz[:, c, :, 1].numpy()[:, :, None]
        r[:, c] = np.hypot(dx[:, :, None, :] / px, dy[:, :, :, None] / py)   # [E, K, Y, X], pixels
    r_lab = np.round(r)
    groups = {"slice": _group("slice", k_lab, D, R, keep, var_D),
              "|z|": _group("|z|", z_lab, D, R, keep, var_D),
              "radius": _group("radius", r_lab, D, R, keep, var_D),
              "bead": _group("bead", e_lab, D, R, keep, var_D)}
    if C > 1:
        c_lab = np.broadcast_to(np.arange(C)[None, :, None, None, None], D.shape)
        groups["channel"] = _group("channel", c_lab, D, R, keep, var_D)
    if batch.field_xy is not None and E >= 8:
        f = batch.field_xy.cpu().double().numpy()
        fr = np.hypot(*(f - f.mean(0)).T)
        edges = np.quantile(fr, [0.25, 0.5, 0.75])
        f_lab = np.broadcast_to(np.searchsorted(edges, fr)[:, None, None, None, None], D.shape)
        groups["field"] = _group("field", f_lab, D, R, keep, var_D)
    # Pixels in one group share fitted parameters, so the groups' chi² is overdispersed: on matched data
    # it reached 3.8 for |z| with 10 groups (4 standard errors: 3.3).  The generic groups get 8.
    checks = {name: (g.chi2, _threshold(len(g.keys) - 1, se=8.0)) for name, g in groups.items()}

    # photons per bead and slice
    resid = ((d - mu) * mask).sum((1, 3, 4)).numpy()                  # [E, K]
    rvar = (var * mask).sum((1, 3, 4)).numpy()
    rr = resid / np.sqrt(rvar)
    m = rr.mean(0)
    if E >= 3:
        within = ((rr - m) ** 2).sum() / ((E - 1) * K)
        # Against the scatter within slices, not against 1: speckle raises both, a laser only this.
        checks["photons per slice, shared"] = (float((E * m ** 2).mean() / max(within, 1.0)), _threshold(K))
        checks["photons per observation"] = (float(within), _threshold((E - 1) * K))

    # background ramp: plane through each bead's (and channel's) mean Pearson residual map
    Rm = np.where(keep, R, np.nan)
    maps = np.nanmean(Rm, axis=2).reshape(E * C, Y, X)                  # [E·C, Y, X]
    counts = keep.sum(2).reshape(E * C, Y, X)                           # pixels averaged per map pixel
    gy, gx = np.mgrid[:Y, :X]
    A = np.c_[np.ones(Y * X), (gx.ravel() - (X - 1) / 2), (gy.ravel() - (Y - 1) / 2)]
    z2 = []
    for e in range(E * C):
        w = counts[e].ravel().astype(float)
        ok = w > 0
        Aw = A[ok] * np.sqrt(w[ok])[:, None]
        bw = maps[e].ravel()[ok] * np.sqrt(w[ok])
        coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        cov = np.linalg.inv(Aw.T @ Aw)
        z2 += [coef[1] ** 2 / cov[1, 1], coef[2] ** 2 / cov[2, 2]]
    checks["background ramp"] = (float(np.mean(z2)), _threshold(len(z2)))

    findings = [Finding(n, v, t, SUGGESTS[n]) for n, (v, t) in checks.items() if v > t]
    return Atlas(float(D[keep].mean()), groups, checks, findings, rr)
