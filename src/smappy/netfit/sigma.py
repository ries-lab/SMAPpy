"""How honest a fitter's claimed uncertainties are, on a simulation where each error is known (step 4,
an honest sigma; ``docs/decode.md`` §10).

`scores` takes the matched localizations' errors and claims, per axis in nm, and gives

* the error over the claim as a robust width (1.4826 MAD; `training.reference.robust_width`) -- 1
  for an honest claim -- overall and in bins of |z|, the photons, the distance to the nearest other
  molecule in the frame, and the frames a molecule is on;
* reliability: in bins of the claim, the error's root mean square against the claim's;
* coverage: the share of errors within 1 and 2 claimed sigma (0.683 and 0.954 for an honest
  Gaussian claim);
* the mean negative log-likelihood of the truth under the claim, ``log sigma + e^2 / (2 sigma^2) +
  log sqrt(2 pi)``: the one number that rewards a small error and an honest claim together.

`fit_scale` / `apply_scale` are variant 3, the baseline: a factor on the claim per bin of |z|, the
photons and the claim itself, fitted on one set of localizations so that the error over the claim
has a robust width of 1 in each bin, and applied to another.  It reads what a fitter reports (its z,
photons and claim), never the truth, so it can be fitted on a simulation and applied to real data.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..training.reference import robust_width

Z_EDGES = (0.0, 200.0, 400.0, 600.0, 800.0, np.inf)
PHOTON_EDGES = (0.0, 2000.0, 5000.0, 10000.0, np.inf)
NN_EDGES = (0.0, 1.5, 3.0, 5.0, np.inf)              # pixels
CLAIM_BINS = 8                                         # quantiles of the claim, reliability
MIN_PER_BIN = 50
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def _binned(ratio, key, edges):
    out = []
    for lo, hi in itertools.pairwise(edges):
        m = (key >= lo) & (key < hi)
        out.append({"lo": float(lo), "hi": None if np.isinf(hi) else float(hi), "n": int(m.sum()),
                    "width": robust_width(ratio[m]) if m.sum() >= MIN_PER_BIN else None})
    return out


def nearest_other(frame, xy):
    """Each molecule's distance to the nearest other one in its frame (pixels; inf alone)."""
    from scipy.spatial import cKDTree
    nn = np.full(len(frame), np.inf)
    for k in np.unique(frame):
        a = np.flatnonzero(frame == k)
        if len(a) > 1:
            nn[a] = cKDTree(xy[a]).query(xy[a], 2)[0][:, 1]
    return nn


def scores(err_nm, claim_nm, z_nm, photons, nn_px=None, frames_on=None) -> dict:
    """``err_nm``, ``claim_nm`` (n, 3) x, y, z; ``z_nm`` and ``photons`` the truth's, per molecule;
    ``nn_px`` (`nearest_other`) and ``frames_on`` optional.  x and y are pooled as "lat"."""
    err, claim = np.asarray(err_nm, float), np.asarray(claim_nm, float)
    out = {}
    for axis, cols in (("lat", (0, 1)), ("ax", (2,))):
        e = np.concatenate([err[:, c] for c in cols])
        s = np.concatenate([claim[:, c] for c in cols])
        rep = len(cols)
        r = e / s
        z, ph = np.tile(np.abs(z_nm), rep), np.tile(photons, rep)
        a = {"width": robust_width(r), "coverage_1": float(np.mean(np.abs(r) <= 1)),
             "coverage_2": float(np.mean(np.abs(r) <= 2)),
             "nll": float(np.mean(np.log(s) + e ** 2 / (2 * s ** 2) + LOG_SQRT_2PI)),
             "by_abs_z": _binned(r, z, Z_EDGES), "by_photons": _binned(r, ph, PHOTON_EDGES)}
        if nn_px is not None:
            a["by_nn_px"] = _binned(r, np.tile(nn_px, rep), NN_EDGES)
        if frames_on is not None:
            fo = np.tile(frames_on, rep)
            a["by_frames_on"] = _binned(r, fo, tuple(np.unique(fo)) + (np.inf,))
        q = np.quantile(s, np.linspace(0, 1, CLAIM_BINS + 1))
        rel = []
        for lo, hi in itertools.pairwise(q):
            m = (s >= lo) & (s <= hi)
            if m.sum() >= MIN_PER_BIN:
                rel.append({"claim_rms_nm": float(np.sqrt(np.mean(s[m] ** 2))), "error_rms_nm": float(np.sqrt(np.mean(e[m] ** 2))),
                            "error_width_nm": robust_width(e[m]), "n": int(m.sum())})
        a["reliability"] = rel
        out[axis] = a
    return out


def _keys(z_nm, photons, claim, claim_edges):
    zb = np.clip(np.digitize(np.abs(z_nm), Z_EDGES) - 1, 0, len(Z_EDGES) - 2)
    pb = np.clip(np.digitize(photons, PHOTON_EDGES) - 1, 0, len(PHOTON_EDGES) - 2)
    cb = np.clip(np.digitize(claim, claim_edges[1:-1]), 0, len(claim_edges) - 2)
    return zb, pb, cb


def fit_scale(err_nm, claim_nm, z_nm, photons, claim_bins: int = 4) -> dict:
    """Variant 3: per axis ("lat", "ax") the factor on the claim in each bin of the fit's |z|, its
    photons and the claim, that makes the error over the claim a robust width of 1 there (bins with
    fewer than `MIN_PER_BIN` take the axis's overall factor)."""
    err, claim = np.asarray(err_nm, float), np.asarray(claim_nm, float)
    out = {}
    for axis, cols in (("lat", (0, 1)), ("ax", (2,))):
        e = np.concatenate([err[:, c] for c in cols])
        s = np.concatenate([claim[:, c] for c in cols])
        z, ph = np.tile(z_nm, len(cols)), np.tile(photons, len(cols))
        edges = np.quantile(s, np.linspace(0, 1, claim_bins + 1))
        zb, pb, cb = _keys(z, ph, s, edges)
        overall = robust_width(e / s)
        table = np.full((len(Z_EDGES) - 1, len(PHOTON_EDGES) - 1, claim_bins), overall)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                for k in range(claim_bins):
                    m = (zb == i) & (pb == j) & (cb == k)
                    if m.sum() >= MIN_PER_BIN:
                        table[i, j, k] = robust_width(e[m] / s[m])
        out[axis] = {"claim_edges": edges, "factor": table}
    return out


def apply_scale(scale: dict, claim_nm, z_nm, photons):
    """The claims (n, 3) times their bins' factors (`fit_scale`)."""
    claim = np.array(claim_nm, float)
    for axis, cols in (("lat", (0, 1)), ("ax", (2,))):
        t = scale[axis]
        for c in cols:
            zb, pb, cb = _keys(z_nm, photons, claim[:, c], t["claim_edges"])
            claim[:, c] = claim[:, c] * t["factor"][zb, pb, cb]
    return claim
