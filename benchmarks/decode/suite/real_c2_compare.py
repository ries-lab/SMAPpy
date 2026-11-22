"""The real movies' fits compared without a truth (plex-plan §16, real data).

* **Agreement** with DECODE-Plex (the paper's run): shares paired within 150 nm, offset, spread.
* **Precision across frames**: a molecule on for several frames is found again; each localization's
  nearest ``GAP`` frames later within ``R_NM`` laterally (3: a 3-frame network's outputs share no
  input frame; 1 understates its error three-fold, the outputs' errors correlated), the differences' robust width / sqrt(2) per axis --
  the precision of one localization (NeNA's idea, on frame pairs), against the reported sigma of the
  same localizations (their RMS / sqrt(1) -- the reported sigma of a pair's mean).  In crowding a
  neighbour can be taken for the molecule; the robust width keeps to the core.
* **Pixelation**: KS of the positions' sub-pixel parts on each fitter's grid.
* **Shares** (colour): the share in channel 1, its histogram's modes.

    GAP=3 python real_c2_compare.py
"""
import glob
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import real_c2 as R
import sets_c2 as S

from smappy.training.reference import ks_uniform, robust_width

R_NM = 100.0
PROB = 0.6
GAP = int(__import__('os').environ.get('GAP', '1'))
PX = np.asarray(S.PIXEL_NM)


def next_frame(L, gap=1):
    """(m, 3) differences (nm) between each localization and its nearest in the next frame within
    ``R_NM``; (m, 3) the pair's reported sigmas combined, sqrt(s1^2 + s2^2) / sqrt(2)."""
    d, s = [], []
    frames = np.unique(L["frame"])
    for f in frames:
        i, j = np.flatnonzero(L["frame"] == f), np.flatnonzero(L["frame"] == f + gap)
        if not len(i) or not len(j):
            continue
        dist, k = cKDTree(np.c_[L["x_px"][j], L["y_px"][j]] * PX).query(np.c_[L["x_px"][i], L["y_px"][i]] * PX)
        ok = dist < R_NM
        a, b = i[ok], j[k[ok]]
        d.append(np.c_[(L["x_px"][a] - L["x_px"][b]) * PX[0], (L["y_px"][a] - L["y_px"][b]) * PX[1], L["z_nm"][a] - L["z_nm"][b]])
        s.append(np.sqrt((np.c_[L["x_err_nm"][a], L["y_err_nm"][a], L["z_err_nm"][a]] ** 2
                          + np.c_[L["x_err_nm"][b], L["y_err_nm"][b], L["z_err_nm"][b]] ** 2) / 2))
    return np.concatenate(d), np.concatenate(s)


def report(L, grid=("x_px", "y_px"), gap=1):
    d, s = next_frame(L, gap)
    n = len(np.unique(L["frame"]))
    prec = [robust_width(d[:, a]) / np.sqrt(2) for a in range(3)]
    sig = [float(np.median(s[:, a])) for a in range(3)]
    out = {"per_frame": len(L["frame"]) / n, "pairs_share": len(d) / len(L["frame"]),
           "prec_nm": np.round(prec, 1).tolist(), "sigma_nm": np.round(sig, 1).tolist(),
           "honesty": np.round(np.array(prec) / np.array(sig), 2).tolist(),
           "pix": round(max(ks_uniform(L[grid[0]]), ks_uniform(L[grid[1]])), 3),
           "photons": float(np.median(L["photons"]))}
    if "photons_c" in L:
        share = L["photons_c"][:, 1] / np.maximum(L["photons_c"].sum(1), 1e-9)
        h, _ = np.histogram(share, 50, (0, 1))
        out["share_hist"] = h.tolist()
        out["share_pct"] = np.percentile(share, [5, 25, 50, 75, 95]).round(3).tolist()
    if "goodness" in L:
        out["goodness_pct"] = np.nanpercentile(L["goodness"], [50, 90]).round(1).tolist()
    return out


def load(path):
    L = dict(np.load(path))
    keep = L["prob"] > PROB
    return {k: v[keep] if np.ndim(v) and len(v) == len(keep) else v for k, v in L.items()}


if __name__ == "__main__":
    out = {}
    for movie in ("fig2a", "fig2c"):
        P = R.plex(movie, 200)
        P = {k: v[P["prob"] > PROB] for k, v in P.items()}
        out[f"{movie} plex"] = report(P, gap=GAP)
        for f in sorted(glob.glob(str(R.OUT / f"{movie}_L_*_+*.npz"))):
            if not f.endswith(("_+5.8.npz", "_+0.0.npz")):
                continue
            L = load(f)
            if len(np.unique(L["frame"])) < 150:
                continue
            r = report(L, ("x_grid_px", "y_grid_px"), gap=GAP)
            r.update({k: v for k, v in R.summary(movie, dict(np.load(f)), R.plex(movie, 200), PROB).items()
                      if k in ("matched_of_ours", "matched_of_plex", "offset_nm", "spread_nm")})
            out[f"{movie} {pathlib.Path(f).stem}"] = r
    for k, v in out.items():
        print(k, json.dumps({a: b for a, b in v.items() if a != "share_hist"}))
    (R.OUT / "compare.json").write_text(json.dumps(out))
