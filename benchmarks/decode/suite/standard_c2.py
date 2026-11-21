"""The C = 2 suite's standard report (plex-plan §16, Jonas 2026-11-21): a few numbers per fitter that
say how often it fails and how good the rest is, instead of every set's every column.

    python standard_c2.py CONTEXT RUN [RUN ...]    # biplane_2c or colour_2a; runs as `run_c2` keeps them,
                                                   # seeds (_s0, _s1 ..) pooled

**The sets**: 10 000 photons and 1 000 photons (isolated), density 15 with molecules on for 1 frame and
for 3 (`sets_c2`'s blinking sets).

**The numbers** -- precision over |z| < ``Z_NM`` (500 nm: normalised by the bound, the order of the
fitters is the in-focus one, and 300 -- 500 nm is not forgotten), on the density sets from
``PHOTONS_MIN`` photons in the frame (a molecule on for part of one holds little):

* **Jaccard**, every set (`run_c2.score`).
* **Outliers**: the share of matched localizations whose error is beyond ``OUTLIER`` times their own
  single-emitter bound, for the frames the fitter reads, on any axis.  How often it fails.
* **Inliers' RMSE / CRLB**, lateral and axial: the rest, against the multi-emitter bound for the frames
  read where it is drawn (the density sets), else the single-emitter one.  RMSE, not a robust width:
  a bias counts.  How good the rest is.
* **Pixelation** (1 000 photons): KS of the inliers' sub-pixel positions on the fitter's grid less
  the truth's own at the same molecules -- 0 is counting noise.  Outliers pixelate no differently.
* **Right dye** (colour; 1 000 photons and density 15): inliers whose share in channel 1 is nearest
  their dye's.
* **Sigma honesty** (density 15): the robust width of the inliers' errors over their reported sigma,
  lateral and axial; 1 is honest, above it overconfident.

Cost -- training time on one idle GPU and ms a frame -- is a model's, not a set's: the study's
`timing.py`.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import report as R
import run_c2 as RC
import sets_c2 as S
from score_sets import match

from smappy.training.reference import ks_uniform, robust_width

Z_NM = 500.0
PHOTONS_MIN = 2000.0
OUTLIER = 5.0
SETS = {"10k": "photons_10000", "1k": "photons_1000", "d15": "density_15", "d15 on 3": "l3_density_15"}


def _bounds(name, frames):
    """The truth's single- and multi-emitter bounds (x, y, z nm, each (n,)) for a fit reading ``frames``."""
    folder = S.SETS_DIR / name
    t = np.load(folder / "truth.npz")
    single = ({a: np.asarray(v) for a, v in np.load(folder / "crlb_w3.npz").items()} if frames == 3
              else {a: np.asarray(t[f"crlb_{a}"]) for a in ("x_nm", "y_nm", "z_nm")})
    jf = folder / ("crlb_joint_w3.npz" if frames == 3 else "crlb_joint_w1.npz")
    multi = {a: v for a, v in np.load(jf).items() if a != "nn_nm"} if jf.exists() else None
    return single, multi


def measure(label, short, context):
    """The standard numbers of one run on one set (``short`` a key of `SETS`) of ``context``."""
    name_set = SETS[short]
    out = {}
    name = f"{context}_blink_{name_set}"
    if not (RC.FITS / label / f"{name}.npz").exists():
        return None
    L0 = dict(np.load(RC.FITS / label / f"{name}.npz"))
    keep = RC.KEEP["plex" if label == "plex" else "netfit"]
    out["jaccard"] = RC.score(name, L0, keep)["jaccard"]
    info = json.loads((RC.FITS / label / f"{name}.json").read_text())
    frames = int(info.get("window") or RC.FRAMES_READ.get(label, {}).get(context, 3))
    t = dict(np.load(S.SETS_DIR / name / "truth.npz"))
    n = len(t["frame"])
    single, multi = _bounds(name, frames)
    t.update({f"s_{a}": v for a, v in single.items()})
    if multi:
        t.update({f"m_{a}": v for a, v in multi.items()})
    t = {k: v for k, v in t.items() if np.ndim(v) == 1 and len(v) == n}
    cal = S.calibration(context)
    shape = tuple(S._layout(cal)[0])
    mask = RC.both_channels(context, shape, bool(cal.parameters["smap"]["EMon"][0]))
    t = {k: v[mask(t["x_px"], t["y_px"])] for k, v in t.items()}
    ok = R.kept(L0, keep, float(max(abs(v) for v in S.CONTEXTS[context]["z_nm"])))
    L = {k: np.asarray(v)[ok] for k, v in L0.items() if np.ndim(v) >= 1 and len(v) == len(ok)}
    L = {k: v[mask(L["x_nm"] / S.PIXEL_NM[0], L["y_nm"] / S.PIXEL_NM[1])] for k, v in L.items()}
    f = {"frame": L["frame"].astype(np.int64), "x_px": L["x_nm"] / S.PIXEL_NM[0], "y_px": L["y_nm"] / S.PIXEL_NM[1],
         "z_nm": L["z_nm"]}
    a, b = match(t, f, S.PIXEL_NM)
    d = np.c_[(f["x_px"][b] - t["x_px"][a]) * S.PIXEL_NM[0], (f["y_px"][b] - t["y_px"][a]) * S.PIXEL_NM[1],
              f["z_nm"][b] - t["z_nm"][a]]
    bs = np.c_[t["s_x_nm"][a], t["s_y_nm"][a], t["s_z_nm"][a]]
    inlier = ~(np.abs(d / bs) > OUTLIER).any(1)
    sel = np.abs(t["z_nm"][a]) < Z_NM
    if "density" in name_set:
        sel &= t["photons"][a] >= PHOTONS_MIN
    out["n"] = int(sel.sum())
    out["outliers"] = float(1 - inlier[sel].mean())
    bm = np.c_[t["m_x_nm"][a], t["m_y_nm"][a], t["m_z_nm"][a]] if multi else bs
    s = sel & inlier
    out["rmse_crlb_lat"] = float(np.sqrt(np.mean(d[s, 0] ** 2 + d[s, 1] ** 2) / np.mean(bm[s, 0] ** 2 + bm[s, 1] ** 2)))
    out["rmse_crlb_ax"] = float(np.sqrt(np.mean(d[s, 2] ** 2) / np.mean(bm[s, 2] ** 2)))
    sig = np.c_[L["x_err_nm"][b], L["y_err_nm"][b], L["z_err_nm"][b]]
    out["sigma_lat"] = robust_width(np.r_[d[s, 0] / sig[s, 0], d[s, 1] / sig[s, 1]])
    out["sigma_ax"] = robust_width(d[s, 2] / sig[s, 2])
    # pixelation over every inlier, on the fitter's grid (DECODE-Plex: the frame's)
    gx, gy = (L["x_grid_px"], L["y_grid_px"]) if "x_grid_px" in L else (f["x_px"], f["y_px"])
    i = np.flatnonzero(inlier)
    out["pixelation"] = (max(ks_uniform(gx[b][i]), ks_uniform(gy[b][i]))
                         - max(ks_uniform(t["x_px"][a][i]), ks_uniform(t["y_px"][a][i])))
    if "photons_c" in L and "dye" in t:
        dyes = np.asarray(S.SETS[name].get("dyes", S.CONTEXTS[context].get("dyes", S.DYES)), float)
        pc = np.asarray(L["photons_c"], float)[b][i]
        share = pc[:, 1] / np.maximum(pc.sum(1), 1e-9)
        out["dye"] = float((np.abs(share[:, None] - dyes[None]).argmin(1) == t["dye"][a][i].astype(int)).mean())
    return out


COLUMNS = [("10k", "jaccard", "J 10k", 3), ("1k", "jaccard", "J 1k", 3), ("d15", "jaccard", "J d15", 3),
           ("d15 on 3", "jaccard", "J on3", 3), ("d15", "outliers", "out d15 %", 1), ("d15 on 3", "outliers", "out on3 %", 1),
           ("10k", "rmse_crlb_lat", "10k lat", 2), ("10k", "rmse_crlb_ax", "10k ax", 2),
           ("1k", "rmse_crlb_lat", "1k lat", 2), ("1k", "rmse_crlb_ax", "1k ax", 2),
           ("d15", "rmse_crlb_lat", "d15 lat", 2), ("d15", "rmse_crlb_ax", "d15 ax", 2),
           ("d15 on 3", "rmse_crlb_lat", "on3 lat", 2), ("d15 on 3", "rmse_crlb_ax", "on3 ax", 2),
           ("1k", "pixelation", "pix 1k", 3), ("1k", "dye", "dye 1k %", 1), ("d15", "dye", "dye d15 %", 1),
           ("d15", "sigma_lat", "σ lat", 2), ("d15", "sigma_ax", "σ ax", 2)]


def table(context, runs):
    """A Markdown table, one row per model (seeds pooled: mean, and sd where several)."""
    groups = {}
    for r in runs:
        groups.setdefault(re.sub(r"_s\d+$", "", r), []).append(r)
    rows = {}
    for g, labels in groups.items():
        per = [{k: measure(lab, k, context) for k in SETS} for lab in labels]
        cells = []
        for short, key, _, digits in COLUMNS:
            v = [p[short][key] for p in per if p[short] and key in p[short]]
            if not v:
                cells.append("")
                continue
            scale = 100 if key in ("outliers", "dye") else 1
            m, sd = scale * np.mean(v), scale * np.std(v)
            cells.append(f"{m:.{digits}f}" + (f" ({sd:.{digits}f})" if len(v) > 1 else ""))
        rows[g] = cells
        print(g, " | ".join(cells), flush=True, file=sys.stderr)
    lines = ["| model | " + " | ".join(c[2] for c in COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    lines += [f"| {g} | " + " | ".join(c) + " |" for g, c in rows.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(sys.argv[1], sys.argv[2:]))
