"""netfit (C = 2) on the first frames of DECODE-Plex's real movies, against the original's localizations.

    python real_c2.py fig2a|fig2c MODEL [SHIFT_ROWS] [FRAMES]

fig2a: colour_2a, the beads' ROI, the splitter moved by SHIFT_ROWS (5.8, plex-plan §13) since the beads;
fig2c: biplane_2c, camera ROI x0 = 175, no shift; DECODE-Plex's run mirrored the columns.
Writes <out>/<movie>_<model>_<shift>.npz and prints the agreement with DECODE-Plex (prob > 0.6)."""
import json
import pathlib
import sys
import time
from dataclasses import replace

import numpy as np
import tifffile
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sets_c2 as S

from smappy.netfit.channels import TwoChannels
from smappy.netfit.predict import Predictor
from smappy.training.reference import ks_uniform

D = pathlib.Path.home() / ".cache/decode_smlm/plex"
OUT = S.SETS_DIR.parent / "real_c2"
MOVIES = {"fig2a": ("colour_2a", "Fig2a-SmallTest/first300.tif", "oracles/plex_fig2a_cpu.npz", None, False),
          "fig2c": ("biplane_2c", "Fig2c-240619_NC_LivingCell_HMSir_Tub_i20_2/first300.tif", "oracles/plex_fig2c_cpu.npz",
                    (175, 0), True)}


def fit(movie, model, shift=0.0, n=200, goodness=True):
    ctx, path, _, corner, _ = MOVIES[movie]
    cal = S.calibration(ctx)
    if shift:       # the secondary's partner row moved by ``shift`` since the beads (as real_fig2a.py)
        cal = replace(cal, transformation=np.asarray(cal.transformation) @ np.array([[1, 0, 0], [0, 1, -shift], [0, 0, 1.]]))
    corner = corner or tuple(cal.geometry["sources"][0]["roi"][:2])
    p = Predictor(model)
    c = S.camera_of(ctx)
    adu = tifffile.imread(D / path)[:n].astype(np.float32)
    two = TwoChannels.from_calibration(cal, corner, main=p.channels.main, shape=adu.shape[1:])
    photons = (adu - c["offset"]) * (c["conversion"] / c["emgain"] if c["em_on"] else c["conversion"])
    k = p.window
    stack = two.cut(photons)
    stack = np.concatenate([np.repeat(stack[:1], k, 0), stack, np.repeat(stack[-1:], k, 0)])
    t0 = time.time()
    r = p(stack, 0, two.offsets())
    seconds = time.time() - t0
    r0, _, c0, _ = two.box
    xy = two.on_frame(np.c_[r["x_pix"] + c0, r["y_pix"] + r0])
    keep = {"frame": np.asarray(r["frame"]), "x_px": xy[:, 0], "y_px": xy[:, 1], "z_nm": np.asarray(r["z_nm"]),
            "x_err_nm": r["x_err_pix"] * S.PIXEL_NM[0], "y_err_nm": r["y_err_pix"] * S.PIXEL_NM[1],
            "z_err_nm": np.asarray(r["z_err_nm"]), "prob": np.asarray(r["prob"]), "photons": np.asarray(r["photons"]),
            "x_grid_px": r["x_pix"] + c0, "y_grid_px": r["y_pix"] + r0}
    if "photons_c" in r:
        keep["photons_c"] = two.in_order(r["photons_c"])
    bg = np.asarray(r["background_c"]) if "background_c" in r else np.repeat(np.asarray(r["background"])[:, None], 2, 1)
    keep["background_c"] = two.in_order(bg)
    if goodness and len(r["frame"]):
        from smappy.netfit.goodness import fit_goodness
        ph = np.asarray(r["photons_c"]) if p.channels.kind == "colour" else np.asarray(r["photons"])
        t1 = time.time()
        keep["goodness"] = fit_goodness(two, cal, p.channels.kind, stack[k:len(stack) - k], np.asarray(r["frame"], int),
                                        np.c_[r["x_pix"], r["y_pix"]], r["z_nm"], ph, None, bool(c["em_on"]))
        keep["goodness_seconds"] = np.array(time.time() - t1)
    keep["seconds"] = np.array(seconds)
    keep["box"] = np.array(two.box)
    return keep


def plex(movie, n=200):
    _, path, oracle, _, flipped = MOVIES[movie]
    o = np.load(D / oracle)
    w = tifffile.imread(D / path, key=0).shape[-1]
    col = o["em_xyz"][:, 1]
    if flipped:
        col = w - 1 - col
    sel = o["em_frame_ix"] < n
    return {"frame": o["em_frame_ix"][sel], "x_px": col[sel], "y_px": o["em_xyz"][sel, 0], "z_nm": o["em_xyz"][sel, 2],
            "x_err_nm": o["em_xyz_sig"][sel, 1] * S.PIXEL_NM[0], "y_err_nm": o["em_xyz_sig"][sel, 0] * S.PIXEL_NM[1],
            "z_err_nm": o["em_xyz_sig"][sel, 2], "prob": o["em_prob"][sel], "photons": o["em_phot"][sel]}


def agree(a, b, radius_nm=150.0):
    """Pairs within ``radius_nm`` frame by frame (nearest of b for each of a): (index a, index b, d (m, 3) nm)."""
    ia, ib, d = [], [], []
    px = np.asarray(S.PIXEL_NM)
    for f in np.intersect1d(np.unique(a["frame"]), np.unique(b["frame"])):
        i, j = np.flatnonzero(a["frame"] == f), np.flatnonzero(b["frame"] == f)
        dist, k = cKDTree(np.c_[b["x_px"][j], b["y_px"][j]] * px).query(np.c_[a["x_px"][i], a["y_px"][i]] * px)
        ok = dist < radius_nm
        ia.append(i[ok]); ib.append(j[k[ok]])
        d.append(np.c_[(a["x_px"][i[ok]] - b["x_px"][j[k[ok]]]) * px[0], (a["y_px"][i[ok]] - b["y_px"][j[k[ok]]]) * px[1],
                       a["z_nm"][i[ok]] - b["z_nm"][j[k[ok]]]])
    return np.concatenate(ia), np.concatenate(ib), np.concatenate(d)


def summary(movie, L, P, prob=0.6):
    L = {k: v[L["prob"] > prob] if np.ndim(v) and len(v) == len(L["prob"]) else v for k, v in L.items()}
    P = {k: v[P["prob"] > prob] for k, v in P.items()}
    n = len(np.unique(P["frame"]))
    ia, ib, d = agree(L, P)
    med = np.median(d, 0)
    spread = 1.4826 * np.median(np.abs(d - med), 0)
    out = {"per_frame": len(L["frame"]) / n, "plex_per_frame": len(P["frame"]) / n,
           "matched_of_ours": len(np.unique(ia)) / len(L["frame"]), "matched_of_plex": len(np.unique(ib)) / len(P["frame"]),
           "offset_nm": med.round(1).tolist(), "spread_nm": spread.round(1).tolist(),
           "sigma_ours": [float(np.median(L[k])) for k in ("x_err_nm", "y_err_nm", "z_err_nm")],
           "sigma_plex": [float(np.median(P[k])) for k in ("x_err_nm", "y_err_nm", "z_err_nm")],
           "pix_ours": max(ks_uniform(L["x_grid_px"]), ks_uniform(L["y_grid_px"])),
           "pix_plex": max(ks_uniform(P["x_px"]), ks_uniform(P["y_px"])),
           "z_pct_ours": np.percentile(L["z_nm"], [5, 50, 95]).round(0).tolist(),
           "z_pct_plex": np.percentile(P["z_nm"], [5, 50, 95]).round(0).tolist(),
           "photons_med": [float(np.median(L["photons"])), float(np.median(P["photons"]))]}
    if "goodness" in L:
        g = L["goodness"]
        out["goodness_pct"] = np.nanpercentile(g, [50, 90, 99]).round(2).tolist()
        out["goodness_seconds"] = float(L["goodness_seconds"])
    if "photons_c" in L:
        pc = L["photons_c"]
        out["share_pct"] = np.percentile(pc[:, 1] / np.maximum(pc.sum(1), 1e-9), [5, 25, 50, 75, 95]).round(3).tolist()
    out["seconds"] = float(L["seconds"])
    return out


if __name__ == "__main__":
    movie, model = sys.argv[1], sys.argv[2]
    shift = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    OUT.mkdir(exist_ok=True)
    L = fit(movie, model, shift, n)
    name = f"{movie}_{pathlib.Path(model).stem}_{shift:+.1f}"
    np.savez(OUT / f"{name}.npz", **L)
    print(name, json.dumps(summary(movie, L, plex(movie, n))), flush=True)
