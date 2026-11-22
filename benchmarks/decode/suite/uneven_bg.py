"""Does a network trained on a flat background follow an uneven one?  (Jonas, 2026-11-22: DECODE trained on a
flat background predicted realistic background maps; Perlin backgrounds in training did not help.)

    python uneven_bg.py CONTEXT MODEL_LABEL MODEL_PATH [PEAK] [flat]

The C = 2 suite's density-15 set with a smooth random field added -- a Gaussian-filtered noise image, 16
pixels wide, scaled to 0 -- ``PEAK`` photons a pixel, the same in every frame and both channels' own --
drawn through the camera (Poisson, the EM register's gamma, the read noise), and fitted by the model.  The
fit is kept as ``fits_c2/UBG_<label>/<set>.npz`` beside the flat set's, so `standard_c2.measure` scores
it against the same truth; and the network's background at each matched molecule against the truth's,
the set's level plus the field where the molecule is, per channel.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

import run_c2 as RC
import sets_c2 as S

WIDTH_PX = 16.0


def field(shape, peak, seed=7, flat=False):
    """The field, or with ``flat`` its mean everywhere: the control, as much light spread evenly."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=shape), WIDTH_PX)
    f = (f - f.min()) / (f.max() - f.min())
    return np.full(shape, peak * f.mean()) if flat else peak * f


def add_field(adu, camera, peak, seed=7, flat=False):
    """``adu`` (n, H, W) with a field's photons drawn into it: Poisson, the EM register, the conversion."""
    rng = np.random.default_rng(seed + 1)
    lam = field(adu.shape[1:], peak, seed, flat)
    n = rng.poisson(np.broadcast_to(lam, adu.shape))
    electrons = rng.gamma(np.maximum(n, 1e-9), camera["emgain"]) * (n > 0) if camera["em_on"] else n
    return adu + electrons / camera["conversion"], lam


def fit(context, frames_adu, model):
    from smappy.netfit.channels import TwoChannels
    from smappy.netfit.predict import Predictor
    cal = S.calibration(context)
    p = Predictor(model, threshold=None)
    two = TwoChannels.from_calibration(cal, cal.geometry["sources"][0]["roi"][:2], main=p.channels.main)
    c = S.camera_of(context)
    photons = (frames_adu - c["offset"]) * (c["conversion"] / c["emgain"] if c["em_on"] else c["conversion"])
    k = p.window
    stack = two.cut(photons)
    stack = np.concatenate([np.repeat(stack[:1], k, 0), stack, np.repeat(stack[-1:], k, 0)])
    r = p(stack, 0, two.offsets())
    r0, _, c0, _ = two.box
    xy = two.on_frame(np.c_[r["x_pix"] + c0, r["y_pix"] + r0])
    out = {"frame": r["frame"], "x_nm": xy[:, 0] * S.PIXEL_NM[0], "y_nm": xy[:, 1] * S.PIXEL_NM[1], "z_nm": r["z_nm"],
           "x_err_nm": r["x_err_pix"] * S.PIXEL_NM[0], "y_err_nm": r["y_err_pix"] * S.PIXEL_NM[1],
           "z_err_nm": r["z_err_nm"], "prob": r["prob"], "photons": r["photons"],
           "x_grid_px": r["x_pix"] + c0, "y_grid_px": r["y_pix"] + r0}
    if "photons_c" in r:
        out["photons_c"] = two.in_order(r["photons_c"])
    bg = np.asarray(r["background_c"]) if "background_c" in r else np.repeat(np.asarray(r["background"])[:, None], 2, 1)
    out["background_c"] = two.in_order(bg)
    return out, two


if __name__ == "__main__":
    context, label, model = sys.argv[1], sys.argv[2], sys.argv[3]
    peak = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
    flat = len(sys.argv) > 5 and sys.argv[5] == "flat"
    name = f"{context}_blink_density_15"
    adu = tifffile.imread(S.SETS_DIR / name / "frames.tif").astype(np.float32)
    noisy, lam = add_field(adu, S.camera_of(context), peak, flat=flat)
    out, two = fit(context, noisy, model)
    folder = RC.FITS / f"{'FBG' if flat else 'UBG'}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{name}.npz", **out)
    (folder / f"{name}.json").write_text(json.dumps({"window": 2 * 1 + 1, "peak": peak}))
    # the network's background at each matched molecule against the truth's: the set's level plus the field
    t = dict(np.load(S.SETS_DIR / name / "truth.npz"))
    fx, fy = out["x_nm"] / S.PIXEL_NM[0], out["y_nm"] / S.PIXEL_NM[1]
    a, b = [], []
    for f in np.unique(out["frame"]):
        i, j = np.flatnonzero(out["frame"] == f), np.flatnonzero(t["frame"] == f)
        if len(i) and len(j):
            d, q = cKDTree(np.c_[t["x_px"][j], t["y_px"][j]]).query(np.c_[fx[i], fy[i]])
            ok = d < 1
            a.append(j[q[ok]])
            b.append(i[ok])
    a, b = np.concatenate(a), np.concatenate(b)
    a, b = a[out["prob"][b] > 0.5], b[out["prob"][b] > 0.5]
    rows = []
    for c, (xs, ys) in enumerate(((t["x_px"][a], t["y_px"][a]), (t["x2_px"][a], t["y2_px"][a]))):
        here = lam[np.clip(np.rint(ys).astype(int), 0, lam.shape[0] - 1), np.clip(np.rint(xs).astype(int), 0, lam.shape[1] - 1)]
        true = t[f"background_{c}"][a] + here
        got = out["background_c"][b, c]
        r = np.corrcoef(here, got - t[f"background_{c}"][a])[0, 1]
        rows.append(f"channel {c}: field {np.percentile(here, 5):.0f}-{np.percentile(here, 95):.0f}; network/true median "
                    f"{np.median(got / true):.3f}, robust width {1.4826 * np.median(np.abs(got / true - np.median(got / true))):.3f}; "
                    f"correlation of the field with the network's excess {r:.3f}")
    print(name, f"peak {peak:.0f}:", "; ".join(rows), flush=True)
