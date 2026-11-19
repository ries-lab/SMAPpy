"""The time models on the real sets (poc-plan, "Step 4: the time model"): pixelation and the pair check
of each run's fit, the pair check with the reporting head's sigma and, beside it, the mixture's.

    python benchmarks/decode/suite/real_time_models.py SET RUN...

Reads ``fits/<RUN>/<SET>.hdf5`` (`run.py`) and scores it as `report.score_real` does: localizations
of at least 1000 photons and p >= 0.5, |z| <= 700 nm; pairs ``2 w + 1`` frames apart for a model of
window ``2 w + 1``, so that the two fits share no frame.  The mixture's sigma is kept by the predictor
as ``x_err_mix_pix`` (pixels; the reported one is ``x_err_nm``)."""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import report as R

from smappy.io.hdf5 import load_localizations


def pair_check(col, i, j, err_x, err_z):
    out = {}
    for axis, err in (("x", err_x), ("z", err_z)):
        claim = np.sqrt(err[i] ** 2 + err[j] ** 2)
        v = (col[f"{axis}_nm"][i] - col[f"{axis}_nm"][j]) / claim
        rows = []
        edges = np.asarray(R.CLAIM_BINS[axis], float) * np.sqrt(2)
        for lo, hi in itertools.pairwise(edges):
            sel = (claim >= lo) & (claim < hi)
            if sel.sum() >= 200:
                rows.append([float(lo / np.sqrt(2)), float(hi / np.sqrt(2)), int(sel.sum()), R.robust_width(v[sel])])
        out[axis] = R.robust_width(v)
        out[f"{axis}_by_claim"] = rows
    return out


def score(name, run):
    locs = load_localizations(R.FITS / run / f"{name}.hdf5")
    cam = R.S.camera(name)
    px = (cam["pixelsize_um"] * 1000, cam.get("pixelsize_y_um", cam["pixelsize_um"]) * 1000)     # as score_set
    ok = R.kept(locs, {"min_prob": 0.5, "min_photons": 1000.0})
    names = ["frame", "x_nm", "y_nm", "z_nm", "photons", "x_err_nm", "z_err_nm"]
    mix = "x_err_mix_pix" in locs.columns
    names += ["x_err_mix_pix", "z_err_mix_nm"] if mix else []
    col = {k: np.asarray(locs[k], np.float64)[ok] for k in names}
    fx, fy = col["x_nm"] / px[0], col["y_nm"] / px[1]
    frames = np.asarray(locs["frame"])
    lag = R.independent_lag(locs)
    i, j = R.pairs_apart(col["frame"].astype(np.int64), col["x_nm"], col["y_nm"], lag)
    out = {"run": run, "lag": lag, "per_frame": float(ok.sum() / (frames.max() - frames.min() + 1)),
           "ks": max(R.ks_uniform(fx), R.ks_uniform(fy)),
           "ks_by_abs_z": [[b["lo"], b["hi"], b["ks"]] for b in R._binned_ks(fx, fy, np.abs(col["z_nm"]), R.Z_BINS, 2000)],
           "ks_by_photons": [[b["lo"], b["hi"], b["ks"]] for b in R._binned_ks(fx, fy, col["photons"], R.PHOTON_BINS, 2000)],
           "n_pairs": len(i),
           "reported": pair_check(col, i, j, col["x_err_nm"], col["z_err_nm"]),
           "median_claim": [float(np.median(col["x_err_nm"])), float(np.median(col["z_err_nm"]))]}
    if mix:
        xm = col["x_err_mix_pix"] * px[0]
        out["mixture"] = pair_check(col, i, j, xm, col["z_err_mix_nm"])
        out["median_mix"] = [float(np.median(xm)), float(np.median(col["z_err_mix_nm"]))]
    return out


if __name__ == "__main__":
    name, runs = sys.argv[1], sys.argv[2:]
    res = {r: score(name, r) for r in runs}
    (R.REPORTS / f"time_models_{name}.json").write_text(json.dumps(res, indent=1))
    for r, o in res.items():
        z = " ".join(f"{b[2]:.3f}" for b in o["ks_by_abs_z"])
        p = " ".join(f"{b[2]:.3f}" for b in o["ks_by_photons"])
        rep = o["reported"]; mx = o.get("mixture", {})
        print(f"{r:16s} lag {o['lag']} {o['per_frame']:6.1f}/fr KS {o['ks']:.3f} | z: {z} | ph: {p} | "
              f"rep x/z {rep['x']:.2f}/{rep['z']:.2f}" + (f" mix {mx['x']:.2f}/{mx['z']:.2f}" if mx else ""))
