"""(docs/4pi.md §29)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_evaluate.py NAME=table.hdf5 ...

Each model's Spline 4Pi fit of Cell01's first 6000 frames judged on the even blocks of 500 frames (those in situ
did not learn from), as `evaluate.py` judges the Nup96: the CRLBs at matched photons, the spline fit's
``logl_rel``, *4Pi z*'s uncertain fringes, and the phase's z against the envelope's (slope, and the scatter in
units of the envelope's CRLB and in nm).  No rings: tubulin has none.  One JSON line per model."""
import json
import sys

import numpy as np
from smappy.io.formats import load

BLOCK = 500


def robust_sd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def judge(path):
    locs = load(path)[0]
    c = {k: np.asarray(locs[k]) for k in locs.keys()}
    sel = ((c["frame"] // BLOCK) % 2 == 0) & np.isfinite(c["z_nm"])
    out = {"rows": int(sel.sum()), "median photons": float(np.median(c["photons"][sel]))}
    mid = sel & (c["photons"] > 3000) & (c["photons"] < 6000)
    for k in ("x_err_nm", "y_err_nm", "z_env_err_nm", "z_err_nm"):
        out["median " + k + " (3000-6000 ph)"] = float(np.median(c[k][mid]))
    out["median logl_rel"] = float(np.nanmedian(c["logl_rel"][sel]))
    p = c["fringe_probability"]
    out["uncertain fringes"] = float(np.mean(1 - p[sel]))
    out["p<0.9"] = float(np.mean(p[sel] < 0.9))
    sure = sel & (p > 0.9) & (np.abs(c["z_env_nm"]) < 500)
    a, b = np.polyfit(c["z_env_nm"][sure], c["z_nm"][sure], 1)
    out["z vs z_env slope"] = float(a)
    r = c["z_nm"][sure] - (a * c["z_env_nm"][sure] + b)
    out["(z - z_env)/z_env_err robust sd"] = robust_sd(r / c["z_env_err_nm"][sure])
    out["z - z_env robust sd nm"] = robust_sd(r)
    out["period nm"] = float((locs.metadata or {}).get("fourpi", {}).get("period_nm", np.nan))
    return out


for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    print(json.dumps({"model": name, **judge(path)}), flush=True)
