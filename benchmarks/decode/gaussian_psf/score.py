"""Fit a test movie four ways and score each against the truth: one-to-one matches per frame within 100 nm in x, y,
inside the frame's interior (4 pixels from its edges), as the DECODE / SMLM-challenge scores take them."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from scipy.optimize import linear_sum_assignment

from smappy import plugins
from smappy.gui.live_session import fit_settings
from smappy.plugins import Context

b = Path(sys.argv[1]); name = sys.argv[2]; which = sys.argv[3].split(";")
t = np.load(b / f"{name}.npz")
cam = {"source.path": str(b / f"{name}.tif"), "camera.conversion": 0.5, "camera.offset": 100.0,
       "camera.pixelsize_um": 0.1, "camera.em_on": False, "output.path": "", "output.raw_frames": 0}
METHODS = {
    # the likelihood fitters with the detection tuned on these movies: an absolute cutoff of 4 was the best of
    # 4 -- 30 on both (the dynamic cutoff rises with the crowding and found nothing at 50 per 40 x 40 px)
    "Gaussian 2D": ("Localize/Gaussian 2D", {"detection.cutoff_mode": "absolute", "detection.cutoff": 4.0}),
    "Spline 3D, true PSF": ("Localize/Spline 3D", {"model.calibration": str(b / "real_psf.h5"), "model.focus": "calibrated",
                                                   "detection.cutoff_mode": "absolute", "detection.cutoff": 4.0}),
    "DECODE, Gaussians": ("Localize/DECODE", {"model.path": str(b / "job_gauss" / "model.safetensors"), "model.device": "cpu"}),
    "DECODE, true PSF": ("Localize/DECODE", {"model.path": str(b / "job_psf" / "model.safetensors"), "model.device": "cpu"}),
}
LO, HI, CUT = 400.0, 6000.0, 100.0          # interior in nm (pixels 4 to 60), match distance


def score(locs):
    f = np.asarray(locs["frame"]); x = np.asarray(locs["x_nm"], float); y = np.asarray(locs["y_nm"], float)
    ph = np.asarray(locs["photons"], float)
    tx, ty = t["x"] * 100, t["y"] * 100
    inside_t = (tx >= LO) & (tx < HI) & (ty >= LO) & (ty < HI)
    inside = (x >= LO) & (x < HI) & (y >= LO) & (y < HI)
    tp, dx, dy, pr, zt, n_loc, n_true = 0, [], [], [], [], int(inside.sum()), int(inside_t.sum())
    for k in np.unique(t["frame"]):
        a = np.flatnonzero((t["frame"] == k) & inside_t)
        c = np.flatnonzero((f == k) & inside)
        if not len(a) or not len(c):
            continue
        d = np.hypot(tx[a, None] - x[None, c], ty[a, None] - y[None, c])
        r, q = linear_sum_assignment(np.where(d < CUT, d, 1e6))
        ok = d[r, q] < CUT
        tp += ok.sum()
        dx += list(x[c[q[ok]]] - tx[a[r[ok]]]); dy += list(y[c[q[ok]]] - ty[a[r[ok]]])
        pr += list(ph[c[q[ok]]] / t["photons"][a[r[ok]]]); zt += list(t["z"][a[r[ok]]])
    dx, dy, pr, zt = map(np.asarray, (dx, dy, pr, zt))
    fp, fn = n_loc - tp, n_true - tp
    rmse = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
    near = np.abs(zt) < 150
    return {"jaccard": tp / (tp + fp + fn), "recall": tp / n_true, "precision": tp / max(n_loc, 1),
            "rmse_lat_nm": rmse, "bias_nm": [float(np.mean(dx)), float(np.mean(dy))],
            "rmse_lat_in_focus_nm": float(np.sqrt(np.mean(dx[near] ** 2 + dy[near] ** 2))),
            "photons_over_true": float(np.median(pr)), "matched": int(tp)}


out = {}
for m in which:
    path, extra = METHODS[m]
    locs = plugins.get(path)().run(Context(), fit_settings(path, {**cam, **extra})).locs
    out[m] = score(locs)
    if "sigma_nm" in locs and m.startswith("DECODE"):
        out[m]["sigma_median_nm"] = float(np.median(locs["sigma_nm"]))
    print(name, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m].items()}, flush=True)
with open(b / f"score_{name}_{'_'.join(w.split()[0] for w in which)}.json", "w") as fh:
    json.dump(out, fh, indent=1)
