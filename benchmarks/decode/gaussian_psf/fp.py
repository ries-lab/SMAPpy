"""Where the Gaussian model's false positives are: their photons, prob and width, and how far from a true molecule."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from scipy.optimize import linear_sum_assignment

from smappy import plugins
from smappy.gui.live_session import fit_settings
from smappy.plugins import Context

b = Path(sys.argv[1]); name = sys.argv[2]; model = sys.argv[3]
t = np.load(b / f"{name}.npz")
cam = {"source.path": str(b / f"{name}.tif"), "camera.conversion": 0.5, "camera.offset": 100.0,
       "camera.pixelsize_um": 0.1, "camera.em_on": False, "output.path": "", "output.raw_frames": 0,
       "model.path": str(b / model / "model.safetensors"), "model.device": "cpu"}
L = plugins.get("Localize/DECODE")().run(Context(), fit_settings("Localize/DECODE", cam)).locs
f = np.asarray(L["frame"]); x = np.asarray(L["x_nm"], float); y = np.asarray(L["y_nm"], float)
tx, ty = t["x"] * 100, t["y"] * 100
inside = (x >= 400) & (x < 6000) & (y >= 400) & (y < 6000)
tp = np.zeros(len(f), bool); dist_true = np.full(len(f), np.inf); err = np.full((len(f), 2), np.nan)
for k in np.unique(f):
    c = np.flatnonzero(f == k); a = np.flatnonzero(t["frame"] == k)
    if not len(a):
        continue
    d = np.hypot(tx[a, None] - x[None, c], ty[a, None] - y[None, c])
    r, q = linear_sum_assignment(np.where(d < 100, d, 1e6)); ok = d[r, q] < 100
    tp[c[q[ok]]] = True
    err[c[q[ok]], 0] = x[c[q[ok]]] - tx[a[r[ok]]]; err[c[q[ok]], 1] = y[c[q[ok]]] - ty[a[r[ok]]]
    dist_true[c] = d.min(0)
fp = inside & ~tp
cols = {k: np.asarray(L[k], float) for k in ("photons", "prob", "sigma_nm") if k in L}
print(name, model, "locs", inside.sum(), "false", fp.sum())
for k, v in cols.items():
    print(f"  {k}: true median {np.median(v[inside & tp]):.3g}, false median {np.median(v[fp]):.3g}")
print("  false: distance to nearest true molecule, quartiles nm", np.percentile(dist_true[fp], [25, 50, 75]).round())
tests = [(f"sigma <= {w}", cols["sigma_nm"] <= w) for w in (160, 180, 200)] if "sigma_nm" in cols else []
tests += [(f"prob >= {p:.2f}", cols["prob"] >= p) for p in (0.9, 0.95, 0.99)]
if "sigma_nm" in cols:
    tests += [("sigma <= 180 and prob >= 0.95", (cols["sigma_nm"] <= 180) & (cols["prob"] >= 0.95))]
for th, keep in tests:
    n_tp, n_fp = (inside & tp & keep).sum(), (fp & keep).sum()
    n_true = ((tx >= 400) & (tx < 6000) & (ty >= 400) & (ty < 6000)).sum()
    print(f"  {th}: precision {n_tp / (n_tp + n_fp):.3f}, recall {n_tp / n_true:.3f}, "
          f"Jaccard {n_tp / (n_true + n_fp):.3f}, RMSE {np.sqrt(np.nanmean((err[inside & tp & keep] ** 2).sum(1))):.1f} nm, "
          f"bias {np.nanmean(err[inside & tp & keep], 0).round(1)}")
