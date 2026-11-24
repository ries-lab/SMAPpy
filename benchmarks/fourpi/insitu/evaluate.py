"""P2: judge fits of the Nup96 movie with several models on the even files' localizations.
python evaluate.py NAME=table.hdf5 ...   -> one JSON line per model"""
import sys, json, numpy as np
from smappy.io.formats import load
FRAMES = 2000
DRIFT = __import__("os").path.expanduser("~/.cache/uipsf/p2/drift.pkl")
def drift_of(locs):
    """One 3D drift for every model's table: RCC on the first table's sure fringes, kept."""
    import os, pickle
    if os.path.exists(DRIFT):
        return pickle.load(open(DRIFT, "rb"))
    from smappy.rcc import RCCSettings, estimate_drift_rcc
    sure = np.asarray(locs["fringe_probability"]) > 0.9
    d = estimate_drift_rcc(locs, RCCSettings(n_timepoints=30, use_z=True), select=sure)
    pickle.dump(d, open(DRIFT, "wb"))
    return d
def judge(path):
    locs = load(path)[0]
    locs = drift_of(locs).apply(locs)
    c = {k: np.asarray(locs[k]) for k in locs.keys()}
    even = (c["frame"] // FRAMES) % 2 == 0
    out = {"rows": int(even.sum())}
    sel = even & np.isfinite(c["z_nm"])
    mid = sel & (c["photons"] > 700) & (c["photons"] < 1400)
    for k in ("x_err_nm", "y_err_nm", "z_env_err_nm", "z_err_nm"):
        if k in c:
            out["median " + k + " (700-1400 ph)"] = float(np.median(c[k][mid]))
    if "logl_rel" in c:
        out["median logl_rel"] = float(np.nanmedian(c["logl_rel"][sel]))
    p = c.get("fringe_probability")
    if p is not None:
        out["uncertain fringes"] = float(np.mean(1 - p[sel]))
        out["p<0.9"] = float(np.mean(p[sel] < 0.9))
        sure = sel & (p > 0.9) & (np.abs(c["z_env_nm"]) < 500)
        a, b = np.polyfit(c["z_env_nm"][sure], c["z_nm"][sure], 1)
        out["z vs z_env slope"] = float(a)
        r = (c["z_nm"][sure] - (a * c["z_env_nm"][sure] + b)) / c["z_env_err_nm"][sure]
        out["(z - z_env)/z_env_err robust sd"] = float(1.4826 * np.median(np.abs(r - np.median(r))))
    out.update(rings(c, sel & (p > 0.9) if p is not None else sel))
    return out
def two_rings(z, iters=200):
    """The separation of two Gaussians of one width fitted to z by EM."""
    m = np.percentile(z, [25, 75]).astype(float); s = np.std(z) / 2 + 1; w = 0.5
    for _ in range(iters):
        a = w * np.exp(-0.5 * ((z - m[0]) / s) ** 2); b = (1 - w) * np.exp(-0.5 * ((z - m[1]) / s) ** 2)
        r = a / (a + b + 1e-300)
        w = r.mean(); m = np.array([np.sum(r * z) / r.sum(), np.sum((1 - r) * z) / (1 - r).sum()])
        s = np.sqrt(np.sum(r * (z - m[0]) ** 2 + (1 - r) * (z - m[1]) ** 2) / len(z)) + 1e-6
    return float(abs(m[1] - m[0]))
def rings(c, sel):
    from smappy.segmentation import find_circles, fit_circle
    from scipy.spatial import cKDTree
    x, y, z = c["x_nm"][sel], c["y_nm"][sel], c["z_nm"][sel].copy()
    sites = find_circles({"x_nm": x, "y_nm": y}, "ring", 40.0, 70.0)
    tree = cKDTree(np.c_[x, y])
    seps, radii = [], []
    for s in sites:
        cx, cy = s["center"]
        idx = tree.query_ball_point([cx, cy], 90.0)
        if len(idx) < 30:
            continue
        zz = z[idx]
        zz = zz[np.isfinite(zz)]
        zz = zz[np.abs(zz - np.median(zz)) < 150] if len(zz) else zz
        if len(zz) < 30:
            continue
        seps.append(two_rings(zz))
        radii.append(float(np.median(np.hypot(x[idx] - cx, y[idx] - cy))))
    return {"pores": len(seps), "ring separation median nm": float(np.median(seps)) if seps else None,
            "ring separation IQR": [float(np.percentile(seps, 25)), float(np.percentile(seps, 75))] if seps else None,
            "radius median nm": float(np.median(radii)) if radii else None}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    print(json.dumps({"model": name, **judge(path)}), flush=True)
