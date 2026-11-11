"""(i)'s scores: every fit of every simulated set against its truth, the same way for every fitter.

    python benchmarks/decode/benchmark/score_sets.py [--calibration A]

**Kept** before matching: DECODE's localizations at ``prob`` at least ``MIN_PROB`` (0.5, its own
training's evaluation threshold); Spline 3D's at a log-likelihood per pixel within ``LOGL_MADS`` MADs of
the median and not at the iteration limit; for both, |z| within the spline's range.
**Matched** per frame by the Hungarian algorithm on the lateral distance, a pair allowed within
``LATERAL_NM`` laterally and ``AXIAL_NM`` axially (250 and 500 nm, as DECODE's evaluation and the SMLM
challenge).  **The metrics**, DECODE's: Jaccard, recall, precision; lateral, axial and volume RMSE of
the matched pairs; the efficiencies of Sage et al. 2019, ``100 - sqrt((100 (1 - J))^2 + a^2 RMSE^2)``
with a 1 a nm laterally and 0.5 axially (as fractions here); and SMAPpy's: the mean offset in x, y and z
(a bias), z's bias in 100 nm bins of the true z, pixelation (KS distance of the matched positions
within a pixel from uniform), and the lateral RMSE against Spline 3D's own CRLB at the same spots
(its ``xy_err_nm``) at the lowest density and at least ``CRLB_PHOTONS`` photons.

Truth and fits are scored inside a margin of ``EDGE_PX`` (7) pixels only: Spline 3D's fit box cannot
leave the frame, DECODE's can, and the borders are not what is compared.  The truth indices the CRLB
reference keeps are of the emitters inside.

Writes ``<calibration>/scores.jsonl`` (one line a set and fitter) and prints a table.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pixelation"))
from common import DATA  # noqa: E402

MIN_PROB = 0.5
LOGL_MADS = 3.0
LATERAL_NM, AXIAL_NM = 250.0, 500.0
CRLB_PHOTONS = 2000.0
# truth and fits are scored inside this margin only: Spline 3D cannot fit a spot whose fit box (13
# pixels) leaves the frame, DECODE localizes to the edge, and the benchmark compares fitters, not borders
EDGE_PX = 7.0
ROOT = DATA / "benchmark"


def kept(locs, z_limit):
    ok = np.abs(np.asarray(locs["z_nm"])) <= z_limit
    if "prob" in locs:
        ok &= np.asarray(locs["prob"]) >= MIN_PROB
    if "logl_rel" in locs:
        logl = np.asarray(locs["logl_rel"])
        med = np.median(logl[ok])
        ok &= logl >= med - LOGL_MADS * 1.4826 * np.median(np.abs(logl[ok] - med))
    if "iterations" in locs:
        it = np.asarray(locs["iterations"])
        ok &= it < it.max()
    return ok


def match(t, f, px):
    """Pairs (truth index, fit index), per frame, Hungarian on the lateral distance in nm."""
    from scipy.optimize import linear_sum_assignment
    ti, fi = [], []
    tf, ff = t["frame"], f["frame"]
    ot, of = np.argsort(tf, kind="stable"), np.argsort(ff, kind="stable")
    frames = np.union1d(tf, ff)
    st = np.searchsorted(tf[ot], frames), np.searchsorted(tf[ot], frames, "right")
    sf = np.searchsorted(ff[of], frames), np.searchsorted(ff[of], frames, "right")
    for k in range(len(frames)):
        a, b = ot[st[0][k]:st[1][k]], of[sf[0][k]:sf[1][k]]
        if not len(a) or not len(b):
            continue
        dx = (t["x_px"][a, None] - f["x_px"][None, b]) * px[0]
        dy = (t["y_px"][a, None] - f["y_px"][None, b]) * px[1]
        dz = t["z_nm"][a, None] - f["z_nm"][None, b]
        lat = np.hypot(dx, dy)
        cost = np.where((lat <= LATERAL_NM) & (np.abs(dz) <= AXIAL_NM), lat, 1e9)
        r, c = linear_sum_assignment(cost)
        good = cost[r, c] < 1e9
        ti.append(a[r[good]])
        fi.append(b[c[good]])
    return (np.concatenate(ti), np.concatenate(fi)) if ti else (np.zeros(0, int), np.zeros(0, int))


def ks_uniform(u):
    u = np.sort(np.asarray(u, float))
    n = len(u)
    if not n:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def inside(x_px, y_px, size):
    return (x_px >= EDGE_PX - 0.5) & (x_px < size - EDGE_PX - 0.5) & (y_px >= EDGE_PX - 0.5) & (y_px < size - EDGE_PX - 0.5)


def score(truth, locs, px, z_limit, crlb_ref=None, size=64):
    ok = kept(locs, z_limit)
    ok &= inside(np.asarray(locs["x_nm"], float) / px[0], np.asarray(locs["y_nm"], float) / px[1], size)
    f = {"frame": np.asarray(locs["frame"])[ok], "x_px": np.asarray(locs["x_nm"], float)[ok] / px[0],
         "y_px": np.asarray(locs["y_nm"], float)[ok] / px[1], "z_nm": np.asarray(locs["z_nm"], float)[ok]}
    keep_t = inside(truth["x_px"], truth["y_px"], size)
    t = {k: truth[k][keep_t] for k in ("frame", "x_px", "y_px", "z_nm", "photons")}
    ti, fi = match(t, f, px)
    tp, fp, fn = len(ti), len(f["frame"]) - len(ti), len(t["frame"]) - len(ti)
    dx = (f["x_px"][fi] - t["x_px"][ti]) * px[0]
    dy = (f["y_px"][fi] - t["y_px"][ti]) * px[1]
    dz = f["z_nm"][fi] - t["z_nm"][ti]
    jac = tp / max(tp + fp + fn, 1)
    lat, ax = float(np.sqrt(np.mean(dx ** 2 + dy ** 2))), float(np.sqrt(np.mean(dz ** 2)))
    eff = lambda rmse, a: (100 - np.sqrt((100 * (1 - jac)) ** 2 + (a * rmse) ** 2)) / 100   # noqa: E731
    zbins = np.arange(-z_limit, z_limit + 1, 100)
    which = np.digitize(t["z_nm"][ti], zbins) - 1
    zbias = [float(np.mean(dz[which == k])) if np.sum(which == k) >= 50 else None for k in range(len(zbins) - 1)]
    out = {"n_truth": len(t["frame"]), "n_kept": len(f["frame"]), "tp": tp, "fp": fp, "fn": fn,
           "jaccard": jac, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
           "rmse_lat_nm": lat, "rmse_ax_nm": ax, "rmse_vol_nm": float(np.sqrt(np.mean(dx ** 2 + dy ** 2 + dz ** 2))),
           "eff_lat": float(eff(lat, 1.0)), "eff_ax": float(eff(ax, 0.5)),
           "bias_x_nm": float(np.mean(dx)), "bias_y_nm": float(np.mean(dy)), "bias_z_nm": float(np.mean(dz)),
           "z_bias_nm": zbias, "z_bins_nm": zbins.tolist(),
           "ks_x": ks_uniform((f["x_px"][fi] + 0.5) % 1), "ks_y": ks_uniform((f["y_px"][fi] + 0.5) % 1)}
    if crlb_ref is not None:              # Spline 3D's precision at the same true emitters, bright ones only
        sel = np.isin(ti, crlb_ref["truth_index"]) & (t["photons"][ti] >= CRLB_PHOTONS)
        if sel.sum() > 100:
            ref = crlb_ref["xy_err_nm"][np.searchsorted(crlb_ref["truth_index"], ti[sel])]
            # xy_err is the RMS of x's and y's error: x and y together are twice its square
            out["rmse_lat_over_crlb"] = float(np.sqrt(np.mean(dx[sel] ** 2 + dy[sel] ** 2))
                                              / np.sqrt(np.mean(2 * ref ** 2)))
            out["crlb_n"] = int(sel.sum())
    return out, ti, fi, ok


def main():
    from smappy.io.hdf5 import load_localizations
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="A")
    a = ap.parse_args()
    root = ROOT / a.calibration
    meta = json.loads((root / "sets.json").read_text())
    px = tuple(meta["pixel_nm"])
    z_limit = meta["z_nm"]
    rows = []
    for d in meta["sets"]:
        folder = root / f"density_{d}"
        truth = dict(np.load(folder / "truth.npz"))
        fits = sorted(p.stem for p in folder.glob("*.hdf5"))
        crlb_ref = None
        if "spline3d" in fits:            # Spline 3D's CRLB at the emitters it found: the reference precision
            locs = load_localizations(folder / "spline3d.hdf5")
            s, ti, fi, ok = score(truth, locs, px, z_limit, size=meta["size"])
            idx = np.flatnonzero(ok)[fi]
            order = np.argsort(ti)
            crlb_ref = {"truth_index": ti[order], "xy_err_nm": np.asarray(locs["xy_err_nm"])[idx][order]}
        for name in fits:
            s, *_ = score(truth, load_localizations(folder / f"{name}.hdf5"), px, z_limit,
                          crlb_ref if int(d) == min(map(int, meta["sets"])) else None, size=meta["size"])
            rows.append({"density": int(d), "fitter": name, **s})
    (root / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{'density':>7s} {'fitter':14s} {'jac':>6s} {'rec':>6s} {'prec':>6s} {'lat':>6s} {'ax':>6s} "
          f"{'eff_lat':>7s} {'eff_ax':>6s} {'bias x,y,z':>17s} {'KS x,y':>11s} {'lat/CRLB':>8s}")
    for r in rows:
        print(f"{r['density']:7d} {r['fitter']:14s} {r['jaccard']:6.3f} {r['recall']:6.3f} {r['precision']:6.3f} "
              f"{r['rmse_lat_nm']:6.1f} {r['rmse_ax_nm']:6.1f} {r['eff_lat']:7.3f} {r['eff_ax']:6.3f} "
              f"{r['bias_x_nm']:5.1f},{r['bias_y_nm']:5.1f},{r['bias_z_nm']:5.1f} "
              f"{r['ks_x']:5.3f},{r['ks_y']:5.3f} {r.get('rmse_lat_over_crlb', float('nan')):8.2f}")


if __name__ == "__main__":
    main()
