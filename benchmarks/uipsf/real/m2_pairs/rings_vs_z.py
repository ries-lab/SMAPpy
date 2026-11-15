"""NPC ring separation against the pore's z, for the fits of the nucleus movie with AB's calibration.

    python rings_vs_z.py NPCDIR calibrated auto [--json FILE]

Per variant: the pores (`pores_<v>.csv`, Line Profile's two Gaussians in z per pore) and the fitted table
(`out_<v>/.../*_proc.h5`, the chain's drift-corrected table); each pore's z the median z of the localizations
within 100 nm of its centre.  Pores kept: the
two-Gaussian fit determined (distance error < 10 nm), 20-120 nm apart, at least 40 localizations.  Prints the
median separation per 150 nm of z and its slope against z; draws `rings_vs_z.png`.
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pore_z(locs_path, xs, ys, r=100.0):
    from smappy.io.formats import load
    locs, _ = load(locs_path)
    x, y, z = (np.asarray(locs[c], float) for c in ("x_nm", "y_nm", "z_nm"))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[x, y])
    return np.array([np.median(z[idx]) if len(idx) else np.nan
                     for idx in tree.query_ball_point(np.c_[xs, ys], r)])


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        k = args.index("--json")
        out_json = args[k + 1]
        del args[k:k + 2]
    d, variants = args[0], args[1:]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for v in variants:
        import csv
        rows_v = list(csv.DictReader(open(os.path.join(d, f"pores_{v}.csv"))))
        f = lambda k, rows_v=rows_v: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan  # noqa: E731
                                               for r in rows_v])
        dist, err, n = f("two_gauss_distance_nm"), f("two_gauss_distance_err_nm"), f("n_locs")
        locs = glob.glob(os.path.join(d, f"out_{v}", "*", "*_proc.h5"))[0]
        z = pore_z(locs, f("x_nm"), f("y_nm"))
        ok = np.isfinite(dist) & (err < 10) & (dist > 20) & (dist < 120) & (n >= 40) & np.isfinite(z)
        a, b = np.polyfit(z[ok], dist[ok], 1)
        print(f"\n{v}: {ok.sum()} of {len(rows_v)} pores; separation median {np.median(dist[ok]):.1f} nm "
              f"(robust sd {1.4826 * np.median(np.abs(dist[ok] - np.median(dist[ok]))):.1f}); "
              f"slope {a * 1000:.1f} nm per µm of z; z of the pores {np.percentile(z[ok], [5, 50, 95]).round(0)}")
        edges = np.arange(np.floor(np.nanmin(z[ok]) / 150) * 150, np.nanmax(z[ok]) + 150, 150)
        centres, meds = [], []
        for lo in edges[:-1]:
            k = ok & (z >= lo) & (z < lo + 150)
            if k.sum() >= 5:
                centres.append(lo + 75)
                meds.append(np.median(dist[k]))
                print(f"  z {lo:6.0f}..{lo + 150:6.0f}: {k.sum():4d} pores, separation median {meds[-1]:5.1f} nm")
        rows.append(dict(variant=v, pores=int(ok.sum()), of=len(rows_v),
                         separation_median_nm=float(np.median(dist[ok])),
                         separation_robust_sd_nm=float(1.4826 * np.median(np.abs(dist[ok] - np.median(dist[ok])))),
                         slope_nm_per_um=float(a * 1000),
                         z_p5_p50_p95_nm=np.percentile(z[ok], [5, 50, 95]).round(0).tolist(),
                         per_150nm=[dict(z_centre_nm=c, median_nm=float(m))
                                    for c, m in zip(centres, meds, strict=True)]))
        axes[0].scatter(z[ok], dist[ok], s=6, alpha=0.4, label=v)
        axes[1].plot(centres, meds, "o-", label=v)
    for a in axes:
        a.set(xlabel="pore z (nm): calibrated = defocus, auto = height above the coverslip",
              ylabel="ring separation (nm)")
        a.legend()
    axes[0].set_title("each pore")
    axes[1].set_title("median per 150 nm of z")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "rings_vs_z.png"), dpi=110)
    if out_json:
        with open(out_json, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
