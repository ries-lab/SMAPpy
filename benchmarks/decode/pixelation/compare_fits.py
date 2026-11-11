"""The same spots fitted with two models: whose fit explains them better, and where in z they disagree.

    python benchmarks/decode/pixelation/compare_fits.py spline_A spline_C

Spline 3D's fits of one movie with two calibrations, matched spot by spot (same frame, positions within
half a pixel, nearest).  For each pair: the difference of the log-likelihood per pixel (``logl_rel``,
the second minus the first: positive where the second model explains the spot better), against the
first fit's z; and where z went (the second's z against the first's).  A model that sharpens z by not
describing the data (a PSF that changes too little with z) fits the spots it moves worse, not better.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import OUT, PIXEL_NM  # noqa: E402


def load(name):
    from smappy.io.hdf5 import load_localizations
    locs = load_localizations(OUT / f"{name}.hdf5")
    ok = np.asarray(locs["photons"]) >= 1000
    if "prob" in locs:
        ok &= np.asarray(locs["prob"]) >= 0.7             # as the analysis keeps DECODE's
    out = {k: np.asarray(locs[k])[ok] for k in ("frame", "x_nm", "y_nm", "z_nm", "photons")}
    out["logl_rel"] = np.asarray(locs["logl_rel"])[ok] if "logl_rel" in locs else None   # DECODE writes none
    return out


def match(a, b, tol_nm=0.5 * min(PIXEL_NM)):
    from scipy.spatial import cKDTree
    ia, ib = [], []
    fa, fb = a["frame"], b["frame"]
    oa, ob = np.argsort(fa, kind="stable"), np.argsort(fb, kind="stable")
    frames = np.intersect1d(fa, fb)
    sa = np.searchsorted(fa[oa], frames), np.searchsorted(fa[oa], frames, "right")
    sb = np.searchsorted(fb[ob], frames), np.searchsorted(fb[ob], frames, "right")
    for k in range(len(frames)):
        pa, pb = oa[sa[0][k]:sa[1][k]], ob[sb[0][k]:sb[1][k]]
        tree = cKDTree(np.stack([b["x_nm"][pb], b["y_nm"][pb]], 1))
        d, j = tree.query(np.stack([a["x_nm"][pa], a["y_nm"][pa]], 1), distance_upper_bound=tol_nm)
        hit = np.isfinite(d)
        ia.append(pa[hit])
        ib.append(pb[j[hit]])
    return np.concatenate(ia), np.concatenate(ib)


def main(first, second):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = load(first), load(second)
    ia, ib = match(a, b)
    za, zb = a["z_nm"][ia], b["z_nm"][ib]
    both = a["logl_rel"] is not None and b["logl_rel"] is not None
    dl = b["logl_rel"][ib] - a["logl_rel"][ia] if both else np.full(len(ia), np.nan)
    print(f"{len(ia):,} spots in both ({len(ia) / len(a['frame']):.0%} of {first}'s, {len(ib) / len(b['frame']):.0%} of {second}'s)")
    if both:
        print(f"logl_rel, {second} - {first}: median {np.median(dl):+.4f}, better in {np.mean(dl > 0):.0%} of spots")
    edges = np.arange(-800, 801, 100)
    print(f"{'z of ' + first:>14s} {'n':>8s} {'median dlogl':>12s} {'better':>7s} {'median z of ' + second:>18s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (za >= lo) & (za < hi)
        if s.sum() < 200:
            continue
        print(f"{lo:6.0f} - {hi:4.0f}  {s.sum():8d} {np.median(dl[s]):+12.4f} {np.mean(dl[s] > 0):6.0%} {np.median(zb[s]):18.0f}"
              f"   spread of {second}'s z {np.subtract(*np.percentile(zb[s], [84, 16])) / 2:5.0f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].hist2d(za, zb, bins=[np.arange(-800, 801, 20)] * 2, cmap="magma", norm=matplotlib.colors.LogNorm())
    ax[0].plot([-800, 800], [-800, 800], "c--", lw=1)
    ax[0].set(xlabel=f"z with {first} (nm)", ylabel=f"z with {second} (nm)", title="the same spots' z")
    if not both:
        ax[1].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"compare_{first}_{second}.png", dpi=100)
        print("figure:", OUT / f"compare_{first}_{second}.png")
        return
    lo, hi = np.percentile(dl, [1, 99])
    ax[1].hist2d(za, dl, bins=[np.arange(-800, 801, 20), np.linspace(lo, hi, 80)], cmap="magma", norm=matplotlib.colors.LogNorm())
    ax[1].axhline(0, color="c", lw=1)
    ax[1].set(xlabel=f"z with {first} (nm)", ylabel=f"logl_rel {second} - {first}", title="which model explains the spot better")
    fig.tight_layout()
    fig.savefig(OUT / f"compare_{first}_{second}.png", dpi=100)
    print("figure:", OUT / f"compare_{first}_{second}.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
