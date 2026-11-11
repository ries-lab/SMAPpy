"""Pixelation in x and y, and how even z is, for the study's fits: histograms and one number each.

    python benchmarks/decode/pixelation/analyse.py spline_A spline_B ...

For each fit (``pixelation/<name>.hdf5``), the localizations a fit of good quality keeps -- photons
at least ``MIN_PHOTONS``, |z| within ``Z_LIMIT_NM``, and (where the fitter writes them) not at the
iteration limit and a log-likelihood per pixel within ``LOGL_MADS`` MADs of the median -- and of them:

DECODE writes no log-likelihood or iterations; its localizations are kept at ``prob`` at least
``MIN_PROB`` instead.

* **the position within a pixel**, x and y apart: a fitter whose PSF model is wrong pulls positions
  towards the pixel's centre or its edges.  Pixel centres are whole numbers (`locs.in_pixels`), so the
  fraction is ``(x_px + 0.5) mod 1``, 0 and 1 the pixel's edges.  The number is the Kolmogorov-Smirnov
  distance from uniform, and the histogram's relative spread (its standard deviation over its mean,
  against what Poisson counting alone gives);
* **z**: the histogram in 10 nm bins, for peaks, gaps and walls; and the position within the spline's
  slice grid of the calibration the fit used (``dz``), where a spline's own steps would show, with its
  KS distance.  A smooth sample has a smooth z histogram; the slice grid's should be flat.

Writes ``pixelation/pixelation.png`` and prints a table.  ``--by-error``: x and y within a pixel in bins
of the predicted error (``pixelation_by_error.png``); ``--z-by-logl``: the z histogram at tighter
log-likelihood cuts, for fitters that write one (``z_by_logl.png``); ``--by-condition``: pixelation
against photons, |z| and the local density (``pixelation_by_condition.png``).
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import OUT, PIXEL_NM  # noqa: E402

MIN_PHOTONS = 1000.0
Z_LIMIT_NM = 700.0
LOGL_MADS = 3.0
MIN_PROB = 0.7
BINS = 50


def keep(locs) -> np.ndarray:
    ok = np.asarray(locs["photons"]) >= MIN_PHOTONS
    ok &= np.abs(np.asarray(locs["z_nm"])) <= Z_LIMIT_NM
    if "logl_rel" in locs:
        logl = np.asarray(locs["logl_rel"])
        med = np.median(logl[ok])
        mad = 1.4826 * np.median(np.abs(logl[ok] - med))
        ok &= logl >= med - LOGL_MADS * mad
    if "prob" in locs:
        ok &= np.asarray(locs["prob"]) >= MIN_PROB
    if "iterations" in locs:
        it = np.asarray(locs["iterations"])
        ok &= it < it.max()
    return ok


def ks_uniform(u) -> float:
    u = np.sort(np.asarray(u, float))
    n = len(u)
    if not n:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def spread(u, bins=BINS) -> float:
    """The histogram's standard deviation over its mean, divided by Poisson's alone: 1 is flat."""
    h, _ = np.histogram(u, bins=bins, range=(0, 1))
    return float(np.std(h) / np.mean(h) / (1 / np.sqrt(np.mean(h))))


def measure(name):
    from smappy.io.hdf5 import load_localizations
    locs = load_localizations(OUT / f"{name}.hdf5")
    ok = keep(locs)
    fx = (np.asarray(locs["x_nm"])[ok] / PIXEL_NM[0] + 0.5) % 1.0
    fy = (np.asarray(locs["y_nm"])[ok] / PIXEL_NM[1] + 0.5) % 1.0
    z = np.asarray(locs["z_nm"])[ok]
    dz = _dz(locs) or _model_dz(name)
    fz = (z / dz) % 1.0 if dz else None
    return {"name": name, "n": int(ok.sum()), "of": len(locs), "fx": fx, "fy": fy, "z": z, "dz": dz, "fz": fz,
            "ks_x": ks_uniform(fx), "ks_y": ks_uniform(fy), "ks_z": ks_uniform(fz) if fz is not None else np.nan,
            "spread_x": spread(fx), "spread_y": spread(fy), "spread_z": spread(fz) if fz is not None else np.nan}


def _dz(locs):
    """The slice step of the spline the fit used, from the file's record; None when not recorded."""
    cal = (locs.metadata.get("calibration") or {}) if hasattr(locs, "metadata") else {}
    dz = cal.get("dz_nm")
    if dz is None and "decode" in locs.metadata:
        dz = (locs.metadata["decode"].get("model_metadata") or {}).get("calibration_dz_nm")
    return float(dz) if dz else None


def _model_dz(name):
    """The slice step of the study's model a fit's name ends in (A, B, C): DECODE's file does not record it."""
    from common import SMAP_SPLINE
    from smappy.io.calibration import load_spline_calibration
    files = {"A": SMAP_SPLINE, "B": OUT / "beads_pupil.h5", "C": OUT / "insitu.h5"}
    path = files.get(name[-1])
    return float(load_spline_calibration(path).dz) if path is not None and path.exists() else None


def main(names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [measure(n) for n in names]
    print(f"{'fit':16s} {'kept':>9s} {'of':>9s}  {'KS x':>6s} {'KS y':>6s} {'KS z':>6s}  "
          f"{'spread x':>8s} {'spread y':>8s} {'spread z':>8s}  dz")
    for r in rows:
        print(f"{r['name']:16s} {r['n']:9d} {r['of']:9d}  {r['ks_x']:6.3f} {r['ks_y']:6.3f} {r['ks_z']:6.3f}  "
              f"{r['spread_x']:8.1f} {r['spread_y']:8.1f} {r['spread_z']:8.1f}  {r['dz']}")
    fig, ax = plt.subplots(len(rows), 4, figsize=(16, 2.8 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        for j, (key, label) in enumerate((("fx", "x within a pixel"), ("fy", "y within a pixel"))):
            ax[i, j].hist(r[key], bins=BINS, range=(0, 1), color="C0")
            ax[i, j].set(title=f"{r['name']}: {label}, KS {r[f'ks_{key[1]}']:.3f}", xlim=(0, 1))
        ax[i, 2].hist(r["z"], bins=np.arange(-Z_LIMIT_NM, Z_LIMIT_NM + 1, 10), color="C1")
        ax[i, 2].set(title=f"{r['name']}: z, 10 nm bins", xlabel="z (nm)")
        if r["fz"] is not None:
            ax[i, 3].hist(r["fz"], bins=BINS, range=(0, 1), color="C2")
            ax[i, 3].set(title=f"z within the spline's {r['dz']:g} nm slices, KS {r['ks_z']:.3f}", xlim=(0, 1))
        for a in ax[i]:
            a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(OUT / "pixelation.png", dpi=100)
    print("figure:", OUT / "pixelation.png")




# ----------------------------------------------------------------------------- two closer looks
ERROR_BINS_NM = (0, 10, 15, 20, 30, 60)       # bins of the predicted lateral error, xy_err_nm
LOGL_CUTS = (None, 3.0, 1.0, 0.0, -1.0)       # MADs below the median log-likelihood kept; -1: only the better half's upper part


def by_error(names):
    """x and y within a pixel in bins of the predicted error: a bias is a larger share of a small error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smappy.io.hdf5 import load_localizations
    bins = list(zip(ERROR_BINS_NM[:-1], ERROR_BINS_NM[1:]))
    fig, ax = plt.subplots(2 * len(names), len(bins), figsize=(3.2 * len(bins), 2.4 * 2 * len(names)), squeeze=False)
    print(f"{'fit':12s} {'xy_err bin':>11s} {'n':>9s} {'KS x':>6s} {'KS y':>6s}  {'1/sqrt(n)':>9s}")
    for i, name in enumerate(names):
        locs = load_localizations(OUT / f"{name}.hdf5")
        ok = keep(locs)
        err = np.asarray(locs["xy_err_nm"])
        for j, (lo, hi) in enumerate(bins):
            sel = ok & (err >= lo) & (err < hi)
            fx = (np.asarray(locs["x_nm"])[sel] / PIXEL_NM[0] + 0.5) % 1.0
            fy = (np.asarray(locs["y_nm"])[sel] / PIXEL_NM[1] + 0.5) % 1.0
            kx, ky = ks_uniform(fx), ks_uniform(fy)
            print(f"{name:12s} {lo:4.0f}-{hi:<4.0f} nm {sel.sum():9d} {kx:6.3f} {ky:6.3f}  {1 / np.sqrt(max(sel.sum(), 1)):9.4f}")
            for k, (f, axis, ks) in enumerate(((fx, "x", kx), (fy, "y", ky))):
                a = ax[2 * i + k, j]
                a.hist(f, bins=BINS, range=(0, 1), color="C0" if axis == "x" else "C4")
                a.set(xlim=(0, 1), title=f"{name} {axis}, xy_err {lo:g}-{hi:g} nm: n {sel.sum():,}, KS {ks:.3f}")
                a.title.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(OUT / "pixelation_by_error.png", dpi=90)
    print("figure:", OUT / "pixelation_by_error.png")


def z_by_logl(names):
    """The z histogram as the log-likelihood cut tightens: misfits go first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smappy.io.hdf5 import load_localizations
    fig, ax = plt.subplots(1, len(names), figsize=(6 * len(names), 4), squeeze=False)
    for i, name in enumerate(names):
        locs = load_localizations(OUT / f"{name}.hdf5")
        base = (np.asarray(locs["photons"]) >= MIN_PHOTONS) & (np.abs(np.asarray(locs["z_nm"])) <= 800)
        it = np.asarray(locs["iterations"])
        base &= it < it.max()
        logl = np.asarray(locs["logl_rel"])
        med = np.median(logl[base])
        mad = 1.4826 * np.median(np.abs(logl[base] - med))
        z = np.asarray(locs["z_nm"])
        for cut in LOGL_CUTS:
            sel = base if cut is None else base & (logl >= med - cut * mad)
            h, e = np.histogram(z[sel], bins=np.arange(-800, 801, 10))
            label = "no cut" if cut is None else f"logl >= median - {cut:g} MAD" if cut >= 0 else f"logl >= median + {-cut:g} MAD"
            ax[0, i].plot(0.5 * (e[1:] + e[:-1]), h / h.max(), label=f"{label} ({sel.sum():,})")
        ax[0, i].set(title=f"{name}: z, each curve scaled to its peak", xlabel="z (nm)")
        ax[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "z_by_logl.png", dpi=100)
    print("figure:", OUT / "z_by_logl.png")


CONDITIONS = {
    "photons": ("photons", (1000, 2000, 4000, 8000, 16000, 1e9)),
    "|z|": ("abs_z", (0, 150, 300, 450, 600, 701)),
    "density": ("neighbours", (0, 1, 2, 4, 8, 1000)),       # others in the same frame within NEIGHBOUR_NM
}
NEIGHBOUR_NM = 1000.0


def _neighbours(locs, ok):
    """For each kept localization, how many others of its frame lie within NEIGHBOUR_NM: the local density."""
    from scipy.spatial import cKDTree
    frame = np.asarray(locs["frame"])[ok]
    xy = np.stack([np.asarray(locs["x_nm"])[ok], np.asarray(locs["y_nm"])[ok]], 1)
    out = np.zeros(len(frame), np.int32)
    order = np.argsort(frame, kind="stable")
    bounds = np.flatnonzero(np.diff(frame[order])) + 1
    for part in np.split(order, bounds):
        if len(part) > 1:
            tree = cKDTree(xy[part])
            out[part] = np.array([len(n) - 1 for n in tree.query_ball_point(xy[part], NEIGHBOUR_NM)])
    return out


def by_condition(names):
    """KS from uniform of x and y within a pixel, in bins of photons, |z| and local density: where the
    signal is poor -- few photons, far from focus, crowded -- a fitter's pixelation shows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smappy.io.hdf5 import load_localizations
    fig, ax = plt.subplots(1, len(CONDITIONS), figsize=(5.5 * len(CONDITIONS), 4), squeeze=False)
    for name in names:
        locs = load_localizations(OUT / f"{name}.hdf5")
        ok = keep(locs)
        fx = (np.asarray(locs["x_nm"])[ok] / PIXEL_NM[0] + 0.5) % 1.0
        fy = (np.asarray(locs["y_nm"])[ok] / PIXEL_NM[1] + 0.5) % 1.0
        values = {"photons": np.asarray(locs["photons"])[ok], "abs_z": np.abs(np.asarray(locs["z_nm"])[ok]),
                  "neighbours": _neighbours(locs, ok)}
        for j, (label, (key, edges)) in enumerate(CONDITIONS.items()):
            v = values[key]
            ks, mid = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (v >= lo) & (v < hi)
                if sel.sum() < 2000:
                    continue
                ks.append(max(ks_uniform(fx[sel]), ks_uniform(fy[sel])))
                mid.append(f"{lo:g}-{hi:g}" if hi < 1e8 and hi != 1000 else f">{lo:g}")
                print(f"{name:12s} {label:8s} {mid[-1]:>12s} n {sel.sum():9d}  KS max(x,y) {ks[-1]:.3f}")
            ax[0, j].plot(range(len(ks)), ks, "o-", label=name)
            ax[0, j].set_xticks(range(len(mid)), mid, fontsize=8)
        print()
    for j, label in enumerate(CONDITIONS):
        unit = {"photons": "photons", "|z|": "|z| (nm)", "density": f"others within {NEIGHBOUR_NM / 1000:g} um"}[label]
        ax[0, j].set(xlabel=unit, ylabel="KS from uniform, the larger of x and y", title=f"pixelation against {label}")
        ax[0, j].grid(alpha=.3)
        ax[0, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pixelation_by_condition.png", dpi=100)
    print("figure:", OUT / "pixelation_by_condition.png")


if __name__ == "__main__":
    if "--by-condition" in sys.argv:
        by_condition([a for a in sys.argv[1:] if not a.startswith("--")])
        sys.exit()
    # python analyse.py NAME ...: the table and pixelation.png; --by-error and --z-by-logl the closer looks
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--by-error" in sys.argv:
        by_error(args)
    elif "--z-by-logl" in sys.argv:
        z_by_logl(args)
    else:
        main(args)
