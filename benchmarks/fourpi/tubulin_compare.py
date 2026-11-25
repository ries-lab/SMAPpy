"""(docs/4pi.md §26)  PYTHONPATH=src python benchmarks/fourpi/tubulin_compare.py TABLE [TABLE ...]

Step 3b on the tubulin data: our fit of Cell01's first 6000 frames (`tubulin_fit.py`) against the authors' on
the same frames.  The authors' are their MATLAB pipeline's (4Pi-SMS ``fit_ast.m``, redone 2020-09):
``Cell01_642_tmpresult_20200912.mat``, every fit -- x, y in pixels of their 168 px crop of channel 1 (our chip
x less 150, their crop starting at 151, counted from 1), the astigmatic z, the phase, photons, background,
CRLB -- and ``Cell01_642v20_60.mat``, the filtered, unwrapped, drift-corrected result, whose ``vutarapr``
names the raw fit each row came from (counted from 1).

Localizations are paired within a frame, nearest first, closer than 0.5 px (64 nm).  Reported: counts and
how many pair; photons and background against theirs; the lateral offset and spread between pairs; our
precision against their CRLB; and z -- ours (*4Pi z*) against their final z, less the median difference:
a pair more than half a fringe period apart is on another fringe in one of the two.  The authors' final z
is drift-corrected (`Cell01_1_drift.mat`: within ±12 nm per 3000-frame cycle) and ours is not, so the
spread includes that.  The first 6000 frames are cached from the .mat files (1.1 GB) in ``smappy_p1/``."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_common import DATA, OUT, PIXEL_UM

from smappy.io.formats import load

PX = PIXEL_UM * 1000
CROP_X = 150.0                       # our chip x (px) = their x + 150
FRAMES = 6000


def authors():
    cache = DATA / "smappy_p1" / "authors_6000.npz"
    if not cache.exists():
        import scipy.io as sio
        raw = sio.loadmat(DATA / "Cell01_642_tmpresult_20200912.mat")
        t = raw["tresult"].ravel()
        keep = np.flatnonzero(t < FRAMES)
        final = sio.loadmat(DATA / "Cell01_642v20_60.mat")
        row = final["vutarapr"].ravel().astype(np.int64) - 1
        z_final = np.full(len(t), np.nan, np.float32)
        z_final[row] = final["vutaraz"].ravel()
        np.savez(cache, frame=t[keep], x=raw["xresult"].ravel()[keep] + CROP_X, y=raw["yresult"].ravel()[keep],
                 photons=raw["Iresult"].ravel()[keep], bg=raw["bgresult"].ravel()[keep],
                 crlb=raw["CRLBresult"][keep], z_ast=raw["zfresult"].ravel()[keep],
                 phase=raw["zangresult"].ravel()[keep], ll=raw["llresult"].ravel()[keep], z_final=z_final[keep])
    return dict(np.load(cache))


def pair(fa, xa, ya, fb, xb, yb, radius=0.5):
    """Indices (a, b) of pairs in the same frame within ``radius`` px, nearest first, each used once."""
    from scipy.spatial import cKDTree
    ia, ib = [], []
    order_a, order_b = np.argsort(fa, kind="stable"), np.argsort(fb, kind="stable")
    sa, sb = np.searchsorted(fa[order_a], np.arange(FRAMES + 1)), np.searchsorted(fb[order_b], np.arange(FRAMES + 1))
    for f in range(FRAMES):
        a, b = order_a[sa[f]:sa[f + 1]], order_b[sb[f]:sb[f + 1]]
        if not len(a) or not len(b):
            continue
        d, j = cKDTree(np.c_[xb[b], yb[b]]).query(np.c_[xa[a], ya[a]], distance_upper_bound=radius)
        for k in np.argsort(d):
            if np.isfinite(d[k]) and b[j[k]] not in ib[-len(b):]:
                ia.append(a[k])
                ib.append(b[j[k]])
    return np.asarray(ia), np.asarray(ib)


def robust_sd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


them = authors()
for path in sys.argv[1:]:
    loaded = load(path, columns="all")
    ours = loaded[0] if isinstance(loaded, tuple) else loaded
    f = np.asarray(ours["frame"]).astype(int)
    x, y = np.asarray(ours["x_nm"]) / PX, np.asarray(ours["y_nm"]) / PX
    n_frames = int(f.max()) + 1
    mine = them["frame"] < n_frames
    a = {k: v[mine] for k, v in them.items()}
    i, j = pair(f, x, y, a["frame"].astype(int), a["x"], a["y"])
    print(f"\n{Path(path).name}: {n_frames} frames")
    print(f"  localizations: ours {len(f)} ({len(f) / n_frames:.1f} a frame), theirs {len(a['frame'])}, "
          f"in their final table {np.isfinite(a['z_final']).sum()}; paired {len(i)} "
          f"({len(i) / len(f):.0%} of ours, {len(i) / len(a['frame']):.0%} of theirs, "
          f"{np.isfinite(a['z_final'][j]).sum() / max(np.isfinite(a['z_final']).sum(), 1):.0%} of their final)")
    alone = np.ones(len(f), bool)
    alone[i] = False
    theirs_alone = np.ones(len(a["frame"]), bool)
    theirs_alone[j] = False
    print(f"  unpaired: ours {alone.sum()}, median {np.median(np.asarray(ours['photons'])[alone]):.0f} photons "
          f"(paired {np.median(np.asarray(ours['photons'])[i]):.0f}); theirs {theirs_alone.sum()}, median "
          f"{np.median(a['photons'][theirs_alone]):.0f} (paired {np.median(a['photons'][j]):.0f})")
    if "fringe_probability" in ours:
        p_all = np.asarray(ours["fringe_probability"])
        print(f"  our fringe probability > 0.9: {np.mean(p_all > 0.9):.0%} of all, {np.mean(p_all[i] > 0.9):.0%} of "
              f"the paired, {np.mean(p_all[alone] > 0.9):.0%} of the unpaired; expected wrong fringes "
              f"{1 - p_all.mean():.1%} over all")
    photons = np.asarray(ours["photons"])[i] / a["photons"][j]
    print(f"  photons: ours median {np.median(np.asarray(ours['photons'])):.0f}, theirs {np.median(a['photons']):.0f}; "
          f"per pair ours/theirs median {np.median(photons):.3f} (16-84 %: {np.percentile(photons, 16):.2f}-"
          f"{np.percentile(photons, 84):.2f})")
    bg = np.asarray(ours["background"])[i]
    print(f"  background: ours median {np.median(bg):.1f} photons/px (all channels' mean), theirs {np.median(a['bg'][j]):.1f}")
    dx, dy = (x[i] - a["x"][j]) * PX, (y[i] - a["y"][j]) * PX
    print(f"  lateral, ours - theirs: x {np.median(dx):+.1f} nm (robust sd {robust_sd(dx):.1f}), "
          f"y {np.median(dy):+.1f} nm ({robust_sd(dy):.1f})")
    print(f"  precision: ours x_err median {np.median(np.asarray(ours['x_err_nm'])[i]):.1f} nm, theirs CRLB x "
          f"{np.median(a['crlb'][j, 0]) * PX:.1f} nm (column 1 x {PX:.0f})")
    # the authors count z the other way: their final z and astigmatic z fall where ours rise (correlation
    # -0.96 on the first 300 frames), while the phases agree -- a convention, compared with theirs turned
    theirs = a["z_final"][j]
    ok = np.isfinite(theirs)
    sign = -1.0 if np.corrcoef(np.asarray(ours["z_nm"])[i][ok], theirs[ok])[0, 1] < 0 else 1.0
    zs = np.asarray(ours["z_nm"])[i] - sign * theirs
    print(f"  z: ours against {'minus ' if sign < 0 else ''}theirs, slope "
          f"{np.polyfit(sign * theirs[ok], np.asarray(ours['z_nm'])[i][ok], 1)[0]:.3f}")
    if ok.sum():
        period = float((ours.metadata or {}).get("fourpi", {}).get("period_nm", 0)) or \
            2 * np.pi / abs(float((ours.metadata or {}).get("fourpi", {}).get("phase_per_nm", 2 * np.pi / 291.5)))
        d = zs[ok] - np.median(zs[ok])
        other = np.abs(d) > period / 2
        print(f"  z against their final ({ok.sum()} pairs): offset {np.median(zs[ok]):+.1f} nm, spread of the "
              f"rest {robust_sd(d[~other]):.1f} nm; on another fringe {other.mean():.1%} "
              f"(period {period:.1f} nm)")
        if "fringe_probability" in ours:
            p = np.asarray(ours["fringe_probability"])[i][ok]
            print(f"    of those ours call 90 % sure: {np.mean(p > 0.9):.0%}, on another fringe among them "
                  f"{other[p > 0.9].mean():.1%}; our expected wrong fringes for these {1 - p.mean():.1%}")
        env = np.asarray(ours["z_env_nm"])[i][ok] - sign * theirs[ok]
        print(f"  envelope z against their final: spread {robust_sd(env - np.median(env)):.1f} nm")
