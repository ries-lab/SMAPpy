"""The benchmark's report: one JSON per run, every set scored the same way (`docs/decode/benchmark.md`).

    python benchmarks/decode/suite/report.py RUN [...]        # score the kept fits again
    python benchmarks/decode/suite/report.py --compare RUN [...]

**Kept** before scoring, by the run's ``keep``: a network's localizations at ``prob`` >= ``min_prob``;
Spline 3D's within ``logl_mads`` MADs of the median log-likelihood and not at the iteration limit;
both with |z| inside the sets' range.  **Matched** per frame by the Hungarian algorithm, within 250 nm
laterally and 500 nm in z (DECODE's evaluation and the challenge's), inside a margin of 7 pixels
(Spline 3D's fit box cannot leave the frame) -- (i)'s `score_sets.match`.

**Simulated and challenge sets:** Jaccard, recall, precision; RMSE laterally and in z, the
challenge's efficiencies; the mean offset in x, y, z and z's in 100 nm bins of the true z; and, where
the truth carries each emitter's CRLB (the simulated sets):

* ``rmse_over_crlb_lat`` / ``_ax``: the RMSE over the root mean square of the matched emitters' bounds;
* ``claim_width_lat`` / ``_ax``, ``claim_width_by_z``: the robust width of each error over the
  fitter's own claimed uncertainty (``x_err_nm`` ...), 1 where its claims are right; z by 200 nm bins
  of the true z, each bin's bias taken off (DECODE's sigma_z is right at the focus and up to twice
  too small away from it, poc-plan);
* ``crlb_width_lat`` / ``_ax``: the robust width (1.4826 MAD) of each localization's error over its own
  emitter's bound, x and y pooled -- the Train window's Follow measure, which mismatches and merged
  pairs do not drag; 1 where a fitter reaches the bound.  Both with the 13 x 13 bound and, as
  ``*_full``, the whole spot's.

**The range that matters**, ``core``: positions, uncertainties, z bias and pixelation of the
emitters with |z| < ``CORE_Z_NM`` (300 nm), read first; the whole range's are kept beside them, but
a variant that gains far from focus at a cost near it is not better.

**Pixelation:** the KS distance from uniform of the matched localizations' place within their
pixel, x and y, all and in bins of |z| (`smappy.training.reference.ks_uniform`), and the sub-pixel
histograms; the truth is uniform within its pixels, so any departure is the fitter's.

**Real sets** have no truth: localizations a frame, the KS distance of the kept ones by |z| and
photons, and the pair check -- a molecule on in two frames localized twice, each pair's difference
over the root of its two claimed variances, 1 where a fitter's claimed precision is what the data
show; the frames as far apart as the fitter needs for the two fits to share no frame
(`score_real`).

**Speed** is the fit's wall time over its frames, file reading and peak finding included, on the
device recorded.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
sys.path.insert(0, str(HERE.parent / "pixelation"))
import sets as S  # noqa: E402
from score_sets import EDGE_PX, match  # noqa: E402

from smappy.training.reference import ks_uniform, robust_width, subpixel_histogram  # noqa: E402

FITS = S.ROOT / "fits"
REPORTS = S.ROOT / "reports"
Z_LIMIT = 700.0
Z_BINS = (0, 150, 300, 450, 600, 700)          # |z| bins of the pixelation
PHOTON_BINS = (0, 2000, 4000, 8000, 16000, 1e9)
CORE_Z_NM = 300.0                              # |z| below it is what a fitter is for; see score_truth


def kept(locs, keep: dict, z_limit=Z_LIMIT):
    ok = np.abs(np.asarray(locs["z_nm"])) <= z_limit
    if "min_prob" in keep and "prob" in locs:
        ok &= np.asarray(locs["prob"]) >= keep["min_prob"]
    if "logl_mads" in keep and "logl_rel" in locs:
        logl = np.asarray(locs["logl_rel"])
        med = np.median(logl[ok])
        ok &= logl >= med - keep["logl_mads"] * 1.4826 * np.median(np.abs(logl[ok] - med))
    if keep.get("below_iteration_limit") and "iterations" in locs:
        it = np.asarray(locs["iterations"])
        ok &= it < it.max()
    if "min_photons" in keep:
        ok &= np.asarray(locs["photons"]) >= keep["min_photons"]
    return ok


def _inside(x, y, w, h):
    return (x >= EDGE_PX - 0.5) & (x < w - EDGE_PX - 0.5) & (y >= EDGE_PX - 0.5) & (y < h - EDGE_PX - 0.5)


def _binned_ks(fx, fy, v, edges, least=500):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v >= lo) & (v < hi)
        if sel.sum() >= least:
            out.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                        "ks": max(ks_uniform(fx[sel]), ks_uniform(fy[sel]))})
    return out


def score_truth(truth, locs, keep, px, size, shift=False):
    """Scores of a fit against a truth (dict of arrays: frame, x_px, y_px, z_nm, photons and, for the
    simulated sets, ``crlb_*``).  ``shift``: take off the fit's median offset from the truth first
    (the challenge: the calibration's centre is the bead average's, not the truth's)."""
    w, h = size
    ok = kept(locs, keep)
    f = {"frame": np.asarray(locs["frame"])[ok].astype(np.int64), "x_px": np.asarray(locs["x_nm"], float)[ok] / px[0],
         "y_px": np.asarray(locs["y_nm"], float)[ok] / px[1], "z_nm": np.asarray(locs["z_nm"], float)[ok]}
    claims = {a: np.asarray(locs[f"{a}_err_nm"], float)[ok] for a in "xyz" if f"{a}_err_nm" in locs}
    out = {}
    if shift:
        import score_sets
        lat, ax = score_sets.LATERAL_NM, score_sets.AXIAL_NM
        score_sets.LATERAL_NM, score_sets.AXIAL_NM = 2 * lat, 2 * ax
        try:
            ti, fi = match(truth, f, px)
        finally:
            score_sets.LATERAL_NM, score_sets.AXIAL_NM = lat, ax
        d = [float(np.median(f[k][fi] - truth[k][ti])) for k in ("x_px", "y_px", "z_nm")]
        f["x_px"], f["y_px"], f["z_nm"] = f["x_px"] - d[0], f["y_px"] - d[1], f["z_nm"] - d[2]
        out["shift_nm"] = [d[0] * px[0], d[1] * px[1], d[2]]
    keep_f = _inside(f["x_px"], f["y_px"], w, h)
    f = {k: v[keep_f] for k, v in f.items()}
    claims = {a: v[keep_f] for a, v in claims.items()}
    keep_t = _inside(truth["x_px"], truth["y_px"], w, h)
    t = {k: np.asarray(v)[keep_t] for k, v in truth.items() if np.ndim(v) == 1 and len(v) == len(keep_t)}
    ti, fi = match(t, f, px)
    tp, fp, fn = len(ti), len(f["frame"]) - len(ti), len(t["frame"]) - len(ti)
    dx = (f["x_px"][fi] - t["x_px"][ti]) * px[0]
    dy = (f["y_px"][fi] - t["y_px"][ti]) * px[1]
    dz = f["z_nm"][fi] - t["z_nm"][ti]
    jac = tp / max(tp + fp + fn, 1)
    lat, ax = float(np.sqrt(np.mean(dx ** 2 + dy ** 2))), float(np.sqrt(np.mean(dz ** 2)))
    eff = lambda rmse, a: float((100 - np.sqrt((100 * (1 - jac)) ** 2 + (a * rmse) ** 2)) / 100)   # noqa: E731
    zb = np.arange(-Z_LIMIT, Z_LIMIT + 1, 100)
    which = np.digitize(t["z_nm"][ti], zb) - 1
    fx, fy = f["x_px"][fi], f["y_px"][fi]
    out.update({
        "n_truth": len(t["frame"]), "n_kept": len(f["frame"]), "tp": tp, "fp": fp, "fn": fn,
        "jaccard": jac, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
        "rmse_lat_nm": lat, "rmse_ax_nm": ax, "eff_lat": eff(lat, 1.0), "eff_ax": eff(ax, 0.5),
        "bias_x_nm": float(np.mean(dx)), "bias_y_nm": float(np.mean(dy)), "bias_z_nm": float(np.mean(dz)),
        "z_bins_nm": zb.tolist(),
        "z_bias_nm": [float(np.mean(dz[which == k])) if np.sum(which == k) >= 50 else None for k in range(len(zb) - 1)],
        "ks_x": ks_uniform(fx), "ks_y": ks_uniform(fy),
        "ks_by_abs_z": _binned_ks(fx, fy, np.abs(t["z_nm"][ti]), Z_BINS),
        "subpixel_x": subpixel_histogram(fx), "subpixel_y": subpixel_histogram(fy)})
    if len(claims) == 3 and tp:
        # the fitter's own uncertainty against its error: 1 where its claims are right.  z in bins of the
        # true z with each bin's bias taken off -- the claim is of the spread, not of a bias
        cx, cy, cz = (claims[a][fi] for a in "xyz")
        out["claim_width_lat"] = robust_width(np.r_[dx / cx, dy / cy])
        out["claim_width_ax"] = robust_width(dz / cz)
        rows = []
        for lo in range(-int(Z_LIMIT), int(Z_LIMIT), 200):
            s = (t["z_nm"][ti] >= lo) & (t["z_nm"][ti] < lo + 200)
            if s.sum() >= 200:
                rows.append({"lo": float(lo), "hi": float(lo + 200), "n": int(s.sum()),
                             "lat": robust_width(np.r_[dx[s] / cx[s], dy[s] / cy[s]]),
                             "ax": robust_width((dz[s] - np.median(dz[s])) / cz[s])})
        out["claim_width_by_z"] = rows
    # the range that matters (Jonas, 2026-11-12): |z| < CORE_Z_NM.  Positions, uncertainties and
    # pixelation of the emitters there, so that no variant is judged better for gains far from focus
    # bought near it; detection stays over the whole range
    core = np.abs(t["z_nm"][ti]) < CORE_Z_NM
    if core.sum() >= 100:
        c = {"n": int(core.sum()),
             "rmse_lat_nm": float(np.sqrt(np.mean(dx[core] ** 2 + dy[core] ** 2))),
             "rmse_ax_nm": float(np.sqrt(np.mean(dz[core] ** 2))),
             "bias_z_nm": float(np.mean(dz[core])),
             "ks": max(ks_uniform(fx[core]), ks_uniform(fy[core]))}
        if len(claims) == 3:
            c["claim_width_lat"] = robust_width(np.r_[dx[core] / cx[core], dy[core] / cy[core]])
            c["claim_width_ax"] = robust_width(dz[core] / cz[core])
        if "crlb_x_nm" in t:
            bx, by, bz = (t[f"crlb_{a}_nm"][ti][core] for a in "xyz")
            c["crlb_width_lat"] = robust_width(np.r_[dx[core] / bx, dy[core] / by])
            c["crlb_width_ax"] = robust_width(dz[core] / bz)
        out["core"] = c
    for key in ("crlb", "crlb_full"):
        if f"{key}_x_nm" not in t:
            continue
        bx, by, bz = (t[f"{key}_{a}_nm"][ti] for a in "xyz")
        sfx = "" if key == "crlb" else "_full"
        out[f"rmse_over_crlb_lat{sfx}"] = float(lat / np.sqrt(np.nanmean(bx ** 2 + by ** 2)))
        out[f"rmse_over_crlb_ax{sfx}"] = float(ax / np.sqrt(np.nanmean(bz ** 2)))
        out[f"crlb_width_lat{sfx}"] = robust_width(np.r_[dx / bx, dy / by])
        out[f"crlb_width_ax{sfx}"] = robust_width(dz / bz)
        if key == "crlb":
            ph = t["photons"][ti]
            out["crlb_width_by_photons"] = [
                {"lo": float(lo), "hi": float(hi), "n": int(s.sum()), "lat": robust_width(np.r_[dx[s] / bx[s], dy[s] / by[s]]),
                 "ax": robust_width(dz[s] / bz[s])}
                for lo, hi in zip(PHOTON_BINS[:-1], PHOTON_BINS[1:]) for s in [(ph >= lo) & (ph < hi)] if s.sum() >= 200]
    return out


def pairs_apart(frame, x, y, lag, within=100.0):
    """Indices (i, j) of localizations ``lag`` frames apart that are each other's nearest within
    ``within`` nm: mostly one molecule localized twice (`pixelation/consecutive_frames.pairs` at a lag)."""
    from scipy.spatial import cKDTree
    order = np.argsort(frame, kind="stable")
    f, xs, ys = frame[order], x[order], y[order]
    first = int(f.min())
    starts = np.searchsorted(f, np.arange(first, f.max() + lag + 2))
    out_i, out_j = [], []
    for t in range(first, int(f.max()) - lag + 1):
        a = slice(starts[t - first], starts[t - first + 1])
        b = slice(starts[t - first + lag], starts[t - first + lag + 1])
        if a.start == a.stop or b.start == b.stop:
            continue
        pa, pb = np.c_[xs[a], ys[a]], np.c_[xs[b], ys[b]]
        d_ab, j = cKDTree(pb).query(pa, distance_upper_bound=within)
        _, i = cKDTree(pa).query(pb, distance_upper_bound=within)
        ok = np.isfinite(d_ab)
        ia = np.nonzero(ok)[0]
        mutual = i[j[ok]] == ia
        out_i.append(a.start + ia[mutual])
        out_j.append(b.start + j[ok][mutual])
    if not out_i:
        return np.zeros(0, int), np.zeros(0, int)
    return order[np.concatenate(out_i)], order[np.concatenate(out_j)]


#: the claims' bins of the pair check, nm (x) and nm (z)
CLAIM_BINS = {"x": (0, 5, 10, 15, 20, 30, 50), "z": (0, 10, 20, 30, 45, 70, 120)}


def independent_lag(locs) -> int:
    """The least number of frames apart two localizations are fitted from no shared frame: a network
    reading ``2 w + 1`` frames (DECODE's ``window`` w, recorded in the fit) localizes frames t and t + 1
    from shared ones -- on the Fig. 4e movie's 3-frame model their errors correlated by 0.73
    (`precision_check.py`) -- so the pair check compares frames that far apart; 1 for a fitter of
    single frames."""
    window = (locs.metadata.get("decode") or {}).get("window") or 0
    return 2 * int(window) + 1


def score_real(locs, keep, px):
    """Scores of a fit of real frames: no truth, so what the localizations say of themselves.

    **The pair check**: a molecule on in frames t and t + lag is localized twice; each pair's
    difference over the root of its two claimed variances has a robust width of 1 where the claims
    are right and the two fits independent.  Normalised per pair -- a robust spread of the pooled
    differences over the root mean claim is below 1 even for right claims, a mixture's robust width
    being below its root mean square -- in bins of the claim, at ``lag`` the fitter's independent one
    (`independent_lag`), and at 1 for the record.  On real data it grows with the lag for every
    fitter (the example's 1-frame model 1.13 at 1, 1.38 at 6): the sample moves between frames, which
    no claim includes, so the least independent lag is the fair one."""
    keep = {"min_photons": 1000.0, **keep}          # the pixelation study's: dim spots are mostly misfits
    ok = kept(locs, keep)
    col = {k: np.asarray(locs[k])[ok] for k in ("frame", "x_nm", "y_nm", "z_nm", "photons", "x_err_nm", "y_err_nm", "z_err_nm")}
    frames = np.asarray(locs["frame"])
    n_frames = int(frames.max() - frames.min() + 1) if len(frames) else 0
    fx, fy = col["x_nm"] / px[0], col["y_nm"] / px[1]
    lag = independent_lag(locs)
    out = {"n_locs": int(len(frames)), "n_kept": int(ok.sum()), "per_frame": float(ok.sum() / max(n_frames, 1)),
           "ks_x": ks_uniform(fx), "ks_y": ks_uniform(fy),
           "ks_by_abs_z": _binned_ks(fx, fy, np.abs(col["z_nm"]), Z_BINS, 2000),
           "ks_by_photons": _binned_ks(fx, fy, col["photons"], PHOTON_BINS, 2000),
           "subpixel_x": subpixel_histogram(fx), "subpixel_y": subpixel_histogram(fy), "pair_lag": lag}
    for at, name in ((lag, "pairs"), (1, "pairs_lag1")):
        if name == "pairs_lag1" and lag == 1:
            continue
        i, j = pairs_apart(col["frame"].astype(np.int64), col["x_nm"], col["y_nm"], at)
        out[name] = {"n": int(len(i))}
        for axis in ("x", "z"):
            claim = np.sqrt(col[f"{axis}_err_nm"][i] ** 2 + col[f"{axis}_err_nm"][j] ** 2)
            v = (col[f"{axis}_nm"][i] - col[f"{axis}_nm"][j]) / claim
            out[name][axis] = robust_width(v)
            rows = []
            edges = np.asarray(CLAIM_BINS[axis], float) * np.sqrt(2)      # the claim is of a difference
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (claim >= lo) & (claim < hi)
                if sel.sum() >= 200:
                    rows.append({"claim_lo": float(lo / np.sqrt(2)), "claim_hi": float(hi / np.sqrt(2)),
                                 "n": int(sel.sum()), "ratio": robust_width(v[sel])})
            out[name][f"{axis}_by_claim"] = rows
    return out


def challenge_truth(name):
    a = np.loadtxt(S.CHALLENGE / name / "activations.csv", delimiter=",", skiprows=1)
    # as (i)'s challenge: pixel k covers k*100 .. (k+1)*100 nm, z the other way round from the bead stack's
    return {"frame": a[:, 1].astype(np.int64) - 1, "x_px": a[:, 2] / 100 - 0.5, "y_px": a[:, 3] / 100 - 0.5,
            "z_nm": -a[:, 4], "photons": a[:, 5]}


def score_set(name, locs, keep):
    spec = S.SETS[name]
    cam = S.camera(name)
    px = (cam["pixelsize_um"] * 1000, cam.get("pixelsize_y_um", cam["pixelsize_um"]) * 1000)
    if spec["kind"] == "simulated":
        truth = dict(np.load(S.SETS_DIR / name / "truth.npz"))
        return score_truth(truth, locs, keep, px, (S.SIZE, S.SIZE))
    if spec["kind"] == "challenge":
        return score_truth(challenge_truth(spec["name"]), locs, keep, px, (64, 64), shift=True)
    return score_real(locs, keep, px)


def rescore(run_name):
    """Score a run's kept fits again, keeping its record of how each was made."""
    from smappy.io.hdf5 import load_localizations
    from run import load_run
    path = REPORTS / f"{run_name}.json"
    report = json.loads(path.read_text())
    run = load_run(run_name)
    for name, entry in report["sets"].items():
        entry["scores"] = score_set(name, load_localizations(FITS / run_name / f"{name}.hdf5"), run["keep"])
    path.write_text(json.dumps(report, indent=1))
    return report


# ----------------------------------------------------------------------------- the comparison
CORE_COLUMNS = [("rmse_lat_nm", "lat", "{:.1f}"), ("rmse_ax_nm", "ax", "{:.1f}"), ("crlb_width_lat", "w/CR lat", "{:.2f}"),
                ("crlb_width_ax", "w/CR ax", "{:.2f}"), ("claim_width_lat", "err/cl lat", "{:.2f}"),
                ("claim_width_ax", "err/cl ax", "{:.2f}"), ("bias_z_nm", "dz", "{:+.1f}"), ("ks", "KS", "{:.3f}")]
COLUMNS = [("jaccard", "J", "{:.3f}"), ("recall", "rec", "{:.3f}"), ("precision", "prec", "{:.3f}"),
           ("rmse_lat_nm", "lat", "{:.1f}"), ("rmse_ax_nm", "ax", "{:.1f}"),
           ("crlb_width_lat", "w/CR lat", "{:.2f}"), ("crlb_width_ax", "w/CR ax", "{:.2f}"),
           ("rmse_over_crlb_lat", "rmse/CR lat", "{:.2f}"), ("claim_width_lat", "err/cl lat", "{:.2f}"),
           ("claim_width_ax", "err/cl ax", "{:.2f}"), ("claim_ax_worst", "err/cl ax max", "{:.2f}"),
           ("bias_z_nm", "dz", "{:+.1f}"),
           ("ks_max", "KS", "{:.3f}"), ("ks_far", "KS |z|>450", "{:.3f}"), ("fps", "frames/s", "{:.0f}")]


def _row(entry):
    s = dict(entry["scores"])
    s["ks_max"] = max((v for v in (s.get("ks_x"), s.get("ks_y")) if v is not None), default=None)
    far = [b["ks"] for b in s.get("ks_by_abs_z", []) if b["lo"] >= 450]
    s["ks_far"] = max(far) if far else None
    s["fps"] = entry.get("frames_per_s")
    by_z = [b["ax"] for b in s.get("claim_width_by_z", []) if b["ax"] is not None]
    s["claim_ax_worst"] = max(by_z) if by_z else None
    return s


def compare(names):
    reports = {n: json.loads((REPORTS / f"{n}.json").read_text()) for n in names}
    for n, r in reports.items():
        c = r.get("cost", {})
        print(f"{n}: {r['run'].get('about', '')}")
        print("   parts: " + ", ".join(f"{k}={v['name']}" for k, v in r["run"].get("parts", {}).items()) or "   (not a network)")
        if c:
            print("   cost: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    sets = [s for s in S.SETS if any(s in r["sets"] for r in reports.values())]
    head = f"{'set':22s} {'run':12s} " + " ".join(f"{h:>10s}" for _, h, _ in COLUMNS)
    print(head)
    for s in sets:
        for n, r in reports.items():
            if s not in r["sets"]:
                continue
            row = _row(r["sets"][s])
            cells = [(fmt.format(row[k]) if row.get(k) is not None else "-") for k, _, fmt in COLUMNS]
            print(f"{s:22s} {n:12s} " + " ".join(f"{c:>10s}" for c in cells))
    print(f"\n|z| < {CORE_Z_NM:.0f} nm, the range that matters: the matched emitters there")
    print(f"{'set':22s} {'run':12s} " + " ".join(f"{h:>10s}" for _, h, _ in CORE_COLUMNS))
    for s in sets:
        for n, r in reports.items():
            core = r["sets"].get(s, {}).get("scores", {}).get("core")
            if core:
                cells = [(fmt.format(core[k]) if core.get(k) is not None else "-") for k, _, fmt in CORE_COLUMNS]
                print(f"{s:22s} {n:12s} " + " ".join(f"{c:>10s}" for c in cells))
    print("\nreal sets: kept a frame, KS by |z|; the pair check (observed / claimed, x) at the independent lag, by claim")
    for s in sets:
        for n, r in reports.items():
            sc = r["sets"].get(s, {}).get("scores", {})
            if "per_frame" in sc:
                p = sc["pairs"]
                cons = "  ".join(f"{b['claim_lo']:.0f}-{b['claim_hi']:.0f}:{b['ratio']:.2f}" for b in p["x_by_claim"])
                ks = "  ".join(f"{b['lo']:.0f}-{b['hi']:.0f}:{b['ks']:.3f}" for b in sc["ks_by_abs_z"])
                lag1 = sc.get("pairs_lag1")
                print(f"{s:10s} {n:12s} {sc['per_frame']:6.1f}/frame  KS {_row({'scores': sc})['ks_max']:.3f}  by |z| {ks}\n"
                      f"{'':23s} pairs {sc['pair_lag']} apart: x {p['x']:.2f} z {p['z']:.2f}"
                      + (f" (1 apart: x {lag1['x']:.2f} z {lag1['z']:.2f})" if lag1 else "") + f";  x by claim {cons}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        compare(args[1:])
    else:
        for n in args:
            rescore(n)
        compare(args)


if __name__ == "__main__":
    main()
