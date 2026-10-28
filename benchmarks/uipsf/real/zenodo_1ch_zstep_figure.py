"""The table and the figure of `zenodo_1ch_zstep.py` (docs/uipsf/zstep.md).

    python benchmarks/uipsf/real/zenodo_1ch_zstep_figure.py [benchmarks/uipsf/results/zenodo_1ch_zstep.jsonl] \\
        [-o zstep.png]

The table: each metric's mean over the splits ± their standard deviation, and in brackets its change from
the finest step (or the widest range) of the same method, taken within each split and then averaged, ±
its sd.  The change is the comparison that matters: one split's held-out beads judge every setting, so
most of the split-to-split spread is the beads', the same at every setting.

The figure plots that change, each metric against the step (left) and the range (right), with the
split-to-split sd of the baseline as a grey band: a change inside the band is smaller than what choosing
other beads does.  A bead average that failed is marked at the bottom; one whose spline covered only part
of the judged window is drawn hollow.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [("z_rms_line_nm", "z rms about a line (nm)"), ("z_bias_nm", "z bias, systematic (nm)"),
           ("slope", "z slope"), ("z_crlb_nm", "z CRLB (nm)"), ("x_rms_nm", "x rms (nm)"),
           ("deviance_per_px", "deviance per pixel"), ("zernike_dist_nm", "Zernikes from reference (nm rms)")]
TABLE = ["fitted_fraction", "z_rms_line_nm", "z_bias_nm", "slope", "z_scatter_nm", "z_crlb_nm", "x_rms_nm",
         "y_rms_nm", "x_crlb_nm", "deviance_per_px", "zernike_dist_nm"]
STYLE = {"pupil": ("#2a78d6", "o", "pupil model"), "pupil@step": ("#7fb2ee", "s", "pupil, spline at the step"),
         "average": ("#eb6834", "^", "bead average"),
         "pupil@wide": ("#1baf7a", "D", "pupil, spline over ±975, judged ±850")}
PARTIAL = 0.95          # below this fraction of the window fitted, a row is drawn hollow


def groups(rows):
    out = defaultdict(dict)
    for r in rows:
        if r["split"] >= 0:
            out[(r["method"], r["step_nm"], r["half_range_nm"])][r["split"]] = r
    return out


def base_of(g, method):
    return g.get(("pupil" if method == "pupil@step" else method, 50.0, 1000.0))


def stats(g, key, base=None):
    """(mean, sd) over the splits of ``key``, or of its change from ``base``'s value in the same split."""
    vals = np.asarray([r.get(key, np.nan) - (base[s].get(key, np.nan) if base else 0.0) for s, r in g.items()
                       if base is None or s in base], float)
    if not np.isfinite(vals).any():
        return np.nan, np.nan
    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan


def table(rows):
    g = groups(rows)
    lines = ["| method | step (nm) | range (nm) | splits | " + " | ".join(TABLE) + " |",
             "|---" * (4 + len(TABLE)) + "|"]
    for (method, step, half) in sorted(g, key=lambda k: (k[0], k[2] != 1000.0, k[1], -k[2])):
        sub = g[(method, step, half)]
        base = base_of(g, method)
        first = next(iter(sub.values()))
        used = f"{first['z_ends_nm'][0]:g} to {first['z_ends_nm'][1]:g}"
        if all("failed" in r for r in sub.values()):
            lines.append(f"| {method} | {step:g} | {used} | {len(sub)} | failed: {first['failed']} |"
                         + " |" * (len(TABLE) - 1))
            continue
        cells = []
        for key in TABLE:
            m, s = stats(sub, key)
            if not np.isfinite(m):
                cells.append("")
                continue
            d = "" if base is None or base is sub or key == "fitted_fraction" else \
                " ({:+.2g} ± {:.1g})".format(*stats(sub, key, base))
            cells.append(f"{m:.4g} ± {s:.2g}{d}")
        lines.append(f"| {method} | {step:g} | {used} | {len(sub)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _series(ax, g, method, key, xs, keys, x_of):
    colour, marker, name = STYLE[method]
    base = base_of(g, method)
    pts = []
    for x, k in zip(xs, keys, strict=True):
        sub = g[k]
        if all("failed" in r for r in sub.values()):
            ax.annotate("fails", (x_of(x), 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7, color=colour)
            continue
        m, s = stats(sub, key, base)
        partial = stats(sub, "fitted_fraction")[0] < PARTIAL
        pts.append((x_of(x), m, s, partial))
    if not pts or not np.isfinite([p[1] for p in pts]).any():
        return
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=1.2, label=name)
    for x, m, s, partial in pts:
        ax.errorbar(x, m, s, color=colour, marker=marker, capsize=3, mfc="white" if partial else colour, ms=5)


def figure(rows, path):
    g = groups(rows)
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(8, 1.85 * len(METRICS)), sharex="col")
    for i, (key, label) in enumerate(METRICS):
        for col in (0, 1):
            ax = axes[i, col]
            ref = g.get(("pupil", 50.0, 1000.0))
            spread = stats(ref, key)[1] if ref else np.nan
            if np.isfinite(spread):
                ax.axhspan(-spread, spread, color="0.88", zorder=0)
            ax.axhline(0, color="0.5", lw=0.8, zorder=0)
            ax.tick_params(labelsize=8)
        for method in ("pupil", "pupil@step", "average"):
            keys = sorted(k for k in g if k[0] == method and k[2] == 1000.0)
            _series(axes[i, 0], g, method, key, [k[1] for k in keys], keys, lambda x: x)
        for method in ("pupil", "pupil@wide"):
            keys = sorted((k for k in g if k[0] == method and k[1] == 50.0), key=lambda k: k[2])
            ends = [next(iter(g[k].values()))["range_used_nm"] for k in keys]
            _series(axes[i, 1], g, method, key, ends, keys, lambda x: x)
        axes[i, 0].set_ylabel("Δ " + label, fontsize=8)
    for col in (0, 1):
        axes[0, col].legend(fontsize=6.5, loc="upper left")
    axes[0, 0].set_title("z step (scan over ±1000 nm)", fontsize=9)
    axes[0, 1].set_title("z range (50 nm step)", fontsize=9)
    axes[-1, 0].set_xlabel("z step (nm)")
    axes[-1, 1].set_xlabel("half range scanned (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?", default="benchmarks/uipsf/results/zenodo_1ch_zstep.jsonl")
    ap.add_argument("-o", "--out", default="zstep.png")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.jsonl)]
    print(table(rows))
    figure(rows, args.out)


if __name__ == "__main__":
    main()
