"""(i)'s figure: the scores of every fitter against density, and z's bias against z at the lowest density.

    python benchmarks/decode/benchmark/plot_scores.py [--calibration A]

Reads ``<calibration>/scores.jsonl`` (`score_sets.py`); writes ``<calibration>/scores.png``.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pixelation"))
from common import DATA  # noqa: E402

STYLE = {"spline3d": ("k", "-", "Spline 3D"), "package": ("C2", "-", "DECODE, the package's model"),
         "original_run1": ("C0", "-", "original, run 1"), "original_run2": ("C0", "--", "original, run 2"),
         "port_run0": ("C3", "-", "port, run 0"), "port_run1": ("C3", "--", "port, run 1"),
         "port_B": ("C3", "-", "port, trained on B")}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="A")
    a = ap.parse_args()
    root = DATA / "benchmark" / a.calibration
    rows = [json.loads(line) for line in (root / "scores.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    panels = (("jaccard", "Jaccard"), ("rmse_lat_nm", "lateral RMSE (nm)"), ("rmse_ax_nm", "axial RMSE (nm)"),
              ("eff_lat", "lateral efficiency"), ("eff_ax", "axial efficiency"))
    for name, (colour, ls, label) in STYLE.items():
        mine = sorted((r for r in rows if r["fitter"] == name), key=lambda r: r["density"])
        if not mine:
            continue
        d = [r["density"] for r in mine]
        for k, (key, title) in enumerate(panels):
            a_ = ax.flat[k]
            a_.plot(d, [r[key] for r in mine], marker="o", color=colour, ls=ls, label=label)
            a_.set(xscale="log", xlabel="emitters per 40 x 40 pixels", title=title)
        low = mine[0]
        z = np.asarray(low["z_bins_nm"], float)
        mid = 0.5 * (z[1:] + z[:-1])
        bias = np.array([np.nan if b is None else b for b in low["z_bias_nm"]], float)
        ax.flat[5].plot(mid, bias, marker=".", color=colour, ls=ls, label=label)
    ax.flat[5].axhline(0, color="0.6", lw=0.8)
    ax.flat[5].set(xlabel="true z (nm)", ylabel="mean z error (nm)", title=f"z bias at density {rows[0]['density']}")
    for a_ in ax.flat:
        a_.grid(alpha=0.3)
    ax.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "scores.png", dpi=100)
    print("figure:", root / "scores.png")


if __name__ == "__main__":
    main()
