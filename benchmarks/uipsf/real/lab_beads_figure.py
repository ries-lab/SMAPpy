"""The figure of `lab_beads.py`: the learnt blur per acquisition, coloured by microscope, and the free pupil
magnitudes side by side.

    python benchmarks/uipsf/real/lab_beads_figure.py [benchmarks/uipsf/results/lab_beads.jsonl] [-o lab_beads.png]

Left: σx and σy with the pixel's share taken out (√(σ² − p²/12)), ± CRLB, one column per acquisition; the
learnt value, pixel included, as a faint mark beside it.  Right: each acquisition's magnitude map,
1 + Σ cⱼ Zⱼ over the pupil (the apodisation and the aperture's own fall-off are the model's and not drawn),
on one colour scale centred on 1 (its range the 98th percentile of |m − 1|; the rim's swings saturate).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uipsf.optics.zernike import zernike

COLOURS = {"M2": "#2a78d6", "M5": "#eb6834"}     # categorical slots 1 and 2


def magnitude_map(row, n=129):
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    m = 1 + sum(c * zernike(j, rho, phi) for j, c in zip(row["magnitude_modes"], row["magnitude"], strict=True))
    return np.where(rho <= 1, m, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?", default="benchmarks/uipsf/results/lab_beads.jsonl")
    ap.add_argument("-o", "--out", default="lab_beads.png")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.jsonl)]
    rows.sort(key=lambda r: (r["microscope"], r["acquisition"]))
    maps = [magnitude_map(r) for r in rows]
    # the scale from the bulk of the pupil: an order-8 fit swings at the very rim, where little light is
    span = float(np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in maps]) - 1), 98))
    n = len(rows)
    fig = plt.figure(figsize=(12, 4.6), layout="constrained")
    sub = fig.subfigures(1, 2, width_ratios=[1, 1.35])
    ax = sub[0].subplots()
    for i, r in enumerate(rows):
        c = COLOURS.get(r["microscope"], "#777777")
        for k, (marker, dx) in enumerate((("o", -0.13), ("s", 0.13))):
            ax.errorbar(i + dx, r["sigma_optics_nm"][k], yerr=r["sigma_crlb_nm"][k], fmt=marker, ms=7, color=c,
                        mec="white", mew=1.5, elinewidth=2, capsize=0)
            ax.plot(i + dx, r["sigma_nm"][k], marker, ms=7, mfc="none", mec=c, alpha=0.35)
    ax.set_xticks(range(n), [f"{r['microscope']} {r['acquisition'][:6]}" for r in rows], rotation=60, ha="right",
                  fontsize=8)
    ax.set_ylabel("extra blur σ (nm)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot([], [], "o", color="#555", label="σx, pixel taken out")
    ax.plot([], [], "s", color="#555", label="σy, pixel taken out")
    ax.plot([], [], "o", mfc="none", mec="#555", alpha=0.5, label="as learnt")
    for name, c in COLOURS.items():
        ax.plot([], [], lw=6, color=c, label=name)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    cols = int(np.ceil(n / 2))
    axes = sub[1].subplots(2, cols, squeeze=False)
    for a in axes.ravel():
        a.set_axis_off()
    for a, r, m in zip(axes.ravel(), rows, maps, strict=False):
        im = a.imshow(m, cmap="BrBG", vmin=1 - span, vmax=1 + span, origin="lower")
        a.set_title(f"{r['microscope']} {r['acquisition'][:6]}", fontsize=8, color=COLOURS.get(r["microscope"]))
    sub[1].colorbar(im, ax=axes, shrink=0.7, label="pupil magnitude", extend="both")
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
