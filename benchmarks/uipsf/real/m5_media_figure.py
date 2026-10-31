"""Figures of study A on M5's beads (`m5_media.py`): radial magnitude, phase in nm, likelihoods, σ, z bends.

    python benchmarks/uipsf/real/m5_media_figure.py OUT_DIR

Reads `results/m5_media_magnitude.jsonl` (six magnitude forms, ±1 µm at 100 nm slices) and
`results/m5_media_window.jsonl` (the z window on the 676 nm sets) and writes PNGs into OUT_DIR.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
ORDER = [(c, m, l) for c in ("600", "676") for m in ("match", "mismatch") for l in ("2D", "3D")]
MEDIUM = {"match": "oil", "mismatch": "buffer"}
# the reference palette's first slots, in fixed order (dataviz skill)
SERIES = {"zernike": "#2a78d6", "cos bead na asym": "#eb6834", "cos steps bead na asym": "#1baf7a",
          "cos signed bead na asym": "#6250d6"}
LABEL = {"zernike": "free Zernikes (m = 0 part)", "cos bead na asym": "cos θ^γ × ring × cutoff",
         "cos steps bead na asym": "… × monotone correction", "cos signed bead na asym": "… × two-way correction"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
N_MED, NA = {"match": 1.518, "mismatch": 1.334}, 1.50
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def load(name):
    path = os.path.join(RESULTS, name)
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def key(r):
    return (r["colour"], r["medium"], r["lens"])


def title(k):
    return f"{k[0]} nm, {MEDIUM[k[1]]}, {k[2]}"


def zernike_profile(r, rho):
    """1 + Σ c_j Z_j over the m = 0 modes along a radius (`ZernikePupil`), relative to the centre."""
    from uipsf.optics.zernike import noll_to_nm, zernike
    m = 1 + sum(c * zernike(int(j), rho, np.zeros_like(rho)) for j, c in r["magnitude"].items()
                if noll_to_nm(int(j))[1] == 0)
    return m / m[0]


def radial(rows, out):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True, sharey=True)
    rho = np.linspace(0, 1, 201)
    for ax, k in zip(axes.T.ravel(), ORDER, strict=True):
        if k[1] == "mismatch":
            ax.axvline(N_MED[k[1]] / NA, color=MUTED, lw=0.8, ls=":")
        for v, colour in SERIES.items():
            r = next((r for r in rows if key(r) == k and r["variant"] == v), None)
            if r is None:
                continue
            if v == "zernike":
                ax.plot(rho, zernike_profile(r, rho), color=colour, lw=2, label=LABEL[v])
            else:
                ax.plot(np.linspace(0, 1, 21), r["profile"], color=colour, lw=2, label=LABEL[v])
        ax.set_title(title(k), fontsize=9, color=INK)
        ax.set_ylim(0, 1.25)
    for ax in axes[-1]:
        ax.set_xlabel("pupil radius ρ (NA 1.50 = 1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("magnitude / centre")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.text(0.5, 0.005, "dotted: the critical angle in buffer, ρ = 1.334/1.50; the ring term starts there",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(out, "radial.png"), dpi=150)


def heat(table, rows_labels, cols_labels, out_png, unit, vmax=None, fmt="{:.0f}", cmap="RdBu_r", note=""):
    a = np.array(table, float)
    vmax = vmax or np.nanmax(np.abs(a))
    fig, ax = plt.subplots(figsize=(2.4 + 1.15 * len(cols_labels), 0.5 + 0.38 * len(rows_labels)))
    ax.imshow(a, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(a):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(cols_labels)), cols_labels)
    ax.set_yticks(range(len(rows_labels)), rows_labels)
    ax.tick_params(length=0)
    ax.set_title(f"{unit}{note}", fontsize=9, color=MUTED, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def phases(rows, out):
    modes = [5, 6, 7, 8, 11, 12, 13, 22, 37]
    z = {key(r): r for r in rows if r["variant"] == "zernike"}
    table = [[z[k]["phase_nm_path"][str(m)] for m in modes] for k in ORDER if k in z]
    names = {5: "5\nastig 45°", 6: "6\nastig 0°", 7: "7\ncoma", 8: "8\ncoma", 11: "11\nspherical",
             12: "12\nastig 2", 13: "13\nastig 2", 22: "22\nspherical 2", 37: "37\nspherical 3"}
    heat(table, [title(k) for k in ORDER if k in z], [names[m] for m in modes], os.path.join(out, "phase.png"),
         "phase aberration, nm of optical path (Noll index)", vmax=60)


def likelihoods(rows, out):
    """Two tables of Δ nats: does the m ≠ 0 magnitude matter, and does a free correction on cos θ^γ help."""
    def delta(k, v, base):
        a = next((r for r in rows if key(r) == k and r["variant"] == v), None)
        b = next((r for r in rows if key(r) == k and r["variant"] == base), None)
        return a["objective"] - b["objective"] if a and b else np.nan
    names = [title(k) for k in ORDER]
    table = [[delta(k, "symmetric", "zernike"), delta(k, "cos bead na", "cos bead na asym")] for k in ORDER]
    heat(table, names, ["Zernike:\nm = 0 only", "cos θ^γ:\nno m ≠ 0"], os.path.join(out, "nats_asym.png"),
         "Δ nats when the m ≠ 0 magnitude is dropped", vmax=300000, fmt="{:+.0f}")
    table = [[delta(k, "cos bead na asym", "zernike"), delta(k, "cos steps bead na asym", "cos bead na asym"),
              delta(k, "cos signed bead na asym", "cos bead na asym")] for k in ORDER]
    heat(table, names, ["cos θ^γ\nvs Zernikes", "+ monotone\nvs cos θ^γ", "+ two-way\nvs cos θ^γ"],
         os.path.join(out, "nats_corr.png"), "Δ nats (negative: the first is better)", vmax=20000, fmt="{:+.0f}")


def sigma(rows, out):
    variants = ["zernike", "symmetric", "cos bead na asym", "cos signed bead na asym"]
    table = [[(lambda r: f"{r['sigma_optics_nm'][0]:.0f}, {r['sigma_optics_nm'][1]:.0f}")(r) if r else ""
              for r in [next((r for r in rows if key(r) == k and r["variant"] == v), None) for v in variants]]
             for k in ORDER]
    return table


def bends(rows, window, out):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, k in zip(axes.T.ravel(), ORDER, strict=True):
        for v, colour in SERIES.items():
            r = next((r for r in rows if key(r) == k and r["variant"] == v), None)
            if r:
                z, c = np.array(r["curve_z_nm"], float), np.array(r["curve_nm"], float)[:, 2]
                ax.plot(z, c, color=colour, lw=2, label=LABEL[v])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title(k), fontsize=9, color=INK)
        ax.set_ylim(-45, 45)
    for ax in axes[-1]:
        ax.set_xlabel("z from the bead's focus (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("slice z − stack z (nm)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bend_magnitude.png"), dpi=150)

    if not window:
        return
    sets = [k for k in ORDER if k[0] == "676"]
    shades = {500: "#2a78d6", 1000: "#eb6834", 2000: "#1baf7a"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
    for ax, k in zip(axes, sets, strict=True):
        for w, colour in shades.items():
            r = next((r for r in window if key(r) == k and r["window_nm"] == w), None)
            if r:
                z, c = np.array(r["curve_z_nm"], float), np.array(r["curve_nm"], float)[:, 2]
                ax.plot(z, c, color=colour, lw=2, label=f"learnt on ±{w / 1000:g} µm")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title(k), fontsize=9, color=INK)
        ax.set_xlabel("z from the bead's focus (nm)")
        ax.set_ylim(-35, 35)
    axes[0].set_ylabel("slice z − stack z (nm)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bend_window.png"), dpi=150)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rows = load("m5_media_magnitude.jsonl")
    window = [r for r in load("m5_media_window.jsonl") if r["dz_nm"] == 100 or r["window_nm"] == 500]
    radial(rows, out)
    phases(rows, out)
    likelihoods(rows, out)
    bends(rows, window, out)
    json.dump({"sigma": sigma(rows, out)}, open(os.path.join(out, "tables.json"), "w"))


if __name__ == "__main__":
    main()
