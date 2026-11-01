"""The consensus of `m5_consensus.py` against each acquisition's own minimal model, as figures.

    python benchmarks/uipsf/real/m5_consensus_figure.py OUT_DIR [--blur fixed] [--form height]

Reads `results/m5_consensus.jsonl`.  For one blur (the user's minimal model holds σ at the pixel's: "fixed")
and one ring form, three figures:

- ``consensus_values.png``: each acquisition's own γ and m ≠ 0 magnitude modes with their CRLBs, and the
  consensus values (median, weighted mean, joint estimate) each acquisition was held at -- the ones from the
  other acquisitions, leave one out, so they differ a little from row to row;
- ``consensus_cost.png``: what holding them costs against the acquisition's own fit, in nats, per estimator;
- ``consensus_positions.png``: the z bend over ±200 and ±600 nm and the x / y bias, own fit against consensus.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "m5_consensus.jsonl")
ESTIMATORS = {"median": "#2a78d6", "weighted": "#eb6834", "joint": "#1baf7a"}
MODES = ["2", "3", "5", "6", "8", "16"]


def short(name: str) -> str:
    return (name.replace("20261029_", "").replace("_50ms", "").removesuffix("_1").replace("mismatch", "buffer")
            .replace("match", "oil"))


def order(name: str) -> tuple:
    bead = 40 if name.startswith("40nm") else 100
    return (bead, "676" in name, "3D" in name, "0.1um" in name, "mismatch" in name)


def bend(r, lim, axis=2):
    z = np.asarray(r["curve_z_nm"], float)
    c = np.asarray(r["curve_nm"], float)[:, axis]
    ok = (np.abs(z) <= lim) & np.isfinite(c)
    return float(np.sqrt(np.mean(c[ok] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--blur", default="fixed", choices=("fixed", "free", "zero"))
    ap.add_argument("--form", default="height")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(line) for line in open(RESULTS)]
    names = sorted({r["acquisition"] for r in rows}, key=order)

    def own(n):
        return next(r for r in rows if r["acquisition"] == n and r["stage"] == "free" and r.get("blur") == args.blur
                    and r["form"] in (args.form, "none"))

    def held(n, e):
        return next((r for r in rows if r["acquisition"] == n and r["stage"] == "consensus"
                     and r.get("blur") == args.blur and r["form"] == args.form and r["estimator"] == e), None)

    blur = {"fixed": "σ = pixel", "free": "σ free", "zero": "σ = 0"}[args.blur]
    y = np.arange(len(names))

    # 1. the values
    fig, axes = plt.subplots(1, 1 + len(MODES), figsize=(2.0 * (1 + len(MODES)), 0.32 * len(names) + 1.6),
                             sharey=True)
    for ax, key in zip(axes, ["g"] + MODES, strict=True):
        v = [own(n)["edge"]["centre_g"] if key == "g" else own(n)["magnitude"][key] for n in names]
        e = [own(n)["edge_crlb"].get("centre_g", 0) if key == "g" else own(n)["magnitude_crlb"].get(key, 0)
             for n in names]
        ax.errorbar(v, y, xerr=e, fmt="o", color="#222222", ms=4, lw=1, label="own fit")
        for est, colour in ESTIMATORS.items():
            h = [held(n, est) for n in names]
            hv = [(x["held_gamma"] if key == "g" else x["held_magnitude"][key]) if x else np.nan for x in h]
            ax.plot(hv, y, "|", color=colour, ms=10, mew=2, label=est)
        ax.set_title("γ (cos θ exponent)" if key == "g" else f"Noll {key}", fontsize=8)
        ax.axvline(0, color="#bbbbbb", lw=0.6)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#eeeeee")
    axes[0].set_yticks(y, [short(n) for n in names], fontsize=7)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle(f"the minimal model's objective numbers ({blur}): each acquisition's own fit (black, ± CRLB) and "
                 "the consensus of the other acquisitions it was held at", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "consensus_values.png"), dpi=140)
    plt.close(fig)

    # 2. the cost
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(names) + 1.4))
    w = 0.26
    for i, (est, colour) in enumerate(ESTIMATORS.items()):
        cost = [(held(n, est)["objective"] - own(n)["objective"]) / 1000 if held(n, est) else np.nan for n in names]
        ax.barh(y + (i - 1) * w, cost, height=w, color=colour, label=est)
    ax.set_yticks(y, [short(n) for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("objective held at the consensus − own fit (k nats)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"what holding γ and the six m ≠ 0 modes costs ({blur})", fontsize=8.5)
    ax.grid(axis="x", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "consensus_cost.png"), dpi=140)
    plt.close(fig)

    # 3. the positions
    measures = [("z bend ±200 nm", lambda r: bend(r, 200)), ("z bend ±600 nm", lambda r: bend(r, 600)),
                ("x bias ±600 nm", lambda r: bend(r, 600, 0)), ("y bias ±600 nm", lambda r: bend(r, 600, 1))]
    fig, axes = plt.subplots(1, len(measures), figsize=(3.0 * len(measures), 0.32 * len(names) + 1.6), sharey=True)
    for ax, (label, f) in zip(axes, measures, strict=True):
        a = [f(own(n)) for n in names]
        b = [f(held(n, "joint")) if held(n, "joint") else np.nan for n in names]
        for yi, ai, bi in zip(y, a, b, strict=True):
            ax.plot([ai, bi], [yi, yi], color="#bbbbbb", lw=1)
        ax.plot(a, y, "o", color="#222222", ms=4, label="own fit")
        ax.plot(b, y, "o", color=ESTIMATORS["joint"], ms=4, label="held at the joint consensus")
        ax.set_title(f"{label} (rms, nm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#eeeeee")
        ax.set_xlim(left=0)
    axes[0].set_yticks(y, [short(n) for n in names], fontsize=7)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle(f"positions: the minimal model's own fit against the consensus ({blur})", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "consensus_positions.png"), dpi=140)
    plt.close(fig)
    print(os.path.join(args.out, "consensus_values.png"))


if __name__ == "__main__":
    main()
