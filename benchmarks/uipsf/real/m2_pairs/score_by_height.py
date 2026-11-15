"""Where in z does each pupil fit the cell better?  Models' scores on the same molecules, per height bin.

    python score_by_height.py OUTDIR REFERENCE MODEL [MODEL ...] [--bins -200,0,200,...] [--json FILE]

`cross_score.py` found that on the nucleus movie's 223 *held out* molecules the in situ pupils beat the stack's
(held, the focus learnt: ``insitu_all_n1345_ab_n1345``) by 1.75-2.75 nats per molecule, while on the 4105
*not drawn* -- the z bins' surplus, crowded where molecules are plentiful -- they lose by 0.36-0.50.  Which
heights the surplus crowds at, and so where the stack's pupil is the better one, was not checked.  This script
does: an excess the in situ pupils carry because of depth (the index, the nucleus) would win deep and lose near
the glass; one from something else would not depend on height.

Every MODEL's ½ Σ deviance per molecule (``heights_<MODEL>.npz`` from `heights.py`), each molecule placed in a
height bin by REFERENCE's height (so a bin holds the same molecules for every model), and per bin the mean
difference against the first MODEL, with its standard error and count, for the *held out* and *not drawn*
molecules apart and together.  The *used* are left out: the in situ models learnt from them.

Negative: the model fits that bin's molecules better than the first MODEL (nats per molecule; 1 ≈ moving each
molecule by 1.4 CRLB).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heights import load  # noqa: E402

BINS_NM = (-200.0, 0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1600.0)
UNSEEN = ("held out", "not drawn")


def by_bin(height, delta, edges):
    """Per bin of ``height``: (count, mean of ``delta``, its standard error)."""
    idx = np.digitize(height, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        d = delta[(idx == b) & np.isfinite(delta)]
        out.append((len(d), float(d.mean()) if len(d) else float("nan"),
                    float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("reference", help="the model whose heights place the molecules in bins")
    ap.add_argument("names", nargs="+", help="models to compare; differences against the first")
    ap.add_argument("--bins", default=",".join(f"{b:g}" for b in BINS_NM))
    ap.add_argument("--json")
    a = ap.parse_args()
    edges = np.array([float(b) for b in a.bins.split(",")])
    ref = load(a.out, a.reference)
    data = {n: load(a.out, n) for n in a.names}
    for n, d in data.items():
        if not np.array_equal(d["units"], ref["units"]):
            raise SystemExit(f"{n} holds other units than {a.reference}: run heights.py with the same --states")
    state, height = ref["state"], ref["h"]
    base = data[a.names[0]]["half_deviance"]
    labels = [f"{edges[b]:g}..{edges[b + 1]:g}" for b in range(len(edges) - 1)]
    for n in a.names[1:]:
        delta = data[n]["half_deviance"] - base
        print(f"\n{n} - {a.names[0]}, nats per molecule by {a.reference}'s height (nm):")
        print(f"   {'height':>12s}  " + "  ".join(f"{s + ' (n)':>22s}" for s in (*UNSEEN, "both")))
        parts = {s: by_bin(height[state == s], delta[state == s], edges) for s in UNSEEN}
        unseen = np.isin(state, UNSEEN)
        parts["both"] = by_bin(height[unseen], delta[unseen], edges)
        for b, label in enumerate(labels):
            cells = []
            for s in (*UNSEEN, "both"):
                k, m, se = parts[s][b]
                cells.append(f"{m:+7.2f} ± {se:4.2f} ({k:5d})" if k else f"{'-':>22s}")
            print(f"   {label:>12s}  " + "  ".join(cells))
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(dict(kind="score_by_height", model=n, against=a.names[0],
                                        reference=a.reference, bins=edges.tolist(),
                                        **{s.replace(" ", "_"): parts[s] for s in parts})) + "\n")


if __name__ == "__main__":
    main()
