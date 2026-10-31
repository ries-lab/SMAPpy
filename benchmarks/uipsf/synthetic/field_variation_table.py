"""The table of `field_variation.py`'s results: σ beyond the truth against δ and the field, and the rest.

    python benchmarks/uipsf/synthetic/field_variation_table.py [benchmarks/uipsf/results/field_variation.jsonl]

One line per truth and fit (the scatter's seeds as mean ± sd): σ beyond the truth in quadrature (x, y, nm),
the phase bias's rms over all modes (nm), the bends' rms in z over ±200 and ±600 nm, and the residual at the
peak pixel at z = −300, 0, +300 nm (% of the peak).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():6.2f}" if len(v) == 1 else f"{v.mean():6.2f}±{v.std(ddof=1):4.2f}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "field_variation.jsonl")
    groups = defaultdict(list)
    for r in map(json.loads, open(path)):
        fit = f"{r['field']}{' σ held' if r['sigma_held'] else ''}"
        groups[(r["truth_sigma"], r["case"], r["delta_nm"], fit)].append(r)
    print(f"{'truth σ':8} {'case':8} {'δ nm':>5} {'fit':15} {'n':>2} {'σx+':>12} {'σy+':>12} {'phase':>11} "
          f"{'z200':>11} {'z600':>11} {'pk-300 %':>11} {'pk0 %':>11} {'pk+300 %':>11}")
    for (ts, case, d, fit), rows in groups.items():
        col = lambda f, rows=rows: fmt([f(r) for r in rows])  # noqa: E731
        print(f"{ts:8} {case:8} {d:5.0f} {fit:15} {len(rows):2} "
              f"{col(lambda r: r['sigma_excess_quad_nm'][0]):>12} {col(lambda r: r['sigma_excess_quad_nm'][1]):>12} "
              f"{col(lambda r: r['phase']['rms_nm']):>11} {col(lambda r: r['bends']['rms_200_nm'][2]):>11} "
              f"{col(lambda r: r['bends']['rms_600_nm'][2]):>11} "
              + " ".join(f"{col(lambda r, z=z: r['peak'][z]['peak_pct']):>11}" for z in ("-300", "+0", "+300")))


if __name__ == "__main__":
    main()
