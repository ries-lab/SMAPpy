#!/usr/bin/env python3
"""Chunked parallel linking against the sequential walk, which is the control.

    python scripts/check_chunked_grouping.py                       # simulated
    python scripts/check_chunked_grouping.py --file RUN_sml.mat    # real data
    python scripts/check_chunked_grouping.py --chunks 2 4 8 16

`smappy._group_chunked.connect_chunked` cuts the frame axis, links the pieces in
parallel and repairs the seams.  Cutting a trace is a real error, so the
question is not whether the ids match -- they are arbitrary -- but whether the
*partition* does.  What is reported per chunk count:

``groups``      how many groups, against the sequential number.
``identical``   the fraction of localizations whose group is exactly the same
                set of localizations in both.  This is the number that matters:
                anything less than 1 means some traces were grouped differently.
``seams``       traces rejoined across the cuts, and how many groups touch a
                seam at all -- the population the error can come from.
``speedup``     wall clock against the sequential walk on the same data.

The simulated data is deliberately harsher than a real acquisition: emitters
blink for several frames at a density high enough that traces compete for the
same localizations, which is what the seam repair can get wrong.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smappy.group import connect                                   # noqa: E402
from smappy._group_chunked import connect_chunked                  # noqa: E402


def simulate(n_emitters=40_000, frames=4000, mean_on=3.0, blinks=4,
             field=20_000.0, precision=12.0, seed=0):
    """Blinking emitters: each is on for a run of consecutive frames, several
    times, and gives one localization per frame with noise.  Returns x, y, frame
    and the true emitter/blink each localization came from."""
    rng = np.random.default_rng(seed)
    ex = rng.uniform(0, field, n_emitters)
    ey = rng.uniform(0, field, n_emitters)
    xs, ys, fs, tid = [], [], [], []
    trace = 0
    for _ in range(blinks):
        start = rng.integers(0, frames, n_emitters)
        length = 1 + rng.geometric(1.0 / mean_on, n_emitters)
        for e in range(n_emitters):
            n = int(min(length[e], frames - start[e]))
            if n <= 0:
                continue
            f = np.arange(start[e], start[e] + n)
            xs.append(rng.normal(ex[e], precision, n))
            ys.append(rng.normal(ey[e], precision, n))
            fs.append(f)
            tid.append(np.full(n, trace))
            trace += 1
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    f = np.concatenate(fs).astype(np.int64)
    return x, y, f, np.concatenate(tid)


def load(path):
    from smappy.io.formats import load as load_any
    locs, _ = load_any(path)
    x = np.asarray(locs["x_nm"], np.float64)
    y = np.asarray(locs["y_nm"], np.float64)
    f = np.asarray(locs["frame"], np.int64)
    blocks = None
    if "channel" in locs:
        blocks = np.asarray(locs["channel"])[:, None]
    return x, y, f, blocks


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of localizations whose group is the same *set* under both.

    Pair up the two labellings and count a localization as agreeing when its
    (a, b) pair is as big as its group is on either side -- that is, when the
    two groups it belongs to contain exactly the same localizations.
    """
    n = len(a)
    pair_key = a.astype(np.int64) * (int(b.max()) + 1) + b
    _, pair_inv, pair_n = np.unique(pair_key, return_inverse=True, return_counts=True)
    a_n = np.bincount(a)[a]
    b_n = np.bincount(b)[b]
    same = (pair_n[pair_inv] == a_n) & (pair_n[pair_inv] == b_n)
    return float(same.sum()) / n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="a localization file instead of simulated data")
    p.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--dx", type=float, default=50.0)
    p.add_argument("--dt", type=int, default=1)
    p.add_argument("--emitters", type=int, default=40_000)
    p.add_argument("--frames", type=int, default=4000)
    p.add_argument("--mean-on", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    blocks = None
    if a.file:
        print(f"reading {Path(a.file).name} ...", flush=True)
        x, y, f, blocks = load(a.file)
        truth = None
    else:
        x, y, f, truth = simulate(a.emitters, a.frames, a.mean_on, seed=a.seed)
    print(f"{len(x):,} localizations over {int(f.max()) - int(f.min()) + 1:,} frames"
          f"   dx={a.dx} dt={a.dt}", flush=True)

    t0 = time.time()
    control = connect(x, y, f, a.dx, a.dt, blocks)
    t_seq = time.time() - t0
    n_seq = int(control.max())
    print(f"\nsequential: {n_seq:,} groups in {t_seq:.2f} s", flush=True)
    if truth is not None:
        print(f"            {agreement(control, truth.astype(np.int64) + 1):.4f} "
              f"of localizations grouped as the simulation made them")

    print(f"\n{'chunks':>7} {'groups':>12} {'vs seq':>9} {'identical':>10} "
          f"{'seams':>18} {'time':>8} {'speedup':>8}")
    for n_chunks in a.chunks:
        stats = {}
        t0 = time.time()
        ids = connect_chunked(x, y, f, a.dx, a.dt, blocks, n_chunks=n_chunks, stats=stats)
        t = time.time() - t0
        n = int(ids.max())
        seams = f"{stats['rejoined']:,}/{stats['at_seams']:,}"
        print(f"{n_chunks:>7} {n:>12,} {n - n_seq:>+9,} {agreement(control, ids):>10.5f} "
              f"{seams:>18} {t:>7.2f}s {t_seq / t:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
