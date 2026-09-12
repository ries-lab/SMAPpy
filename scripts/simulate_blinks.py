#!/usr/bin/env python3
"""Write the simulated blinking datasets the examples and tests use.

    python scripts/simulate_blinks.py [--out DIR] [--frames 20000] [--seed 0]

The simulation itself is `smappy.simulate`, which is also the
``File/Simulate/Blinking Structure`` plugin; this only drives it and saves.

Emitters sit on a 3D structure (a tilted ring, two crossing lines, a few
scattered points) spread over ~10 um, far apart compared with a PSF.  Each
one blinks a few times; a blink lasts one to a few frames and every frame
gives one localization with noise from its photon count (precision ~
150 / sqrt(N) nm laterally, three times that in z).  Two emitters that are
active in the same frame closer than ``--min-separation`` (a PSF width)
could not have been fitted apart, so both are dropped -- the labelling is
dense, the activation sparse, as in a real experiment.  The drifted copy adds a
smooth random walk plus a slow linear creep of ~100 nm; the true drift per
frame is stored in the file's metadata as ``drift_truth`` (x, y, z in nm).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smappy.io.hdf5 import save_localizations  # noqa: E402
from smappy.simulate import simulate            # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".", help="directory (default: here)")
    p.add_argument("--frames", type=int, default=20000)
    p.add_argument("--density", type=float, default=1.0,
                   help="scale the number of blinks per emitter")
    p.add_argument("--min-separation", type=float, default=250.0,
                   help="two emitters closer than this in one frame are both dropped (nm)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    for drift, name in ((False, "sim_blinks.hdf5"), (True, "sim_blinks_drift.hdf5")):
        locs = simulate(args.frames, args.seed, drift, args.density, args.min_separation)
        save_localizations(out / name, locs)
        per_frame = len(locs) / args.frames
        print(f"{name}: {len(locs)} localizations, {locs.metadata['n_emitters']} emitters, "
              f"{args.frames} frames, {per_frame:.1f} per frame"
              + f", {locs.metadata['n_unresolvable_dropped']} overlapping dropped"
              + (", drift ~150 nm" if drift else ""))


if __name__ == "__main__":
    main()
