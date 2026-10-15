"""Link tracks: the simple linker, the global assignment and trackpy, on
simulated tracks over density and jump size.

    python studies/tracking/bench.py            # writes bench.csv beside it
    PYTHONPATH=/path/to/trackpy python studies/tracking/bench.py

trackpy is not a dependency of smappy: it is benchmarked when it imports and
left out when it does not (its 0.7 sdist does not build on Python 3.13; the
source unpacked on PYTHONPATH works).  It is given the same maximum jump and
memory, its own default linking strategy and no adaptive search; a
subnetwork too large for it (`SubnetOversizeException`) is recorded as a
failure, which is what a user would see.

The grid: densities 0.02 to 1 particles per um^2 (nearest neighbour 3.5 um
to 0.5 um), root-mean-square jumps 50 to 400 nm per frame, blinking with
p_dark 0.1 and dark runs of 1.5 frames on average (so memory 3 bridges
nearly all of them), 20 nm precision, the maximum jump three times the
root-mean-square jump as it is measured -- with the precision of both ends,
``sqrt(jump^2 + 4 sigma^2)`` -- as the page tells a user to set it.  `score` is `smappy.tracking.link.score`.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from smappy.simulate import jump_nm, simulate_tracks, spacing_nm
from smappy.tracking import link_tracks, score

HERE = Path(__file__).resolve().parent
DENSITIES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
JUMPS = (50.0, 100.0, 200.0, 400.0)
FRAME_MS = 10.0
MEMORY = 3
PRECISION = 20.0


def d_for(jump: float) -> float:
    """The diffusion coefficient (um^2/s) whose 2D rms jump is ``jump`` nm."""
    return jump ** 2 / (4 * FRAME_MS * 1e-3) / 1e6


def trackpy_ids(locs, max_jump: float, memory: int):
    import pandas as pd
    import trackpy as tp
    tp.quiet()
    frame = pd.DataFrame({"x": np.asarray(locs["x_nm"], float),
                          "y": np.asarray(locs["y_nm"], float),
                          "frame": np.asarray(locs["frame"])})
    out = tp.link(frame, max_jump, memory=memory)
    particle = out["particle"].to_numpy()
    # trackpy keeps lone localizations as particles of their own: 0, as here
    sizes = np.bincount(particle)
    return np.where(sizes[particle] >= 2, particle + 1, 0)


def main(argv=None):
    try:
        import trackpy  # noqa: F401
        have_trackpy = True
    except ImportError:
        have_trackpy = False
        print("trackpy not importable: benchmarking the two linkers only")
    rows = []
    for density in DENSITIES:
        for jump in JUMPS:
            locs = simulate_tracks(n_frames=300, field_um=20.0, density_um2=density,
                                   d_um2_s=(d_for(jump),), track_frames=20.0,
                                   frame_time_ms=FRAME_MS, p_dark=0.1, dark_frames=1.5,
                                   precision_nm=PRECISION, seed=7)
            max_jump = 3.0 * np.hypot(jump, 2 * PRECISION)
            methods = ["simple", "global"] + (["trackpy"] if have_trackpy else [])
            for method in methods:
                t0 = time.perf_counter()
                try:
                    if method == "trackpy":
                        ids = trackpy_ids(locs, max_jump, MEMORY)
                    else:
                        ids, _ = link_tracks(locs["x_nm"], locs["y_nm"], locs["frame"],
                                             max_jump_nm=max_jump, memory=MEMORY,
                                             algorithm=method)
                    seconds = time.perf_counter() - t0
                    found = score(ids, locs["track_id_true"], locs["frame"])
                except Exception as error:                 # noqa: BLE001
                    seconds = time.perf_counter() - t0
                    print(f"  {method} failed: {type(error).__name__}")
                    found = {"links_found": float("nan"), "links_wrong": float("nan"),
                             "tracks_whole": float("nan")}
                row = {"density_um2": density, "jump_nm": jump,
                       "jump_over_spacing": round(jump / spacing_nm(density), 3),
                       "method": method, "n_locs": len(locs),
                       "links_found": round(found["links_found"], 4),
                       "links_wrong": round(found["links_wrong"], 4),
                       "tracks_whole": round(found["tracks_whole"], 4),
                       "seconds": round(seconds, 4),
                       "us_per_loc": round(1e6 * seconds / max(len(locs), 1), 2)}
                rows.append(row)
                print(row, flush=True)
    with open(HERE / "bench.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # speed on a large table: a million localizations, sparse
    locs = simulate_tracks(n_frames=4000, field_um=50.0, density_um2=0.1,
                           d_um2_s=(d_for(100.0),), p_dark=0.1, dark_frames=1.5, seed=3)
    print(f"\n{len(locs):,} localizations, 4000 frames:")
    for method in ["simple", "global"] + (["trackpy"] if have_trackpy else []):
        t0 = time.perf_counter()
        if method == "trackpy":
            trackpy_ids(locs, 300.0, MEMORY)
        else:
            link_tracks(locs["x_nm"], locs["y_nm"], locs["frame"], max_jump_nm=300.0,
                        memory=MEMORY, algorithm=method)
        print(f"  {method}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
