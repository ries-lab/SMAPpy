"""Linking two mobilities at once: how many links go wrong, and what the
wrong ones do to the analyses that read the tracks (survey item 5).

    python studies/tracking/two_states.py       # writes two_states.csv beside it

A slow state (0.02 um^2/s, 28 nm rms jump per 10 ms frame) and a fast one
(1 um^2/s, 200 nm), half the time each, 20 nm precision, blinking as in
`bench.py` (p_dark 0.1, dark runs of 1.5 frames, memory 3), tracks of 30
frames on average; without switching and with a switch every 40 frames
(2.5 /s each way).  Each density is linked three ways:

* ``truth``: the true ids -- what a perfect linker would give;
* ``global``: the global assignment with R for the fast state, three times
  its rms jump with both ends' precision, as the page tells a user;
* ``simple``: the same R, nearest first;
* ``global_slow``: R for the slow state, the other horn of the dilemma:
  fast tracks are cut.

and scored as `bench.py` does, plus the share of wrong links that start in
the slow state, and the jump fit (2 states), the D spectrum's peaks and the
HMM's rates on the result.  The gap between ``truth`` and the linkers bounds
what any better linking cost (item 5) could gain.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from smappy.simulate import simulate_tracks, spacing_nm
from smappy.tracking import link_tracks, score
from smappy.tracking.analysis import fit_jumps, gather, jumps
from smappy.tracking.hmm import fit_hmm
from smappy.tracking.link import true_links
from smappy.tracking.spectrum import peaks, spectrum

HERE = Path(__file__).resolve().parent
DENSITIES = (0.05, 0.1, 0.2, 0.5, 1.0)
D = (0.02, 1.0)
FRAME_MS = 10.0
MEMORY = 3
PRECISION = 20.0
SWITCH = (0.0, 0.05)        # redraw probability: leaving a state at 0.025 /frame


def rms_jump(d):
    return np.sqrt(4 * d * 1e6 * FRAME_MS * 1e-3)


def radius(d):
    return 3.0 * np.hypot(rms_jump(d), 2 * PRECISION)


def wrong_from_slow(ids, locs):
    """Of the wrong links, the fraction whose first localization is slow,
    and the wrong links per slow link found."""
    frame = np.asarray(locs["frame"])
    truth = np.asarray(locs["track_id_true"], np.int64)
    state = np.asarray(locs["state_true"])
    n = len(ids)
    ta, tb = true_links(truth, frame)
    fa, fb = true_links(ids, frame)
    wrong = ~np.isin(fa.astype(np.int64) * n + fb, ta.astype(np.int64) * n + tb)
    slow = state[fa] == 0
    return (float(slow[wrong].mean()) if wrong.any() else float("nan"),
            float(wrong[slow].mean()) if slow.any() else float("nan"))


def analyse(ids, locs):
    t = gather(ids, locs["frame"], locs["x_nm"], locs["y_nm"],
               err_xy=locs["xy_err_nm"], frame_time_s=FRAME_MS * 1e-3)
    d2, err = jumps(t, 1)
    jf = fit_jumps(d2, err, 2, FRAME_MS * 1e-3, states=2)
    sp = spectrum(t)
    pk = peaks(sp)
    slow_peak = sum(f for d, f in pk if d < 0.15)
    out = {"jump_D_slow": jf.D_um2_per_s[0], "jump_D_fast": jf.D_um2_per_s[-1],
           "jump_f_slow": jf.fraction[0],
           "spec_n_peaks": len(pk), "spec_f_slow": slow_peak}
    hm = fit_hmm(t, states=2)
    Q = hm.params.Q_per_s
    out.update({"hmm_D_slow": hm.params.D_um2_per_s[0],
                "hmm_D_fast": hm.params.D_um2_per_s[1],
                "hmm_k_slow_fast": Q[0, 1], "hmm_k_fast_slow": Q[1, 0],
                "hmm_f_slow": hm.params.stationary()[0]})
    return out


def main(argv=None):
    rows = []
    for switch in SWITCH:
        for density in DENSITIES:
            locs = simulate_tracks(n_frames=300, field_um=20.0, density_um2=density,
                                   d_um2_s=D, fractions=(0.5, 0.5),
                                   switch_probability=switch, track_frames=30.0,
                                   frame_time_ms=FRAME_MS, p_dark=0.1, dark_frames=1.5,
                                   precision_nm=PRECISION, seed=11)
            for method in ("truth", "global", "simple", "global_slow"):
                t0 = time.perf_counter()
                if method == "truth":
                    ids = np.asarray(locs["track_id_true"], np.int64)
                else:
                    r = radius(D[0] if method == "global_slow" else D[1])
                    ids, _ = link_tracks(locs["x_nm"], locs["y_nm"], locs["frame"],
                                         max_jump_nm=r, memory=MEMORY,
                                         algorithm=method.split("_")[0])
                found = score(ids, locs["track_id_true"], locs["frame"])
                share, per_slow = wrong_from_slow(ids, locs)
                row = {"switch_per_frame": switch / 2, "density_um2": density,
                       "fast_jump_over_spacing": round(rms_jump(D[1]) / spacing_nm(density), 3),
                       "method": method, "n_locs": len(locs),
                       "links_found": found["links_found"],
                       "links_wrong": found["links_wrong"],
                       "tracks_whole": found["tracks_whole"],
                       "wrong_from_slow": share, "wrong_per_slow_link": per_slow}
                row.update(analyse(ids, locs))
                row["seconds"] = time.perf_counter() - t0
                row = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                       for k, v in row.items()}
                rows.append(row)
                print(row, flush=True)
    with open(HERE / "two_states.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
