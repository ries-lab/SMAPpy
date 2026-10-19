"""uipsf's bead finder against SMAPpy's `detect_beads` and v1's beads, on Zenodo `1ch_40nm_bead` (plan §9 M3).

Since October 2026 uipsf's finder follows SMAPpy's convention, so the two should agree bead for bead;
this script shows whether they still do, and how both relate to v1's beads.

    python benchmarks/uipsf/real/compare_finders.py [--full benchmarks/uipsf/results/zenodo_1ch_full.jsonl]

Both finders on the same stacks, with the ROI size of the M3 run (25 px): SMAPpy's with its calibration
defaults (σ 2 px, 6 robust σ, beads closer than 25 px both dropped, its brightness range of 6), uipsf's
with its own (`data.rois.find_beads`).  Prints one JSON line: the counts, the beads only one of them
finds and why, v1's 20 beads in each, and, given the full run's bead table, what uipsf's rejection made
of the beads SMAPpy does not take.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
from zenodo_camera import CAMERA

from uipsf.data.io import read_stack
from uipsf.data.rois import find_beads

RESULT = "results/1ch_40nm_bead/psfmodel_zernike_vector_single.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--full", default=None, help="the full run's JSON lines, for the rejection states")
    args = ap.parse_args()
    import tifffile
    from smappy.calibrate.core import CalibrationSettings, detect_beads
    files = sorted(glob.glob(os.path.join(args.data, "1ch_40nm_bead", "*", "*.ome.tif")))
    stacks = [read_stack(f, CAMERA) for f in files]
    ours = find_beads(stacks, 25)
    ours_set = {(int(s), int(x), int(y)) for x, y, s in zip(ours.x, ours.y, ours.stack, strict=True)}
    settings = CalibrationSettings(roi_size=25)
    theirs, bright = set(), {}
    for i, f in enumerate(files):
        adu = tifffile.imread(f).astype(np.float32)
        yx, _ = detect_beads(adu, settings)
        r = settings.roi_size // 2 + settings.padding
        for y, x in yx:
            vol = adu[:, y - r:y + r + 1, x - r:x + r + 1]
            border = np.concatenate((vol[:, 0], vol[:, -1], vol[:, 1:-1, 0], vol[:, 1:-1, -1]), axis=1)
            bright[(i, int(x), int(y))] = float((vol - np.median(border)).sum((1, 2)).max())
            theirs.add((i, int(x), int(y)))
    med = np.median(list(bright.values()))
    accepted = {k for k, b in bright.items() if med / np.sqrt(6) <= b <= med * np.sqrt(6)}

    def near(a, pool, tol=2):
        return any(a[0] == b[0] and abs(a[1] - b[1]) <= tol and abs(a[2] - b[2]) <= tol for b in pool)

    with h5py.File(os.path.join(args.data, RESULT), "r") as f:
        cor, fid, call = f["rois/cor"][()], f["rois/fileID"][()], f["res/cor_all"][()]
    pos13 = next(i for i, p in enumerate(files) if "Pos13_" in p)
    kept = {tuple(c) for c in cor}
    v1 = [(int(s), int(c[0]), int(c[1])) for c, s in zip(cor, fid, strict=True)]
    v1 += [(pos13, int(c[0]), int(c[1])) for c in call if tuple(c) not in kept]
    only_ours = sorted(b for b in ours_set if not near(b, theirs))
    only_theirs = sorted(b for b in theirs if not near(b, ours_set))
    row = dict(uipsf=len(ours_set), smappy_detected=len(theirs), smappy_accepted=len(accepted),
               both=sum(near(b, theirs) for b in ours_set), only_uipsf=only_ours, only_smappy=only_theirs,
               v1_in_uipsf=sum(near(b, ours_set) for b in v1), v1_in_smappy=sum(near(b, theirs) for b in v1),
               v1_in_smappy_accepted=sum(near(b, accepted) for b in v1), v1=len(v1))
    if args.full:
        rows = [json.loads(line) for line in open(args.full)]
        states = {}
        order = sorted(ours_set, key=lambda b: (b[0], b[1], b[2]))
        idx = {(int(s), int(x), int(y)): e for e, (x, y, s) in enumerate(zip(ours.x, ours.y, ours.stack, strict=True))}
        for r in rows:
            if r.get("stage") == "bead":
                states[r["bead"]] = (r["state"], r["reason"])
        row["uipsf_state_of_beads_smappy_does_not_accept"] = {
            str(b): states.get(idx[b]) for b in order if not near(b, accepted)}
    print(json.dumps(row))


if __name__ == "__main__":
    main()
