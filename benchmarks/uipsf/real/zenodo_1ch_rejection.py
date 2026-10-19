"""Bead rejection on Zenodo `1ch_40nm_bead`, against v1's: the 20 beads v1 found, of which it kept 19.

    python benchmarks/uipsf/real/zenodo_1ch_rejection.py [--out benchmarks/uipsf/results/zenodo_1ch_rejection.jsonl]

The beads are v1's candidates (`res/cor_all`; the stack of the one v1 dropped is the one with a bead at
that position, Pos13), cut from the raw stacks in the camera's frame, slices 1–39, in SMAPpy's photons
with a read noise of 21 e⁻ and 116 × 127 nm pixels (docs/uipsf/data.md).  `learn_beads` runs with its defaults
and nothing held out, as v1 held out none; the bead table is printed, one JSON line per bead, then a
summary line.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import h5py
import numpy as np
from zenodo_camera import CAMERA

import uipsf
from uipsf.data.io import read_stack
from uipsf.data.rois import Beads, extract_rois
from uipsf.model import ForwardModel
from uipsf.workflows import learn_beads

RESULT = "results/1ch_40nm_bead/psfmodel_zernike_vector_single.h5"


def v1_candidates(root):
    """(beads, files): v1's 19 kept beads with their stacks, plus the one it dropped."""
    with h5py.File(os.path.join(root, RESULT), "r") as f:
        cor, call, fid = f["rois/cor"][()], f["res/cor_all"][()], f["rois/fileID"][()]
    files = sorted(glob.glob(os.path.join(root, "1ch_40nm_bead", "*", "*.ome.tif")))
    kept = {tuple(c) for c in cor}
    dropped = [tuple(int(v) for v in c) for c in call if tuple(c) not in kept]
    pos13 = next(i for i, p in enumerate(files) if "Pos13_" in p)
    x = [int(c[0]) for c in cor] + [d[0] for d in dropped]
    y = [int(c[1]) for c in cor] + [d[1] for d in dropped]
    stack = [int(s) for s in fid] + [pos13] * len(dropped)
    return x, y, stack, files, len(cor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    x, y, stack, files, n_kept = v1_candidates(args.data)
    used = sorted(set(stack))
    stacks = [read_stack(files[i], CAMERA) for i in used]
    beads = Beads(np.array(x), np.array(y), np.array([used.index(s) for s in stack]), np.zeros(len(x)))
    batch = extract_rois(stacks, beads, 25, slices=slice(1, 40))
    optics = uipsf.Optics(NA=1.43, wavelength_nm=600, n_imm=1.516, n_med=1.35, pixel_size_nm=(116, 127))
    grid = uipsf.PupilGrid.build(optics, 64, device="cpu")
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=8), (25, 25), bead_radius_nm=25.0,
                          sigma_nm=(30.0, 30.0))
    t = time.perf_counter()
    cal = learn_beads(ForwardModel(psf), batch, heldout=0.0)
    seconds = time.perf_counter() - t
    lines = []
    for row in cal.beads.as_dicts():
        row["v1"] = "kept" if row["bead"] < n_kept else "rejected"
        row["stack_file"] = os.path.basename(os.path.dirname(files[stack[row["bead"]]]))
        lines.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    summary = dict(summary=True, beads=len(x), used=len(cal.beads.used), rejected=cal.beads.ids("rejected"),
                   thresholds={k: round(v, 4) for k, v in cal.beads.thresholds.items()},
                   iterations=cal.result.iterations, seconds=round(seconds, 1), stop=cal.result.stop_reason)
    lines.append(summary)
    for line in lines:
        print(json.dumps(line), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(line) + "\n" for line in lines)


if __name__ == "__main__":
    main()
