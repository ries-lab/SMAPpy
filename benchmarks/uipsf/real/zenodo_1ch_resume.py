"""Resume `zenodo_1ch_full.py` from its saved fit: re-judge the held-out beads, report them, probe the terms.

    python benchmarks/uipsf/real/zenodo_1ch_resume.py RUN.jsonl [--probe-beads 16] [--out RESULTS.jsonl]

The full run takes hours on a CPU and its probes, which refit the whole model per term, did not survive a
container restart.  This rebuilds the same batch (the finder is deterministic), loads the fitted
parameters saved beside RUN.jsonl, and:
- re-decides every bead from the scores the run recorded, with the current rules (held-out beads are
  now judged too), and reports the held-out beads that pass;
- probes the optional terms by likelihood ratio on a random subset of the used beads, each refit capped
  at a few iterations: a capped refit gains at most what a full one would, so a flagged term is still a
  finding; an unflagged one is weaker evidence than a full probe.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.pupil import single_rois
from zenodo_camera import CAMERA, read

import uipsf
from uipsf.diagnostics import heldout_report, probe_terms
from uipsf.model import ForwardModel
from uipsf.params import Params
from uipsf.solve import LevenbergMarquardt
from uipsf.workflows import RejectionSettings, decide
from uipsf.workflows.scoring import BeadScores, shared_values


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--probe-beads", type=int, default=16)
    ap.add_argument("--probe-iter", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rows = [json.loads(line) for line in open(args.run)]
    beads_rows = sorted((r for r in rows if r["stage"] == "bead"), key=lambda r: r["bead"])
    saved = torch.load(os.path.splitext(args.run)[0] + "_params.pt")
    files = sorted(glob.glob(os.path.join(args.data, "1ch_40nm_bead", "*", "*.ome.tif")))
    rois = single_rois(read(files), CAMERA, CalibrationSettings(roi_size=25))
    batch = rois.batch
    # Runs made before the finder took SMAPpy's convention (October 2026) found other beads.
    assert batch.n_emitters == len(beads_rows), "the finder found different beads than the run did"
    optics = uipsf.Optics(NA=1.43, wavelength_nm=600, n_imm=1.516, n_med=1.35, pixel_size_nm=(116, 127))
    grid = uipsf.PupilGrid.build(optics, 64, device="cpu")
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=8), (25, 25), bead_radius_nm=25.0,
                          sigma_nm=(30.0, 30.0))
    model, noise = ForwardModel(psf), rois.noise
    used = saved["used"]
    sub = batch.subset(torch.tensor(used))
    params = Params(model.specs(sub), len(used))
    params.values = {k: v for k, v in saved["params"].items()}
    shared = shared_values(params)

    # Re-judge with the current rules, from the scores the run recorded.
    names = BeadScores.NAMES
    scores = BeadScores(*[np.array([r[n] for r in beads_rows]) for n in names], fit=None)
    held = [r["bead"] for r in beads_rows if r["state"] == "held out"]
    table = decide(scores, RejectionSettings(), held)
    newly = [r.bead for r in table.rows if r.bead in held and r.state == "rejected"]
    ids = table.ids("held out")
    rep = heldout_report(model, shared, batch.subset(torch.tensor(ids)), noise, ids)
    emit(dict(stage="heldout", held_out=ids, now_rejected={b: table.rows[b].reason for b in newly},
              summary=rep.summary(), z_bias_nm=rep.z_bias_nm, z_spread_nm=rep.z_spread_nm, z_crlb_nm=rep.z_crlb_nm,
              z_nm=rep.z_nm, minutes=round((time.perf_counter() - t0) / 60, 1)), args.out)

    # Probes on a subset of the used beads.
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(used), size=min(args.probe_beads, len(used)), replace=False).tolist())
    psub = sub.subset(torch.tensor(pick))
    pparams = params.subset(torch.tensor(pick))
    probes = probe_terms(model, pparams, psub, noise, solver=LevenbergMarquardt(max_iter=args.probe_iter))
    emit(dict(stage="probes", beads=[used[i] for i in pick], probe_iter=args.probe_iter,
              probes=[dict(term=q.term, gain=q.gain, threshold=q.threshold, dof=q.dof, flagged=q.flagged,
                           values={k: v for k, v in q.values.items()}) for q in probes],
              minutes=round((time.perf_counter() - t0) / 60, 1)), args.out)


if __name__ == "__main__":
    main()
