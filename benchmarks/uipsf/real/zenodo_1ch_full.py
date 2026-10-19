"""The whole M3 pipeline on Zenodo `1ch_40nm_bead`, from the raw stacks (plan §9 M3, end-to-end).

    python benchmarks/uipsf/real/zenodo_1ch_full.py [--out benchmarks/uipsf/results/zenodo_1ch_full.jsonl] \\
        [--device mps]

Stacks read with the camera SMAPpy's database names (4.7 e⁻/ADU, offset 398.6) and a read noise of
21 e⁻ (`zenodo_camera.py`); 116 × 127 nm pixels (docs/uipsf/data.md); beads found by SMAPpy
(`calibrate.pupil.single_rois`),
25 × 25 ROIs, all 41 slices.  `learn_beads` with its defaults (20 % held out, rejection on), v1's optics and n_max 8, a
25 nm bead.  Then the residual atlas of the used beads and forward selection of the optional terms
(`diagnostics.select_terms`), and the held-out report again on the selected model.

Prints JSON lines: the bead table, the atlas checks, the probes, the held-out summaries and the
Zernikes; the fitted parameters are saved next to ``--out`` (never committed).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.pupil import single_rois
from zenodo_camera import CAMERA, read

import uipsf
from uipsf.diagnostics import heldout_report, residual_atlas, select_terms
from uipsf.model import ForwardModel
from uipsf.workflows import learn_beads
from uipsf.workflows.scoring import shared_values


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--device", default="cpu", help="where the model runs, e.g. mps")
    ap.add_argument("--localiser", default="pupil", choices=["pupil", "spline"],
                    help="single-slice fits of the scoring: the pupil model, or SMAPpy's spline fitter")
    ap.add_argument("--no-probes", action="store_true", help="stop after the calibration")
    args = ap.parse_args()
    t0 = time.perf_counter()
    files = sorted(glob.glob(os.path.join(args.data, "1ch_40nm_bead", "*", "*.ome.tif")))
    stacks = read(files)
    rois = single_rois(stacks, CAMERA, CalibrationSettings(roi_size=25))
    batch, beads = rois.batch.to(args.device), rois.records
    optics = uipsf.Optics(NA=1.43, wavelength_nm=600, n_imm=1.516, n_med=1.35, pixel_size_nm=(116, 127))
    grid = uipsf.PupilGrid.build(optics, 64, device=args.device)
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=8), (25, 25), bead_radius_nm=25.0,
                          sigma_nm=(30.0, 30.0))
    model, noise = ForwardModel(psf), rois.noise
    emit(dict(stage="data", device=args.device, stacks=len(stacks), beads=len(beads), s=float(batch.s),
              camera=CAMERA.to_dict()), args.out)
    localiser = None
    if args.localiser == "spline":
        from uipsf.workflows.spline import SplineFrames
        localiser = SplineFrames()
    cal = learn_beads(model, batch, noise, chunk_size=args.chunk, localiser=localiser,
                      progress=lambda e: print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                                           for k, v in e.items()}), flush=True))
    for row in cal.beads.as_dicts():
        emit(dict(stage="bead", **row), args.out)
    p = cal.result.params.physical()
    emit(dict(stage="calibration", localiser=args.localiser, used=len(cal.beads.used),
              rejected=cal.beads.ids("rejected"),
              held_out=cal.beads.ids("held out"), thresholds=cal.beads.thresholds,
              iterations=cal.result.iterations, stop=cal.result.stop_reason,
              minutes=round((time.perf_counter() - t0) / 60, 1), atlas_mean_deviance=cal.atlas.mean_deviance,
              atlas=cal.atlas.checks, atlas_flagged=cal.atlas.flagged(),
              heldout=cal.heldout_report.summary() if cal.heldout_report else None,
              sigma_nm=p["psf.sigma"], phase=dict(zip(psf.pupil.phase_modes, p["pupil.phase"].tolist(), strict=True)),
              phase_crlb=cal.result.std["pupil.phase"]), args.out)
    if args.out:
        torch.save({"params": cal.result.params.values, "used": cal.beads.used},
                   os.path.splitext(args.out)[0] + "_params.pt")
    if args.no_probes:
        return
    used = batch.subset(torch.tensor(cal.beads.used))
    selected, params, added = select_terms(model, cal.result.params, used, noise, max_terms=3)
    emit(dict(stage="probes", added=[dict(term=a.term, gain=a.gain, threshold=a.threshold, dof=a.dof,
                                          values={k: v for k, v in a.values.items()}) for a in added],
              minutes=round((time.perf_counter() - t0) / 60, 1)), args.out)
    if added:
        atlas = residual_atlas(selected, params, used, noise)
        held = batch.subset(torch.tensor(cal.beads.ids("held out")))
        rep = heldout_report(selected, shared_values(params), held, noise, cal.beads.ids("held out"))
        emit(dict(stage="selected", terms=[a.term for a in added], atlas_mean_deviance=atlas.mean_deviance,
                  atlas=atlas.checks, atlas_flagged=atlas.flagged(), heldout=rep.summary(),
                  minutes=round((time.perf_counter() - t0) / 60, 1)), args.out)


if __name__ == "__main__":
    main()
