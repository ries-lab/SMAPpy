"""How coarse an astigmatic bead scan's z step can be, and how long its range must be (docs/uipsf/zstep.md).

    UIPSF_DEVICE=cpu python benchmarks/uipsf/real/zenodo_1ch_zstep.py --splits 0,1,2 \\
        --out benchmarks/uipsf/results/zenodo_1ch_zstep.jsonl
    UIPSF_DEVICE=cpu python benchmarks/uipsf/real/zenodo_1ch_zstep.py --splits 0,1,2 --wide 850 --out ...
    python benchmarks/uipsf/real/zenodo_1ch_zstep_figure.py      # the table and the figure

uiPSF's `1ch_40nm_bead` stacks (31 positions, 41 slices 50 nm apart, ±1000 nm; strong vertical astigmatism,
Noll 6 at 0.56 rad in v1's model; docs/uipsf/data.md), read as SMAPpy's PSF calibration window reads them.
The beads are split at random into training (two thirds) and held-out beads, ``--splits`` times.  On the
training beads, at each setting:

* *z step*: every k-th slice, through the middle one (the nominal focus), over the whole stack: 50, 100,
  150 and 200 nm.  The native step is 50 nm, so 20 nm cannot be made; 150 nm reaches ±900 only.  A
  coarser scan here has fewer frames of the same exposure, as a lab's coarser scan would: the photons go
  down with the step.
* *z range*: the 50 nm slices within ±500, ±750 and ±1000 nm (the stacks hold no more).

Each setting learns uiPSF's pupil model (`uipsf.workflows.calibrate_beads`, the window's defaults, a 40 nm
bead) and, for the step, smappy's bead average (`calibrate.core.build_calibration`).  Both become a
`SplineCalibration`, and every slice of the held-out beads' full 50 nm stacks is fitted with it by Spline 3D
from five z starts, as the window's *Fit quality* does (`pupil.fit_quality`), so both methods are judged
by the fitter the data will meet.  The pupil model's spline is sampled at 50 nm whatever the step (the
learning, not the spline's knots, is compared); ``pupil@step`` is the same model sampled at the scan's
step, as the window's spline on *auto* would (`SplineSettings.dz_nm` None).

Per held-out bead, in the slices within ±``--window`` nm of its own focus (the reference model's fit):
z against the stage, a line per bead -- its slope, the rms about it, and the *bias*: the rms over z of
the median across beads of the centred error (the systematic curve, which the noise averages out of); the
scatter about that curve; x and y about each bead's mean; the median CRLBs; the deviance per pixel of
the fits (Spline 3D's logl is minus half the Poisson deviance).  For the pupil model also the rms
distance of its phase Zernikes (Noll ≥ 5) from the reference: the model learnt on every bead at 50 nm over
±1000 nm.  ``--wide W``: instead, the range study's pupil models sampled over the whole stack (±975 nm),
past what a short scan covered, and judged within ±W -- does a model of a short scan extrapolate?  One JSON
line per (split, method, setting); a line already in ``--out`` is not run again.  50 nm, ±1000 nm and
learning take 1 min a split on 4 CPU threads; the whole study (5 splits) about 40 min in two processes.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zenodo_camera import CAMERA, read  # noqa: E402

NATIVE_NM = 50.0
OPTICS = dict(NA=1.43, wavelength_nm=600.0, n_immersion=1.516, n_medium=1.35)   # v1's, from its result file
BEAD_NM = 40.0


def slices(stage, every: int, half_range_nm: float) -> np.ndarray:
    """Every ``every``-th slice of the stage z through the one nearest 0, the nominal focus, within
    ±``half_range_nm``."""
    mid = int(np.argmin(np.abs(stage)))
    idx = np.arange(mid % every, len(stage), every)
    return idx[np.abs(stage[idx]) <= half_range_nm + 1e-6]


def pick(batch, beads, idx):
    """The batch's ``beads`` and slices ``idx``."""
    b = batch.subset(torch.as_tensor(beads))
    t = torch.as_tensor(idx)

    def along_k(v):
        return v.index_select(2, t) if isinstance(v, torch.Tensor) and v.dim() == b.data.dim() else v
    return dataclasses.replace(b, data=b.data.index_select(2, t), z_nm=b.z_nm.index_select(-1, t),
                               s=along_k(b.s), mask=along_k(b.mask))


def learn_pupil(rois, beads, idx):
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, PupilRun

    from uipsf.workflows import calibrate_beads
    batch = pick(rois.batch, beads, idx)
    optics = OpticsSettings(**OPTICS).optics(rois.camera, 1)
    cal = calibrate_beads(batch, optics, BeadSettings(bead_diameter_nm=BEAD_NM), noise=rois.noise)
    sub = dataclasses.replace(rois, batch=batch, records=[rois.records[i] for i in beads])
    return PupilRun(sub, cal)


def pupil_spline(run, dz_nm, z_range_nm=None):
    """The window's spline of the model: over the range every used bead covers, unless ``z_range_nm``."""
    from smappy.calibrate.pupil import _auto_z_range, spline_calibration
    batch = run.rois.batch
    z_range = z_range_nm or _auto_z_range(run.cal, batch)
    return spline_calibration(run.cal, dz_nm, z_range, int(batch.data.shape[-1]), camera=run.rois.camera)


def learn_average(collection, beads, idx):
    from smappy.calibrate.core import BeadCollection, build_calibration
    records = [dict(collection.records[i], id=n) for n, i in enumerate(beads)]
    sub = BeadCollection(collection.volumes[beads][:, idx], records, collection.projections, collection.sources,
                         collection.dz_nm * (idx[1] - idx[0]), collection.settings)
    return build_calibration(sub).calibration


def phase_zernikes(cal) -> dict:
    from uipsf.diagnostics import zernikes
    modes, values, _ = zernikes(cal, "phase", 0)
    return {int(j): float(v) for j, v in zip(modes, values, strict=True)}


def zernike_distance_nm(a: dict, b: dict) -> float:
    d = [a[j] - b.get(j, 0.0) for j in a if j >= 5]
    return float(np.sqrt(np.mean(np.square(d))) * OPTICS["wavelength_nm"] / (2 * np.pi))


def fit_slices(spline, batch, z_starts=5, iterations=75) -> dict:
    """Every slice of every bead in ``batch`` fitted by Spline 3D, best of ``z_starts`` (`pupil.fit_quality`)."""
    from smappy.calibrate.pupil import _best_of
    from smappy.psf import SplinePSF
    data = batch.data.detach().cpu().double().numpy()[:, 0]                  # [E, K, Y, X]
    E, K, Y, X = data.shape
    s = np.broadcast_to(torch.as_tensor(batch.s).detach().cpu().double().numpy(), batch.data.shape)[:, 0]
    box = np.ascontiguousarray(np.maximum(data + s, 0)[..., 2:Y - 2, 2:X - 2].reshape(E * K, Y - 4, X - 4),
                               np.float32)
    best = _best_of(lambda z: SplinePSF(spline, z_start_nm=z).fit(box, iterations=iterations),
                    spline.z_index_to_nm(np.linspace(1, spline.shape[0] - 2, z_starts)))
    v = SplinePSF(spline).unpack(best)
    ok = np.isfinite(best.logl) & np.isfinite(best.theta).all(axis=1)
    ok &= (best.theta[:, 4] >= 1) & (best.theta[:, 4] <= spline.shape[0] - 2)  # not stopped at the spline's end
    px = np.asarray(CAMERA.pixelsize_um, float) * 1000
    out = {"z": v["z_nm"], "x": v["x_roi"] * px[0], "y": v["y_roi"] * px[1], "z_err": v["z_err_nm"],
           "x_err": v["x_err_pix"] * px[0], "y_err": v["y_err_pix"] * px[1],
           "deviance": -2 * best.logl / box[0].size}
    return {k: np.where(ok, val, np.nan).reshape(E, K) for k, val in out.items()}


def judge(fits, stage, centre, window) -> dict:
    """The metrics over each bead's slices within ±window of its own focus (``centre``, nm of stage z)."""
    rel = stage[None, :] - centre[:, None]                                   # [E, K] stage z about focus
    near = np.abs(rel) <= window
    z = np.where(near, fits["z"], np.nan)
    use = np.isfinite(z)
    slopes, resid = [], np.full_like(z, np.nan)
    for e in range(z.shape[0]):
        u = use[e]
        if u.sum() < 5:
            continue
        b, a = np.polyfit(rel[e, u], z[e, u], 1)
        slopes.append(b)
        resid[e, u] = z[e, u] - (a + b * rel[e, u])
    # the systematic curve: per bead the centred error against the common line, the median over beads
    slope = float(np.median(slopes))
    err = z - slope * rel
    err -= np.nanmedian(err, axis=1, keepdims=True)
    k = np.round(rel / NATIVE_NM).astype(int)
    curve = {}
    for kk in np.unique(k[near]):
        vals = err[(k == kk) & use]
        if len(vals) >= 3:
            curve[int(kk)] = float(np.median(vals))
    bias = np.sqrt(np.mean(np.square(list(curve.values()))))
    sys_at = np.vectorize(lambda kk: curve.get(int(kk), np.nan))(k)
    scatter = err - sys_at

    def about_mean(q):
        q = np.where(use, q, np.nan)
        return float(np.sqrt(np.nanmean(np.square(q - np.nanmean(q, axis=1, keepdims=True)))))
    return {"slope": slope, "z_rms_line_nm": float(np.sqrt(np.nanmean(resid ** 2))),
            "z_bias_nm": float(bias), "z_scatter_nm": float(np.sqrt(np.nanmean(scatter ** 2))),
            "x_rms_nm": about_mean(fits["x"]), "y_rms_nm": about_mean(fits["y"]),
            "z_crlb_nm": float(np.nanmedian(np.where(use, fits["z_err"], np.nan))),
            "x_crlb_nm": float(np.nanmedian(np.where(use, fits["x_err"], np.nan))),
            "y_crlb_nm": float(np.nanmedian(np.where(use, fits["y_err"], np.nan))),
            "deviance_per_px": float(np.nanmean(np.where(use, fits["deviance"], np.nan))),
            "fitted_fraction": float(use.sum() / max(near.sum(), 1)),
            "curve_nm": {kk * int(NATIVE_NM): round(v, 1) for kk, v in sorted(curve.items())}}


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
             if k not in ("curve_nm", "zernikes")}
    print(json.dumps(short), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def done_keys(out):
    if not out or not os.path.exists(out):
        return set()
    return {(r["split"], r["method"], r["step_nm"], r["half_range_nm"]) for r in map(json.loads, open(out))}


def main():
    from smappy.calibrate.core import CalibrationSettings, collect_beads
    from smappy.calibrate.pupil import single_rois
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/.cache/uipsf/zenodo/1ch_40nm_bead"))
    ap.add_argument("--splits", default="0,1,2")
    ap.add_argument("--steps", default="50,100,150,200")
    ap.add_argument("--ranges", default="500,750,1000")
    ap.add_argument("--train", type=float, default=2 / 3)
    ap.add_argument("--window", type=float, default=400.0, help="± about each bead's focus that is judged, nm")
    ap.add_argument("--wide", type=float, default=None,
                    help="instead: the range study's models sampled over the whole stack, judged within ±this nm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    stacks = read(sorted(glob.glob(os.path.join(args.data, "*", "*.ome.tif"))))
    settings = CalibrationSettings()
    rois = single_rois(stacks, CAMERA, settings)
    collection = collect_beads(stacks, settings)
    where = {(r["stack"], r["x_px"], r["y_px"]): i for i, r in enumerate(collection.records)}
    beads = [i for i, r in enumerate(rois.records) if (r["stack"], r["x_px"], r["y_px"]) in where]
    in_avg = {i: where[(rois.records[i]["stack"], rois.records[i]["x_px"], rois.records[i]["y_px"])] for i in beads}
    K = rois.batch.data.shape[2]
    # the window drops each stack's first slice (`pupil.KEEP`), the bead average keeps it
    first = collection.volumes.shape[1] - K
    assert first in (0, 1), "the bead average's stacks are cropped otherwise: slices would not match"
    stage = rois.batch.z_nm.detach().cpu().double().numpy()
    print(f"{rois.batch.n_emitters} ROIs, {len(collection.records)} in the bead average, {len(beads)} in both; "
          f"{K} slices, stage z {stage[0]:.0f} to {stage[-1]:.0f} nm; {time.time() - t0:.0f} s", flush=True)
    done = done_keys(args.out)

    # the reference: every bead, 50 nm, ±1000 nm -- its Zernikes, and each bead's focus from its fit
    t = time.time()
    ref = learn_pupil(rois, beads, slices(stage, 1, 1e9))
    ref_z = phase_zernikes(ref.cal)
    ref_fit = fit_slices(pupil_spline(ref, NATIVE_NM), pick(rois.batch, beads, np.arange(K)))
    centre = {}
    for n, i in enumerate(beads):            # focus: where the line of fitted z against the stage crosses 0
        u = np.isfinite(ref_fit["z"][n]) & (np.abs(stage) <= 600)
        b, a = np.polyfit(stage[u], ref_fit["z"][n][u], 1)
        centre[i] = -a / b
    print("reference", f"{(time.time() - t) / 60:.1f} min", "used", len(ref.cal.beads.used), "of", len(beads),
          "focus offsets nm", np.round(np.percentile(list(centre.values()), [0, 50, 100]), 0).tolist(),
          "Zernikes", {j: round(v, 3) for j, v in ref_z.items() if abs(v) > 0.05}, flush=True)
    emit({"split": -1, "method": "reference", "step_nm": NATIVE_NM, "half_range_nm": 1000.0, "zernikes": ref_z,
          "focus_nm": [centre[i] for i in beads]}, None)

    steps = [float(s) for s in args.steps.split(",")]
    ranges = [float(r) for r in args.ranges.split(",")]
    for split in (int(s) for s in args.splits.split(",")):
        order = np.random.default_rng(split).permutation(beads)
        n_train = int(round(args.train * len(order)))
        train, held = sorted(order[:n_train]), sorted(order[n_train:])
        held_batch = pick(rois.batch, held, np.arange(K))
        held_centre = np.array([centre[i] for i in held])
        settings_list = [(s, 1000.0) for s in steps] + [(NATIVE_NM, r) for r in ranges if r < 1000.0]
        if args.wide:
            settings_list = [(NATIVE_NM, r) for r in ranges]
        for step, half in settings_list:
            every = int(round(step / NATIVE_NM))
            idx = slices(stage, every, half)
            # 40 slices (the first is dropped) have no slice at 0: a thinned scan reaches one side further
            span = float(np.abs(stage[idx]).max())
            base = {"split": split, "step_nm": step, "half_range_nm": half, "range_used_nm": span,
                    "z_ends_nm": [float(stage[idx].min()), float(stage[idx].max())],
                    "slices": len(idx), "train": len(train), "held": len(held)}
            if args.wide:
                if (split, "pupil@wide", step, half) not in done:
                    run = learn_pupil(rois, train, idx)
                    whole = float(np.abs(stage).max())
                    r = judge(fit_slices(pupil_spline(run, NATIVE_NM, whole), held_batch), stage, held_centre,
                              args.wide)
                    emit({**base, "method": "pupil@wide", "spline_dz_nm": NATIVE_NM, "spline_range_nm": whole,
                          "window_nm": args.wide, **r}, args.out)
                continue
            if (split, "pupil", step, half) not in done:
                t = time.time()
                run = learn_pupil(rois, train, idx)
                minutes = (time.time() - t) / 60
                z = phase_zernikes(run.cal)
                for method, dz in (("pupil", NATIVE_NM), ("pupil@step", step)):
                    if method == "pupil@step" and step == NATIVE_NM:
                        continue
                    r = judge(fit_slices(pupil_spline(run, dz), held_batch), stage, held_centre, args.window)
                    emit({**base, "method": method, "spline_dz_nm": dz, "minutes": minutes,
                          "used": len(run.cal.beads.used), "stop": run.cal.result.stop_reason,
                          "zernike_dist_nm": zernike_distance_nm(z, ref_z), "zernikes": z, **r}, args.out)
            if half == 1000.0 and (split, "average", step, half) not in done:
                t = time.time()
                try:
                    spline = learn_average(collection, [in_avg[i] for i in train], idx + first)
                except ValueError as error:                  # what the window would say: no calibration
                    emit({**base, "method": "average", "failed": str(error)}, args.out)
                    continue
                minutes = (time.time() - t) / 60
                r = judge(fit_slices(spline, held_batch), stage, held_centre, args.window)
                emit({**base, "method": "average", "spline_dz_nm": float(spline.dz), "minutes": minutes, **r},
                     args.out)


if __name__ == "__main__":
    main()
