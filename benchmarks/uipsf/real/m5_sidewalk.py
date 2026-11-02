"""Where M5's lateral shift with z comes from: the stage, the same for every bead of a frame, or the optics, by place.

    python benchmarks/uipsf/real/m5_sidewalk.py FOLDER [...] [--z-range 600] [--window 1000] [--dz 100] \\
        [--memory 0.5] [--out benchmarks/uipsf/results/m5_sidewalk.jsonl] [--npz DIR]

`m5_updown.py` (2 November 2026) found that each stack's linear drift has the same rate per µm of z scanned up as
down: a shift with z, dx(z), not with time.  The user's question: the stage, or the model?  What tells them apart
is where on the chip a bead is:

- *the stage* moving sideways as it steps -- or, the same thing to the image, its z axis tilted against the
  optical axis -- shifts every bead of a frame alike: one slope per stack, whatever the bead's place;
- *the optics*: a non-telecentric image changes its magnification with defocus, a shift that grows with the
  distance from the axis, (x − x0, y − y0) times one coefficient; coma the pupil does not hold would shift each
  bead by its own amount.

Per bead, each slice is localised alone on the learnt minimal model (`scoring.slice_fits`'s ``found``: a free
x, y, z per slice, so the drift the stack fit learnt is not in it), and a line is fitted to x and to y against the
stage's z within ``--z-range`` of its focus, weighted by the CRLB: dx/dz and dy/dz, nm per µm.  The slopes are
then explained by a slope per stack, by a plane over the chip (the camera ROI's offset added, so places compare
across acquisitions), and by both; and the residual of each slice about its bead's line is averaged over the
beads of each camera frame, whose scatter beyond the beads' own is what the stage does frame by frame.
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402
from m5_consensus import GRID_NA, PIXEL, minimal, summary  # noqa: E402
from m5_media import N_IMM, N_MED, ORDER, WAVELENGTH, condition, select, subset  # noqa: E402


def metadata(tif: str) -> dict:
    """The camera ROI's origin (px) and the stage's x, y (µm) of a Micro-Manager stack."""
    text = open(re.sub(r"\.ome\.tif$", "_metadata.txt", tif)).read(400000)
    roi = re.search(r'"ROI":\s*"(\d+)-(\d+)-\d+-\d+"', text)
    num = lambda key: float(re.search(rf'"{key}":\s*(-?[\d.]+)', text)[1])  # noqa: E731
    return dict(roi=(int(roi[1]), int(roi[2])) if roi else (0, 0), stage=(num("XPositionUm"), num("YPositionUm")))


def lines(found, crlb, zk, centre, z_range):
    """Each bead's weighted line of x and y against the stage's z (µm) within ``z_range`` of its focus:
    slopes [E, 2] nm/µm, their standard errors [E, 2], and each slice's residual about its line [E, K, 2]."""
    E = found.shape[0]
    slope, se, resid = np.full((E, 2), np.nan), np.full((E, 2), np.nan), np.full(found.shape[:2] + (2,), np.nan)
    for e in range(E):
        t = (zk[e] - centre[e]) / 1000.0
        for a in range(2):
            ok = (np.abs(t) <= z_range / 1000.0) & np.isfinite(found[e, :, a]) & (crlb[e, :, a] > 0)
            if ok.sum() < 4:
                continue
            w = 1 / crlb[e, ok, a] ** 2
            A = np.stack([np.ones(ok.sum()), t[ok]], 1)
            cov = np.linalg.inv(A.T @ (A * w[:, None]))
            coef = cov @ A.T @ (w * found[e, ok, a])
            r = found[e, ok, a] - A @ coef
            # the scatter about the line is 3-4x the CRLB on M5 (`m5_media.md`): the slope's error from it
            chi = float(np.sum(w * r ** 2) / max(ok.sum() - 2, 1))
            slope[e, a], se[e, a] = coef[1], np.sqrt(cov[1, 1] * chi)
            resid[e, ok, a] = r
    return slope, se, resid


def explain(slope, se, stack, chip):
    """How much of the beads' slopes one slope per stack, a plane over the chip, or both explain: the weighted rms
    of what is left, nm/µm, per axis; the plane's coefficients (nm/µm per 100 px) from the joint fit."""
    out = {}
    stacks = np.unique(stack)
    S = (stack[:, None] == stacks[None]).astype(float)
    X = np.column_stack([chip - chip.mean(0)]) / 100.0
    designs = {"one slope": np.ones((len(stack), 1)), "per stack": S,
               "plane": np.column_stack([np.ones(len(stack)), X]), "per stack + plane": np.column_stack([S, X])}
    for a, axis in enumerate("xy"):
        ok = np.isfinite(slope[:, a]) & np.isfinite(se[:, a])
        w = 1 / se[ok, a] ** 2
        for name, D in designs.items():
            D = D[ok]
            keep = D.any(0)
            D = D[:, keep]
            coef, *_ = np.linalg.lstsq(D * np.sqrt(w)[:, None], slope[ok, a] * np.sqrt(w), rcond=None)
            r = slope[ok, a] - D @ coef
            dof = max(ok.sum() - D.shape[1], 1)
            out.setdefault(name, {})[axis] = dict(
                rms=round(float(np.sqrt(np.mean(r ** 2))), 2),
                chi2_dof=round(float(np.sum(w * r ** 2) / dof), 2), params=D.shape[1])
            if name == "per stack + plane":
                out[name][axis]["plane_per_100px"] = np.round(coef[-2:], 2).tolist()
            if name == "plane":
                out[name][axis]["plane_per_100px"] = np.round(coef[-2:], 2).tolist()
            if name == "per stack":
                out[name][axis]["stack_slopes"] = np.round(coef, 2).tolist()
    out["beads"] = int(len(stack))
    out["slope_se_median"] = np.round(np.nanmedian(se, 0), 2).tolist()
    return out


def frame_jitter(resid, frame):
    """The residual about each bead's line averaged over the beads of each camera frame, against what the beads'
    own scatter would give such an average: rms per axis (nm), and that rms over the expected."""
    out = {}
    for a, axis in enumerate("xy"):
        means, expect = [], []
        for f in np.unique(frame[frame >= 0]):
            r = resid[..., a][frame == f]
            r = r[np.isfinite(r)]
            if len(r) >= 3:
                means.append(r.mean())
                expect.append(r.var(ddof=1) / len(r))
        means, expect = np.asarray(means), np.asarray(expect)
        rms = float(np.sqrt(np.mean(means ** 2))) if len(means) else float("nan")
        out[axis] = dict(frames=len(means), rms_nm=round(rms, 2),
                         over_expected=round(rms / float(np.sqrt(np.mean(expect))), 2) if len(means) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="each searched for .ome.tif stacks, at any depth")
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.5)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_sidewalk.jsonl"))
    ap.add_argument("--npz", default=os.path.join(HERE, "..", "results", "m5_sidewalk"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        bead_rois,
        camera_for,
        run_calibration,
    )

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=1, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1))

    t0 = time.perf_counter()
    files = [f for d in args.folders for f in sorted(glob.glob(os.path.join(d, "**", "*.ome.tif"), recursive=True))]
    meta = [metadata(f) for f in files]
    cond = condition(os.path.dirname(files[0]))
    label = args.label or "+".join(os.path.basename(os.path.normpath(d)) for d in args.folders)
    rois = bead_rois(files, camera_for(files[0]), dual=False)
    # one acquisition's consensus beads where `m5_beads.py` chose them, else every bright bead in focus
    # (only a folder that holds its stacks itself: the up/down groups' stacks share the 25-bead set's name)
    single = len(args.folders) == 1 and all(os.path.dirname(f) == os.path.normpath(args.folders[0]) for f in files)
    name = os.path.basename(os.path.normpath(args.folders[0])) if single else None
    keep, focus, cut = select(rois, name)
    # each kept bead's window's first plane, as `subset` cuts it: the slices' absolute stage z (`m5_wobble.py`)
    K = rois.batch.data.shape[2]
    half = min(int(round(args.window / float(rois.batch.z_step_nm))), (K - 1) // 2)
    first = np.array([min(max(int(focus[i]) - half, 0), K - 2 * half - 1) for i in keep])
    rois = subset(rois, keep, focus, args.window)
    ring = cond["medium"] == "mismatch"
    optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                            n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
    psf = PSFModelSettings(pixel_blur=False, zernike_order=ORDER, blur="round", blur_nm=30.0,
                           magnitude=minimal("height", ring, (), {}, {}, 100.0))
    run = run_calibration(files, False, optics, BeadSettings(psf=psf, bead_diameter_nm=100.0),
                          run=RunSettings(save_model=False), rois=rois,
                          decisions={i: "include" for i in range(len(rois.records))})
    cal = run.cal
    used = list(cal.beads.used)
    rec = [rois.records[i] for i in used]
    batch = run.rois.batch.subset(torch.as_tensor(np.asarray(used))).thinned(beads.MAX_DZ_NM)
    found, crlb, _ = scoring.slice_fits(cal.model, cal.shared(), batch, run.rois.noise, cal.result.params)
    found, crlb = np.asarray(found), np.asarray(crlb)
    zk = np.broadcast_to(batch.z_nm.cpu().double().numpy(), found.shape[:2])
    centre = np.array([r["focus_nm"] for r in rec])
    slope, se, resid = lines(found, crlb, zk, centre, args.z_range)
    stack = np.array([int(r["stack"]) for r in rec])
    chip = np.array([[r["x_px"] + meta[s]["roi"][0], r["y_px"] + meta[s]["roi"][1]]
                     for r, s in zip(rec, stack, strict=True)], float)
    frame = batch.frame.cpu().numpy() if batch.frame is not None else np.full(found.shape[:2], -1)

    p = cal.result.params.physical()
    rate = p["geometry.drift_rate"].detach().cpu().double().numpy() if "geometry.drift_rate" in p else None
    row = dict(label=label, **cond, beads=cut, used=len(used), stacks=len(files))
    row.update({k: v for k, v in summary(run, cond).items() if k in ("edge", "sigma_nm", "objective")})
    row["explain"] = explain(slope, se, stack, chip)
    row["frame_jitter"] = frame_jitter(resid, frame)
    row["chip_centre_px"] = chip.mean(0).round(0).tolist()
    row["per_stack"] = [dict(stack=int(s), stage_um=list(meta[s]["stage"]), beads=int((stack == s).sum()),
                             median_slope=np.nanmedian(slope[stack == s], 0).round(2).tolist(),
                             model_rate=None if rate is None else rate[s].round(2).tolist())
                        for s in np.unique(stack)]
    row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
    emit(row, args.out)
    os.makedirs(args.npz, exist_ok=True)
    np.savez_compressed(os.path.join(args.npz, f"{label}.npz"), slope=slope, se=se, stack=stack, chip=chip,
                        resid=resid, frame=frame, found=found, crlb=crlb, zk=zk, centre=centre, first=first[used],
                        stage=np.array([meta[s]["stage"] for s in stack]),
                        rate=rate if rate is not None else np.nan)


if __name__ == "__main__":
    main()
