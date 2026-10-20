"""The learnt blur σ and pupil magnitude across the Zenodo bead sets: trends, not one number (plan §10, 3 and 7).

    python benchmarks/uipsf/real/zenodo_sigma.py 1ch|2ch|fd [--region R] [--magnitude free] [--device mps]
                                                [--out benchmarks/uipsf/results/zenodo_sigma.jsonl]

Each set is learnt the way SMAPpy's *PSF calibration* window learns it: SMAPpy's ROIs (`bead_rois`),
`calibrate_beads` with its defaults (n_max 6, magnitude free, blur free, pixel centres), beads in water
(n_med 1.33) for the 40 nm sets, v1's settings for FD, the bead's diameter given.  Per channel it records σ,
σ less the pixel's share
(√(σ² − a²/12): learning at pixel centres puts the pixel's width into σ, `synthetic/pixels.py`), the
magnitude (rms, the rotationally symmetric share, the largest modes), the largest phase modes and the atlas's
mean deviance.

*fd* (`1ch_FD_bead`, Prime 95B, 1608 × 1608 px): ``--region R`` keeps the beads within R pixels of the
chip's centre, so a small central field can be compared with the whole one: aberrations varying over the
field should widen σ (and perhaps the magnitude) for the whole field only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.dual import DualColorSettings
from smappy.calibrate.pupil import BeadSettings, bead_rois
from smappy.metadata import CameraMetadata
from zenodo_camera import CAMERA

import uipsf
from uipsf.diagnostics import zernikes
from uipsf.model.channels import prefix
from uipsf.optics.zernike import noll_to_nm
from uipsf.workflows import calibrate_beads

#: The FD set's Prime 95B (HDR, 16 bit, offset 100 from its tags; no conversion there): 0.83 e⁻/ADU from the
#: background, 138 ADU above the offset with a variance of 166 ADU² between neighbouring slices; read noise
#: the camera's typical 1.6 e⁻.  Its four DESPECKLE filters were on (PP 1–4 in the tags).
PRIME95B = CameraMetadata(conversion=0.83, offset=100.0, read_noise_e=1.6, pixelsize_um=[0.11, 0.11])
DATA = os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo"))
#: (folder, optics, bead diameter nm, camera or None for the files' own, dual).  The 40 nm sets are M2's
#: (Evolve 512 EMCCD, beads in water); the FD set's optics are v1's own settings for it (results/1ch_FD_bead:
#: NA 1.5, 680 nm, n 1.515 / 1.406 / 1.525, 110 nm pixels, Prime 95B sCMOS, bead radius 0 in v1).
SETS = {"1ch": ("1ch_40nm_bead", dict(NA=1.43, wavelength_nm=600.0, n_imm=1.518, n_med=1.33,
                                      pixel_size_nm=(116.0, 127.0)), 40.0, CAMERA, False),
        # v1's indices: with 1.518 / 1.33 this set's fit stalls at its start (deviance 20.8, October 2026)
        "2ch": ("2ch_40nm_bead", dict(NA=1.43, wavelength_nm=680.0, n_imm=1.516, n_med=1.35,
                                      pixel_size_nm=(116.0, 127.0)), 40.0, CAMERA, True),
        "fd": ("1ch_FD_bead", dict(NA=1.5, wavelength_nm=680.0, n_imm=1.515, n_med=1.406, n_cov=1.525,
                                   pixel_size_nm=(110.0, 110.0)), 0.0, PRIME95B, False)}


def summary(cal, c, pixel):
    out = {}
    p = cal.shared()
    sig = [float(s) for s in p[prefix(c) + "psf.sigma"]]
    out["sigma_nm"] = [round(s, 1) for s in sig]
    out["sigma_less_pixel_nm"] = [round(math.sqrt(max(s * s - a * a / 12, 0.0)), 1)
                                  for s, a in zip(sig, pixel, strict=True)]
    for which in ("magnitude", "phase"):
        modes, values, crlb = zernikes(cal, which, c)
        v = np.asarray(values, float)
        if not len(v):
            continue
        top = np.argsort(-np.abs(v))[:5]
        entry = dict(rms=round(float(np.sqrt(np.mean(v ** 2))), 4),
                     top={f"{int(modes[i])} {noll_to_nm(int(modes[i]))}": round(float(v[i]), 3) for i in top})
        if which == "magnitude":
            radial = sum(x * x for j, x in zip(modes, v, strict=True) if noll_to_nm(int(j))[1] == 0)
            entry["radial_share"] = round(float(radial / max((v ** 2).sum(), 1e-30)), 2)
        out[which] = entry
    return out


def wander(cal, batch, noise, stacks):
    """Each used bead's slices localised alone on the learnt pupil (`scoring.slice_fits`), against where the
    stack fit puts them: the residual d [E, K, 2].  A stage that wanders laterally between z steps moves
    every bead of an acquisition alike, so the per-stack median over beads at each slice is that wander;
    over a shared model it widens the PSF as a blur of the same rms would.  Also the residuals' excess
    over the CRLB, common and per-bead jitter together."""
    from uipsf.workflows.scoring import slice_fits
    used = list(cal.beads.used)
    sub = batch.subset(torch.tensor(used))
    found, crlb, where = slice_fits(cal.model, cal.shared(), sub, noise, cal.result.params)
    d, c = found[..., :2] - where[..., :2], crlb[..., :2]
    ok = (np.abs(d) < 300).all(-1) & np.isfinite(d).all(-1)
    stack = np.asarray(stacks)[used]
    common, floor = [], []
    for s in np.unique(stack):
        rows = np.flatnonzero(stack == s)
        if len(rows) < 2:
            continue
        dd = np.where(ok[rows][..., None], d[rows], np.nan)
        m = np.nanmedian(dd, 0)                                       # [K, 2]: the stack's shift per slice
        common.append(m - np.nanmean(m, 0))                           # its drift-free part (drift is fitted)
        floor.append(np.nanmedian(c[rows], 0) * 1.2533 / np.sqrt(len(rows)))   # the median's noise
    common, floor = np.concatenate(common), np.concatenate(floor)
    rms = np.sqrt(np.nanmean(common ** 2, 0))
    noise_rms = np.sqrt(np.nanmean(floor ** 2, 0))
    resid = np.where(ok[..., None], d - np.nanmean(np.where(ok[..., None], d, np.nan), 1, keepdims=True), np.nan)
    excess = np.sqrt(np.maximum(np.nanmean(resid ** 2, (0, 1)) - np.nanmean(c ** 2, (0, 1)), 0))
    return dict(stacks_with_2_beads=int(len(np.unique(stack)) - sum((stack == s).sum() < 2 for s in np.unique(stack))),
                common_rms_nm=[round(float(v), 1) for v in rms],
                common_noise_rms_nm=[round(float(v), 1) for v in noise_rms],
                common_less_noise_nm=[round(float(np.sqrt(max(a * a - b * b, 0))), 1)
                                      for a, b in zip(rms, noise_rms, strict=True)],
                residual_excess_over_crlb_nm=[round(float(v), 1) for v in excess],
                crlb_median_nm=[round(float(v), 1) for v in np.nanmedian(c, (0, 1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=list(SETS))
    ap.add_argument("--region", type=float, default=None, help="fd: beads within R px of the chip's centre")
    ap.add_argument("--magnitude", default="free", choices=["free", "symmetric", "fixed"])
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--bead", type=float, default=None, help="bead diameter nm (default the set's)")
    ap.add_argument("--max-beads", type=int, default=None, help="the brightest N after the region cut")
    ap.add_argument("--stacks", type=int, default=None, help="only the first N acquisitions (fd: 1 GB each)")
    ap.add_argument("--step", type=int, default=1, help="every step-th slice (fd: 10 nm apart)")
    ap.add_argument("--first", type=int, default=1, help="the first slice kept (fd 41: ±600 nm with --last 161)")
    ap.add_argument("--last", type=int, default=None, help="the slice after the last one kept")
    ap.add_argument("--wander", action="store_true", help="the per-slice common lateral shift (slice_fits)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    folder, optics_kw, bead, camera, dual = SETS[args.set]
    pixel = optics_kw["pixel_size_nm"]
    bead = bead if args.bead is None else args.bead
    t0 = time.perf_counter()
    paths = sorted(glob.glob(os.path.join(DATA, folder, "*", "*.ome.tif")))[:args.stacks]
    settings = (DualColorSettings if dual else CalibrationSettings)(roi_size=25)
    rois = bead_rois(paths, camera, dual=dual, settings=settings, slices=slice(args.first, args.last, args.step))
    batch = rois.batch
    keep = np.arange(batch.n_emitters)
    if args.region is not None:
        shapes = [s["shape"] for s in rois.sources]
        cx, cy = [(s[-1] - 1) / 2 for s in shapes], [(s[-2] - 1) / 2 for s in shapes]
        r = np.array([math.hypot(rec["x_px"] - cx[rec["stack"]], rec["y_px"] - cy[rec["stack"]])
                      for rec in rois.records])
        keep = keep[r <= args.region]
    if args.max_beads and len(keep) > args.max_beads:
        bright = np.array([rois.records[i].get("brightness_adu", 0) for i in keep])
        keep = np.sort(keep[np.argsort(-bright)[:args.max_beads]])
    batch = batch.subset(torch.tensor(keep))
    optics = [uipsf.Optics(**optics_kw) for _ in range(2 if dual else 1)]
    model_settings = BeadSettings(magnitude=args.magnitude, bead_diameter_nm=bead, zernike_order=args.order)
    cal = calibrate_beads(batch, optics, model_settings, transforms=[None, rois.transform] if dual else None,
                          geometry=rois.geometry, noise=rois.noise, device=args.device)
    xy = np.array([[rois.records[i]["x_px"], rois.records[i]["y_px"]] for i in keep])
    camera = dict(conversion=rois.camera.conversion, offset=rois.camera.offset, read_noise_e=rois.camera.read_noise_e)
    row = dict(set=args.set, stacks=len(paths), step=args.step, first=args.first, last=args.last, camera=camera,
               region=args.region, magnitude=args.magnitude, order=model_settings.order,
               pixel_nm=pixel, beads=len(keep), used=len(cal.beads.used),
               field_px=dict(x=[round(float(xy[:, 0].min())), round(float(xy[:, 0].max()))],
                             y=[round(float(xy[:, 1].min())), round(float(xy[:, 1].max()))]),
               slices=int(batch.data.shape[2]), z_step_nm=float(batch.z_step_nm),
               mean_deviance=round(float(cal.atlas.mean_deviance), 3) if cal.atlas is not None else None,
               channels=[summary(cal, c, pixel) for c in range(2 if dual else 1)],
               minutes=round((time.perf_counter() - t0) / 60, 1))
    if args.wander:
        row["wander"] = wander(cal, batch, rois.noise, [rois.records[i]["stack"] for i in keep])
    print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
