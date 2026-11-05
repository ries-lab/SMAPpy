"""The same beads scanned up and straight back down: a drift in time against the stage's own shear, and whose bend.

    python benchmarks/uipsf/real/m5_updown.py GROUP [...] [--window 1000] [--dz 100] [--memory 0.5] \\
        [--out benchmarks/uipsf/results/m5_updown.jsonl]

M5, 2 November 2026 (the user's test 2): four groups of 100 nm beads in buffer at 676 nm with the 3D lens, each
scanned 35.4 → 39.4 µm (``…_3D_1``) and 11 s later 39.4 → 35.4 µm (``…_3D_2``), 100 nm steps.  The reader sorts
every stack into ascending z (`smappy.calibrate.input`), so in a downward stack time runs against z:

- a drift in *time* has a rate per µm of z of the opposite sign up and down; a stage that moves sideways as it
  steps, or coma (a shift with defocus), has the same sign.  The linear drift per stack
  (`BeadStackGeometry`, ``geometry.drift_rate``) is read per stack and direction;
- the *z bend* (each slice fitted alone against its bead's stack fit, `m5_bends.per_bead`) is the model's if it
  is the same both ways and the stage's (hysteresis, a nonlinear piezo) if it differs; the beads are paired up
  against down, so each bead's own part cancels in the difference;
- each bead's focus and place between its two scans: the axial and lateral drift over the 20 s.

Three fits of the minimal model (minimal0: cos θ^γ, the ring as a height h, the aperture held at NA 1.50 on a
grid at 1.51; one round σ; every phase mode to order 8), on the same beads: both directions together, the
upward stacks alone and the downward alone -- whether the direction moves what is learnt.  Every bright bead in
focus is used (`m5_media.select`), the whole field: the comparison is paired.
"""
from __future__ import annotations

import argparse
import functools
import glob
import math
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402
from m5_bends import per_bead  # noqa: E402
from m5_consensus import GRID_NA, PIXEL, minimal, summary  # noqa: E402
from m5_media import N_IMM, N_MED, ORDER, WAVELENGTH, condition, select, subset  # noqa: E402

#: a bead up and the same bead down: their places on the chip at most this far apart, px
PAIR_PX = 3.0
Z_RANGE, BIN_NM = 600.0, 100.0


def direction(tif: str) -> int:
    """+1 for a stack acquired upwards, −1 downwards, from its Micro-Manager metadata's first two z positions."""
    meta = re.sub(r"\.ome\.tif$", "_metadata.txt", tif)
    z = [float(v) for v in re.findall(r'"ZPositionUm":\s*(-?[\d.]+)', open(meta).read(400000))[:2]]
    return 1 if z[1] > z[0] else -1


def curves(each, rows, z_range=Z_RANGE, step=BIN_NM):
    """The median over ``rows`` (beads) of the slice error about each bead's own offset, in bins of z about its
    focus: [bins, 3] and the bins' z."""
    e, z = each["error"][rows], each["z"][rows]
    near = np.abs(z) <= z_range
    d = np.where(near[..., None] & np.isfinite(e), e, np.nan)
    d = d - np.nanmedian(d, axis=1, keepdims=True)
    b = np.where(near, np.round(z / step), np.nan)
    bins = np.unique(b[np.isfinite(b)])
    return np.stack([np.nanmedian(d[b == v], axis=0) for v in bins]), bins * step, d, b


def rms(c):
    c = c[np.isfinite(c)]
    return round(float(np.sqrt(np.mean(c ** 2))), 2) if len(c) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="folders of one group: its upward and downward acquisitions")
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_updown.jsonl"))
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

    files = [f for g in args.groups for f in sorted(glob.glob(os.path.join(g, "*", "*.ome.tif")))]
    group = [i for i, g in enumerate(args.groups) for _ in sorted(glob.glob(os.path.join(g, "*", "*.ome.tif")))]
    sign = [direction(f) for f in files]
    cond = condition(os.path.dirname(files[0]))
    ring = cond["medium"] == "mismatch"
    optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                            n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
    print(len(files), "stacks, directions", sign, flush=True)

    def prepared(which):
        """The kept beads of the stacks of direction ``which`` (0: both), each on its window about its focus;
        and each kept bead's absolute focus slice z, to place the stack fit's z on the stage's."""
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, cut = select(rois)
        keep = [i for i in keep if which == 0 or sign[int(rois.records[i]["stack"])] == which]
        z = rois.batch.z_nm.numpy()
        z = z if z.ndim == 1 else z[0]
        K = rois.batch.data.shape[2]
        half = min(int(round(args.window / float(rois.batch.z_step_nm))), (K - 1) // 2)
        centre = [z[min(max(int(focus[i]) - half, 0), K - 2 * half - 1) + half] for i in keep]
        return subset(rois, keep, focus, args.window), np.asarray(centre), cut

    for label, which in (("both", 0), ("up", 1), ("down", -1)):
        t0 = time.perf_counter()
        rois, centre, cut = prepared(which)
        psf = PSFModelSettings(pixel_blur=False, zernike_order=ORDER, blur="round", blur_nm=30.0,
                               magnitude=minimal("height", ring, (), {}, {}, 100.0))
        run = run_calibration(files, False, optics, BeadSettings(psf=psf, bead_diameter_nm=100.0),
                              run=RunSettings(save_model=False), rois=rois,
                              decisions={i: "include" for i in range(len(rois.records))})
        cal = run.cal
        used = list(cal.beads.used)
        rec = [rois.records[i] for i in used]
        p, std = cal.result.params.physical(), cal.result.std
        row = dict(fit=label, groups=[os.path.basename(g) for g in args.groups], **cond, beads=cut, used=len(used))
        row.update(summary(run, cond))
        # the linear drift per stack, nm per µm of z as the stack is sorted: a drift in time flips sign down
        rate = p["geometry.drift_rate"].detach().cpu().double().numpy()
        rate_se = std["geometry.drift_rate"].detach().cpu().double().numpy() if "geometry.drift_rate" in std else None
        stacks = sorted({int(r["stack"]) for r in rec})
        row["drift"] = [dict(stack=s, group=group[s], direction=sign[s], beads=sum(int(r["stack"]) == s for r in rec),
                             rate_nm_per_um=rate[s].round(2).tolist(),
                             crlb=None if rate_se is None else rate_se[s].round(2).tolist()) for s in stacks]
        each = per_bead(run, [r["focus_nm"] for r in rec])
        for name, d in (("up", 1), ("down", -1)):
            rows = [k for k, r in enumerate(rec) if sign[int(r["stack"])] == d]
            if not rows:
                continue
            c, cz, _, _ = curves(each, rows)
            row[f"bend_{name}"] = dict(beads=len(rows), z_nm=cz.round(0), curve_nm=c.round(1),
                                       rms_nm=[rms(c[:, a]) for a in range(3)])
        # the same bead up and down: its slice errors differenced in each z bin, its focus and place compared
        xyz = p["emitter.xyz"].detach().cpu().double().numpy()
        pairs = []
        for k, r in enumerate(rec):
            if sign[int(r["stack"])] != 1:
                continue
            for m, q in enumerate(rec):
                if (sign[int(q["stack"])] == -1 and group[int(q["stack"])] == group[int(r["stack"])]
                        and math.hypot(q["x_px"] - r["x_px"], q["y_px"] - r["y_px"]) <= PAIR_PX):
                    pairs.append((k, m))
                    break
        if pairs:
            _, cz, d, b = curves(each, list(range(len(rec))))
            bins = np.unique(b[np.isfinite(b)])
            diff = []
            for k, m in pairs:
                at = {v: d[k][b[k] == v].mean(0) for v in bins if (b[k] == v).any()}
                bt = {v: d[m][b[m] == v].mean(0) for v in bins if (b[m] == v).any()}
                diff.append([bt[v] - at[v] if v in at and v in bt else np.full(3, np.nan) for v in bins])
            diff = np.asarray(diff)                                       # [pairs, bins, 3] down − up
            med = np.nanmedian(diff, 0)
            se = 1.2533 * np.nanstd(diff, 0) / np.sqrt(np.sum(np.isfinite(diff), 0))
            shift = np.array([[*(np.array([rec[m]["x_px"], rec[m]["y_px"]]) - [rec[k]["x_px"], rec[k]["y_px"]]),
                               np.nan] for k, m in pairs])
            fit_shift = np.array([xyz[m] - xyz[k] for k, m in pairs])
            window_shift = np.array([centre[used[m]] - centre[used[k]] for k, m in pairs])
            row["pairs"] = dict(n=len(pairs), z_nm=(bins * BIN_NM).round(0), down_minus_up_nm=med.round(1),
                                se_nm=se.round(1), rms_nm=[rms(med[:, a]) for a in range(3)],
                                # each pair's stack fit down − up (x, y, z in its window, nm), beside its ROI's
                                # offset (px) and its window's centre on the stage (nm): the focus moved by both
                                roi_shift_px=shift[:, :2].tolist(), fit_shift_nm=fit_shift.round(1).tolist(),
                                window_shift_nm=window_shift.round(0).tolist(),
                                fit_shift_median_nm=np.nanmedian(fit_shift, 0).round(1).tolist())
        row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
        emit(row, args.out)
        print(label, f"{row['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
