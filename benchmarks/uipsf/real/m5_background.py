"""Is the core every M5 model misses a normalisation, not the PSF?  Background, haze, per-slice light, masks.

    python benchmarks/uipsf/real/m5_background.py ACQUISITION [...] [--radii 6 4] [--dz 100] [--memory 0.7] \\
        [--out benchmarks/uipsf/results/m5_background.jsonl]

Every model, whatever its magnitude or blur, has its peak 4-12 % below the bead average at z = ±300 nm, less in
focus, with a ring of the opposite sign just outside (`m5_media_psfs.py`), and σ(z) does not move it
(`m5_zblur.py`).  The user's suspicion (1 November 2026): the core's few pixels are outweighed by the many
around it, so anything that sets the amplitude between model and data -- a background that is not
constant, a haze or halo the model has no term for, light that changes from slice to slice -- could
make the shared model widen the PSF to fit the wings and lose the peak.  Per acquisition, free Zernikes to
order 8 and σx, σy free, on the consensus beads:

1. *reference*, every pixel of the ROI;
2. *masked*: only the pixels within ``--radii`` pixels of the ROI's centre count in the likelihood
   (`Batch.mask`), the model drawn as before -- if the wings set the shape, the core comes back;
3. *probes* on the reference (`diagnostics.probe_terms`): a planar background per bead, a z scale and a
   photon factor per slice shared by the beads, each refitted from the fit, its gain in nats and its values.

Per fit: σ, the objective, the bends (`bead_slices.measure`), the core -- model minus bead average at the
peak pixel, % of the peak, at z = −300, 0, +300 -- and the radial profile of model minus data in rings of one
pixel at five planes (`model_views.pupil_views`: both aligned and background removed alike), whose sum
beyond the core says whether the data hold light the model has no place for.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from bead_slices import emit, measure  # noqa: E402,I001
from m5_media import GRID_NA, N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402

PIXEL = (104.7, 106.4)
PLANES_NM = (-600.0, -300.0, 0.0, 300.0, 600.0)


def core_and_rings(views) -> dict:
    """The model minus the bead average at the peak pixel (% of the peak) at ±300 and 0, and in rings."""
    data, model, z = views.data[0], views.model[0], views.z_nm
    Y, X = data.shape[-2:]
    yy, xx = np.mgrid[:Y, :X]
    out = dict(core_pct={}, rings_pct={}, wings_pct={})
    for p in PLANES_NM:
        k = int(np.argmin(np.abs(z - p)))
        d, m = data[k], model[k]
        top = float(np.nanmax(d))
        cy, cx = np.unravel_index(np.nanargmax(d), d.shape)
        r = np.hypot(yy - cy, xx - cx).round().astype(int)
        diff = (m - d) / top * 100
        out["core_pct"][int(p)] = round(float(diff[cy, cx]), 2)
        out["rings_pct"][int(p)] = [round(float(np.nanmean(diff[r == i])), 3) for i in range(int(r.max()) + 1)]
        # the light beyond the core, model minus data, as % of the data's own light in the plane
        total = float(np.nansum(d))
        out["wings_pct"][int(p)] = round(float(np.nansum((m - d)[r > 3])) / total * 100, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--radii", type=float, nargs="*", default=[6.0, 4.0], help="masks: the pixels kept, radius px")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--memory", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_background.jsonl"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.model_views import pupil_views
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration

    import uipsf.diagnostics.probes as probes
    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    from uipsf.device import on_model_device
    from uipsf.model.pupil import ZernikePupil
    from uipsf.solve import Problem
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=1, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise, chunk_size=int(2 * args.window // args.dz + 1))
    probes.Problem = functools.partial(Problem, chunk_size=1)            # the probes' refits, one bead at a time
    calibrate.ZernikePupil = ZernikePupil
    done = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    have = {(r["acquisition"], r["fit"]) for r in done}

    for path in args.acquisitions:
        name, cond, files, bead_nm = acquisition(path)
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, cut = select(rois, name)
        rois = subset(rois, keep, focus, args.window)
        optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
        model = BeadSettings(zernike_order=ORDER, magnitude="free", blur="free", blur_nm=30.0, bead_diameter_nm=bead_nm)
        Y, X = rois.batch.data.shape[-2:]
        yy, xx = np.mgrid[:Y, :X]
        for radius in [None] + list(args.radii):
            fit = "reference" if radius is None else f"mask r{radius:g}"
            if (name, fit) in have:
                continue
            t0 = time.perf_counter()
            r = copy.copy(rois)
            if radius is not None:
                keep_px = torch.as_tensor(np.hypot(yy - Y // 2, xx - X // 2) <= radius)
                base = r.batch.mask if r.batch.mask is not None else torch.ones_like(r.batch.data, dtype=torch.bool)
                r.batch = dataclasses.replace(r.batch, mask=base & keep_px.to(base.device))
            run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=r,
                                  decisions={i: "include" for i in range(len(keep))})
            cal = run.cal
            p = cal.result.params.physical()
            row = dict(acquisition=name, **cond, fit=fit, radius_px=radius, roi_px=[Y, X], used=len(cal.beads.used),
                       objective=round(float(cal.result.objective), 1),
                       sigma_nm=p["psf.sigma"].detach().cpu().double().numpy().round(2).tolist(),
                       atlas_mean_deviance=round(float(cal.atlas.mean_deviance), 3))
            row.update(core_and_rings(pupil_views(run)))
            row.update(measure(run, 600.0, centre_nm=[rois.records[i]["focus_nm"] for i in cal.beads.used]))
            if radius is None and not args.no_probes:
                batch = on_model_device(rois.batch.thinned(args.dz), cal.model).subset(torch.tensor(cal.beads.used))
                found = probes.probe_terms(cal.model, cal.result.params, batch, rois.noise)
                row["probes"] = {q.term: dict(gain=round(float(q.gain), 1), threshold=round(q.threshold, 1),
                                              values={k: v.cpu().double().numpy().round(4).tolist()
                                                      for k, v in q.values.items()}) for q in found}
            row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
            emit(row, args.out)
            print(name, fit, "σ", row["sigma_nm"], "core", row["core_pct"], "wings", row["wings_pct"],
                  {k: v["gain"] for k, v in row.get("probes", {}).items()}, f"{row['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
