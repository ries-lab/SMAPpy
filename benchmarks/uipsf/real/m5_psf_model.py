"""The one PSF model end to end on M5's beads: a minimal model and a free-Zernike one, each exported as a spline that
Spline 3D fits, and an in situ run started from the minimal one.

    python benchmarks/uipsf/real/m5_psf_model.py ACQUISITION [--order 8] [--dz 100] [--chunk 2] [--memory 0.4] \\
        [--out benchmarks/uipsf/results/m5_psf_model.jsonl]

For each model (the window's defaults, the magnitude *minimal* or *Zernike*; the same beads, `m5_media.select`
and `subset`, every one included):

1. learnt by `run_calibration`, as the PSF calibration window learns it;
2. its spline (`spline_calibration`: the point-emitter view, `VectorPSF.for_emitters`) fitted to every slice of
   every bead by SMAPpy's Spline 3D (`fit_quality`, the window's *Fit quality*);
3. the z bias: the median over beads of each slice's error with each bead's own offset taken out (*centred*),
   as a curve over the stage's z; its rms and largest value within ±``--z-range``.

Then in situ from the minimal model (decision 3 of November 2026: γ and the phase above the order held): a movie
simulated with that model (`uipsf.movie`, through the same point-emitter view), learnt by `run_insitu` with
*start from* the saved minimal model, one round; the row says whether γ was held at the beads' value.

The 100 nm beads are in the model (*bead diameter*); the spline is a point's, so both models' curves include the
bead's size alike, and their difference is the models'.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from m5_media import N_IMM, N_MED, WAVELENGTH, acquisition, select, subset  # noqa: E402


def z_bias(quality: dict, z_range: float, step: float) -> dict:
    """The centred error's median per slice of the stage's z, over the used beads, and its rms and largest value
    within ±``z_range``."""
    used = quality["state"] == "used"
    ok = used & quality["fit_valid"] & np.isfinite(quality["centred_error_nm"])
    z = np.round(quality["nominal_z_nm"] / step) * step
    grid = np.unique(z[ok & (np.abs(z) <= z_range)])
    curve = np.array([np.median(quality["centred_error_nm"][ok & (z == g)]) for g in grid])
    spread = np.array([np.std(quality["centred_error_nm"][ok & (z == g)]) for g in grid])
    return dict(z_nm=grid.round(1).tolist(), bias_nm=curve.round(2).tolist(), scatter_nm=spread.round(2).tolist(),
                bias_rms_nm=round(float(np.sqrt(np.mean(curve ** 2))), 2),
                bias_max_nm=round(float(np.max(np.abs(curve))), 2),
                valid=round(float(ok.sum() / max(used.sum(), 1)), 3))


def insitu_from(model_file: str, optics, out_dir: str, order: int, device=None) -> dict:
    """A movie drawn with the minimal model's in situ view and an in situ run started from it: γ held?"""
    from smappy.calibrate.insitu import InSituSettings
    from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu
    from smappy.calibrate.pupil import BeadSettings, PSFModelSettings, RunSettings, load_start
    from smappy.metadata import CameraMetadata
    from smappy.simulate import BlinkingSettings
    from tifffile import imwrite

    import uipsf
    from uipsf.model import InSituGeometry
    from uipsf.movie import MovieSettings, simulate_movie
    beads = load_start(model_file)
    psf = beads.model.channels[0].psf
    gamma = float(beads.shared["pupil.gamma"])
    # the movie: the beads' pupil, γ and blur, molecules 0-1 µm above the coverslip seen through a 600 nm focus
    model = uipsf.ForwardModel(psf.for_emitters(15), geometry=InSituGeometry(600.0),
                               intensity=uipsf.model.Bleaching(order=0))
    shared = {k: v for k, v in beads.shared.items() if not k.startswith(("geometry.", "ch"))}
    shared["geometry.focus"] = torch.tensor(600.0)
    camera = CameraMetadata(conversion=1.0, offset=100.0, pixelsize_um=tuple(np.array(psf.grid.optics.pixel_size_nm)
                                                                              / 1000), read_noise_e=1.5)
    movie = simulate_movie(model, shared, MovieSettings(size_px=(96, 96), n_frames=400, fluorophores=300,
                                                         heights_nm=(0.0, 1000.0), on_coverslip=0.25, seed=2,
                                                         blinking=BlinkingSettings(photons=5000.0, photons_std=2000.0,
                                                                                   on_time=1.5, blinks=3.0)))
    folder = os.path.join(out_dir, "cells")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "movie.tif")
    imwrite(path, np.round(movie.frames / camera.adu_to_photons + camera.offset).astype(np.uint16))
    t0 = time.perf_counter()
    result = run_insitu([path], optics, BeadSettings(psf=PSFModelSettings(zernike_order=order)),
                        InSituModelSettings(focus_nm=600.0, max_rounds=1), InSituSettings(molecules=300),
                        camera=camera, run=RunSettings(start_from=model_file, save_model=False), device=device)
    pupil = result.model.psf.pupil
    learnt = result.cal.shared()
    return dict(beads_gamma=round(gamma, 4), insitu_held_gamma=pupil.held_gamma,
                gamma_held=pupil.held_gamma == round(gamma, 4) or pupil.held_gamma == gamma,
                gamma_learnt="pupil.gamma" in learnt, ring=bool(getattr(pupil, "ring", False)),
                held_phase_modes=sorted(pupil.held_phase), learnt_phase_modes=list(pupil.phase_modes),
                focus_nm=round(float(learnt["geometry.focus"]), 1), sigma_nm=round(float(learnt["psf.sigma"][0]), 2),
                messages=result.messages, minutes=round((time.perf_counter() - t0) / 60, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisition")
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--dz", type=float, default=100.0, help="the slices' largest spacing in learning, nm")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--memory", type=float, default=0.4, help="MPS memory fraction")
    ap.add_argument("--models", nargs="+", default=["minimal", "free"])
    ap.add_argument("--insitu-from", default=None, help="a saved minimal model: only the in situ step, from it")
    ap.add_argument("--insitu-device", default="cpu",
                    help="where in situ learns: its 300 molecules on the NA 1.5 pupil overran 10 GB of MPS memory")
    ap.add_argument("--spline-ring", action="store_true",
                    help="draw the minimal model's spline with the beads' ring damping (h as learnt): the beads' own "
                         "view less their size, to tell the ring's share of the beads' z bias from the model's")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.spline_ring:
        from uipsf.model import MinimalPupil
        MinimalPupil.without_ring = lambda self: self
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        bead_rois,
        camera_for,
        fit_quality,
        run_calibration,
    )
    name, cond, files, bead_nm = acquisition(args.acquisition)
    optics = OpticsSettings(NA=1.50, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                            n_medium=N_MED[cond["medium"]], pixel_x_nm=104.7, pixel_y_nm=106.4)
    out_dir = tempfile.mkdtemp(prefix="m5_psf_model_")
    if args.insitu_from:
        emit(dict(acquisition=name, model="in situ from minimal",
                  **insitu_from(args.insitu_from, optics, out_dir, 6, args.insitu_device)), args.out)
        return
    rois = bead_rois(files, camera_for(files[0]), dual=False)
    keep, focus, cut = select(rois, name)
    rois = subset(rois, keep, focus, args.window)
    decisions = {i: "include" for i in range(len(keep))}
    rows = {}
    for magnitude in args.models:
        t0 = time.perf_counter()
        settings = BeadSettings(psf=PSFModelSettings(zernike_order=args.order, magnitude=magnitude),
                                bead_diameter_nm=bead_nm)
        run = run_calibration(files, False, optics, settings, run=RunSettings(save_model=True), rois=rois,
                              decisions=decisions, save_to=os.path.join(out_dir, f"{magnitude}_3dcal.h5"))
        quality = fit_quality(run)
        shared = run.cal.shared()
        row = dict(acquisition=name, **cond, model=magnitude + (" (spline with the beads' ring)" if args.spline_ring
                                                                 else ""), beads=cut, used=len(run.cal.beads.used),
                   objective=round(float(run.cal.result.objective), 1), stop=run.cal.result.stop_reason,
                   sigma_nm=round(float(shared["psf.sigma"][0]), 2),
                   gamma=round(float(shared["pupil.gamma"]), 4) if "pupil.gamma" in shared else None,
                   ring_h_nm=round(float(shared["pupil.ring_h"]), 1) if "pupil.ring_h" in shared else None,
                   saved=str(run.saved), **z_bias(quality, args.z_range, float(rois.batch.z_step_nm)),
                   minutes=round((time.perf_counter() - t0) / 60, 2))
        rows[magnitude] = row
        emit(row, args.out)
    if {"minimal", "free"} <= set(rows):
        a, b = rows["minimal"], rows["free"]
        common = sorted(set(a["z_nm"]) & set(b["z_nm"]))
        d = np.array([a["bias_nm"][a["z_nm"].index(z)] - b["bias_nm"][b["z_nm"].index(z)] for z in common])
        emit(dict(acquisition=name, model="minimal - free", z_nm=common, bias_difference_nm=d.round(2).tolist(),
                  difference_rms_nm=round(float(np.sqrt(np.mean(d ** 2))), 2),
                  difference_max_nm=round(float(np.max(np.abs(d))), 2)), args.out)
    if "minimal" in rows and not args.spline_ring:
        emit(dict(acquisition=name, model="in situ from minimal",
                  **insitu_from(os.path.join(out_dir, "minimal_3dcal.uipsf.h5"), optics, out_dir, 6,
                                args.insitu_device)), args.out)


def emit(row: dict, out: str | None) -> None:
    line = json.dumps(row, default=float)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
