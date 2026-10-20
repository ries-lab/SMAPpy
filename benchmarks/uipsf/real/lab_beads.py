"""Is the learnt blur σ and the free pupil magnitude the microscope's?  The lab's own bead stacks, M2 and M5.

    python benchmarks/uipsf/real/lab_beads.py ACQUISITION_FOLDER [...] --microscope M2 \\
        [--out benchmarks/uipsf/results/lab_beads.jsonl] [--pixel 116 127] [--no-probe]

Each acquisition (a folder of Micro-Manager stacks, one bead field each) goes through SMAPpy's headless
PSF calibration as the window runs it: `calibrate.pupil.run_calibration` with the camera `camera_for`
resolves, beads found by `single_rois`, `BeadSettings(zernike_order=8, magnitude="free", blur="free")` and
100 nm beads, nothing saved; then `pupil.fit_quality` (every slice refitted by Spline 3D on the sampled
spline) and the likelihood-ratio probe of a free piezo z scale on the used beads.

The optics are the user's (2026-10-20): M2 NA 1.43, oil 1.516, 116 × 127 nm pixels (the astigmatic path's;
which axis is which ``--pixel`` and ``--include-all`` tested: 116 127 for the Evolve, 127 116 for the iXon
mounted after July 2026); M5 NA 1.50, oil 1.518, 105.6 nm square (the file's); both TetraSpeck beads
(100 nm) in water at 676 nm (the 676/37 channel).  The runs of `results/lab_beads.jsonl`, the stacks in
~/.cache/uipsf/lab_beads (`docs/uipsf/microscopes.md`):

    lab_beads.py M2/260622_AA_BeadsCalZ_638_M2 --microscope M2 --pixel 116 127
    lab_beads.py M2/2608* M2/2609* M2/2610* --microscope M2 --pixel 127 116
    lab_beads.py M5/<each> --microscope M5 --chunk 1          # one at a time: the pupil needs 128 samples

Prints and appends one JSON line per acquisition: σ as learnt with its CRLB and with the pixel's share
taken out (uipsf learns at pixel centres, so the blur holds the pixel's width: √(σ² − p²/12)), the
magnitude's and the phase's Zernike coefficients, the atlas, the probe, the beads and the fit quality.
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

MICROSCOPES = {
    "M2": dict(NA=1.43, n_immersion=1.516, pixel=(116.0, 127.0)),
    "M5": dict(NA=1.50, n_immersion=1.518, pixel=(105.6, 105.6)),
}
WAVELENGTH_NM, N_MEDIUM, BEAD_NM = 676.0, 1.33, 100.0


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def _rms(values):
    return float(math.sqrt(sum(v * v for v in values)))


def summarise(acq, microscope, pixel, run, quality, probe, seconds):
    from uipsf.optics.zernike import noll_to_nm
    cal = run.cal
    pupil = cal.model.psf.pupil
    p, std = cal.result.params.physical(), cal.result.std
    sigma = p["psf.sigma"].detach().cpu().double().numpy()
    sigma_crlb = std["psf.sigma"].detach().cpu().double().numpy()
    # the pixel's own width, taken out: a box of width p has variance p²/12
    sigma_optics = [math.sqrt(max(s * s - q * q / 12, 0.0)) for s, q in zip(sigma, pixel, strict=True)]
    mag = p["pupil.magnitude"].detach().cpu().double().numpy()
    mag_crlb = std.get("pupil.magnitude")
    phase = p["pupil.phase"].detach().cpu().double().numpy()
    phase_crlb = std["pupil.phase"].detach().cpu().double().numpy()
    mag_nm = [noll_to_nm(j) for j in pupil.magnitude_modes]
    phase_nm = [noll_to_nm(j) for j in pupil.phase_modes]
    radial = [c for c, (_, m) in zip(mag, mag_nm, strict=True) if m == 0]
    other = [c for c, (_, m) in zip(mag, mag_nm, strict=True) if m != 0]
    order = np.argsort(-np.abs(phase))[:6]
    err = quality["centred_error_nm"]
    used_rows = np.isin(quality["bead_id"], np.asarray(cal.beads.used))
    in_range = used_rows & np.isfinite(err) & (np.abs(quality["nominal_z_nm"]) <= 600)
    return dict(
        acquisition=os.path.basename(os.path.normpath(acq)), microscope=microscope, pixel_nm=list(pixel),
        NA=MICROSCOPES[microscope]["NA"], n_immersion=MICROSCOPES[microscope]["n_immersion"],
        wavelength_nm=WAVELENGTH_NM, n_medium=N_MEDIUM, bead_nm=BEAD_NM,
        camera=run.rois.camera.camera_name, stacks=len(run.rois.sources), beads=len(run.rois.records),
        used=len(cal.beads.used), rejected=cal.beads.ids("rejected"),
        iterations=cal.result.iterations, stop=cal.result.stop_reason, minutes=round(seconds / 60, 2),
        objective=float(cal.result.objective),
        sigma_nm=sigma.round(2).tolist(), sigma_crlb_nm=sigma_crlb.round(2).tolist(),
        sigma_optics_nm=[round(s, 2) for s in sigma_optics],
        magnitude_modes=list(pupil.magnitude_modes), magnitude_nm=mag_nm, magnitude=mag.round(5).tolist(),
        magnitude_crlb=None if mag_crlb is None else mag_crlb.detach().cpu().double().numpy().round(5).tolist(),
        magnitude_radial_rms=round(_rms(radial), 5), magnitude_nonsymmetric_rms=round(_rms(other), 5),
        phase_modes=list(pupil.phase_modes), phase_nm=phase_nm, phase_rad=phase.round(4).tolist(),
        phase_crlb_rad=phase_crlb.round(4).tolist(),
        phase_main=[dict(noll=int(pupil.phase_modes[i]), nm=phase_nm[i], rad=round(float(phase[i]), 4))
                    for i in order],
        atlas_mean_deviance=float(cal.atlas.mean_deviance), atlas_flagged=cal.atlas.flagged(),
        z_scale_probe=probe,
        fit_quality_median_abs_centred_nm=round(float(np.nanmedian(np.abs(err[in_range]))), 2),
        fit_quality_rms_centred_nm=round(float(np.sqrt(np.nanmean(err[in_range] ** 2))), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--microscope", required=True, choices=sorted(MICROSCOPES))
    ap.add_argument("--pixel", type=float, nargs=2, default=None, help="x y on the sample, nm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--include-all", type=int, default=0,
                    help="every bead 0 … N−1 included by hand: the same beads in runs to compare by likelihood")
    ap.add_argument("--memory", type=float, default=0.3, help="MPS memory fraction (the Mac is shared)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="emitters per chunk of the solver; the headless path has none, and M5's pupil "
                         "(NA 1.50 under oil of 1.518: 128 samples, M2's 64) does not fit 10 GB unchunked")
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, fit_quality, run_calibration

    from uipsf.diagnostics import probe_terms
    if args.chunk:
        # memory only: a chunk is a unit of memory, the sums are the same (uipsf.solve.problem)
        import functools

        import uipsf.diagnostics.probes as probes
        import uipsf.workflows.calibrate as calibrate
        calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk)
        probes.Problem = functools.partial(probes.Problem, chunk_size=args.chunk)
    scope = MICROSCOPES[args.microscope]
    pixel = tuple(args.pixel) if args.pixel else scope["pixel"]
    optics = OpticsSettings(NA=scope["NA"], wavelength_nm=WAVELENGTH_NM, n_immersion=scope["n_immersion"],
                            n_medium=N_MEDIUM, pixel_x_nm=pixel[0], pixel_y_nm=pixel[1])
    model = BeadSettings(zernike_order=8, magnitude="free", blur="free", bead_diameter_nm=BEAD_NM)
    for acq in args.acquisitions:
        files = sorted(glob.glob(os.path.join(acq, "*", "*.ome.tif")))
        t0 = time.perf_counter()
        decisions = {i: "include" for i in range(args.include_all)} or None
        run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), decisions=decisions)
        seconds = time.perf_counter() - t0
        probe = None
        if not args.no_probe:
            cal = run.cal
            # the slices learning saw: `learn_beads` thins them to MAX_DZ_NM
            from uipsf.workflows.beads import MAX_DZ_NM
            used = run.rois.batch.subset(torch.tensor(cal.beads.used)).thinned(MAX_DZ_NM)
            params = cal.result.params
            got = probe_terms(cal.model, params, used, run.rois.noise, terms=["z scale"])
            if got:
                q = got[0]
                probe = dict(gain_nats=round(float(q.gain), 1), threshold=round(q.threshold, 1),
                             flagged=bool(q.flagged),
                             z_scale=float(q.values["geometry.z_scale"].reshape(-1)[0]))
                # what the blur and the magnitude become with the z scale free: they can trade against it
                phys = q.params.physical()
                s_free = phys["psf.sigma"].detach().cpu().double().numpy()
                probe["sigma_nm"] = s_free.round(2).tolist()
                probe["sigma_optics_nm"] = [round(math.sqrt(max(v * v - w * w / 12, 0.0)), 2)
                                            for v, w in zip(s_free, pixel, strict=True)]
                probe["magnitude"] = phys["pupil.magnitude"].detach().cpu().double().numpy().round(5).tolist()
        quality = fit_quality(run)
        emit(summarise(acq, args.microscope, pixel, run, quality, probe, seconds), args.out)


if __name__ == "__main__":
    main()
