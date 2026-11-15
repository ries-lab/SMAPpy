"""Control for in situ on M2's nucleus movie: a simulated movie with a known truth, learnt as the real one was.

    python insitu_control.py OUTDIR

Truth: AB's model on the glass at n_med 1.345 (every aberration, γ, σ 31.5 nm) as the in situ run that held it
carries it (`insitu_all_n1345_tilt_s315_ab_n1345`), its tilt set to 0 and its focus kept (607 nm).  The movie:
molecules 0-1000 nm above the glass, a tenth on it, the real movie's background and per-frame photons
(measured on its chosen molecules), 160 x 160 px, 4000 frames.  Then `run_insitu` as for the real movie: started
from AB (n 1.345), γ and the phase above order 8 held, focus learnt, σ round and free.  If in situ is sound
on this geometry it stays at the truth: the phase within its errors, no extra coma or spherical, the twin test
on the right sign.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

# the data are not in git: M2_DATA names the folder that holds 261112_new's sets
ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
MOVIE = "261112_MR_Intercalib2_BGAAF647_50ms_30i_nucleus_1/*_MMStack_Pos0.ome.tif"
WL = 690.0


def main():
    out = sys.argv[1]
    from smappy.calibrate.insitu import InSituSettings, insitu_rois
    from smappy.calibrate.insitu_run import InSituModelSettings, movie_rois, run_insitu
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        camera_for,
        load_start,
    )
    from smappy.simulate import BlinkingSettings

    from uipsf.movie import MovieSettings, simulate_movie

    real = sorted(glob.glob(os.path.join(ROOT, MOVIE)))
    camera = camera_for(real[0])
    rr = movie_rois(real)
    used = rr.ids("used")
    g = rr.units.gauss
    bg = float(np.nanmedian(np.asarray(g["background"])[used]))
    ph = float(np.nanmedian(np.asarray(g["photons"])[used]))
    print(f"real movie: background {bg:.1f} photons/px/frame, median photons per chosen molecule {ph:.0f}",
          flush=True)

    truth_model = load_start(os.path.join(out, "insitu_all_n1345_tilt_s315_ab_n1345_3dcal.h5"))
    model = truth_model.model
    shared = {k: torch.as_tensor(v).detach().clone() for k, v in truth_model.shared.items()}
    shared["geometry.tilt"] = torch.zeros_like(shared["geometry.tilt"])
    print("truth focus", float(shared["geometry.focus"]), "σ", shared["psf.sigma"].tolist(), flush=True)
    movie = simulate_movie(model, shared, MovieSettings(
        size_px=(160, 160), n_frames=4000, fluorophores=2500, heights_nm=(0.0, 1000.0), on_coverslip=0.1,
        blinking=BlinkingSettings(photons=1.5 * ph, photons_std=0.75 * ph), background=bg,
        read_noise_var=float(camera.readout_variance), seed=1))
    rois = insitu_rois(movie.frames, model.pixel_size_nm, InSituSettings(),
                       read_noise_var=float(camera.readout_variance), excess_noise=float(camera.excess_noise))
    rois.camera = camera
    print("simulated:", rois.messages[-1] if rois.messages else len(rois.ids("used")), flush=True)

    optics = OpticsSettings(NA=1.43, wavelength_nm=WL, n_immersion=1.516, n_medium=1.345,
                            pixel_x_nm=127.0, pixel_y_nm=116.0)
    res = run_insitu([], optics, BeadSettings(psf=PSFModelSettings(zernike_order=8)), InSituModelSettings(),
                     run=RunSettings(save_model=True, start_from=os.path.join(out, "ab_n1345_3dcal.h5")),
                     rois=rois, camera=camera, save_to=os.path.join(out, "insitu_control_3dcal.h5"),
                     progress=lambda *a: print(*a, flush=True))
    cal = res.cal
    p = cal.result.params.physical()
    pupil = cal.model.psf.pupil
    learnt = {f"Z{int(j)}": round(float(v) * WL / (2 * np.pi), 1)
              for j, v in zip(pupil.phase_modes, p["pupil.phase"].detach().cpu().reshape(-1), strict=False)}
    ab = load_start(os.path.join(out, "ab_n1345_3dcal.h5"))
    abp = ab.model.channels[0].psf.pupil
    abv = torch.as_tensor(ab.shared["pupil.phase"]).cpu().reshape(-1)
    truth = {f"Z{int(j)}": round(float(v) * WL / (2 * np.pi), 1) for j, v in zip(abp.phase_modes, abv, strict=False)}
    row = dict(focus=float(p["geometry.focus"]), sigma=p["psf.sigma"].detach().cpu().tolist(),
               messages=res.messages, used=int(len(rois.ids("used"))),
               phase={k: (truth.get(k), learnt.get(k)) for k in
                      ["Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12", "Z16", "Z17", "Z22", "Z37"]})
    print(json.dumps(row), flush=True)
    with open(os.path.join(out, "insitu_control.json"), "w") as f:
        json.dump(row, f, indent=1)


if __name__ == "__main__":
    main()
