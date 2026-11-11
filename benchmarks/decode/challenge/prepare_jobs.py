"""(i)'s SMLM Challenge 2016 (3D-AS): SMAPpy's bead calibration and the two DECODE jobs, the Train window's way.

    python benchmarks/decode/challenge/prepare_jobs.py

The data (``$DECODE_SMLM_DATA/challenge``, from srm.epfl.ch, ``challenge-3D-simulation``): the
astigmatic training sets MT0.N1.LD, MT0.N1.HD, MT0.N2.LD and MT0.N2.HD as stacks (64 x 64 pixels of
100 nm), their ``activations.csv`` (the truth), and the astigmatic bead z-stack (6 beads, 151 slices
10 nm apart).  N1 is a dye's photons (a median 4100 a frame), N2 a fluorescent protein's (440).

**The calibration** is SMAPpy's bead average of the bead stack (`calibrate.core`, the window's
defaults, z step 10 nm, which the file does not record), marked as taken with EM gain: the beads and
the movies were simulated with one camera, so nothing is mirrored between them.

**The camera** is the challenge's EMCCD as DECODE's own defaults have it (``param_friendly.yaml``,
written for these data): offset 100 ADU, 45 e-/ADU, EM gain 300, read noise 74.4 e-.

**The jobs**, one a noise level, prepared as the Train window prepares one (`gui.train_window`:
`run_prefit` of the LD movie, the estimates as they come, `write_job`), at the window's *LD* density
(15 a 40 x 40 pixels; the HD sets have about 5): ``challenge/jobs/N1`` and ``N2``.
"""
import os
import pathlib
from dataclasses import replace

DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm"))
ROOT = DATA / "challenge"
SETS = ("MT0.N1.LD", "MT0.N1.HD", "MT0.N2.LD", "MT0.N2.HD")
CAMERA = {"conversion": 45.0, "offset": 100.0, "em_on": True, "emgain": 300.0, "pixelsize_um": 0.1,
          "read_noise_e": 74.4}
CALIBRATION = ROOT / "cal" / "beads_AS_3dcal.h5"


def movie(name):
    return ROOT / name / "data" / f"sequence-as-stack-{name}-AS-Exp.tif"


def calibrate():
    from smappy.calibrate.core import CalibrationSettings, build_calibration, collect_beads
    from smappy.io.calibration import load_spline_calibration, save_spline_calibration
    raw = CALIBRATION.with_name("beads_AS_raw_3dcal.h5")
    beads = collect_beads([ROOT / "beads" / "sequence-as-stack-Beads-AS-Exp.tif"], CalibrationSettings(dz_nm=10.0))
    build_calibration(beads).save(raw, overwrite=True)
    save_spline_calibration(CALIBRATION, replace(load_spline_calibration(raw), em_on=True))
    return CALIBRATION


def main():
    from smappy.gui.train_window import PrepareSettings, run_prefit, write_job
    from smappy.plugins.fit import CameraSettings
    cal = CALIBRATION if CALIBRATION.exists() else calibrate()
    for noise in ("N1", "N2"):
        prepared = run_prefit(PrepareSettings(movie=str(movie(f"MT0.{noise}.LD")), calibration=str(cal), density="LD",
                                              camera=CameraSettings(**CAMERA)), report=lambda e: None)
        est = prepared.estimates
        print(noise, f"{est.n_localizations} localizations; photons {est.intensity_mu_sig}, background {est.background},"
                     f" z {est.z_nm}, on {est.on_time_frames:.2f} frames")
        folder = write_job(prepared, est, ROOT / "jobs" / noise)
        print("  ", folder)


if __name__ == "__main__":
    main()
