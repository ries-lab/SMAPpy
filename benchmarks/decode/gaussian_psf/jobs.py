import sys
from pathlib import Path

sys.path.insert(0, "src")
from smappy.gui.train_window import PrepareSettings, run_prefit, write_job
from smappy.plugins.fit import CameraSettings

b = Path(sys.argv[1])
cam = CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1, em_on=False)
common = {"movie": str(b / "d15.tif"), "density": "manual", "density_manual": 3.125, "model": "optimised", "camera": cam}
g = run_prefit(PrepareSettings(psf="gaussian", **common))
print("gaussian: widths", [round(v) for v in g.sigma_nm], "photons", [round(v) for v in g.estimates.photons_range],
      "bg", [round(v, 1) for v in g.estimates.background], "on", round(g.estimates.on_time_frames, 2))
write_job(g, g.estimates, b / "job_gauss")
s = PrepareSettings(calibration=str(b / "real_psf.h5"), **common)
s.sample.focus = "calibrated"
p = run_prefit(s)
print("true PSF: z", [round(v) for v in p.estimates.z_nm], "photons", [round(v) for v in p.estimates.photons_range],
      "bg", [round(v, 1) for v in p.estimates.background], "on", round(p.estimates.on_time_frames, 2))
write_job(p, p.estimates, b / "job_psf")
