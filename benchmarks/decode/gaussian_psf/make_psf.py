"""The 'real' PSF of the benchmark: a vector pupil (NA 1.45, oil, water sample, 670 nm, 100 nm pixels) with mild
aberrations (astigmatism, coma, spherical, a few tens of nm) and a 40 nm blur, sampled into a spline."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "src")
from smappy.calibrate.pupil import spline_calibration
from smappy.io.calibration import save_spline_calibration
from uipsf.export import from_description
from uipsf.optics import Optics

out = Path(sys.argv[1])
desc = {"kind": "zernike", "phase_nm": {"5": 12.0, "6": -8.0, "7": 10.0, "8": -6.0, "11": 25.0},
        "magnitude": {}, "sigma_nm": [40.0, 40.0]}
optics = Optics(NA=1.45, wavelength_nm=670.0, n_imm=1.518, n_med=1.33, pixel_size_nm=(100.0, 100.0))
learned = from_description(desc, optics, roi_px=27)
cal = spline_calibration(learned, 20.0, 800.0, 27, camera=SimpleNamespace(is_em=False, pixelsize_um=[0.1, 0.1]))
save_spline_calibration(out, cal, overwrite=True)
psf = cal.psf
c = psf.shape[1] // 2
for z in (-400, -200, 0, 200, 400):
    k = round(cal.z_nm_to_index(z))
    p = psf[k] / psf[k].sum()
    r2 = ((np.arange(27) - c) ** 2)
    sx = np.sqrt((p.sum(0) * r2).sum()); sy = np.sqrt((p.sum(1) * r2).sum())
    print(f"z {z:5d} nm: peak {p.max():.3f}, rms width x {sx*100:.0f} y {sy*100:.0f} nm, in 7x7 {p[c-3:c+4, c-3:c+4].sum():.2f}")
