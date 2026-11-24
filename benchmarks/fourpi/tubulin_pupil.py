"""(docs/4pi.md §26)  PYTHONPATH=src python benchmarks/fourpi/tubulin_pupil.py [wavelength_nm] [bead_nm] [held|free]

Step 2b on the tubulin data: uiPSF's 4Pi pupil model (`fourpi_beads.learn`) learnt from the three one-bead
scans Cell03 -- 05 together.  Each bead -- each scan -- has its own phase (``emitter.phase``) and its own
position (``emitter.xyz``: the stage's offset between scans); the channels' maps held at the bead fields'
registration (``held``, the default: the three beads lie within a pixel of each other and fix no turn or
scale), or learnt from the beads (``free``).  The model's fringe period is checked against the beads' own
(`fourpi_beads.period_mismatch`), and it is fitted back to the scans.  Saved as
``smappy_p1/pupil_<wavelength>_<bead>_<maps>_4pi.h5``.  Optics: NA 1.35, silicone immersion 1.406 (the
4Pi-SMS's objectives), the sample index-matched, as the PSF window's 4Pi mode takes them."""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_common import OUT, PIXEL_UM, REGISTRATION, SCANS, camera

from smappy.calibrate import fourpi_beads as fb
from smappy.channels import FourPiChannelsSettings

wavelength = float(sys.argv[1]) if len(sys.argv) > 1 else 620.0
bead_nm = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
maps = sys.argv[3] if len(sys.argv) > 3 else "held"
settings = fb.FourPiBeadSettings(channels=FourPiChannelsSettings(layout="yale"), mirror_steps="",
                                 registration=fb.ScanRegistrationSettings(source="file", path=str(REGISTRATION)))
beads, layout, _ = fb.scan_beads([str(p) for p in SCANS], settings, camera())
try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else None
except ImportError:
    device = None
t0 = time.time()
fit = fb.learn(beads, NA=1.35, wavelength_nm=wavelength, n_imm=1.406, pixel_size_nm=PIXEL_UM * 1000,
               steps_rad=fb.steps_of(settings, 1), bead_radius_nm=bead_nm / 2, device=device,
               maps=layout.maps if maps == "held" else None)
seconds = time.time() - t0
name = f"pupil_{wavelength:.0f}_{bead_nm:.0f}_{maps}"
cal = fb.save(fit, OUT / f"{name}_4pi.h5", layout=layout)
period = 2 * np.pi / abs(cal.phase_per_nm)
beads_period = fb.beads_period_nm(beads)
result = fit.result
print(f"{name}: {seconds:.0f} s, objective {float(result.objective):.1f}, model period {period:.1f} nm, "
      f"beads' {beads_period:.1f} nm; {fb.period_mismatch(period, beads_period, wavelength) or 'periods agree'}")
p = cal.parameters
print("  coherence", np.round(p["coherence"], 3).tolist(), "modulation", np.round(p["modulation"], 3).tolist())
print("  deltas", np.round(p["deltas_rad"], 3).tolist(), "offset_nm", np.round(p["offset_nm"], 1).tolist(),
      "sigma", np.round(p["sigma_nm"], 1).tolist())
fits = fb.bead_fits(beads, cal)
for b in range(beads.data.shape[0]):
    print(f"  scan {b}: z slope {fits['slope'][b]:.3f}, scatter {fits['scatter'][b]:.1f} nm; "
          f"phase slope {fits['phase_slope'][b]:.3f}")
