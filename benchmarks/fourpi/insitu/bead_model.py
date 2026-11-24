"""P2: the v2 bead model of 4pi_40nm_bead, as the window's 4Pi pupil method learns it; kept with its shared values."""
import os, glob, time, pickle, torch, numpy as np
from smappy.calibrate.psf_methods import FOURPI
from smappy.calibrate.fourpi_beads import FourPiBeadSettings, learn, save, steps_of
from smappy.calibrate.pupil import RunSettings
from smappy.metadata import CameraMetadata
out = os.path.expanduser("~/.cache/uipsf/p2/")
paths = sorted(glob.glob(os.path.expanduser("~/.cache/uipsf/zenodo/4pi_40nm_bead/bead*")))
inputs = {"fourpi": FourPiBeadSettings(), "camera": CameraMetadata(conversion=0.44, offset=100.0, pixelsize_um=0.129),
          "run": RunSettings(save_model=False), "detection": None}
print(inputs["fourpi"].channels)
det = FOURPI["pupil"].detect(paths, False, inputs)
print(det.text, flush=True)
beads = det.found["beads"]
t = time.time()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
settings = inputs["fourpi"]
fit = learn(beads, NA=1.35, wavelength_nm=600, n_imm=1.406, pixel_size_nm=129, steps_rad=steps_of(settings, beads.data.shape[3]),
            spec=settings.psf, bead_radius_nm=20.0, device=dev)
print("learnt in", time.time() - t, fit.result.history[-1], flush=True)
cal = save(fit, out + "bead_v2_4pi.h5", overwrite=True, layout=det.found["layout"])
shared = {k: v.detach().cpu() for k, v in fit.shared().items()}
pickle.dump({"shared": shared, "layout": det.found["layout"].to_dict(), "spec": settings.psf}, open(out + "bead_v2_shared.pkl", "wb"))
print("period", cal.period_nm, "coherence", shared["fourpi.coherence"])
