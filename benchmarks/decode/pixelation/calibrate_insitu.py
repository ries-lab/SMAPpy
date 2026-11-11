"""Model C: model B refined in situ on the (flipped) movie's blinking molecules, the window's defaults otherwise.

    python benchmarks/decode/pixelation/calibrate_insitu.py

U2OS cells lie on the coverslip, so its lowest molecules pin the focus (*on the coverslip*, the
default), and *hold start* keeps B's pupil where molecules cannot tell it.  The spline's z is the
height above the focal plane, as the beads' is; its step the in situ default, 20 nm (B's and SMAP's
are 10).  Saved as ``pixelation/insitu.h5``.
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import FLIPPED, OUT, camera, optics  # noqa: E402

from smappy.calibrate.insitu import InSituSettings  # noqa: E402
from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu  # noqa: E402
from smappy.calibrate.pupil import BeadSettings, RunSettings  # noqa: E402

t0 = time.perf_counter()


def progress(entry):
    if entry.get("message"):
        print(f"{time.perf_counter() - t0:7.0f} s  {entry['message']}", flush=True)


done = run_insitu([str(FLIPPED)], optics(), BeadSettings(), InSituModelSettings(), InSituSettings(),
                  run=RunSettings(start_from=str(OUT / "beads_pupil.uipsf.h5"), save_model=True),
                  camera=camera("movie"), save_to=OUT / "insitu.h5", progress=progress)
print("saved", done.saved)
for line in done.messages:
    print(line)
