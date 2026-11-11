"""Model B: SMAPpy's pupil model (uiPSF) of the example's 11 bead stacks, the window's defaults otherwise.

    python benchmarks/decode/pixelation/calibrate_beads.py

Saved as ``pixelation/beads_pupil.h5`` (the spline the fitters read, with the pupil model beside it).
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import BEADS, OUT, camera, optics  # noqa: E402

from smappy.calibrate.pupil import BeadSettings, RunSettings, run_calibration  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()


def progress(entry):
    if entry.get("message"):
        print(f"{time.perf_counter() - t0:7.0f} s  {entry['message']}", flush=True)


run = run_calibration([str(p) for p in BEADS], False, optics(), BeadSettings(), camera=camera("beads"),
                      run=RunSettings(save_model=True), save_to=OUT / "beads_pupil.h5", progress=progress)
print("saved", run.saved)
for line in getattr(run, "messages", []) or []:
    print(line)
