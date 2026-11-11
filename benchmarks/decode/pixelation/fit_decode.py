"""Localize/DECODE on the example movie with the study's model A, B or C.

    python benchmarks/decode/pixelation/fit_decode.py A|B|C

A is the port's reference training on SMAP's spline (gpuvm ``jobs/port_run0``), trained before the
model file recorded its calibration's EM state: the fitter flips nothing for it, so it runs on the
flipped movie, which is in its calibration's orientation.  B and C record beads without EM gain
(C: learnt on the flipped movie, which is the beads' orientation), so they run on the movie as it was
taken and the fitter flips it into the beads' orientation itself (`fit_decode._mirrored`); their
positions come back on that movie, and are mirrored into the flipped movie's for the analysis
(``x_px -> width - 1 - x_px``, written as ``x_nm`` again), so that every fit is in one frame.
Threshold DECODE's 0.1; the analysis filters further.  Saved as ``pixelation/decode_<model>.hdf5``.
"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import FLIPPED, MOVIE, MOVIE_CAMERA, OUT, PIXEL_NM  # noqa: E402

from smappy import plugins  # noqa: E402
from smappy.gui.live_session import fit_settings  # noqa: E402
from smappy.plugins import Context  # noqa: E402

name = sys.argv[1]
model = OUT / f"decode_{name}.safetensors"
source = FLIPPED if name == "A" else MOVIE
out = OUT / f"decode_{name}.hdf5"
settings = fit_settings("Localize/DECODE", {
    "source.path": str(source), "camera.conversion": MOVIE_CAMERA["conversion"], "camera.offset": MOVIE_CAMERA["offset"],
    "camera.em_on": True, "camera.emgain": MOVIE_CAMERA["emgain"], "camera.pixelsize_um": MOVIE_CAMERA["pixelsize_um"][0],
    "camera.pixelsize_y_um": MOVIE_CAMERA["pixelsize_um"][1], "model.path": str(model), "model.threshold": 0.1,
    "output.path": str(out), "output.raw_frames": 0})        # rewritten below, mirrored where it must be
t0 = time.perf_counter()
result = plugins.get("Localize/DECODE")().run(Context(), settings)
locs = result.locs
mirrored = name != "A"
if mirrored:                                   # into the flipped movie's frame, as the Spline 3D fits are
    import tifffile
    with tifffile.TiffFile(MOVIE) as t:
        width = t.pages[0].shape[1]
    x_px = np.asarray(locs["x_nm"], np.float64) / PIXEL_NM[0]
    locs.columns["x_nm"] = ((width - 1 - x_px) * PIXEL_NM[0]).astype(np.float32)
locs.metadata["pixelation_study"] = {"model": name, "source": str(source), "mirrored_back": mirrored}
from smappy.io.hdf5 import save_localizations  # noqa: E402
save_localizations(out, locs, {"decode": {"model": str(model)}, "pixelation_study": locs.metadata["pixelation_study"]})
print(result.text.splitlines()[0], f"({time.perf_counter() - t0:.0f} s)")
for line in result.text.splitlines()[1:]:
    print(line)
