"""Spline 3D on the flipped movie with one of the study's models: A (SMAP's spline), B (beads), C (in situ).

    python benchmarks/decode/pixelation/fit_spline.py A|B|C

The fit box is SMAP's fit's (13 px).  *focus* is *as calibrated*: each model as it was made, not redrawn
for a focus in the sample (B and C have a pupil model beside their spline, which *auto* would redraw).
Nothing after the fit.  Saved as ``pixelation/spline_<model>.hdf5``.
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import FLIPPED, MOVIE_CAMERA, OUT, SMAP_SPLINE  # noqa: E402

from smappy import plugins  # noqa: E402
from smappy.gui.live_session import fit_settings  # noqa: E402
from smappy.plugins import Context  # noqa: E402

MODELS = {"A": SMAP_SPLINE, "B": OUT / "beads_pupil.h5", "C": OUT / "insitu.h5"}

name = sys.argv[1]
out = OUT / f"spline_{name}.hdf5"
settings = fit_settings("Localize/Spline 3D", {
    "source.path": str(FLIPPED), "camera.conversion": MOVIE_CAMERA["conversion"], "camera.offset": MOVIE_CAMERA["offset"],
    "camera.em_on": True, "camera.emgain": MOVIE_CAMERA["emgain"], "camera.pixelsize_um": MOVIE_CAMERA["pixelsize_um"][0],
    "camera.pixelsize_y_um": MOVIE_CAMERA["pixelsize_um"][1], "model.calibration": str(MODELS[name]),
    "model.focus": "calibrated", "fit.roisize": 13, "output.path": str(out), "output.raw_frames": 0})
t0 = time.perf_counter()
result = plugins.get("Localize/Spline 3D")().run(Context(), settings)
print(result.text.splitlines()[0], f"({time.perf_counter() - t0:.0f} s)")
for line in result.text.splitlines()[1:]:
    print(line)
