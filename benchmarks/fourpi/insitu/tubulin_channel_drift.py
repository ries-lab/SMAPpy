"""(docs/4pi.md §34)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_channel_drift.py MODEL

Spline 4Pi's *channel drift* on Cell01 with ``MODEL`` (x and y free per channel on blocks spread over the movie,
`FourPiFit.channel_drift`, kept beside the movie): how far each channel's molecules sit from where the
registration puts them -- a constant part is the registration's (or the model's centring), the rest drift."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_misfit import fitter

plugin, settings = fitter(sys.argv[1])
settings.model.channel_drift = True
source = plugin.open_source(settings)
camera = plugin._camera(settings)
drift = plugin.channel_drift(settings, camera, plugin.region_finder(settings, camera), source)
off = np.asarray(drift.offsets)
print("frames", np.asarray(drift.frames)[[0, -1]])
for c in range(1, off.shape[1]):
    print(f"ch{c + 1}: mean dx, dy {off[:, c].mean(0).round(3)} px; range over time "
          f"{np.ptp(off[:, c], axis=0).round(3)} px")
