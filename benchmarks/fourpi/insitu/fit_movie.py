"""P2: Spline 4Pi + 4Pi z on the whole Nup96 movie with one model.  python fit_movie.py MODEL OUT [STOP]"""
import sys, time, glob, os
from smappy.plugins import Context
from smappy.plugins.fit import CameraSettings, OutputSettings, SourceSettings
from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
model, out = sys.argv[1], sys.argv[2]
stop = int(sys.argv[3]) if len(sys.argv) > 3 else None
paths = sorted(glob.glob(os.path.expanduser("~/.cache/uipsf/zenodo/4pi_smlm_nup96/*.mat")))
s = FourPiFitSettings(source=SourceSettings(path=paths[0], stop=stop),
                      camera=CameraSettings(conversion=0.44, offset=100.0, pixelsize_um=0.129),
                      model=FourPiModelSettings(calibration=model), output=OutputSettings(path=out))
t = time.time()
r = FourPiFit().run(Context(), s)
print(r.text)
print("rows", len(r.locs), "minutes", (time.time() - t) / 60, "columns", sorted(r.locs.keys()), flush=True)
