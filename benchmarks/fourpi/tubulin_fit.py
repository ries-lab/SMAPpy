"""(docs/4pi.md §26)  PYTHONPATH=src python benchmarks/fourpi/tubulin_fit.py MODEL [frames] [drift]

Step 3a on the tubulin data: *Spline 4Pi* and, after it, *4Pi z* on the first two DCIMG files of Cell01 (6000
frames of the 360 000), as the GUI would run them: the 4Pi-SMS layout, the channels registered from the bead
fields Cell06 -- 12 (a registration file, `tubulin_layout.py`), the camera of `tubulin_common`, the model
``MODEL`` (a ``smappy_p1/*_4pi.h5``), everything else at its default; ``drift`` turns on the channel drift.
The table goes to ``smappy_p1/cell01_<model>.hdf5``; `tubulin_compare.py` sets it against the authors'."""
import sys
import time
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_common import CONVERSION, MOVIE, OFFSET, OUT, PIXEL_UM, REGISTRATION

from smappy.channels import FourPiChannelsSettings
from smappy.plugins import Context
from smappy.plugins.fit import CameraSettings, OutputSettings, SourceSettings
from smappy.plugins.fit_4pi import (
                                    FourPiFit,
                                    FourPiFitSettings,
                                    FourPiModelSettings,
                                    FourPiRegistrationSettings,
)

model = Path(sys.argv[1])
frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
drift = len(sys.argv) > 3 and sys.argv[3] == "drift"
name = model.name.replace("_4pi.h5", "") + ("_drift" if drift else "") + (f"_{frames}" if frames else "")
settings = FourPiFitSettings(
    source=SourceSettings(path=str(MOVIE), stop=frames),
    camera=CameraSettings(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM),
    channels=FourPiChannelsSettings(layout="yale"),
    model=FourPiModelSettings(calibration=str(model), channel_drift=drift,
                              registration=FourPiRegistrationSettings(source="file", path=str(REGISTRATION))),
    output=OutputSettings(path=str(OUT / f"cell01_{name}.hdf5")))
t0 = time.time()
result = FourPiFit().run(Context(), settings)
seconds = time.time() - t0
n = len(result.locs)
print(f"{name}: {n} localizations in {seconds:.0f} s ({(frames or 6000) / seconds:.1f} frames/s)")
print(result.text)
