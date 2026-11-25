"""(docs/4pi.md §34)  PYTHONPATH=src python benchmarks/fourpi/insitu/tubulin_share_check.py MODEL [photons] [n_per_frame]

What the *channels* figure shows when the model is the truth: frames drawn with MODEL (`fourpi.render_frames`) at
the tubulin's brightness and background (5500 photons, 34 a pixel per channel), the cavity's phase random per
frame, fitted by the 4Pi engine with the same model as Spline 4Pi fits them, and the split's residual measured as
`tubulin_misfit.py` measures it on the movie.  Any residual here is the figure's own, not the model's."""
import sys

import numpy as np

from smappy import fourpi
from smappy.channels import from_corners
from smappy.locs import concat
from smappy.metadata import CameraMetadata
from smappy.multichannel import MultiChannelEngine
from smappy.pipeline import FitSettings
from smappy.plugins.fit import DetectionSettings
from smappy.psf import Global4PiPSF

cal = fourpi.load_fourpi_calibration(sys.argv[1])
photons = float(sys.argv[2]) if len(sys.argv) > 2 else 5500.0
per_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 4
layout = from_corners([0, 110, 220, 330], [0] * 4, 100, 100, flip_x=[0, 0, 1, 1])
rng = np.random.default_rng(0)
n_frames = 3000
frame = np.repeat(np.arange(n_frames), per_frame)
x, y = rng.uniform(10, 90, (2, len(frame)))
z = rng.uniform(-500, 500, len(frame))
phase0 = rng.uniform(-np.pi, np.pi, n_frames)[frame]
camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.128)
engine = MultiChannelEngine(camera, DetectionSettings().finder(), Global4PiPSF(cal), layout, FitSettings(roisize=13))
tables = []
for first in range(0, n_frames, 500):
    sel = (frame >= first) & (frame < first + 500)
    frames = fourpi.render_frames(cal, layout, frame[sel] - first, x[sel], y[sel], z[sel], photons,
                                  phase0=phase0[sel], n_frames=500, background=34.0, seed=first)
    engine.push(layout.cut(frames), first)
    tables.append(engine.flush())
locs = concat([t for t in tables if t is not None])
names = [f"roi_photons_ch{c}" for c in range(cal.n_channels)]
sums = np.stack([np.asarray(locs[n], float) for n in names], 1)
phase, zf = np.asarray(locs["phase_rad"], float), np.asarray(locs["z_env_nm"], float)
ends = np.sort(cal.z_index_to_nm(np.array([1.0, cal.shape[0] - 2.0])))
inside = (zf > ends[0]) & (zf < ends[1])
binned = fourpi.channel_shares(sums[inside], phase[inside],
                               model=fourpi.model_shares(cal, zf[inside], phase[inside], 13))
c = binned["centres"]
r = binned["median"] - binned["model"]
near_pi = np.argsort(np.abs(np.abs(c) - np.pi))[:4]
A = np.c_[np.ones(len(c)), np.cos(c), np.sin(c)]
amp = [np.hypot(*np.linalg.lstsq(A, binned[k], rcond=None)[0][1:]) for k in ("median", "model")]
print(f"{int(inside.sum())} fits; residual rms per channel {np.sqrt(np.nanmean(r ** 2, 0)).round(4).tolist()}, "
      f"near ±pi {np.nanmean(r[near_pi], 0).round(4).tolist()}; swing data / model {(amp[0] / amp[1]).round(3).tolist()}")
