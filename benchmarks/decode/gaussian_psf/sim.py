"""Test movies drawn with the 'real' PSF: blinking molecules, z over +-300 nm, 500 -- 5000 photons a frame,
a flat background of 40, an sCMOS-like camera; density per 40 x 40 pixels as given."""
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, "src")
from smappy.io.calibration import load_spline_calibration
from smappy.simulate.camera import _camera, _draw_spline

cal = load_spline_calibration(sys.argv[1]); out = Path(sys.argv[2]); density = float(sys.argv[3])
n_frames = int(sys.argv[4]); size = 64; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.default_rng(seed)
on_mean = 2.0
per_frame = density * size * size / 1600            # molecules on in a frame
starts_per_frame = per_frame / on_mean
rows = []
mid = 0
for f0 in range(-4, n_frames):
    for _ in range(rng.poisson(starts_per_frame)):
        dur = rng.geometric(1 / on_mean)
        x, y = rng.uniform(-2, size + 2, 2)
        z = rng.uniform(-300, 300)
        ph = rng.uniform(500, 5000)
        for f in range(f0, f0 + dur):
            if 0 <= f < n_frames:
                rows.append((f, x, y, z, ph, mid))
        mid += 1
t = np.array(rows)
image = np.full((n_frames, size, size), 40.0)
_draw_spline(image, t[:, 0].astype(int), t[:, 1:3], t[:, 3], t[:, 4], cal)
frames = np.stack([_camera(image[k], np.random.default_rng(1000 * seed + k), 0.5, 100.0, 1.5) for k in range(n_frames)])
tifffile.imwrite(out.with_suffix(".tif"), frames, photometric="minisblack")
np.savez(out.with_suffix(".npz"), frame=t[:, 0].astype(int), x=t[:, 1], y=t[:, 2], z=t[:, 3], photons=t[:, 4],
         id=t[:, 5].astype(int))
print(out.name, len(t), "localizations,", round(len(t) / n_frames, 1), "a frame")
