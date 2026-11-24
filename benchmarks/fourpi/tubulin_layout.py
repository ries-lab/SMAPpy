"""(docs/4pi.md §26)  PYTHONPATH=src python benchmarks/fourpi/tubulin_layout.py

Step 1 on the tubulin data: the channel registration from the many-bead fields Cell06 -- 12 on the 4Pi-SMS
layout, compared with the authors' (``align_642_FMTtransform_20200914.mat``, from Cell12), and the camera's
conversion from the photon transfer of the movie's background (each pixel's robust variance against its
median over 1000 frames, the variance from frame-to-frame differences so that a slow background does not
count)."""
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tubulin_common import DATA, MOVIE, OUT, REGISTRATION, FIELDS, layout  # noqa: E402

from smappy.channels import bead_projections, describe_map, register_channels, save_registration  # noqa: E402
from smappy.io.dcimg import open_dcimg  # noqa: E402

OUT.mkdir(exist_ok=True)
lay = layout()
reg = register_channels(bead_projections(FIELDS, lay))
save_registration(REGISTRATION, lay.with_maps(reg.maps), reg, source="Cell06-12")
fmt = sio.loadmat(DATA / "align_642_FMTtransform_20200914.mat")
print("channel  pairs  rms px   ours: scale x, y   rotation mrad   | authors: zoom x, y   angle mrad")
for c in range(1, 4):
    d = describe_map(reg.maps[c], lay.size)
    print(f"{c + 1:7d} {reg.pairs[c]:6d} {reg.rms_px[c]:7.3f}   {d['scale'][0]:.4f} {d['scale'][1]:.4f}"
          f"   {d['rotation_mrad']:8.1f}       | {fmt['zm_all'][c, 0]:.4f} {fmt['zm_all'][c, 1]:.4f}"
          f"   {1000 * fmt['ang_all'][c, 0]:8.1f}")

# photon transfer: per pixel (variance - read variance) / (median - offset) = 1 / conversion, the read
# variance and the offset from the strip outside the channels.  Blinking raises a pixel's variance, so the
# median ratio is a lower bound of the conversion and its 20th percentile nearer the truth; a regression
# across pixels is worse still (pixels on the filaments are both brighter and noisier)
frames = np.concatenate([b for _, b in open_dcimg(MOVIE).frames(chunk=300, start=1000, stop=1300)])
frames = frames.astype(np.float32)
dif = np.diff(frames, axis=0)
var = (1.4826 * np.median(np.abs(dif - np.median(dif, 0)), 0)) ** 2 / 2
mean = np.median(frames, 0)
read, offset = np.median(var[:, 700:1300]), np.median(mean[:, 700:1300])
print(f"outside the channels (x 700-1300): offset {offset:.1f} ADU, read variance {read:.2f} ADU^2")
for c, (x, y, w, h) in enumerate(lay.regions):
    ratio = ((var[:, x + 20:x + w - 20] - read) / (mean[:, x + 20:x + w - 20] - offset)).ravel()
    print(f"channel {c + 1}: background {np.median(mean[:, x:x + w]) - offset:.0f} ADU; conversion "
          f"{1 / np.median(ratio):.3f} (median) to {1 / np.percentile(ratio, 20):.3f} (20th percentile) e-/ADU")
