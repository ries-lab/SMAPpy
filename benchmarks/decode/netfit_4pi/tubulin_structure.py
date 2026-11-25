"""Whether the tubulin localizations nobody else reports lie on the filaments (docs/4pi.md §30, stage 3).

Each table's localizations are paired with the authors' (Zhang et al. 2020) in the same frame within 64 nm, as
`benchmarks/fourpi/tubulin_compare.py` pairs them; for paired and unpaired, the distance to the nearest authors'
localization of another blink (more than 3 frames away) -- the structure the movie traces -- beside random
positions in the same field.

    python tubulin_structure.py TABLE.hdf5 [TABLE.hdf5 ...]
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

from smappy.io.formats import load

a = np.load("/Users/ries/.cache/uipsf/4pi_tubulin_2017-11-28/smappy_p1/authors_6000.npz")
ax, ay = a["x"] * 128, a["y"] * 128
af = a["frame"].astype(int)
tree = cKDTree(np.c_[ax, ay])
rng = np.random.default_rng(0)
def near(x, y, f, k=12):
    d, idx = tree.query(np.c_[x, y], k=k)
    d = np.where(np.abs(af[idx] - f[:, None]) > 3, d, np.inf)
    return d.min(1)
for path in sys.argv[1:]:
    n = load(path)[0]
    nx, ny, nf = (np.asarray(n[k], float) for k in ("x_nm", "y_nm", "frame"))
    pair = np.zeros(len(nx), bool)
    for f in range(6000):
        i = np.flatnonzero(nf == f); j = np.flatnonzero(af == f)
        if len(i) and len(j):
            d, _ = cKDTree(np.c_[ax[j], ay[j]]).query(np.c_[nx[i], ny[i]]); pair[i[d < 64]] = True
    print(path.split("/")[-1])
    for name, sel in (("paired", pair), ("unpaired", ~pair)):
        s = rng.choice(np.flatnonzero(sel), min(20000, sel.sum()), replace=False)
        d = near(nx[s], ny[s], nf[s])
        print(f"  {name:9s} {sel.sum():7d}: nearest authors' localization of another blink: median {np.median(d):3.0f} nm, "
              f"< 30 nm {np.mean(d < 30):.2f}, < 60 nm {np.mean(d < 60):.2f}")
rx, ry = rng.uniform(ax.min(), ax.max(), 20000), rng.uniform(ay.min(), ay.max(), 20000)
d = near(rx, ry, rng.integers(0, 6000, 20000).astype(float))
print(f"  random positions:  median {np.median(d):3.0f} nm, < 30 nm {np.mean(d < 30):.2f}, < 60 nm {np.mean(d < 60):.2f}")
