"""The example movie flipped along x into the beads' orientation (`common.py` says why), a frame at a time.

    python benchmarks/decode/pixelation/flip_movie.py
"""
import sys

import numpy as np
import tifffile

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import FLIPPED, MOVIE  # noqa: E402

with tifffile.TiffFile(MOVIE) as src, tifffile.TiffWriter(FLIPPED, bigtiff=True) as out:
    n = len(src.pages)
    for i, page in enumerate(src.pages):
        out.write(np.ascontiguousarray(page.asarray()[:, ::-1]), contiguous=True, photometric="minisblack")
        if i % 2000 == 0:
            print(f"{i} of {n}", flush=True)
print(f"{n} frames to {FLIPPED}")
