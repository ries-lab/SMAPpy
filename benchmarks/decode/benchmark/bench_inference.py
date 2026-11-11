"""How fast DECODE localizes on this machine: frames a second through the predictor, post-processing included.

    python bench_inference.py MODEL.safetensors FRAMES.tif [--device cuda] [--sizes 256 512] [--seconds 20]

The frames are (i)'s simulated ones (``density_15/frames.tif``: 64 x 64, 15 emitters per 40 x 40
pixels), tiled 4 x 4 and 8 x 8 into frames of 256 x 256 and 512 x 512 -- real content at a camera's
size, at no cost of drawing.  Each size is run in blocks of 64 frames, the first block a warm-up,
until ``--seconds`` have passed; the frames a second is the median block's.  TF32 off, as the
fitter runs it.  One JSON line to stdout.
"""
import argparse
import json
import platform
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("frames")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    import tifffile
    import torch

    from decode_smlm.predict import load_predictor
    from decode_smlm.utils import hardware

    device = a.device or hardware.default_device()
    predictor = load_predictor(a.model, device, batch_size=a.batch)
    tiles = tifffile.imread(a.frames).astype(np.float32)
    photons = (tiles - 398.6) * 5.0 / 100.0                 # the sets' camera (simulate_sets.CAMERA)
    out = {"device": device, "torch": torch.__version__, "batch": a.batch,
           "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
           "threads": torch.get_num_threads(), "frames_per_s": {}, "localizations_per_frame": {}}
    for size in a.sizes:
        k = size // tiles.shape[-1]
        n = 64
        rng = np.random.default_rng(0)
        block = np.stack([np.block([[photons[rng.integers(len(photons))] for _ in range(k)] for _ in range(k)])
                          for _ in range(n)])
        times, found, t_end = [], [], time.perf_counter() + a.seconds
        while time.perf_counter() < t_end or len(times) < 3:
            t0 = time.perf_counter()
            cols = predictor(block, 0)
            times.append(time.perf_counter() - t0)
            found.append(len(cols["frame"]))
        rate = n / float(np.median(times[1:]))
        out["frames_per_s"][size] = round(rate, 2)
        out["localizations_per_frame"][size] = round(float(np.mean(found[1:])) / n, 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
