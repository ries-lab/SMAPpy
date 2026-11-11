"""(i)'s simulated sets: single frames of emitters drawn as the example package's training draws them.

    python benchmarks/decode/benchmark/simulate_sets.py [--calibration A] [--out DIR]

One set per density -- 1, 5, 15, 25 and 50 emitters per 40 x 40 pixels, the plan's -- each of frames
of 64 x 64 pixels and about ``EMITTERS`` emitters in all, independent frames (the package's models see
one frame at a time).  Each emitter: x and y uniform over the frame (pixel centres at whole numbers),
z uniform within +-``Z_NM``, photons normal 7000 +- 3000 cut at 1000, the frame's background
uniform in 20 -- 200 photons a pixel -- the package's training (``param_run.yaml``), so that every
model is asked about data like its training.  The PSF is the calibration's spline, drawn by
`smappy.simulate` (`_draw_spline`, `evaluate_spline`); the camera the training's EMCCD (Poisson,
gamma EM of gain 100, 5 e-/ADU, read noise 58.8 e-, offset 398.6; `_camera`).  The number of emitters
a frame is Poisson, as DECODE draws it.

Writes ``<out>/density_<d>/frames.tif`` (uint16) and ``truth.npz`` (frame, x_px, y_px, z_nm, photons;
background per frame), and ``sets.json`` with the settings and seeds.  A is SMAP's spline of the
example package, in the beads' orientation as SMAPpy loads it; the models trained on it see frames
in that orientation, so nothing is flipped anywhere.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pixelation"))
from common import DATA, SMAP_SPLINE  # noqa: E402

DENSITIES = (1, 5, 15, 25, 50)          # emitters per 40 x 40 pixels
SIZE = 64
EMITTERS = 20000
Z_NM = 700.0
PHOTONS = (7000.0, 3000.0, 1000.0)      # mean, spread, the least
BACKGROUND = (20.0, 200.0)
CAMERA = dict(conversion=5.0, offset=398.6, em_gain=100.0, read_noise_adu=58.8 / 5.0)
PIXEL_NM = (117.0, 127.0)
CALIBRATIONS = {"A": SMAP_SPLINE}


def simulate(cal, density: int, seed: int):
    from smappy.simulate.camera import _camera, _draw_spline
    rng = np.random.default_rng(seed)
    per_frame = density * SIZE * SIZE / (40 * 40)
    n_frames = int(np.ceil(EMITTERS / per_frame))
    counts = rng.poisson(per_frame, n_frames)
    frame = np.repeat(np.arange(n_frames), counts)
    n = len(frame)
    x = rng.uniform(-0.5, SIZE - 0.5, n)
    y = rng.uniform(-0.5, SIZE - 0.5, n)
    z = rng.uniform(-Z_NM, Z_NM, n)
    photons = np.maximum(rng.normal(PHOTONS[0], PHOTONS[1], n), PHOTONS[2])
    background = rng.uniform(*BACKGROUND, n_frames)
    image = np.empty((n_frames, SIZE, SIZE))
    image[:] = background[:, None, None]
    _draw_spline(image, frame, np.stack([x, y], 1), z, photons, cal)
    frames = np.empty(image.shape, np.uint16)
    for k in range(n_frames):
        frames[k] = _camera(image[k], np.random.default_rng([seed, 7, k]), CAMERA["conversion"], CAMERA["offset"],
                            CAMERA["read_noise_adu"], CAMERA["em_gain"])
    truth = dict(frame=frame, x_px=x, y_px=y, z_nm=z, photons=photons, background=background)
    return frames, truth


def main():
    import tifffile
    from smappy.io.calibration import load_spline_calibration
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="A", choices=sorted(CALIBRATIONS))
    ap.add_argument("--out", type=pathlib.Path, default=DATA / "benchmark")
    a = ap.parse_args()
    cal = load_spline_calibration(CALIBRATIONS[a.calibration])
    root = a.out / a.calibration
    sets = {}
    for i, d in enumerate(DENSITIES):
        seed = 1000 + i
        frames, truth = simulate(cal, d, seed)
        folder = root / f"density_{d}"
        folder.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(folder / "frames.tif", frames, photometric="minisblack")
        np.savez(folder / "truth.npz", **truth)
        sets[d] = {"seed": seed, "frames": len(frames), "emitters": len(truth["frame"])}
        print(f"density {d}: {len(frames)} frames, {len(truth['frame'])} emitters", flush=True)
    meta = {"calibration": str(CALIBRATIONS[a.calibration]), "size": SIZE, "z_nm": Z_NM, "photons": PHOTONS,
            "background": BACKGROUND, "camera": CAMERA, "pixel_nm": PIXEL_NM, "sets": sets}
    (root / "sets.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
