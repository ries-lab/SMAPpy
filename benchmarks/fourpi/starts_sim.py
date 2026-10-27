"""The 4Pi fit's starts against the truth: molecules simulated with an experimental model, at the photon
numbers and backgrounds of the data it was learnt for.

    python benchmarks/fourpi/starts_sim.py MODEL.h5 [LOCS.pkl]

Photons and background are drawn from a fitted table of the data (``LOCS``: columns photons and
background_ch0..3), else lognormal about 3000 photons and 15 per pixel.  Each setting fits the same ROIs;
a fit is *wrong* when z or the phase is off by more than five of its own errors (CRLB), and for the wrong
ones of one start the phase and z they ended on is tallied: half a turn of phase off, and z with it, is a
second valley of the likelihood, not noise.
"""
import sys
import time

import numpy as np

from smappy import fourpi
from smappy.psf import Global4PiPSF

SETTINGS = [("1 z start", 1, 1), ("2 z starts", 2, 1), ("3 z starts", 3, 1), ("1 z, 2 phases", 1, 2),
            ("2 z, 2 phases", 2, 2)]


def main(model_path, locs_path=None, n=20000, seed=3):
    cal = fourpi.load_fourpi_calibration(model_path)
    rng = np.random.default_rng(seed)
    if locs_path:
        import pickle
        cols = pickle.load(open(locs_path, "rb"))["cols"]
        pick = rng.integers(0, len(cols["photons"]), n)
        photons = np.asarray(cols["photons"], float)[pick]
        background = np.median([np.asarray(cols[f"background_ch{c}"], float) for c in range(4)], axis=0)[pick]
    else:
        photons = rng.lognormal(np.log(3000), 0.6, n)
        background = np.full(n, 15.0)
    z_range = cal.z_index_to_nm(np.array([2.0, cal.shape[0] - 2.0]))
    # the model's expectation per pixel (simulate_rois draws Poisson; the expectation is wanted, so draw it
    # without noise from the spots and add the background before one shared Poisson draw)
    centre = 6.0
    x = centre + rng.uniform(-0.5, 0.5, n)
    y = centre + rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(*sorted(z_range * 0.8), n)
    phase = cal.phase_per_nm * z + rng.uniform(-np.pi, np.pi, n)
    clean = photons[:, None, None, None] * np.clip(fourpi.spots(cal, x, y, cal.z_nm_to_index(z), phase, 13), 0, None)
    truth = {"z_nm": z, "phase": np.angle(np.exp(1j * phase))}
    print(f"{n} molecules, photons median {np.median(photons):.0f}, background median {np.median(background):.1f} "
          f"per pixel and channel, z over {np.round(sorted(z_range * 0.8))} nm")
    for label, z_starts, phase_starts in SETTINGS:
        model = Global4PiPSF(cal, n_starts=z_starts, phase_starts=phase_starts)
        t = time.perf_counter()
        noisy = rng_noise(clean, background, seed)
        result = model.fit(noisy, iterations=50)
        seconds = time.perf_counter() - t
        v = model.unpack(result)
        dz = np.abs(v["z_nm"] - truth["z_nm"])
        dph = np.abs(np.angle(np.exp(1j * (v["phase_rad"] - truth["phase"]))))
        # off by more than the fit's own error can explain: a second valley, not noise
        wrong = (dz > 5 * v["z_err_nm"]) | (dph > 5 * v["phase_err_rad"])
        half = wrong & (np.abs(dph - np.pi) < 0.6)
        print(f"  {label:14}: {seconds:5.1f} s; wrong {100 * wrong.mean():5.2f} %, of them half a turn off in "
              f"phase {100 * half.sum() / max(wrong.sum(), 1):3.0f} % (median z off {np.median(dz[wrong]) if wrong.any() else 0:.0f} nm)")


def rng_noise(clean, background, seed):
    """The same Poisson draw for every setting: the settings fit identical data."""
    return np.random.default_rng(seed + 1).poisson(clean + background[:, None, None, None]).astype(np.float32)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
