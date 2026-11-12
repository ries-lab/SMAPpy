"""Does a model's claimed precision hold, and what does the consecutive-frame check see of it?

    python benchmarks/decode/suite/precision_check.py [--context fig4e] [--run decode_port]

The consecutive-frame check (`report.score_real`) compares the spread of a molecule's two positions
in frames t and t + 1 with the root of their two claimed variances: 1 when the claims are right and
the two localizations independent.  A model that reads several frames (DECODE's 3-frame models: t-1,
t, t+1) localizes t and t + 1 from two shared frames, so their errors correlate and the difference is
smaller than the claims say -- on the Fig. 4e movie 0.3 of them, against about 1 for the example's
1-frame model.  This draws a movie with the truth known to tell the two apart:

**The movie**: 64 x 64 frames of the context's calibration and camera (mirrored as an EMCCD reads
them), molecules fixed in place and on for whole frames, each for a geometric number of frames of
mean ``ON_FRAMES`` from a random start, about ``DENSITY`` on per 40 x 40 pixels; photons normal
5000 +- 2000 (at least 1000), background 50 -- 150 a pixel per frame, z uniform in +-700 nm -- inside
the Fig. 4e training's ranges.

**Measured**: each localization's error against its own claim (``x_err_nm``, ``z_err_nm``), the
robust width of ``error / claim`` (1 where the claims are right), the correlation of a molecule's
errors in frames ``lag`` apart, and the consecutive-frame ratio at each lag from the truth's pairs
(the same molecule): the robust width of the pair's difference over the root of its two claims.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402
import sets as S  # noqa: E402

SIZE = 64
FRAMES = 3000
DENSITY = 5.0                   # molecules on per 40 x 40 pixels
ON_FRAMES = 4.0
LAGS = (1, 2, 3, 4, 6)
CAMERAS = {"fig4e": S.SETS["fig4e"]["camera"],
           "B": {**S.SIM_CAMERA}}


def simulate(cal, camera, seed=0):
    from smappy.simulate.camera import _camera, _draw_spline
    rng = np.random.default_rng(seed)
    n = int(DENSITY * SIZE * SIZE / 1600 * FRAMES / ON_FRAMES)
    start = rng.integers(-int(3 * ON_FRAMES), FRAMES, n)
    length = rng.geometric(1 / ON_FRAMES, n)
    x, y = rng.uniform(-0.5, SIZE - 0.5, n), rng.uniform(-0.5, SIZE - 0.5, n)
    z = rng.uniform(-700, 700, n)
    photons = np.maximum(rng.normal(5000, 2000, n), 1000)
    mol = np.repeat(np.arange(n), length)
    frame = start[mol] + (np.arange(len(mol)) - np.repeat(np.cumsum(length) - length, length))
    on = (frame >= 0) & (frame < FRAMES)
    mol, frame = mol[on], frame[on]
    background = rng.uniform(50, 150, FRAMES)
    image = np.empty((FRAMES, SIZE, SIZE))
    image[:] = background[:, None, None]
    _draw_spline(image, frame, np.stack([x[mol], y[mol]], 1), z[mol], photons[mol], cal)
    frames = np.empty(image.shape, np.uint16)
    for k in range(FRAMES):
        frames[k] = _camera(image[k], np.random.default_rng([seed, 7, k]), camera["conversion"], camera["offset"],
                            camera.get("read_noise_e", 0.0) / camera["conversion"], camera["emgain"])
    truth = {"frame": frame, "mol": mol, "x_px": x[mol], "y_px": y[mol], "z_nm": z[mol], "photons": photons[mol]}
    return frames, truth


def fit(movie, camera, model, out, threshold=0.1):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.plugins import Context
    s = fit_settings("Localize/DECODE", {"source.path": str(movie), **{f"camera.{k}": v for k, v in camera.items()},
                                         "model.path": str(model), "model.threshold": threshold,
                                         "output.path": str(out), "output.raw_frames": 0})
    plugins.get("Localize/DECODE")().run(Context(), s)


def robust(v):
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def measure(truth, locs, camera, keep):
    from score_sets import match
    px = (camera["pixelsize_um"] * 1000, camera.get("pixelsize_y_um", camera["pixelsize_um"]) * 1000)
    ok = R.kept(locs, keep)
    f = {"frame": np.asarray(locs["frame"])[ok].astype(np.int64), "x_px": np.asarray(locs["x_nm"], float)[ok] / px[0],
         "y_px": np.asarray(locs["y_nm"], float)[ok] / px[1], "z_nm": np.asarray(locs["z_nm"], float)[ok]}
    err = {a: np.asarray(locs[f"{a}_err_nm"], float)[ok] for a in ("x", "z")}
    inside = lambda x, y: (x >= 6.5) & (x < SIZE - 7.5) & (y >= 6.5) & (y < SIZE - 7.5)   # noqa: E731
    keep_t = inside(truth["x_px"], truth["y_px"])
    t = {k: v[keep_t] for k, v in truth.items()}
    ti, fi = match(t, f, px)
    d = {"x": (f["x_px"][fi] - t["x_px"][ti]) * px[0], "z": f["z_nm"][fi] - t["z_nm"][ti]}
    out = {"matched": int(len(ti)), "recall": float(len(ti) / len(t["frame"]))}
    for a in ("x", "z"):
        out[f"error_over_claim_{a}"] = robust(d[a] / err[a][fi])
    # the same molecule matched in frames lag apart: its errors' correlation and the check's ratio
    key = {(m, fr): k for k, (m, fr) in enumerate(zip(t["mol"][ti], t["frame"][ti]))}
    for lag in LAGS:
        a = np.array([k for (m, fr), k in key.items() if (m, fr + lag) in key])
        b = np.array([key[(t["mol"][ti][k], t["frame"][ti][k] + lag)] for k in a])
        row = {"pairs": int(len(a))}
        for ax in ("x", "z"):
            e1, e2 = d[ax][a], d[ax][b]
            s1, s2 = err[ax][fi][a], err[ax][fi][b]
            good = np.abs(e1 / s1) < 5
            good &= np.abs(e2 / s2) < 5
            row[f"corr_{ax}"] = float(np.corrcoef(e1[good] / s1[good], e2[good] / s2[good])[0, 1])
            # each pair normalised by its own claims: a pooled spread over the root mean claim is not 1
            # even for right claims -- a mixture's robust width is below its root mean square
            row[f"ratio_{ax}"] = robust((e1 - e2) / np.sqrt(s1 ** 2 + s2 ** 2))
        out[f"lag_{lag}"] = row
    return out


def main():
    import tifffile
    from smappy.io.calibration import load_spline_calibration, matched_to_em
    from smappy.io.hdf5 import load_localizations
    from run import load_run, model_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="fig4e")
    ap.add_argument("--run", default="decode_port")
    a = ap.parse_args()
    folder = S.ROOT / "precision_check" / a.context
    folder.mkdir(parents=True, exist_ok=True)
    camera = CAMERAS[a.context]
    if not (folder / "truth.npz").exists():
        cal = load_spline_calibration(S.CONTEXTS[a.context]["calibration"])
        frames, truth = simulate(matched_to_em(cal, True), camera)
        tifffile.imwrite(folder / "frames.tif", frames, photometric="minisblack")
        np.savez(folder / "truth.npz", **truth)
    truth = dict(np.load(folder / "truth.npz"))
    run = load_run(a.run)
    out = folder / f"{a.run}.hdf5"
    if not out.exists():
        fit(folder / "frames.tif", camera, model_path(run, a.context), out)
    result = measure(truth, load_localizations(out), camera, run["keep"])
    (folder / f"{a.run}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
