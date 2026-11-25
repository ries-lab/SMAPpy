"""netfit against Spline 4Pi on the same simulated 4Pi movie, each with its own detection (docs/4pi.md §30).

The movie: molecules switching on at a rate that keeps ``density`` on per 40 x 40 px, each for an exponential
time of mean ``on`` frames (its photons shared over the frames by overlap, as netfit's training draws them),
at x, y uniform over the region every channel sees, z uniform in +-500 nm, the cavity's phase drifting over the
movie (`smappy.fourpi.cavity_drift`); drawn by `smappy.fourpi.render_frames` in the model's own layout with
its maps, a background, Poisson noise -- the simulation Spline 4Pi's tests use, not netfit's own drawing.
Spline 4Pi: its engine with the plugin's default detection, fit and starts (`MultiChannelEngine`, as
`FourPiFit.engine` builds it).  netfit: `netfit.fourpi.localize` with the trained model.  Both scored against
the truth of each frame with `netfit.metrics.evaluate` and `netfit.fourpi.phase_scores` (the cavity's phase
known for the wrong-fringe share), only where both read: the grid's box less 8 px.

Several short movies (``--movies`` of ``--frames``), each drawn, fitted and scored on its own and the matches
pooled: one long movie's frames take more memory than the Mac has to spare.

    python movie_compare.py MODEL.safetensors CALIBRATION.h5 --density 5 --on 2 --movies 5 --frames 60 [--device mps]
"""
import argparse
import json
import time

import numpy as np

from smappy import fourpi
from smappy.netfit import fourpi as NF
from smappy.netfit import metrics
from smappy.netfit.predict import Predictor

MARGIN = 8


def movie(cal, layout, four, n_frames, density, on, photons, background, rate, walk, seed):
    """The frames (n, H, W) in photons and the truth per frame: frame, x, y (channel 0's own pixels), z,
    photons in the frame, the cavity's phase in the frame."""
    rng = np.random.default_rng(seed)
    r0, r1, c0, c1 = four.box
    area = (r1 - r0) * (c1 - c0) / 1600.0
    rate_on = density * area / (on + 1)                       # switched on a frame, as `simulate.emitters`
    lo, hi = -3 * on, n_frames + 3 * on
    n = rng.poisson(rate_on * (hi - lo))
    start = rng.uniform(lo, hi, n)
    end = start + rng.exponential(on, n)
    x, y = rng.uniform(c0 - 0.5, c1 - 0.5, n), rng.uniform(r0 - 0.5, r1 - 0.5, n)
    z = rng.uniform(-500, 500, n)
    flux = np.clip(rng.normal(photons[0], photons[1], n), max(photons[2], 1e-3), None)
    cavity = fourpi.cavity_drift(n_frames, rate_rad=rate, wander_rad=walk, start_rad=rng.uniform(-np.pi, np.pi),
                                 seed=seed + 1)
    rows = []
    for f in range(n_frames):
        part = np.clip(np.minimum(end, f + 1) - np.maximum(start, f), 0, None)
        k = np.flatnonzero(part > 0)
        rows.append(np.c_[np.full(len(k), f), x[k], y[k], z[k], flux[k] * part[k], np.full(len(k), cavity[f])])
    t = np.concatenate(rows)
    frames = fourpi.render_frames(cal, layout, t[:, 0].astype(int), t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5],
                                  n_frames=n_frames, background=background, seed=seed + 2)
    return frames, t


def score(t, found, four, cal, pixel_nm, background):
    """metrics.evaluate and phase_scores of ``found`` (frame, x, y on the frame, z, errors, phase) against
    the truth ``t``, both inside the box less `MARGIN`."""
    r0, r1, c0, c1 = four.box
    tx = four.on_frame(t[:, 1:3])
    inner = lambda xy0: ((xy0[:, 0] >= c0 + MARGIN) & (xy0[:, 0] <= c1 - 1 - MARGIN)
                         & (xy0[:, 1] >= r0 + MARGIN) & (xy0[:, 1] <= r1 - 1 - MARGIN))
    keep_t = inner(t[:, 1:3])
    # the found positions back in channel 0's own pixels, to judge them by the same region
    f_own = four.layout.from_frame(0, np.c_[found["x"], found["y"]])
    keep_f = inner(f_own)
    tt = t[keep_t]
    shift = four.remainder(tt[:, 1:3])
    phase = np.angle(np.exp(1j * (cal.phase_per_nm * tt[:, 3] + tt[:, 5])))
    bound = NF.crlb(cal, tt[:, 1], tt[:, 2], shift, tt[:, 3], phase, tt[:, 4],
                    np.full((len(tt), cal.n_channels), float(background)))
    truth = {"frame": tt[:, 0].astype(int), "xyz": np.c_[tx[keep_t], tt[:, 3]], "photons": tt[:, 4],
             "crlb": np.c_[bound[:, 0] * pixel_nm[0], bound[:, 1] * pixel_nm[1], bound[:, 2]],
             "phase": phase, "crlb_phase": bound[:, 3]}
    f = {k: np.asarray(v)[keep_f] for k, v in found.items()}
    fd = {"frame": f["frame"].astype(int), "xyz": np.c_[f["x"], f["y"], f["z"]],
          "err": np.c_[f["x_err"] * pixel_nm[0], f["y_err"] * pixel_nm[1], f["z_err"]],
          "phase": f["phase"], "phase_err": f["phase_err"]}
    r = metrics.evaluate(truth, fd, pixel_nm)
    r.update(NF.phase_scores(truth, fd, pixel_nm, cal.phase_per_nm))
    return r


def spline_4pi(frames, layout, cal, pixel_um):
    from smappy.metadata import CameraMetadata
    from smappy.multichannel import MultiChannelEngine
    from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings
    from smappy.psf import Global4PiPSF
    s = FourPiFitSettings()
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=pixel_um)
    finder = FourPiFit().finder(s, camera)
    model = Global4PiPSF(cal, n_starts=int(s.model.starts), phase_starts=int(s.model.phase_starts))
    engine = MultiChannelEngine(camera, finder, model, layout, s.fit, frame_shape=frames.shape[-2:])
    parts = []
    for a in range(0, len(frames), 200):
        parts.append(engine.push(layout.cut(frames[a:a + 200]), a))
    parts.append(engine.flush())
    from smappy.locs import concat
    locs = concat([v for v in parts if v is not None and len(v)])
    p = pixel_um * 1000
    col = lambda k: np.asarray(locs[k], float)
    return {"frame": col("frame"), "x": col("x_nm") / p, "y": col("y_nm") / p, "z": col("z_nm"),
            "x_err": col("x_err_nm") / p, "y_err": col("y_err_nm") / p, "z_err": col("z_err_nm"),
            "phase": col("phase_rad"), "phase_err": col("phase_err_rad")}


def netfit(frames, layout, four, model, device):
    pred = Predictor(model, device)
    out = NF.localize(pred, four, layout.cut(frames))
    return {"frame": out["frame"], "x": out["x_pix"], "y": out["y_pix"], "z": out["z_nm"],
            "x_err": out["x_err_pix"], "y_err": out["y_err_pix"], "z_err": out["z_err_nm"],
            "phase": out["phase_rad"], "phase_err": out["phase_err_rad"]}


KEYS = ("jaccard", "recall", "precision", "core_rmse_lat_nm", "core_rmse_ax_nm", "core_crlb_width_lat",
        "core_crlb_width_ax", "core_crlb_width_phase", "core_wrong_fringe_share", "core_rmse_z_phase_nm",
        "rmse_lat_nm", "rmse_ax_nm", "crlb_width_lat", "crlb_width_ax", "crlb_width_phase", "wrong_fringe_share")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calibration")
    ap.add_argument("--density", type=float, default=5.0)
    ap.add_argument("--on", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--movies", type=int, default=5)
    ap.add_argument("--photons", type=float, nargs=3, default=None, help="mean, sd, min (default: the training's)")
    ap.add_argument("--background", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    cal = fourpi.load_fourpi_calibration(a.calibration)
    four = NF.FourChannels.from_calibration(cal)
    layout = four.layout
    pred_meta = Predictor(a.model, "cpu")
    sim = pred_meta.metadata["settings"]["simulation"]
    cam = pred_meta.metadata["settings"]["camera"]
    photons = a.photons or (sim["photons_mean"], sim["photons_sd"], sim["photons_min"])
    bg = a.background if a.background is not None else float(np.mean(pred_meta.channels.background[0]))
    pixel_nm = tuple(cam["pixel_nm"])
    pooled = {"netfit": ([], [], 0.0), "spline4pi": ([], [], 0.0)}
    for m in range(a.movies):
        frames, t = movie(cal, layout, four, a.frames, a.density, a.on, photons, bg, 0.002, 0.005, a.seed + 10 * m)
        for name in ("netfit", "spline4pi"):
            t0 = time.time()
            found = (netfit(frames, layout, four, a.model, a.device) if name == "netfit"
                     else spline_4pi(frames, layout, cal, pixel_nm[0] / 1000))
            ts, fs, sec = pooled[name]
            offset = m * a.frames
            found["frame"] = np.asarray(found["frame"]) + offset
            tm = t.copy()
            tm[:, 0] += offset
            ts.append(tm)
            fs.append(found)
            pooled[name] = (ts, fs, sec + time.time() - t0)
    results = {}
    for name, (ts, fs, sec) in pooled.items():
        found = {k: np.concatenate([f[k] for f in fs]) for k in fs[0]}
        r = score(np.concatenate(ts), found, four, cal, pixel_nm, bg)
        r["seconds"] = sec
        r["frames"] = a.movies * a.frames
        results[name] = r
        print(name, f"{sec:.0f} s", {k: round(r[k], 4) for k in KEYS if r.get(k) is not None}, flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "results": results}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
