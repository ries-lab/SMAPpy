"""4Pi against astigmatic imaging at one density: the same molecules drawn both ways, each fitted by its netfit
and its spline fitter with their own detection (docs/4pi.md §33).

The truth -- molecules blinking with a mean on-time of ``--on`` frames at ``--density`` per 40 x 40 px, z in
+-500 nm, the cavity drifting -- is `netfit_4pi/movie_compare.py`'s `movie`, which also draws the 4Pi frames
(`fourpi.render_frames`).  The same truth is drawn a second time as a single-channel astigmatic movie of channel
0's pixels (`simulate.camera._draw_spline`, the first objective's PSF, `make_astig.py`): the same total photons
and the 4Pi channels' background summed (``full``), or half of both (``half``, one objective).  Six fits:

    4Pi:   netfit_4pi (network A, §30), spline_4pi
    astig: netfit_astig, spline_3d                 (full photons)
           netfit_astig_half, spline_3d_half       (half photons)

Every one scored in channel 0's own pixels, inside the box all channels see less 8 px, against its own mode's
single-frame CRLB, by `netfit.metrics.evaluate` (4Pi also `netfit.fourpi.phase_scores`, the cavity known).  A
molecule below 100 photons (50 for ``half``) counts neither way.  Beside the JSON of scores, an ``.npz`` of every
molecule judged -- its z, photons, nearest neighbour on in the frame (distance, phase and z difference, photon
ratio) and, per fit, whether it was found and its error -- which `plot.py` bins (the close pairs by phase
difference).

Several short movies (``--movies`` of ``--frames``), each drawn, fitted and scored on its own and pooled.

    python density_compare.py --density 5 --on 1 --movies 4 --frames 100 --out results/d5_on1 [--device cuda]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "netfit_4pi"))
from movie_compare import movie, spline_4pi

from smappy import fourpi
from smappy.io.calibration import load_spline_calibration
from smappy.netfit import fourpi as NF
from smappy.netfit import metrics
from smappy.netfit.predict import Predictor

MARGIN = 8
PIXEL_NM = (129.0, 129.0)
PHOTONS = (1000.0, 800.0, 100.0)        # network A's training range: mean, sd, floor
BACKGROUND = 3.75                       # a 4Pi channel's, A's range's mean; the astigmatic image has 4x (or 2x)
CAVITY = (0.002, 0.005)                 # rate and walk, rad a frame, as movie_compare
#: a neighbour this close (nm, laterally) makes a molecule one of a close pair
PAIR_NM = 250.0
FITS = ("netfit_4pi", "spline_4pi", "netfit_astig", "spline_3d", "netfit_astig_half", "spline_3d_half")


def astig_frames(acal, shape, t, scale, background, seed):
    """The truth ``t`` drawn by the astigmatic spline on channel 0's pixels: (n, H, W) photons with Poisson noise,
    ``scale`` times its photons, ``background`` a pixel."""
    n = int(t[:, 0].max()) + 1 if len(t) else 0
    from smappy.simulate.camera import _draw_spline
    image = np.zeros((n,) + tuple(shape), np.float32)
    _draw_spline(image, t[:, 0].astype(int), t[:, 1:3], t[:, 3], t[:, 4] * scale, acal)
    image += background
    return np.random.default_rng(seed).poisson(image).astype(np.float32)


def spline_3d(frames, cal_path):
    """Spline 3D with the plugin's default detection and fit, *focus* as calibrated: x, y in pixels."""
    from smappy.locs import concat
    from smappy.metadata import CameraMetadata
    from smappy.pipeline import LocalizationEngine
    from smappy.plugins.fit import SplineFit, SplineFitSettings
    s = SplineFitSettings()
    s.model.calibration = str(cal_path)
    s.model.focus = "calibrated"
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=PIXEL_NM[0] / 1000)
    plugin = SplineFit()
    engine = LocalizationEngine(camera, plugin.finder(s, camera), plugin.model(s, camera), s.fit)
    parts = [engine.push(frames[a:a + 200], a) for a in range(0, len(frames), 200)] + [engine.flush()]
    locs = concat([v for v in parts if v is not None and len(v)])
    p = PIXEL_NM[0]
    col = lambda k: np.asarray(locs[k], float)
    return {"frame": col("frame"), "x": col("x_nm") / p, "y": col("y_nm") / p, "z": col("z_nm"),
            "x_err": col("x_err_nm") / p, "y_err": col("y_err_nm") / p, "z_err": col("z_err_nm")}


def netfit_one(frames, predictor, block=256):
    """A single-channel model over a movie in blocks, each with the window's context (the ends repeated), as
    `netfit.fourpi.localize` reads a 4Pi movie."""
    k, n, parts = predictor.window, len(frames), []
    for a in range(0, n, block):
        b = min(a + block, n)
        parts.append(predictor(frames[np.clip(np.arange(a - k, b + k), 0, n - 1)], a))
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != "background_map"}
    return {"frame": out["frame"], "x": out["x_pix"], "y": out["y_pix"], "z": out["z_nm"],
            "x_err": out["x_err_pix"], "y_err": out["y_err_pix"], "z_err": out["z_err_nm"]}


def netfit_4pi(frames, four, predictor):
    out = NF.localize(predictor, four, four.layout.cut(frames))
    return {"frame": out["frame"], "x": out["x_pix"], "y": out["y_pix"], "z": out["z_nm"],
            "x_err": out["x_err_pix"], "y_err": out["y_err_pix"], "z_err": out["z_err_nm"],
            "phase": out["phase_rad"], "phase_err": out["phase_err_rad"]}


def neighbours(t, phase):
    """Each molecule's nearest other molecule on in the same frame: lateral distance (nm), the wrapped phase
    difference (rad), the z difference (nm) and its photons over the molecule's.  Every molecule drawn counts
    as a neighbour, also those outside the scored box or below the floor."""
    n = len(t)
    dist, dphi, dz, ratio = (np.full(n, np.inf), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    from scipy.spatial import cKDTree
    for f in np.unique(t[:, 0]):
        k = np.flatnonzero(t[:, 0] == f)
        if len(k) < 2:
            continue
        xy = t[k, 1:3] * PIXEL_NM[0]
        d, j = cKDTree(xy).query(xy, 2)
        other = k[j[:, 1]]
        dist[k], dz[k] = d[:, 1], t[other, 3] - t[k, 3]
        dphi[k] = np.angle(np.exp(1j * (phase[other] - phase[k])))
        ratio[k] = t[other, 4] / t[k, 4]
    return dist, dphi, dz, ratio


def judged(t, found, inner_t, inner_f, crlb, photons_min, extra_truth=None, phase_per_nm=None):
    """`metrics.evaluate` (and `phase_scores` with a phase) of the molecules inside, and per molecule judged
    whether it was found and its error (nan where not)."""
    tt = t[inner_t]
    truth = {"frame": tt[:, 0].astype(int), "xyz": tt[:, 1:4], "photons": tt[:, 4], "crlb": crlb}
    truth.update(extra_truth or {})
    f = {k: np.asarray(v)[inner_f] for k, v in found.items()}
    fd = {"frame": f["frame"].astype(int), "xyz": np.c_[f["x"], f["y"], f["z"]],
          "err": np.c_[f["x_err"] * PIXEL_NM[0], f["y_err"] * PIXEL_NM[1], f["z_err"]]}
    if "phase" in f:
        fd.update(phase=f["phase"], phase_err=f["phase_err"])
    r = metrics.evaluate(truth, fd, PIXEL_NM, photons_min=photons_min)
    if phase_per_nm is not None:
        r.update(NF.phase_scores(truth, fd, PIXEL_NM, phase_per_nm))
    ti, fi = metrics.match(truth["frame"], truth["xyz"], fd["frame"], fd["xyz"], PIXEL_NM)
    err = np.full((len(tt), 4), np.nan)
    err[ti, 0] = (fd["xyz"][fi, 0] - tt[ti, 1]) * PIXEL_NM[0]
    err[ti, 1] = (fd["xyz"][fi, 1] - tt[ti, 2]) * PIXEL_NM[1]
    err[ti, 2] = fd["xyz"][fi, 2] - tt[ti, 3]
    if "phase" in fd:
        z = tt[ti, 3]
        phase0 = truth["phase"][ti] - phase_per_nm * z
        err[ti, 3] = NF.unwrap_known(fd["phase"][fi], fd["xyz"][fi, 2], phase0, phase_per_nm) - z
    return r, err


def run_movie(m, a, cal, acal, acal_path, four, preds):
    """One movie: drawn both ways, fitted six ways, judged.  Returns the per-fit (result, error) and the
    molecules' columns."""
    seed = a.seed + 10 * m
    frames4, t = movie(cal, four.layout, four, a.frames, a.density, a.on, PHOTONS, BACKGROUND, *CAVITY, seed)
    phase = np.angle(np.exp(1j * (cal.phase_per_nm * t[:, 3] + t[:, 5])))
    h, w = four.layout.sizes[0][1], four.layout.sizes[0][0]
    r0, r1, c0, c1 = four.box
    inside = lambda xy: ((xy[:, 0] >= c0 + MARGIN) & (xy[:, 0] <= c1 - 1 - MARGIN)
                         & (xy[:, 1] >= r0 + MARGIN) & (xy[:, 1] <= r1 - 1 - MARGIN))
    inner_t = inside(t[:, 1:3])
    tt = t[inner_t]
    shift = four.remainder(tt[:, 1:3])
    b4 = NF.crlb(cal, tt[:, 1], tt[:, 2], shift, tt[:, 3], phase[inner_t], tt[:, 4],
                 np.full((len(tt), cal.n_channels), BACKGROUND))
    crlb4 = np.c_[b4[:, 0] * PIXEL_NM[0], b4[:, 1] * PIXEL_NM[1], b4[:, 2]]
    out, secs, frames_of, crlb_of = {}, {}, {}, {}
    for name in FITS:
        if name not in a.fits:
            continue
        t0 = time.time()
        if name.endswith("4pi"):
            found = (netfit_4pi(frames4, four, preds["4pi"]) if name == "netfit_4pi"
                     else spline_4pi(frames4, four.layout, cal, PIXEL_NM[0] / 1000))
            own = four.layout.from_frame(0, np.c_[found["x"], found["y"]])
            found["x"], found["y"] = own[:, 0], own[:, 1]
            r, err = judged(t, found, inner_t, inside(own), crlb4, 100.0,
                            {"phase": phase[inner_t], "crlb_phase": b4[:, 3]}, cal.phase_per_nm)
        else:
            scale = 0.5 if name.endswith("half") else 1.0
            bg = 4 * BACKGROUND * scale
            key = "half" if scale < 1 else "full"
            if key not in frames_of:
                frames_of[key] = astig_frames(acal, (h, w), t, scale, bg, seed + 5 + (scale < 1))
                crlb_of[key] = metrics.crlbs(acal, tt[:, 4] * scale, np.full(len(tt), bg), tt[:, 3],
                                             PIXEL_NM, False)
            frames = frames_of[key]
            found = (netfit_one(frames, preds["astig_" + key]) if name.startswith("netfit")
                     else spline_3d(frames, acal_path))
            r, err = judged(t, found, inner_t, inside(np.c_[found["x"], found["y"]]), crlb_of[key],
                            100.0 * scale)
        secs[name] = time.time() - t0
        out[name] = (r, err)
    dist, dphi, dz, ratio = neighbours(t, phase)
    cols = {"frame": tt[:, 0] + m * a.frames, "z": tt[:, 3], "photons": tt[:, 4], "phase": phase[inner_t],
            "nn_nm": dist[inner_t], "nn_dphi": dphi[inner_t], "nn_dz": dz[inner_t], "nn_ratio": ratio[inner_t],
            "crlb4": crlb4, "phase_crlb4": b4[:, 3], **{"crlb_" + k: v for k, v in crlb_of.items()}}
    return out, secs, cols


def pooled(results):
    """The movies' scores pooled, weighted by the molecules each judged: a simple mean of rates and RMSEs as
    squares -- close to the numbers of the movies scored together, which `plot.py` recomputes from the
    molecules where it matters."""
    keys = set().union(*[r.keys() for r in results])
    out = {}
    for k in keys:
        v = [r[k] for r in results if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        if not v:
            continue
        out[k] = float(np.sqrt(np.mean(np.square(v)))) if "rmse" in k else float(np.mean(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, required=True)
    ap.add_argument("--on", type=float, required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--movies", type=int, default=4)
    ap.add_argument("--model4pi", default="models/netfit_4pi_A.safetensors")
    ap.add_argument("--astig", default="models/netfit_astig_full.safetensors")
    ap.add_argument("--astig-half", default="models/netfit_astig_half.safetensors")
    ap.add_argument("--cal", default="cal/bead_v2_4pi.h5")
    ap.add_argument("--acal", default="cal/astig_v2_upper_3dcal.h5")
    ap.add_argument("--fits", nargs="+", default=list(FITS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = fourpi.load_fourpi_calibration(a.cal)
    acal = load_spline_calibration(a.acal)
    four = NF.FourChannels.from_calibration(cal)
    preds = {}
    if "netfit_4pi" in a.fits:
        preds["4pi"] = Predictor(a.model4pi, a.device)
    if "netfit_astig" in a.fits:
        preds["astig_full"] = Predictor(a.astig, a.device)
    if "netfit_astig_half" in a.fits:
        preds["astig_half"] = Predictor(a.astig_half, a.device)
    per_fit = {name: [] for name in a.fits}
    errs = {name: [] for name in a.fits}
    seconds = {name: 0.0 for name in a.fits}
    cols = []
    for m in range(a.movies):
        out, secs, c = run_movie(m, a, cal, acal, a.acal, four, preds)
        for name, (r, e) in out.items():
            per_fit[name].append(r)
            errs[name].append(e)
            seconds[name] += secs[name]
        cols.append(c)
        print(f"movie {m}: {len(c['z'])} molecules, " + ", ".join(f"{k} {v:.0f} s" for k, v in secs.items()),
              flush=True)
    molecules = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    for name in a.fits:
        molecules["err_" + name] = np.concatenate(errs[name]).astype(np.float32)
    np.savez_compressed(a.out + ".npz", **molecules)
    summary = {name: dict(pooled(rs), seconds=seconds[name]) for name, rs in per_fit.items()}
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "per_um2": a.density / (40 * PIXEL_NM[0] / 1000) ** 2,
                   "results": summary, "movies": per_fit}, f, indent=1, default=float)
    for name, r in summary.items():
        print(name, {k: round(r[k], 3) for k in ("jaccard", "recall", "precision", "core_rmse_lat_nm",
                                                  "core_rmse_ax_nm", "core_crlb_width_lat", "core_crlb_width_ax",
                                                  "core_wrong_fringe_share", "core_rmse_z_phase_nm") if k in r})


if __name__ == "__main__":
    main()
