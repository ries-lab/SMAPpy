"""The C = 2 baselines on the C = 2 sets, scored as every run is (`docs/decode/plex-plan.md` §11).

    python benchmarks/decode/suite/run_c2.py plex   [--sets NAME ...] [--refit]
    python benchmarks/decode/suite/run_c2.py spline [--sets NAME ...] [--refit]
    python benchmarks/decode/suite/run_c2.py table  [RUN ...]

**plex**: DECODE-Plex's published model of the set's context (`sets_c2.CONTEXTS`), through its
notebook's steps (`decode_smlm.plex`), its raw threshold the notebooks' 0.1, the second channel
shifted by the registration's own offset (the sets use the beads' registration as it is).  On MPS
or CUDA where there is one; the device is recorded.

**spline**: SMAPpy's Spline 3D 2C with the context's converted calibration, written ROI-local for
frames that carry no camera ROI (the same map, the beads' corner taken off); photons linked for
biplane (one count, each plane's share in its spline), free for colour (a count per channel).

**The scoring** is `report.score_truth`, where both channels see the sample, the published
network reads and its model was trained (`trained_sensor`) -- the same area for the truth and every fitter's localizations: a molecule's partner
inside the secondary half (7 pixels in, where Spline 3D 2C's 13-pixel box fits; 76 % of the
molecules nearer a half's edge are not fitted), inside the main half's crop a network reads (DECODE-Plex
cuts the frame to a multiple of 8, from its own corner, after its EM mirror).  ``keep``: the
network's localizations at prob >= 0.5, Spline 3D 2C's within 3 MADs of the median log-likelihood
and not at the iteration limit, as for C = 1.  Positions in the main channel's pixels (117 x 127 nm).

Fits are kept under ``fits_c2/<RUN>/<set>.npz``, the report as ``reports/<RUN>_c2.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R
import sets_c2 as S

FITS = S.ROOT / "fits_c2"
REPORTS = S.ROOT / "reports"
KEEP = {"plex": {"min_prob": 0.5}, "spline": {"logl_mads": 3.0, "below_iteration_limit": True}}
FRAMES_READ = {"plex": {"biplane_2c": 1, "colour_2a": 3, "biplane_s4": 1, "colour_s4": 1},
               "spline": {"biplane_2c": 1, "colour_2a": 1, "biplane_s4": 1, "colour_s4": 1}}
EDGE = 7                                       # pixels in from a half's edge or the crop's: a 13-px fit box
                                               # stays inside, as the C = 1 scoring's 7 (score_sets.EDGE_PX)


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _plex_cfg(context, dev):
    import decode_smlm.plex as dp
    ctx = S.CONTEXTS[context]
    cfg = dp.io.param.load(ctx["model"] / "param_run.yaml")
    for s in cfg["Hardware"]["device"]:
        cfg["Hardware"]["device"][s] = dev
    # the frames are as the camera records them: DECODE-Plex undoes the EM register's mirror (``gain``)
    # where the beads were taken with EM gain, as Fig. 2c's run does, and the split's mirror
    # (``channel``) always, as the notebooks set it; the S4 models' runs (a perfect camera) name neither
    em = bool(S.calibration(context).parameters["smap"]["EMon"][0])
    for cam in cfg["Camera"]:
        cam["specs"]["flip"] = {"gain": -1 if em else None, "channel": -2}
    glob = cfg["Simulation"]["Transformation"]["Pos"]["glob"]["offset"]
    cfg["Test"]["Transformation"]["Pos"]["glob"]["offset"] = copy.deepcopy(glob)
    return cfg


def crop_of(shape):
    """The main half's frame a DECODE-Plex model reads: rows and columns, a multiple of 8."""
    h, w = shape
    return [math.floor(h / 2 / 8) * 8, math.floor(w / 8) * 8]


def fit_plex(name, out):
    import tifffile
    import torch

    import decode_smlm.plex.neuralfitter.inference.functional as infer_func
    spec = S.SETS[name]
    ctx = S.CONTEXTS[spec["context"]]
    dev = _device()
    cfg = _plex_cfg(spec["context"], dev)
    frames = torch.from_numpy(tifffile.imread(S.SETS_DIR / name / "frames.tif").astype(np.float32))
    h, w = frames.shape[-2:]
    trafo = pathlib.Path(str(ctx["smap"]).replace("_3dcal.mat", "_trafo.mat"))
    t0 = time.time()
    em, _ = infer_func.infer(frames, frame_crop=crop_of((h, w)), cfg=cfg, model=next(ctx["model"].glob("*.ckpt")),
                             mode="multi", trafo=trafo, mode_camera="rois", roi_shift=None, batch_size=4,
                             num_workers=0, device=[dev])
    seconds = time.time() - t0
    xyz, sig = em.xyz_px.cpu().numpy(), em.xyz_sig.cpu().numpy()
    row, col, srow, scol = xyz[:, 0], xyz[:, 1], sig[:, 0], sig[:, 1]
    if cfg["Camera"][0]["specs"]["flip"]["gain"] is not None:     # it mirrored the frame in x first
        col = w - 1 - col
    phot = em.phot.cpu().numpy().reshape(len(em), -1)
    np.savez(out, frame=em.frame_ix.cpu().numpy(), x_nm=col * S.PIXEL_NM[0], y_nm=row * S.PIXEL_NM[1], z_nm=xyz[:, 2],
             x_err_nm=scol * S.PIXEL_NM[0], y_err_nm=srow * S.PIXEL_NM[1], z_err_nm=sig[:, 2],
             prob=em.prob.cpu().numpy(), photons=phot.sum(1), photons_c=phot)
    return {"device": dev, "seconds": seconds, "frames": len(frames), "model": str(ctx["model"])}


def local_calibration(context):
    """The context's calibration ROI-local: the map with the beads' corner taken off, the frames'
    pixels from 0 -- for movies that carry no camera ROI -- with the splines the frames are drawn with
    (`sets_c2.drawn_splines`), so that Spline 3D 2C fits the physics the sets hold."""
    from smappy.calibrate import dual
    path = S.SETS_DIR / f"{context}_{S.READING}_2ct_local.h5"
    if not path.exists():
        cal = S.calibration(context)
        main, secondary = S.drawn_splines(context, cal)
        x0, y0, w, h = cal.geometry["sources"][0]["roi"]
        corner = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
        hmat = np.linalg.inv(corner) @ cal.transformation @ corner
        g = cal.geometry
        split = dual.geometry_layout(g).regions[1][1 if "up-down" in g["layout"] else 0] - (y0 if "up-down" in g["layout"] else x0)
        geometry = dual.split_geometry(g["layout"], g["main_channel"], (h, w), split_position=split, roi_local=True)
        dual.save_dual_color_calibration(path, dual.DualColorCalibration(main, secondary, hmat / hmat[2, 2],
                                                                          geometry, cal.parameters))
    return path


def fit_spline(name, out):
    from smappy.pipeline import FitSettings as PipelineFitSettings
    from smappy.plugins import Context
    from smappy.plugins.fit import (
        CameraSettings,
        DualModelSettings,
        DualSplineFit,
        DualSplineFitSettings,
        OutputSettings,
        SourceSettings,
    )
    spec = S.SETS[name]
    c = S.camera_of(spec["context"])
    settings = DualSplineFitSettings(
        source=SourceSettings(path=str(S.SETS_DIR / name / "frames.tif")),
        camera=CameraSettings(conversion=c["conversion"], offset=c["offset"], em_on=c["em_on"], emgain=c["emgain"],
                              pixelsize_um=c["pixelsize_um"], pixelsize_y_um=c["pixelsize_y_um"],
                              read_noise_e=c["read_noise_e"]),
        model=DualModelSettings(calibration=str(local_calibration(spec["context"])),
                                link_photons=S.CONTEXTS[spec["context"]]["kind"] == "biplane"),
        fit=PipelineFitSettings(roisize=13), output=OutputSettings(save=False))
    t0 = time.time()
    locs = DualSplineFit().run(Context(), settings).locs
    seconds = time.time() - t0
    keys = [k for k in ("frame", "x_nm", "y_nm", "z_nm", "x_err_nm", "y_err_nm", "z_err_nm", "photons", "ratio",
                        "logl_rel", "iterations") if k in locs]
    np.savez(out, **{k: np.asarray(locs[k]) for k in keys})
    return {"device": "cpu", "seconds": seconds, "calibration": str(local_calibration(spec["context"]))}


FITTERS = {"plex": fit_plex, "spline": fit_spline}


def trained_sensor(context):
    """The camera the context's model was trained on (``Camera[0].specs.sensor_size``, rows and columns
    in DECODE-Plex's frame): its training placed the frame anywhere on it, and nowhere else -- the
    Fig. S4 colour model's is 256 x 256, and it misses 94 -- 100 % of the molecules beyond it."""
    import yaml
    run = yaml.safe_load((S.CONTEXTS[context]["model"] / "param_run.yaml").read_text())
    return tuple(int(v) for v in run["Camera"][0]["specs"]["sensor_size"])


def both_channels(context, shape, flip_x: bool):
    """A mask maker on main-channel frame pixels: inside the area both channels see, a network reads
    and its model was trained on (`trained_sensor`, in DECODE-Plex's frame: x mirrored for EM)."""
    cal = S.calibration(context)
    _, regions, to_sec, _ = S._layout(cal)
    crop = crop_of(shape)
    w = shape[1]
    rows, cols = trained_sensor(context)

    def mask(x, y):
        ok = S._inside(to_sec(np.c_[x, y]), regions[1], margin=EDGE) & (y > EDGE) & (y < min(crop[0], rows) - EDGE)
        lo, hi = (w - crop[1], w) if flip_x else (0, crop[1])
        xd = (w - 1 - x) if flip_x else x                 # the column in DECODE-Plex's frame
        return ok & (x > lo + EDGE) & (x < hi - 1 - EDGE) & (xd < cols - EDGE)
    return mask


def score(name, locs, keep):
    spec = S.SETS[name]
    context = spec["context"]
    t = dict(np.load(S.SETS_DIR / name / "truth.npz"))
    shape = tuple(S._layout(S.calibration(context))[0])
    flip = bool(S.calibration(context).parameters["smap"]["EMon"][0])   # DECODE-Plex mirrors an EM frame first
    mask = both_channels(context, shape, flip)
    n = len(t["frame"])
    truth = {k: v for k, v in t.items() if np.ndim(v) == 1 and len(v) == n}
    sel = mask(truth["x_px"], truth["y_px"])
    truth = {k: v[sel] for k, v in truth.items()}
    lx, ly = np.asarray(locs["x_nm"]) / S.PIXEL_NM[0], np.asarray(locs["y_nm"]) / S.PIXEL_NM[1]
    lsel = mask(lx, ly)
    locs = {k: np.asarray(v)[lsel] for k, v in locs.items() if np.ndim(v) >= 1 and len(v) == len(lsel)}
    # the suite's z range is C = 1's (+-700 nm); the context's is its model's training range
    z_limit = float(max(abs(v) for v in S.CONTEXTS[context]["z_nm"]))
    R.Z_LIMIT, R.kept.__defaults__ = z_limit, (z_limit,)
    out = R.score_truth(truth, locs, keep, S.PIXEL_NM, (shape[1], shape[0]))
    out["area_molecules"] = int(sel.sum())
    out["by_z"] = by_z(truth, locs, keep, shape)
    return out


def by_z(truth, locs, keep, shape, width=100.0):
    """The paper's CRLB figure: in bins of the true z, each axis's RMSE of the matched localizations
    against the root mean square of their molecules' CRLB, and its ratio -- 1 at the bound."""
    from score_sets import match
    ok = R.kept(locs, keep)
    f = {"frame": np.asarray(locs["frame"])[ok].astype(np.int64), "x_px": np.asarray(locs["x_nm"])[ok] / S.PIXEL_NM[0],
         "y_px": np.asarray(locs["y_nm"])[ok] / S.PIXEL_NM[1], "z_nm": np.asarray(locs["z_nm"])[ok]}
    w, h = shape[1], shape[0]
    fi = R._inside(f["x_px"], f["y_px"], w, h)
    f = {k: v[fi] for k, v in f.items()}
    ti = R._inside(truth["x_px"], truth["y_px"], w, h)
    t = {k: v[ti] for k, v in truth.items()}
    a, b = match(t, f, S.PIXEL_NM)
    d = np.c_[(f["x_px"][b] - t["x_px"][a]) * S.PIXEL_NM[0], (f["y_px"][b] - t["y_px"][a]) * S.PIXEL_NM[1],
              f["z_nm"][b] - t["z_nm"][a]]
    crlb = np.c_[t["crlb_x_nm"][a], t["crlb_y_nm"][a], t["crlb_z_nm"][a]]
    z = t["z_nm"][a]
    rows = []
    for lo in np.arange(np.floor(z.min() / width) * width, z.max(), width):
        sel = (z >= lo) & (z < lo + width)
        if sel.sum() < 50:
            continue
        rmse = np.sqrt(np.mean(d[sel] ** 2, 0))
        bound = np.sqrt(np.nanmean(crlb[sel] ** 2, 0))
        rows.append({"lo": float(lo), "hi": float(lo + width), "n": int(sel.sum()), "rmse_nm": rmse.tolist(),
                     "bias_nm": d[sel].mean(0).tolist(), "crlb_nm": bound.tolist(), "ratio": (rmse / bound).tolist()})
    return rows


def run(fitter, names=None, refit=False):
    folder = FITS / fitter
    folder.mkdir(parents=True, exist_ok=True)
    report = {"fitter": fitter, "keep": KEEP[fitter], "frames_read": FRAMES_READ[fitter],
              "platform": platform.platform(), "sets": {}}
    path = REPORTS / f"{fitter}_c2.json"
    if path.exists():
        report["sets"] = json.loads(path.read_text()).get("sets", {})
    for name in names or S.SETS:
        out = folder / f"{name}.npz"
        meta = folder / f"{name}.json"
        if refit or not out.exists():
            info = FITTERS[fitter](name, out)
            meta.write_text(json.dumps(info))
        info = json.loads(meta.read_text())
        locs = dict(np.load(out))
        s = score(name, locs, KEEP[fitter])
        s["fit"] = info
        s["ms_per_frame"] = 1000 * info["seconds"] / max(int(np.load(S.SETS_DIR / name / "truth.npz")["frame_background_0"].shape[0]), 1)
        report["sets"][name] = s
        core = s.get("core", {})
        print(f"{fitter} {name}: J {s['jaccard']:.3f}  RMSE lat/ax {s['rmse_lat_nm']:.1f}/{s['rmse_ax_nm']:.1f}  "
              f"|z|<300: {core.get('rmse_lat_nm', float('nan')):.1f}/{core.get('rmse_ax_nm', float('nan')):.1f}  "
              f"/CRLB {s.get('rmse_over_crlb_lat', float('nan')):.2f}/{s.get('rmse_over_crlb_ax', float('nan')):.2f}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=float))


def table(fitters):
    """A Markdown table of the runs' reports, set by set."""
    reports = {f: json.loads((REPORTS / f"{f}_c2.json").read_text()) for f in fitters}
    head = ("| set | fitter | frames | Jaccard | RMSE lat / ax (nm) | |z| < 300: lat / ax | crlb width lat / ax, |z| < 300 "
            "| RMSE / CRLB lat / ax | ms / frame |")
    print(head)
    print("|" + "---|" * 9)
    for name in S.SETS:
        for f, rep in reports.items():
            s = rep["sets"].get(name)
            if not s:
                continue
            c = s.get("core", {})
            print(f"| {name} | {f} | {rep['frames_read'][S.SETS[name]['context']]} | {s['jaccard']:.3f} | "
                  f"{s['rmse_lat_nm']:.1f} / {s['rmse_ax_nm']:.1f} | {c.get('rmse_lat_nm', np.nan):.1f} / "
                  f"{c.get('rmse_ax_nm', np.nan):.1f} | {c.get('crlb_width_lat', np.nan):.2f} / {c.get('crlb_width_ax', np.nan):.2f} | "
                  f"{s.get('rmse_over_crlb_lat', np.nan):.2f} / {s.get('rmse_over_crlb_ax', np.nan):.2f} | "
                  f"{s['ms_per_frame']:.0f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plex", "spline", "table"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    if a.what == "table":
        table(a.runs or ["plex", "spline"])
    else:
        run(a.what, a.sets, a.refit)


if __name__ == "__main__":
    sys.exit(main())
