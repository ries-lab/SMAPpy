"""The C = 2 baselines on the C = 2 sets, scored as every run is (`docs/decode/plex-plan.md` §11).

    python benchmarks/decode/suite/run_c2.py plex   [--sets NAME ...] [--refit]
    python benchmarks/decode/suite/run_c2.py spline [--sets NAME ...] [--refit]
    python benchmarks/decode/suite/run_c2.py netfit RUN --models CONTEXT=MODEL ... [--sets NAME ...] [--refit]
    python benchmarks/decode/suite/run_c2.py table  [RUN ...]

**plex**: DECODE-Plex's published model of the set's context (`sets_c2.CONTEXTS`), through its
notebook's steps (`decode_smlm.plex`), its raw threshold the notebooks' 0.1, the second channel
shifted by the registration's own offset (the sets use the beads' registration as it is), the
network reading only where both channels hold the camera's pixels (`overlap`, `fit_plex`).  On MPS
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

**netfit**: a two-channel netfit model per context (`smappy.netfit`, plex-plan §16), on the frames cut
onto channel 0's grid by SMAPpy's own dual calibration (`netfit.channels.TwoChannels`) -- the overlap
alone, the registration's remainder as its offset maps -- and its positions put back on the frame.
The frames' ends get their first and last frame repeated as the window's context.

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
KEEP = {"plex": {"min_prob": 0.5}, "spline": {"logl_mads": 3.0, "below_iteration_limit": True},
        "netfit": {"min_prob": 0.5}}
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


def _trafo_path(context):
    return pathlib.Path(str(S.CONTEXTS[context]["smap"]).replace("_3dcal.mat", "_trafo.mat"))


def overlap(context, shape):
    """Where both channels hold the camera's pixels, in DECODE-Plex's frame of the main half: rows
    ``r0:r1`` and columns ``c0:c1``.  Its inference moves the secondary channel by the registration's
    global offset and pads what comes from beyond the chip with zeros (`FrameShiftCropPad`); the
    network never saw a channel without background and answers the step with bright molecules far
    from focus (plex-plan §15).  A fill would break the intensity relation between the channels
    (biplane) or leave a channel dark (colour), so the network reads only this box (to a multiple
    of 8: `fit_plex`'s ``edges``), and every fitter is scored only in it.  Found by sending a constant frame through the inference's own splitter."""
    import torch

    from decode_smlm.plex.io import param
    from decode_smlm.plex.neuralfitter.train import setup_cfg
    crop = crop_of(shape)
    cfg = param.patch_img_size(_plex_cfg(context, "cpu"), crop, scenario=["Simulation", "Test"])
    proc = setup_cfg.setup_processor_inference(cfg, cfg_sim=cfg["Test"], crop_to=crop, mode_camera="rois")
    frame = torch.full((1, *shape), 3000.0)
    for step in proc._m_input._frame_pre:
        frame = step.forward(frame)
    empty = torch.stack([(f == 0).reshape(f.shape[-2:]) for f in frame]).any(0).numpy()
    rows, cols = np.flatnonzero(~empty.all(1)), np.flatnonzero(~empty.all(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


EDGES = "trim"        # what fills the U-Net's multiple of 8 at the overlap's far edges: `fit_plex`
HALF_BOX_PX = 13      # half the spots' box (26 px): the real pixels every kept localization has around it


def fit_plex(name, out, edges=None):
    """DECODE-Plex on the channels' overlap (`overlap`), by `infer`'s steps.  The U-Net (depth 3)
    reads a multiple of 8, where the overlap is not one; ``edges`` says what is done about it:

    * ``"trim"`` (the default, the fastest): DECODE-Plex's way, the box cut to it from its upper left
      (`AutoLeftUpperCrop`), the last 0 -- 7 rows and columns not read;
    * ``"mirror"``: both channels mirrored out to it at the lower and right edges, what is found in
      the padding dropped -- false molecules up to 15 px inside (plex-plan §15);
    * ``"corners"``: the trimmed box read once from each corner of the overlap, every pixel real;
      each pass keeps the localizations in its own quarter, split at the overlap's middle (`_owned`);
    * ``"strips"``: the trimmed box once, and thin strips (a multiple of 8) along the
      lower and right edges and in their corner, each reaching `HALF_BOX_PX` past the line where it
      takes over, which lies `HALF_BOX_PX` inside the box's cut (`_strips`): every kept localization
      had half a spot's box of real pixels on every side; 1.4 x the trim's time, the same scores but
      the cut rows' molecules found (plex-plan §15).

    The offset maps are cut from the whole crop's at each pass's pixels -- the training's maps there
    (DECODE-Plex's own ``roi_shift`` would do it through the transformation, but is off by twice the
    shift along a mirrored axis)."""
    import tifffile
    import torch

    from decode_smlm.plex import io as plex_io
    from decode_smlm.plex.neuralfitter import indicator
    from decode_smlm.plex.neuralfitter.inference import inference
    from decode_smlm.plex.neuralfitter.train import setup_cfg
    edges = edges or EDGES
    spec = S.SETS[name]
    ctx = S.CONTEXTS[spec["context"]]
    dev = _device()
    cfg = _plex_cfg(spec["context"], dev)
    frames = torch.from_numpy(tifffile.imread(S.SETS_DIR / name / "frames.tif").astype(np.float32))
    h, w = frames.shape[-2:]
    r0, r1, c0, c1 = overlap(spec["context"], (h, w))
    box = [r1 - r0, c1 - c0]
    size = [-(-n // 8) * 8 for n in box] if edges == "mirror" else [n // 8 * 8 for n in box]
    if edges == "corners":
        corners = sorted({(r, c) for r in (r0, r1 - size[0]) for c in (c0, c1 - size[1])})
        mid = ((r0 + r1) / 2, (c0 + c1) / 2)
        passes = [(ra, ca, size[0], size[1], (r0 if ra == r0 else mid[0], r1 if ra != r0 or len({c[0] for c in corners}) == 1
                                              else mid[0]), (c0 if ca == c0 else mid[1], c1 if ca != c0 or len({c[1] for c in corners}) == 1
                                                             else mid[1])) for ra, ca in corners]
    elif edges == "strips":
        passes = _strips((r0, r1, c0, c1), size)
    else:
        passes = [(r0, c0, size[0], size[1], (r0, r1), (c0, c1))]
    cfg["Paths"]["trafo"] = _trafo_path(spec["context"])
    aux_all = indicator.trafo_aux_factory(setup_cfg.setup_trafo(cfg, device="cpu"), crop_of((h, w)))
    cfg = plex_io.param.patch_img_size(cfg, size, scenario=["Simulation", "Test"])
    model = plex_io.model.load_model(next(ctx["model"].glob("*.ckpt")), cfg, device=dev)
    parts, seconds = [], 0.0
    for k, (ra, ca, ph, pw, _, _) in enumerate(passes):
        cfg = plex_io.param.patch_img_size(cfg, [ph, pw], scenario=["Simulation", "Test"])
        rb, cb = min(ra + ph, r1), min(ca + pw, c1)
        aux = torch.nn.functional.pad(aux_all[None, :, ra:rb, ca:cb], (0, pw - (cb - ca), 0, ph - (rb - ra)),
                                      mode="replicate")[0]
        proc = setup_cfg.setup_processor_inference(
            cfg, cfg_sim=cfg["Test"], crop_to=(slice(ra, rb), slice(ca, cb)), mode_camera="rois")
        if edges == "mirror":
            _pad_not_crop(proc, [ph, pw])
        inf = inference.Infer(model=model, window=cfg["Trainer"]["frame_window"], pre=proc.pre_inference,
                              post=proc.post, batch_size=max(4, 4 * (size[0] * size[1]) // (ph * pw)),
                              num_workers=0, device=[dev],    # a strip: as many pixels a batch as the box
                              validator=setup_cfg.setup_inference_validator(cfg))
        t0 = time.time()
        em = inf.forward(frames=frames, aux=aux)
        seconds += time.time() - t0
        xyz, sig = em.xyz_px.cpu().numpy(), em.xyz_sig.cpu().numpy()
        parts.append({"frame": em.frame_ix.cpu().numpy(), "row": xyz[:, 0] + ra, "col": xyz[:, 1] + ca,
                      "z": xyz[:, 2], "srow": sig[:, 0], "scol": sig[:, 1], "sz": sig[:, 2],
                      "prob": em.prob.cpu().numpy(), "phot": em.phot.cpu().numpy().reshape(len(em), -1),
                      "pass": np.full(len(em), k)})
    loc = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    loc = {key: v[_owned(loc, passes)] for key, v in loc.items()}
    col = loc["col"]
    if cfg["Camera"][0]["specs"]["flip"]["gain"] is not None:     # it mirrored the frame in x first
        col = w - 1 - col
    np.savez(out, frame=loc["frame"], x_nm=col * S.PIXEL_NM[0], y_nm=loc["row"] * S.PIXEL_NM[1], z_nm=loc["z"],
             x_err_nm=loc["scol"] * S.PIXEL_NM[0], y_err_nm=loc["srow"] * S.PIXEL_NM[1], z_err_nm=loc["sz"],
             prob=loc["prob"], photons=loc["phot"].sum(1), photons_c=loc["phot"], plex_pass=loc["pass"])
    return {"device": dev, "seconds": seconds, "frames": len(frames), "model": str(ctx["model"]),
            "overlap": [r0, r1, c0, c1], "edges": edges, "read": size,
            "passes": [[int(v) for v in p[:4]] for p in passes]}


BAND_PX = 0.5         # either side of a split line both passes' localizations are kept, then paired


def _strips(box, size):
    """The passes of ``edges="strips"``: ``(row, col, height, width, rows owned, cols owned)``.

    Along an axis the box's read loses ``lost`` pixels: it takes over at ``HALF_BOX_PX`` inside the
    read's cut, and its strip, aligned to the overlap's far edge, reaches ``HALF_BOX_PX`` before that
    line -- ``lost + 2 HALF_BOX_PX`` rounded up to 8, or the whole read where that is longer."""
    r0, r1, c0, c1 = box
    def axis(a0, a1, n):
        if a1 - a0 == n:
            return [(a0, n, (a0, a1))]
        split = a0 + n - HALF_BOX_PX
        m = min(-(-(a1 - split + HALF_BOX_PX) // 8) * 8, n)
        return [(a0, n, (a0, split)), (a1 - m, m, (split, a1))]
    return [(ra, ca, h, w, rown, cown) for ra, h, rown in axis(r0, r1, size[0]) for ca, w, cown in axis(c0, c1, size[1])]


BAND_PX = 0.5         # either side of a line between two passes' parts both are kept, then paired


def _owned(loc, passes):
    """Which localizations to keep, each pass its own part of the overlap (``passes``' last two:
    the rows and columns it owns).

    A localization outside its pass's part is dropped, so no molecule is reported twice -- except
    near a line between two parts, where two passes may place the same molecule on either side of
    it (reported by both, or by neither).  Within `BAND_PX` of a line both passes' localizations are
    kept and the duplicates removed: in a frame, the most probable first, any other pass's within
    1 px of it dropped."""
    k = loc["pass"]
    own = np.array([[p[4][0], p[4][1], p[5][0], p[5][1]] for p in passes], float)[k]
    read = np.array([[p[0], p[0] + p[2], p[1], p[1] + p[3]] for p in passes], float)[k]
    row, col = loc["row"], loc["col"]
    inside = (row > read[:, 0] - 0.5) & (row < read[:, 1] - 0.5) & (col > read[:, 2] - 0.5) & (col < read[:, 3] - 0.5)
    keep = inside & (row >= own[:, 0] - BAND_PX) & (row < own[:, 1] + BAND_PX) \
        & (col >= own[:, 2] - BAND_PX) & (col < own[:, 3] + BAND_PX)
    lines_r = {v for p in passes for v in p[4]} - {min(p[4][0] for p in passes), max(p[4][1] for p in passes)}
    lines_c = {v for p in passes for v in p[5]} - {min(p[5][0] for p in passes), max(p[5][1] for p in passes)}
    near = np.zeros(len(row), bool)
    for v in lines_r:
        near |= np.abs(row - v) < BAND_PX
    for v in lines_c:
        near |= np.abs(col - v) < BAND_PX
    near &= keep
    for f in np.unique(loc["frame"][near]):
        idx = np.flatnonzero(near & (loc["frame"] == f))
        idx = idx[np.argsort(-loc["prob"][idx])]
        for i in idx:
            if not keep[i]:
                continue
            others = idx[(k[idx] != k[i]) & keep[idx]]
            dup = others[np.hypot(row[others] - row[i], col[others] - col[i]) < 1.0]
            keep[dup] = False
    return keep


class _ReflectPad:
    """Mirror a frame out at its lower and right edges to ``size`` (the edge pixel not repeated)."""

    def __init__(self, size):
        self.size = size

    def forward(self, frame):
        import torch
        h, w = frame.shape[-2:]
        x = frame.reshape(-1, 1, h, w)
        x = torch.nn.functional.pad(x, (0, self.size[1] - w, 0, self.size[0] - h), mode="reflect")
        return x.reshape(*frame.shape[:-2], *self.size)


def _pad_not_crop(proc, size):
    """Put `_ReflectPad` where DECODE-Plex's splitter crops each channel to a multiple of 8."""
    from decode_smlm.plex.neuralfitter import frame_processing as fp
    from decode_smlm.plex.neuralfitter.processing.multichannel import (
        MultiChannelSplitter,
    )
    found = 0
    for step in proc._m_input._frame_pre:
        if isinstance(step, MultiChannelSplitter):
            for chain in step._per_channel:
                for i, q in enumerate(chain):
                    if isinstance(q, fp.AutoLeftUpperCrop):
                        chain[i] = _ReflectPad(size)
                        found += 1
    if found != 2:
        raise RuntimeError(f"expected a fold crop per channel in DECODE-Plex's splitter, found {found}")


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


def fit_netfit(name, out, model):
    """A two-channel netfit model on one set (the module's docstring)."""
    import tifffile

    from smappy.netfit.channels import TwoChannels
    from smappy.netfit.predict import Predictor
    spec = S.SETS[name]
    cal = S.calibration(spec["context"])
    two = TwoChannels.from_calibration(cal, cal.geometry["sources"][0]["roi"][:2])
    c = S.camera_of(spec["context"])
    adu = tifffile.imread(S.SETS_DIR / name / "frames.tif").astype(np.float32)
    photons = (adu - c["offset"]) * (c["conversion"] / c["emgain"] if c["em_on"] else c["conversion"])
    p = Predictor(model)
    k = p.window
    stack = two.cut(photons)
    stack = np.concatenate([np.repeat(stack[:1], k, 0), stack, np.repeat(stack[-1:], k, 0)])
    t0 = time.time()
    r = p(stack, 0, two.offsets())
    seconds = time.time() - t0
    r0, _, c0, _ = two.box
    xy = two.on_frame(np.c_[r["x_pix"] + c0, r["y_pix"] + r0])
    keep = {"frame": r["frame"], "x_nm": xy[:, 0] * S.PIXEL_NM[0], "y_nm": xy[:, 1] * S.PIXEL_NM[1], "z_nm": r["z_nm"],
            "x_err_nm": r["x_err_pix"] * S.PIXEL_NM[0], "y_err_nm": r["y_err_pix"] * S.PIXEL_NM[1],
            "z_err_nm": r["z_err_nm"], "prob": r["prob"], "photons": r["photons"]}
    if "photons_c" in r:
        keep["photons_c"] = r["photons_c"]
    np.savez(out, **keep)
    return {"device": p.device, "seconds": seconds, "frames": len(adu), "model": str(model),
            "read": two.to_dict(), "window": 2 * k + 1}


FITTERS = {"plex": fit_plex, "spline": fit_spline}


def trained_sensor(context):
    """The camera the context's model was trained on (``Camera[0].specs.sensor_size``, rows and columns
    in DECODE-Plex's frame): its training placed the frame anywhere on it, and nowhere else -- the
    Fig. S4 colour model's is 256 x 256, and it misses 94 -- 100 % of the molecules beyond it."""
    import yaml
    run = yaml.safe_load((S.CONTEXTS[context]["model"] / "param_run.yaml").read_text())
    return tuple(int(v) for v in run["Camera"][0]["specs"]["sensor_size"])


def both_channels(context, shape, flip_x: bool):
    """A mask maker on main-channel frame pixels: inside the area both channels see -- a partner inside
    the secondary half and the frame inside the channels' overlap the network reads (`overlap`) -- and
    its model was trained on (`trained_sensor`, in DECODE-Plex's frame: x mirrored for EM)."""
    cal = S.calibration(context)
    _, regions, to_sec, _ = S._layout(cal)
    r0, r1, c0, c1 = overlap(context, shape)
    w = shape[1]
    rows, cols = trained_sensor(context)

    def mask(x, y):
        ok = S._inside(to_sec(np.c_[x, y]), regions[1], margin=EDGE)
        xd = (w - 1 - x) if flip_x else x                 # the column in DECODE-Plex's frame
        return (ok & (y > r0 + EDGE) & (y < min(r1, rows) - 1 - EDGE)
                & (xd > c0 + EDGE) & (xd < min(c1, cols) - 1 - EDGE))
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


def run(fitter, names=None, refit=False, edges=None, run_name=None, models=None):
    """``fitter`` on the sets; netfit's run is named ``run_name`` and reads ``models`` (context -> model
    file), the sets of other contexts left out."""
    label = run_name or fitter
    folder = FITS / label
    folder.mkdir(parents=True, exist_ok=True)
    frames_read = FRAMES_READ.get(fitter, {})
    if fitter == "netfit":
        names = [n for n in (names or S.SETS) if S.SETS[n]["context"] in models]
        frames_read = {}
    report = {"fitter": fitter, "keep": KEEP[fitter], "frames_read": frames_read,
              "platform": platform.platform(), "sets": {}, **({"models": models} if models else {})}
    path = REPORTS / f"{label}_c2.json"
    if path.exists():
        report["sets"] = json.loads(path.read_text()).get("sets", {})
    for name in names or S.SETS:
        out = folder / f"{name}.npz"
        meta = folder / f"{name}.json"
        if refit or not out.exists():
            if fitter == "plex":
                info = fit_plex(name, out, edges)
            elif fitter == "netfit":
                info = fit_netfit(name, out, models[S.SETS[name]["context"]])
            else:
                info = FITTERS[fitter](name, out)
            meta.write_text(json.dumps(info))
        info = json.loads(meta.read_text())
        if fitter == "netfit":
            report["frames_read"][S.SETS[name]["context"]] = info["window"]
        locs = dict(np.load(out))
        if not len(locs["frame"]):
            print(f"{label} {name}: no localizations", flush=True)
            continue
        s = score(name, locs, KEEP[fitter])
        s["fit"] = info
        s["ms_per_frame"] = 1000 * info["seconds"] / max(int(np.load(S.SETS_DIR / name / "truth.npz")["frame_background_0"].shape[0]), 1)
        report["sets"][name] = s
        core = s.get("core", {})
        print(f"{label} {name}: J {s['jaccard']:.3f}  RMSE lat/ax {s['rmse_lat_nm']:.1f}/{s['rmse_ax_nm']:.1f}  "
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
    ap.add_argument("what", choices=("plex", "spline", "netfit", "table"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--edges", choices=("trim", "strips", "corners", "mirror"), default=EDGES,
                    help="what fills the U-Net's multiple of 8 at the overlap's far edges (fit_plex)")
    ap.add_argument("--models", nargs="*", default=[], help="netfit: CONTEXT=MODEL, one per context")
    a = ap.parse_args()
    if a.what == "table":
        table(a.runs or ["plex", "spline"])
    elif a.what == "netfit":
        if len(a.runs) != 1 or not a.models:
            ap.error("netfit: one RUN name and --models CONTEXT=MODEL")
        run("netfit", a.sets, a.refit, run_name=a.runs[0], models=dict(m.split("=", 1) for m in a.models))
    else:
        run(a.what, a.sets, a.refit, a.edges)


if __name__ == "__main__":
    sys.exit(main())
