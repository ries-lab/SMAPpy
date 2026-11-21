"""The benchmark's two-channel sets (`docs/decode/plex-plan.md` §6): drawn once and kept, as `sets.py`'s.

    python benchmarks/decode/suite/sets_c2.py draw
    python benchmarks/decode/suite/sets_c2.py list

**Contexts**: the calibrations of DECODE-Plex's paper, so that its published models are the baseline
without training -- SMAP's global two-channel calibrations read by `smappy.calibrate.smap_dual`
(each kept beside the sets as SMAPpy's ``_2ct.h5``):

* ``biplane_2c``: the bifocal beads of Fig. 1c (EM gain on, a 512 x 512 camera ROI from x = 2), the
  context of the live-cell model of Fig. 2c -- one photon count, drawn into both planes by their
  splines, which hold each plane's share (SMAP's ``normf``);
* ``colour_2a``: the two-colour beads of Fig. 2a (no EM gain, 226 x 512 from x = 138), the context
  of its actin / microtubule model -- two dyes, each emitter one of them, its photons split between
  the channels by its dye's ratio.  The ratios are read off DECODE-Plex's own fit of the Fig. 2a
  movie (`plex-plan.md` §6): the peak of log10(ch1 / ch0) at 0.55 and the shoulder at 1.05, 78 % and
  92 % of the light in channel 1.  Each channel's spline is made to sum to one at its focus, as
  DECODE-Plex's ``norm: auto`` does for colour.

**Frames as DECODE-Plex's models were trained on them** (plex-plan §12): the beads' whole camera ROI,
each half drawn with its own spline, the secondary where the registration *as DECODE-Plex reads it*
puts a molecule's partner -- SMAP's registration without its projective division
(`smap_dual`, ``reading="decode-plex"``; up to 0.29 px from SMAP's own on a 512-pixel chip, which costs
its biplane CRLB model 3 -- 7 times the bound where it otherwise reaches it) -- and channel 1's spline
oriented as each model was trained (``channel_1``), each clipped to its own half; the camera the paper's (EMCCD, gain 100, 5 e-/ADU, read noise 74.4 e-,
offset 398.6), the pixels 117 x 127 nm (x, y) as its runs convert them.  A molecule is placed in the
main half where its partner lands inside the secondary half.

**Two sweeps a context**, as for C = 1: density 1, 5, 15, 25, 50 molecules per 40 x 40 pixels of the
main half, photons and background drawn as the context's model was trained (`TRAINING`), the background
changing slowly over the movie (`slow_background`: a new level every 50 frames, straight in between);
photons at 1000, 3000 and 10 000 at density 1, each channel's background 2 % of its mean photons.  z
uniform over the model's training range.

**Blinking twins** (``<context>_blink_...``, plex-plan §16): the same two sweeps with molecules that
blink as DECODE draws them and as both DECODE-Plex and netfit were trained (``lifetime_avg`` 1): each
switches on at a time uniform over the movie and 3 on-times either side, stays on for an exponential
time of mean ``LIFETIME`` frames, and gives each frame its on-time overlaps that share of its photons;
a molecule is a row of the truth in every frame it is on in, at the same place.  As many molecules on
per frame as the sets without blinking.  In those, each molecule is in one frame alone -- a network
that reads several frames, trained on molecules that stay on, takes a bright spot in a neighbouring
frame for a faint tail and reports it (netfit's Round 0, §16).

**Each molecule's CRLB** is kept with its truth: both channels together, one molecule, its spot
alone and noise-free -- the Fisher matrix of each channel's spot (``training.joint_crlb``'s spots, the
spline's whole extent less 3 pixels), the secondary's x and y derivatives taken through the
registration's Jacobian, summed over the shared parameters (x, y, z, and the photons: one count for
biplane, two for colour) with a background per channel; doubled in variance for the EM register.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm"))
ROOT = DATA / "benchmark" / "v1"
SETS_DIR = ROOT / "sets_c2"
PLEX = DATA / "plex"

EMITTERS = 20000
PIXEL_NM = (117.0, 127.0)                     # x, y: as DECODE-Plex's runs convert (px_size [127, 117], rows first)
CAMERA = {"conversion": 5.0, "offset": 398.6, "em_on": True, "emgain": 100.0, "read_noise_e": 74.4,
          "pixelsize_um": PIXEL_NM[0] / 1000, "pixelsize_y_um": PIXEL_NM[1] / 1000}
EXCESS_NOISE = 2.0
#: DECODE-Plex's CameraPerfect, Fig. S4's: Poisson noise alone, counts in photons
PERFECT = {"conversion": 1.0, "offset": 0.0, "em_on": False, "emgain": 0.0, "read_noise_e": 0.0,
           "pixelsize_um": PIXEL_NM[0] / 1000, "pixelsize_y_um": PIXEL_NM[1] / 1000}
CRLB_PHOTONS = (2000, 5000, 8000)              # the S4 models' training range, its middle the paper's figure's
CRLB_BACKGROUND = 20.0                        # a pixel, each channel, as the paper's CRLB notebook
GRID_PX = 32                                  # isolated: a spline's spot is 26 px across
DENSITIES = (1, 5, 15, 25, 50)
LIFETIME = 1.0                                # mean on-time of the blinking sets, frames (DECODE-Plex's lifetime_avg)
PHOTONS = (1000, 3000, 10000)
BACKGROUND_KNOTS = 50                         # frames between the drawn levels of a slow background
DYES = (0.78, 0.92)                           # fraction of a dye's light in channel 1 (colour_2a)

CONTEXTS = {
    "biplane_2c": {
        "smap": PLEX / "calibration" / "Fig1c_2c-Pos0_240621_NC_BeadCal_BifocalPlane_Z_1_MMStack_Default.ome_3dcal.mat",
        "model": PLEX / "outputs" / "Fig2c-biplane-live_cell-separated_ch-2026-05-15_13-39-38-146351",
        "kind": "biplane", "z_nm": (-1000.0, 1000.0),
        "training": {"photons": [(500.0, 30000.0)], "background": [(0.0, 300.0), (0.0, 300.0)]},
        "about": "DECODE-Plex Fig. 1c's bifocal beads; the Fig. 2c live-cell model",
    },
    "colour_2a": {
        "smap": PLEX / "calibration" / "Fig2a-Pos0_MM_100ms_638i1_T700(100)_R685(70)_665LP_20nmstep_Z_1_up_noEM_3dcal.mat",
        "model": PLEX / "outputs" / "Fig2a-dual_color-actin_microtubes-separated_ch-2026-05-14_17-22-25-738645",
        "kind": "colour", "z_nm": (-600.0, 600.0),
        # per channel, as the model was trained; a molecule's total is drawn so that both stay inside
        "training": {"photons": [(200.0, 20000.0), (1000.0, 100000.0)], "background": [(0.0, 100.0), (0.0, 500.0)],
                     "total": (3000.0, 40000.0)},
        "about": "DECODE-Plex Fig. 2a's two-colour beads; its actin / microtubule model",
    },
    # Fig. S4's CRLB demonstration (notebook/crlb.ipynb): its own models, trained with a perfect camera
    # on single frames, photons 2000 -- 8000, background 0 -- 40.  Its biplane model's channel 1 is the
    # secondary spline not flipped along the split as Fig. 2c's run flips it, so in the camera's frame it
    # is the secondary mirrored along the split (plex-plan §12: DECODE-Plex's own PSFs, rendered)
    "biplane_s4": {
        "smap": PLEX / "calibration" / "Fig1c_2c-Pos0_240621_NC_BeadCal_BifocalPlane_Z_1_MMStack_Default.ome_3dcal.mat",
        "model": PLEX / "outputs" / "FigS4-biplane-crlb-2026-05-28_12-12-58-710801",
        "kind": "biplane", "z_nm": (-600.0, 600.0), "camera": PERFECT, "excess": 1.0, "placement": "grid",
        "channel_1": "secondary mirrored",
        "training": {"photons": [(2000.0, 8000.0)], "background": [(0.0, 40.0), (0.0, 40.0)]},
        "about": "Fig. 1c's bifocal beads; the Fig. S4 biplane CRLB model (perfect camera, isolated molecules)",
    },
    # its colour model draws channel 0's spline in both channels (channels_out [0, 0]), each normalised,
    # the photons split half and half: in the camera's frame channel 1 is the main spline mirrored
    "colour_s4": {
        "smap": PLEX / "calibration" / "FigS4-Pos0_231130_BeadsCal_Z_1_MMStack_Default.ome_updown_ul_normal_em_3dcal.mat",
        "model": PLEX / "outputs" / "FigS4-dual_color-crlb-2026-05-28_12-10-32-901897",
        "kind": "colour", "z_nm": (-600.0, 600.0), "camera": PERFECT, "excess": 1.0, "placement": "grid",
        "channel_1": "main mirrored", "dyes": (0.5,),
        "training": {"photons": [(1000.0, 4000.0), (1000.0, 4000.0)], "background": [(0.0, 40.0), (0.0, 40.0)],
                     "total": (2000.0, 8000.0)},
        "about": "Fig. S4's two-colour beads; the Fig. S4 colour CRLB model (perfect camera, isolated molecules)",
    },
}


def camera_of(context):
    return CONTEXTS[context].get("camera", CAMERA)


def excess_of(context):
    return CONTEXTS[context].get("excess", EXCESS_NOISE)


def _sets():
    out = {}
    for c, (name, ctx) in enumerate(CONTEXTS.items()):
        if ctx.get("placement") == "grid":
            for i, p in enumerate(CRLB_PHOTONS):
                out[f"{name}_crlb_{p}"] = {"kind": "simulated", "context": name, "seed": 5070 + 100 * c + i,
                                           "density": None, "photons": float(p), "background": CRLB_BACKGROUND}
            continue
        for i, d in enumerate(DENSITIES):
            out[f"{name}_density_{d}"] = {"kind": "simulated", "context": name, "seed": 5000 + 100 * c + i,
                                          "density": d, "photons": "training", "background": "training"}
        for i, p in enumerate(PHOTONS):
            out[f"{name}_photons_{p}"] = {"kind": "simulated", "context": name, "seed": 5050 + 100 * c + i,
                                          "density": 1, "photons": float(p), "background": "2%"}
        for i, d in enumerate(DENSITIES):
            out[f"{name}_blink_density_{d}"] = {"kind": "simulated", "context": name, "seed": 5200 + 100 * c + i,
                                                "density": d, "photons": "training", "background": "training",
                                                "lifetime": LIFETIME}
        for i, p in enumerate(PHOTONS):
            out[f"{name}_blink_photons_{p}"] = {"kind": "simulated", "context": name, "seed": 5250 + 100 * c + i,
                                                "density": 1, "photons": float(p), "background": "2%",
                                                "lifetime": LIFETIME}
    return out


SETS = _sets()


#: how a context's registration is read (`smappy.calibrate.smap_dual`): as DECODE-Plex reads it, so
#: that its published models see the microscope they were trained for
READING = "decode-plex"


def calibration(context: str):
    """The context's dual calibration, converted once from SMAP's files (read as `READING`) and kept as
    ``<context>_<READING>_2ct.h5``."""
    from smappy.calibrate import dual
    from smappy.calibrate.smap_dual import load_smap_dual_calibration
    kept = SETS_DIR / f"{context}_{READING}_2ct.h5"
    if not kept.exists():
        kept.parent.mkdir(parents=True, exist_ok=True)
        dual.save_dual_color_calibration(kept, load_smap_dual_calibration(CONTEXTS[context]["smap"], reading=READING))
    return dual.load_dual_color_calibration(kept)


def _unit(cal):
    """``cal`` scaled to sum to one at its focus (its knots' plane at z0)."""
    from dataclasses import replace
    s = float(cal.psf[round(cal.z0)].sum())
    return replace(cal, coeff=(cal.coeff / s).astype(np.float32), psf=cal.psf / s)


def drawn_splines(context, cal):
    """The two splines a context's frames are drawn with, as the camera sees them: colour's each
    normalised at focus (DECODE-Plex's ``norm: auto``), biplane's with each plane's share; channel 1
    the secondary's, or (``channel_1``) as the context's model was trained -- mirrored along the split,
    or the main channel's mirrored along it."""
    from smappy.calibrate.dual import mirror_axis, mirrored_spline
    ctx = CONTEXTS[context]
    colour = ctx["kind"] == "colour"
    main = _unit(cal.main) if colour else cal.main
    how = ctx.get("channel_1", "secondary")
    second = {"secondary": cal.secondary, "secondary mirrored": cal.secondary, "main": cal.main,
              "main mirrored": cal.main}[how]
    second = _unit(second) if colour else second
    if how.endswith("mirrored"):
        second = mirrored_spline(second, axis=2 - int(mirror_axis(cal.geometry)))
    return main, second


def _layout(cal):
    """The frame (height, width), the two halves as (x0, y0, w, h) in frame pixels, and the maps
    between frame pixels: main -> secondary and its Jacobian."""
    from smappy.calibrate.dual import geometry_layout
    x0, y0, w, h = cal.geometry["sources"][0]["roi"]
    regions = [(r[0] - x0, r[1] - y0, r[2], r[3]) for r in geometry_layout(cal.geometry).regions]
    corner = np.array([x0, y0], float)

    def to_secondary(xy):
        return cal.to_secondary(np.asarray(xy, float) + corner) - corner

    def jacobian(xy, step=1e-3):                    # d(secondary) / d(main), (n, 2, 2)
        xy = np.asarray(xy, float)
        jx = (to_secondary(xy + [step, 0]) - to_secondary(xy - [step, 0])) / (2 * step)
        jy = (to_secondary(xy + [0, step]) - to_secondary(xy - [0, step])) / (2 * step)
        return np.stack([jx, jy], -1)
    return (h, w), regions, to_secondary, jacobian


def _inside(xy, region, margin=0.5):
    x0, y0, w, h = region
    return ((xy[:, 0] >= x0 - 0.5 + margin) & (xy[:, 0] <= x0 + w - 0.5 - margin)
            & (xy[:, 1] >= y0 - 0.5 + margin) & (xy[:, 1] <= y0 + h - 0.5 - margin))


def slow_background(lo, hi, n_frames, rng, every=BACKGROUND_KNOTS):
    """A background that changes slowly over the movie, as a real one does: straight between levels drawn
    uniformly in [lo, hi] every ``every`` frames, so it moves by at most (hi - lo) / ``every`` a frame.
    A new level every frame (as the sets were drawn first) put DECODE-Plex's three-frame model 700 times
    off: its training holds the background over a window, and a frame brighter than its neighbours is
    molecules to it (plex-plan §14)."""
    knots = rng.uniform(lo, hi, int(np.ceil((n_frames - 1) / every)) + 1)
    return np.interp(np.arange(n_frames), every * np.arange(len(knots)), knots)


def draw_spline_whole(image, frame, xy_px, z_nm, photons, cal) -> None:
    """Add each spot into ``image`` (n, y, x) over the spline's whole grid, as DECODE-Plex's
    ``CubicSplinePSF`` draws it (its native ROI, the grid's size), where smappy's `_draw_spline` and
    netfit's renderer keep a square 3 cells smaller: far from focus that square cut 4 -- 8 % of the light
    and narrowed the spot by 5 % (z = +-1000 nm, Fig. 1c's spline), so the sets were not the spots
    DECODE-Plex is trained on (it changed no score: plex-plan §15).  The same pixel-to-spline map as theirs -- spline coordinate u = pixel - x + n / 2
    -- with a pixel outside the grid drawn as nothing, not as the edge cell's polynomial extended."""
    from smappy.io.calibration import compute_delta3d
    if len(frame) == 0:
        return
    height, width = image.shape[1:]
    _, nz, ny, nx = cal.coeff.shape
    size_x, size_y = nx, ny                          # the grid's size, around the spot's nearest pixel:
    zi = cal.z_nm_to_index(np.asarray(z_nm, float))  # DECODE-Plex's totals to 0.4 % at every z
    for k in range(len(frame)):
        cx, cy = xy_px[k]
        ox, oy = int(np.rint(cx)) - nx // 2, int(np.rint(cy)) - ny // 2
        u0, v0 = ox - cx + nx / 2, oy - cy + ny / 2     # spline coordinates of the square's first pixel
        iu = np.floor(u0).astype(int) + np.arange(size_x)
        iv = np.floor(v0).astype(int) + np.arange(size_y)
        iz = int(np.clip(np.floor(zi[k]), 0, nz - 1))
        delta = compute_delta3d(u0 - np.floor(u0), v0 - np.floor(v0), zi[k] - np.floor(zi[k])).astype(np.float32)
        okx, oky = (iu >= 0) & (iu < nx), (iv >= 0) & (iv < ny)
        spot = np.zeros((size_y, size_x), np.float32)
        block = cal.coeff[:, iz][:, iv[oky][:, None], iu[okx][None, :]]
        spot[np.ix_(oky, okx)] = np.tensordot(delta, block, axes=(0, 0))
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + size_x, width), min(oy + size_y, height)
        if x1 <= x0 or y1 <= y0:
            continue
        image[frame[k], y0:y1, x0:x1] += photons[k] * np.clip(spot[y0 - oy:y1 - oy, x0 - ox:x1 - ox], 0, None)


def simulate(spec, cal):
    """Frames (uint16, as the camera records them) and truth of one set."""
    from smappy.simulate.camera import _camera
    ctx = CONTEXTS[spec["context"]]
    colour = ctx["kind"] == "colour"
    rng = np.random.default_rng(spec["seed"])
    shape, regions, to_secondary, _ = _layout(cal)
    main_r, sec_r = regions
    splines = drawn_splines(spec["context"], cal)

    if ctx.get("placement") == "grid":
        # isolated molecules: one in each cell of a grid of GRID_PX over the main half, anywhere within
        # its middle, kept where the partner is inside the secondary half
        gx = np.arange(main_r[0] + GRID_PX / 2, main_r[0] + main_r[2] - GRID_PX / 2, GRID_PX)
        gy = np.arange(main_r[1] + GRID_PX / 2, main_r[1] + main_r[3] - GRID_PX / 2, GRID_PX)
        cells = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        cells = cells[_inside(to_secondary(cells), sec_r, margin=GRID_PX / 2)]
        n_frames = int(np.ceil(EMITTERS / len(cells)))
        frame = np.repeat(np.arange(n_frames), len(cells))
        xy = np.tile(cells, (n_frames, 1)) + rng.uniform(-GRID_PX / 4, GRID_PX / 4, (len(frame), 2))
    else:
        # molecules in the main half whose partner is inside the secondary half: the share that is,
        # measured on 20 000 trial positions, sets the frames
        def place(n):
            xy = np.c_[rng.uniform(main_r[0] - 0.5, main_r[0] + main_r[2] - 0.5, n),
                       rng.uniform(main_r[1] - 0.5, main_r[1] + main_r[3] - 0.5, n)]
            return xy[_inside(to_secondary(xy), sec_r)]
        share = len(place(20000)) / 20000
        per_frame = spec["density"] * main_r[2] * main_r[3] / (40 * 40) * share
        n_frames = int(np.ceil(EMITTERS / per_frame))
        life = spec.get("lifetime")
        if life:
            # molecules switching on over the movie and 3 on-times either side, so many that per_frame
            # are on in a frame on average (each overlaps life + 1 frames)
            total = rng.poisson(per_frame / (life + 1) * (n_frames + 6 * life))
        else:
            counts = rng.poisson(per_frame, n_frames)
            total = counts.sum()
        xy = np.empty((0, 2))
        while len(xy) < total:
            xy = np.r_[xy, place(2 * (total - len(xy)) + 100)]
        xy = xy[:total]
        if life:
            start = rng.uniform(-3 * life, n_frames + 3 * life, total)
            end = start + rng.exponential(life, total)
            first = np.floor(start).astype(int)
            span = int(np.max(np.floor(end) - first)) + 1
            k = first[:, None] + np.arange(span)[None]                       # (molecules, span)
            over = np.clip(np.minimum(end[:, None], k + 1) - np.maximum(start[:, None], k), 0, None)
            on = (over > 0) & (k >= 0) & (k < n_frames)
            mol, j = np.nonzero(on)
            order = np.argsort(k[mol, j], kind="stable")
            mol, frame, frac = mol[order], k[mol, j][order], over[mol, j][order]
        else:
            mol, frame, frac = np.arange(total), np.repeat(np.arange(n_frames), counts), np.ones(total)
    if ctx.get("placement") == "grid":
        mol, frac = np.arange(len(frame)), np.ones(len(frame))
    n = len(frame)
    m = int(mol.max()) + 1 if n else 0
    z = rng.uniform(*ctx["z_nm"], m)[mol]
    xy = xy[mol] if ctx.get("placement") != "grid" else xy
    xy2 = to_secondary(xy)

    train = ctx["training"]
    truth = {"frame": frame, "x_px": xy[:, 0], "y_px": xy[:, 1], "z_nm": z, "x2_px": xy2[:, 0], "y2_px": xy2[:, 1],
             "molecule": mol}
    if colour:
        dyes = np.asarray(ctx.get("dyes", DYES))
        dye = rng.integers(0, len(dyes), m)[mol]
        fraction = dyes[dye]
        # a molecule's flux while on, times the share of the frame it is on for
        total = (rng.uniform(*train["total"], m)[mol] if spec["photons"] == "training" else np.full(n, spec["photons"])) * frac
        phot = np.c_[total * (1 - fraction), total * fraction]
        truth.update(dye=dye, photons=total, photons_0=phot[:, 0], photons_1=phot[:, 1])
        mean_c = [np.mean(1 - dyes), np.mean(dyes)]
    else:
        total = (rng.uniform(*train["photons"][0], m)[mol] if spec["photons"] == "training"
                 else np.full(n, spec["photons"])) * frac
        phot = np.c_[total, total]                  # each plane's share is in its spline
        truth.update(photons=total)
        z0 = [round(s.z0) for s in splines]
        mean_c = [float(splines[c].psf[z0[c]].sum()) for c in range(2)]
    if spec["background"] == "training":
        bg = np.stack([slow_background(*train["background"][c], n_frames, rng) for c in range(2)], 1)
    elif not isinstance(spec["background"], str):
        bg = np.full((n_frames, 2), float(spec["background"]))
    else:
        bg = np.tile([0.02 * spec["photons"] * mean_c[c] for c in range(2)], (n_frames, 1))
    truth.update(background_0=bg[frame, 0], background_1=bg[frame, 1], frame_background_0=bg[:, 0],
                 frame_background_1=bg[:, 1])

    image = np.zeros((n_frames, *shape))
    for c, (region, pos) in enumerate(((main_r, xy), (sec_r, xy2))):
        x0, y0, w, h = region
        half = np.empty((n_frames, h, w))
        half[:] = bg[:, c, None, None]
        draw_spline_whole(half, frame, pos - [x0, y0], z, phot[:, c], splines[c])
        image[:, y0:y0 + h, x0:x0 + w] = half
    c = camera_of(spec["context"])
    frames = np.empty(image.shape, np.uint16)
    for k in range(n_frames):
        frames[k] = _camera(image[k], np.random.default_rng([spec["seed"], 7, k]), c["conversion"], c["offset"],
                            c["read_noise_e"] / c["conversion"], c["emgain"])
    return frames, truth, splines


def crlb(cal, splines, truth, colour: bool, excess=EXCESS_NOISE):
    """Each molecule's bound in x, y (nm, the main channel's pixels) and z, both channels together, on
    the splines the frames were drawn with; ``excess`` the camera's, 2 for an EM register, 1 perfect."""
    info = position_information(cal, splines, truth, colour, excess)
    return _bounds(info)


def _bounds(info):
    out = np.full((len(info), 3), np.nan)
    for k, I in enumerate(info):
        if np.all(np.isfinite(I)):
            try:
                out[k] = np.sqrt(np.diag(np.linalg.inv(I)))
            except np.linalg.LinAlgError:
                pass
    return out * [PIXEL_NM[0], PIXEL_NM[1], 1.0]


def window_crlb(info, truth, half=1):
    """The bound of a fit that reads the frames ``f - half .. f + half``: a molecule stays put while it is
    on, so the frames it is on within the window share x, y and z and each has its own photons and
    backgrounds -- their position information (`position_information`, the nuisances already taken out)
    adds.  ``half`` 1 is the 3-frame networks' window; a molecule on in one frame only keeps its bound."""
    mol, frame = np.asarray(truth["molecule"]), np.asarray(truth["frame"])
    order = np.lexsort((frame, mol))
    total = info.copy()
    for d in range(1, half + 1):
        a, b = order[:-d], order[d:]
        near = (mol[a] == mol[b]) & (frame[b] - frame[a] <= half)
        total[a[near]] += info[b[near]]
        total[b[near]] += info[a[near]]
    return _bounds(total)


def position_information(cal, splines, truth, colour: bool, excess=EXCESS_NOISE):
    """Each molecule-frame's Fisher information on x, y (pixels) and z alone, its photons and the
    channels' backgrounds taken out (the Schur complement): the part that adds over frames."""
    from smappy.training.joint_crlb import spline_spots
    _, _, _, jacobian = _layout(cal)
    n = len(truth["frame"])
    shape = (1 << 15, 1 << 15)                      # a spot alone: never clipped
    spots = []
    for c, (xs, ys) in enumerate((("x_px", "y_px"), ("x2_px", "y2_px"))):
        phot = truth["photons_0" if c == 0 else "photons_1"] if colour else truth["photons"]
        bg = truth[f"background_{c}"]
        s = spline_spots(splines[c], np.arange(n), truth[xs] + 4096, truth[ys] + 4096, truth["z_nm"], phot, bg, shape)
        spots.append((s, bg))
    J = jacobian(np.c_[truth["x_px"], truth["y_px"]])
    n_par = 7 if colour else 6                      # x, y, z, N (or N0, N1), b0, b1
    out = np.full((n, 3, 3), np.nan)
    for k in range(n):
        F = np.zeros((n_par, n_par))
        for c, (s, bg) in enumerate(spots):
            g = s.grads[k]                              # x, y (this channel's pixels), z, photons
            mu = s.model[k] + bg[k]
            d = np.zeros((n_par,) + mu.shape)
            jac = np.eye(2) if c == 0 else J[k]
            d[0] = jac[0, 0] * g[0] + jac[1, 0] * g[1]    # d/dx of the main position
            d[1] = jac[0, 1] * g[0] + jac[1, 1] * g[1]
            d[2] = g[2]
            if colour:
                d[3 + c] = g[3]
                d[5 + c] = 1.0
            else:
                d[3] = g[3]
                d[4 + c] = 1.0
            D = d.reshape(n_par, -1)
            F += (D / (mu.reshape(-1) * excess)) @ D.T
        try:
            out[k] = np.linalg.inv(np.linalg.inv(F)[:3, :3])
        except np.linalg.LinAlgError:
            pass
    return out


def draw_window_crlb(names=None, half=1):
    """``crlb_w{2 half + 1}.npz`` beside each set's truth: `window_crlb` on the splines it was drawn with."""
    for name, spec in SETS.items():
        if names and name not in names:
            continue
        folder = SETS_DIR / name
        if not (folder / "truth.npz").exists() or spec["context"] not in CONTEXTS:
            continue
        ctx = spec["context"]
        cal, colour = calibration(ctx), CONTEXTS[ctx]["kind"] == "colour"
        t = dict(np.load(folder / "truth.npz"))
        n = len(t["frame"])
        truth = {k: v for k, v in t.items() if np.ndim(v) == 1 and len(v) == n}
        info = position_information(cal, drawn_splines(ctx, cal), truth, colour, excess=excess_of(ctx))
        b = window_crlb(info, truth, half)
        one = _bounds(info)
        np.savez(folder / f"crlb_w{2 * half + 1}.npz", x_nm=b[:, 0], y_nm=b[:, 1], z_nm=b[:, 2])
        print(f"{name}: median bound x / z, 1 frame {np.nanmedian(one[:, 0]):.1f} / {np.nanmedian(one[:, 2]):.1f}, "
              f"{2 * half + 1} frames {np.nanmedian(b[:, 0]):.1f} / {np.nanmedian(b[:, 2]):.1f} nm; "
              f"stored 1-frame x {np.nanmedian(t['crlb_x_nm']):.1f}", flush=True)


def fingerprint(path) -> dict:
    import hashlib
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        head = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256_first_mb": head}


def draw(names=None):
    import tifffile
    manifest = {"pixel_nm": PIXEL_NM, "camera": CAMERA, "excess_noise": EXCESS_NOISE, "dyes": DYES,
                "contexts": {}, "sets": {}}
    cals = {}
    for name, ctx in CONTEXTS.items():
        if not ctx["smap"].exists():
            print(f"{name}: {ctx['smap']} is missing; skipped", flush=True)
            continue
        cals[name] = calibration(name)
        manifest["contexts"][name] = {"calibration": fingerprint(SETS_DIR / f"{name}_{READING}_2ct.h5"), "reading": READING,
                                      "smap": str(ctx["smap"]), "model": str(ctx["model"]),
                                      "about": ctx["about"], "z_nm": ctx["z_nm"], "training": ctx["training"],
                                      "camera": camera_of(name), "excess_noise": excess_of(name),
                                      "channel_1": ctx.get("channel_1", "secondary"),
                                      "placement": ctx.get("placement", "random"),
                                      "frame_shape": list(_layout(cals[name])[0])}
    for name, spec in SETS.items():
        if spec["context"] not in cals:
            continue
        cal, colour = cals[spec["context"]], CONTEXTS[spec["context"]]["kind"] == "colour"
        folder = SETS_DIR / name
        if (names and name in names) or not (folder / "truth.npz").exists():
            frames, truth, splines = simulate(spec, cal)
            b = crlb(cal, splines, truth, colour, excess=excess_of(spec["context"]))
            truth.update(crlb_x_nm=b[:, 0], crlb_y_nm=b[:, 1], crlb_z_nm=b[:, 2])
            folder.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(folder / "frames.tif", frames, photometric="minisblack")
            np.savez(folder / "truth.npz", **truth)
            print(f"{name}: {len(frames)} frames of {frames.shape[1]} x {frames.shape[2]}, {len(truth['frame'])} "
                  f"molecules; median CRLB x / y / z {np.nanmedian(b[:, 0]):.1f} / {np.nanmedian(b[:, 1]):.1f} / "
                  f"{np.nanmedian(b[:, 2]):.1f} nm", flush=True)
        t = np.load(folder / "truth.npz")
        manifest["sets"][name] = {**spec, "frames": int(t["frame_background_0"].shape[0]),
                                  "molecules": len(t["frame"]), "movie": str(folder / "frames.tif"),
                                  "file": fingerprint(folder / "frames.tif")}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest_c2.json").write_text(json.dumps(manifest, indent=1))
    print("manifest:", ROOT / "manifest_c2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("draw", "list"))
    ap.add_argument("--only", nargs="*", default=None, help="draw these again (the rest only if missing)")
    a = ap.parse_args()
    if a.what == "draw":
        draw(a.only)
    else:
        for name, spec in SETS.items():
            print(f"{name:28s} {spec['context']}")


if __name__ == "__main__":
    sys.exit(main())
