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

**Frames as the camera records them**: the beads' whole camera ROI, each half drawn with its own
spline, the secondary where the registration puts a molecule's partner (mirror included), each
clipped to its own half; the camera the paper's (EMCCD, gain 100, 5 e-/ADU, read noise 74.4 e-,
offset 398.6), the pixels 117 x 127 nm (x, y) as its runs convert them.  A molecule is placed in the
main half where its partner lands inside the secondary half.

**Two sweeps a context**, as for C = 1: density 1, 5, 15, 25, 50 molecules per 40 x 40 pixels of the
main half, photons and background drawn as the context's model was trained (`TRAINING`); photons at
1000, 3000 and 10 000 at density 1, each channel's background 2 % of its mean photons.  z uniform
over the model's training range.

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
DENSITIES = (1, 5, 15, 25, 50)
PHOTONS = (1000, 3000, 10000)
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
}


def _sets():
    out = {}
    for c, (name, ctx) in enumerate(CONTEXTS.items()):
        for i, d in enumerate(DENSITIES):
            out[f"{name}_density_{d}"] = {"kind": "simulated", "context": name, "seed": 5000 + 100 * c + i,
                                          "density": d, "photons": "training", "background": "training"}
        for i, p in enumerate(PHOTONS):
            out[f"{name}_photons_{p}"] = {"kind": "simulated", "context": name, "seed": 5050 + 100 * c + i,
                                          "density": 1, "photons": float(p), "background": "2%"}
    return out


SETS = _sets()


def calibration(context: str):
    """The context's dual calibration, converted once from SMAP's files and kept as ``_2ct.h5``."""
    from smappy.calibrate import dual
    from smappy.calibrate.smap_dual import load_smap_dual_calibration
    kept = SETS_DIR / f"{context}_2ct.h5"
    if not kept.exists():
        kept.parent.mkdir(parents=True, exist_ok=True)
        dual.save_dual_color_calibration(kept, load_smap_dual_calibration(CONTEXTS[context]["smap"]))
    return dual.load_dual_color_calibration(kept)


def _unit(cal):
    """``cal`` scaled to sum to one at its focus (its knots' plane at z0)."""
    from dataclasses import replace
    s = float(cal.psf[round(cal.z0)].sum())
    return replace(cal, coeff=(cal.coeff / s).astype(np.float32), psf=cal.psf / s)


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


def simulate(spec, cal):
    """Frames (uint16, as the camera records them) and truth of one set."""
    from smappy.simulate.camera import _camera, _draw_spline
    ctx = CONTEXTS[spec["context"]]
    colour = ctx["kind"] == "colour"
    rng = np.random.default_rng(spec["seed"])
    shape, regions, to_secondary, _ = _layout(cal)
    main_r, sec_r = regions
    splines = (_unit(cal.main), _unit(cal.secondary)) if colour else (cal.main, cal.secondary)

    # molecules in the main half whose partner is inside the secondary half: the share that is,
    # measured on 20 000 trial positions, sets the frames
    def place(n):
        xy = np.c_[rng.uniform(main_r[0] - 0.5, main_r[0] + main_r[2] - 0.5, n),
                   rng.uniform(main_r[1] - 0.5, main_r[1] + main_r[3] - 0.5, n)]
        return xy[_inside(to_secondary(xy), sec_r)]
    share = len(place(20000)) / 20000
    per_frame = spec["density"] * main_r[2] * main_r[3] / (40 * 40) * share
    n_frames = int(np.ceil(EMITTERS / per_frame))
    counts = rng.poisson(per_frame, n_frames)
    xy = np.empty((0, 2))
    while len(xy) < counts.sum():
        xy = np.r_[xy, place(2 * (counts.sum() - len(xy)) + 100)]
    xy = xy[: counts.sum()]
    frame = np.repeat(np.arange(n_frames), counts)
    n = len(frame)
    z = rng.uniform(*ctx["z_nm"], n)
    xy2 = to_secondary(xy)

    train = ctx["training"]
    truth = {"frame": frame, "x_px": xy[:, 0], "y_px": xy[:, 1], "z_nm": z, "x2_px": xy2[:, 0], "y2_px": xy2[:, 1]}
    if colour:
        dye = rng.integers(0, 2, n)
        fraction = np.asarray(DYES)[dye]
        total = rng.uniform(*train["total"], n) if spec["photons"] == "training" else np.full(n, spec["photons"])
        phot = np.c_[total * (1 - fraction), total * fraction]
        truth.update(dye=dye, photons=total, photons_0=phot[:, 0], photons_1=phot[:, 1])
        mean_c = [np.mean(1 - np.asarray(DYES)), np.mean(DYES)]
    else:
        total = rng.uniform(*train["photons"][0], n) if spec["photons"] == "training" else np.full(n, spec["photons"])
        phot = np.c_[total, total]                  # each plane's share is in its spline
        truth.update(photons=total)
        z0 = [round(s.z0) for s in splines]
        mean_c = [float(splines[c].psf[z0[c]].sum()) for c in range(2)]
    if spec["background"] == "training":
        bg = np.stack([rng.uniform(*train["background"][c], n_frames) for c in range(2)], 1)
    else:
        bg = np.tile([0.02 * spec["photons"] * mean_c[c] for c in range(2)], (n_frames, 1))
    truth.update(background_0=bg[frame, 0], background_1=bg[frame, 1], frame_background_0=bg[:, 0],
                 frame_background_1=bg[:, 1])

    image = np.zeros((n_frames, *shape))
    for c, (region, pos) in enumerate(((main_r, xy), (sec_r, xy2))):
        x0, y0, w, h = region
        half = np.empty((n_frames, h, w))
        half[:] = bg[:, c, None, None]
        _draw_spline(half, frame, pos - [x0, y0], z, phot[:, c], splines[c])
        image[:, y0:y0 + h, x0:x0 + w] = half
    c = CAMERA
    frames = np.empty(image.shape, np.uint16)
    for k in range(n_frames):
        frames[k] = _camera(image[k], np.random.default_rng([spec["seed"], 7, k]), c["conversion"], c["offset"],
                            c["read_noise_e"] / c["conversion"], c["emgain"])
    return frames, truth, splines


def crlb(cal, splines, truth, colour: bool, excess=EXCESS_NOISE):
    """Each molecule's bound in x, y (nm, the main channel's pixels) and z, both channels together."""
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
    out = np.full((n, 3), np.nan)
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
            cov = np.linalg.inv(F)
            out[k] = np.sqrt(np.diag(cov)[:3])
        except np.linalg.LinAlgError:
            pass
    return out * [PIXEL_NM[0], PIXEL_NM[1], 1.0]


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
        manifest["contexts"][name] = {"calibration": fingerprint(SETS_DIR / f"{name}_2ct.h5"),
                                      "smap": str(ctx["smap"]), "model": str(ctx["model"]),
                                      "about": ctx["about"], "z_nm": ctx["z_nm"], "training": ctx["training"],
                                      "frame_shape": list(_layout(cals[name])[0])}
    for name, spec in SETS.items():
        if spec["context"] not in cals:
            continue
        cal, colour = cals[spec["context"]], CONTEXTS[spec["context"]]["kind"] == "colour"
        folder = SETS_DIR / name
        if (names and name in names) or not (folder / "truth.npz").exists():
            frames, truth, splines = simulate(spec, cal)
            b = crlb(cal, splines, truth, colour)
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
