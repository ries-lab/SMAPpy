"""The benchmark's sets (`docs/decode/benchmark.md`): what every run is scored on, drawn once and kept.

    python benchmarks/decode/suite/sets.py draw            # the simulated sets, with each emitter's CRLB
    python benchmarks/decode/suite/sets.py list

**Simulated** (calibration B, SMAPpy's pupil model of the example package's beads): single frames of
64 x 64 pixels, about ``EMITTERS`` emitters a set, x and y uniform over the frame (pixel centres at
whole numbers), z uniform within +-``Z_NM``.  Two sweeps, each changing one thing:

* **density** -- 1, 5, 15, 25 and 50 emitters per 40 x 40 pixels, photons and background drawn as
  the example package's training draws them (normal 7000 +- 3000 cut at 1000; 20 -- 200 a pixel);
* **photons** -- 1000, 3000 and 10 000 at density 1, the background 2 % of the photons a pixel
  (20, 60, 200): the signal to noise from poor to good at a fixed ratio, inside the training's range.

z is not a set of its own: every set spreads it, and the report bins it.  The frames are drawn as an
EMCCD reads them -- the calibration's beads were taken without EM gain, so the PSF is mirrored in x
(`matched_to_em`) -- and the camera is the training's (gain 100, 5 e-/ADU, read noise 58.8 e-).  Every
fitter then gets the frames as a real movie's and mirrors its own calibration where it must.

**Each emitter's CRLB** is kept with its truth: the Poisson bound of SMAPpy's spline fitter
(`psf.SplinePSF`) on the spot alone, noise-free, doubled in variance for the EM register's excess
noise -- in 13 x 13 pixels (``crlb_*``, Spline 3D's own fit box) and in 25 x 25, the calibration's
whole spot (``crlb_*_full``, which a network seeing the whole frame could reach; 19 % lower in z).

**The SMLM Challenge 2016 3D-AS** training sets and **real data** -- a fixed range of frames of the
example package's movie and of the DECODE paper's Fig. 4e raw movie -- are named here, not copied: the
files are kept where they are, their size and first megabyte's hash recorded.

Each set names its *context*, the calibration and camera a model is trained for: a run (`runs/`)
gives one model per context, so one training covers every simulated set and the example movie.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm"))
ROOT = DATA / "benchmark" / "v1"
SETS_DIR = ROOT / "sets"

SIZE = 64
EMITTERS = 20000
Z_NM = 700.0
PIXEL_NM = (117.0, 127.0)                      # x, y
# the example package's training camera, as an EMCCD reads it
SIM_CAMERA = {"conversion": 5.0, "offset": 398.6, "em_on": True, "emgain": 100.0,
              "pixelsize_um": PIXEL_NM[0] / 1000, "pixelsize_y_um": PIXEL_NM[1] / 1000, "read_noise_e": 58.8}
EXCESS_NOISE = 2.0                             # an EM register's at high gain, in variance
CRLB_ROI, CRLB_ROI_FULL = 13, 25

FIG4E = DATA / "fig4e_raw"
FIG4E_JOB = FIG4E / "200723_Ulf_U2OS_MT_a+b_tubulin_15ms_EM50_LP75b0_3_MMStack_Default_decode" / "261111_beads_3Dcal_2026-11-12"
CHALLENGE = DATA / "challenge"

# the calibration each context's frames were drawn or taken with: Spline 3D's model, and what a
# network for that context is trained on
CONTEXTS = {
    "B": {"calibration": DATA / "pixelation" / "beads_pupil.h5",
          "about": "SMAPpy's pupil model of the example package's 11 bead stacks"},
    "challenge_N1": {"calibration": CHALLENGE / "cal" / "beads_AS_3dcal.h5",
                     "about": "SMAPpy's calibration of the SMLM Challenge's AS beads"},
    "challenge_N2": {"calibration": CHALLENGE / "cal" / "beads_AS_3dcal.h5",
                     "about": "as challenge_N1; a model trained for the N2 noise level"},
    "fig4e": {"calibration": FIG4E_JOB / "calibration.h5",
              "about": "the Fig. 4e beads (261111_beads_3Dcal), as the GUI's training job copied it"},
}

DENSITIES = (1, 5, 15, 25, 50)
PHOTONS = (1000, 3000, 10000)
TRAINING_PHOTONS = (7000.0, 3000.0, 1000.0)    # mean, spread, the least
TRAINING_BACKGROUND = (20.0, 200.0)


def _simulated():
    out = {}
    for i, d in enumerate(DENSITIES):
        out[f"density_{d}"] = {"kind": "simulated", "context": "B", "seed": 2000 + i, "density": d,
                               "photons": {"normal": TRAINING_PHOTONS}, "background": {"uniform": TRAINING_BACKGROUND}}
    for i, p in enumerate(PHOTONS):
        out[f"photons_{p}"] = {"kind": "simulated", "context": "B", "seed": 2100 + i, "density": 1,
                               "photons": {"fixed": float(p)}, "background": {"fixed": 0.02 * p}}
    return out


def _challenge():
    camera = {"conversion": 45.0, "offset": 100.0, "em_on": True, "emgain": 300.0, "pixelsize_um": 0.1,
              "pixelsize_y_um": 0.1, "read_noise_e": 74.4}
    return {f"challenge_{n}": {"kind": "challenge", "context": f"challenge_{n.split('.')[1]}", "name": n,
                               "movie": CHALLENGE / n / "data" / f"sequence-as-stack-{n}-AS-Exp.tif",
                               "truth": CHALLENGE / n / "activations.csv", "camera": camera}
            for n in ("MT0.N1.LD", "MT0.N1.HD", "MT0.N2.LD", "MT0.N2.HD")}


def _real():
    return {
        # the example package's movie as it was taken (EM gain 50); SMAP's fit of it for the camera
        "example": {"kind": "real", "context": "B", "movie": DATA / "experimental_data_workflow" / "frames.tif",
                    "start": 0, "stop": 5000,
                    "camera": {"conversion": 5.0, "offset": 398.6, "em_on": True, "emgain": 50.0,
                               "pixelsize_um": 0.117, "pixelsize_y_um": 0.127}},
        # the Fig. 4e raw movie, a Micro-Manager file of 32 263 frames of 257 x 213; the camera as the
        # GUI's training job read it from the microscope database
        "fig4e": {"kind": "real", "context": "fig4e",
                  "movie": FIG4E / "200723_Ulf_U2OS_MT_a+b_tubulin_15ms_EM50_LP75b0_3_MMStack_Default.ome.tif",
                  "start": 2000, "stop": 7000,
                  "camera": {"conversion": 6.7, "offset": 398.6, "em_on": True, "emgain": 50.0,
                             "pixelsize_um": 0.127, "pixelsize_y_um": 0.127, "read_noise_e": 74.4}},
    }


SETS = {**_simulated(), **_challenge(), **_real()}


# ----------------------------------------------------------------------------- drawing
def _draw(spec, n, rng):
    (kind, arg), = spec.items()
    if kind == "fixed":
        return np.full(n, float(arg))
    if kind == "uniform":
        return rng.uniform(*arg, n)
    mu, sd, least = arg
    return np.maximum(rng.normal(mu, sd, n), least)


def simulate(spec, cal):
    """Frames (uint16) and truth of one simulated set, ``cal`` already in the EM camera's orientation."""
    from smappy.simulate.camera import _camera, _draw_spline
    rng = np.random.default_rng(spec["seed"])
    per_frame = spec["density"] * SIZE * SIZE / (40 * 40)
    n_frames = int(np.ceil(EMITTERS / per_frame))
    frame = np.repeat(np.arange(n_frames), rng.poisson(per_frame, n_frames))
    n = len(frame)
    x = rng.uniform(-0.5, SIZE - 0.5, n)
    y = rng.uniform(-0.5, SIZE - 0.5, n)
    z = rng.uniform(-Z_NM, Z_NM, n)
    photons = _draw(spec["photons"], n, rng)
    background = _draw(spec["background"], n_frames, rng)
    image = np.empty((n_frames, SIZE, SIZE))
    image[:] = background[:, None, None]
    _draw_spline(image, frame, np.stack([x, y], 1), z, photons, cal)
    c = SIM_CAMERA
    frames = np.empty(image.shape, np.uint16)
    for k in range(n_frames):
        frames[k] = _camera(image[k], np.random.default_rng([spec["seed"], 7, k]), c["conversion"], c["offset"],
                            c["read_noise_e"] / c["conversion"], c["emgain"])
    return frames, {"frame": frame, "x_px": x, "y_px": y, "z_nm": z, "photons": photons,
                    "background": background[frame], "frame_background": background}


def crlb(cal, photons, background, z, roi, excess=EXCESS_NOISE, seed=0):
    """Each emitter's bound in x, y (nm) and z: SMAPpy's spline fitter on its spot alone, noise-free,
    in ``roi`` x ``roi`` pixels, at a random place within the centre pixel; times sqrt(``excess``)."""
    from smappy.psf import SplinePSF
    from smappy.simulate.camera import _draw_spline
    rng = np.random.default_rng(seed)
    out = np.full((len(z), 3), np.nan)
    psf = SplinePSF(cal)
    for a in range(0, len(z), 5000):
        s = slice(a, a + 5000)
        m = len(z[s])
        image = np.repeat(np.asarray(background[s], float)[:, None, None], roi, 1).repeat(roi, 2)
        xy = roi // 2 + rng.uniform(-0.5, 0.5, (m, 2))
        _draw_spline(image, np.arange(m), xy, z[s], photons[s], cal)
        found = psf.unpack(psf.fit(np.ascontiguousarray(image, np.float32)))
        out[s] = np.c_[found["x_err_pix"] * PIXEL_NM[0], found["y_err_pix"] * PIXEL_NM[1], found["z_err_nm"]]
    return out * np.sqrt(excess)


def fingerprint(path) -> dict:
    """A kept file's size and the sha256 of its first megabyte: enough to see that it was swapped."""
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        head = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256_first_mb": head}


def draw(names=None):
    import tifffile
    from smappy.io.calibration import load_spline_calibration, matched_to_em
    cal = load_spline_calibration(CONTEXTS["B"]["calibration"])
    drawn = matched_to_em(cal, True)
    manifest = {"pixel_nm": PIXEL_NM, "size": SIZE, "z_nm": Z_NM, "camera": SIM_CAMERA, "excess_noise": EXCESS_NOISE,
                "contexts": {k: {"calibration": fingerprint(v["calibration"]), "about": v["about"]}
                             for k, v in CONTEXTS.items() if pathlib.Path(v["calibration"]).exists()},
                "sets": {}}
    for name, spec in SETS.items():
        entry = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in spec.items()}
        if spec["kind"] == "simulated":
            folder = SETS_DIR / name
            if (names and name in names) or not (folder / "truth.npz").exists():
                frames, truth = simulate(spec, drawn)
                for key, roi in (("crlb", CRLB_ROI), ("crlb_full", CRLB_ROI_FULL)):
                    b = crlb(cal, truth["photons"], truth["background"], truth["z_nm"], roi, seed=spec["seed"])
                    truth.update({f"{key}_x_nm": b[:, 0], f"{key}_y_nm": b[:, 1], f"{key}_z_nm": b[:, 2]})
                folder.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(folder / "frames.tif", frames, photometric="minisblack")
                np.savez(folder / "truth.npz", **truth)
                print(f"{name}: {len(frames)} frames, {len(truth['frame'])} emitters; median CRLB x / z "
                      f"{np.nanmedian(truth['crlb_x_nm']):.1f} / {np.nanmedian(truth['crlb_z_nm']):.1f} nm", flush=True)
            t = np.load(folder / "truth.npz")
            entry.update(frames=int(t["frame_background"].shape[0]), emitters=int(len(t["frame"])),
                         movie=str(folder / "frames.tif"), file=fingerprint(folder / "frames.tif"))
        else:
            entry["file"] = fingerprint(spec["movie"]) if pathlib.Path(spec["movie"]).exists() else None
        manifest["sets"][name] = entry
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("manifest:", ROOT / "manifest.json")


def movie(name) -> pathlib.Path:
    spec = SETS[name]
    return SETS_DIR / name / "frames.tif" if spec["kind"] == "simulated" else pathlib.Path(spec["movie"])


def camera(name) -> dict:
    return SIM_CAMERA if SETS[name]["kind"] == "simulated" else SETS[name]["camera"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("draw", "list"))
    ap.add_argument("--only", nargs="*", default=None, help="draw these again (the rest only if missing)")
    a = ap.parse_args()
    if a.what == "draw":
        draw(a.only)
    else:
        for name, spec in SETS.items():
            print(f"{name:20s} {spec['kind']:10s} {spec['context']}")


if __name__ == "__main__":
    sys.exit(main())
