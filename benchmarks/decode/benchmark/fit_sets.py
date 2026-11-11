"""Every fitter on every simulated set of (i): Spline 3D and the DECODE models trained on the same spline.

    python benchmarks/decode/benchmark/fit_sets.py [--calibration A] [--only NAME ...]

The fitters on A (on B, Spline 3D and the port's model trained on B, `models`): Spline 3D (fit box 13, *focus as calibrated*); the package's published model; the
original's two reference runs (``original_run{1,2}``, their final weights, exported to model files
by `decode_smlm.predict.export_original`); the port's two (``port_run{0,1}``).  DECODE's models are
run through the predictor and `smappy.window_engine.WindowEngine`, the plugin's own path, with the
frames as drawn (no mirror: the sets are in the spline's orientation), at DECODE's threshold 0.1;
the scoring filters further.  Each fit goes to ``<set>/<fitter>.hdf5``.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pixelation"))
from common import DATA, OUT, PACKAGE, SMAP_SPLINE  # noqa: E402

ROOT = DATA / "benchmark"
MODELS = ROOT / "models"


def models(calibration="A"):
    """The DECODE models, as model files: the package's and the original runs' exported once.

    On B, the port's model trained on SMAPpy's bead calibration (the pixelation study's ``decode_B``,
    trained after the ``x0`` fix) is the only DECODE model: the others were trained on A's spline.
    """
    if calibration == "B":
        return {"port_B": OUT / "decode_B.safetensors"}
    from decode_smlm.predict import export_original
    from smappy.io.calibration import load_spline_calibration
    cal = load_spline_calibration(SMAP_SPLINE)
    out = {}
    for name, (param, weights) in {"package": (PACKAGE / "param_run.yaml", PACKAGE / "model.pt"),
                                   "original_run1": (MODELS / "original_run1.yaml", MODELS / "original_run1.pt"),
                                   "original_run2": (MODELS / "original_run2.yaml", MODELS / "original_run2.pt")}.items():
        path = MODELS / f"{name}.safetensors"
        if not path.exists():
            export_original(param, weights, path, calibration=cal)
        out[name] = path
    for name in ("port_run0", "port_run1"):
        out[name] = MODELS / f"{name}.safetensors"
    return out


def camera(meta):
    from smappy.metadata import CameraMetadata
    c = meta["camera"]
    return CameraMetadata(conversion=c["conversion"], offset=c["offset"], em_on=True, emgain=c["em_gain"],
                          pixelsize_um=tuple(p / 1000 for p in meta["pixel_nm"]),
                          read_noise_e=c["read_noise_adu"] * c["conversion"])


def fit_spline(folder, meta):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.plugins import Context
    c = meta["camera"]
    settings = fit_settings("Localize/Spline 3D", {
        "source.path": str(folder / "frames.tif"), "camera.conversion": c["conversion"], "camera.offset": c["offset"],
        "camera.em_on": True, "camera.emgain": c["em_gain"], "camera.pixelsize_um": meta["pixel_nm"][0] / 1000,
        "camera.pixelsize_y_um": meta["pixel_nm"][1] / 1000, "camera.read_noise_e": c["read_noise_adu"] * c["conversion"],
        "model.calibration": meta["calibration"], "model.focus": "calibrated", "fit.roisize": 13,
        "output.path": str(folder / "spline3d.hdf5"), "output.raw_frames": 0})
    return plugins.get("Localize/Spline 3D")().run(Context(), settings).text.splitlines()[0]


def fit_decode(folder, meta, name, path, device):
    import tifffile
    from decode_smlm.predict import load_predictor
    from smappy.io.hdf5 import save_localizations
    from smappy.window_engine import WindowEngine, _joined
    frames = tifffile.imread(folder / "frames.tif")
    engine = WindowEngine(camera(meta), load_predictor(path, device, frames_mirrored=False, raw_th=0.1))
    parts = [engine.push(frames[a:a + 500], a) for a in range(0, len(frames), 500)] + [engine.end()]
    locs = _joined(parts)
    save_localizations(folder / f"{name}.hdf5", locs, {"decode": {"model": str(path)}})
    return f"{len(locs)} localizations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="A")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    root = ROOT / a.calibration
    meta = json.loads((root / "sets.json").read_text())
    fitters = {"spline3d": None, **models(a.calibration)}
    for d in meta["sets"]:
        folder = root / f"density_{d}"
        for name, path in fitters.items():
            if a.only and name not in a.only:
                continue
            t0 = time.perf_counter()
            text = fit_spline(folder, meta) if path is None else fit_decode(folder, meta, name, path, a.device)
            print(f"density {d:>2} {name:14s} {text}  ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
