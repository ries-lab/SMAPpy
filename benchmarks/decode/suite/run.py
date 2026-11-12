"""Fit a run on the benchmark's sets and write its report (`docs/decode/benchmark.md`).

    python benchmarks/decode/suite/run.py RUN [--sets NAME ...] [--refit] [--device DEV]

A run is ``runs/<RUN>.yaml``: the fitter that runs it (an adapter below), its parts by name, one model
per context, the fitter's settings and what the scoring keeps.  Each fit is kept under
``fits/<RUN>/<set>.hdf5`` and fitted again only with ``--refit``; the report, ``reports/<RUN>.json``,
holds the run's spec, what each model file records (architecture, parameters, the training's ranges
and epochs), the cost, and each set's scores and speed.

**Adapters** run a fitter on one set the way a user would, through its plugin, the frames as they
are: SMAPpy's Spline 3D and the port's DECODE (`Localize/DECODE`, which mirrors the frames where the
model's calibration was taken the other way).  The new fitter adds its own; a run of it differs from
another only in what its ``parts`` name.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402
import sets as S  # noqa: E402


def load_run(name) -> dict:
    """A run's YAML; ``base: OTHER`` takes OTHER's and changes what this one names -- a part replaced
    whole, ``fit`` and ``keep`` merged key by key -- so that a variant states only its difference."""
    import yaml
    run = yaml.safe_load((HERE / "runs" / f"{name}.yaml").read_text())
    if run.get("base"):
        base = load_run(run["base"])
        for key in ("parts", "fit", "keep", "models"):
            run[key] = {**base.get(key, {}), **(run.get(key) or {})}
        for key in ("fitter", "about"):
            run.setdefault(key, base.get(key))
    run.setdefault("parts", {})
    run.setdefault("models", {})
    run.setdefault("fit", {})
    run.setdefault("keep", {})
    return run


def _camera_settings(cam) -> dict:
    return {f"camera.{k}": v for k, v in cam.items()}


def _source(name) -> dict:
    spec = S.SETS[name]
    out = {"source.path": str(S.movie(name))}
    if "start" in spec:
        out.update({"source.start": spec["start"], "source.stop": spec["stop"]})
    return out


def fit_spline3d(run, name, out, device=None):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.plugins import Context
    cal = S.CONTEXTS[S.SETS[name]["context"]]["calibration"]
    s = fit_settings("Localize/Spline 3D", {
        **_source(name), **_camera_settings(S.camera(name)), "model.calibration": str(cal),
        "model.focus": run["fit"].get("focus", "calibrated"), "fit.roisize": run["fit"].get("roisize", 13),
        "output.path": str(out), "output.raw_frames": 0})
    plugins.get("Localize/Spline 3D")().run(Context(), s)
    return {"device": "cpu", "calibration": str(cal)}


def fit_decode_smlm(run, name, out, device=None):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.plugins import Context
    from extraction import extraction
    model = model_path(run, S.SETS[name]["context"])
    s = fit_settings("Localize/DECODE", {
        **_source(name), **_camera_settings(S.camera(name)), "model.path": str(model),
        "model.threshold": run["fit"].get("threshold", 0.1), "model.device": device or "auto",
        "output.path": str(out), "output.raw_frames": 0})
    with extraction(run["fit"].get("extraction")):
        plugins.get("Localize/DECODE")().run(Context(), s)
    from smappy.io.hdf5 import load_localizations
    dec = load_localizations(out).metadata.get("decode", {})           # the fitter's record, as saved
    return {"device": dec.get("device"), "model": str(model), "frames_mirrored": dec.get("frames_mirrored")}


ADAPTERS = {"spline3d": fit_spline3d, "decode_smlm": fit_decode_smlm}


def model_path(run, context):
    p = run["models"].get(context)
    return None if p is None else S.DATA / p


def model_record(path) -> dict:
    """What a model file says of itself: parameters (counted from its tensors' shapes), architecture,
    the training's ranges, epochs and batch."""
    import struct
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    meta = json.loads(header.pop("__metadata__", {}).get("smappy", "{}"))
    n = 0
    for v in header.values():
        k = 1
        for d in v["shape"]:
            k *= d
        n += k
    keys = ("method", "architecture", "arch_param", "channels_in", "post_processing", "trained_on", "epoch",
            "epochs", "batch_size", "learning_rate", "base_model")
    return {"parameters": n, **{k: meta[k] for k in keys if k in meta}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from smappy.io.hdf5 import load_localizations
    run = load_run(a.run)
    adapter = ADAPTERS[run["fitter"]]
    path = R.REPORTS / f"{a.run}.json"
    report = json.loads(path.read_text()) if path.exists() else {"sets": {}}
    report.update(run=run, host=platform.node(), models={c: model_record(model_path(run, c)) for c in run["models"]
                                                         if model_path(run, c).exists()})
    (R.FITS / a.run).mkdir(parents=True, exist_ok=True)
    R.REPORTS.mkdir(parents=True, exist_ok=True)
    for name, spec in S.SETS.items():
        if a.sets and name not in a.sets:
            continue
        if run["models"] and spec["context"] not in run["models"]:
            print(f"{name}: no model for context {spec['context']}, skipped")
            continue
        out = R.FITS / a.run / f"{name}.hdf5"
        entry = report["sets"].get(name, {})
        if a.refit or not out.exists():
            t0 = time.perf_counter()
            how = adapter(run, name, out, a.device)
            seconds = time.perf_counter() - t0
            frames = (spec["stop"] - spec["start"]) if "stop" in spec else None
            if frames is None:
                import tifffile
                with tifffile.TiffFile(S.movie(name)) as t:
                    frames = len(t.pages)
            entry = {"fit": how, "seconds": seconds, "frames": frames, "frames_per_s": frames / seconds}
        entry["scores"] = R.score_set(name, load_localizations(out), run["keep"])
        report["sets"][name] = entry
        path.write_text(json.dumps(report, indent=1, default=str))
        sc = entry["scores"]
        ks = max((v for v in (sc["ks_x"], sc["ks_y"]) if v is not None), default=float("nan"))
        line = (f"J {sc['jaccard']:.3f}  lat {sc['rmse_lat_nm']:.1f}  ax {sc['rmse_ax_nm']:.1f}  KS {ks:.3f}"
                if "jaccard" in sc else f"{sc['per_frame']:.1f} a frame  KS {ks:.3f}")
        print(f"{name:22s} {line}  ({entry.get('frames_per_s', float('nan')):.0f} frames/s)", flush=True)
    report["cost"] = {"parameters": {c: m["parameters"] for c, m in report["models"].items()},
                      "epochs": {c: m.get("epochs") for c, m in report["models"].items()}} if report["models"] else {}
    path.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
