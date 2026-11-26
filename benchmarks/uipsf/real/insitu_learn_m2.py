"""(docs/insitu-fitting-plan.md, the open question)  Spline 3D's *focus in situ* with each *learn* -- the focus
alone, the aberrations, everything -- on M2's movies, run through the fitter itself: how large is the held-out
gain a learnt pupil wins by, and does it say anything the fit's own measures agree with?

    PYTHONPATH=src python benchmarks/uipsf/real/insitu_learn_m2.py OUT [DATASET ...] [--frames N]

Per data set a bead calibration with a pupil model is made once with the window's defaults (kept in OUT), then
Spline 3D fits the first N frames with *focus* in situ at each *learn*.  Printed, one JSON line per run (and in
``benchmarks/uipsf/results/insitu_learn_m2.jsonl``): the focus, the held-out gain in nats per molecule, whether
it fell back, and two measures that need no truth --

* *blink pairs*: one blink's fits in consecutive frames (laterally within 80 nm), the robust sd of their
  differences over the root sum of their errors, per axis: 1 where the errors are honest and the model fits;
  a wrong model shows as more than 1, in z first;
* *below the coverslip*: the share of localizations more than 50 nm under it (z is the height above it), which
  a model with the depth's aberrations wrong pushes up.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

D = Path(os.environ.get("SMAPPY_M2_DATA",
                      "/Users/ries/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/smappy test"))
DATASETS = {
    "jan": ("260105_AA_BeadsCalZ_638_M2", "260105_AA_CLTA-B-SNAP_AF647_C8_638i75_UVi20_25ms_3D_1",
            "_MMStack_Default.ome.tif", 680.0),
    "aug": ("260821_AASJ_BeadsCalZ_638_665LP_M2", "260821_AASJ_sun1_mRuby_pS8_CF_him8_A647_3D_2",
            "_MMStack_Pos0.ome.tif", 680.0),
    "sep": ("260917_AA_BeadsCalZ_561_SingleMode_M2", "260917_AA_CLC-SNAP-BG-5xR2_Cy3B_561i20_PCAD_3D_200ms_1",
            "_MMStack_Pos0.ome.tif", 590.0),
}
LEARN = ("focus", "aberrations", "everything")


def calibration(out: Path, name: str, beads: str, wavelength: float) -> Path:
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, run_calibration
    path = out / f"{name}_beads_3dcal.h5"
    if not path.exists():
        files = sorted(str(p) for p in (D / beads).glob("*/*.ome.tif"))
        # the pixels the camera's (M2's iXon: 127 x 116 nm), as the fit takes them: typed 116 x 127 they were
        # swapped against the fit's, and a learnt pupil had that to make up for too
        o = OpticsSettings(NA=1.43, wavelength_nm=wavelength, n_immersion=1.516, n_medium=1.33)
        run_calibration(files, False, o, BeadSettings(), save_to=path)
    return path


def pairs(locs, err) -> dict:
    """The blink pairs' robust sd over their errors, per axis, and how many pairs."""
    from scipy.spatial import cKDTree
    keep = np.asarray(locs["photons"]) > 1000
    x, y, z = (np.asarray(locs[c], float)[keep] for c in ("x_nm", "y_nm", "z_nm"))
    f = np.asarray(locs["frame"], float)[keep]
    big = 1e5
    tree = cKDTree(np.c_[x, y, f * big])
    d, j = tree.query(np.c_[x, y, (f + 1) * big], distance_upper_bound=80.0)
    i = np.flatnonzero(np.isfinite(d))
    j = j[i]
    out = {"pairs": int(len(i))}
    for axis, v in (("x", x), ("y", y), ("z", z)):
        e = np.asarray(locs[err[axis]], float)[keep]
        r = (v[i] - v[j]) / np.sqrt(e[i] ** 2 + e[j] ** 2)
        r = r[np.isfinite(r)]
        out[axis] = round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3) if len(r) else None
    return out


def error_columns(columns) -> dict:
    lateral = "x_err_nm" if "x_err_nm" in columns else "xy_err_nm"
    return {"x": lateral, "y": "y_err_nm" if "y_err_nm" in columns else lateral, "z": "z_err_nm"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--frames", type=int, default=10000)
    p.add_argument("--learn", nargs="*", default=list(LEARN), help="which learn levels")
    p.add_argument("--cpu", action="store_true", help="learn on the CPU; default: uipsf's device (CUDA, MPS)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import smappy.calibrate.insitu_run as ir
    from smappy.calibrate import depth
    from smappy.plugins import Context
    from smappy.plugins.fit import OutputSettings, SourceSettings, SplineFit, SplineFitSettings

    from uipsf.device import default_device
    if args.cpu:
        base = ir.run_insitu
        ir.run_insitu = lambda *a, **k: base(*a, **dict(k, device="cpu"))
    results = Path(__file__).resolve().parents[1] / "results" / "insitu_learn_m2.jsonl"

    for name in args.datasets:
        beads, movie, suffix, wavelength = DATASETS[name]
        t = time.time()
        cal = calibration(out, name, beads, wavelength)
        print(f"{name}: calibration {cal} ({(time.time() - t) / 60:.1f} min)", flush=True)
        for learn in args.learn:
            s = SplineFitSettings(
                source=SourceSettings(path=str(D / movie / f"{movie}{suffix}"), stop=args.frames),
                output=OutputSettings(path=str(out / f"{name}_{learn}_locs.hdf5")))
            s = replace(s, model=replace(s.model, calibration=str(cal), focus="insitu", learn=learn))
            t = time.time()
            ctx = Context(progress=lambda text: None)
            result = SplineFit().run(ctx, s)
            said = list(getattr(ctx, "warnings", []))
            minutes = (time.time() - t) / 60
            found = depth.kept(SplineFit()._focus_key(s))
            locs = result.locs
            z = np.asarray(locs["z_nm"], float)
            row = dict(dataset=name, learn=learn, device="cpu" if args.cpu else str(default_device()),
                       frames=args.frames, localizations=int(len(locs)),
                       focus_nm=None if found is None else round(found.focus_nm, 1),
                       blur_nm=None if found is None or found.blur_nm is None else round(found.blur_nm, 1),
                       fallback=None if found is None else found.fallback,
                       heldout=None if found is None else found.heldout,
                       blink_pairs=pairs(locs, error_columns(locs.columns)),
                       below_coverslip=round(float(np.mean(z < -50.0)), 4),
                       z_percentiles=np.nanpercentile(z, [5, 25, 50, 75, 95]).round(0).tolist(),
                       warnings=[str(w) for w in said], minutes=round(minutes, 1))
            line = json.dumps(row, default=str)
            print(line, flush=True)
            with results.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
