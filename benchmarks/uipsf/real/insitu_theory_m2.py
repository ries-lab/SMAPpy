"""(docs/psf-model-parts-plan.md, "How it is measured")  In situ without beads on M2: Spline 3D's *learn* = *phase*
started from the microscope's optics in theory (`microscope_psf.theory_pupil`) against the same learning started
from a bead calibration of the day, on the same movies -- does a theory start reach the bead start's z?

    PYTHONPATH=src python benchmarks/uipsf/real/insitu_theory_m2.py OUT [DATASET ...] [--frames N] [--beads DIR]

Runs per data set, all with *focus* on the coverslip's edge:

* ``beads/phase``  -- the bead calibration (made once with the window's defaults, or taken from ``--beads``), the
  phase learnt: the reference;
* ``theory/phase`` -- *calibration* = microscope with no pupil kept (`microscope_db.pupil_of` answers None), the
  theory drawn from M2's entry: its NA, its 3D lens's astigmatism for the file's camera, γ from the pupil it keeps
  (a bead value: what a microscope with a γ in its entry gets);
* ``nominal/phase`` -- the same with no γ known either (`gamma_of` answers none): the nominal 0.36, what a
  microscope never calibrated gets;
* ``theory/nothing`` -- the theory fitted as it is, nothing learnt: what learning has to make up.

Printed, one JSON line per run (and in ``benchmarks/uipsf/results/insitu_theory_m2.jsonl``): the focus, the
blur, the held-out decision, the share below the coverslip, the blink pairs (`insitu_learn_m2.pairs`'s measure:
consecutive-frame differences over their errors, 1 where the model fits) and, for every run against
``beads/phase``, the localizations matched within 50 nm in the same frame: the median z difference (the bias)
and its robust sd.  Passes (the plan): |bias| < 20 nm, and the blink pairs' z no more than 10 % above the bead
start's.
"""
import argparse
import json
import os
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
RUNS = ("beads/phase", "theory/phase", "nominal/phase", "theory/nothing")


def calibration(out: Path, beads_dir, name: str, beads: str, wavelength: float) -> Path:
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, run_calibration
    for folder in filter(None, (beads_dir, out)):
        path = Path(folder) / f"{name}_beads_3dcal.h5"
        if path.exists():
            return path
    path = out / f"{name}_beads_3dcal.h5"
    files = sorted(str(p) for p in (D / beads).glob("*/*.ome.tif"))
    o = OpticsSettings(NA=1.43, wavelength_nm=wavelength, n_immersion=1.516, n_medium=1.33)
    run_calibration(files, False, o, BeadSettings(), save_to=path)
    return path


def pairs(locs, err) -> dict:
    """The blink pairs' robust sd over their errors, per axis, and how many pairs (`insitu_learn_m2.pairs`)."""
    from scipy.spatial import cKDTree
    keep = np.asarray(locs["photons"]) > 1000
    x, y, z = (np.asarray(locs[c], float)[keep] for c in ("x_nm", "y_nm", "z_nm"))
    f = np.asarray(locs["frame"], float)[keep]
    big = 1e5
    d, j = cKDTree(np.c_[x, y, f * big]).query(np.c_[x, y, (f + 1) * big], distance_upper_bound=80.0)
    i = np.flatnonzero(np.isfinite(d))
    j = j[i]
    out = {"pairs": int(len(i))}
    for axis, v in (("x", x), ("y", y), ("z", z)):
        e = np.asarray(locs[err[axis]], float)[keep]
        r = (v[i] - v[j]) / np.sqrt(e[i] ** 2 + e[j] ** 2)
        r = r[np.isfinite(r)]
        out[axis] = round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3) if len(r) else None
    return out


def matched_z(a, b, within_nm: float = 50.0) -> dict:
    """The localizations of ``a`` and ``b`` in the same frame within ``within_nm`` laterally: how many, the median
    of b's z less a's, and its robust sd."""
    from scipy.spatial import cKDTree
    big = 1e6
    pa = np.c_[np.asarray(a["x_nm"], float), np.asarray(a["y_nm"], float), np.asarray(a["frame"], float) * big]
    pb = np.c_[np.asarray(b["x_nm"], float), np.asarray(b["y_nm"], float), np.asarray(b["frame"], float) * big]
    d, j = cKDTree(pb).query(pa, distance_upper_bound=within_nm)
    i = np.flatnonzero(np.isfinite(d))
    dz = np.asarray(b["z_nm"], float)[j[i]] - np.asarray(a["z_nm"], float)[i]
    dz = dz[np.isfinite(dz)]
    if not len(dz):
        return {"matched": 0}
    med = float(np.median(dz))
    return {"matched": int(len(dz)), "share": round(len(dz) / len(pa), 3), "bias_nm": round(med, 1),
            "sd_nm": round(float(1.4826 * np.median(np.abs(dz - med))), 1)}


def error_columns(columns) -> dict:
    lateral = "x_err_nm" if "x_err_nm" in columns else "xy_err_nm"
    return {"x": lateral, "y": "y_err_nm" if "y_err_nm" in columns else lateral, "z": "z_err_nm"}


class Theory:
    """The microscope as one that keeps no pupil (and, ``nominal``, knows no γ either), while it is entered."""

    def __init__(self, nominal: bool):
        self.nominal = nominal

    def __enter__(self):
        import smappy.microscope_db as db
        self.saved = db.pupil_of, db.gamma_of
        db.pupil_of = lambda summary: None
        if self.nominal:
            db.gamma_of = lambda summary, wavelength_nm=None: (None, None, "")
        return self

    def __exit__(self, *exc):
        import smappy.microscope_db as db
        db.pupil_of, db.gamma_of = self.saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--frames", type=int, default=10000)
    p.add_argument("--runs", nargs="*", default=list(RUNS))
    p.add_argument("--beads", help="a folder with <dataset>_beads_3dcal.h5 made before")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from smappy.calibrate import depth
    from smappy.calibrate.microscope_psf import MICROSCOPE
    from smappy.plugins import Context
    from smappy.plugins.fit import OutputSettings, SourceSettings, SplineFit, SplineFitSettings

    from uipsf.device import default_device
    results = Path(__file__).resolve().parents[1] / "results" / "insitu_theory_m2.jsonl"

    for name in args.datasets:
        beads, movie, suffix, wavelength = DATASETS[name]
        cal = calibration(out, args.beads, name, beads, wavelength)
        reference = None
        for run in args.runs:
            start, learn = run.split("/")
            path = out / f"{name}_{start}_{learn}_locs.hdf5"
            s = SplineFitSettings(source=SourceSettings(path=str(D / movie / f"{movie}{suffix}"), stop=args.frames),
                                  output=OutputSettings(path=str(path)))
            s = replace(s, model=replace(s.model, calibration=str(cal) if start == "beads" else MICROSCOPE,
                                         focus="edge", learn=learn))
            t = time.time()
            ctx = Context(progress=lambda text: None)
            plugin = SplineFit()
            if start == "beads":
                result = plugin.run(ctx, s)
                key_settings = s
            else:
                with Theory(nominal=start == "nominal"):
                    result = plugin.run(ctx, s)
                    key_settings = plugin.with_calibration(s)
            minutes = (time.time() - t) / 60
            found = depth.found(plugin._focus_key(key_settings))
            locs = result.locs
            z = np.asarray(locs["z_nm"], float)
            row = dict(dataset=name, run=run, device=str(default_device()), frames=args.frames,
                       localizations=int(len(locs)),
                       focus_nm=None if found is None else round(found.focus_nm, 1),
                       edge_nm=None if found is None or found.edge_nm is None else round(found.edge_nm, 1),
                       blur_nm=None if found is None or found.blur_nm is None else round(found.blur_nm, 1),
                       fallback=None if found is None else found.fallback,
                       blink_pairs=pairs(locs, error_columns(locs.columns)),
                       below_coverslip=round(float(np.mean(z < -50)), 4),
                       z_percentiles=[round(float(v)) for v in np.nanpercentile(z, [5, 25, 50, 75, 95])],
                       warnings=list(getattr(ctx, "warnings", [])), minutes=round(minutes, 1))
            if run == "beads/phase":
                reference = locs
            elif reference is not None:
                row["against_beads"] = matched_z(reference, locs)
            print(json.dumps(row), flush=True)
            with open(results, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            del locs, result


if __name__ == "__main__":
    main()
