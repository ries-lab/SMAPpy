"""(i)'s SMLM Challenge 2016 (3D-AS): Spline 3D and DECODE on the four training sets, against the truth.

    python benchmarks/decode/challenge/fit_and_score.py [--models DIR] [--only spline3d|decode]

Each set is fitted by Spline 3D with SMAPpy's bead calibration of the challenge's beads
(`prepare_jobs.py`; fit box 13, *as calibrated*) and by the port's DECODE model trained for its
noise level (``N1`` or ``N2``, `prepare_jobs.py`'s jobs, 1000 epochs on the H100; the model files in
``--models``, ``<noise>.safetensors``).  The camera is `prepare_jobs.CAMERA`.

**The truth** is the set's ``activations.csv`` (frame from 1, x and y in nm from the image's
corner, z in nm, the photons), its z negated: the challenge's z runs the other way from the z
SMAPpy reads off the bead stack.  **The scoring** is (i)'s (`score_sets.score`): kept as there (DECODE
at prob >= 0.5, Spline 3D within 3 MADs of its median log-likelihood and not at the iteration
limit), matched per frame by the Hungarian algorithm within 250 nm laterally and 500 nm in z,
inside a margin of 7 pixels.  **One shift** is taken off each fit first: the median offset of
its matches in x, y and z, found with the tolerances doubled.  The calibration's centre and focus
are the bead average's, not the challenge's -- a constant offset of tens of nm that the scoring
would count as error; it is the same for both fitters, which share the calibration, and is printed.

Writes ``challenge/scores.jsonl`` and prints the table.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmark"))
import score_sets  # noqa: E402
from prepare_jobs import CALIBRATION, CAMERA, ROOT, SETS, movie  # noqa: E402

PX = (100.0, 100.0)
Z_LIMIT = 750.0
SIZE = 64


def truth(name):
    a = np.loadtxt(ROOT / name / "activations.csv", delimiter=",", skiprows=1)
    # pixel k covers k*100 .. (k+1)*100 nm from the corner; SMAPpy's pixel centres are whole numbers
    # z the other way round: SMAPpy reads a bead stack's slices as the objective's steps, so a bead's
    # z is the opposite of the stage's, and the challenge's bead stack runs from its -750 to +750 nm
    # in the emitters' z; fitted z against the truth's had a slope of -0.975 before this
    return {"frame": a[:, 1].astype(np.int64) - 1, "x_px": a[:, 2] / PX[0] - 0.5, "y_px": a[:, 3] / PX[1] - 0.5,
            "z_nm": -a[:, 4], "photons": a[:, 5]}


def fit(name, fitter, models):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.io.hdf5 import load_localizations
    from smappy.plugins import Context
    out = ROOT / name / (f"{fitter}.hdf5" if fitter == "spline3d" else f"decode_{models.name}.hdf5")
    if out.exists():
        return load_localizations(out)
    camera = {f"camera.{k}": v for k, v in CAMERA.items()}
    if fitter == "spline3d":
        s = fit_settings("Localize/Spline 3D", {"source.path": str(movie(name)), **camera,
                                                "model.calibration": str(CALIBRATION), "model.focus": "calibrated",
                                                "fit.roisize": 13, "output.path": str(out), "output.raw_frames": 0})
        return plugins.get("Localize/Spline 3D")().run(Context(), s).locs
    noise = name.split(".")[1]
    s = fit_settings("Localize/DECODE", {"source.path": str(movie(name)), **camera,
                                         "model.path": str(models / f"{noise}.safetensors"), "model.threshold": 0.1,
                                         "output.path": str(out), "output.raw_frames": 0})
    return plugins.get("Localize/DECODE")().run(Context(), s).locs


def shifted(locs, t):
    """The fit with its median offset from the truth taken off; the offset."""
    keep = score_sets.kept(locs, Z_LIMIT)
    f = {"frame": np.asarray(locs["frame"])[keep], "x_px": np.asarray(locs["x_nm"], float)[keep] / PX[0],
         "y_px": np.asarray(locs["y_nm"], float)[keep] / PX[1], "z_nm": np.asarray(locs["z_nm"], float)[keep]}
    lat, ax = score_sets.LATERAL_NM, score_sets.AXIAL_NM
    score_sets.LATERAL_NM, score_sets.AXIAL_NM = 2 * lat, 2 * ax
    try:
        ti, fi = score_sets.match(t, f, PX)
    finally:
        score_sets.LATERAL_NM, score_sets.AXIAL_NM = lat, ax
    d = [float(np.median(f[k][fi] - t[k][ti])) for k in ("x_px", "y_px", "z_nm")]
    from smappy.locs import Localizations
    cols = {k: np.asarray(v) for k, v in locs.columns.items()}
    cols["x_nm"] = cols["x_nm"] - d[0] * PX[0]
    cols["y_nm"] = cols["y_nm"] - d[1] * PX[1]
    cols["z_nm"] = cols["z_nm"] - d[2]
    return Localizations(cols, dict(locs.metadata)), (d[0] * PX[0], d[1] * PX[1], d[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=pathlib.Path, default=ROOT / "models")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    rows = []
    for name in SETS:
        t = truth(name)
        for fitter in ("spline3d", "decode"):
            if a.only and fitter != a.only:
                continue
            locs, shift = shifted(fit(name, fitter, a.models), t)
            out, *_ = score_sets.score(t, locs, PX, Z_LIMIT, size=SIZE)
            row = {"set": name, "fitter": fitter, "shift_nm": shift, **out}
            rows.append(row)
            print(f"{name:10s} {fitter:9s} J {out['jaccard']:.3f}  rec {out['recall']:.3f}  prec {out['precision']:.3f}  "
                  f"lat {out['rmse_lat_nm']:5.1f}  ax {out['rmse_ax_nm']:5.1f}  eff {out['eff_lat']:.3f} / {out['eff_ax']:.3f}"
                  f"   shift {shift[0]:+.0f}, {shift[1]:+.0f}, {shift[2]:+.0f} nm", flush=True)
    with open(ROOT / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
