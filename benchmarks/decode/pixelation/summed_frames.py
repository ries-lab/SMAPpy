"""(i) on the example's real frames: density, by adding one frame's molecules to another's.

    python benchmarks/decode/pixelation/summed_frames.py

There is no truth in a real movie, but there is an answer for a sum: frame t with frame t + 10 000
added holds the molecules of both, and each fitter's own localizations in the two frames alone are
what it should find in the sum.  The molecules of the second frame are added without its
background -- a per-pixel median over its 11 neighbouring frames taken off -- so the mean background
stays the data's (the models were trained for 20 -- 200 photons a pixel, and twice the data's would
leave that), and only the background's noise grows (its variance by up to half again: the median's
own noise is small).  The sum is written in camera counts again.

**The sets:** ``N_SUMS`` sums of frames 0 ... N_SUMS - 1 with frames 10 000 ... (``summed_raw.tif``,
for DECODE with B, which flips it into the beads' orientation itself, its positions mirrored back as
`fit_decode.py` does; ``summed_flipped.tif`` for Spline 3D with B).  **The reference** of a fitter:
its localizations in the two frames alone, kept as the study keeps them (`analyse.keep`); **its
answer**: its localizations in the sum, kept the same way.  Matched per frame by the Hungarian
algorithm within 100 nm laterally and 250 nm in z: recall (of the reference found again), precision
(of the sum's that are in the reference) and Jaccard, against the density -- the reference's
localizations a frame, in a 40 x 40 pixel area.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyse import keep  # noqa: E402
from common import FLIPPED, MOVIE, MOVIE_CAMERA, OUT, PIXEL_NM  # noqa: E402

N_SUMS = 2000
SHIFT = 10000
LATERAL_NM, AXIAL_NM = 100.0, 250.0
GAIN = MOVIE_CAMERA["conversion"] / MOVIE_CAMERA["emgain"]          # photons a count


def summed(movie, out):
    import tifffile
    stack = tifffile.memmap(movie) if str(movie).endswith(".tif") else tifffile.imread(movie)
    offset = MOVIE_CAMERA["offset"]
    sums = np.empty((N_SUMS,) + stack.shape[1:], np.float32)
    for t in range(N_SUMS):
        a = (stack[t].astype(np.float64) - offset) * GAIN
        b = (stack[SHIFT + t].astype(np.float64) - offset) * GAIN
        around = (stack[SHIFT + t - 5:SHIFT + t + 6].astype(np.float64) - offset) * GAIN
        sums[t] = ((a + b - np.median(around, axis=0)) / GAIN + offset).astype(np.float32)
    tifffile.imwrite(out, sums, photometric="minisblack")
    return out


def fit_spline(path, out):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.plugins import Context
    c = MOVIE_CAMERA
    s = fit_settings("Localize/Spline 3D", {
        "source.path": str(path), "camera.conversion": c["conversion"], "camera.offset": c["offset"],
        "camera.em_on": True, "camera.emgain": c["emgain"], "camera.pixelsize_um": c["pixelsize_um"][0],
        "camera.pixelsize_y_um": c["pixelsize_um"][1], "model.calibration": str(OUT / "beads_pupil.h5"),
        "model.focus": "calibrated", "fit.roisize": 13, "output.path": str(out), "output.raw_frames": 0})
    return plugins.get("Localize/Spline 3D")().run(Context(), s).locs


def fit_decode(path, out):
    import tifffile
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.io.hdf5 import save_localizations
    from smappy.plugins import Context
    c = MOVIE_CAMERA
    s = fit_settings("Localize/DECODE", {
        "source.path": str(path), "camera.conversion": c["conversion"], "camera.offset": c["offset"],
        "camera.em_on": True, "camera.emgain": c["emgain"], "camera.pixelsize_um": c["pixelsize_um"][0],
        "camera.pixelsize_y_um": c["pixelsize_um"][1], "model.path": str(OUT / "decode_B.safetensors"),
        "model.threshold": 0.1, "output.path": "", "output.raw_frames": 0})
    locs = plugins.get("Localize/DECODE")().run(Context(), s).locs
    with tifffile.TiffFile(path) as t:
        width = t.pages[0].shape[1]
    x_px = np.asarray(locs["x_nm"], np.float64) / PIXEL_NM[0]
    locs.columns["x_nm"] = ((width - 1 - x_px) * PIXEL_NM[0]).astype(np.float32)
    save_localizations(out, locs, {})
    return locs


def table(locs, ok, frames=None):
    f = np.asarray(locs["frame"])[ok]
    out = {k: np.asarray(locs[k], np.float64)[ok] for k in ("x_nm", "y_nm", "z_nm")}
    out["frame"] = f
    if frames is not None:
        sel = np.isin(f, frames)
        out = {k: v[sel] for k, v in out.items()}
    return out


def score(reference, found):
    from scipy.optimize import linear_sum_assignment
    tp = n_ref = n_found = 0
    for t in range(N_SUMS):
        r = {k: v[reference["frame"] == t] for k, v in reference.items()}
        s = {k: v[found["frame"] == t] for k, v in found.items()}
        n_ref += len(r["x_nm"])
        n_found += len(s["x_nm"])
        if not len(r["x_nm"]) or not len(s["x_nm"]):
            continue
        d = np.hypot(r["x_nm"][:, None] - s["x_nm"][None], r["y_nm"][:, None] - s["y_nm"][None])
        far = (d > LATERAL_NM) | (np.abs(r["z_nm"][:, None] - s["z_nm"][None]) > AXIAL_NM)
        cost = np.where(far, 1e9, d)
        i, j = linear_sum_assignment(cost)
        tp += int((cost[i, j] < 1e9).sum())
    return {"reference": n_ref, "found": n_found, "matched": tp, "recall": tp / max(n_ref, 1),
            "precision": tp / max(n_found, 1), "jaccard": tp / max(n_ref + n_found - tp, 1)}


def reference_of(name):
    """The fitter's own localizations in frames t and t + SHIFT, t < N_SUMS, as frames of the sums."""
    from smappy.io.hdf5 import load_localizations
    locs = load_localizations(OUT / f"{name}.hdf5")
    ok = keep(locs)
    first = table(locs, ok, np.arange(N_SUMS))
    second = table(locs, ok, np.arange(SHIFT, SHIFT + N_SUMS))
    second["frame"] = second["frame"] - SHIFT
    single = table(locs, ok, np.arange(N_SUMS))
    return {k: np.concatenate([first[k], second[k]]) for k in first}, single


def main():
    raw, flipped = OUT / "summed_raw.tif", OUT / "summed_flipped.tif"
    if not raw.exists():
        summed(MOVIE, raw)
        summed(FLIPPED, flipped)
    area_px = None
    results = []
    for name, fit, path in (("spline_B", fit_spline, flipped), ("decode_B", fit_decode, raw)):
        out = OUT / f"summed_{name}.hdf5"
        if out.exists():
            from smappy.io.hdf5 import load_localizations
            locs = load_localizations(out)
        else:
            locs = fit(path, out)
        if area_px is None:
            import tifffile
            with tifffile.TiffFile(path) as t:
                area_px = t.pages[0].shape[0] * t.pages[0].shape[1]
        reference, single = reference_of(name)
        found = table(locs, keep(locs))
        per_40 = 1600.0 / area_px / N_SUMS
        r = {"name": name, "density_single": len(single["x_nm"]) * per_40,
             "density_sum": len(reference["x_nm"]) * per_40, **score(reference, found)}
        results.append(r)
        print(f"{name:9s} the frames alone: {r['density_single']:.2f} a 40 x 40 px, summed {r['density_sum']:.2f};  "
              f"in the sum recall {r['recall']:.3f}, precision {r['precision']:.3f}, Jaccard {r['jaccard']:.3f}")
    with open(OUT / "summed_frames.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
