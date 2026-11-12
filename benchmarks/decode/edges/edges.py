"""The frame's edge pixels: DECODE's crop against padding and against windows set on the edges.

    python benchmarks/decode/edges/edges.py [--densities 5 15] [--device mps]

The U-Net takes frames of a multiple of 8 pixels, so the predictor cuts the frames as DECODE's
``AutoCenterCrop(8)`` does and loses up to 7 rows and 7 columns at the edges (plan §6, item 17).
This compares ways of keeping them, on (i)'s simulated sets of calibration B with the port's model
trained on B (``decode_B``), the sets' 64 x 64 frames cut to 59 x 61 (rows 2 -- 60, columns 1 -- 61)
so that the crop loses rows 0, 1 and 58 and columns 0, 1, 2 and 59 -- the truth kept whole, its
emitters outside the cut drawn into its edges as a real frame's are:

* ``crop``: the predictor as it is;
* ``pad <v>``: the frames padded to the next multiple of 8, the same on either side as the crop
  would cut, with the value whose network input is ``v`` (``(photons - 110) / 140`` for this
  model's scaling): ``v = 0`` is what the network's own convolutions pad with at every layer
  (110 photons, a typical background), ``-0.79`` is 0 photons, ``-5`` and ``+5`` values no frame
  holds; ``edge`` repeats the edge pixels, ``reflect`` mirrors them; detections in the padding
  dropped;
* ``windows``: no padding -- the centre crop as now, and for the pixels it loses windows of 32
  pixels set on the frame's edges (a strip along each side, a square in each corner), each a
  multiple of 8 and with its outer edge on the data's edge, which is how the network saw an edge in
  its training; each detection taken from the one window whose part of the frame it lies in, so
  the inside is the crop's exactly.  It costs a quarter more network time on 512 x 512 frames.

Scored as (i) (`score_sets`: prob >= 0.5, matched within 250 nm and 500 nm in z) by the distance of
the truth (recall, lateral RMSE) or of the detection (precision) from the nearest edge of the cut
frame, in pixels.  The truth is the whole 64 x 64 frame's: an emitter outside the cut, whose spot
reaches into it, counts for precision when a detection is matched to it and is not asked for in
recall (no bin holds it).  ``windows <m>`` takes the centre crop's own ``m`` pixels along its cut
edges from the windows too: the crop's edge pulls in the spots beyond it.  ``same inside`` is the fraction of the crop's detections at least 8 pixels from
an edge that another method found at the same place (within 0.01 px): what padding does to the
inside.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmark"))
sys.path.insert(0, str(HERE.parent / "pixelation"))
import score_sets  # noqa: E402
from common import DATA, OUT  # noqa: E402

ROOT = DATA / "benchmark" / "B"
MODEL = OUT / "decode_B.safetensors"
CUT = (slice(2, 61), slice(1, 62))           # 59 x 61
FOLD = 8
STRIP = 32
BINS = (0, 1, 2, 3, 4, 8, 16, 99)
SAME_PX = 0.01
METHODS = ("crop", "pad 0", "pad -0.79", "pad -5", "pad 5", "edge", "reflect", "edge aligned", "reflect aligned",
           "pad 0 aligned", "windows", "windows 2", "windows 4", "uncut")


def photons_of(frames, meta):
    c = meta["camera"]
    return ((frames.astype(np.float32) - c["offset"]) * c["conversion"] / c["em_gain"]).astype(np.float32)


def run(predictor, photons):
    """Predictor columns (frame, x_pix, y_pix, ...) of frames that need no context (a 1-frame model)."""
    assert predictor.window == 0
    return predictor(photons, 0)


def shifted(cols, rows=0, columns=0):
    out = dict(cols)
    out["x_pix"] = cols["x_pix"] + columns
    out["y_pix"] = cols["y_pix"] + rows
    return out


def inside(cols, r0, r1, c0, c1):
    """Rows of ``cols`` with their position in rows [r0, r1) and columns [c0, c1) of pixel centres."""
    y, x = cols["y_pix"], cols["x_pix"]
    ok = (y >= r0 - 0.5) & (y < r1 - 0.5) & (x >= c0 - 0.5) & (x < c1 - 0.5)
    return {k: v[ok] for k, v in cols.items()}


def joined(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def padded(predictor, photons, how):
    from decode_smlm.io import crop_of
    n, h, w = photons.shape
    r0, c0, th, tw = crop_of(photons.shape)
    aligned = how.endswith(" aligned")
    how = how.removesuffix(" aligned")
    if aligned:
        # the crop's first row and column where they are in the crop, on the network's grid of 8:
        # the inside then sees the crop's grid exactly, and differs from it only within reach of
        # the edges
        top, left = (FOLD - r0) % FOLD, (FOLD - c0) % FOLD
        ph, pw = top + (-(h + top)) % FOLD, left + (-(w + left)) % FOLD
    else:
        # the next multiple of 8, the larger part on the side where the crop cut the larger part
        ph, pw = (-h) % FOLD, (-w) % FOLD
        top, left = ph - ph // 2, pw - pw // 2
    pad = ((0, 0), (top, ph - top), (left, pw - left))
    if how in ("edge", "reflect"):
        x = np.pad(photons, pad, mode=how)
    else:
        v = float(how.split()[1])           # "nan" too: float("nan")
        s = predictor.metadata["scaling"]
        cam = predictor.metadata.get("camera") or {}
        value = (v * s["input_scale"] + s["input_offset"]) * float(cam.get("qe") or 1) + float(cam.get("spur_noise") or 0)
        x = np.pad(photons, pad, mode="constant", constant_values=value)
    cols = shifted(run(predictor, x), -top, -left)
    return inside(cols, 0, h, 0, w)


def windows(predictor, photons, margin=0):
    """The centre crop, and for what it cuts off, windows on the edges; the centre crop's own
    ``margin`` pixels along its cut edges taken from the windows too."""
    from decode_smlm.io import crop_of
    n, h, w = photons.shape
    r0, c0, th, tw = crop_of(photons.shape)
    r1, c1 = r0 + th, c0 + tw
    # a cut edge moves in by the margin; an edge on the data's own edge (nothing cut) stays
    r0, c0 = (r0 + margin if r0 else 0), (c0 + margin if c0 else 0)
    r1, c1 = (r1 - margin if r1 < h else h), (c1 - margin if c1 < w else w)
    s = min(STRIP, th, tw)
    parts = [inside(run(predictor, photons), r0, r1, c0, c1)]
    assert max(r0, c0, h - r1, w - c1) <= s - 16, "the windows' inner edges come too close"
    # (rows of the frame this window owns, columns it owns, its own rows, its own columns)
    bands_r = [((0, r0), (0, s)), ((r0, r1), (r0, r1)), ((r1, h), (h - s, h))]
    bands_c = [((0, c0), (0, s)), ((c0, c1), (c0, c1)), ((c1, w), (w - s, w))]
    for i, (own_r, win_r) in enumerate(bands_r):
        for j, (own_c, win_c) in enumerate(bands_c):
            if (i, j) == (1, 1) or own_r[0] == own_r[1] or own_c[0] == own_c[1]:
                continue
            sub = photons[:, win_r[0]:win_r[1], win_c[0]:win_c[1]]
            cols = shifted(run(predictor, sub), win_r[0], win_c[0])
            parts.append(inside(cols, own_r[0], own_r[1], own_c[0], own_c[1]))
    return joined(parts)


def edge_distance(x, y, h, w):
    """Pixels from the nearest edge of the frame (its edge half a pixel beyond the outer centres)."""
    return np.minimum.reduce([x + 0.5, w - 0.5 - x, y + 0.5, h - 0.5 - y])


def score(t, cols, h, w, px):
    ok = cols["prob"] >= score_sets.MIN_PROB
    f = {"frame": cols["frame"][ok], "x_px": cols["x_pix"][ok], "y_px": cols["y_pix"][ok], "z_nm": cols["z_nm"][ok]}
    ti, fi = score_sets.match(t, f, px)
    dt = edge_distance(t["x_px"], t["y_px"], h, w)          # < 0 outside the frame
    df = edge_distance(f["x_px"], f["y_px"], h, w)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        in_t, in_f = (dt >= lo) & (dt < hi), (df >= lo) & (df < hi)
        m = in_t[ti]
        err = np.hypot((f["x_px"][fi[m]] - t["x_px"][ti[m]]) * px[0], (f["y_px"][fi[m]] - t["y_px"][ti[m]]) * px[1])
        found = np.zeros(len(df), bool)
        found[fi] = True
        rows.append({"lo": lo, "hi": hi, "truth": int(in_t.sum()), "recall": float(m.sum() / max(in_t.sum(), 1)),
                     "precision": float(found[in_f].sum() / max(in_f.sum(), 1)),
                     "rmse_lat_nm": float(np.sqrt(np.mean(err ** 2))) if m.any() else float("nan")})
    return rows, f


def same_inside(ref, other, h, w):
    """Of ref's detections >= 8 px from an edge, the fraction other has at the same place; the
    median distance to other's nearest, in pixels."""
    from scipy.spatial import cKDTree
    keep = edge_distance(ref["x_px"], ref["y_px"], h, w) >= 8
    hit, dist = 0, []
    for fr in np.unique(ref["frame"][keep]):
        a = keep & (ref["frame"] == fr)
        b = other["frame"] == fr
        if not b.any():
            continue
        d, _ = cKDTree(np.c_[other["x_px"][b], other["y_px"][b]]).query(np.c_[ref["x_px"][a], ref["y_px"][a]])
        hit += int((d <= SAME_PX).sum())
        dist.append(d)
    return hit / max(int(keep.sum()), 1), float(np.median(np.concatenate(dist))) if dist else float("inf")


def main():
    import tifffile
    from decode_smlm.predict import load_predictor
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", nargs="*", default=["5", "15"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="these only (crop is always run)")
    a = ap.parse_args()
    meta = json.loads((ROOT / "sets.json").read_text())
    px = tuple(meta["pixel_nm"])
    predictor = load_predictor(MODEL, a.device, frames_mirrored=False, raw_th=0.1)
    results = []
    for d in a.densities:
        folder = ROOT / f"density_{d}"
        frames = photons_of(tifffile.imread(folder / "frames.tif")[:, CUT[0], CUT[1]], meta)
        n, h, w = frames.shape
        tz = np.load(folder / "truth.npz")
        t = {"frame": tz["frame"], "x_px": tz["x_px"] - CUT[1].start, "y_px": tz["y_px"] - CUT[0].start, "z_nm": tz["z_nm"]}
        fits = {}
        for method in ("crop", *[m for m in (a.methods or METHODS) if m != "crop"]):
            if method == "crop":
                cols = run(predictor, frames)
            elif method == "uncut":
                # the oracle: the network on the whole 64 x 64 frame, which holds what lies beyond the cut
                whole = photons_of(tifffile.imread(folder / "frames.tif"), meta)
                cols = inside(shifted(run(predictor, whole), -CUT[0].start, -CUT[1].start), 0, h, 0, w)
            elif method.startswith("windows"):
                cols = windows(predictor, frames, int(method.split()[1]) if " " in method else 0)
            else:
                cols = padded(predictor, frames, method)
            rows, fits[method] = score(t, cols, h, w, px)
            same, moved = same_inside(fits["crop"], fits[method], h, w)
            results.append({"density": d, "method": method, "same_inside": same, "moved_px": moved, "bins": rows})
            print(f"density {d:>2} {method:15s} same inside {same:.3f}, moved {moved:.4f} px  " + "  ".join(
                f"[{r['lo']}-{r['hi']}) rec {r['recall']:.2f} prec {r['precision']:.2f} {r['rmse_lat_nm']:4.0f}"
                for r in rows), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / ("edges.jsonl" if not a.methods else "edges_more.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
