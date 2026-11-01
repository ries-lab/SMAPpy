"""A scorecard per learnt model: the few numbers that say how good a PSF model of M5's beads is.

    python benchmarks/uipsf/real/m5_scorecard.py DIR [DIR ...] [--csv OUT.csv]

Reads the per-fit .npz files of `m5_media_psfs.py` (and of `m5_consensus.py --npz-dir`) and prints one row per
fit; whatever a file lacks (older runs kept no per-bead fits, or no deviance) is left empty.  The user's
selection (1 November 2026), in three groups:

**Localisation**, each slice fitted alone against its bead's stack fit (`bead_error`, `bead_z`):
1. *z bend*: rms of the median over beads of the z error, over ±200 and ±600 nm about focus;
2. *x, y bias*: the same in x and y;
3. *z scale*: the median over beads of each bead's slope of z error against z, %, and *bend − line*: the
   bend's rms about its own straight line, what a z scale cannot remove;
4. *drift*: the median over beads of dx/dz and dy/dz, nm per µm of defocus (coma, tilt, non-telecentricity);
5. *field*: the spread over beads of those slopes (1.4826·MAD) and their gradient across the chip, per 100 px;
6. *scatter / CRLB*: each slice's error about its bead's own line in z, over its CRLB (1.4826·MAD) -- 1 if the
   model limits nothing.

**The PSF**, the bead average against the model on the same planes (`data`, `model`):
7. *deviance* per pixel (1: the model is right to the noise) and the objective, nats;
8. *rings*: model minus bead average in the core (r ≤ 1 px from the peak), the ring around it (2-4 px) and the
   wings (> 4 px), mean % of the peak, at z = −300, 0, +300 nm -- the blue core and the red ring as numbers;
9. *orders*: the residual split by azimuthal order about the peak, m = 0 (radial: blur, defocus, core), m = 1
   (a shift, coma) and m = 2 (astigmatism), rms % of the peak over the five planes ±600 nm;
10. *significance*: the averaged residual's χ² per pixel against its standard error over beads (`resid_se`), 1
    when it is noise.

**The model**: σ, the z scale and NA where learnt, the immersion index, the beads used.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from m5_bends import bead_slopes  # noqa: E402

PLANES = (-600.0, -300.0, 0.0, 300.0, 600.0)


def _mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def localisation(z, err, crlb, x_px, y_px) -> dict:
    """Items 1-6 from every bead's slice fits."""
    out = {}
    e = err - np.nanmedian(np.where(np.abs(z)[..., None] <= 600, err, np.nan), axis=1, keepdims=True)
    for lim in (200, 600):
        grid = np.arange(-lim, lim + 1, 100.0)
        med = np.array([[np.nanmedian(e[..., a][np.abs(z - g) <= 50]) if np.any(np.abs(z - g) <= 50) else np.nan
                         for g in grid] for a in range(3)])
        for a, name in enumerate("xyz"):
            out[f"{name}_bias_{lim}"] = float(np.sqrt(np.nanmean(med[a] ** 2)))
        if lim == 600:
            ok = np.isfinite(med[2])
            line = np.polyval(np.polyfit(grid[ok], med[2][ok], 1), grid[ok])
            out["bend_minus_line_600"] = float(np.sqrt(np.mean((med[2][ok] - line) ** 2)))
    s = bead_slopes(z, err)
    out.update(z_scale_pct=float(np.nanmedian(s["scale"])), dxdz=float(np.nanmedian(s["dx"])),
               dydz=float(np.nanmedian(s["dy"])), spread_dxdz=_mad(s["dx"]), spread_dydz=_mad(s["dy"]),
               spread_scale_pct=_mad(s["scale"]))
    ok = np.isfinite(s["dx"]) & np.isfinite(s["dy"])
    if ok.sum() >= 5:
        A = np.stack([np.ones(ok.sum()), (x_px[ok] - x_px[ok].mean()) / 100, (y_px[ok] - y_px[ok].mean()) / 100], 1)
        gx, *_ = np.linalg.lstsq(A, s["dx"][ok], rcond=None)
        gy, *_ = np.linalg.lstsq(A, s["dy"][ok], rcond=None)
        out.update(field_dxdz_per_x=float(gx[1]), field_dydz_per_y=float(gy[2]))
    # each slice about its bead's own line in z, over its CRLB
    near = np.abs(z) <= 600
    for a, name in enumerate("xyz"):
        r = []
        for b in range(len(z)):
            k = near[b] & np.isfinite(err[b, :, a]) & np.isfinite(crlb[b, :, a]) & (crlb[b, :, a] > 0)
            if k.sum() >= 4:
                fit = np.polyval(np.polyfit(z[b][k], err[b, k, a], 1), z[b][k])
                r.extend((err[b, k, a] - fit) / crlb[b, k, a])
        out[f"{name}_scatter_crlb"] = _mad(r)
    return out


def psf(z_nm, data, model, se=None) -> dict:
    """Items 8-10 from the bead average and the model on the same planes."""
    out = {}
    Y, X = data.shape[-2:]
    yy, xx = np.mgrid[:Y, :X]
    orders = {0: [], 1: [], 2: []}
    chi = []
    for p in PLANES:
        k = int(np.argmin(np.abs(z_nm - p)))
        d, m = data[k], model[k]
        top = float(np.nanmax(d))
        cy, cx = np.unravel_index(np.nanargmax(d), d.shape)
        r = np.hypot(yy - cy, xx - cx)
        phi = np.arctan2(yy - cy, xx - cx)
        res = (m - d) / top * 100
        if p in (-300.0, 0.0, 300.0):
            tag = f"{int(p):+d}"
            out[f"core_{tag}"] = float(np.nanmean(res[r <= 1.0]))
            out[f"ring_{tag}"] = float(np.nanmean(res[(r > 1.5) & (r <= 4.5)]))
            out[f"wings_{tag}"] = float(np.nanmean(res[r > 4.5]))
        # azimuthal orders in rings one pixel wide: the m-th Fourier part of each ring
        ok = np.isfinite(res)
        for mm in orders:
            power = 0.0
            for rr in range(int(r.max()) + 1):
                ring = ok & (np.round(r) == rr)
                if ring.sum() < 3:
                    continue
                if mm == 0:
                    c = np.mean(res[ring])
                    power += ring.sum() * c ** 2
                else:
                    a = 2 * np.mean(res[ring] * np.cos(mm * phi[ring]))
                    b = 2 * np.mean(res[ring] * np.sin(mm * phi[ring]))
                    power += ring.sum() * (a ** 2 + b ** 2) / 2
            orders[mm].append(power / ok.sum())
        if se is not None:
            s = se[k] / top * 100
            good = ok & np.isfinite(s) & (s > 0)
            chi.append(np.mean((res[good] / s[good]) ** 2))
    for mm, v in orders.items():
        out[f"order{mm}_rms"] = float(np.sqrt(np.mean(v)))
    if chi:
        out["resid_chi2"] = float(np.mean(chi))
    return out


def score(path: str) -> dict:
    d = np.load(path)
    name = os.path.basename(path)[:-4]
    acq, _, variant = name.partition("__")
    row = dict(acquisition=acq, variant=variant.replace("_", " "), folder=os.path.basename(os.path.dirname(path)),
               objective=float(d["objective"]), used=int(d["used"]) if "used" in d.files else np.nan,
               sigma_nm="/".join(f"{v:.1f}" for v in np.atleast_1d(d["sigma_nm"])) if "sigma_nm" in d.files else "")
    for k in ("atlas_mean_deviance", "z_scale", "effective_na", "n_immersion", "window_nm"):
        if k in d.files:
            row[k] = float(d[k])
    if "bead_z" in d.files:
        row.update(localisation(d["bead_z"], d["bead_error"], d["bead_crlb"], d["bead_x_px"], d["bead_y_px"]))
    row.update(psf(d["z_nm"], d["data"], d["model"], d["resid_se"] if "resid_se" in d.files else None))
    return row


COLUMNS = ["acquisition", "variant", "used", "objective", "atlas_mean_deviance", "sigma_nm", "z_bias_200",
           "z_bias_600", "bend_minus_line_600", "z_scale_pct", "x_bias_600", "y_bias_600", "dxdz", "dydz",
           "spread_dxdz", "spread_dydz", "field_dxdz_per_x", "field_dydz_per_y", "z_scatter_crlb", "core_-300",
           "core_+0", "core_+300", "ring_-300", "wings_-300", "order0_rms", "order1_rms", "order2_rms", "resid_chi2",
           "z_scale", "effective_na", "n_immersion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = [score(f) for folder in args.folders for f in sorted(glob.glob(os.path.join(folder, "*.npz")))]
    # Δ nats against the best fit of the same acquisition and folder
    for r in rows:
        same = [q["objective"] for q in rows if q["acquisition"] == r["acquisition"] and q["folder"] == r["folder"]]
        r["delta_nats"] = r["objective"] - min(same)
    cols = [c for c in COLUMNS if any(c in r for r in rows)] + ["delta_nats"]
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["folder"] + cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(" | ".join(f"{c}={r[c]:.2f}" if isinstance(r.get(c), float) else f"{c}={r.get(c, '')}"
                         for c in ["acquisition", "variant"] + cols[2:] if c in r))


if __name__ == "__main__":
    main()
