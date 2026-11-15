"""Is the coverslip tilted?  Its edge measured per tile of the field, a plane through the tiles, no pupil tilt.

    python tilt_from_edge.py OUTDIR MODEL [MODEL ...] [--tiles 3] [--stack STACK_MODEL] [--json FILE]

The in situ runs on M2's nucleus movie learn a coverslip tilt of 1.6° about y (2.9° with every aberration of
the stack held), which takes most of their excess astigmatism, coma and spherical over the molecule stack
(architecture.md, "In situ against molecule stacks on M2").  That is a tilt of the interfaces in the *pupil*.
A real tilt would also raise the coverslip across the field: the field is ~26 µm wide, so 1.6° is ~28 nm per µm,
~700 nm across.  The molecules' heights show that without any pupil: the lowest molecules of each tile lie on the
coverslip, and their edge (`coverslip_edge`, the rounds' own fit) moves with it.

Per model (``heights_<MODEL>.npz`` from `heights.py`): the field cut into ``tiles`` × ``tiles``, each tile's edge
and its error, then a weighted plane through the tiles, edge = a + b_x·u + b_y·v (u, v in µm from the field's
centre).  Reported: b_x, b_y in nm per µm with their errors (scaled by the fit's χ² where it exceeds 1), the
slope's size as an angle, and each tile's edge.  A model without a learnt tilt should be used (the edge then
carries the whole tilt); the edge's slope barely depends on the pupil, since it is the molecules' positions.

``--stack``: the molecule stack's own focal-plane slope (``geometry.plane``, nm per µm, `CoverslipStackGeometry`)
printed beside it -- the coverslip's tilt as the z sweeps saw it, two hours later.

What to read: a slope of ~28-50 nm per µm along x would say the tilt is real; a slope within a few nm per µm
of 0 says the learnt tilt stands in for something else (the nucleus' index, focus drift, the pores' own
orientation).  A tile without molecules on the glass (its spike fraction near 0) gives the lower edge of the
sample instead: inside a thick nucleus that is not the coverslip, so look at the spike column.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heights import PIXEL_NM, coverslip_edge, load  # noqa: E402

MIN_PER_TILE = 60


def tile_edges(x_px, y_px, h, sd, tiles: int, pixel_nm=PIXEL_NM):
    """Each tile's (u, v) centre in µm from the field's centre, its edge, error, spike fraction and count."""
    x0, x1 = np.nanmin(x_px), np.nanmax(x_px) + 1e-9
    y0, y1 = np.nanmin(y_px), np.nanmax(y_px) + 1e-9
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ix = np.minimum(((x_px - x0) / (x1 - x0) * tiles).astype(int), tiles - 1)
    iy = np.minimum(((y_px - y0) / (y1 - y0) * tiles).astype(int), tiles - 1)
    rows = []
    for j in range(tiles):
        for i in range(tiles):
            sel = (ix == i) & (iy == j)
            if sel.sum() < MIN_PER_TILE:
                continue
            e = coverslip_edge(h[sel], sd[sel])
            if e is None or not np.isfinite(e.sd_nm):
                continue
            u = (np.mean(x_px[sel]) - cx) * pixel_nm[0] / 1000
            v = (np.mean(y_px[sel]) - cy) * pixel_nm[1] / 1000
            rows.append(dict(tile=(i, j), u_um=float(u), v_um=float(v), edge_nm=e.shift_nm, sd_nm=e.sd_nm,
                             spike=e.spike, n=int(sel.sum())))
    return rows


def plane(rows):
    """Weighted least squares edge = a + b_x u + b_y v: (a, b_x, b_y), their errors and χ² per degree of freedom."""
    u = np.array([r["u_um"] for r in rows])
    v = np.array([r["v_um"] for r in rows])
    z = np.array([r["edge_nm"] for r in rows])
    w = 1 / np.maximum(np.array([r["sd_nm"] for r in rows]), 1.0) ** 2
    A = np.c_[np.ones_like(u), u, v]
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * z))
    dof = max(len(z) - 3, 1)
    chi2 = float(np.sum(w * (z - A @ coef) ** 2) / dof)
    err = np.sqrt(np.diag(cov) * max(chi2, 1.0))
    return coef, err, chi2


def angle_deg(bx, by) -> float:
    return math.degrees(math.atan(math.hypot(bx, by) / 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--stack", help="a molecule stack model in OUTDIR: its geometry.plane printed beside")
    ap.add_argument("--json")
    a = ap.parse_args()
    print(f"a tilt of 1.6° is {1000 * math.tan(math.radians(1.6)):.0f} nm per µm, 2.9° is "
          f"{1000 * math.tan(math.radians(2.9)):.0f}")
    if a.stack:
        import torch
        from smappy.calibrate.pupil import load_start
        m = load_start(os.path.join(a.out, f"{a.stack}_3dcal.h5"))
        p = torch.as_tensor(m.shared["geometry.plane"]).detach().cpu().double().reshape(-1).tolist()   # MPS: no float64
        print(f"stack {a.stack}: focal plane's slope {p[0]:+.1f}, {p[1]:+.1f} nm per µm (x, y), "
              f"{angle_deg(*p):.2f}°")
    for name in a.names:
        d = load(a.out, name)
        rows = tile_edges(d["x_px"], d["y_px"], d["h"], d["sd_h"], a.tiles)
        if len(rows) < 4:
            print(f"{name}: only {len(rows)} tiles with an edge; fewer --tiles")
            continue
        coef, err, chi2 = plane(rows)
        print(f"\n{name}: {len(rows)} tiles of {a.tiles}x{a.tiles}; edge's slope b_x {coef[1]:+.1f} ± {err[1]:.1f}, "
              f"b_y {coef[2]:+.1f} ± {err[2]:.1f} nm per µm -> {angle_deg(coef[1], coef[2]):.2f}°; "
              f"χ²/dof {chi2:.1f}")
        for r in rows:
            print(f"   tile {r['tile']}: u {r['u_um']:+6.1f} v {r['v_um']:+6.1f} µm  edge {r['edge_nm']:+7.1f} ± "
                  f"{r['sd_nm']:5.1f} nm  spike {r['spike']:.2f}  n {r['n']}")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(dict(kind="tilt_from_edge", model=name, tiles=a.tiles,
                                        b_x_nm_per_um=float(coef[1]), b_y_nm_per_um=float(coef[2]),
                                        b_x_err=float(err[1]), b_y_err=float(err[2]), chi2_dof=chi2,
                                        angle_deg=angle_deg(coef[1], coef[2]), tile_rows=rows)) + "\n")


if __name__ == "__main__":
    main()
