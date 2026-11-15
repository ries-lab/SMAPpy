"""How flat and level is a glass slide on M2?  The AB molecule stack's molecules, each with its own z free.

    python stack_flatness.py OUTDIR STACK_MODEL [--tiles 4] [--near 400] [--json FILE]

A reference for `tilt_from_edge.py`: the nucleus movie's coverslip edge falls 23-29 nm per µm across the few
tiles that have molecules on the glass, and the in situ pupils learn a tilt of 1.6-2.9°.  The AB sweeps were
taken on a glass slide the same evening, molecules over the whole field, so the glass's own variation can be
read without a pupil question: STACK_MODEL's learnt geometry (each sweep's focus, the focal plane's slope
``geometry.plane``) and its pupil held, each spot localised again with its own offset δz free
(`CoverslipStackGeometry.checking`, the z check's model), the z bend (the median δz per 100 nm of defocus)
taken out, then:

- the learnt plane: the slope of the focal plane against the glass over the field (nm per µm, its angle);
- the residual δz per tile of the field and a plane through the tiles: what the learnt plane leaves;
- the rms of the tiles' residuals about that plane: the glass's unevenness over a few µm;
- each sweep's learnt focus: how much the focus moved over the 10 sweeps (~10 min).

Only spots within ``--near`` nm of focus are used: there δz is best determined.  Every spot that passes the
selection (significance 5, all of them, no draw) is used; the stack learnt from 3468 of them, which is fine
here: the question is the glass, not the pupil.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
STACK = "261112_MR_Intercalib2_AB_50ms_coverslip/*/*.ome.tif"
BEND_STEP_NM = 100.0


def weighted_plane(u, v, z, sd):
    w = 1 / np.maximum(sd, 1.0) ** 2
    A = np.c_[np.ones_like(u), u, v]
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * z))
    dof = max(len(z) - 3, 1)
    chi2 = float(np.sum(w * (z - A @ coef) ** 2) / dof)
    return coef, np.sqrt(np.diag(cov) * max(chi2, 1.0)), chi2, z - A @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("stack")
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--near", type=float, default=400.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    from smappy.calibrate.molecule_stack import MoleculeStackSettings, molecule_rois
    from smappy.calibrate.pupil import load_start

    from uipsf.model import ForwardModel
    from uipsf.workflows.scoring import localise

    rois = molecule_rois(sorted(glob.glob(os.path.join(ROOT, STACK))),
                         MoleculeStackSettings(molecules=10 ** 6, significance=5.0, height_nm=None))
    ids = rois.ids("used")
    d = np.asarray(rois.defocus_nm, float)[ids]
    ids = ids[np.abs(d) <= a.near]
    d = np.asarray(rois.defocus_nm, float)[ids]
    batch, noise = rois.batch(ids=ids)
    m = load_start(os.path.join(a.out, f"{a.stack}_3dcal.h5"))
    model = m.model
    checking = ForwardModel(model.channels, geometry=model.geometry.checking(), intensity=model.intensity,
                            background=model.background)
    shared = {k: torch.as_tensor(v) for k, v in m.shared.items()}
    fit, _ = localise(checking, shared, batch, noise)
    dz = fit.params.physical()["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    sd = fit.std["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    ok = np.isfinite(dz) & np.isfinite(sd) & (sd > 0) & (np.abs(dz) < np.maximum(3 * sd, 150.0))
    # the z bend: a function of defocus, the pupil's, not the glass's
    bins = np.round(d / BEND_STEP_NM)
    bend = np.zeros_like(dz)
    for b in np.unique(bins):
        k = ok & (bins == b)
        if k.any():
            bend[bins == b] = np.median(dz[k])
    r = dz - bend
    px = np.asarray(rois.pixel_size_nm, float)
    xy = np.asarray(rois.units.centre, float)[ids]
    cx, cy = rois.field_centre()
    u, v = (xy[:, 0] - cx) * px[0] / 1000, (xy[:, 1] - cy) * px[1] / 1000
    plane = torch.as_tensor(shared["geometry.plane"]).detach().cpu().double().reshape(-1).tolist()
    focus = torch.as_tensor(shared["geometry.focus"]).detach().cpu().double().reshape(-1).numpy()
    ang = np.degrees(np.arctan(np.hypot(*plane) / 1000))
    print(f"{ok.sum()} of {len(ids)} spots within {a.near:g} nm of focus; δz CRLB median {np.median(sd[ok]):.1f} nm")
    print(f"learnt focal plane's slope: {plane[0]:+.2f}, {plane[1]:+.2f} nm per µm (x, y), {ang:.3f}°")
    print(f"sweep foci: {np.round(focus, 1).tolist()} nm; range {np.ptp(focus):.1f}, sd {focus.std():.1f}")
    # tiles
    x0, x1, y0, y1 = u[ok].min(), u[ok].max() + 1e-9, v[ok].min(), v[ok].max() + 1e-9
    iu = np.minimum(((u - x0) / (x1 - x0) * a.tiles).astype(int), a.tiles - 1)
    iv = np.minimum(((v - y0) / (y1 - y0) * a.tiles).astype(int), a.tiles - 1)
    tiles = []
    for j in range(a.tiles):
        for i in range(a.tiles):
            k = ok & (iu == i) & (iv == j)
            if k.sum() < 20:
                continue
            med = float(np.median(r[k]))
            err = float(1.2533 * 1.4826 * np.median(np.abs(r[k] - med)) / np.sqrt(k.sum()))
            tiles.append(dict(tile=(i, j), u_um=float(u[k].mean()), v_um=float(v[k].mean()), dz_nm=med,
                              sd_nm=err, n=int(k.sum())))
    T = {k: np.array([t[k] for t in tiles]) for k in ("u_um", "v_um", "dz_nm", "sd_nm")}
    coef, err, chi2, res = weighted_plane(T["u_um"], T["v_um"], T["dz_nm"], T["sd_nm"])
    rms = float(np.sqrt(np.mean(res ** 2)))
    print(f"residual δz per tile ({len(tiles)} of {a.tiles}x{a.tiles}): plane b_x {coef[1]:+.2f} ± {err[1]:.2f}, "
          f"b_y {coef[2]:+.2f} ± {err[2]:.2f} nm per µm; rms about it {rms:.1f} nm; χ²/dof {chi2:.1f}; "
          f"tiles' range {np.ptp(T['dz_nm']):.1f} nm")
    for t in tiles:
        print(f"   tile {t['tile']}: u {t['u_um']:+6.1f} v {t['v_um']:+6.1f} µm  δz {t['dz_nm']:+6.1f} ± "
              f"{t['sd_nm']:4.1f} nm  n {t['n']}")
    total = (plane[0] + coef[1], plane[1] + coef[2])
    print(f"the glass against the focal plane, learnt plane + residual: {total[0]:+.2f}, {total[1]:+.2f} nm per µm, "
          f"{np.degrees(np.arctan(np.hypot(*total) / 1000)):.3f}°")
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(dict(kind="stack_flatness", model=a.stack, tiles=a.tiles, near_nm=a.near,
                                    spots=int(ok.sum()), learnt_plane_nm_per_um=plane, learnt_plane_deg=float(ang),
                                    sweep_focus_nm=focus.tolist(), sweep_focus_range_nm=float(np.ptp(focus)),
                                    residual_plane_nm_per_um=[float(coef[1]), float(coef[2])],
                                    residual_plane_err=[float(err[1]), float(err[2])], residual_rms_nm=rms,
                                    chi2_dof=chi2, tiles_range_nm=float(np.ptp(T["dz_nm"])),
                                    total_slope_nm_per_um=list(total), tile_rows=tiles)) + "\n")


if __name__ == "__main__":
    main()
