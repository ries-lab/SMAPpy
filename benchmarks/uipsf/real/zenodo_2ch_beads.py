"""Two channels on Zenodo `2ch_40nm_bead`, against uiPSF v1 and SMAPpy's dual calibration (plan §9 M4).

    python benchmarks/uipsf/real/zenodo_2ch_beads.py [--out benchmarks/uipsf/results/zenodo_2ch_beads.jsonl] \\
        [--device mps]

The 14 stacks (41 slices of 512 × 236, the chip split up–down, the secondary half mirrored): read with
the camera SMAPpy's database names (M2's Evolve 512, normal port: 4.7 e⁻/ADU, offset 398.6) and 21 e⁻ of
read noise (`zenodo_camera.py`), 116 × 127 nm pixels in the camera's frame (docs/uipsf/data.md), v1's optics
(NA 1.43, 680 nm in both channels, n_med 1.35), a 20 nm bead radius, n_max 8, 25 × 25 ROIs.  Pairs by
SMAPpy (`calibrate.pupil.dual_rois`), `learn_beads` with its defaults.

Then:
- **the transform** against SMAPpy's (`calibrate.dual.calibrate_dual` on the same stacks) and v1's
  (``res/T``, converted to chip pixels through v1's frame, `v1_to_chip`; checked on v1's own ROI
  centres):
  where each puts the partner of every used bead, and each one's residuals on SMAPpy's bead positions;
- **the Zernikes per channel** against v1's, mapped into v1's frame (`v1_frame`);
- **the photon split** against SMAPpy's ``photon_normalization`` (the shares its dual calibration
  stores), its brightness ratio and v1's per-channel intensities.

Prints JSON lines; the fitted parameters are saved next to ``--out`` (never committed).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import h5py
import numpy as np
import torch
from smappy.calibrate.dual import DualColorSettings, fit_projective
from smappy.calibrate.pupil import dual_rois
from zenodo_camera import CAMERA, read

import uipsf
from uipsf.diagnostics.camera import noise_transfer
from uipsf.model import Channel, ForwardModel
from uipsf.optics.zernike import zernike_basis
from uipsf.params import Params
from uipsf.solve import Problem
from uipsf.workflows import learn_beads

PIXEL_NM = (116.0, 127.0)
V1_FILE = os.path.join("results", "2ch_40nm_bead", "psfmodel_zernike_vector_multi.h5")


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def apply(H, xy):
    q = np.c_[xy, np.ones(len(xy))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:]


def v1_to_chip(channel, rc, origin, height=512):
    """v1's (row, column) in its channel ``channel``'s array -> chip (x, y).  Found from v1's 74 ROI centres
    (``rois/cor``), every one on a bead uipsf detects: rows are the camera's x (``swapxy``); v1's channel 0,
    its reference, is the camera's *lower* half with y reversed (column = 511 − y, ``mirrortype``), its
    channel 1 the upper half (column = y).  So v1's reference is SMAPpy's secondary channel."""
    rc = np.asarray(rc, float)
    y = height - 1 - rc[:, 1] if channel == 0 else rc[:, 1]
    return np.c_[rc[:, 0] + origin[0], y + origin[1]]


def v1_transform(f, origin) -> np.ndarray:
    """v1's ``T`` as a 3×3 secondary -> main chip map.  v1 maps its reference channel's (row, column),
    relative to ``imgcenter``, to the other's: other = (ref − c)·T + c (row vectors,
    `PSFMultiChannel.calc_positions_from_trafos`); its reference is SMAPpy's secondary (`v1_to_chip`)."""
    T, c = f["res/T"][()].reshape(-1, 3, 3)[0], f["res/imgcenter"][()].ravel()
    rr, cc = np.meshgrid(np.linspace(0, 235, 9), np.linspace(0, 255, 9), indexing="ij")
    ref = np.c_[rr.ravel(), cc.ravel()]
    other = (np.c_[ref, np.ones(len(ref))] - c) @ T
    other = other[:, :2] + c[:2]
    h = fit_projective(v1_to_chip(0, ref, origin), v1_to_chip(1, other, origin))
    return h / h[2, 2]


def v1_frame(grid, modes, channel):
    """[M, M]: our coefficients of ``channel`` (camera frame) -> v1's (its frame).  v1's pupil is ours with
    u and v swapped (``swapxy``), and for the secondary channel (v1's channel 0) v1's columns run against
    the camera's y: W_v1(u', v') = W(v', u'), and W(v', −u') for the secondary.  Found numerically on the
    pupil grid: the transformed basis T_i = Σ_j M_ij Z_j (exact, a signed permutation), so Σ c_i T_i has
    the coefficients Mᵀc."""
    Z = zernike_basis(grid, modes).cpu().double()               # [M, P, P], rows v, columns u (MPS: no float64)
    T = Z.transpose(-1, -2)                                                   # u ↔ v
    if channel == 1:
        T = T.flip(-1)                                                        # v1's u' = −(camera v)
    inside = grid.aperture.cpu() > 0
    A, B = Z[:, inside], T[:, inside]
    M = torch.linalg.lstsq(A.T, B.T).solution.T.numpy()                      # T_i = Σ_j M_ij Z_j
    return M.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--from-saved", default=None, help="a saved fit (*_params.pt): only the comparisons")
    args = ap.parse_args()
    t0 = time.perf_counter()
    files = sorted(glob.glob(os.path.join(args.data, "2ch_40nm_bead", "*", "*.ome.tif")))
    stacks = read(files)
    rois = dual_rois(stacks, CAMERA, DualColorSettings(roi_size=25))
    batch, start = rois.batch.to(args.device), rois.transform / rois.transform[2, 2]
    channels = []
    for c in range(2):
        optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.35, pixel_size_nm=PIXEL_NM)
        grid = uipsf.PupilGrid.build(optics, 64, device=args.device)
        psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=args.n_max), (25, 25), bead_radius_nm=20.0,
                              sigma_nm=(30.0, 30.0))
        channels.append(Channel(psf, transform=start if c else None))
    model, noise = ForwardModel(channels), rois.noise
    emit(dict(stage="data", stacks=len(stacks), pairs=batch.n_emitters, layout=rois.geometry["layout"],
              split=rois.geometry["split_position"], start_transform=start, s=float(batch.s),
              seconds=round(time.perf_counter() - t0, 1)),
         args.out)
    if args.from_saved:
        saved = torch.load(args.from_saved)
        used_ids = saved["used"]
        params = Params(model.specs(batch.subset(torch.tensor(used_ids))), len(used_ids), device=args.device)
        params.values = saved["params"]
        p, std = params.physical(), saved["std"]
        H = uipsf.model.channels.unpack_transform(p["ch1.transform"].detach().cpu().double()).numpy()
    else:
        cal = learn_beads(model, batch, noise, chunk_size=args.chunk, geometry=rois.geometry,
                          progress=lambda e: print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                                               for k, v in e.items()}), flush=True))
        minutes = round((time.perf_counter() - t0) / 60, 2)
        p, std, used_ids = cal.result.params.physical(), cal.result.std, cal.beads.used
        used = batch.subset(torch.tensor(used_ids))
        mu = Problem(model, noise, used).expected(cal.result.params)
        nt = noise_transfer(used.data.cpu(), mu)
        H = uipsf.model.channels.unpack_transform(p["ch1.transform"].detach().cpu().double()).numpy()
        emit(dict(stage="calibration", minutes=minutes, used=len(used_ids), rejected=cal.beads.ids("rejected"),
                  iterations=cal.result.iterations, stop=cal.result.stop_reason, objective=cal.result.objective,
                  atlas_mean_deviance=cal.atlas.mean_deviance, atlas=cal.atlas.checks, flagged=cal.atlas.flagged(),
                  channel_deviance=dict(zip(map(int, cal.atlas.groups["channel"].keys),
                                            cal.atlas.groups["channel"].deviance.tolist(), strict=True)),
                  transform=H, share=float(p["ch1.share"]), share_crlb=float(std["ch1.share"]),
                  dz_nm=float(p["ch1.dz"]), dz_crlb=float(std["ch1.dz"]),
                  sigma_nm=[p["psf.sigma"], p["ch1.psf.sigma"]],
                  noise_transfer=dict(slope=nt.slope, intercept=nt.intercept)), args.out)
        if args.out:
            torch.save({"params": cal.result.params.values, "used": used_ids, "std": cal.result.std},
                       os.path.splitext(args.out)[0] + "_params.pt")

    # The transform, against SMAPpy's and v1's: where each puts every used bead's partner.
    main_chip = rois.points[used_ids, 0]
    rows = {"uipsf": H}
    try:
        from smappy.calibrate.dual import calibrate_dual
        from smappy.calibrate.input import read_bead_stacks
        res = calibrate_dual([s for f in files for s in read_bead_stacks(f)])
        Hs = res.calibration.transformation / res.calibration.transformation[2, 2]
        rows["smappy"] = Hs
        sm, ss = res.beads.main_points, res.beads.secondary_points
        ok = res.transform_accepted
        brightness = float(res.calibration.parameters["secondary_main_brightness_ratio"])
        normalisation = [float(m.parameters["photon_normalization"])
                         for m in (res.calibration.main, res.calibration.secondary)]
    except ImportError:
        sm = ss = ok = brightness = normalisation = None
    origin = tuple(stacks[0].roi[:2]) if stacks[0].roi is not None else (0, 0)
    with h5py.File(os.path.join(args.data, V1_FILE)) as f:
        rows["v1"] = v1_transform(f, origin)
        cor = f["rois/cor"][()]
        v1_pairs = np.abs(apply(rows["uipsf"], v1_to_chip(0, cor[0], origin)) - v1_to_chip(1, cor[1], origin))
        # in uipsf's channel order: main (v1's channel 1), secondary (v1's 0)
        v1z = [f[f"res/channel{c}/zernike_coeff"][()] for c in (1, 0)]
        v1_intensity = [float(np.median(np.real(f[f"res/channel{c}/intensity"][()]))) for c in (1, 0)]
        v1_sigma = [f[f"res/channel{c}/sigma"][()].tolist() for c in (1, 0)]
    for name, Hx in rows.items():
        partner = apply(np.linalg.inv(Hx), main_chip)
        row = dict(stage="transform", source=name, transform=Hx,
                   vs_uipsf_px=dict(rms=float(np.sqrt(((partner - apply(np.linalg.inv(H), main_chip)) ** 2)
                                                      .sum(1).mean())),
                                    max=float(np.abs(partner - apply(np.linalg.inv(H), main_chip)).max())))
        if sm is not None:
            r = apply(Hx, ss[ok]) - sm[ok]
            row["residual_on_smappy_beads_px"] = dict(rms=float(np.sqrt((r ** 2).sum(1).mean())),
                                                      median=float(np.median(np.linalg.norm(r, axis=1))))
        emit(row, args.out)
    emit(dict(stage="v1 frame check", note="v1's ROI centres through uipsf's transform: whole-pixel centres",
              max_px=float(v1_pairs.max()), median_px=float(np.median(v1_pairs))), args.out)

    # Zernikes per channel, in v1's frame; v1's Noll j at index j − 1, magnitude normalised by its piston.
    for c in range(2):
        pre = "" if c == 0 else "ch1."
        pupil = channels[c].psf.pupil
        modes = pupil.phase_modes
        M = v1_frame(channels[c].psf.grid, modes, c)
        ours = M @ p[pre + "pupil.phase"].detach().cpu().double().numpy()
        crlb = np.sqrt(np.einsum("ij,j->i", M ** 2, std[pre + "pupil.phase"].cpu().double().numpy() ** 2))
        theirs = np.array([v1z[c][1][j - 1] for j in modes])
        emit(dict(stage="zernikes", channel=c, modes=modes, uipsf_in_v1_frame=ours, crlb=crlb, v1=theirs,
                  rms_difference=float(np.sqrt(np.mean((ours - theirs) ** 2))),
                  within_2_crlb=int((np.abs(ours - theirs) <= 2 * crlb).sum()),
                  correlation=float(np.corrcoef(ours, theirs)[0, 1])), args.out)
    share = float(p["ch1.share"])
    emit(dict(stage="photons", uipsf_shares=[1 - share, share], smappy_photon_normalization=normalisation,
              uipsf_ratio=share / (1 - share), smappy_brightness_ratio=brightness,
              v1_intensity_ratio=v1_intensity[1] / v1_intensity[0], v1_sigma=v1_sigma), args.out)


if __name__ == "__main__":
    main()
