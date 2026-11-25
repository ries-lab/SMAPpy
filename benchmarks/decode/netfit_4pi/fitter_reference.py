"""Spline 4Pi on the sets `evaluate.py` scores netfit on, as a reference (docs/4pi.md §30): drawn the same way
(the training's settings, the same densities and on-times, CPU draws, so not the same samples), each molecule of
the middle frame fitted at its true position -- 13-px ROIs in every channel where the registration puts it,
the link carrying the sub-pixel offsets, as Spline 4Pi fits a detected molecule.  Detection is perfect here,
so the numbers are the fitter's best: its error over the CRLB, its z from the phase with the cavity known and
its wrong-fringe share, near focus (|z| < 300 nm) and over all, and near focus for the molecules alone
(no other on in the middle frame within 10 px), the fitter's own limit without neighbours in its ROIs.  Molecules within 6 px of the crop's edge, and
below 100 photons in the frame, are left out; a fit that failed (NaN) is counted, not scored.

    python fitter_reference.py SETTINGS.json CALIBRATION.h5 [--samples 512]
"""
import argparse
from dataclasses import replace

import numpy as np
import torch
from smappy import fourpi
from smappy.netfit import fourpi as NF
from smappy.netfit import train
from smappy.netfit.metrics import robust_width
from smappy.psf import Global4PiPSF

from evaluate import SETS

HALF = 6


def reference(settings, calibration, density, life, samples, seed):
    sim = replace(settings.simulation, density_per_40px=density, lifetime_frames=life, lifetime_max_frames=0.0)
    s = replace(settings, simulation=sim, calibration=calibration)
    drawer = train.Drawer(s, "cpu")
    cal = drawer.cal
    g = torch.Generator("cpu").manual_seed(seed)
    rois, links, truth = [], [], []
    for a in range(0, samples, 64):
        b = drawer(min(64, samples - a), g)
        mid = b.frames[:, :, sim.window // 2].numpy()
        m = b.mask.numpy() & (b.photons.numpy() >= 100)
        f, k = np.nonzero(m)
        xyz, d = b.xyz.numpy()[f, k], b.shift.numpy()[f, k]
        X = np.c_[xyz[:, 0], xyz[:, :1] + d[:, 0::2]]
        Y = np.c_[xyz[:, 1], xyz[:, 1:2] + d[:, 1::2]]
        ox, oy = np.rint(X).astype(int) - HALF, np.rint(Y).astype(int) - HALF
        s_ = sim.frame_px
        ok = (ox.min(1) >= 0) & (oy.min(1) >= 0) & (ox.max(1) + 2 * HALF < s_) & (oy.max(1) + 2 * HALF < s_)
        # alone: no other molecule on in the middle frame within 10 px (channel 0), whatever its brightness
        on = b.mask.numpy()
        alone = np.array([np.sort(np.hypot(*(b.xyz.numpy()[f[i]][on[f[i]], :2] - xyz[i, :2]).T))[1:2].min(initial=99) > 10
                          for i in range(len(f))])
        for i in np.nonzero(ok)[0]:
            rois.append(np.stack([mid[f[i], c, oy[i, c]:oy[i, c] + 2 * HALF + 1, ox[i, c]:ox[i, c] + 2 * HALF + 1]
                                  for c in range(len(X[i]))]))
            xr, yr = X[i] - ox[i], Y[i] - oy[i]
            links.append((xr, yr))
            bgs = b.background.numpy()[f[i]]
            truth.append((xr[0], yr[0], xyz[i, 2], b.phase.numpy()[f[i], k[i]], b.photons.numpy()[f[i], k[i]],
                          d[i], bgs, xyz[i, 0], xyz[i, 1], alone[i]))
    rois = np.asarray(rois, np.float32)
    xr = np.array([v[0] for v in links])
    yr = np.array([v[1] for v in links])
    link = fourpi.link(len(rois), cal.n_channels, dx=xr - xr[:, :1], dy=yr - yr[:, :1])
    model = Global4PiPSF(cal, phase_starts=2)
    v = model.unpack(model.fit(rois, link, iterations=100))
    t = np.array([row[:5] for row in truth], float)
    shift = np.array([row[5] for row in truth])
    bgs = np.array([row[6] for row in truth])
    pos = np.array([row[7:9] for row in truth])
    alone = np.array([row[9] for row in truth], bool)
    bound = NF.crlb(cal, pos[:, 0], pos[:, 1], shift, t[:, 2], t[:, 3], t[:, 4], bgs, bool(s.camera.em_gain))
    px = s.camera.pixel_nm
    dx, dy, dz = (v["x_roi"] - t[:, 0]) * px[0], (v["y_roi"] - t[:, 1]) * px[1], v["z_nm"] - t[:, 2]
    dphi = np.angle(np.exp(1j * (v["phase_rad"] - t[:, 3])))
    k = cal.phase_per_nm
    zp = NF.unwrap_known(v["phase_rad"], v["z_nm"], t[:, 3] - k * t[:, 2], k)
    wrong = np.abs(zp - t[:, 2]) > np.pi / abs(k)
    fine = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) & np.isfinite(dphi)
    out = {"n": len(t), "failed_share": float(np.mean(~fine)), "alone_share": float(np.mean(alone)),
           "alone_failed_share": float(np.mean(~fine[alone]))}
    for name, sel in (("core_", (np.abs(t[:, 2]) < 300) & fine), ("", fine), ("alone_core_", (np.abs(t[:, 2]) < 300) & fine & alone)):
        b = bound[sel]
        out.update({
            name + "rmse_lat_nm": float(np.sqrt(np.mean(dx[sel] ** 2 + dy[sel] ** 2))),
            name + "rmse_ax_nm": float(np.sqrt(np.mean(dz[sel] ** 2))),
            name + "crlb_width_lat": robust_width(np.r_[dx[sel] / (b[:, 0] * px[0]), dy[sel] / (b[:, 1] * px[1])]),
            name + "crlb_width_ax": robust_width(dz[sel] / b[:, 2]),
            name + "rmse_phase_rad": float(np.sqrt(np.mean(dphi[sel] ** 2))),
            name + "crlb_width_phase": robust_width(dphi[sel] / b[:, 3]),
            name + "wrong_fringe_share": float(np.mean(wrong[sel])),
            name + "rmse_z_phase_nm": float(np.sqrt(np.mean((zp - t[:, 2])[sel & ~wrong] ** 2)))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings")
    ap.add_argument("calibration")
    ap.add_argument("--samples", type=int, default=512)
    a = ap.parse_args()
    settings = train.load_settings(a.settings)
    for i, (name, (density, life)) in enumerate(SETS.items()):
        r = reference(settings, a.calibration, density, life, a.samples, 777 + i)
        print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
