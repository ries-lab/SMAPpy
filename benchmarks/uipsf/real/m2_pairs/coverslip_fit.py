"""Control: fluorophores at known heights in the sample (AB's model propagated, n_med 1.345, focus 607 nm, no tilt)
localised with the AB coverslip calibration's spline, as a user would (Spline 3D, best of 5 starts).

    python coverslip_fit.py OUTDIR

The spline describes a molecule on the glass against defocus; the fluorophores sit 0-1200 nm deep, imaged with
the focus 607 nm into the sample.  Prints per height the median fitted z, its spread, the x and y error, and the
slope and the rms about a straight line over 0-1000 nm: what ignoring the depth costs in linearity.
"""
import os
import sys

import numpy as np
import torch

HEIGHTS = np.arange(0.0, 1201.0, 100.0)
REP, PHOTONS, BG = 40, 4570.0, 98.0


def main():
    out = sys.argv[1]
    from smappy.calibrate.pupil import _best_of, load_start
    from smappy.io.calibration import load_spline_calibration
    from smappy.psf import SplinePSF

    from uipsf.simulate import parameters, simulate, template

    t = load_start(os.path.join(out, "insitu_all_n1345_tilt_s315_ab_n1345_3dcal.h5"))
    tm = t.model
    shared = {k: torch.as_tensor(v).detach().clone() for k, v in t.shared.items()}
    shared["geometry.tilt"] = torch.zeros_like(shared["geometry.tilt"])
    rng = np.random.default_rng(0)
    E = len(HEIGHTS) * REP
    z_gt = np.repeat(HEIGHTS, REP)
    px, py = tm.pixel_size_nm
    xy = rng.uniform(-0.5, 0.5, (E, 2)) * np.array([px, py])
    cal = load_spline_calibration(os.path.join(out, "ab_n1345_3dcal.h5"))
    box = min(int(cal.coeff.shape[-1]) - 3, 15) | 1
    roi = (box, box)
    b0 = template(E, torch.zeros(1, dtype=torch.float64), roi, dtype=torch.float32)
    values = dict(shared)
    values.update({"emitter.xyz": torch.tensor(np.c_[xy, z_gt], dtype=torch.float64),
                   "emitter.photons": torch.full((E,), PHOTONS), "emitter.background": torch.full((E,), BG)})
    batch = simulate(tm, parameters(tm, b0, values), b0, seed=1)
    full = batch.data.cpu().numpy().reshape(E, *batch.data.shape[-2:])
    c0 = (full.shape[-1] - box) // 2
    rois = np.ascontiguousarray(full[:, c0:c0 + box, c0:c0 + box], np.float32)
    best = _best_of(lambda z: SplinePSF(cal, z_start_nm=z).fit(rois, iterations=75),
                    cal.z_index_to_nm(np.linspace(1, cal.shape[0] - 2, 5)))
    z = cal.z_index_to_nm(best.theta[:, 4])
    h = box // 2
    x_nm = (best.theta[:, 0] - h) * px
    y_nm = (best.theta[:, 1] - h) * py
    print(f"spline z range {cal.z_index_to_nm(0):.0f} .. {cal.z_index_to_nm(cal.shape[0] - 1):.0f} nm, box {box} px")
    print(" z_gt   z_fit median  sd    x err  y err")
    med = []
    for hh in HEIGHTS:
        k = z_gt == hh
        m = float(np.median(z[k]))
        med.append(m)
        mad = 1.4826 * np.median(np.abs(z[k] - m))
        ex, ey = np.median(x_nm[k] - xy[k, 0]), np.median(y_nm[k] - xy[k, 1])
        print(f" {hh:5.0f}  {m:9.1f}  {mad:5.1f}  {ex:6.1f} {ey:6.1f}")
    med = np.array(med)
    inner = HEIGHTS <= 1000
    a, b = np.polyfit(HEIGHTS[inner], med[inner], 1)
    r = med[inner] - np.polyval([a, b], HEIGHTS[inner])
    print(f"slope {a:.3f} (z_fit per nm of height), rms about the line {np.sqrt((r ** 2).mean()):.1f} nm")


if __name__ == "__main__":
    main()
