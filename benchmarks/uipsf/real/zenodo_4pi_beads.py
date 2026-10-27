"""4Pi from beads on Zenodo `4pi_40nm_bead`, against uiPSF v1's model of the same beads (docs/4pi.md, M5).

    python benchmarks/uipsf/real/zenodo_4pi_beads.py [--device mps] [--n-max 6] [--out DIR]

The beads are v1's own ROIs (``rois/psf_data`` of ``results/4pi_40nm_bead/psfmodel_zernike_4pi.h5``: 18
beads, 4 channels, 3 mirror steps, 21 slices 50 nm apart, 21 × 21 pixels, photons), so both models are
learnt from the same pixels, in v1's (transposed) frame.  Each channel's ROI centre is v1's ``rois/cor``.
The optics are v1's: NA 1.35, n 1.406 matched, 600 nm, 129 nm pixels.  z_nm = −(slice − 10)·50, the
direction v1's model gives its beads (SMAPpy's sign).

Then both models are exported to SMAPpy's IAB calibration and fit the beads with `Global4PiPSF`:
- **the fringe:** the phase against the stage, per bead and mirror step, fitted by a line: its slope (the
  period) and its scatter about the line;
- **z:** the same for z;
- **the mirror steps** as the phase measures them.
Prints one JSON line; the calibration is written to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from smappy import fourpi
from smappy.psf import Global4PiPSF

import uipsf
from uipsf.model import PerObservation
from uipsf.model.fourpi import FourPiModel
from uipsf.noise import ShiftedPoisson
from uipsf.workflows.fourpi import calibration, learn_fourpi

ROOT = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
V1 = ROOT / "results/4pi_40nm_bead/psfmodel_zernike_4pi.h5"
STEP_NM = 50.0
CROP = 15


def v1_batch(read_noise_var: float = 1.0):
    with h5py.File(V1, "r") as f:
        data = f["rois/psf_data"][...]                         # (C, bead, phase, z, y, x)
        cor = f["rois/cor"][...]                               # (C, bead, (row, column))
    C, E, M, J, Y, X = data.shape
    stack = np.transpose(data, (1, 0, 3, 2, 4, 5)).reshape(E, C, J * M, Y, X)   # z-major
    z = -(np.arange(J) - (J - 1) / 2) * STEP_NM
    roi_xy = np.transpose(cor, (1, 0, 2))[..., ::-1].astype(np.float32)        # (bead, C, (x, y))
    return uipsf.Batch(data=torch.as_tensor(stack, dtype=torch.float32),
                       z_nm=torch.as_tensor(np.repeat(z, M), dtype=torch.float32), s=read_noise_var,
                       roi_xy=torch.as_tensor(roi_xy.copy())), data


def tracks(cal, data):
    """Each model's fit of the beads: the fringe and z against the stage, per bead and mirror step."""
    C, E, M, J, Y, X = data.shape
    o = (Y - CROP) // 2
    rois = np.moveaxis(data[..., o:o + CROP, o:o + CROP], 0, 3).reshape(-1, C, CROP, CROP)
    model = Global4PiPSF(cal)
    v = model.unpack(model.fit(rois.astype(np.float32)))
    phase = v["phase_rad"].reshape(E, M, J)
    z = v["z_nm"].reshape(E, M, J)
    stage = np.arange(J) * STEP_NM
    core = slice(3, J - 3)
    slopes, phase_rms, z_rms, z_slopes = [], [], [], []
    for e in range(E):
        for m in range(M):
            u = np.unwrap(phase[e, m, core])
            fit = np.polyfit(stage[core], u, 1)
            slopes.append(fit[0])
            phase_rms.append(np.sqrt(np.mean((u - np.polyval(fit, stage[core])) ** 2)))
            fz = np.polyfit(stage[core], z[e, m, core], 1)
            z_slopes.append(fz[0])
            z_rms.append(np.sqrt(np.mean((z[e, m, core] - np.polyval(fz, stage[core])) ** 2)))
    steps = np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :1])))
    return {"period_nm": float(2 * np.pi / abs(np.median(slopes))),
            "phase_rms_rad": float(np.median(phase_rms)), "phase_rms_nm": float(
                np.median(phase_rms) / abs(np.median(slopes))),
            "z_rms_nm": float(np.median(z_rms)), "z_slope": float(np.median(z_slopes)),
            "steps_rad": np.median(steps, axis=(0, 2)).tolist(),
            "phase_err_rad": float(np.median(v["phase_err_rad"])), "z_err_nm": float(np.median(v["z_err_nm"])),
            "photons": float(np.median(v["photons"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--bead-radius", type=float, default=0.0)
    ap.add_argument("--photons", choices=("bleaching", "observation"), default="bleaching",
                    help="a bleaching curve per bead, or a free count per bead and observation (v1's var_photon)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(Path.home() / ".cache/uipsf/m5"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    batch, data = v1_batch()
    optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
    grid = uipsf.PupilGrid.build(optics, 64, device=a.device)
    model = FourPiModel(grid, batch.data.shape[-2:], n_max=a.n_max, steps_rad=[2, 0, -2],
                        bead_radius_nm=a.bead_radius,
                        intensity=PerObservation() if a.photons == "observation" else None)
    t = time.time()
    fit = learn_fourpi(model, batch, noise=ShiftedPoisson(),
                       progress=lambda e: print(e.get("stage"), e.get("iteration"), e.get("objective"),
                                                flush=True) if e.get("iteration", 0) % 10 == 0 else None)
    seconds = time.time() - t
    shared = {k: v.detach().cpu().double().numpy().tolist() for k, v in fit.shared().items()}
    deviance = float(fit.result.objective) * 2 / batch.data.numel()
    cal = calibration(fit, z_range_nm=600, dz_nm=25, roi_px=21)
    fourpi.save_fourpi_calibration(out / f"v2_4pi_nmax{a.n_max}{a.tag}.h5", cal, overwrite=True)
    (out / f"v2_4pi_nmax{a.n_max}{a.tag}.json").write_text(json.dumps(shared))
    v1 = fourpi.load_uipsf_v1(V1, transpose=False)
    row = {"n_max": a.n_max, "bead_radius": a.bead_radius, "photons": a.photons, "seconds": seconds,
           "iterations": len(fit.result.history), "stop": fit.result.stop_reason, "mean_deviance": deviance,
           "start": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fit.start.items()
                     if k in ("steps", "steps_flipped", "deltas", "period_nm")},
           "period_nm": cal.period_nm, "params": cal.parameters,
           "v2": tracks(cal, data), "v1": tracks(v1, data)}
    print(json.dumps(row))


if __name__ == "__main__":
    main()
