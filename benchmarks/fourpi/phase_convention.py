"""Which phase convention fits real beads: SMAPpy's, or DECODE_Internal's 4Pi branch's (docs/4pi.md §30).

SMAPpy's IAB model is mu_c = bg + N (I_c + A_c cos phi + B_c sin phi): each channel's phase offset is
inside its own A_c and B_c (`smappy.fourpi`).  The branch `4pi-simulation` of DECODE_Internal reads
the same uiPSF v1 files the same way (A = 2 Re, B = -2 Im) and then adds o_c = 0, pi/2, pi, 3 pi/2 per
channel, cos(phi + o_c) and sin(phi + o_c) (`psf_kernel.py:1812`).  Both are fitted here to the beads
uiPSF v1 learned its model from, by the C++ fitter: the branch's convention as a calibration whose A
and B are turned by o_c, A' = A cos o + B sin o, B' = B cos o - A sin o, which is the same model.

    python benchmarks/fourpi/phase_convention.py
"""
import os
from pathlib import Path

import numpy as np

from smappy import fourpi
from smappy.psf import Global4PiPSF


def rois_and_model():
    import h5py
    root = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
    path = root / "results/4pi_40nm_bead/psfmodel_zernike_4pi.h5"
    cal = fourpi.load_uipsf_v1(path, transpose=False)
    with h5py.File(path, "r") as f:
        data = f["rois/psf_data"][...]                 # (C, bead, phase, z, y, x)
    C, nb, nph, nz, size, _ = data.shape
    o = (size - 15) // 2
    rois = np.moveaxis(data[..., o:o + 15, o:o + 15], 0, 3).reshape(-1, C, 15, 15)
    return cal, rois.astype(np.float32), (nb, nph, nz)


def turned(cal, offsets):
    """The same calibration with each channel's interference turned by ``offsets``: its A and B
    rewritten so that A' cos phi + B' sin phi = A cos(phi + o) + B sin(phi + o)."""
    I, A, B = (cal.volumes[:, k].astype(float) for k in range(3))
    o = np.asarray(offsets, float)[:, None, None, None]
    return fourpi.from_volumes(I, A * np.cos(o) + B * np.sin(o), B * np.cos(o) - A * np.sin(o),
                               dz=cal.dz, z0=cal.z0, period_nm=cal.period_nm,
                               phase_per_nm=cal.phase_per_nm)


def main():
    cal, rois, (nb, nph, nz) = rois_and_model()
    ours = fourpi.from_volumes(*(cal.volumes[:, k] for k in range(3)), dz=cal.dz, z0=cal.z0,
                               period_nm=cal.period_nm, phase_per_nm=cal.phase_per_nm)
    hao = turned(cal, np.arange(cal.n_channels) * np.pi / 2)
    stage = np.arange(nz) * 50.0
    print(f"{len(rois)} bead ROIs ({nb} beads x {nph} mirror phases x {nz} planes)")
    print("convention   sum logL   median logL/ROI   period from the phase (nm)   steps (rad, set 2 and -2)")
    lls = {}
    for name, c in (("SMAPpy", ours), ("branch", hao)):
        model = Global4PiPSF(c, phase_starts=4)
        result = model.fit(rois, iterations=100)
        v = model.unpack(result)
        phase = v["phase_rad"].reshape(nb, nph, nz)
        slopes = [np.polyfit(stage[3:-3], np.unwrap(phase[b, 0, 3:-3]), 1)[0] for b in range(nb)]
        period = 2 * np.pi / abs(np.median(slopes))
        steps = np.median(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :1]))), axis=(0, 2))
        lls[name] = ll = result.logl
        print(f"{name:10s} {np.nansum(ll):10.0f} {np.nanmedian(ll):12.1f} {period:20.1f}"
              f"            {steps[0]:+.2f} {steps[1]:+.2f}")
    better = np.mean(lls["SMAPpy"] > lls["branch"])
    print(f"SMAPpy's convention fits {100 * better:.1f} % of the ROIs better")


if __name__ == "__main__":
    main()
