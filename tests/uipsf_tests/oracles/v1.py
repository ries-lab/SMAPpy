"""uiPSF v1's PSFs (tests/uipsf_tests/fixtures/oracle_v1.npz, by scripts/uipsf/make_oracles_v1.py), and v2's."""
import os

import numpy as np
import torch

import uipsf

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "oracle_v1.npz")


def load():
    f = np.load(FIXTURE)
    cases = {}
    for name in f["cases"]:
        cases[str(name)] = {k.split("/", 1)[1]: f[k] for k in f.files if k.startswith(f"{name}/")}
    return f, cases


def v2_psf(f, case, dtype=torch.float64, device="cpu"):
    """v2's stack for one case, [E, K, Y, X], in v2's normalisation."""
    NA, lam, nimm, nmed, ncov = (float(x) for x in f["optics"])
    optics = uipsf.Optics(NA=NA, wavelength_nm=lam, n_imm=nimm, n_med=nmed, n_cov=ncov,
                          pixel_size_nm=tuple(float(x) for x in case["pixel_nm"]))
    grid = uipsf.PupilGrid.build(optics, int(f["pupil_size"]), device=device, dtype=dtype)
    n = len(case["phase"])
    # v1 uses every Noll mode; with zeros where v2 leaves modes out, the two pupils are the same.
    pupil = uipsf.ZernikePupil(grid, phase_modes=list(range(1, n + 1)), magnitude_modes=list(range(2, n + 1)))
    analyser = str(case["analyser"]) or None
    psf = uipsf.VectorPSF(grid, pupil, (21, 21), upsample=int(case["bin"]), kind=str(case["kind"]),
                          analyser=analyser, bead_radius_nm=float(case["bead_radius_nm"]))
    mag = case["magnitude"] / case["magnitude"][0]          # v1's piston is the magnitude's scale
    p = {"pupil.phase": torch.tensor(case["phase"], dtype=dtype, device=device),
         "pupil.magnitude": torch.tensor(mag[1:], dtype=dtype, device=device),
         "psf.sigma": torch.tensor(case["sigma_nm"], dtype=dtype, device=device)}
    xyz = torch.tensor(case["xyz"], dtype=dtype, device=device)
    depth = torch.tensor(case["depth"], dtype=dtype, device=device) if np.any(case["depth"]) else None
    return psf(p, xyz, depth, z_step_nm=50.0)


def compare(v2, v1):
    """Largest difference relative to the peak, after one global scale (v1 normalises in the region)."""
    v2 = np.asarray(v2, float)
    v1 = np.asarray(v1, float)
    c = (v2 * v1).sum() / (v1 * v1).sum()
    return np.abs(v2 - c * v1).max() / np.abs(v2).max(), c
