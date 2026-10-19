"""v2's forward model against uiPSF v1 (TensorFlow) and against the TU Delft vectorial model (vecfit).

The v1 fixtures were made by scripts/uipsf/make_oracles_v1.py from uiPSF test_roi b367f00.  One global
scale is allowed: v1 normalises within the region, v2 to the whole image plane.
"""
import numpy as np
import pytest
import torch

import uipsf
from oracles import vecfit
from oracles.v1 import compare, load, v2_psf
from uipsf.optics.kernels import gaussian_otf, sphere_otf

F, CASES = load()
EXACT = ["ideal", "aberrated", "bin2", "scalar", "analyser_x", "analyser_y", "shifted_bin1", "depth"]


@pytest.mark.parametrize("name", EXACT)
def test_v2_reproduces_uipsf_v1_to_float32_precision(name):
    err, _ = compare(v2_psf(F, CASES[name]).numpy(), CASES[name]["image"])
    assert err < 1e-6


@pytest.mark.parametrize("name", ["bin2_blur", "shifted_bin2"])
def test_with_an_even_region_v1s_blur_is_off_by_half_a_frequency_sample_and_nothing_else(name):
    # v1 builds the blur on (i − L/2 + ½)/L, the FFT's frequencies are (i − L/2)/L: same for odd L,
    # half a sample off for even L (bin 2).  With v1's grid put back, the two agree exactly.
    case = CASES[name]
    v2 = v2_psf(F, case).numpy()
    assert 1e-6 < compare(v2, case["image"])[0] < 3e-4
    b = int(case["bin"])
    L = 21 * b
    sd = case["sigma_nm"] / (case["pixel_nm"] / b)                    # in upsampled pixels
    q = (np.arange(L) - L / 2 + 0.5) / L
    Hc = np.exp(-2 * np.pi ** 2 * (sd[0] ** 2 * q[None, :] ** 2 + sd[1] ** 2 * q[:, None] ** 2))
    H = torch.tensor(np.fft.ifftshift(Hc / Hc.max()))
    psf, p, xyz = _psf_for(case)
    I = psf.intensity(p, xyz)
    out = uipsf.optics.kernels.bin_pixels(torch.fft.ifft2(torch.fft.fft2(I) * H).real, b)
    assert compare(out.numpy(), case["image"])[0] < 1e-6


def test_the_bead_kernel_is_v1s_and_only_v1s_wrap_around_differs():
    case = CASES["bead"]
    psf, p, xyz = _psf_for(case)
    # v1 convolves its 47 planes circularly; done the same way, v2 matches exactly.
    I = psf.intensity(p, xyz)
    H = gaussian_otf((21, 21), psf.spacing, p["psf.sigma"]) * sphere_otf(
        (47, 21, 21), (*psf.spacing, 50.0), 50.0, dtype=torch.float64)
    circular = torch.fft.ifftn(torch.fft.fftn(I, dim=(-3, -2, -1)) * H, dim=(-3, -2, -1)).real
    assert compare(circular.numpy(), case["image"])[0] < 1e-6
    # v2 pads the stack instead; away from the ends the two agree.
    padded = psf(p, xyz, z_step_nm=50.0).numpy()
    assert compare(padded[:, 6:41], case["image"][:, 6:41])[0] < 5e-4


def _psf_for(case):
    """Rebuild the PSF object of a case (v2_psf builds and evaluates in one go)."""
    NA, lam, nimm, nmed, ncov = (float(x) for x in F["optics"])
    o = uipsf.Optics(NA=NA, wavelength_nm=lam, n_imm=nimm, n_med=nmed, n_cov=ncov,
                     pixel_size_nm=tuple(float(x) for x in case["pixel_nm"]))
    g = uipsf.PupilGrid.build(o, int(F["pupil_size"]), device="cpu", dtype=torch.float64)
    n = len(case["phase"])
    pupil = uipsf.ZernikePupil(g, phase_modes=list(range(1, n + 1)), magnitude_modes=list(range(2, n + 1)))
    psf = uipsf.VectorPSF(g, pupil, (21, 21), upsample=int(case["bin"]), bead_radius_nm=float(case["bead_radius_nm"]))
    mag = case["magnitude"] / case["magnitude"][0]
    p = {"pupil.phase": torch.tensor(case["phase"]), "pupil.magnitude": torch.tensor(mag[1:]),
         "psf.sigma": torch.tensor(case["sigma_nm"])}
    return psf, p, torch.tensor(case["xyz"])


@pytest.mark.parametrize("aberrated", [False, True])
def test_v2_reproduces_the_tu_delft_vectorial_model(optics, aberrated):
    g = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)
    pupil = uipsf.ZernikePupil(g, n_max=6, magnitude_n_max=0)
    psf = uipsf.VectorPSF(g, pupil, (21, 21))
    phase = torch.zeros(len(pupil.phase_modes), dtype=torch.float64)
    if aberrated:
        for j, v in {5: 0.6, 6: -0.3, 7: 0.25, 8: -0.2, 11: 0.4}.items():
            phase[pupil.phase_modes.index(j)] = v
    p = {"pupil.phase": phase, "pupil.magnitude": torch.zeros(0, dtype=torch.float64),
         "psf.sigma": torch.zeros(2, dtype=torch.float64)}
    z = np.linspace(-1000, 1000, 41)
    xyz = torch.zeros(1, 41, 3, dtype=torch.float64)
    xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2] = 60.0, -35.0, torch.tensor(z)
    ours = psf(p, xyz)[0].numpy()
    W = (torch.tensordot(phase, pupil.phase_basis, 1) * g.aperture).numpy()
    theirs = vecfit.psf(1.43, 680, 1.516, 1.335, 1.516, 64, (21, 21), (127, 116), z, W, 60.0, -35.0)
    assert compare(ours, theirs)[0] < 1e-8
