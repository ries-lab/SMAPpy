import math

import numpy as np
import pytest
import torch

import uipsf
from uipsf.optics import choose_pupil_size


def test_optics_refuses_an_NA_above_the_immersion_index_and_defaults_the_coverslip():
    with pytest.raises(ValueError):
        uipsf.Optics(NA=1.6, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(100, 100))
    o = uipsf.Optics(NA=1.4, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(100, 100))
    assert o.n_cov == 1.518


def test_the_pupil_size_reaches_pi_per_sample_where_the_review_says_it_aliases(optics):
    # Review §A: at NA 1.43 and 64 samples the defocus step reaches π per sample near |z| = 2.5-3 µm.
    assert choose_pupil_size(optics, 2500, max_phase_step=math.pi) == 64
    assert choose_pupil_size(optics, 3500, max_phase_step=math.pi) > 64


def test_the_pupil_size_grows_with_z_in_steps_of_16_and_never_drops_below_64(optics):
    sizes = [choose_pupil_size(optics, z) for z in (0, 1000, 2000, 4000, 8000)]
    assert sizes == sorted(sizes) and sizes[0] == 64
    assert all(s % 16 == 0 for s in sizes)


def test_fresnel_transmission_at_normal_incidence_is_the_product_of_the_two_interfaces(optics):
    g = uipsf.PupilGrid.build(optics, 65, device="cpu", dtype=torch.float64)    # odd: a sample at ρ = 0
    nm, nc, ni = optics.n_med, optics.n_cov, optics.n_imm
    expected = 2 * nm / (nm + nc) * 2 * nc / (nc + ni)
    assert g.rho[32, 32] == 0
    assert abs(g.fresnel_p[32, 32].item() - expected) < 1e-12
    assert abs(g.fresnel_s[32, 32].item() - expected) < 1e-12


def test_above_the_critical_angle_kz_in_the_medium_is_positive_imaginary_so_depth_decays(optics):
    g = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)
    beyond = (g.rho * optics.NA > optics.n_med) & (g.aperture > 0)
    assert beyond.any()
    kz = g.kz_med[beyond]
    assert torch.all(kz.imag > 0) and torch.all(kz.real.abs() < 1e-15)
    assert torch.all(g.kz_imm[g.aperture > 0].imag == 0)


def test_pupil_samples_are_centred_and_spaced_by_two_NA_over_lambda_P(optics):
    g = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)
    assert np.isclose(g.dk, 2 * 1.43 / 680 / 64)
    assert torch.allclose(g.kx[1:] - g.kx[:-1], torch.full((63,), g.dk, dtype=torch.float64))
    assert abs(g.kx.sum().item()) < 1e-15
    assert torch.equal(g.kx, g.ky)
    assert g.u[0, 1] > g.u[0, 0] and g.v[1, 0] > g.v[0, 0]             # u along columns, v along rows
