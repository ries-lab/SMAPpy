import numpy as np
import pytest
import torch

import uipsf
from uipsf_tests.conftest import make_psf, stack


def test_an_unaberrated_psf_in_focus_is_mirror_symmetric():
    o = uipsf.Optics(NA=1.4, wavelength_nm=600, n_imm=1.518, n_med=1.33, pixel_size_nm=(100, 100))
    psf, p = make_psf(o)
    I = psf(p, stack([0.0, 300.0]))[0]
    assert torch.allclose(I, I.flip(-1), atol=1e-12) and torch.allclose(I, I.flip(-2), atol=1e-12)
    assert torch.allclose(I, I.transpose(-1, -2), atol=1e-12)


def test_moving_the_emitter_by_one_pixel_moves_the_image_by_one_pixel(optics):
    psf, p = make_psf(optics)
    p["pupil.phase"][:4] = torch.tensor([0.5, -0.3, 0.2, 0.1], dtype=torch.float64)
    a = psf(p, stack([200.0], x=30.0, y=-20.0))[0, 0]
    b = psf(p, stack([200.0], x=30.0 + 127, y=-20.0 - 116))[0, 0]
    assert torch.allclose(b[:-1, 1:], a[1:, :-1], atol=1e-12)


def test_the_image_integrates_to_one_photon_and_aberrations_keep_that(optics):
    # A region of exactly one DFT period (1/Δk) holds the whole image, and the discrete Parseval
    # theorem makes its sum exact.
    period = 680 * 64 / (2 * 1.43)
    o = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335,
                     pixel_size_nm=(period / 151, period / 151))
    psf, p = make_psf(o, P=64, roi=(151, 151), n_max=4)
    z = [0.0, 600.0, -600.0]
    one = torch.ones(1, 3, dtype=torch.float64)
    assert torch.allclose(psf(p, stack(z)).sum((-2, -1)), one, atol=1e-10)
    p["pupil.phase"][:] = 0.4
    p["pupil.magnitude"][:] = 0.1                       # the magnitude's energy is normalised away
    assert torch.allclose(psf(p, stack(z)).sum((-2, -1)), one, atol=1e-10)


def test_depth_in_the_sample_loses_supercritical_light(optics):
    psf, p = make_psf(optics)
    near = psf(p, stack([0.0]), depth_nm=torch.tensor([0.0], dtype=torch.float64)).sum()
    deep = psf(p, stack([0.0]), depth_nm=torch.tensor([1000.0], dtype=torch.float64)).sum()
    assert deep < 0.85 * near


def test_positive_vertical_astigmatism_stretches_the_spot_along_y_above_focus(optics):
    # docs/uipsf/conventions.md: the Zernike sign, anchored on what the image shows.
    psf, p = make_psf(optics)
    p["psf.sigma"] = torch.zeros(2, dtype=torch.float64)
    p["pupil.phase"][psf.pupil.phase_modes.index(6)] = 0.5            # Noll 6, cos 2φ
    I = psf(p, stack([400.0, -400.0]))[0].numpy()
    yy, xx = np.mgrid[:21, :21] - 10.0
    sd = [(np.sqrt((w * (xx * 127) ** 2).sum()), np.sqrt((w * (yy * 116) ** 2).sum()))
          for w in I / I.sum((-2, -1), keepdims=True)]
    assert sd[0][1] > 1.15 * sd[0][0]          # above focus (z > 0): longer along y
    assert sd[1][0] > 1.15 * sd[1][1]          # below focus: longer along x


def test_float32_and_float64_agree(optics):
    out = []
    for dt in (torch.float32, torch.float64):
        psf, p = make_psf(optics, dtype=dt)
        p["pupil.phase"][:5] = torch.tensor([0.6, -0.3, 0.25, -0.2, 0.4], dtype=dt)
        p["psf.sigma"] = torch.tensor([40.0, 30.0], dtype=dt)
        out.append(psf(p, stack(np.linspace(-1000, 1000, 11), x=25.0, dtype=dt)).double())
    assert (out[0] - out[1]).abs().max() / out[1].max() < 1e-5


def test_each_emitter_sees_only_its_own_parameters(optics):
    # The solver's block structure relies on this (plan §5.4).
    psf, p = make_psf(optics, P=32, roi=(9, 9))
    xyz = stack([-200.0, 0.0, 200.0], n=3)
    xyz[:, :, 0] = torch.tensor([[-50.0], [0.0], [80.0]])
    tangent = torch.zeros_like(xyz)
    tangent[0] = 1.0
    _, dI = torch.func.jvp(lambda q: psf(p, q), (xyz,), (tangent,))
    assert dI[0].abs().max() > 0
    assert dI[1:].abs().max() == 0


def test_one_pupil_per_emitter_is_the_same_as_separate_calls(optics):
    psf, p = make_psf(optics, P=32, roi=(9, 9))
    coeff = torch.randn(2, len(psf.pupil.phase_modes), dtype=torch.float64) * 0.3
    xyz = stack([-300.0, 300.0], n=2)
    both = psf({**p, "pupil.phase": coeff}, xyz)
    for e in range(2):
        one = psf({**p, "pupil.phase": coeff[e]}, xyz[e:e + 1])
        assert torch.allclose(both[e], one[0], atol=1e-14)


def test_gradients_of_every_parameter_match_finite_differences(optics):
    psf, p = make_psf(optics, P=12, roi=(5, 5), n_max=3, bead_radius_nm=60.0)
    xyz0 = stack([-100.0, 0.0, 100.0], x=20.0, y=-10.0)

    def f(phase, magnitude, sigma, xyz, depth):
        q = {"pupil.phase": phase, "pupil.magnitude": magnitude, "psf.sigma": sigma}
        return psf(q, xyz, depth, z_step_nm=100.0)

    n_phase, n_mag = len(psf.pupil.phase_modes), len(psf.pupil.magnitude_modes)
    f64 = torch.float64
    args = (torch.linspace(-0.3, 0.3, n_phase, dtype=f64), torch.full((n_mag,), 0.05, dtype=f64),
            torch.tensor([30.0, 45.0], dtype=f64), xyz0, torch.tensor([150.0], dtype=f64))
    args = tuple(a.clone().requires_grad_(True) for a in args)
    assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)


def test_a_bead_kernel_needs_a_z_stack_and_its_step(optics):
    psf, p = make_psf(optics, P=16, roi=(5, 5), bead_radius_nm=50.0)
    with pytest.raises(ValueError, match="z stack"):
        psf(p, stack([0.0]), z_step_nm=50.0)
    with pytest.raises(ValueError, match="z stack"):
        psf(p, stack([0.0, 50.0]))


def test_the_padding_continues_the_stack_in_its_own_direction():
    xyz = stack([300.0, 250.0, 200.0])
    xyz[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    out = uipsf.VectorPSF._padded_stack(xyz, 50.0, 2)
    assert out[0, :, 2].tolist() == [400.0, 350.0, 300.0, 250.0, 200.0, 150.0, 100.0]
    assert out[0, :, 0].tolist() == [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]


def test_the_bead_kernel_blurs_along_z_without_changing_the_total(optics):
    psf0, p = make_psf(optics, P=32, roi=(15, 15))
    psf1, _ = make_psf(optics, P=32, roi=(15, 15), bead_radius_nm=100.0)
    z = np.linspace(-500, 500, 21)
    a, b = psf0(p, stack(z)), psf1(p, stack(z), z_step_nm=50.0)
    assert b.max() < a.max()
    assert abs(b.sum() / a.sum() - 1) < 0.01
