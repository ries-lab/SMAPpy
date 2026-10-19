import numpy as np
import torch

import uipsf
from oracles.v1 import load
from uipsf.optics.zernike import nm_to_noll, noll_to_nm, orthogonalise, zernike_basis, zernike_modes


def test_noll_indices_follow_nolls_table():
    table = [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3), (4, 0)]
    assert [noll_to_nm(j) for j in range(1, 12)] == table
    assert all(nm_to_noll(*noll_to_nm(j)) == j for j in range(1, 67))


def test_piston_tilt_and_defocus_are_left_out_of_the_phase_by_default():
    assert zernike_modes(2) == [5, 6]
    assert zernike_modes(3, exclude=("piston",))[:4] == [2, 3, 4, 5]
    assert len(zernike_modes(8, exclude=())) == 45


def test_the_modes_are_orthonormal_over_the_disc(optics):
    g = uipsf.PupilGrid.build(optics, 512, device="cpu", dtype=torch.float64)
    Z = zernike_basis(g, list(range(1, 16)))[:, g.aperture > 0]
    gram = (Z @ Z.T / Z.shape[1]).numpy()
    assert np.abs(gram - np.eye(15)).max() < 0.02


def test_the_basis_is_uipsf_v1s_genZern1_sample_for_sample(optics):
    f, _ = load()
    g = uipsf.PupilGrid.build(optics, int(f["pupil_size"]), device="cpu", dtype=torch.float64)
    Z = zernike_basis(g, list(range(1, 46))).numpy()
    assert np.abs(Z - f["zernike_v1"]).max() < 1e-10


def test_high_NA_modes_are_orthogonal_to_tilt_and_to_the_exact_defocus(optics):
    g = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)
    inside = g.aperture > 0
    Z = zernike_basis(g, zernike_modes(6), high_na=True)[:, inside]
    kz = g.kz_imm.real[inside]
    against = torch.stack([torch.ones_like(kz), g.u[inside], g.v[inside], kz])
    assert (Z @ against.T).abs().max() / Z.shape[1] < 1e-10
    assert torch.allclose((Z ** 2).mean(1), torch.ones(Z.shape[0], dtype=torch.float64))
    # the paraxial modes are not: spherical (Noll 11) leaks into the exact defocus at NA 1.43
    Z11 = zernike_basis(g, [11])[0][inside]
    assert abs((Z11 * (kz - kz.mean())).mean()) > 1e-6


def test_orthogonalising_against_a_mode_removes_it():
    mask = np.ones((4, 4))
    a = np.arange(16.0).reshape(1, 4, 4)
    out = orthogonalise(np.concatenate([a, a + 1]), np.ones((1, 4, 4)), mask)
    assert np.allclose(out[0], out[1]) and np.isclose(out.mean(), 0)
