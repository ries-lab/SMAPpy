import pytest
import torch

from uipsf.params import Bounded, Exp, Gaussian, Identity, Params, ParamSpec, Softplus


@pytest.mark.parametrize("t,values", [(Identity(), [-3.0, 0.0, 2.0]), (Softplus(), [1e-3, 0.5, 40.0]),
                                      (Softplus(floor=1.0), [1.5, 3.0]), (Exp(), [1e-4, 7.0]),
                                      (Bounded(-2.0, 5.0), [-1.9, 0.0, 4.9])])
def test_transforms_invert(t, values):
    y = torch.tensor(values, dtype=torch.float64)
    assert torch.allclose(t.forward(t.inverse(y)), y, rtol=1e-10, atol=1e-12)


def _specs():
    return [ParamSpec("pupil.phase", (3,)), ParamSpec("psf.sigma", (2,), init=30.0, transform=Softplus()),
            ParamSpec("emitter.xyz", (3,), scope="emitter"),
            ParamSpec("emitter.photons", (), scope="emitter", init=1000.0, transform=Softplus(),
                      prior=Gaussian(1000.0, 100.0))]


def test_the_blocks_pack_and_unpack_without_loss():
    p = Params(_specs(), n_emitters=4, dtype=torch.float64)
    s, e = p.shared_vector(), p.emitter_matrix()
    assert s.shape == (5,) and e.shape == (4, 4)
    s2, e2 = torch.randn(5, dtype=torch.float64), torch.randn(4, 4, dtype=torch.float64)
    q = p.replace(shared=s2, emitter=e2)
    assert torch.equal(q.shared_vector(), s2) and torch.equal(q.emitter_matrix(), e2)
    assert torch.equal(p.shared_vector(), s)                            # the original is unchanged


def test_physical_values_are_the_inits_and_fixed_parameters_leave_the_blocks():
    p = Params(_specs(), n_emitters=2, dtype=torch.float64)
    phys = p.physical()
    assert torch.allclose(phys["psf.sigma"], torch.full((2,), 30.0, dtype=torch.float64))
    assert phys["emitter.xyz"].shape == (2, 3)
    q = p.fix("psf.sigma", "emitter.xyz")
    assert q.shared_vector().shape == (3,) and q.emitter_matrix().shape == (2, 1)
    assert q.free("psf.sigma").shared_vector().shape == (5,)


def test_a_gaussian_prior_is_half_the_squared_standardised_distance():
    p = Params(_specs(), n_emitters=1, dtype=torch.float64, values={"emitter.photons": 1200.0})
    assert abs(p.prior_penalty().item() - 0.5 * 2 ** 2) < 1e-6


def test_parameter_names_must_be_unique():
    with pytest.raises(ValueError):
        Params([ParamSpec("a", ()), ParamSpec("a", ())])
