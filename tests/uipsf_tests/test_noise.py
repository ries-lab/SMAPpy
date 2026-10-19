import math

import pytest
import torch

from uipsf.noise import EMCCD, Poisson, ShiftedPoisson

f64 = torch.float64


def test_the_deviance_is_zero_where_the_model_equals_the_data_and_positive_elsewhere():
    d = torch.tensor([0.0, 1.0, 5.0, 300.0], dtype=f64)
    for noise, s in [(Poisson(), 0.0), (ShiftedPoisson(), 2.0), (EMCCD(), 0.0)]:
        assert noise.deviance(d.clamp(min=1e-12), d, s).abs().max() < 1e-10
        assert (noise.deviance(d + 0.7, d, s) > 0).all()


def test_the_gradient_is_the_derivative_of_half_the_deviance():
    d = torch.tensor([0.0, 3.0, 40.0, 900.0], dtype=f64)
    mu = torch.tensor([0.5, 2.0, 55.0, 870.0], dtype=f64, requires_grad=True)
    for noise, s in [(Poisson(), 0.0), (ShiftedPoisson(), 1.7), (EMCCD(), 0.3)]:
        (auto,) = torch.autograd.grad(0.5 * noise.deviance(mu, d, s).sum(), mu)
        grad, _ = noise.derivatives(mu.detach(), d, s)
        assert torch.allclose(grad, auto, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("noise,s", [(Poisson(), 0.0), (ShiftedPoisson(), 2.5), (EMCCD(), 0.0)])
def test_the_weight_is_the_fisher_information_of_the_counts(noise, s):
    # E[∂²(½D)/∂μ²] over the counts' own distribution: Poisson in d/F, exactly summed.
    mu = torch.tensor(7.0, dtype=f64)
    F = noise.excess
    k = torch.arange(0, 200, dtype=f64)
    pmf = torch.exp(k * torch.log(mu / F) - mu / F - torch.lgamma(k + 1))
    m = mu.clone().requires_grad_(True)
    d = k * F
    (g,) = torch.autograd.grad(0.5 * noise.deviance(m, d, s), m, grad_outputs=pmf, create_graph=True)
    (h,) = torch.autograd.grad(g, m)
    _, w = noise.derivatives(mu, torch.tensor(0.0, dtype=f64), s)
    assert abs(h.item() - w.item()) < 1e-10


def test_the_mean_deviance_of_poisson_counts_is_one_where_the_counts_are_many():
    g = torch.Generator().manual_seed(0)
    mu = torch.full((200_000,), 400.0, dtype=f64)
    d = torch.poisson(mu, generator=g)
    assert abs(Poisson().deviance(mu, d).mean().item() - 1) < 0.01


def test_read_noise_shifts_data_and_model_and_negative_values_are_clamped_and_counted():
    noise = ShiftedPoisson()
    d = torch.tensor([-5.0, -1.0, 3.0], dtype=f64)
    assert noise.clamped(d, 2.0) == 1
    D = noise.deviance(torch.tensor([1.0, 1.0, 3.0], dtype=f64), d, 2.0)
    assert math.isclose(D[0].item(), 2 * 3.0, rel_tol=1e-12)              # d′ = 0, μ′ = 3: D = 2μ′
    assert D[2].item() == 0


def test_a_mask_removes_pixels_from_the_deviance_and_the_derivatives():
    noise = Poisson()
    d, mu = torch.tensor([3.0, 9.0], dtype=f64), torch.tensor([5.0, 5.0], dtype=f64)
    mask = torch.tensor([True, False])
    assert noise.deviance(mu, d, mask=mask)[1] == 0
    g, w = noise.derivatives(mu, d, mask=mask)
    assert g[1] == 0 and w[1] == 0 and w[0] > 0


def test_electron_multiplication_halves_the_information():
    mu, d = torch.tensor([50.0], dtype=f64), torch.tensor([60.0], dtype=f64)
    assert torch.allclose(EMCCD().deviance(mu, d), Poisson().deviance(mu, d) / 2)
    assert torch.allclose(EMCCD().derivatives(mu, d)[1], Poisson().derivatives(mu, d)[1] / 2)
