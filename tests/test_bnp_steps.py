"""BNP-Step (`smappy.bnp_steps`): the squared error from prefix sums is the
model's, and the sampler learns the number of steps, their places and the
noise of simulated staircases, batched or one at a time."""
import numpy as np
import pytest

from smappy.bnp_steps import _Batch, _ss, bnp_steps, noise_sd

from test_find_steps import staircase


def test_the_squared_error_from_prefix_sums_is_the_models():
    x, _ = staircase(n=60, seed=3)
    batch = _Batch([x])
    k = np.array([[5, 20, 31, 59]])
    a = np.array([[2.0, -1.5, 3.0, 0.5]])
    F = np.array([0.7])
    n = np.arange(60)
    model = F[0] + sum(a[0, j] * (n <= k[0, j]) for j in range(4))
    assert _ss(batch, F, a, k)[0] == pytest.approx(np.sum((x - x.mean() - model) ** 2))


def test_the_noise_from_consecutive_differences_ignores_the_steps():
    x, _ = staircase(n=3000, noise=3.0, seed=1)
    assert noise_sd(x) == pytest.approx(3.0, rel=0.1)


def test_the_sampler_finds_the_steps_of_every_trace_in_a_batch():
    # kinesin-like dwells (two events a step): rarely shorter than the noise resolves
    data = [staircase(n=500, noise=3.0, shape=2.0, seed=s) for s in range(12)]
    res = bnp_steps([x for x, _ in data], sweeps=150, seed=0)
    found_n = sum(len(f) for f in res.boundaries)
    true_n = sum(len(t) for _, t in data)
    assert found_n == pytest.approx(true_n, rel=0.1)
    for (x, truth), found, sd in zip(data, res.boundaries, res.noise_sd):
        near = np.min(np.abs(found[None, :] - truth[:, None]), axis=1) <= 2
        assert near.mean() > 0.8
        assert sd == pytest.approx(3.0, rel=0.2)


def test_from_the_prior_alone_the_sampler_still_finds_the_steps():
    x, truth = staircase(n=250, noise=2.5, dwell=30.0, shape=2.0, seed=4)
    res = bnp_steps([x], sweeps=1500, weak_limit=25, start="prior", seed=1)
    assert abs(res.n_steps[0] - len(truth)) <= 1


def test_noise_alone_has_no_steps():
    x = np.random.default_rng(5).normal(0, 3.0, 400)
    res = bnp_steps([x], sweeps=200, seed=2)
    assert res.n_steps[0] <= 1
