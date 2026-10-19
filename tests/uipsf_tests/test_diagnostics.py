import numpy as np

from uipsf.diagnostics import noise_transfer


def _counts(conversion_error=1.0, read_noise=0.0, seed=0):
    """Poisson photons with read noise, then given to the fit in units off by ``conversion_error``."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(5, 300, 400_000)
    d = rng.poisson(mu) + rng.normal(0, read_noise, mu.size)
    return d * conversion_error, mu * conversion_error


def test_the_right_camera_gives_slope_one_and_the_read_noise_as_intercept():
    t = noise_transfer(*_counts(read_noise=3.0))
    assert abs(t.slope - 1) < 0.03
    assert abs(t.read_noise_photons - 3.0) < 0.3


def test_a_conversion_three_times_too_high_shows_as_slope_three():
    # Photons = ADU · conversion: too high a conversion multiplies counts and variance alike,
    # so the variance grows as 3 μ instead of μ.
    t = noise_transfer(*_counts(conversion_error=3.0))
    assert abs(t.conversion_factor() - 3) < 0.1
