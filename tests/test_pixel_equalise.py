import numpy as np

from smappy import pixel_equalise as pe
from smappy.training.reference import pixel_modulation


def _pulled(n, pull, seed=0):
    """Uniform true positions, found pulled toward their pixel's centre by ``pull`` (a fraction)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 40, n)
    c = np.round(t)
    return t, c + (t - c) * (1 - pull), rng.uniform(-500, 500, n), rng.uniform(0.05, 0.3, n)


def test_a_pull_to_the_pixel_centre_without_noise_is_undone_exactly():
    t, f, z, s = _pulled(200_000, 0.3)
    assert pixel_modulation(f) > 0.2
    x, _ = pe.equalise(f, f, z, s, s)
    assert pixel_modulation(x) < 0.01
    # a monotone pull: the equalised place is the true one, within the map's 40 knots
    assert np.sqrt(np.mean((x - t) ** 2)) < 0.01


def test_the_maps_learnt_on_one_half_flatten_the_other_and_keep_every_localization_in_its_pixel():
    _, f, z, s = _pulled(200_000, 0.3)
    f = f + np.random.default_rng(1).normal(0, 0.02, len(f))
    a = np.arange(len(f)) % 2 == 0
    eq = pe.fit(f[a], f[a], z[a], s[a], s[a])
    x, _ = eq(f[~a], f[~a], z[~a], s[~a], s[~a])
    assert pixel_modulation(x) < 0.02 < pixel_modulation(f[~a])
    assert np.array_equal(np.round(x), np.round(f[~a]))


def test_a_group_with_too_few_localizations_is_left_as_it_is():
    _, f, z, s = _pulled(300, 0.3)
    x, y = pe.equalise(f, f, z, s, s)
    assert np.array_equal(x, f) and np.array_equal(y, f)
