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


def _pulled_by_z(n, seed=0):
    """Uniform true places pulled toward the pixel's centre more at large |z|, positions in pixels
    with centres at whole numbers."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 40, n)
    z = rng.uniform(-600, 600, n)
    s = rng.uniform(5, 20, n)
    edge = np.round(t) - 0.5
    u = t - edge
    strength = 0.02 + 0.1 * np.abs(z) / 600
    return t, edge + u + strength * np.sin(2 * np.pi * u) / (2 * np.pi), z, s


def test_the_smooth_map_undoes_a_pull_that_grows_with_z_and_keeps_every_localization_in_its_pixel():
    t, f, z, s = _pulled_by_z(300_000)
    m = pe.fit_smooth_axis(f, z, s)
    g = m(f, z, s)
    assert pixel_modulation(f) > 0.05 and pixel_modulation(g) < 0.01
    assert np.sqrt(np.mean((g - t) ** 2)) < 0.4 * np.sqrt(np.mean((f - t) ** 2))
    assert np.array_equal(np.round(g), np.round(f))
    # monotone: the order within a pixel is kept, for localizations of one z and uncertainty
    u = np.linspace(-0.49, 0.49, 999)
    assert np.all(np.diff(m(u, np.full(999, 500.0), np.full(999, 10.0))) > 0)


def test_the_smooth_map_leaves_uniform_places_exactly_where_they_are():
    rng = np.random.default_rng(2)
    n = 200_000
    t = rng.uniform(0, 40, n)
    m = pe.fit_smooth_axis(t, rng.uniform(-600, 600, n), rng.uniform(5, 20, n))
    assert m.harmonics_kept == 0
    assert np.array_equal(m(t, np.zeros(n), np.full(n, 10.0)), t)


def test_the_smooth_map_learnt_on_one_half_flattens_the_other_and_survives_a_round_trip_through_a_dict():
    _, f, z, s = _pulled_by_z(300_000, seed=3)
    a = np.arange(len(f)) % 2 == 0
    eq = pe.fit_smooth(f[a], f[a], z[a], s[a], s[a])
    again = pe.SmoothEqualiser.from_dict(eq.to_dict())
    x, _ = again(f[~a], f[~a], z[~a], s[~a], s[~a])
    assert pixel_modulation(x) < 0.015 < pixel_modulation(f[~a])
    assert np.array_equal(x, eq(f[~a], f[~a], z[~a], s[~a], s[~a])[0])


def test_a_2d_fit_without_z_or_uncertainties_is_equalised_by_one_map():
    _, f, _, _ = _pulled_by_z(200_000, seed=4)
    n = len(f)
    m = pe.fit_smooth_axis(f, np.zeros(n), np.ones(n))
    assert m.a.shape[0] == 1 and pixel_modulation(m(f, np.zeros(n), np.ones(n))) < 0.01
