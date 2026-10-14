"""Neighbourhood: the neighbours within r, a statistic of a column around each
localization and the Voronoi density, written onto the around layer."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.mathparse import recipes
from smappy.plugins import neighbourhood as nb
from smappy.plugins.neighbourhood import (Neighbourhood, count_neighbours,
                                          neighbour_statistic, voronoi_density)
from smappy.session import Session

SIDE = 20_000.0          # nm


def uniform(n, seed=0, side=SIDE, **extra):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, side, (n, 2))
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(n, dtype=np.int64),
                          "photons": rng.gamma(4, 400, n), **extra}, {})


def inner(xy, margin=300.0, side=SIDE):
    """Away from the edges, where a circle is whole."""
    return ((xy > margin) & (xy < side - margin)).all(axis=1)


def test_a_uniform_field_has_pi_r_squared_rho_neighbours():
    n, r = 40_000, 80.0
    xy = np.random.default_rng(0).uniform(0, SIDE, (n, 2))
    counts = count_neighbours(xy, xy, r, same=np.arange(n))
    keep = inner(xy)
    expected = (n - 1) / SIDE ** 2 * np.pi * r ** 2         # self not counted
    assert counts[keep].mean() == pytest.approx(expected, rel=0.02)
    # Poisson: the variance is the mean
    assert counts[keep].var() == pytest.approx(expected, rel=0.08)


def test_the_gaussian_weight_integrates_to_two_pi_sigma_squared_rho():
    n, sigma = 40_000, 40.0
    xy = np.random.default_rng(1).uniform(0, SIDE, (n, 2))
    weights = count_neighbours(xy, xy, sigma, gaussian=True, same=np.arange(n))
    rho = (n - 1) / SIDE ** 2
    # the integral of exp(-r^2/2s^2) over the plane, cut at 3 sigma
    expected = 2 * np.pi * sigma ** 2 * rho * (1 - np.exp(-4.5))
    assert weights[inner(xy)].mean() == pytest.approx(expected, rel=0.03)


def test_a_cylinder_counts_only_the_neighbours_within_its_half_height():
    n, r, h, depth = 40_000, 100.0, 100.0, 1000.0
    rng = np.random.default_rng(2)
    p = np.column_stack([rng.uniform(0, SIDE, (n, 2)), rng.uniform(0, depth, n)])
    counts = count_neighbours(p, p, r, h, same=np.arange(n))
    keep = inner(p[:, :2]) & (p[:, 2] > h) & (p[:, 2] < depth - h)
    expected = (n - 1) / (SIDE ** 2 * depth) * np.pi * r ** 2 * 2 * h
    assert counts[keep].mean() == pytest.approx(expected, rel=0.03)


def test_the_voronoi_density_of_a_uniform_field_is_its_density():
    n = 40_000
    xy = np.random.default_rng(3).uniform(0, SIDE, (n, 2))
    density = voronoi_density(xy)                           # per um^2
    rho_um2 = n / (SIDE / 1000) ** 2
    keep = inner(xy, 1000)
    assert np.isfinite(density[keep]).all()
    # the mean cell area is 1/rho, so the harmonic mean of the densities is rho
    assert 1 / np.mean(1 / density[keep]) == pytest.approx(rho_um2, rel=0.02)
    assert np.isnan(density).any()                          # open cells at the hull


def test_points_at_one_position_share_a_cell_and_each_gets_their_number():
    xy = np.random.default_rng(4).uniform(0, 1000, (300, 2))
    alone = voronoi_density(xy)
    twice = voronoi_density(np.vstack([xy, xy[:5]]))
    closed = np.isfinite(alone[:5])
    assert np.allclose(twice[:5][closed], 2 * alone[:5][closed])
    assert np.allclose(twice[-5:][closed], twice[:5][closed])


def test_a_statistic_follows_a_known_gradient():
    n, r = 30_000, 150.0
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, SIDE, (n, 2))
    slope = 0.05                                            # photons per nm in x
    photons = 500 + slope * xy[:, 0] + rng.normal(0, 50, n)
    for statistic in ("mean", "median"):
        got = neighbour_statistic(xy, xy, photons, r, statistic, same=np.arange(n))
        keep = inner(xy)
        # a disc is symmetric, so the mean over it is the value at its centre
        residual = got[keep] - (500 + slope * xy[keep, 0])
        assert abs(residual.mean()) < 2
        assert residual.std() < 50 / np.sqrt(np.pi * r ** 2 * n / SIDE ** 2) * 1.5


def test_each_statistic_is_numpys_over_the_neighbours():
    rng = np.random.default_rng(11)
    a = np.column_stack([rng.uniform(0, 2000, (400, 2)), rng.uniform(0, 400, 400)])
    b = np.column_stack([rng.uniform(0, 2000, (300, 2)), rng.uniform(0, 400, 300)])
    values = rng.normal(size=len(b))
    values[::17] = np.nan                                   # left out
    lateral = ((a[:, None, :2] - b[None, :, :2]) ** 2).sum(-1) <= 120 ** 2
    within = lateral & (np.abs(a[:, None, 2] - b[None, :, 2]) <= 80)
    numpy = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max,
             "std": np.std, "quantile": lambda v: np.quantile(v, 0.8)}
    for statistic, f in numpy.items():
        for z_half, near in ((None, lateral), (80.0, within)):
            got = neighbour_statistic(a if z_half else a[:, :2], b if z_half else b[:, :2],
                                      values, 120, statistic, 0.8, z_half)
            ok = near & np.isfinite(values)[None, :]
            want = np.array([f(values[row]) if row.any() else np.nan for row in ok])
            assert np.allclose(got, want, equal_nan=True), statistic


def test_a_localization_alone_has_no_neighbours_and_its_own_value():
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [1010.0, 0.0]])
    same = np.arange(3)
    assert list(count_neighbours(xy, xy, 50, same=same)) == [0, 1, 1]
    values = np.array([7.0, 1.0, 3.0])
    assert list(neighbour_statistic(xy, xy, values, 50, "max", same=same)) == [7, 3, 3]


def test_the_voronoi_cells_in_tiles_are_those_of_one_triangulation():
    rng = np.random.default_rng(6)
    # uniform, a dense cluster, and a sparse patch beyond an empty gap
    xy = np.unique(np.vstack([rng.uniform(0, 20_000, (20_000, 2)),
                              rng.normal(5000, 100, (10_000, 2)),
                              rng.uniform(30_000, 40_000, (300, 2))]), axis=0)
    whole = nb._cell_areas(xy)[0]
    tiled = nb.voronoi_areas(xy, tile=2000)
    assert np.array_equal(np.isnan(whole), np.isnan(tiled))
    closed = np.isfinite(whole)
    assert np.allclose(tiled[closed], whole[closed], rtol=1e-6)


# ---------------------------------------------------------------- the plugin

def test_the_plugin_writes_a_recipe_column_and_nan_outside_the_filter():
    locs = uniform(20_000)
    session = Session(locs)
    session.show_grouped(0, False)
    session.layers[0].set_bound("photons", 1500.0, None)
    result = session.run(Neighbourhood(), Neighbourhood.Settings(radius_nm=100))
    assert result.data["column"] == "density_r100"
    values = np.asarray(session.locs["density_r100"])
    bright = np.asarray(session.locs["photons"]) >= 1500
    assert np.isnan(values[~bright]).all()
    assert np.isfinite(values[bright]).all()
    # the neighbours counted are the filtered ones too
    xy = np.column_stack([locs["x_nm"], locs["y_nm"]])
    keep = bright & inner(xy)
    expected = (bright.sum() - 1) / SIDE ** 2 * np.pi * 100 ** 2
    assert values[keep].mean() == pytest.approx(expected, rel=0.03)
    rules = {r["field"]: r["grouped"] for r in recipes(session.locs)}
    assert rules["density_r100"] == "mean"


def test_two_layers_give_a_cross_density_and_never_across_files():
    n = 20_000
    rng = np.random.default_rng(7)
    channel = rng.integers(1, 3, n)
    files = (rng.uniform(size=n) < 0.5).astype(np.int64)
    locs = uniform(n, seed=8, channel=channel, filenumber=files)
    session = Session(locs)
    session.add_layer()
    for i in (0, 1):
        session.show_grouped(i, False)
        session.layers[i].set_files(None)                   # every file
        session.layers[i].set_bound("channel", i + 1, i + 1)
    session.run(Neighbourhood(), Neighbourhood.Settings(
        around=1, neighbours_from=2, radius_nm=150))
    name = "density_r150_from2"
    values = np.asarray(session.locs[name])
    assert np.isnan(values[channel == 2]).all()
    xy = np.column_stack([locs["x_nm"], locs["y_nm"]])
    keep = (channel == 1) & inner(xy)
    for f in (0, 1):
        # only the other channel's localizations of the same file count
        n_b = ((channel == 2) & (files == f)).sum()
        expected = n_b / SIDE ** 2 * np.pi * 150 ** 2
        assert values[keep & (files == f)].mean() == pytest.approx(expected, rel=0.04)


def test_on_merged_blinks_the_blinks_are_counted_and_their_value_goes_onto_them():
    blinks = uniform(10_000, seed=9)
    n = len(blinks)
    # each blink on three frames, the localizations a few nm apart
    rng = np.random.default_rng(10)
    xy = np.repeat(np.column_stack([blinks["x_nm"], blinks["y_nm"]]), 3, axis=0)
    xy = xy + rng.normal(0, 2, xy.shape)
    frame = np.repeat(np.arange(n) * 10, 3) + np.tile([0, 1, 2], n)
    locs = Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1], "frame": frame,
                          "photons": np.full(3 * n, 1000.0),
                          "xy_err_nm": np.full(3 * n, 5.0)}, {})
    settings = Neighbourhood.Settings(radius_nm=200)
    grouped = Session(Localizations(dict(locs.columns), {}))
    grouped.show_grouped(0, True)
    grouped.run(Neighbourhood(), settings)
    on_blinks = np.asarray(grouped.locs["density_r200"])
    # every localization of a blink has the blink's value
    assert np.allclose(on_blinks.reshape(-1, 3), on_blinks.reshape(-1, 3)[:, :1])
    keep = inner(xy)
    expected = (n - 1) / SIDE ** 2 * np.pi * 200 ** 2
    assert on_blinks[keep].mean() == pytest.approx(expected, rel=0.04)

    ungrouped = Session(Localizations(dict(locs.columns), {}))
    ungrouped.show_grouped(0, False)
    ungrouped.run(Neighbourhood(), settings)
    on_locs = np.asarray(ungrouped.locs["density_r200"])
    # three localizations per blink: three times the neighbours, and the
    # blink's own other two
    assert on_locs[keep].mean() == pytest.approx(3 * expected + 2, rel=0.04)


def test_the_voronoi_density_is_of_one_layer_only():
    session = Session(uniform(1000))
    session.add_layer()
    with pytest.raises(ValueError, match="one layer"):
        session.run(Neighbourhood(), Neighbourhood.Settings(
            quantity="voronoi", around=1, neighbours_from=2))
    session.run(Neighbourhood(), Neighbourhood.Settings(quantity="voronoi"))
    assert "voronoi_density" in session.locs


def test_a_column_of_the_localizations_own_is_never_overwritten():
    session = Session(uniform(500))
    with pytest.raises(ValueError, match="own"):
        session.run(Neighbourhood(), Neighbourhood.Settings(name="photons"))


def test_the_form_shows_only_the_quantitys_settings():
    plugin = Neighbourhood()
    count = plugin.describe(None, Neighbourhood.Settings())
    assert count.shown["gaussian"] and not count.shown["column"]
    stat = plugin.describe(None, Neighbourhood.Settings(quantity="statistic",
                                                        statistic="quantile"))
    assert stat.shown["column"] and stat.shown["quantile"] and not stat.shown["gaussian"]
    vor = plugin.describe(None, Neighbourhood.Settings(quantity="voronoi"))
    assert not vor.shown["radius_nm"]
    assert vor.hints["name"] == "voronoi_density"


def test_the_estimate_asks_before_a_long_run():
    s = Neighbourhood.Settings(quantity="statistic", radius_nm=500)
    assert nb.expected_seconds(5_000_000, 5_000_000, 40_000.0 ** 2, s) > nb.ASK_ABOVE_S
    assert nb.expected_seconds(100_000, 100_000, 40_000.0 ** 2,
                               Neighbourhood.Settings()) < nb.ASK_ABOVE_S


@pytest.mark.parametrize("quantity", ["count", "voronoi"])
def test_a_run_reports_its_progress_and_a_stop_writes_nothing(monkeypatch, quantity):
    import threading
    from smappy.plugins import Stopped
    monkeypatch.setattr(nb, "CHUNK", 1000)                  # many steps
    monkeypatch.setattr(nb, "VORONOI_TILE", 1000)
    session = Session(uniform(10_000))
    settings = Neighbourhood.Settings(quantity=quantity)
    lines = []
    Neighbourhood().run(session.context(0, progress=lines.append), settings)
    percents = [int(line.rsplit(" ", 1)[1].rstrip("%")) for line in lines if "%" in line]
    assert len(percents) > 5 and percents == sorted(percents) and percents[-1] == 100

    stop = threading.Event()

    def press(line):                                        # Stop, at the first report
        stop.set()

    ctx = session.context(0, progress=press, stop=stop)
    with pytest.raises(Stopped, match="nothing written"):
        Neighbourhood().run(ctx, settings)
    assert not any(name.startswith(("density", "voronoi")) for name in session.locs.keys())
