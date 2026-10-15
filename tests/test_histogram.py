"""Analysis/Measure/Histogram: any column, per layer, its numbers, per region."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.histogram import bin_edges, mode, value_range
from smappy.session import Session

PLUGIN = "Analysis/Measure/Histogram"


def normal_table(n=20000, mean=-120.0, sd=80.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(mean, sd, n)
    z[:5] = 1e6                                          # failed fits, far off
    return Localizations({"x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n),
                          "frame": rng.integers(0, 300, n), "z_nm": z,
                          "photons": rng.exponential(1000, n) + 100,
                          "xy_err_nm": np.full(n, 10.0)})


def run(locs_or_ctx, **settings):
    plugin = plugins.get(PLUGIN)
    ctx = locs_or_ctx if isinstance(locs_or_ctx, Context) else Context(locs=locs_or_ctx)
    return plugin().run(ctx, plugin.Settings(**settings))


def test_the_numbers_of_a_normal_column_come_back_without_its_outliers():
    result = run(normal_table(), column="z_nm")
    s = result.data["layers"][0].stats
    assert abs(s["mean"] + 120) < 3 and abs(s["median"] + 120) < 3
    assert abs(s["std"] - 80) < 3
    assert abs(s["mode"] + 120) < 20                     # a quarter of a sigma
    assert s["outside"] >= 5                             # the 1e6 ones
    assert result.values["z_nm_median"] == s["median"]


def test_the_range_leaves_off_the_extremes_or_takes_all_or_what_is_typed():
    values = [np.concatenate([np.arange(1000.0), [1e6]])]
    lo, hi = value_range(values, "quantile", 0.001)
    assert hi < 1000 and lo >= 0
    assert value_range(values, "all") == (0.0, 1e6)
    assert value_range(values, "fixed", lo=10.0) == (10.0, 1e6)
    assert value_range([np.array([5.0, 5.0])], "all") == (4.5, 5.5)


def test_whole_numbers_get_a_bin_each_and_the_rest_freedman_diaconis():
    frames = [np.arange(300) % 50]
    edges = bin_edges(frames, 0, 49)
    assert np.allclose(np.diff(edges), 1.0) and edges[0] == -0.5 and len(edges) == 51
    continuous = [np.random.default_rng(1).normal(0, 1, 100000)]
    n = len(bin_edges(continuous, -4, 4)) - 1
    assert 20 <= n <= 200
    few = [np.array([0.1, 0.2, 0.3])]
    assert len(bin_edges(few, 0.1, 0.3)) - 1 == 20       # never fewer than 20
    assert np.allclose(np.diff(bin_edges(continuous, 0, 1, width=0.25)), 0.25)
    with pytest.raises(ValueError, match="bins"):
        bin_edges(continuous, 0, 1, width=1e-9)


def test_the_mode_is_the_vertex_near_the_tallest_bin():
    edges = np.arange(11.0)
    counts = np.array([0, 1, 3, 8, 10, 8, 3, 1, 0, 0])   # symmetric about 4.5
    assert mode(counts, edges) == pytest.approx(4.5, abs=1e-6)
    assert np.isnan(mode(np.zeros(10), edges))


def test_each_visible_layer_is_its_own_histogram_on_the_same_bins():
    session = Session(normal_table())
    session.show_grouped(0, False)
    second = session.add_layer(like=0)
    second.set_bound("photons", 2000.0, None)            # the bright ones only
    result = run(session.context(), column="photons")
    first, bright = result.data["layers"]
    assert first.counts.shape == bright.counts.shape
    assert bright.stats["n"] < first.stats["n"]
    assert bright.stats["median"] > first.stats["median"]
    assert result.values["photons_n_L2"] == bright.stats["n"]


def test_a_missing_column_is_named_with_the_ones_there_are():
    with pytest.raises(ValueError, match="no column 'prob'.*photons"):
        run(normal_table(), column="prob")
    with pytest.raises(ValueError, match="choose a column"):
        run(normal_table())


def test_it_runs_per_cluster():
    locs = normal_table(n=3000)
    locs.columns["cluster_id"] = (np.arange(3000) % 3 + 1).astype(np.int32)
    locs.columns["z_nm"][locs.columns["cluster_id"] == 2] += 500
    session = Session(locs)
    session.show_grouped(0, False)
    step = plugins.get(PLUGIN)
    from smappy.roi_manager.pipeline import Step
    table = session.evaluate_regions(
        "cluster_id", [Step(label="z", path=PLUGIN, plugin=step(),
                            settings=step.Settings(column="z_nm"))])["z"]
    medians = table.columns()["z_nm_median"]
    assert medians[1] - medians[0] == pytest.approx(500, abs=30)


def test_the_form_shows_the_quantile_or_the_bounds_as_the_range_needs():
    plugin = plugins.get(PLUGIN)
    describe = plugin().describe
    assert describe(None, plugin.Settings()).shown == {"quantile": True, "lo": False,
                                                       "hi": False}
    assert describe(None, plugin.Settings(range="fixed")).shown == \
        {"quantile": False, "lo": True, "hi": True}
