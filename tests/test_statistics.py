"""The photon, precision and on-time distributions, and what is read off them."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Selection
from smappy.plugins.statistics import (HALF_MAX_U, LocalizationStatistics,
                                       MODE_OVER_SIGMA_C, StatisticsSettings,
                                       exponential_mean, histogram_landmarks,
                                       ontime_decay, ontime_distribution,
                                       photon_decay, photon_distribution,
                                       precision_density, precision_distribution,
                                       precision_model, statistics)

N0 = 1200.0                 # mean photons
PSF_SCALE = 300.0           # S in sigma = S / sqrt(N), so sigma_c = 8.66 nm
SIGMA_C = PSF_SCALE / np.sqrt(N0)
TAU = 2.5                   # frames


def simulate(n=200_000, threshold=150.0, seed=0):
    """Exponential photons above a detection threshold, and what follows."""
    rng = np.random.default_rng(seed)
    photons = rng.exponential(N0, n)
    photons = photons[photons > threshold]
    sigma = PSF_SCALE / np.sqrt(photons)
    on_time = rng.geometric(1 - np.exp(-1 / TAU), len(photons)).astype(float)
    return Localizations(
        {"frame": np.arange(len(photons), dtype=np.int64),
         "x_nm": rng.uniform(0, 1e4, len(photons)),
         "y_nm": rng.uniform(0, 1e4, len(photons)),
         "photons": photons, "loc_precision_nm": sigma,
         "loc_precision_z_nm": 2.5 * sigma, "n_in_group": on_time},
        {"units": "nm"})


# ------------------------------------------------------------------ estimators

def test_the_half_maximum_constant_solves_its_own_equation():
    def curve(u):
        return u ** 1.5 * np.exp(-u)

    assert curve(HALF_MAX_U) == pytest.approx(0.5 * curve(1.5), rel=1e-6)
    assert HALF_MAX_U > 1.5                      # the root above the peak


def test_the_model_is_normalized_and_peaks_where_the_closed_form_says():
    a = SIGMA_C ** 2
    sigma = np.linspace(1e-3, 200 * SIGMA_C, 200_000)
    density = precision_density(sigma, a)
    area = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(sigma)))
    assert area == pytest.approx(1.0, rel=1e-3)
    assert sigma[np.argmax(density)] == pytest.approx(SIGMA_C * MODE_OVER_SIGMA_C,
                                                      rel=1e-3)


def test_an_exponential_mean_survives_a_cut_at_both_ends():
    rng = np.random.default_rng(1)
    values = rng.exponential(10.0, 200_000)
    assert exponential_mean(values) == pytest.approx(10.0, rel=0.02)
    # the same sample seen only between 2 and 25: the estimate is of the whole
    # exponential, not of what is left of it
    assert exponential_mean(values, lo=2.0, hi=25.0) == pytest.approx(10.0, rel=0.05)
    assert values[(values >= 2) & (values <= 25)].mean() < 9.5   # the naive one


def test_the_photon_fit_finds_the_decay_constant_above_the_threshold():
    locs = simulate()
    stats = photon_decay(locs["photons"], start=200.0)
    assert stats["n0"] == pytest.approx(N0, rel=0.03)
    # and the automatic start, at the maximum of the histogram, does as well
    assert photon_distribution(locs["photons"]).stats["n0"] == pytest.approx(N0, rel=0.05)


def test_the_precision_fit_finds_sigma_c_and_both_landmarks():
    locs = simulate()
    stats = precision_model(locs["loc_precision_nm"])
    assert stats["sigma_c"] == pytest.approx(SIGMA_C, rel=0.02)
    assert stats["max"] == pytest.approx(SIGMA_C * np.sqrt(2 / 3), rel=0.02)
    assert stats["rising"] == pytest.approx(SIGMA_C / np.sqrt(HALF_MAX_U), rel=0.02)
    assert stats["rising"] < stats["max"] < stats["median"]

    # the histogram is read independently of the fit, and agrees with it
    dist = precision_distribution(locs["loc_precision_nm"])
    assert dist.stats["histogram_max"] == pytest.approx(stats["max"], rel=0.05)
    assert dist.stats["histogram_rising"] == pytest.approx(stats["rising"], rel=0.05)


def test_the_precision_fit_ignores_a_handful_of_absurdly_good_rows():
    """A few sub-nanometre rows are bad fits, not good localizations, and the
    estimator weighs every row by 1/sigma^2 -- so the trimming is what keeps
    them from carrying the answer."""
    sigma = PSF_SCALE / np.sqrt(np.random.default_rng(2).exponential(N0, 100_000))
    spoilt = np.concatenate([sigma, np.full(200, 0.05)])
    assert precision_model(spoilt)["sigma_c"] == pytest.approx(SIGMA_C, rel=0.05)


def test_the_binned_precision_fit_survives_what_the_likelihood_does_not():
    """Five per cent of collapsed rows is past any trimming: the likelihood
    comes back an order of magnitude too small, the histogram fit does not."""
    rng = np.random.default_rng(3)
    sigma = PSF_SCALE / np.sqrt(rng.exponential(N0, 100_000))
    spoilt = np.concatenate([sigma, rng.uniform(0.01, 1.0, 5_000)])
    assert precision_model(spoilt)["sigma_c"] == pytest.approx(SIGMA_C, rel=0.05)
    assert precision_model(spoilt, method="mle")["sigma_c"] < 0.5 * SIGMA_C


def test_the_binned_precision_fit_agrees_with_the_likelihood_on_clean_data():
    """Robustness that cost accuracy would be no bargain."""
    sigma = PSF_SCALE / np.sqrt(np.random.default_rng(4).exponential(N0, 100_000))
    binned = precision_model(sigma)["sigma_c"]
    assert binned == pytest.approx(SIGMA_C, rel=0.02)
    assert binned == pytest.approx(precision_model(sigma, method="mle")["sigma_c"],
                                   rel=0.02)


def test_the_precision_curve_is_drawn_at_the_height_it_was_fitted_at():
    """The fitted amplitude, not the row count: the histogram stops at the
    99.5th percentile and the curve must sit on the bars that are shown."""
    sigma = PSF_SCALE / np.sqrt(np.random.default_rng(5).exponential(N0, 50_000))
    dist = precision_distribution(sigma)
    peak_bin = dist.counts.max()
    assert dist.curve is not None
    assert dist.curve[1].max() == pytest.approx(peak_bin, rel=0.1)


def test_the_on_time_fit_finds_the_lifetime():
    locs = simulate()
    stats = ontime_decay(locs["n_in_group"])
    assert stats["tau"] == pytest.approx(TAU, rel=0.05)
    dist = ontime_distribution(locs["n_in_group"], exposure_ms=20.0)
    assert dist.log_y and dist.edges[0] == 0.5      # one bin per frame count
    assert "ms" in dist.summary


def test_a_single_frame_on_time_has_no_decay_to_fit():
    stats = ontime_decay(np.ones(1000))
    assert np.isnan(stats["tau"]) and stats["mean"] == 1.0


def test_the_landmarks_are_read_off_a_histogram_between_its_bins():
    edges = np.linspace(0, 10, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.exp(-0.5 * ((centers - 4.03) / 1.0) ** 2) * 1000
    found = histogram_landmarks(counts, edges)
    assert found["max"] == pytest.approx(4.03, abs=0.05)
    # the half maximum of a Gaussian is 1.177 sigma below its centre
    assert found["rising"] == pytest.approx(4.03 - 1.1774, abs=0.05)


# --------------------------------------------------------------------- plugin

def test_every_available_distribution_is_produced_in_reading_order():
    found = statistics(simulate(n=20_000))
    assert [d.key for d in found] == ["photons", "precision", "precision_z", "on_time"]
    assert all(d.counts.sum() > 0 and d.curve is not None for d in found)


def test_the_plugin_describes_the_selection_by_default_and_all_on_request():
    locs = simulate(n=20_000)
    mask = locs["photons"] > 2000                 # a filter the selection stands for
    plugin = LocalizationStatistics()
    from smappy.plugins import Context
    ctx = Context(locs=locs, selection=Selection(mask))

    selected = plugin.run(ctx, StatisticsSettings())
    assert selected.data["n"] == int(mask.sum())
    assert selected.data["stats"]["photons"]["median"] > 2000

    everything = plugin.run(ctx, StatisticsSettings(source="all"))
    assert everything.data["n"] == len(locs)
    assert everything.data["stats"]["photons"]["n0"] == pytest.approx(N0, rel=0.1)


def test_a_table_without_an_on_time_says_so_rather_than_linking_anything():
    locs = simulate(n=20_000)
    del locs.columns["n_in_group"]
    said = []
    from smappy.plugins import Context
    result = LocalizationStatistics().run(
        Context(locs=locs, progress=said.append), StatisticsSettings())
    assert "on_time" not in result.data["stats"]
    assert "grouped" in result.data["note"] and said


def test_a_table_with_nothing_to_describe_is_refused():
    locs = Localizations({"x_nm": np.zeros(50), "y_nm": np.zeros(50)}, {})
    from smappy.plugins import Context
    with pytest.raises(ValueError, match="no photons"):
        LocalizationStatistics().run(Context(locs=locs), StatisticsSettings())


def test_the_plugin_is_in_the_tree_and_draws_one_panel_per_distribution():
    assert "Analysis/Measure/Localization Statistics" in plugins.refs()
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    result = LocalizationStatistics()(simulate(n=20_000))
    figure = Figure()
    result.plot.draw_into(figure)
    assert result.plot.panels == 4 and len(figure.axes) == 4

    # the panels compose onto a page of small multiples, which is what
    # declaring them is for: a subfigure can be drawn in, not resized
    page = Figure()
    result.plot.draw_into(page.subfigures(1, 1, squeeze=False).ravel()[0])
    assert len(page.axes) == 4
