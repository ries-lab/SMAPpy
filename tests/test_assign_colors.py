"""Colours from the photon split: the histogram, the cut, and the posterior."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Selection
from smappy.plugins.assign_colors import (AssignColorSettings, assign_by_minima,
                                          assign_by_probability, find_modes,
                                          posteriors, ratios)
from smappy.session import Session

RATIOS = (-0.6, 0.6)            # two dyes, well apart
PHOTONS = 2000.0
# and two that are not: 0.15 apart from the middle with 150 photons is about
# 1.9 sigma each way, so the modes genuinely overlap and a decision costs
# something -- which is the only regime in which the two methods differ
OVERLAP = dict(ratios_=(-0.15, 0.15), photons=150.0)


def simulate(n=4000, ratios_=RATIOS, photons=PHOTONS, seed=0, errors=True):
    """Two species split between two channels, with honest shot noise."""
    rng = np.random.default_rng(seed)
    species = rng.integers(0, len(ratios_), n)
    rho = np.array(ratios_)[species]
    total = rng.poisson(photons, n).astype(float)
    n1 = rng.binomial(total.astype(int), (1 + rho) / 2).astype(float)
    n2 = total - n1
    columns = {"frame": np.arange(n), "x_nm": rng.uniform(0, 1e4, n),
               "y_nm": rng.uniform(0, 1e4, n),
               "photons": total, "photons_ch0": n1, "photons_ch1": n2}
    if errors:
        columns["photons_err_ch0"] = np.sqrt(np.maximum(n1, 1))
        columns["photons_err_ch1"] = np.sqrt(np.maximum(n2, 1))
    return Localizations(columns, {"units": "nm"}), species + 1


def test_the_ratio_and_its_effective_photons_match_the_shot_noise():
    locs, _ = simulate()
    values = ratios(locs)
    assert values.fitted_errors
    assert np.all(np.abs(values.r) <= 1)
    # Poisson errors are what the effective photon number is calibrated on, so
    # it must come back as the total that was actually detected
    assert np.allclose(values.n_eff, values.total, rtol=0.05)

    # ...and without the error columns it is the total by definition
    plain, _ = simulate(errors=False)
    bare = ratios(plain)
    assert not bare.fitted_errors
    assert np.array_equal(bare.n_eff, bare.total)


def test_the_modes_are_found_where_the_species_are():
    locs, _ = simulate()
    modes = find_modes(ratios(locs).r, colors=2)
    assert np.allclose(modes.maxima, RATIOS, atol=0.03)
    assert abs(modes.minima[0]) < 0.05           # the dip sits between them
    assert np.allclose(modes.prior, 0.5, atol=0.05)


def test_expected_ratios_replace_the_maxima_but_not_the_boundary():
    locs, _ = simulate()
    modes = find_modes(ratios(locs).r, colors=2, expected=(-0.5, 0.7))
    assert np.allclose(modes.maxima, (-0.5, 0.7))
    assert abs(modes.minima[0]) < 0.05


def test_the_exclusion_zone_leaves_the_middle_unassigned():
    locs, truth = simulate()
    values = ratios(locs)
    modes = find_modes(values.r, colors=2)
    channel = assign_by_minima(values.r, modes, exclusion=0.0)
    assert (channel == truth).mean() > 0.99
    assert (channel == 0).sum() == 0

    excluded = assign_by_minima(values.r, modes, exclusion=0.1)
    near = np.abs(values.r - modes.minima[0]) < 0.1
    assert np.array_equal(excluded == 0, near)

    # dim enough that the modes overlap: throwing the middle away is what
    # buys the accuracy back, which is the whole of the exclusion zone's job
    dim, truth = simulate(n=20000, seed=5, **OVERLAP)
    values = ratios(dim)
    modes = find_modes(values.r, colors=2)
    flat = assign_by_minima(values.r, modes, exclusion=0.0)
    excluded = assign_by_minima(values.r, modes, exclusion=0.1)
    kept = excluded > 0
    assert 0 < (~kept).mean() < 0.5
    assert (excluded[kept] == truth[kept]).mean() > (flat == truth).mean()


def test_the_posterior_follows_the_photons_not_only_the_ratio():
    """The same r, twice the photons: the dim one is rejected, the bright one not."""
    modes = find_modes(np.linspace(-1, 1, 500), colors=2, expected=(-0.6, 0.6))
    r = np.array([0.02, 0.02])          # just off the boundary, either way
    channel, p = assign_by_probability(r, np.array([30.0, 5000.0]), modes,
                                       crosstalk=0.05, use_prior=False)
    assert p[1] > p[0]
    assert channel[0] == 0 and channel[1] == 2


def test_the_crosstalk_budget_bounds_the_mistakes():
    locs, truth = simulate(n=20000, seed=3, **OVERLAP)
    values = ratios(locs)
    modes = find_modes(values.r, colors=2)
    for crosstalk in (0.2, 0.05, 0.01):
        channel, p = assign_by_probability(values.r, values.n_eff, modes,
                                           crosstalk=crosstalk)
        kept = channel > 0
        wrong = (channel[kept] != truth[kept]).mean()
        assert wrong <= crosstalk            # the bound holds...
        assert np.all(p[kept] >= 1 - crosstalk)
    # ...and a tighter budget keeps fewer and is righter
    loose = assign_by_probability(values.r, values.n_eff, modes, crosstalk=0.2)[0]
    tight = assign_by_probability(values.r, values.n_eff, modes, crosstalk=0.01)[0]
    assert (tight > 0).sum() < (loose > 0).sum()
    assert ((tight > 0) & (loose == 0)).sum() == 0     # a subset, not a shuffle


def test_the_posteriors_are_a_partition_of_one():
    modes = find_modes(np.linspace(-1, 1, 500), colors=3, expected=(-0.6, 0, 0.6))
    p = posteriors(np.linspace(-1, 1, 21), np.full(21, 1000.0), modes)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_the_plugin_writes_a_channel_column_and_the_preview_does_not():
    locs, truth = simulate(n=20000, seed=7, **OVERLAP)
    plugin = plugins.get("Analysis/Dual-Color/AssignColors")()
    result = plugin(locs, Selection.all(len(locs)),
                    AssignColorSettings(mode="minima", exclusion=0.05))
    assert result.locs is not None
    channel = result.locs["channel"]
    assert channel.dtype == np.int32 and set(np.unique(channel)) == {0, 1, 2}
    kept = channel > 0
    assert (channel[kept] == truth[kept]).mean() > (channel == truth).mean()
    assert "color_ratio" in result.locs and "channel_p" in result.locs
    assert "2 colours" in result.text and result.plot is not None

    session = Session(locs)
    preview = plugin.preview(session.context(), AssignColorSettings())
    assert preview.locs is None and preview.plot is not None
    session.run(plugin, AssignColorSettings(mode="probabilistic"))
    assert "channel" in session.locs                  # run does change it

    # a plugin with a preview that is not of a frame gets no frame number
    cls = plugins.get("Analysis/Dual-Color/AssignColors")
    assert cls.has_preview() and not cls.preview_wants_frame()


def test_the_summary_measures_a_species_wider_than_shot_noise():
    """`spread` is meant to be set by looking at what the preview prints."""
    rng = np.random.default_rng(1)
    n = 40000
    species = rng.integers(0, 2, n)
    # the same two dyes, but each molecule's own split wanders by 0.05
    rho = np.array([-0.15, 0.15])[species] + rng.normal(0, 0.05, n)
    total = rng.poisson(150, n)
    n1 = rng.binomial(total, np.clip((1 + rho) / 2, 0, 1)).astype(float)
    locs = Localizations({"photons_ch0": n1, "photons_ch1": total - n1}, {})

    plugin = plugins.get("Analysis/Dual-Color/AssignColors")()
    blind = plugin(locs, Selection.all(n),
                   AssignColorSettings(mode="probabilistic"))
    width = [line for line in blind.text.splitlines() if "wide against" in line]
    assert len(width) == 1
    assert "spread = 0.04" in width[0] or "spread = 0.05" in width[0]

    # and taking the spread into account brings the promise back in line with
    # what actually happens: without it the model is sure of more than it knows
    def crosstalk(spread):
        result = plugin(locs, Selection.all(n),
                        AssignColorSettings(mode="probabilistic", spread=spread))
        channel = result.locs["channel"]
        kept = channel > 0
        return (channel[kept] != species[kept] + 1).mean()

    assert crosstalk(0.05) < crosstalk(0.0)


def test_a_table_without_two_channels_says_so():
    locs = Localizations({"photons": np.ones(10)}, {})
    with pytest.raises(ValueError, match="no per-channel photon columns"):
        ratios(locs)


def test_asking_for_more_colours_than_there_are_says_so():
    locs, _ = simulate(ratios_=(0.6,))
    with pytest.raises(ValueError, match="not 3"):
        find_modes(ratios(locs).r, colors=3)


def test_the_figure_draws(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    locs, _ = simulate()
    plugin = plugins.get("Analysis/Dual-Color/AssignColors")()
    for mode in ("minima", "probabilistic"):
        result = plugin(locs, Selection.all(len(locs)),
                        AssignColorSettings(mode=mode))
        fig, ax = plt.subplots()
        result.plot(ax)
        fig.savefig(tmp_path / f"{mode}.png")
        plt.close(fig)
