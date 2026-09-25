"""Line profiles: the projection, and the models fitted to it unbinned."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Context, Selection
from smappy.regions import Region
from smappy.plugins.line_profile import (LineProfile, LineProfileSettings,
                                         auto_bin, fit_models, fit_profile,
                                         line_ends, profiles, project)

WIDTH = 200.0
LENGTH = 1000.0
ROI = Region.line((0.0, 5000.0), (LENGTH, 5000.0), WIDTH)


def simulate(n, across, precision=10.0, seed=0, sigma=0.0):
    """Localizations in the ROI: ``across`` nm off the line, blurred.

    The structure is at `across` (a number, or one per localization); each
    localization is then displaced by its own precision, which is what the
    fit is asked to take out again.
    """
    rng = np.random.default_rng(seed)
    precision = np.full(n, precision, float) if np.isscalar(precision) else precision
    along = rng.uniform(0, LENGTH, n)
    offset = np.asarray(across, float) + rng.normal(0, sigma, n)
    x = along + rng.normal(0, precision)
    y = 5000.0 + offset + rng.normal(0, precision)
    return Localizations({
        "x_nm": x.astype(np.float32), "y_nm": y.astype(np.float32),
        "frame": np.arange(n, dtype=np.int64),
        "photons": np.full(n, 800.0, np.float32),
        "xy_err_nm": precision.astype(np.float32),
    }, {"units": "nm"})


def context(locs, roi=ROI):
    """What a session with this ROI drawn would hand a plugin."""
    ctx = Context(locs=locs,
                  selection=Selection(roi.mask(locs["x_nm"], locs["y_nm"]),
                                      roi=roi))

    class OneRoi:
        def __init__(self, region):
            self.roi = region

    ctx.session = OneRoi(roi)
    return ctx


# ------------------------------------------------------------ the geometry

def test_the_line_is_recovered_from_the_corners_of_its_roi():
    p0, p1, width = line_ends(ROI)
    assert np.allclose(p0, (0.0, 5000.0)) and np.allclose(p1, (LENGTH, 5000.0))
    assert width == WIDTH


def test_a_position_becomes_a_distance_along_and_a_distance_across():
    along, across = project([0.0, 100.0, 0.0], [0.0, 0.0, 30.0],
                            (0.0, 0.0), (100.0, 0.0))
    assert np.allclose(along, [0.0, 100.0, 0.0])
    assert np.allclose(across, [0.0, 0.0, 30.0])
    # a diagonal line: the projection is the rotation, not the coordinates
    along, across = project([10.0], [10.0], (0.0, 0.0), (10.0, 10.0))
    assert np.allclose(along, np.hypot(10, 10)) and np.allclose(across, 0.0)


def test_a_roi_that_is_not_a_line_is_refused_by_name():
    with pytest.raises(ValueError, match="rect"):
        line_ends(Region.rect(0, 0, 10, 10))
    with pytest.raises(ValueError, match="a line ROI"):
        LineProfile().run(context(simulate(100, 0.0), Region.rect(0, 0, 1e4, 1e4)),
                          LineProfileSettings())


def test_the_profiles_are_of_the_same_localizations_and_of_the_roi_window():
    locs = simulate(500, 0.0, seed=2)
    found = profiles(locs, ROI)
    assert found["across"].window == (-WIDTH / 2, WIDTH / 2)
    assert found["along"].window == (0.0, LENGTH)
    assert len(found["across"].values) == len(found["along"].values)
    assert np.all(np.abs(found["across"].values) <= WIDTH / 2)


def test_a_given_length_replaces_the_roi_and_drops_what_is_outside_it():
    """Two ROIs of different lengths compare directly when both are cut."""
    locs = simulate(2000, 0.0, seed=3)
    found = profiles(locs, ROI, length=400.0)
    assert found["along"].window == (300.0, 700.0)
    assert len(found["along"].values) < 2000
    assert np.all(found["along"].values >= 300.0)


# -------------------------------------------------------------- the fitting

def test_a_gaussian_profile_comes_back_with_the_width_it_was_made_with():
    locs = simulate(4000, 0.0, sigma=12.0, seed=4)
    profile = profiles(locs, ROI)["across"]
    fit = fit_profile(profile.values, profile.precision, model="gauss",
                      window=profile.window)
    assert fit.values()["sigma"] == pytest.approx(12.0, abs=1.5)
    assert abs(fit.values()["centre"]) < 2.0


def test_the_localization_precision_is_what_separates_structure_from_blur():
    """The point of the unbinned fit: what is fitted is the structure.

    Without the precisions the same data gives the width of the *picture*,
    sqrt(s^2 + sigma^2), which is a different and larger number.
    """
    locs = simulate(4000, 0.0, precision=15.0, sigma=10.0, seed=5)
    profile = profiles(locs, ROI)["across"]
    with_it = fit_profile(profile.values, profile.precision, model="gauss",
                          window=profile.window)
    without = fit_profile(profile.values, None, model="gauss",
                          window=profile.window)
    assert with_it.values()["sigma"] == pytest.approx(10.0, abs=1.5)
    assert without.values()["sigma"] == pytest.approx(np.hypot(10, 15), abs=1.5)


def test_two_structures_give_back_the_distance_between_them():
    n = 4000
    rng = np.random.default_rng(6)
    across = np.where(rng.random(n) < 0.5, -25.0, 25.0)
    locs = simulate(n, across, precision=8.0, sigma=6.0, seed=7)
    profile = profiles(locs, ROI)["across"]
    fit = fit_profile(profile.values, profile.precision, model="two_gauss",
                      window=profile.window)
    assert fit.values()["distance"] == pytest.approx(50.0, abs=3.0)
    assert fit.values()["sigma"] == pytest.approx(6.0, abs=2.0)


def test_an_edge_is_found_where_the_localizations_stop():
    n = 4000
    rng = np.random.default_rng(8)
    across = rng.uniform(-WIDTH / 2, 20.0, n)      # labelled up to +20 nm
    locs = simulate(n, across, precision=8.0, seed=9)
    profile = profiles(locs, ROI)["across"]
    fit = fit_profile(profile.values, profile.precision, model="step",
                      window=profile.window)
    assert fit.values()["edge"] == pytest.approx(20.0, abs=6.0)
    assert fit.extra["side"] < 0                   # the density falls with t


def test_the_model_with_the_better_aic_is_the_one_the_data_came_from():
    """The comparison is the point: one structure or two is a number."""
    n = 3000
    rng = np.random.default_rng(10)
    two = np.where(rng.random(n) < 0.5, -30.0, 30.0)
    profile = profiles(simulate(n, two, precision=8.0, sigma=5.0, seed=11),
                       ROI)["across"]
    best = fit_models(profile.values, profile.precision,
                      models=("gauss", "two_gauss", "step"),
                      window=profile.window)[0]
    assert best.model == "two_gauss"

    one = profiles(simulate(n, 0.0, precision=8.0, sigma=20.0, seed=12),
                   ROI)["across"]
    best = fit_models(one.values, one.precision,
                      models=("gauss", "two_gauss", "step"),
                      window=one.window)[0]
    assert best.model == "gauss"


def test_the_fit_does_not_depend_on_the_bin_width_and_the_binned_one_does():
    """Unbinned is unbinned: the bins are the picture, not the estimator."""
    locs = simulate(600, 0.0, precision=10.0, sigma=15.0, seed=13)
    profile = profiles(locs, ROI)["across"]
    fits = [fit_profile(profile.values, profile.precision, model="gauss",
                        window=profile.window, bin_size=size)
            for size in (2.0, 25.0)]
    assert fits[0].values()["sigma"] == fits[1].values()["sigma"]

    binned = [fit_profile(profile.values, profile.precision, model="gauss",
                          window=profile.window, method="binned", bin_size=size)
              for size in (2.0, 25.0)]
    assert binned[0].values()["sigma"] != binned[1].values()["sigma"]
    # and it should still be the right answer, to within the bins
    assert binned[0].values()["sigma"] == pytest.approx(15.0, abs=3.0)


def test_a_uniform_background_is_fitted_rather_than_widening_the_peak():
    n = 3000
    rng = np.random.default_rng(14)
    structure = rng.normal(0, 10.0, n)
    flat = rng.uniform(-WIDTH / 2, WIDTH / 2, n // 3)
    across = np.concatenate([structure, flat])
    locs = simulate(len(across), across, precision=6.0, seed=15)
    profile = profiles(locs, ROI)["across"]
    with_background = fit_profile(profile.values, profile.precision,
                                  model="gauss", window=profile.window)
    without = fit_profile(profile.values, profile.precision, model="gauss",
                          window=profile.window, background=False)
    assert with_background.background == pytest.approx(0.25, abs=0.08)
    assert with_background.values()["sigma"] == pytest.approx(10.0, abs=2.0)
    assert without.values()["sigma"] > with_background.values()["sigma"] + 2


def test_too_few_localizations_is_said_rather_than_fitted():
    with pytest.raises(ValueError, match="too few"):
        fit_profile(np.zeros(3), None, model="gauss", window=(-50, 50))


def test_the_bin_width_for_the_picture_stays_between_two_and_two_hundred_bins():
    for n in (10, 1000, 100000):
        values = np.random.default_rng(16).normal(0, 20, n)
        width = auto_bin(values, (-100, 100))
        assert 1.0 <= width <= 40.0


# --------------------------------------------------------------- the plugin

def test_the_plugin_reports_every_model_and_draws_what_it_fitted():
    locs = simulate(1500, 0.0, precision=8.0, sigma=14.0, seed=17)
    result = LineProfile().run(context(locs),
                               LineProfileSettings(axis="across", model="all"))
    assert set(result.data["fits"]) == {"gauss", "two_gauss", "step",
                                       "disk", "ring"}
    assert "best by AIC: Gaussian" in result.text
    assert result.data["values"]["gauss"]["sigma"] == pytest.approx(14.0, abs=2.0)

    figure = _figure()
    result.plot(figure.subplots())
    result.plots["scatter"].draw(figure)


def test_the_plugin_refuses_a_profile_the_table_has_no_column_for():
    locs = simulate(200, 0.0, seed=18)
    with pytest.raises(ValueError, match="no z profile"):
        LineProfile().run(context(locs), LineProfileSettings(axis="z"))


def test_a_profile_along_the_line_is_what_a_line_is_drawn_for_and_the_default():
    locs = simulate(1500, 0.0, seed=19)
    result = LineProfile().run(context(locs), LineProfileSettings(model="step"))
    assert LineProfileSettings().axis == "along"
    assert "position along the line" in result.text
    across = LineProfile().run(context(locs),
                               LineProfileSettings(axis="across", model="step"))
    assert "position across the line" in across.text


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.figure()


# ------------------------------------------------------- starting values

def test_the_peak_is_read_off_the_profile_and_not_off_the_median():
    """A peak sitting on a background: the median follows the background."""
    rng = np.random.default_rng(20)
    window = (-100.0, 100.0)
    peak = rng.normal(30.0, 8.0, 1500)
    background = rng.uniform(-100.0, 0.0, 1500)      # all on the other side
    values = np.concatenate([peak, background])
    from smappy.plugins.line_profile import peak_position
    assert peak_position(values, window) == pytest.approx(30.0, abs=4.0)
    assert abs(np.median(values) - 30.0) > 20.0


def test_the_half_maximum_span_gives_the_width_the_eye_reads():
    from smappy.plugins.line_profile import half_max_span
    values = np.random.default_rng(21).normal(0.0, 20.0, 4000)
    low, high = half_max_span(values, (-100.0, 100.0))
    assert (high - low) / 2.355 == pytest.approx(20.0, abs=3.0)


def test_the_edge_start_survives_an_empty_margin_and_a_blob_beyond_it():
    """Both are ordinary: a ROI is wider than the structure, and something
    bright often sits further along it."""
    from smappy.plugins.line_profile import edge_position
    rng = np.random.default_rng(22)
    window = (-100.0, 100.0)
    labelled = rng.uniform(-90.0, 10.0, 2000)        # the edge is at +10
    blob = rng.normal(70.0, 5.0, 300)
    values = np.concatenate([labelled, blob])
    assert edge_position(values, window, side=-1.0) == pytest.approx(10.0, abs=6.0)
    rising = rng.uniform(-20.0, 95.0, 1500)
    assert edge_position(rising, window, side=1.0) == pytest.approx(-20.0, abs=6.0)


def test_em_finds_two_components_that_overlap_and_a_gradient_start_would_miss():
    """The distance is what people come for and what a bad start loses."""
    from smappy.plugins.line_profile import em_two_gaussians
    rng = np.random.default_rng(23)
    n, window = 3000, (-100.0, 100.0)
    precision = np.full(n, 8.0)
    centres = np.where(rng.random(n) < 0.5, -12.5, 12.5)    # 25 nm apart
    values = centres + rng.normal(0, 8.0, n) + rng.normal(0, precision)
    found = em_two_gaussians(values, precision, window)
    assert found["distance"] == pytest.approx(25.0, abs=5.0)
    assert found["sigma"] == pytest.approx(8.0, abs=2.5)

    # and the likelihood agrees with it rather than improving on it
    fit = fit_profile(values, precision, model="two_gauss", window=window)
    assert fit.values()["distance"] == pytest.approx(found["distance"], abs=2.0)


def test_two_peaks_far_apart_and_of_very_different_height_are_both_found():
    """The small peak is below half the tall one, so the profile's top is the
    tall peak alone; a start read only from the top put both components on it
    and reported a distance of about one peak width."""
    rng = np.random.default_rng(31)
    window = (-300.0, 300.0)
    values = np.concatenate([rng.normal(-150.0, 15.0, 1000),
                             rng.normal(150.0, 15.0, 60),
                             rng.uniform(*window, 60)])
    fit = fit_profile(values, None, model="two_gauss", window=window)
    assert fit.values()["distance"] == pytest.approx(300.0, abs=10.0)
    assert fit.values()["sigma"] == pytest.approx(15.0, abs=3.0)
    assert fit.values()["fraction"] == pytest.approx(1000 / 1060, abs=0.03)


def test_em_puts_the_unspecific_localizations_in_the_background_component():
    from smappy.plugins.line_profile import em_two_gaussians
    rng = np.random.default_rng(24)
    window = (-100.0, 100.0)
    pair = np.where(rng.random(2000) < 0.5, -20.0, 20.0) + rng.normal(0, 6.0, 2000)
    flat = rng.uniform(-100.0, 100.0, 1000)
    values = np.concatenate([pair, flat])
    precision = np.full(len(values), 5.0)
    found = em_two_gaussians(values, precision, window)
    assert found["background"] == pytest.approx(1 / 3, abs=0.12)
    assert found["distance"] == pytest.approx(40.0, abs=5.0)


def test_a_bad_start_is_what_the_starting_values_are_there_to_avoid():
    """The failure the EM start removes, shown on the same data."""
    from smappy.plugins.line_profile import (TWO_GAUSS, _density_of, _from_fit,
                                             _to_fit)
    from scipy.optimize import minimize
    rng = np.random.default_rng(25)
    n, window = 2000, (-100.0, 100.0)
    precision = np.full(n, 8.0)
    values = (np.where(rng.random(n) < 0.5, -12.5, 12.5)
              + rng.normal(0, 8.0, n) + rng.normal(0, precision))

    def negative_log_likelihood(vector):
        vector = _from_fit(np.asarray(vector, float), [2])
        density = _density_of(TWO_GAUSS, vector[:-1], window, float(vector[-1]),
                              True, {})
        return float(-np.sum(np.log(np.maximum(density(values, precision),
                                               1e-300))))

    bounds = [(-100, 100), (0, 200), (0, 200 ** 2), (0, 1), (0, 0.95)]
    stuck = minimize(negative_log_likelihood,
                     _to_fit(np.array([0.0, 2.0, 15.0, 0.5, 0.05]), [2]),
                     method="L-BFGS-B", bounds=bounds)
    assert _from_fit(stuck.x, [2])[1] < 10.0            # it never gets out
    assert fit_profile(values, precision, model="two_gauss",
                       window=window).values()["distance"] > 20.0


# ---------------------------------------------------------- round shapes

def round_shape(kind, radius, n, blur=0.0, precision=8.0, seed=0):
    """Localizations on a ring, or filling a disk, seen edge-on."""
    rng = np.random.default_rng(seed)
    errors = np.full(n, float(precision))
    angle = rng.uniform(0, 2 * np.pi, n)
    r = radius if kind == "ring" else radius * np.sqrt(rng.random(n))
    across = r * np.sin(angle)
    return across + rng.normal(0, np.hypot(errors, blur)), errors


def test_a_ring_and_a_disk_give_back_the_radius_they_were_made_with():
    window = (-150.0, 150.0)
    for kind in ("ring", "disk"):
        values, precision = round_shape(kind, 60.0, 3000, seed=26)
        fit = fit_profile(values, precision, model=kind, window=window)
        assert fit.values()["radius"] == pytest.approx(60.0, abs=4.0)
        assert abs(fit.values()["centre"]) < 4.0


def test_a_ring_is_told_from_a_disk_and_from_two_points():
    """What the comparison is for: three models that all have two humps."""
    window = (-150.0, 150.0)
    every = ("gauss", "two_gauss", "step", "disk", "ring")
    for kind in ("ring", "disk"):
        values, precision = round_shape(kind, 60.0, 4000, seed=27)
        best = fit_models(values, precision, models=every, window=window)[0]
        assert best.model == kind


def test_the_round_shapes_are_normalized_over_the_window():
    """The quadrature is a density, so it has to integrate to one."""
    from smappy.plugins.line_profile import arc_density
    integrate = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.26
    window = (-200.0, 200.0)
    grid = np.linspace(*window, 4001)
    for kind in ("ring", "disk"):
        for blur in (2.0, 20.0):
            density = arc_density(grid, 0.0, 60.0, blur, window, kind=kind)
            assert integrate(density, grid) == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------- bootstrap

def test_the_bootstrap_interval_covers_the_truth_about_as_often_as_it_claims():
    """The only test a confidence interval really has: count how often it is
    right.  Twelve samples of forty localizations, a nominal 95% -- the
    percentile bootstrap undercovers a width a little at this size, so this
    asks for three quarters rather than for nineteen in twenty."""
    from smappy.plugins.line_profile import bootstrap
    window, covered = (-100.0, 100.0), 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        precision = np.full(40, 8.0)
        values = rng.normal(0.0, 12.0, 40) + rng.normal(0, precision)
        fit = fit_profile(values, precision, model="gauss", window=window)
        bootstrap(fit, values, precision, rounds=60, seed=seed)
        low, high = fit.intervals["sigma"]
        covered += low <= 12.0 <= high
    assert covered >= 9


def test_the_bootstrap_is_wider_than_the_curvature_on_a_small_sample():
    """Forty localizations: the fit's own error bar is the optimistic one."""
    from smappy.plugins.line_profile import bootstrap
    rng = np.random.default_rng(28)
    n, window = 40, (-100.0, 100.0)
    precision = np.full(n, 8.0)
    values = rng.normal(0.0, 12.0, n) + rng.normal(0, precision)
    fit = fit_profile(values, precision, model="gauss", window=window)
    bootstrap(fit, values, precision, rounds=200, seed=1)

    low, high = fit.intervals["sigma"]
    curvature = fit.uncertainties()["sigma"]
    assert high - low > 2 * curvature        # wider than +- one sigma either way
    assert fit.replicates.shape == (200, 3)  # centre, sigma, background


def test_the_bootstrap_and_the_curvature_agree_when_there_is_enough_data():
    """The asymptotic error bar is asymptotically right; this is the check."""
    from smappy.plugins.line_profile import bootstrap
    rng = np.random.default_rng(29)
    n, window = 4000, (-100.0, 100.0)
    precision = np.full(n, 8.0)
    values = rng.normal(0.0, 15.0, n) + rng.normal(0, precision)
    fit = fit_profile(values, precision, model="gauss", window=window)
    bootstrap(fit, values, precision, rounds=150, seed=2)
    low, high = fit.intervals["sigma"]
    assert (high - low) / (2 * 1.96 * fit.uncertainties()["sigma"]) == \
        pytest.approx(1.0, abs=0.35)


def test_the_bootstrap_shows_a_width_that_the_data_cannot_resolve():
    """A structure narrower than the precision: the interval runs to zero,
    which is the answer, and a symmetric error bar cannot say it."""
    from smappy.plugins.line_profile import bootstrap
    rng = np.random.default_rng(30)
    n, window = 60, (-100.0, 100.0)
    precision = np.full(n, 12.0)
    values = rng.normal(0.0, 1.0, n) + rng.normal(0, precision)   # a point
    fit = fit_profile(values, precision, model="gauss", window=window)
    bootstrap(fit, values, precision, rounds=200, seed=3)
    low, high = fit.intervals["sigma"]
    assert low == pytest.approx(0.0, abs=0.5) and high < 12.0


def test_the_same_seed_gives_the_same_interval():
    from smappy.plugins.line_profile import bootstrap
    rng = np.random.default_rng(31)
    n, window = 200, (-100.0, 100.0)
    precision = np.full(n, 8.0)
    values = rng.normal(0.0, 14.0, n) + rng.normal(0, precision)
    made = [bootstrap(fit_profile(values, precision, model="gauss",
                                  window=window),
                      values, precision, rounds=60, seed=4).intervals["sigma"]
            for _ in range(2)]
    assert made[0] == made[1]


def test_a_step_keeps_the_direction_it_was_fitted_with_through_the_resamples():
    """Otherwise half the resamples answer about the other edge."""
    from smappy.plugins.line_profile import bootstrap
    rng = np.random.default_rng(32)
    window = (-100.0, 100.0)
    values = rng.uniform(-90.0, 10.0, 300)
    precision = np.full(len(values), 8.0)
    fit = fit_profile(values, precision, model="step", window=window)
    assert fit.extra["side"] < 0
    bootstrap(fit, values, precision, rounds=100, seed=5)
    low, high = fit.intervals["edge"]
    assert low <= fit.values()["edge"] <= high and high - low < 20.0


def test_the_plugin_reports_the_intervals_and_offers_their_figure():
    locs = simulate(300, 0.0, precision=8.0, sigma=12.0, seed=33)
    result = LineProfile().run(context(locs),
                               LineProfileSettings(axis="across", bootstrap=80,
                                                   confidence=90.0))
    assert "90% confidence intervals, 80 resamples:" in result.text
    low, high = result.data["intervals"]["gauss"]["sigma"]
    fitted = result.data["values"]["gauss"]["sigma"]
    assert low <= fitted <= high
    assert low == pytest.approx(12.0, abs=3.0) and high == pytest.approx(12.0, abs=3.0)
    result.plots["bootstrap"].draw(_figure())


def test_a_run_without_the_bootstrap_has_no_intervals_and_no_extra_figure():
    locs = simulate(300, 0.0, precision=8.0, sigma=12.0, seed=34)
    result = LineProfile().run(context(locs), LineProfileSettings())
    assert result.data["intervals"] == {}
    assert "bootstrap" not in result.plots


# ------------------------------------------------------- one fit per layer

def _two_channel_session():
    """Two channels over the same line, 60 nm apart, one layer each."""
    from smappy.session import Session

    one = simulate(800, -30.0, precision=8.0, sigma=6.0, seed=51)
    two = simulate(800, +30.0, precision=8.0, sigma=6.0, seed=52)
    columns = {name: np.concatenate([np.asarray(one[name]), np.asarray(two[name])])
               for name in one.keys()}
    columns["channel"] = np.concatenate([np.zeros(len(one), np.int64),
                                         np.ones(len(two), np.int64)])
    session = Session(Localizations(columns, {"units": "nm"}))
    session.add_layer(like=0)
    for i, layer in enumerate(session.layers):
        layer.set_bound("channel", i - 0.5, i + 0.5)
        layer.name = f"channel {i + 1}"
    session.set_roi(ROI)
    return session


def test_a_profile_is_fitted_per_layer_and_drawn_in_each_layer_s_colour():
    """A line is drawn over two channels and the measurement is how they
    differ, so one fit for both would answer the wrong question."""
    session = _two_channel_session()
    result = LineProfile().run(session.context(),
                               LineProfileSettings(axis="across"))
    layers = result.data["layer_profiles"]
    assert [l.name for l in layers] == ["channel 1", "channel 2"]
    assert layers[0].colour != layers[1].colour
    centres = [l.fits[0].values()["centre"] for l in layers]
    assert centres[0] == pytest.approx(-30.0, abs=4.0)
    assert centres[1] == pytest.approx(+30.0, abs=4.0)
    assert "channel 1:" in result.text and "channel 2:" in result.text
    assert set(result.data["fits"]) == {"channel 1", "channel 2"}

    figure = _figure()
    result.plot(figure.subplots())
    result.plots["scatter"].draw(_figure())


def test_a_hidden_layer_is_not_fitted():
    session = _two_channel_session()
    session.layers[1].visible = False
    result = LineProfile().run(session.context(), LineProfileSettings(axis="across"))
    assert [l.name for l in result.data["layer_profiles"]] == ["channel 1"]


def test_the_selection_can_still_be_measured_as_one():
    session = _two_channel_session()
    result = LineProfile().run(session.context(),
                               LineProfileSettings(axis="across", source="selection",
                                                   model="two_gauss"))
    assert len(result.data["layer_profiles"]) == 1
    # one layer's filter, so this is channel 1 alone rather than both
    assert result.data["fits"].keys() == {"two_gauss"}


def test_a_layer_colour_is_read_off_its_lut_and_falls_back_when_it_cannot_be():
    from dataclasses import dataclass as _dataclass

    from smappy.plugins.line_profile import FALLBACK_COLOURS, layer_colour

    @_dataclass
    class Display:
        lut: str = "hot"
        invert: bool = False

    assert layer_colour(Display(lut="red")) == "#bf0000"
    # a grey ramp has no colour to take, and a nonsense one no LUT
    assert layer_colour(Display(lut="gray"), 1) == FALLBACK_COLOURS[1]
    assert layer_colour(Display(lut="nonsense"), 2) == FALLBACK_COLOURS[2]
    # and a colour another layer already has is not used twice
    mine = layer_colour(Display(lut="red"))
    assert layer_colour(Display(lut="red"), 1, taken=[mine]) != mine


def test_a_run_is_applied_to_the_session_without_touching_its_layers():
    """Run hands its result to `Session.apply`, which takes data["layers"] as
    layer set-ups (Chain/Layers); the per-layer profiles were under that key
    once, and every Run from the GUI failed there."""
    session = _two_channel_session()
    before = [l.name for l in session.layers]
    result = session.run(LineProfile(), LineProfileSettings(axis="across"))
    assert "layers" not in result.data and len(result.data["layer_profiles"]) == 2
    assert [l.name for l in session.layers] == before
