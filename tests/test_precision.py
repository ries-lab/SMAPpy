"""The precision plugin, against data whose precision is known by construction."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Context, Selection
from smappy.plugins.precision import (LocalizationPrecision, crlb_statistics,
                                      displacement_pairs, drift_free_sigma,
                                      fit_axis, fit_radial, measure,
                                      sigma_at_photons)


def blinking(sigma=8.0, sigma_z=None, n_emitters=600, blinks=6, on_time=3,
             n_frames=400, extent=5000.0, seed=1, walk=0.0, precision=None,
             singles=0, spread=False, kappa=1.0):
    """Emitters that blink for `on_time` frames, localized with a known error.

    Every localization of one blink is the same molecule, so every pair of them
    in adjacent frames is a true pair whose displacement carries exactly
    ``2 sigma^2`` -- which is the number the fits have to find.  ``walk`` adds a
    random walk to the whole field, the way a stage drifts.  ``precision`` is
    what the table *claims* (the CRLB column), which is `sigma` unless a test
    wants the two to disagree.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, extent, (n_emitters, 3))
    truth[:, 2] = rng.uniform(-300, 300, n_emitters)
    emitter, frame = [], []
    for index in range(n_emitters):
        for _ in range(blinks):
            start = int(rng.integers(0, n_frames - on_time))
            emitter.extend([index] * on_time)
            frame.extend(range(start, start + on_time))
    emitter, frame = np.array(emitter), np.array(frame, dtype=np.int64)
    n = len(frame)
    if spread:
        # photons are exponential, and a precision is S / sqrt(N)
        photons = rng.exponential(2000.0, n) + 300.0
        claimed_nm = sigma * np.sqrt(2000.0 / photons)
    else:
        photons = np.full(n, 2000.0)
        claimed_nm = np.full(n, float(sigma))
    lateral = kappa * claimed_nm
    axial = lateral * ((sigma_z / sigma) if sigma_z else 1.0)
    xyz = truth[emitter] + np.column_stack([
        rng.normal(0, lateral), rng.normal(0, lateral), rng.normal(0, axial)])
    if walk:
        steps = np.cumsum(rng.normal(0, walk, (n_frames + on_time, 3)), axis=0)
        xyz = xyz + steps[frame]
    extra = singles * n_frames
    if singles:
        xyz = np.vstack([xyz, rng.uniform(0, extent, (extra, 3))])
        frame = np.concatenate([frame, np.repeat(np.arange(n_frames), singles)])
        emitter = np.concatenate([emitter, -np.arange(1, extra + 1)])
        n += extra
    if singles:
        claimed_nm = np.concatenate([claimed_nm, np.full(extra, float(sigma))])
        photons = np.concatenate([photons, np.full(extra, 2000.0)])
    if precision is not None:                  # what the table claims it is
        claimed_nm = np.full(n, float(precision))
    columns = {"frame": frame,
               "x_nm": xyz[:, 0].astype(np.float32),
               "y_nm": xyz[:, 1].astype(np.float32),
               "z_nm": xyz[:, 2].astype(np.float32),
               "photons": photons.astype(np.float32),
               "xy_err_nm": claimed_nm.astype(np.float32),
               "z_err_nm": (claimed_nm
                                      * ((sigma_z / sigma) if sigma_z else 1.0)
                                      ).astype(np.float32),
               "emitter": emitter.astype(np.int32)}
    return Localizations(columns, {"units": "nm", "sigma": sigma})


def test_the_pairwise_fit_finds_the_precision_the_data_was_given():
    locs = blinking(sigma=8.0)
    pairs = displacement_pairs(locs["x_nm"], locs["y_nm"], locs["frame"],
                               d_max=48.0, gaps=(1,))[1]
    fit = fit_radial(pairs.d, pairs.d_max)
    assert fit.ok
    assert fit.sigma == pytest.approx(8.0, abs=0.4)
    assert fit.fraction > 0.8               # this sample is nearly all true pairs
    assert fit.sigma_error < 0.3


def test_the_per_axis_fits_find_the_axial_precision_separately():
    locs = blinking(sigma=6.0, sigma_z=18.0)
    found = measure(locs, max_gap=1)
    assert found.axes["x"].sigma == pytest.approx(6.0, abs=0.4)
    assert found.axes["y"].sigma == pytest.approx(6.0, abs=0.4)
    assert found.axes["z"].sigma == pytest.approx(18.0, abs=1.2)


def test_the_fit_survives_a_background_of_other_molecules():
    # a crowd of molecules that never repeat: most neighbours in the next frame
    # are now a different one, which is what the background term is for
    locs = blinking(sigma=8.0, extent=2000.0, singles=400, seed=3)
    found = measure(locs, max_gap=1)
    assert found.radial.sigma == pytest.approx(8.0, abs=0.8)
    assert found.radial.fraction < 0.6            # most pairs are not the pair


def split_blinks(n_emitters=900, blinks=6, n_frames=400, extent=5000.0,
                 total=4000.0, factor=360.0, seed=2, gain=1.0):
    """Blinks that straddle two frames and share their photons between them.

    This is the case NeNA actually measures: the molecule switches part-way
    into an exposure, so both halves of every pair are dimmer than a whole
    localization and the displacement between them is correspondingly worse.
    `factor` is the constant of ``sigma = factor / sqrt(N)``; `gain` scales the
    photon column alone, the way a mis-calibrated camera would.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, extent, (n_emitters, 2))
    emitter = np.repeat(np.arange(n_emitters), 2 * blinks)
    start = rng.integers(0, n_frames - 1, n_emitters * blinks)
    frame = np.repeat(start, 2) + np.tile([0, 1], n_emitters * blinks)
    share = np.repeat(rng.uniform(0.15, 0.85, n_emitters * blinks), 2)
    share[1::2] = 1 - share[1::2]
    photons = total * share
    sigma = factor / np.sqrt(photons)
    xy = truth[emitter] + np.column_stack([rng.normal(0, sigma),
                                           rng.normal(0, sigma)])
    return Localizations(
        {"frame": frame.astype(np.int64),
         "x_nm": xy[:, 0].astype(np.float32), "y_nm": xy[:, 1].astype(np.float32),
         "photons": (photons * gain).astype(np.float32),
         "xy_err_nm": sigma.astype(np.float32),
         "emitter": emitter.astype(np.int32)}, {"units": "nm"})


def test_the_split_pairs_make_nena_worse_than_a_whole_localization():
    # every pair here is one blink cut in two, so both its halves are dimmer
    # than the molecule was: the plain sigma inherits that, and the photon law
    # is what puts the number back on a localization of full brightness
    found = measure(split_blinks(), max_gap=1)
    whole = sigma_at_photons(found.photon_law, 4000.0)
    assert whole == pytest.approx(360.0 / np.sqrt(4000.0), rel=0.10)
    assert found.radial.sigma > whole * 1.2


def test_the_photon_law_does_not_care_what_the_camera_gain_is_said_to_be():
    honest = measure(split_blinks(), max_gap=1)
    wrong = measure(split_blinks(gain=10.0), max_gap=1)
    assert (wrong.photon_law.amplitude
            == pytest.approx(10 * honest.photon_law.amplitude, rel=0.02))
    # the prediction reads the same column it was fitted from, so it survives
    assert (sigma_at_photons(wrong.photon_law, 10 * 4000.0)
            == pytest.approx(sigma_at_photons(honest.photon_law, 4000.0), rel=0.02))


def test_kappa_is_the_one_number_a_wrong_calibration_moves():
    # the table claims twice the precision it has: kappa halves, while the
    # plain sigma and the photon law stay where the coordinates put them
    honest = measure(split_blinks(), max_gap=1, reach=48.0)
    claimed = split_blinks()
    columns = dict(claimed.columns)
    columns["xy_err_nm"] = claimed["xy_err_nm"] * 2
    # the radius is held, because left to itself it follows the median
    # precision -- the one place the claimed column reaches the plain sigma
    optimistic = measure(Localizations(columns, claimed.metadata), max_gap=1,
                         reach=48.0)
    assert optimistic.radial.sigma == pytest.approx(honest.radial.sigma, rel=0.001)
    assert (optimistic.photon_law.amplitude
            == pytest.approx(honest.photon_law.amplitude, rel=0.001))
    assert optimistic.scaled.kappa == pytest.approx(honest.scaled.kappa / 2, rel=0.03)


def test_the_scaled_fit_is_one_when_the_data_reaches_its_bound():
    locs = blinking(sigma=9.0)              # the column says 9, the data is 9
    found = measure(locs, max_gap=1)
    assert found.scaled.kappa == pytest.approx(1.0, abs=0.06)
    assert found.scaled.crlb == pytest.approx(9.0, abs=0.01)


def test_the_scaled_fit_measures_how_far_the_data_misses_the_bound():
    # the fitter claims 6 nm and the localizations scatter by 9: kappa = 1.5,
    # which is the number that carries over to any other set of localizations
    locs = blinking(sigma=9.0, precision=6.0)
    found = measure(locs, max_gap=1)
    assert found.scaled.kappa == pytest.approx(1.5, abs=0.1)
    assert found.scaled.sigma == pytest.approx(9.0, abs=0.6)


def test_the_frame_gap_curve_separates_drift_from_precision():
    locs = blinking(sigma=5.0, on_time=6, walk=3.0, seed=5)
    found = measure(locs, max_gap=4)
    sigmas = [found.by_gap[gap].sigma for gap in sorted(found.by_gap)]
    assert sigmas == sorted(sigmas)          # a walking sample only gets worse
    assert found.gap_line["sigma0"] == pytest.approx(5.0, abs=0.7)
    assert found.gap_line["step"] == pytest.approx(3.0, abs=0.6)


def test_a_still_sample_has_the_same_precision_at_every_gap():
    found = measure(blinking(sigma=7.0, on_time=5), max_gap=3)
    sigmas = [found.by_gap[gap].sigma for gap in sorted(found.by_gap)]
    assert max(sigmas) - min(sigmas) < 0.6
    assert found.gap_line["step"] < 1.5


def test_the_displacements_carry_the_drift_of_the_gap_in_their_mean():
    locs = blinking(sigma=5.0, on_time=4, seed=7)
    shifted = dict(locs.columns)
    shifted["x_nm"] = locs["x_nm"] + 2.0 * locs["frame"]      # 2 nm per frame
    found = measure(Localizations(shifted, locs.metadata), max_gap=1)
    assert found.axes["x"].offset == pytest.approx(2.0, abs=0.3)
    assert found.axes["y"].offset == pytest.approx(0.0, abs=0.3)
    assert found.axes["x"].sigma == pytest.approx(5.0, abs=0.4)


def test_the_truncated_crlb_fit_recovers_what_a_filter_cut_away():
    rng = np.random.default_rng(11)
    photons = rng.exponential(2000.0, 200000)
    sigma = 400.0 / np.sqrt(photons)                 # sigma_c = 400/sqrt(2000)
    whole = crlb_statistics(Localizations({"xy_err_nm": sigma}, {}))
    kept = sigma[sigma <= 12.0]
    cut = crlb_statistics(Localizations({"xy_err_nm": kept}, {}),
                          bounds={"xy_err_nm": (None, 12.0)})
    expected = 400.0 / np.sqrt(2000.0)
    assert whole["lateral"]["sigma_c"] == pytest.approx(expected, rel=0.03)
    assert cut["lateral"]["sigma_c"] == pytest.approx(expected, rel=0.05)
    # without the bound the cut sample would report the cut, not the sample
    assert np.median(kept) < np.median(sigma)


def test_the_gap_line_falls_back_to_the_one_gap_it_has():
    line = drift_free_sigma([1], [7.0], [0.1])
    assert line["sigma0"] == pytest.approx(7.0)
    assert line["n"] == 1


def test_a_fit_with_too_few_pairs_says_so_rather_than_inventing_a_number():
    fit = fit_axis(np.array([1.0, -2.0, 0.5]), 40.0)
    assert not fit.ok and "too few" in fit.message


def test_the_plugin_reports_the_selection_and_the_whole_table_side_by_side():
    locs = blinking(sigma=8.0, seed=13, spread=True)
    selection = Selection(np.asarray(locs["y_nm"]) < 2500.0, name="half")
    ctx = Context(locs=locs, selection=selection)
    result = LocalizationPrecision()(ctx=ctx, max_gap=2)
    assert len(result.data["measurements"]) == 2
    assert set(result.data["sigma"]) == {"half", "all localizations"}
    # every localization has its own precision here, so one sigma over all of
    # them is an effective value above the median rather than the median
    for sigma in result.data["sigma"].values():
        assert 8.0 < sigma < 12.0
    for amplitude in result.data["amplitude"].values():
        assert np.sqrt(amplitude / 2000.0) == pytest.approx(8.0, abs=0.8)
    assert "kappa" in result.text and "CRLB" in result.text
    assert {"per axis", "frame gap", "CRLB"} <= set(result.plots)


def sigma_at_photons_of(amplitude, photons):
    return float(np.sqrt(amplitude / photons))


def test_the_kept_numbers_are_plain_enough_to_write_into_a_file():
    import json

    locs = blinking(sigma=8.0, seed=17)
    plugin = LocalizationPrecision()
    result = plugin(ctx=Context(locs=locs), max_gap=2)
    kept = plugin.keep(result)
    assert json.loads(json.dumps(kept))["measurements"][0]["sigma"] == \
        pytest.approx(8.0, abs=0.6)


def test_a_table_without_frames_says_which_columns_it_has():
    locs = Localizations({"x_nm": np.zeros(50), "y_nm": np.zeros(50)}, {})
    with pytest.raises(ValueError, match="frame"):
        measure(locs)
