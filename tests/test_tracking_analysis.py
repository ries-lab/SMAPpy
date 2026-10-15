"""Tracking analysis: the MSD's covariance and its optimal lags, D and alpha
per track and pooled, the diffusive states of the jumps, and the plugin --
on simulated tracks with known D, states and precision."""
import numpy as np
import pytest

from smappy.mathparse import recipes
from smappy.plugins import Context, Selection
from smappy.plugins.tracking_analysis import TrackingAnalysis
from smappy.simulate import simulate_tracks
from smappy.tracking.analysis import (fit_jumps, gather, jumps, lag_errors,
                                      msd_analysis, msd_covariance, optimal_lags,
                                      steps_of)


def tracks_of(locs, **values):
    return gather(locs["track_id_true"], locs["frame"], locs["x_nm"], locs["y_nm"],
                  err_xy=locs["xy_err_nm"], frame_time_s=0.01, **values)


def test_the_msd_covariance_is_the_one_brownian_tracks_with_errors_have():
    rng = np.random.default_rng(0)
    n, p, x, m = 25, 4, 2.0, 20000
    steps = rng.normal(0, np.sqrt(2.0), (m, n))
    steps[:, 0] = 0
    pos = np.cumsum(steps, axis=1) + rng.normal(0, np.sqrt(x), (m, n))
    msd = np.stack([((pos[:, k:] - pos[:, :-k]) ** 2).mean(1) for k in range(1, p + 1)], 1)
    assert np.allclose(np.cov(msd.T), msd_covariance(n, p, x), rtol=0.05)


def test_the_optimal_lags_grow_with_the_localization_error():
    # Michalet 2010: two points (here one, the offset known) when the error is
    # small, more as it dominates; and the curve has its minimum there
    assert optimal_lags(100, 0.01)[0] == 1
    assert optimal_lags(100, 0.01, fitted=True)[0] == 2
    assert optimal_lags(100, 100.0)[0] > 10
    errors = lag_errors(100, 100.0)
    best = int(np.nanargmin(errors))
    assert errors[0] > 2 * errors[best]


@pytest.mark.parametrize("d, precision", [(0.5, 20.0), (0.05, 20.0), (0.01, 30.0)])
def test_the_msd_gives_back_d_and_alpha_of_free_diffusion(d, precision):
    locs = simulate_tracks(n_frames=1500, density_um2=0.02, d_um2_s=(d,),
                           track_frames=50, precision_nm=precision, seed=1)
    result = msd_analysis(tracks_of(locs))
    assert result.pooled.D_um2_per_s[0] == pytest.approx(d, rel=0.05)
    assert result.pooled.alpha[0] == pytest.approx(1.0, abs=0.08)
    per = result.per_track.D_um2_per_s
    assert np.nanmean(per) == pytest.approx(d, rel=0.1)
    fitted = msd_analysis(tracks_of(locs), fitted=True)
    assert fitted.pooled.D_um2_per_s[0] == pytest.approx(d, rel=0.08)


def test_tracks_shorter_than_the_minimum_get_no_d():
    locs = simulate_tracks(n_frames=300, density_um2=0.02, track_frames=8, seed=4)
    t = tracks_of(locs)
    result = msd_analysis(t, min_length=6)
    short = t.counts() < 6
    assert short.any() and not short.all()
    assert np.isnan(result.per_track.D_um2_per_s[short]).all()
    assert np.isfinite(result.per_track.D_um2_per_s[~short]).all()


def test_the_jumps_give_back_two_states_and_their_fractions():
    locs = simulate_tracks(n_frames=800, density_um2=0.05, d_um2_s=(0.02, 1.0),
                           fractions=(0.3, 0.7), track_frames=30, precision_nm=20.0,
                           seed=2)
    t = tracks_of(locs)
    d2, err = jumps(t, 1)
    one = fit_jumps(d2, err, 2, 0.01, 1)
    two = fit_jumps(d2, err, 2, 0.01, 2)
    assert two.D_um2_per_s == pytest.approx([0.02, 1.0], rel=0.1)
    assert two.fraction == pytest.approx([0.3, 0.7], abs=0.03)
    assert two.bic < one.bic


def test_jumps_at_a_lag_of_two_frames_see_twice_the_spread():
    locs = simulate_tracks(n_frames=600, density_um2=0.03, d_um2_s=(0.2,), seed=3)
    t = tracks_of(locs)
    for lag in (1, 2):
        d2, err = jumps(t, lag)
        assert fit_jumps(d2, err, 2, lag * 0.01, 1).D_um2_per_s[0] == \
            pytest.approx(0.2, rel=0.05)


def test_an_irregular_clock_is_put_on_its_median_interval():
    ids = np.array([1, 1, 1, 1, 2, 2])
    # intervals 0.01, 0.021, 0.009 and 0.01: the median is 0.01, and 0.031
    # is three steps
    time_s = np.array([0.0, 0.01, 0.031, 0.04, 5.0, 5.01])
    step, dt = steps_of(ids, time_s)
    assert dt == pytest.approx(0.01)
    assert step.tolist() == [0, 1, 3, 4, 0, 1]


def test_the_plugin_measures_and_writes_d_per_track():
    locs = simulate_tracks(n_frames=600, density_um2=0.03, d_um2_s=(0.02, 0.5),
                           fractions=(0.4, 0.6), track_frames=30, precision_nm=20.0,
                           seed=5)
    keep = np.asarray(locs["x_nm"]) < 15000
    plugin = TrackingAnalysis()
    settings = TrackingAnalysis.Settings(id_column="track_id_true")
    settings.msd_fit.write_columns = True
    assert plugin.changes_locs_for(settings)
    result = plugin.run(Context(locs=locs, selection=Selection(keep)), settings)
    v = result.values
    assert v["jump_D1_um2_per_s"] == pytest.approx(0.02, rel=0.15)
    assert v["jump_D2_um2_per_s"] == pytest.approx(0.5, rel=0.1)
    assert v["msd_alpha"] == pytest.approx(1.0, abs=0.1)
    D = np.asarray(result.locs["track_D_um2_per_s"])
    assert np.isnan(D[~keep]).all()
    # a track's D is the same on each of its localizations
    truth = np.asarray(locs["track_id_true"])
    some = truth[keep & np.isfinite(D)][0]
    assert np.unique(D[truth == some][keep[truth == some]]).size == 1
    assert {r["field"] for r in recipes(result.locs)} >= {"track_D_um2_per_s",
                                                         "track_alpha"}


def test_the_plugin_takes_the_files_frame_time():
    locs = simulate_tracks(n_frames=500, density_um2=0.03, d_um2_s=(0.2,),
                           frame_time_ms=20.0, seed=6)
    settings = TrackingAnalysis.Settings(id_column="track_id_true", jumps=False)
    result = TrackingAnalysis().run(Context(locs=locs), settings)
    assert result.values["msd_D_um2_per_s"] == pytest.approx(0.2, rel=0.06)
    assert result.locs is None


def test_the_plugin_says_how_to_get_tracks():
    locs = simulate_tracks(n_frames=50, seed=1)
    with pytest.raises(ValueError, match="Link tracks"):
        TrackingAnalysis().run(Context(locs=locs), TrackingAnalysis.Settings())
