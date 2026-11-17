"""Tracking analysis: the MSD's covariance and its optimal lags, D and alpha
per track and pooled, the diffusive states of the jumps, and the plugin --
on simulated tracks with known D, states and precision."""
import numpy as np
import pytest

from smappy.mathparse import recipes
from smappy.plugins import Context, Selection
from smappy.plugins.tracking_analysis import TrackingAnalysis
from smappy.simulate import simulate_tracks
from smappy.tracking.analysis import (fit_jumps, gather, in_focus, jumps,
                                      jumps_over_lags, lag_errors, msd_analysis,
                                      msd_covariance, optimal_lags, steps_of)


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
    from smappy.locs import Localizations
    locs = simulate_tracks(n_frames=50, seed=1)
    bare = Localizations({c: locs[c] for c in ("x_nm", "y_nm", "frame")},
                         dict(locs.metadata))
    with pytest.raises(ValueError, match="Link tracks"):
        TrackingAnalysis().run(Context(locs=bare), TrackingAnalysis.Settings())


def test_auto_takes_a_simulations_truth_when_nothing_is_linked():
    locs = simulate_tracks(n_frames=300, density_um2=0.03, d_um2_s=(0.2,), seed=6)
    result = TrackingAnalysis().run(Context(locs=locs),
                                    TrackingAnalysis.Settings(jumps=False))
    assert "(track_id_true)" in result.text
    assert result.values["msd_D_um2_per_s"] == pytest.approx(0.2, rel=0.08)


def test_over_the_view_each_tracks_fit_is_kept_as_a_row_of_the_track_id():
    """D and alpha per track, which the MSD fits for every track at once,
    go back as a region table, current as a block's over the tracks."""
    from smappy.region_results import Evaluation, is_current, rows
    from smappy.session import Session
    locs = simulate_tracks(n_frames=1500, density_um2=0.02, d_um2_s=(0.1,),
                           track_frames=60, seed=5)
    session = Session(locs)
    session.show_grouped(0, False)
    for layer in session.layers:
        layer.remove_bound("z_nm")
    settings = TrackingAnalysis.Settings(id_column="track_id_true", jumps=False)
    result = session.run(TrackingAnalysis(), settings)
    ev = Evaluation("track_id_true", "Tracking analysis")
    ctx = session.context()
    assert is_current(ctx, ev)
    found = rows(ctx, ev)
    per = result.data["msd"].per_track
    assert len(found) == len(per.D_um2_per_s)
    fitted = np.array([r["D_um2_per_s"] for r in found], float)
    assert np.nanmedian(fitted) == pytest.approx(0.1, rel=0.3)


# ------------------------------------------- the focus and the jumps at several lags

def test_the_fraction_kept_in_focus_is_what_particles_diffusing_in_z_keep():
    rng = np.random.default_rng(0)
    for D in (0.05, 2.0):
        z = rng.uniform(-350, 350, 200_000)
        alive = np.ones(len(z), bool)
        kept = []
        for _ in range(5):
            z = z + rng.normal(size=len(z)) * np.sqrt(2 * D * 1e6 * 0.01)
            alive &= np.abs(z) <= 350
            kept.append(alive.mean())
        assert in_focus([D], 0.01, 700.0, 5)[0] == pytest.approx(kept, abs=3e-3)
    # no focus, nothing lost; a particle that does not move stays
    assert (in_focus([1.0, 0.0], 0.01, 0.0, 3) == 1).all()
    assert (in_focus([0.0], 0.01, 700.0, 3) == 1).all()


def test_a_focus_ends_tracks_and_ends_the_fast_ones_sooner():
    locs = simulate_tracks(n_frames=800, density_um2=0.02, d_um2_s=(0.02, 3.0),
                           fractions=(0.5, 0.5), track_frames=30.0, use_z=True,
                           focal_range_nm=600.0, seed=4)
    assert np.abs(np.asarray(locs["z_nm"])).max() < 300 + 5 * 50   # 50 nm: its error
    truth, state = np.asarray(locs["track_id_true"]), np.asarray(locs["state_true"])
    length = np.bincount(truth)
    slow = np.unique(truth[state == 0])
    fast = np.unique(truth[state == 1])
    assert length[fast].mean() < 0.5 * length[slow].mean()
    # without a focus the simulation is as it was: same seed, same table
    a = simulate_tracks(n_frames=100, seed=1)
    b = simulate_tracks(n_frames=100, seed=1, focal_range_nm=0.0)
    assert np.array_equal(np.asarray(a["x_nm"]), np.asarray(b["x_nm"]))


def test_several_lags_fitted_together_give_back_d():
    locs = simulate_tracks(n_frames=600, density_um2=0.03, d_um2_s=(0.3,),
                           track_frames=30.0, precision_nm=25.0, seed=8)
    d2, err, lag = jumps_over_lags(tracks_of(locs), 5)
    assert set(np.unique(lag)) == {1, 2, 3, 4, 5}
    fit = fit_jumps(d2, err, 2, lag * 0.01, 1)
    assert fit.D_um2_per_s[0] == pytest.approx(0.3, rel=0.05)
    assert list(fit.lag_s) == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])


def test_the_axial_range_gives_back_the_fraction_of_fast_particles():
    # half slow, half fast: the fast leave a 700 nm focus within a few
    # frames, so they make far fewer jumps than their number
    locs = simulate_tracks(n_frames=2000, density_um2=0.02, d_um2_s=(0.05, 3.0),
                           fractions=(0.5, 0.5), track_frames=30.0, precision_nm=25.0,
                           focal_range_nm=700.0, seed=3)
    d2, err, lag = jumps_over_lags(tracks_of(locs), 7)
    plain = fit_jumps(d2, err, 2, lag * 0.01, 2)
    fixed = fit_jumps(d2, err, 2, lag * 0.01, 2, focal_range_nm=700.0, step_s=0.01)
    assert plain.fraction[1] < 0.35
    assert fixed.fraction == pytest.approx([0.5, 0.5], abs=0.05)   # seeds scatter by 0.03
    assert fixed.D_um2_per_s == pytest.approx([0.05, 3.0], rel=0.1)
    # the fast state's share of the jumps falls with lag, as its particles leave
    assert (np.diff(fixed.seen[1]) < 0).all()
    assert fixed.kept[1, 0] < fixed.kept[0, 0]


def test_in_3d_the_axial_range_fits_the_lateral_jumps():
    # what stays in focus moved less in z, so z's jumps would make D too small
    locs = simulate_tracks(n_frames=1200, density_um2=0.02, d_um2_s=(1.0,),
                           track_frames=30.0, precision_nm=25.0, use_z=True,
                           z_precision_nm=40.0, focal_range_nm=800.0, seed=5)
    settings = TrackingAnalysis.Settings(id_column="track_id_true", msd=False)
    settings.jump_fit.states = 1
    settings.axial_range_nm = 800.0
    result = TrackingAnalysis().run(Context(locs=locs), settings)
    assert result.values["jump_D1_um2_per_s"] == pytest.approx(1.0, rel=0.05)
    assert "lateral jumps" in result.text
    assert result.data["jumps"].focal_range_nm == 800.0
    assert result.plot is not None


# ------------------------------------------------------ the D spectrum

def test_a_pieces_likelihood_is_the_gaussian_of_its_shared_localizations():
    from scipy.stats import multivariate_normal
    from smappy.tracking.spectrum import _tridiagonal_loglik
    rng = np.random.default_rng(1)
    L = 6
    x = rng.normal(size=(1, L)) * 100
    lag = np.array([[1, 1, 2, 1, 1, 3]]) * 0.01
    s2 = rng.uniform(100, 900, (1, L + 1))
    D = np.array([0.3e6])
    C = np.diag(2 * D[0] * lag[0] + s2[0, :-1] + s2[0, 1:])
    for j in range(1, L):
        C[j, j - 1] = C[j - 1, j] = -s2[0, j]       # the localization they share
    got = _tridiagonal_loglik(x, lag, s2[:, :-1] + s2[:, 1:], s2[:, 1:-1], D)
    assert got[0, 0] == pytest.approx(multivariate_normal(np.zeros(L), C).logpdf(x[0]))


def test_tracks_are_cut_into_pieces_that_hold_every_jump_once():
    from smappy.tracking.spectrum import pieces_of
    locs = simulate_tracks(n_frames=200, density_um2=0.03, track_frames=25.0, seed=2)
    t = tracks_of(locs)
    p = pieces_of(t, 7)
    assert p.n_jumps.max() == 7 and p.n_jumps.min() >= 1
    assert p.n_jumps.sum() == (t.track[1:] == t.track[:-1]).sum()
    # a piece stays in its track
    assert (t.track[p.first + p.n_jumps] == p.track).all()


def test_the_spectrum_finds_three_populations_without_being_told_how_many():
    from smappy.tracking.spectrum import peaks, spectrum
    locs = simulate_tracks(n_frames=1500, density_um2=0.02, d_um2_s=(0.02, 0.3, 2.0),
                           fractions=(0.3, 0.3, 0.4), track_frames=30.0,
                           precision_nm=25.0, seed=7)
    found = peaks(spectrum(tracks_of(locs)))
    assert len(found) == 3
    D, f = np.array(found).T
    assert D == pytest.approx([0.02, 0.3, 2.0], rel=0.2)
    assert f == pytest.approx([0.3, 0.3, 0.4], abs=0.04)


def test_the_spectrum_with_an_axial_range_gives_the_fractions_of_the_particles():
    from smappy.tracking.spectrum import spectrum
    locs = simulate_tracks(n_frames=2000, density_um2=0.02, d_um2_s=(0.05, 3.0),
                           fractions=(0.5, 0.5), track_frames=30.0, precision_nm=25.0,
                           focal_range_nm=700.0, seed=3)
    t = tracks_of(locs)
    plain, fixed = spectrum(t), spectrum(t, focal_range_nm=700.0)
    assert plain.fraction_between(0.5, 100) < 0.45
    assert fixed.fraction_between(0.5, 100) == pytest.approx(0.5, abs=0.04)
    # each piece's D tells its state
    state = np.asarray(locs["state_true"])[t.rows][fixed.pieces.first]
    assert np.mean((fixed.D_post_um2_per_s > 0.4) == (state == 1)) > 0.95


def test_the_plugin_writes_each_pieces_d_onto_its_localizations():
    locs = simulate_tracks(n_frames=800, density_um2=0.02, d_um2_s=(0.02, 1.0),
                           fractions=(0.5, 0.5), track_frames=30.0, seed=9)
    settings = TrackingAnalysis.Settings(id_column="track_id_true", msd=False,
                                         jumps=False, spectrum=True)
    settings.spectrum_fit.write_column = True
    plugin = TrackingAnalysis()
    assert plugin.changes_locs_for(settings)
    result = plugin.run(Context(locs=locs), settings)
    v = result.values
    assert v["spectrum_D1_um2_per_s"] == pytest.approx(0.02, rel=0.25)
    assert v["spectrum_D2_um2_per_s"] == pytest.approx(1.0, rel=0.25)
    D = np.asarray(result.locs["D_post_um2_per_s"])
    state = np.asarray(locs["state_true"])
    have = np.isfinite(D)
    assert have.mean() > 0.9                    # all but tracks of one localization
    assert np.mean((D[have] > 0.15) == (state[have] == 1)) > 0.95
    assert "D_post_um2_per_s" in {r["field"] for r in recipes(result.locs)}
    assert result.plot is not None


# ------------------------------------------------------- state switching

def switching_tracks(**values):
    base = dict(n_frames=1500, density_um2=0.02, d_um2_s=(0.02, 1.0), fractions=(0.5, 0.5),
                switch_probability=0.05, track_frames=40.0, precision_nm=25.0, seed=4)
    base.update(values)
    return simulate_tracks(**base)


def test_with_one_state_the_filter_is_the_exact_likelihood_of_a_track():
    # nothing is merged with one state, so the filter must agree with the
    # Gaussian of the whole jump vector, shared localizations and all
    from smappy.tracking.hmm import HMMParams, steps_of, track_loglik
    from smappy.tracking.spectrum import piece_loglik, pieces_of
    locs = simulate_tracks(n_frames=200, density_um2=0.02, d_um2_s=(0.3,), p_dark=0.1,
                           track_frames=20.0, precision_nm=40.0, seed=3)
    t = tracks_of(locs)
    s = steps_of(t)
    filtered = track_loglik(s, HMMParams(np.array([0.3]), np.zeros((1, 1)), np.ones(1)))
    pieces = pieces_of(t, 10_000)
    exact = piece_loglik(t, pieces, np.array([0.3]))[:, 0]
    by_track = dict(zip(pieces.track, exact))
    assert filtered == pytest.approx([by_track[k] for k in s.track], rel=1e-9)


def test_the_hmm_gives_back_the_rates_and_the_states_of_switching_particles():
    from smappy.tracking.hmm import fit_hmm, per_localization
    locs = switching_tracks()
    t = tracks_of(locs)
    fit = fit_hmm(t, 2)
    p = fit.params
    assert p.D_um2_per_s == pytest.approx([0.02, 1.0], rel=0.1)
    # a switch drawn with 5% a frame, half the time to the other state
    assert [p.Q_per_s[0, 1], p.Q_per_s[1, 0]] == pytest.approx([2.5, 2.5], rel=0.15)
    assert fit.fractions == pytest.approx([0.5, 0.5], abs=0.04)
    assert p.dwell_s() == pytest.approx([0.4, 0.4], rel=0.15)
    prob, state = per_localization(t, p, len(locs))
    truth = np.asarray(locs["state_true"])
    have = state >= 0
    assert np.mean(state[have] == truth[have]) > 0.95
    assert np.allclose(prob[have].sum(axis=1), 1.0)


@pytest.mark.slow
def test_the_hmm_gives_back_three_states_and_their_six_rates():
    from smappy.tracking.hmm import fit_hmm
    locs = switching_tracks(d_um2_s=(0.02, 0.3, 2.0), fractions=(0.3, 0.3, 0.4))
    fit = fit_hmm(tracks_of(locs), 3)
    p = fit.params
    assert p.D_um2_per_s == pytest.approx([0.02, 0.3, 2.0], rel=0.1)
    # into state b at 5% a frame times its fraction
    truth = np.array([[0, 1.5, 2.0], [1.5, 0, 2.0], [1.5, 1.5, 0]])
    off = ~np.eye(3, dtype=bool)
    assert p.Q_per_s[off] == pytest.approx(truth[off], rel=0.3)
    assert fit.fractions == pytest.approx([0.3, 0.3, 0.4], abs=0.04)


def test_the_plugin_writes_each_localizations_state():
    locs = switching_tracks(n_frames=800)
    settings = TrackingAnalysis.Settings(id_column="track_id_true", msd=False,
                                         jumps=False, switching=True)
    settings.switching_fit.write_columns = True
    plugin = TrackingAnalysis()
    assert plugin.changes_locs_for(settings)
    result = plugin.run(Context(locs=locs), settings)
    v = result.values
    assert v["hmm_D2_um2_per_s"] == pytest.approx(1.0, rel=0.1)
    assert v["hmm_rate12_per_s"] == pytest.approx(2.5, rel=0.25)
    assert v["hmm_fraction1"] + v["hmm_fraction2"] == pytest.approx(1.0)
    state = np.asarray(result.locs["hmm_state"])
    truth = np.asarray(locs["state_true"])
    assert np.mean(state[state >= 0] == truth[state >= 0]) > 0.95
    assert {"hmm_state", "hmm_p1", "hmm_p2"} <= {r["field"] for r in recipes(result.locs)}
    assert result.plot is not None


def test_the_spacing_measured_per_frame_is_the_nearest_neighbour_distance_of_the_density():
    from smappy.simulate import spacing_nm
    from smappy.tracking.hmm import nearest_spacing_nm
    rng = np.random.default_rng(5)
    density, side, frames = 0.2, 40_000.0, 50                # per um^2, nm
    n = rng.poisson(density * (side / 1000) ** 2, frames)
    frame = np.repeat(np.arange(frames), n)
    x, y = rng.uniform(0, side, (2, len(frame)))
    # the field's edges lengthen it a little
    assert nearest_spacing_nm(x, y, frame) == pytest.approx(spacing_nm(density), rel=0.05)


def test_state_switching_warns_when_the_fast_jump_is_large_against_the_spacing():
    settings = TrackingAnalysis.Settings(id_column="track_id_true", msd=False,
                                         jumps=False, switching=True)
    def warnings_at(density):
        locs = switching_tracks(density_um2=density, n_frames=150)
        ctx = Context(locs=locs)
        result = TrackingAnalysis().run(ctx, settings)
        return result.values["hmm_crowding"], [w for w in ctx.warnings if "between" in w]
    # a 200 nm jump: 0.4 of the spacing at 1 /um^2, 0.09 at 0.05
    crowded, warned = warnings_at(1.0)
    assert crowded == pytest.approx(0.4, rel=0.15) and warned
    sparse, quiet = warnings_at(0.05)
    assert sparse < 0.12 and not quiet
