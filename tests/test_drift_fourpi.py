"""Drift correction on a 4Pi table's phase z (docs/4pi.md §31).

Simulated 4Pi fits (`smappy.simulate.fourpi_drift`): a filament sample
drifting in x, y and z, a cavity wandering on its own, *4Pi z* run on them
as on a real fit -- so the ghosts and the cavity phase it misses are its
own, not drawn.
"""
from dataclasses import replace

import numpy as np
import pytest

from smappy.drift import Drift, DriftSettings, estimate_drift
from smappy.plugins import Context
from smappy.plugins.fourpi_z import FourPiZ, FourPiZSettings
from smappy.rcc import RCCSettings, estimate_drift_rcc
from smappy.simulate import apparent_drift, simulate_fourpi_drift
from smappy.simulate.fourpi_drift import cavity_wander, fourpi_drift


def slow(n_frames):
    """The drift and the cavity of the first ``n_frames`` of a 40 000-frame
    movie: an hour's drift is tens of nm, not tens of nm a minute."""
    return dict(drift_nm=fourpi_drift(40000)[:n_frames],
                cavity_rad=cavity_wander(40000, np.random.default_rng(1))[:n_frames])


def fourpi_z(locs, **settings):
    return FourPiZ().run(Context(locs=locs), FourPiZSettings(**settings)).locs


def rms(estimate, truth, axis):
    e = estimate[:, axis] - estimate[:, axis].mean() - (truth[:, axis] - truth[:, axis].mean())
    return float(np.sqrt(np.mean(e ** 2)))


@pytest.fixture(scope="module")
def tubulin():
    raw, truth = simulate_fourpi_drift(n_frames=8000, per_frame=15, seed=3)
    return fourpi_z(raw), truth


def test_z_on_knots_of_its_own_follows_the_cavity_that_4pi_z_missed_and_leaves_x_and_y_alone(tubulin):
    locs, truth = tubulin
    seen = apparent_drift(truth, locs)
    # what *4Pi z*'s 1000-frame windows leave of the cavity, as z: a few nm
    assert np.std(seen[:, 2] - truth.drift_nm[:, 2]) > 1.5
    one = estimate_drift(locs, DriftSettings(split_half=False, z_knot_frames=2000))
    # on auto a 4Pi table's z gets an eighth of the knot spacing: 250 frames
    own = estimate_drift(locs, DriftSettings(split_half=False))
    assert rms(own.drift, seen, 2) < 0.6 * rms(one.drift, seen, 2)
    assert rms(own.drift, seen, 2) < 1.2
    for axis in (0, 1):
        assert abs(rms(own.drift, truth.drift_nm, axis) - rms(one.drift, truth.drift_nm, axis)) < 0.05
        assert rms(own.drift, truth.drift_nm, axis) < 0.6


def test_ghosts_a_fringe_away_do_not_move_the_drift_estimate():
    # an envelope four times as wrong as it says: one localization in six
    # on another fringe, 294 nm off -- far beyond the overlap's reach
    raw, truth = simulate_fourpi_drift(n_frames=6000, per_frame=15, envelope_scale=4.0,
                                       **slow(6000), seed=5)
    locs = fourpi_z(raw)
    z = np.asarray(locs["z_nm"], float)
    wrong = np.abs(z - truth.xyz_nm[:, 2]) > 147
    assert 0.1 < wrong.mean() < 0.3
    seen = apparent_drift(truth, locs)
    for estimate, settings in ((estimate_drift, DriftSettings(split_half=False, sure_fringe=0)),
                               (estimate_drift_rcc, RCCSettings(split_half=False, sure_fringe=0))):
        every = estimate(locs, settings)
        right = estimate(locs, settings, select=~wrong)
        # with the ghosts the curve is barely worse, and from the sure
        # fringes (the default) as good as from the right ones alone
        sure = estimate(locs, replace(settings, sure_fringe=0.9))
        assert rms(every.drift, seen, 2) < 1.1 * rms(right.drift, seen, 2) < 2.0
        assert rms(sure.drift, seen, 2) < 1.03 * rms(right.drift, seen, 2)
        assert sure.n_used < 0.9 * every.n_used


def test_a_drift_correction_moves_the_envelope_z_with_z_so_4pi_z_run_again_keeps_it():
    raw, truth = simulate_fourpi_drift(n_frames=3000, per_frame=15, **slow(3000), seed=7)
    locs = fourpi_z(raw)
    drift = Drift(truth.drift_nm)
    corrected = drift.apply(locs)
    frame = np.asarray(locs["frame"])
    for name in ("z_nm", "z_env_nm"):
        assert np.allclose(np.asarray(corrected[name], float),
                           np.asarray(locs[name], float) - truth.drift_nm[frame, 2], atol=1e-2)
    assert "z_env_nm" in corrected.metadata["drift_correction"]["columns"]
    # *4Pi z* reads the envelope and the phase, never its own z: with the
    # envelope moved, the cavity phase takes up the shift and z stays put --
    # as far as its windows follow it, which a drift as slow as a sample's is
    again = fourpi_z(corrected)
    p = np.asarray(again["fringe_probability"], float)
    sure = p > 0.99
    difference = np.asarray(again["z_nm"], float)[sure] - np.asarray(corrected["z_nm"], float)[sure]
    assert np.median(np.abs(difference - np.median(difference))) < 1.0
    assert abs(np.median(difference)) < 2.0


def test_only_a_4pi_table_gets_z_knots_of_its_own_on_auto(tubulin):
    from smappy.drift import z_knot_frames
    locs, _ = tubulin
    settings = DriftSettings()
    assert z_knot_frames(settings, locs) == 250
    from smappy.locs import Localizations
    plain = Localizations({c: locs[c] for c in ("frame", "x_nm", "y_nm", "z_nm")}, {})
    assert z_knot_frames(settings, plain) == 2000
    assert z_knot_frames(DriftSettings(z_knot_frames=700), locs) == 700
