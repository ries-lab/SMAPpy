"""Redundant cross-correlation: recover a known drift from images."""
import numpy as np
import pytest

from smappy.rcc import RCCSettings, estimate_drift_rcc, _solve
from tests.test_drift import simulate


def test_recovers_known_drift():
    locs, truth, _ = simulate(n_points=300, n_frames=200, per_frame=200)
    drift = estimate_drift_rcc(locs, RCCSettings(
        n_timepoints=10, pixelsize_nm=10.0, max_drift_nm=300.0,
        z_pixelsize_nm=10.0, tile_nm=250.0, axial_max_drift_nm=300.0))

    assert len(drift) == 200
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error[:, :2]).mean() < 15.0   # image-based: a fraction of a pixel
    assert np.abs(error[:, 2]).mean() < 15.0    # z, from sliced 1-D correlations


def test_the_redundant_solve_outvotes_a_bad_pair():
    true = np.array([0.0, 10.0, 25.0, 5.0, -12.0])
    true -= true.mean()
    shifts = true[:, None] - true[None, :]
    shifts[0, 3] = 400.0                  # one correlation peak found in the wrong place
    shifts[3, 0] = -400.0

    assert np.abs(_solve(shifts, 5) - true).max() < 1.0


def test_every_pair_is_used():
    # with only consecutive pairs the middle window would be unconstrained
    true = np.array([0.0, 3.0, -6.0, 2.0]); true -= true.mean()
    shifts = true[:, None] - true[None, :]
    assert np.allclose(_solve(shifts, 4), true, atol=1e-6)


def test_a_table_corrected_by_rcc_says_it_was_rcc(tmp_path):
    """The drift curve records its estimator, through the table's metadata,
    the saved result and a saved file -- it used to say COMET whatever made it."""
    from smappy.drift import Drift, load_drift, save_drift_corrected
    from smappy.locs import Localizations
    locs, _, _ = simulate()
    assert estimate_drift_rcc(locs, RCCSettings(n_timepoints=5)).method == "rcc"
    drift = Drift(np.zeros((10, 3)), method="rcc")
    table = Localizations({"frame": np.arange(10), "x_nm": np.zeros(10),
                           "y_nm": np.zeros(10)}, {})
    assert drift.apply(table).metadata["drift_correction"]["method"] == "rcc"
    assert Drift.from_dict(drift.to_dict()).method == "rcc"
    path = save_drift_corrected(tmp_path / "t.hdf5", drift.apply(table), drift)
    assert load_drift(path).method == "rcc"
    prepass = Drift(np.zeros((10, 3)), method="rcc+comet")
    path = save_drift_corrected(tmp_path / "p.hdf5", prepass.apply(table), prepass)
    assert load_drift(path).method == "rcc+comet"
