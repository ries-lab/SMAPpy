"""Analysis/Measure/Ground Truth: a fit of a simulation scored against it."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.ground_truth import (GroundTruth, GroundTruthSettings, compare,
                                         match, truth_for)
from smappy.simulate import (CameraOutputSettings, LabellingSettings,
                             SimulationSettings, simulate)


def _table(x, y, frame, **more):
    cols = {"x_nm": np.asarray(x, float), "y_nm": np.asarray(y, float),
            "frame": np.asarray(frame, np.int64)}
    cols.update({k: np.asarray(v, float) for k, v in more.items()})
    return Localizations(cols, {"units": "nm"})


def test_matching_is_one_to_one_within_a_frame_and_the_radius():
    fit = np.array([[0, 0], [30, 0], [500, 0], [0, 0]], float)
    true = np.array([[5, 0], [20, 0], [0, 0]], float)
    fi, ti = match(fit, [0, 0, 0, 1], true, [0, 0, 2], radius=50)
    pairs = sorted(zip(fi.tolist(), ti.tolist()))
    # fit 2 is beyond the radius, fit 3 and truth 2 are alone in their frames
    assert len(pairs) == 2 and {p[0] for p in pairs} == {0, 1}
    assert {p[1] for p in pairs} == {0, 1}


def test_the_assignment_is_the_least_total_distance_not_greedy():
    # fit a sits nearest truth 1, but giving it truth 1 leaves fit b nothing
    fit = np.array([[0, 0], [-40, 0]], float)
    true = np.array([[-45, 0], [5, 0]], float)
    fi, ti = match(fit, [0, 0], true, [0, 0], radius=20)
    assert dict(zip(fi.tolist(), ti.tolist())) == {0: 1, 1: 0}


def test_the_counts_errors_and_pulls_come_back_from_a_known_table():
    rng = np.random.default_rng(0)
    n = 4000
    frame = np.repeat(np.arange(n // 4), 4)
    tx, ty = rng.uniform(0, 1e4, n), rng.uniform(0, 1e4, n)
    tz = rng.uniform(-400, 400, n)
    truth = _table(tx, ty, frame, z_nm=tz, photons=np.full(n, 1000.0))
    err = rng.uniform(5, 15, n)
    keep = rng.random(n) > 0.1                                      # 10 % missed
    fx = tx + rng.normal(0, err) + 2.0                               # a 2 nm bias
    fy = ty + rng.normal(0, err)
    fz = 0.9 * tz + rng.normal(0, 3 * err)                          # z scaled
    extra = 200
    fitted = _table(np.r_[fx[keep], rng.uniform(0, 1e4, extra)],
                    np.r_[fy[keep], rng.uniform(0, 1e4, extra)],
                    np.r_[frame[keep], rng.integers(0, n // 4, extra)],
                    z_nm=np.r_[fz[keep], np.zeros(extra)],
                    xy_err_nm=np.r_[err[keep], np.full(extra, 10.0)],
                    z_err_nm=np.r_[3 * err[keep], np.full(extra, 30.0)],
                    photons=np.full(keep.sum() + extra, 1000.0))
    c = compare(fitted, truth, radius=100)
    # a random extra can land on a missed spot, rarely: 200 in 10^4 x 10^4
    assert c.fn == pytest.approx(n - keep.sum(), abs=3)
    assert c.fp == pytest.approx(extra, abs=3)
    assert c.jaccard == pytest.approx(keep.sum() / (n + extra), abs=0.002)
    assert c.axes["x"]["bias_nm"] == pytest.approx(2.0, abs=0.5)
    assert c.axes["x"]["pull"] == pytest.approx(1.0, abs=0.05)
    assert c.axes["y"]["pull"] == pytest.approx(1.0, abs=0.05)
    assert c.z_slope == pytest.approx(0.9, abs=0.01)


def test_spots_that_do_not_count_set_their_fits_aside():
    truth = _table([0, 1000], [0, 0], [0, 0], photons=[20, 2000])
    fitted = _table([3, 1002], [0, 0], [0, 0], xy_err_nm=[10, 10], photons=[20, 2000])
    c = compare(fitted, truth, 50, counted=np.array([False, True]))
    assert (c.tp, c.fp, c.fn, c.set_aside) == (1, 0, 0, 1)


def test_a_simulated_table_is_scored_against_its_own_truth():
    """The localization simulator's errors are drawn from its precision, and
    what it removes as too close is what it misses."""
    locs = simulate(n_frames=2000, seed=1)
    result = GroundTruth().run(Context(locs=locs), GroundTruthSettings())
    c = result.data["comparison"]
    assert c["correct"] > 0.99
    # every localization is found, false or set aside; the dim ones -- kept
    # down to 10 photons, as SMAP keeps them -- are not scored
    assert c["tp"] + c["fp"] + c["set_aside"] == len(locs)
    assert c["tp"] == pytest.approx(np.sum(locs["photons"] >= 100), rel=0.02)
    assert locs["photons"].min() >= 10 and np.sum(locs["photons"] < 100) > 100
    for axis in "xyz":
        assert c["axes"][axis]["pull"] == pytest.approx(1.0, abs=0.05), axis
    assert c["z_slope"] == pytest.approx(1.0, abs=0.01)
    assert "Jaccard" in result.text


def test_drift_is_taken_out_of_the_truth_when_the_table_has_been_corrected():
    locs = simulate(n_frames=2000, seed=2, drift=True)
    drift = np.asarray(locs.metadata["drift_truth"])[locs["frame"]]
    corrected = Localizations({**locs.columns, "x_nm": locs["x_nm"] - drift[:, 0],
                               "y_nm": locs["y_nm"] - drift[:, 1],
                               "z_nm": locs["z_nm"] - drift[:, 2]}, locs.metadata)
    plugin = GroundTruth()
    right = plugin.run(Context(locs=corrected), GroundTruthSettings(drift_corrected=True))
    wrong = plugin.run(Context(locs=corrected), GroundTruthSettings())
    assert right.data["comparison"]["axes"]["x"]["pull"] == pytest.approx(1.0, abs=0.05)
    assert (wrong.data["comparison"]["axes"]["x"]["rmse_nm"]
            > 3 * right.data["comparison"]["axes"]["x"]["rmse_nm"])


def test_a_fit_of_simulated_frames_finds_the_isolated_spots_and_is_honest(tmp_path):
    from smappy.plugins import get
    from smappy.plugins.fit import GaussianFit, GaussianFitSettings, OutputSettings, SourceSettings
    settings = SimulationSettings(
        output="camera", n_frames=300, seed=2,
        labelling=LabellingSettings(efficiency=0.1),
        camera=CameraOutputSettings(path=str(tmp_path / "run.sim.yaml")))
    made = get("File/Simulate/Blinking Structure")().run(Context(), settings)
    fitted = GaussianFit().run(Context(), GaussianFitSettings(
        source=SourceSettings(path=made.data["path"]),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))).locs
    assert fitted.metadata["source"] == made.data["path"]
    result = GroundTruth().run(Context(locs=fitted), GroundTruthSettings(
        min_photons=300, isolated=True))
    c = result.data["comparison"]
    assert c["recall"] > 0.98 and c["correct"] > 0.98
    for axis in "xy":
        assert abs(c["axes"][axis]["bias_nm"]) < 1.0
        assert c["axes"][axis]["pull"] == pytest.approx(1.0, abs=0.1)
    # counting every spot, crowding and slivers cost detections
    everything = GroundTruth().run(Context(locs=fitted), GroundTruthSettings())
    assert everything.data["comparison"]["jaccard"] < c["jaccard"]


def test_a_table_that_is_no_simulation_is_refused_with_a_reason():
    table = _table([0], [0], [0])
    with pytest.raises(ValueError, match="nothing of a simulation"):
        truth_for(table)
    table.metadata["source"] = "/data/run.tif"
    with pytest.raises(ValueError, match="not a simulation"):
        truth_for(table)


def test_a_z_radius_refuses_pairs_that_agree_only_laterally():
    truth = _table([0, 1000], [0, 0], [0, 0], z_nm=[0, 0], photons=[1000, 1000])
    fitted = _table([5, 1005], [0, 0], [0, 0], z_nm=[50, 500], xy_err_nm=[10, 10],
                    z_err_nm=[30, 30], photons=[1000, 1000])
    lateral = compare(fitted, truth, 100)
    both = compare(fitted, truth, 100, z_radius=300)
    assert (lateral.tp, lateral.fp, lateral.fn) == (2, 0, 0)
    assert (both.tp, both.fp, both.fn) == (1, 1, 1)


def test_a_drift_corrected_table_is_scored_without_the_mean_drift_as_a_bias():
    """A drift estimate is fixed up to a constant (RCC and COMET make it
    average zero); taking out the whole true drift left its mean as a bias,
    40 nm here, and the error over the precision at six."""
    from smappy.plugins import Context
    from smappy.plugins.ground_truth import GroundTruth, GroundTruthSettings
    from smappy.rcc import RCCSettings, estimate_drift_rcc
    from smappy.simulate import simulate
    locs = simulate(n_frames=6000, seed=1, drift=True)
    corrected = estimate_drift_rcc(locs, RCCSettings(n_timepoints=10)).apply(locs)
    c = GroundTruth().run(Context(locs=corrected), GroundTruthSettings(
        drift_corrected=True)).data["comparison"]
    for axis in ("x", "y"):
        assert abs(c["axes"][axis]["bias_nm"]) < 1.0, axis
        # what is left above one is RCC's own error on the drift, which is real
        assert c["axes"][axis]["pull"] < 2.0, axis
