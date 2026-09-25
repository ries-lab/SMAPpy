"""Drift correction: recover a known drift, and correct everything with it."""
import re

import numpy as np
import pytest

from smappy.drift import (Drift, DriftCost, DriftSettings, correct_drift,
                           drift_corrected_path, estimate_cost, estimate_drift,
                           load_drift, save_drift_corrected)
from smappy.filter import LocFilter
from smappy.locs import Localizations

# COMET is vendored (smappy._comet) and its cost function is compiled with the
# package, so drift correction has nothing optional left to skip on.
from smappy import _comet as comet                                # noqa: E402


def simulate(n_points=200, n_frames=50, per_frame=30, seed=1):
    """Localizations of a fixed random structure, moved by a known drift."""
    rng = np.random.default_rng(seed)
    structure = rng.random((n_points, 3)) * 1000.0
    t = np.arange(n_frames)
    base = (3 * np.sin(t / 20.0 + 0.37) + 1.25 * np.cos(t / 6.0 + 1.2) + t / 25) * 10.0
    drift = np.column_stack([base, 0.7 * base, -0.5 * base])

    frame = np.repeat(t, per_frame)
    rng.shuffle(frame)
    true_xyz = structure[rng.integers(0, n_points, frame.size)]
    xyz = true_xyz + drift[frame]
    locs = Localizations({"frame": frame.astype(np.int64),
                          "x_nm": xyz[:, 0].astype(np.float32),
                          "y_nm": xyz[:, 1].astype(np.float32),
                          "z_nm": xyz[:, 2].astype(np.float32),
                          "xy_err_nm": rng.random(frame.size).astype(np.float32) * 40},
                         {"units": "nm"})
    return locs, drift, true_xyz


# the per-window fit; the package default is the spline, tested separately
SETTINGS = DriftSettings(segmentation_var=2, initial_sigma_nm=120, max_drift_nm=100,
                         target_sigma_nm=10, backend="cpu", spline=False, group=False)


def test_recovers_known_drift():
    locs, truth, _ = simulate()
    drift = estimate_drift(locs, SETTINGS)

    assert len(drift) == 50
    # both are defined up to a constant offset: compare after centring
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 1.0


def test_corrects_localizations_the_filter_hides():
    locs, _, true_xyz = simulate()
    # estimate from the good half only, correct all of it
    keep = LocFilter(locs, xy_err_nm=(None, 20))
    corrected, drift = correct_drift(locs, SETTINGS, select=keep)

    assert drift.n_used == len(keep.indices) < len(locs)
    assert len(corrected) == len(locs)
    # the hidden localizations must land back on the undrifted structure
    # (up to the constant offset drift is defined against)
    hidden = ~keep.mask
    for axis, column in enumerate(("x_nm", "y_nm", "z_nm")):
        residual = corrected[column][hidden] - true_xyz[hidden, axis]
        assert residual.std() < 1.0
        assert (locs[column][hidden] - true_xyz[hidden, axis]).std() > 5


def test_pixel_table_uses_the_pixel_size():
    locs, _, true_xyz = simulate()
    pixels = Localizations({"frame": locs["frame"],
                            "x_pix": locs["x_nm"] / 100.0,
                            "y_pix": locs["y_nm"] / 100.0},
                           {"units": "pixel", "pixelsize_nm": 100.0})
    drift = estimate_drift(locs, SETTINGS)
    corrected = drift.apply(pixels)

    assert np.allclose(corrected["x_pix"] * 100.0,
                       drift.apply(locs)["x_nm"], atol=1e-2)


def test_save_and_reload(tmp_path):
    locs, _, true_xyz = simulate()
    drift = Drift(np.arange(150, dtype=float).reshape(50, 3))
    path = save_drift_corrected(tmp_path / "run_driftc.hdf5", drift.apply(locs), drift)

    from smappy.io.hdf5 import load_localizations
    assert len(load_localizations(path)) == len(locs)
    assert np.allclose(load_drift(path).drift, drift.drift)
    assert drift_corrected_path(tmp_path / "run.hdf5").name == "run_driftc.hdf5"


def test_drift_from_another_acquisition_is_refused():
    locs, _, true_xyz = simulate()
    with pytest.raises(ValueError, match="different acquisition"):
        Drift(np.zeros((10, 3))).apply(locs)


def test_grouped_estimate_still_corrects_every_localization():
    locs, truth, true_xyz = simulate()
    # two localizations of the same emitter in consecutive frames, so grouping
    # has something to collapse
    settings = DriftSettings(segmentation_var=2, initial_sigma_nm=120, max_drift_nm=100,
                             target_sigma_nm=10, backend="cpu", spline=False,
                             group=True, group_dx_nm=30.0)
    corrected, drift = correct_drift(locs, settings)

    assert drift.n_used < len(locs)          # grouping collapsed rows
    assert len(corrected) == len(locs)       # every row is still corrected
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 5.0


def test_two_stage_matches_the_single_pass():
    locs, truth, _ = simulate()
    single = estimate_drift(locs, SETTINGS)
    two = estimate_drift(locs, DriftSettings(
        segmentation_var=2, backend="cpu", spline=False, two_stage=True,
        group_dx_nm=30.0, two_stage_radius_nm=60.0))

    for drift in (single, two):
        error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
        assert np.abs(error).mean() < 5.0


def test_an_rcc_prepass_recovers_the_same_drift_over_a_small_radius():
    """RCC takes out the bulk, COMET refines what is left, and the two add.

    The point of it is the search radius: COMET sees a few tens of nanometres
    where a single pass has to look over the whole drift, and the pair count
    -- which is the wait -- goes with the radius.
    """
    locs, truth, _ = simulate()
    drift = estimate_drift(locs, DriftSettings(
        segmentation_var=2, backend="cpu", spline=False, group=False,
        rcc_prepass=True, rcc_prepass_windows=8, rcc_prepass_max_drift_nm=400.0,
        max_drift_nm=50.0, initial_sigma_nm=None, target_sigma_nm=10.0))
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 8.0


def test_the_slow_run_dialogue_offers_rcc_first():
    """"No" is not the only useful answer to "this will take an afternoon"."""
    from smappy.plugins import Context, PreflightQuestion
    from smappy.plugins.drift_comet import CometDrift

    rng = np.random.default_rng(0)
    n = 300_000
    locs = Localizations({"x_nm": rng.uniform(0, 20_000, n),
                          "y_nm": rng.uniform(0, 20_000, n),
                          "frame": np.sort(rng.integers(0, 20_000, n)).astype(np.int64),
                          "xy_err_nm": np.full(n, 12.0)}, {})
    plugin = CometDrift()
    asked = DriftSettings(max_drift_nm=1500.0)
    question = plugin.preflight(Context(locs=locs), asked)
    assert isinstance(question, PreflightQuestion) and len(question.choices) == 1
    cheap = question.choices[0].settings
    assert cheap.rcc_prepass and cheap.max_drift_nm == plugin.prepass_max_drift_nm
    assert cheap.rcc_prepass_max_drift_nm == asked.max_drift_nm
    # and it really is cheaper, which is the whole reason for offering it
    assert estimate_cost(locs, cheap).seconds < estimate_cost(locs, asked).seconds
    # a run that already has the prepass on is asked the plain question
    again = plugin.preflight(Context(locs=locs), cheap)
    assert again is None or isinstance(again, str)


def test_spline_drift_needs_no_time_windows():
    locs, truth, _ = simulate()
    drift = estimate_drift(locs, DriftSettings(
        backend="cpu", spline=True, group=False, spline_knot_frames=5,
        max_drift_nm=100, initial_sigma_nm=120, target_sigma_nm=10))

    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 1.0
    # the curve is the fit itself, evaluated per frame -- not interpolated
    assert len(drift) == 50


def test_spline_penalty_smooths():
    locs, _, _ = simulate()
    rough, smooth = (estimate_drift(locs, DriftSettings(
        backend="cpu", spline=True, group=False, spline_knot_frames=2,
        max_drift_nm=100, initial_sigma_nm=120, target_sigma_nm=10,
        spline_penalty=p)).drift
        for p in (0.0, 1e4))
    curvature = lambda d: np.abs(np.diff(d, 2, axis=0)).mean()
    assert curvature(smooth) < curvature(rough)


# --- the compiled cost function ---------------------------------------------
# COMET's reference implementation is the specification; the kernel that runs is
# C++ (csrc/drift.hpp), so what has to be checked is that they agree.

def _cost_case(seed=3, n=1500, pairs=15_000, segments=30):
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 3)) * 2000.0
    times = rng.integers(0, segments, n).astype(np.int64)
    idx_i = rng.integers(0, n, pairs).astype(np.int64)
    idx_j = rng.integers(0, n, pairs).astype(np.int64)
    mu = rng.normal(0, 20, (segments, 3))
    return coords, times, idx_i, idx_j, mu


def test_the_compiled_kernel_matches_comets_reference():
    from smappy import _drift
    from smappy._comet.core.cpu_wrapper import _cost_and_gradient_reference

    case = _cost_case()
    want_total, want_grad = _cost_and_gradient_reference(*case, 30.0, 1.0)
    got_total, got_grad = _drift.cost_and_gradient(*case, 30.0, 1.0)

    assert got_total == pytest.approx(want_total, rel=1e-12)
    assert np.allclose(got_grad, want_grad, rtol=1e-10, atol=1e-15)


def test_threading_does_not_change_the_answer():
    """The gradient is a scatter-add; per-thread buffers must sum to the same."""
    from smappy import _drift

    case = _cost_case()
    one = _drift.cost_and_gradient(*case, 30.0, 1.0, 0.0, 1)
    many = _drift.cost_and_gradient(*case, 30.0, 1.0, 0.0, 8)
    assert many[0] == pytest.approx(one[0], rel=1e-12)
    assert np.allclose(many[1], one[1], rtol=1e-10, atol=1e-15)


def test_the_cutoff_costs_little_accuracy():
    """Pairs beyond six sigma contribute less than the estimate's own noise."""
    from smappy import _drift

    case = _cost_case()
    exact_total, exact_grad = _drift.cost_and_gradient(*case, 30.0, 1.0)
    cut_total, cut_grad = _drift.cost_and_gradient(*case, 30.0, 1.0, 6.0)
    assert cut_total == pytest.approx(exact_total, rel=1e-3)
    assert np.abs(cut_grad - exact_grad).max() < 1e-2 * np.abs(exact_grad).max()


def test_the_kernel_takes_the_optimizers_own_dtypes():
    """float32 coords and int32 indices must not be copied on every call."""
    from smappy import _drift

    coords, times, idx_i, idx_j, mu = _cost_case()
    small = (coords.astype(np.float32), times.astype(np.int32),
             idx_i.astype(np.int32), idx_j.astype(np.int32), mu)
    wide = _drift.cost_and_gradient(coords, times, idx_i, idx_j, mu, 30.0, 1.0)
    narrow = _drift.cost_and_gradient(*small, 30.0, 1.0)
    # float32 coordinates, so agreement is to single precision, not double
    assert narrow[0] == pytest.approx(wide[0], rel=1e-5)


def test_quality_control_counts_match_the_reference():
    from smappy import _drift

    coords, times, idx_i, idx_j, mu = _cost_case()
    observed, null, counts = _drift.overlap_per_segment(
        coords, times, idx_i, idx_j, mu, 30.0, 1.0)
    cross = times[idx_i] != times[idx_j]
    assert counts.sum() == pytest.approx(2 * cross.sum())
    assert (observed >= 0).all() and (null >= 0).all()


def test_drift_correction_needs_no_plotting_library(monkeypatch):
    """A bare `pip install smappy-smlm` has no matplotlib; drift must still run.

    COMET imports pyplot at module level for diagnostics that are all behind
    flags, which made a correction that draws nothing depend on a plotting
    library.  The vendored copy imports it where it is used.
    """
    import subprocess
    import sys

    code = '''
import sys
class Blocker:
    def find_module(self, name, path=None):
        if name.split(".")[0] == "matplotlib":
            return self
    def load_module(self, name):
        raise ImportError("no matplotlib")
sys.meta_path.insert(0, Blocker())

import numpy as np
from smappy.drift import DriftSettings, estimate_drift
from smappy.locs import Localizations

rng = np.random.default_rng(0)
n = 2000
locs = Localizations({"x_nm": rng.uniform(0, 3000, n), "y_nm": rng.uniform(0, 3000, n),
                      "frame": rng.integers(0, 60, n).astype(np.int64),
                      "xy_err_nm": np.full(n, 5.0)}, {"units": "nm"})
drift = estimate_drift(locs, DriftSettings(segmentation_var=10, use_z=False))
assert len(drift.frames) > 0
assert "matplotlib" not in sys.modules
print("ok")
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


# --------------------------------------------- what it will cost, before running

def _pairs_reported(lines):
    """The pair count the estimator says it actually built."""
    match = re.search(r"([\d,]+) pairs", " ".join(lines))
    return int(match.group(1).replace(",", ""))


@pytest.mark.parametrize("group", [False, True])
def test_the_cost_estimate_predicts_the_pairs_that_get_built(group):
    """The estimate is the whole point: it has to be right before the run.

    Pairs are counted on a sample and scaled by N^2, and with grouping on the
    sample is grouped too -- so this compares the prediction against the pair
    count the real run reports, on a table large enough (80 k) that the
    sampling is exercised rather than bypassed.
    """
    locs, _, _ = simulate(n_points=500, n_frames=800, per_frame=100)
    settings = DriftSettings(max_drift_nm=150, target_sigma_nm=30, group=group)

    cost = estimate_cost(locs, settings)
    lines = []
    estimate_drift(locs, settings, progress=lines.append)

    assert 0.8 < cost.n_pairs / _pairs_reported(lines) < 1.25
    assert cost.seconds > 0 and cost.memory_bytes > 0


def test_the_cost_grows_with_the_search_radius():
    """`max_drift_nm` is the neighbour radius, which is why a round number
    picked for safety is expensive rather than free."""
    locs, _, _ = simulate(n_points=300, n_frames=400, per_frame=40)
    near = estimate_cost(locs, DriftSettings(max_drift_nm=100, group=False))
    far = estimate_cost(locs, DriftSettings(max_drift_nm=300, group=False))
    assert far.n_pairs > 3 * near.n_pairs
    assert far.seconds > near.seconds


def test_a_cheap_run_is_not_worth_asking_about():
    locs, _, _ = simulate()
    cost = estimate_cost(locs, DriftSettings(max_drift_nm=100, group=False))
    assert cost.question() is None


def test_a_slow_run_asks_first_and_says_what_it_will_cost():
    cost = DriftCost(n_localizations=2_000_000, n_pairs=4e9, n_evaluations=138,
                     seconds=1400.0, memory_bytes=96e9, max_drift_nm=500.0,
                     machine_bytes=34_000_000_000)
    question = cost.question()
    assert question is not None
    assert "23 min" in question and "96.0 GB" in question
    assert not cost.fits_in_memory        # 96 GB into 34 GB of machine
    assert "max drift" in question        # and what to do about it


def test_a_run_that_fits_and_is_quick_enough_asks_nothing():
    cost = DriftCost(n_localizations=200_000, n_pairs=3e8, n_evaluations=138,
                     seconds=90.0, memory_bytes=7e9, max_drift_nm=150.0,
                     machine_bytes=34_000_000_000)
    assert cost.fits_in_memory and cost.question() is None


def test_an_empty_selection_costs_nothing_rather_than_raising():
    locs, _, _ = simulate()
    cost = estimate_cost(locs, DriftSettings(), select=np.zeros(len(locs), bool))
    assert cost.n_pairs == 0 and cost.question() is None


# ------------------------------------------------------------------- progress

def test_the_estimate_reports_each_stage_it_spends_time_in():
    """A drift estimate is minutes of one call; silence is indistinguishable
    from a hang, which is what this is here to prevent."""
    locs, _, _ = simulate(n_points=300, n_frames=400, per_frame=40)
    lines = []
    correct_drift(locs, DriftSettings(max_drift_nm=150, target_sigma_nm=30,
                                      group=True),
                  progress=lines.append)
    text = "\n".join(lines)
    assert "grouping" in text                  # what it is doing now
    assert "neighbour pairs within 150 nm" in text
    assert "sigma" in text                     # and how far through it is
    assert "applying the drift" in text
    assert any("left" in line for line in lines)   # with an estimate of what is left


def test_progress_is_optional_everywhere():
    """Every caller that does not want progress must be unaffected by it."""
    locs, _, _ = simulate()
    drift = estimate_drift(locs, DriftSettings(max_drift_nm=100, group=False))
    assert len(drift) == 50
