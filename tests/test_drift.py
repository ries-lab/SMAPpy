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
                         target_sigma_nm=10, spline=False, group=False)


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
                             target_sigma_nm=10, spline=False,
                             group=True, group_dx_nm=30.0)
    corrected, drift = correct_drift(locs, settings)

    assert drift.n_used < len(locs)          # grouping collapsed rows
    assert len(corrected) == len(locs)       # every row is still corrected
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 5.0


def test_the_drift_plugins_merge_blinks_as_the_session_does_unless_told_otherwise():
    from smappy.drift import merge_from_session
    from smappy.group import GroupSettings
    from smappy.plugins.drift_comet import CometDrift
    from smappy.plugins.drift_rcc import RCCDrift
    from smappy.rcc import RCCSettings
    from smappy.session import Session

    # no session: the session's own defaults, so a script's numbers do not move
    for settings in (DriftSettings(), RCCSettings()):
        alone = merge_from_session(settings, None)
        assert (alone.group_dx_nm, alone.group_dt) == (50.0, 1)
    session = Session()
    session.set_group_settings(GroupSettings(dx=30.0, dt=3))
    for settings in (DriftSettings(), RCCSettings()):
        used = merge_from_session(settings, session)
        assert (used.group_dx_nm, used.group_dt) == (30.0, 3)
        own = merge_from_session(type(settings)(group_dx_nm=80.0, group_dt=0), session)
        assert (own.group_dx_nm, own.group_dt) == (80.0, 0)
    for plugin, settings in ((CometDrift(), DriftSettings()), (RCCDrift(), RCCSettings())):
        hints = plugin.describe(session.context(), settings).hints
        assert {k: hints[k] for k in ("group_dx_nm", "group_dt")} == \
            {"group_dx_nm": 30.0, "group_dt": 3}
    # and in a two-colour fit's own drift step, which runs after the fit
    from smappy import plugins
    fit = plugins.get("Localize/Gaussian 2D 2C")()
    hints = fit.describe(session.context(), fit.Settings()).hints
    assert hints["finish.comet.group_dx_nm"] == 30.0 and hints["finish.rcc.group_dt"] == 3
    # by precision, the widest distance that mode allows
    session.set_group_settings(GroupSettings(link="precision", dx_max=120.0, dt=2))
    used = merge_from_session(DriftSettings(), session)
    assert (used.group_dx_nm, used.group_dt) == (120.0, 2)


def test_two_stage_matches_the_single_pass():
    locs, truth, _ = simulate()
    single = estimate_drift(locs, SETTINGS)
    two = estimate_drift(locs, DriftSettings(
        segmentation_var=2, spline=False, two_stage=True,
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
        segmentation_var=2, spline=False, group=False,
        rcc_prepass=True, rcc_prepass_windows=8, rcc_prepass_max_drift_nm=400.0,
        max_drift_nm=50.0, initial_sigma_nm=None, target_sigma_nm=10.0))
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 8.0


def test_an_rcc_prepass_runs_before_the_spline_fit_too():
    """The spline is the default, and the prepass used to be skipped under it.

    The spline branch returned before the prepass was looked at, so "RCC
    first" -- which the slow-run dialogue offers -- ran one spline fit over
    the prepass's small radius and could not see drift beyond it.
    """
    locs, truth, _ = simulate()
    told = []
    drift = estimate_drift(locs, DriftSettings(
        spline=True, spline_knot_frames=5, group=False,
        rcc_prepass=True, rcc_prepass_windows=8, rcc_prepass_max_drift_nm=400.0,
        max_drift_nm=30.0, initial_sigma_nm=None, target_sigma_nm=10.0),
        progress=told.append)
    assert any("RCC" in line for line in told)
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 8.0


def test_two_stage_runs_both_passes_with_the_spline_fit():
    locs, truth, _ = simulate()
    told = []
    drift = estimate_drift(locs, DriftSettings(
        spline=True, spline_knot_frames=5, two_stage=True,
        group_dx_nm=30.0, two_stage_radius_nm=60.0, max_drift_nm=100,
        initial_sigma_nm=120, target_sigma_nm=10), progress=told.append)
    assert any("pass 2 of 2" in line for line in told)
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 5.0


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
    # 2000 nm, not 1500: that estimated 299.3 s on Windows, against 300 for "slow"
    asked = DriftSettings(max_drift_nm=2000.0)
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
        spline=True, group=False, spline_knot_frames=5,
        max_drift_nm=100, initial_sigma_nm=120, target_sigma_nm=10))

    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    assert np.abs(error).mean() < 1.0
    # the curve is the fit itself, evaluated per frame -- not interpolated
    assert len(drift) == 50


def test_spline_penalty_smooths():
    locs, _, _ = simulate()
    rough, smooth = (estimate_drift(locs, DriftSettings(
        spline=True, group=False, spline_knot_frames=2,
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


@pytest.mark.parametrize("z_factor,weighted", [(2.5, False), (1.0, True), (2.0, True)])
def test_the_kernel_with_a_z_width_and_precisions_matches_the_reference(z_factor,
                                                                        weighted):
    """Per axis: z's own width, and each localization's own precision."""
    from smappy import _drift
    from smappy._comet.core.cpu_wrapper import _cost_and_gradient_reference

    case = _cost_case()
    rng = np.random.default_rng(8)
    variances = (rng.uniform(5, 40, (len(case[0]), 3)) ** 2) if weighted else None
    want_total, want_grad = _cost_and_gradient_reference(
        *case, 30.0, 1.0, z_factor=z_factor, variances=variances)
    got_total, got_grad = _drift.cost_and_gradient(
        *case, 30.0, 1.0, z_factor=z_factor, variances=variances)
    assert got_total == pytest.approx(want_total, rel=1e-12)
    assert np.allclose(got_grad, want_grad, rtol=1e-10, atol=1e-15)


def test_precise_localizations_weigh_what_they_did_without_weights():
    """Zero precision is COMET's kernel exactly, and a wider z is a z
    measured in larger units."""
    from smappy import _drift

    coords, times, idx_i, idx_j, mu = _cost_case()
    plain = _drift.cost_and_gradient(coords, times, idx_i, idx_j, mu, 30.0, 1.0)
    zero = _drift.cost_and_gradient(coords, times, idx_i, idx_j, mu, 30.0, 1.0,
                                    variances=np.zeros((len(coords), 3)))
    assert zero[0] == pytest.approx(plain[0], rel=1e-12)
    np.testing.assert_allclose(zero[1], plain[1], rtol=1e-12, atol=1e-18)

    scaled = coords * [1.0, 1.0, 0.5]
    wide = _drift.cost_and_gradient(coords, times, idx_i, idx_j, mu, 30.0, 1.0,
                                    z_factor=2.0)
    narrow = _drift.cost_and_gradient(scaled, times, idx_i, idx_j, mu * [1, 1, 0.5],
                                      30.0, 1.0)
    assert wide[0] == pytest.approx(narrow[0], rel=1e-12)


def test_an_imprecise_pair_counts_less():
    from smappy import _drift

    coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    times = np.array([0, 1])
    pair = (np.array([0]), np.array([1]))
    mu = np.zeros((2, 3))
    sharp = _drift.cost_and_gradient(coords, times, *pair, mu, 30.0, 1.0,
                                     variances=np.full((2, 3), 25.0))[0]
    blurred = _drift.cost_and_gradient(coords, times, *pair, mu, 30.0, 1.0,
                                       variances=np.full((2, 3), 900.0))[0]
    assert blurred < sharp


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


def test_quality_control_measures_z_in_its_own_width():
    """The overlap per window is the cost's, *z width* included: the sum of
    both over windows is twice the cost over the pairs between windows."""
    from smappy import _drift

    coords, times, idx_i, idx_j, mu = _cost_case()
    cross = times[idx_i] != times[idx_j]
    for z_factor in (1.0, 2.5):
        observed, _, _ = _drift.overlap_per_segment(
            coords, times, idx_i, idx_j, mu, 30.0, 1.0, z_factor=z_factor)
        total, _ = _drift.cost_and_gradient(coords, times, idx_i[cross], idx_j[cross],
                                            mu, 30.0, 1.0, z_factor=z_factor)
        assert observed.sum() == pytest.approx(2 * total, rel=1e-10)


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
@pytest.mark.slow
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


def test_a_subset_per_window_is_the_same_on_every_run():
    """The subset was drawn from numpy's global state, so two runs differed."""
    locs, _, _ = simulate()
    settings = DriftSettings(segmentation_var=5, spline=False,
                             group=False, max_locs_per_segment=100,
                             max_drift_nm=100, initial_sigma_nm=120,
                             target_sigma_nm=10)
    first = estimate_drift(locs, settings).drift
    np.random.seed(123)                      # whatever else has drawn numbers
    np.testing.assert_array_equal(first, estimate_drift(locs, settings).drift)


class _StopsAfter:
    """A stop that is set once it has been asked ``n`` times."""

    def __init__(self, n):
        self.n, self.asked = n, 0

    def is_set(self):
        self.asked += 1
        return self.asked > self.n


def test_comet_stopped_mid_fit_raises_stopped_and_corrects_nothing():
    """The spline fit is one optimisation: half refined is not a curve to
    subtract, so a stop leaves the table as it was."""
    import threading

    from smappy.plugins import Context, Stopped
    from smappy.plugins.drift_comet import CometDrift
    from dataclasses import replace
    spline = replace(SETTINGS, spline=True, spline_knot_frames=10)
    locs, _, _ = simulate()
    with pytest.raises(Stopped, match="COMET stopped at sigma"):
        estimate_drift(locs, spline, stop=_StopsAfter(5))
    stop = threading.Event()
    stop.set()
    with pytest.raises(Stopped, match="nothing was corrected"):
        CometDrift().run(Context(locs=locs, stop=stop), spline)
    # a stop never set changes nothing
    assert np.array_equal(estimate_drift(locs, spline).drift,
                          estimate_drift(locs, spline, stop=_StopsAfter(10**6)).drift)


# ---------------------------------------------------------------- a time axis

def _minflux_like(n=500, seed=3):
    """Localizations with no frames to speak of: times irregular, frame the rank."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(5.0, 95.0, n))
    return Localizations({"x_nm": rng.uniform(0, 1000, n), "y_nm": rng.uniform(0, 1000, n),
                          "z_nm": rng.uniform(-100, 100, n), "time_s": t,
                          "x_nc_nm": np.zeros(n), "frame": np.arange(n)}, {})


def test_a_drift_on_a_time_axis_moves_each_localization_by_the_curve_at_its_time():
    locs = _minflux_like()
    times = np.arange(10.0, 91.0, 2.0)                    # a point every 2 s, as MBM
    curve = np.column_stack([0.5 * times, -0.2 * times, np.sin(times / 10)])
    drift = Drift(curve, method="mbm", times_s=times)
    out = drift.apply(locs)
    t = locs["time_s"]
    inside = (t >= 10) & (t <= 90)
    assert np.allclose((locs["x_nm"] - out["x_nm"])[inside], 0.5 * t[inside])
    assert np.allclose((locs["y_nm"] - out["y_nm"])[inside], -0.2 * t[inside])
    # held at the ends, not extrapolated
    assert np.allclose((locs["x_nm"] - out["x_nm"])[t < 10], 5.0)
    assert np.allclose((locs["x_nm"] - out["x_nm"])[t > 90], 45.0)
    assert np.array_equal(out["x_nc_nm"], locs["x_nc_nm"])            # not a position column
    assert out.metadata["drift_correction"]["axis"] == "time_s"
    assert str(drift).startswith("drift over 80 s")
    # part of the table: the other rows stay
    rows = t > 50
    part = drift.apply(locs, rows=rows)
    assert np.array_equal(part["x_nm"][~rows], locs["x_nm"][~rows])
    assert np.allclose(part["x_nm"][rows], out["x_nm"][rows])


def test_a_drift_on_a_time_axis_is_kept_saved_and_read_back_with_its_times(tmp_path):
    from smappy.drift import load_drift, save_drift_corrected
    times = np.array([0.0, 2.0, 4.5, 7.0])
    drift = Drift(np.arange(12.0).reshape(4, 3), method="mbm", times_s=times)
    again = Drift.from_dict(drift.to_dict())
    assert np.array_equal(again.times_s, times) and np.array_equal(again.drift, drift.drift)
    assert Drift.from_dict(Drift(np.zeros((3, 3))).to_dict()).times_s is None
    path = save_drift_corrected(tmp_path / "x_driftc.hdf5", drift.apply(_minflux_like()), drift)
    loaded = load_drift(path)
    assert np.array_equal(loaded.times_s, times) and loaded.method == "mbm"
    assert np.allclose(loaded.drift, drift.drift)


def test_a_drift_on_a_time_axis_is_referenced_and_added_on_its_own_axis_only():
    times = np.array([1.0, 2.0, 3.0])
    drift = Drift(np.array([[1.0, 0, 0], [2, 0, 0], [4, 0, 0]]), times_s=times)
    assert np.allclose(drift.referenced("first frame").drift[:, 0], [0, 1, 3])
    assert np.allclose(drift.referenced("last frame").drift[:, 0], [-3, -2, 0])
    assert np.array_equal(drift.referenced("last frame").times_s, times)
    both = drift.then(drift)
    assert np.allclose(both.drift[:, 0], [2, 4, 8]) and np.array_equal(both.times_s, times)
    with pytest.raises(ValueError, match="same axis"):
        drift.then(Drift(np.zeros((3, 3))))
    with pytest.raises(ValueError, match="same axis"):
        drift.then(Drift(np.zeros((3, 3)), times_s=times + 1))
    with pytest.raises(ValueError, match="increase"):
        Drift(np.zeros((3, 3)), times_s=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="no transformation"):
        Drift(np.zeros((3, 3)), times_s=times, matrix=np.tile(np.eye(3), (3, 1, 1)))
    with pytest.raises(ValueError, match="no time_s"):
        drift.apply(Localizations({"x_nm": np.zeros(2), "frame": np.arange(2)}, {}))


# ------------------------------------------- precisions, and the error bar

def simulate_3d(n_points=300, n_frames=2000, per_frame=15, seed=1, pinned=0.0,
                amplitude=40.0):
    """A 3D structure seen with its precisions, z three times worse than x
    and y, moved by a smooth drift; ``pinned`` of the localizations are
    out-of-focus fits held at z = +-400 nm whatever the drift, and say so in
    their precisions."""
    rng = np.random.default_rng(seed)
    structure = np.column_stack([rng.uniform(0, 4000, n_points),
                                 rng.uniform(0, 4000, n_points),
                                 rng.uniform(-300, 300, n_points)])
    t = np.arange(n_frames) / n_frames
    drift = amplitude * np.column_stack([np.sin(5 * t + 0.3), np.cos(4 * t),
                                         np.sin(6 * t + 1.0)])
    frame = np.repeat(np.arange(n_frames), per_frame)
    true = structure[rng.integers(0, n_points, frame.size)]
    xy_err = rng.uniform(4, 12, frame.size)
    z_err = 3 * xy_err
    xyz = (true + drift[frame]
           + rng.normal(size=(frame.size, 3)) * np.column_stack([xy_err, xy_err, z_err]))
    bad = rng.random(frame.size) < pinned
    xy_err[bad], z_err[bad] = 30.0, 200.0
    xyz[bad, :2] = (true + drift[frame])[bad, :2] + rng.normal(0, 30, (bad.sum(), 2))
    xyz[bad, 2] = np.where(true[bad, 2] >= 0, 400.0, -400.0)
    locs = Localizations({"frame": frame, "x_nm": xyz[:, 0], "y_nm": xyz[:, 1],
                          "z_nm": xyz[:, 2], "xy_err_nm": xy_err, "z_err_nm": z_err},
                         {"units": "nm"})
    return locs, drift


SPLINE_3D = DriftSettings(max_drift_nm=150, group=False, spline_knot_frames=250)


def _rms_error(drift, truth):
    error = (drift.drift - drift.drift.mean(0)) - (truth - truth.mean(0))
    return np.sqrt(np.mean(error ** 2, axis=0))


def test_precision_weights_keep_pinned_fits_from_holding_z():
    """Out-of-focus fits pinned in z do not follow the drift and pull the
    axial estimate towards none; their precisions say they are poor, and
    weighted by them they count for little -- what needed a hand-set z cut."""
    from dataclasses import replace

    locs, truth = simulate_3d(pinned=0.3)
    plain = estimate_drift(locs, replace(SPLINE_3D, precision_weights=False,
                                         split_half=False))
    weighted = estimate_drift(locs, replace(SPLINE_3D, split_half=False))
    assert _rms_error(plain, truth)[2] > 2.5
    assert _rms_error(weighted, truth)[2] < 0.6 * _rms_error(plain, truth)[2]


def test_the_error_bar_is_the_noise_of_a_curve_where_there_is_no_drift():
    """With no drift the curve is its noise, and the halves of the field
    measure it: the curve's spread about its mean is about the noise they
    give, and no axis is called drift."""
    locs, _ = simulate_3d(amplitude=0.0)
    drift = estimate_drift(locs, SPLINE_3D)
    assert drift.halves.shape == (2, len(drift), 3)
    spread = np.sqrt(np.mean((drift.drift - drift.drift.mean(0)) ** 2, axis=0))
    assert np.all((0.4 < spread / drift.noise_nm) & (spread / drift.noise_nm < 2.0))
    assert drift.insignificant_axes() == "xyz"
    assert "no drift above the noise in x, y, z" in str(drift)


def test_a_real_drift_is_more_than_its_noise():
    locs, truth = simulate_3d()
    drift = estimate_drift(locs, SPLINE_3D)
    assert drift.insignificant_axes() == ""
    assert np.all(drift.noise_nm < 1.0)
    # the halves lie about the curve, as far apart as the noise says
    for half in drift.halves:
        assert np.sqrt(np.mean((half - drift.drift) ** 2)) < 5 * drift.noise_nm.max()


def test_the_halves_are_kept_moved_with_the_reference_and_drawn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataclasses import replace

    locs, _ = simulate_3d(n_frames=600, seed=2)
    drift = estimate_drift(locs, replace(SPLINE_3D, reference="first frame"))
    np.testing.assert_allclose(drift.drift[0], 0.0, atol=1e-9)
    # moved with the curve: about it still, not each to zero at frame 0
    assert np.abs(drift.halves[:, 0]).max() < 5.0
    again = Drift.from_dict(drift.to_dict())
    np.testing.assert_allclose(again.halves, drift.halves, atol=1e-3)
    np.testing.assert_allclose(again.noise_nm, drift.noise_nm, atol=1e-3)
    assert drift.apply(locs).metadata["drift_correction"]["noise_nm"] is not None
    ax = drift.plot()
    assert sum(line.get_linestyle() == "--" for line in ax.get_lines()) == 6
    plt.close(ax.figure)


def test_without_the_error_bar_there_are_no_halves_and_half_the_cost():
    from dataclasses import replace

    locs, _ = simulate_3d(n_frames=600)
    drift = estimate_drift(locs, replace(SPLINE_3D, split_half=False))
    assert drift.halves is None and drift.noise_nm is None
    assert "noise" not in str(drift)
    with_bar = estimate_cost(locs, SPLINE_3D)
    without = estimate_cost(locs, replace(SPLINE_3D, split_half=False))
    assert with_bar.seconds > 1.5 * without.seconds


def test_rcc_has_the_error_bar_too():
    from smappy.rcc import RCCSettings, estimate_drift_rcc

    locs, _ = simulate_3d(n_frames=600)
    drift = estimate_drift_rcc(locs, RCCSettings(n_timepoints=6, group=False))
    assert drift.halves.shape == (2, len(drift), 3)
    assert drift.noise_nm is not None
