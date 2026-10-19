import os
from pathlib import Path

import numpy as np
import pytest

from smappy.beads import MadRule, bead_metrics, fit_offsets_and_spline, select_beads
from smappy.locs import Localizations
from smappy.simulate import simulate_bead_tracks

T = np.arange(10.0, 200.0, 2.0)                    # a point every 2 s, as MBM
DRIFT = np.column_stack([10 * np.sin(T / 30), 0.05 * T, 3 * np.cos(T / 50)])


def _positions(b):
    return np.column_stack([b["x_nm"], b["y_nm"], b["z_nm"]])


def _rms_from_truth(curve, truth):
    err = (curve - curve.mean(0)) - (truth - truth.mean(0))
    return np.sqrt(np.mean(err ** 2, axis=0))


def test_the_fit_places_a_late_bead_by_its_offset_and_finds_the_drift():
    b = simulate_bead_tracks(T, DRIFT, n_beads=4, seed=2)
    late = (b["tid"] == 2) & (b["time_s"] < 100)
    keep = ~late                                     # bead 2 only from 100 s
    t, p = b["time_s"][keep], _positions(b)[keep]
    index = b["tid"][keep] - 1
    offsets, curve = fit_offsets_and_spline(t, p, index, 4, T[0], T[-1],
                                            n_coeffs=int((T[-1] - T[0]) / 8) + 3)
    assert np.allclose(curve([T[0]]), 0.0, atol=1e-9)            # the reference
    assert np.all(_rms_from_truth(curve(T), DRIFT) < 0.6)
    # each offset is where the bead is when the drift is zero
    truth0 = DRIFT[0]
    for k in range(4):
        rows = b["tid"] == k + 1
        where = _positions(b)[rows][0] - DRIFT[0]
        assert np.allclose(offsets[k], where + truth0, atol=3.0)


def test_a_bead_that_wanders_jumps_or_is_noisy_is_left_out_and_the_others_kept():
    b = simulate_bead_tracks(T, DRIFT, n_beads=7, wander=[(1, 8.0)],
                             jumps=[(3, 120.0, 15.0)], noisy=[(4, 6.0)], seed=1)
    s = select_beads(b["time_s"], _positions(b), b["tid"], knot=8.0)
    bad = set(b.metadata["bad"])
    assert {k for k in range(7) if not s.used[k]} == bad
    assert all(s.outlier[k] and s.reasons[k] for k in bad)
    assert "jump" in s.reasons[3] and "noise" in s.reasons[4] and "wander" in s.reasons[1]
    assert np.all(_rms_from_truth(s.drift_nm(T), DRIFT) < 0.6)
    # all of them in, the bad ones move the curve
    every = select_beads(b["time_s"], _positions(b), b["tid"], knot=8.0, rule=MadRule(k=1e9))
    assert every.used.all()
    assert _rms_from_truth(every.drift_nm(T), DRIFT)[:2].max() > \
        2 * _rms_from_truth(s.drift_nm(T), DRIFT)[:2].max()


def test_a_bead_among_four_is_judged_against_the_other_three():
    # in the fit, a bead pulls the curve a quarter of the way to itself;
    # against the other three its whole wander shows
    b = simulate_bead_tracks(T, DRIFT, n_beads=4, wander=[(2, 6.0)], seed=4)
    s = select_beads(b["time_s"], _positions(b), b["tid"], knot=8.0)
    assert list(s.used) == [True, True, False, True]
    assert s.metrics["wander_nm"][2] > 4.5


def test_a_choice_made_by_hand_is_never_overruled():
    b = simulate_bead_tracks(T, DRIFT, n_beads=5, jumps=[(0, 100.0, 20.0)], seed=5)
    names = ["R0", "R1", "R2", "R3", "R4"]
    s = select_beads(b["time_s"], _positions(b), b["tid"], names, knot=8.0,
                     overrides={"R0": True, "R2": False, "nonexistent": False})
    assert s.used[0] and not s.outlier[0] and s.state(0) == "in, by hand"
    assert not s.used[2] and s.state(2) == "out, by hand"
    assert s.overrides == {"R0": True, "R2": False}
    assert s.used[[1, 3, 4]].all()


def test_too_strict_a_rule_still_leaves_min_beads():
    b = simulate_bead_tracks(T, DRIFT, n_beads=5, seed=6)
    strict = MadRule(k=0.0, margin_nm=0.0)        # everything above the median is out
    s = select_beads(b["time_s"], _positions(b), b["tid"], knot=8.0, rule=strict,
                     min_beads=3)
    assert s.used.sum() >= 3


def test_the_metrics_tell_noise_from_wander_and_find_a_step():
    rng = np.random.default_rng(0)
    t = np.arange(100.0)
    noisy = rng.normal(0, 2.0, (100, 2))
    slow = np.column_stack([np.linspace(-6, 6, 100), np.zeros(100)])
    step = np.where(t[:, None] >= 50, [10.0, 0.0], [0.0, 0.0])
    r = np.concatenate([noisy, slow + rng.normal(0, 0.1, (100, 2)), step])
    index = np.repeat([0, 1, 2], 100)
    m = bead_metrics(np.tile(t, 3), r, index, 3)
    assert 2.4 < m["noise_nm"][0] < 3.3 and m["wander_nm"][0] < 1.5   # sqrt(2) x 2 nm
    assert m["noise_nm"][1] < 0.3 and m["wander_nm"][1] > 3.0
    assert 9.0 < m["jump_nm"][2] < 11.0 and m["jump_nm"][0] < 6.0


def test_the_bead_set_gives_a_drift_on_a_time_axis_that_corrects_a_table():
    b = simulate_bead_tracks(T, DRIFT, n_beads=5, seed=7)
    s = select_beads(b["time_s"], _positions(b), b["tid"], knot=8.0)
    drift = s.drift(method="mbm")
    assert drift.times_s is not None and drift.n_used == 5 and drift.method == "mbm"
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(T[0], T[-1], 400))
    truth = np.column_stack([np.interp(t, T, DRIFT[:, a]) for a in range(3)])
    true_xyz = rng.uniform(0, 1000, (400, 3))
    locs = Localizations({"x_nm": true_xyz[:, 0] + truth[:, 0],
                          "y_nm": true_xyz[:, 1] + truth[:, 1],
                          "z_nm": true_xyz[:, 2] + truth[:, 2], "time_s": t,
                          "frame": np.arange(400)}, {})
    out = drift.apply(locs)
    moved = np.column_stack([out["x_nm"], out["y_nm"], out["z_nm"]]) - true_xyz
    # what is left is a constant (the reference) and the fit's error
    assert np.all(np.std(moved, axis=0) < 0.6)


@pytest.mark.skipif(not os.environ.get("SMAPPY_TEST_MSR"),
                    reason="SMAPPY_TEST_MSR=dir of real .msr files")
def test_the_spline_through_imspectors_beads_is_imspectors_correction_smoothed():
    """With every bead Imspector used and nothing rejected, the curve is its
    correction (`lnc - loc`) to within the beads' scatter: a knot every 8 s
    left 0.7 nm in x and y of a 3D measurement whose beads scatter by 10 nm
    point to point, and 0.2 nm in 2D."""
    from smappy.io import msr
    checked = 0
    for path in sorted(Path(os.environ["SMAPPY_TEST_MSR"]).glob("*.msr")):
        for m in msr.contents(path).measurements:
            if not m.n_beads:
                continue
            locs, _ = msr.load(path, measurements=[m.number + 1])
            beads, info = msr.load(path, measurements=[m.number + 1], beads=True)
            if "x_nc_nm" not in locs or beads["time_s"].max() < locs["time_s"].max() - 5:
                continue
            axes = [a for a in "xyz" if f"{a}_nm" in beads and f"{a}_nc_nm" in locs]
            used = {b["tid"] for b in info.metadata["mbm_beads"] if b["used"]}
            keep = np.isin(beads["tid"], list(used))
            s = select_beads(beads["time_s"][keep],
                             np.column_stack([beads[f"{a}_nm"][keep] for a in axes]),
                             beads["tid"][keep], knot=8.0, rule=MadRule(k=1e9))
            left = (np.column_stack([locs[f"{a}_nc_nm"] - locs[f"{a}_nm"] for a in axes])
                    - s.curve(locs["time_s"]))
            assert np.all(np.std(left, axis=0) < 1.5), (path.name, m.label)
            checked += 1
    if not checked:
        pytest.skip("no file kept its beads' tracks through a measurement")
