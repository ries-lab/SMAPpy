"""*Find steps* (`smappy.steps`, `plugins/find_steps.py`): staircases with
known steps and dwells come back."""
import numpy as np
import pytest

from smappy import plugins
from smappy import steps as S
from smappy.locs import Localizations
from smappy.session import Session

PLUGIN = "Analysis/Tracking/Find steps"


def staircase(n=2000, step=8.0, dwell=30.0, noise=3.0, shape=1.0, seed=0):
    """Positions climbing by ``step`` after dwells drawn from a gamma of
    ``shape`` and mean ``dwell`` points; the true plateau starts."""
    rng = np.random.default_rng(seed)
    levels, starts, pos = [], [], 0.0
    while len(levels) < n:
        d = max(3, int(round(rng.gamma(shape, dwell / shape))))
        starts.append(len(levels))
        levels += [pos] * d
        pos += step
    x = np.array(levels[:n]) + rng.normal(0, noise, n)
    return x, np.array([s for s in starts if 0 < s < n])


def test_autostepfinder_finds_about_the_steps_there_are():
    x, truth = staircase()
    found = S.autostepfinder(x)
    assert found.accepted[0] > 0
    assert len(found.boundaries) == pytest.approx(len(truth), rel=0.2)
    sizes = np.diff(S.step_table(x, np.arange(len(x)), found.boundaries)["level"])
    assert np.median(sizes) == pytest.approx(8.0, abs=1.0)


def test_noise_alone_gives_no_steps():
    x = np.random.default_rng(3).normal(0, 3, 1000)
    assert len(S.autostepfinder(x).boundaries) <= 2


def test_split_and_merge_to_the_expected_size_cleans_the_step_sizes():
    x, truth = staircase(noise=4.0, seed=2)
    b = S.split_merge(x, S.autostepfinder(x).boundaries, 8.0)
    sizes = np.diff(S.step_table(x, np.arange(len(x)), b)["level"])
    assert np.median(sizes) == pytest.approx(8.0, abs=0.5)
    assert len(b) == pytest.approx(len(truth), rel=0.15)


def test_refining_moves_a_boundary_onto_the_true_step():
    x = np.r_[np.zeros(50), np.full(50, 10.0)] + np.random.default_rng(1).normal(0, 1, 100)
    assert S.refine(x, [52]).tolist() == [50]


def test_the_erlang_shape_of_dwells_counts_the_hidden_substeps():
    rng = np.random.default_rng(4)
    assert S.erlang_shape(rng.exponential(10, 5000)) == pytest.approx(1.0, rel=0.1)
    assert S.erlang_shape(rng.gamma(3, 5, 5000)) == pytest.approx(3.0, rel=0.1)


def motors(n_traces=10, seed=0):
    rng = np.random.default_rng(seed)
    cols = {"x_nm": [], "y_nm": [], "time_s": [], "tid": []}
    for k in range(n_traces):
        x, _ = staircase(n=400, seed=seed + k)
        a = rng.uniform(0, 2 * np.pi)
        cols["x_nm"].append(2000 * k + x * np.cos(a))
        cols["y_nm"].append(x * np.sin(a) + rng.normal(0, 3, 400))
        cols["time_s"].append(np.arange(400) * 1e-3)
        cols["tid"].append(np.full(400, k + 1))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    c["frame"] = np.arange(len(c["tid"]))
    return Localizations(c, {})


def test_the_plugin_measures_every_trace_along_its_track():
    session = Session(motors())
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    result = session.run(cls(), cls.Settings())
    pooled = result.data["pooled"]
    assert pooled["step_mean_nm"] == pytest.approx(8.0, abs=1.5)
    assert pooled["forward"] > 0.85
    assert pooled["velocity"] == pytest.approx(8.0 / 0.030, rel=0.3)    # nm/s
    assert set(session.id_regions["tid"]["Find steps"].ids) == {str(k) for k in range(1, 11)}
    assert session.locs is not None and "plateau" not in session.locs


def test_the_plateaus_go_on_the_localizations_or_into_a_file():
    session = Session(motors(n_traces=3))
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    settings = cls.Settings(output="columns")
    assert cls().changes_locs_for(settings)
    session.run(cls(), settings)
    plateau = np.asarray(session.locs["plateau"])
    assert (plateau > 0).all() and len(np.unique(plateau)) > 30
    session.run(cls(), cls.Settings(output="file"))
    steps = session.locs[np.asarray(session.locs["filenumber"]) == len(session.files) - 1]
    assert len(steps) == len(np.unique(plateau)) and "step_level_nm" in steps


def test_both_threads_of_a_pair_step_in_the_first_ones_frame():
    from smappy import pairing as P
    rng = np.random.default_rng(5)
    # dwells of 60 points, so each thread, seeing every other one, has 30; the
    # noise across the track projects onto it, so each is 2 nm (3 nm along in all)
    x, _ = staircase(n=1600, dwell=60.0, noise=2.0, seed=5)
    a = np.radians(40)
    th = np.arange(1600) % 2                     # the threads alternate
    shift = np.where(th == 1, 15.0, 0.0)         # the second colour 15 nm further on
    along = x + shift
    cols = {"x_nm": (along * np.cos(a)).astype(np.float32),
            "y_nm": (along * np.sin(a) + rng.normal(0, 2, 1600)).astype(np.float32),
            "time_s": np.arange(1600) * 1e-3, "frame": np.arange(1600),
            "tid": np.where(th == 0, 1, 2), "thi": th}
    cols["pair_id"] = P.pair_traces(cols["tid"], cols["thi"], cols["time_s"])
    session = Session(Localizations(cols, {}))
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    result = session.run(cls(), cls.Settings(pairs=True, min_locs=20))
    table = session.id_regions["pair_id"]["Find steps"]
    values = dict(zip(table.columns().keys(), [v[0] for v in table.columns().values()]))
    assert values["step_median_thi0_nm"] == pytest.approx(8.0, abs=1.5)
    assert values["step_median_thi1_nm"] == pytest.approx(8.0, abs=1.5)
    assert result.data["pooled"]["forward"] > 0.8


def test_the_dwell_models_pick_the_one_the_dwells_come_from():
    rng = np.random.default_rng(6)
    for dwells, best in ((rng.exponential(20, 800) + rng.exponential(40, 800), "hypoexponential"),
                         (rng.gamma(3, 10, 800), "gamma")):
        m = S.dwell_models(dwells)
        assert min(m, key=lambda k: m[k]["AIC"]) == best
    # one rate: the gamma holds it (shape 1), so the exponential is within
    # chance of the best, and the gamma's shape near 1
    m = S.dwell_models(rng.exponential(30, 800))
    assert m["exponential"]["AIC"] - min(v["AIC"] for v in m.values()) < 2
    assert m["gamma"]["shape"] == pytest.approx(1, abs=0.12)
    m = S.dwell_models(rng.gamma(3, 10, 2000))
    assert m["gamma"]["shape"] == pytest.approx(3, rel=0.1)


def test_bnp_step_finds_the_step_size_without_being_told_it():
    session = Session(motors())
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    greedy = session.run(cls(), cls.Settings()).data["pooled"]
    pooled = session.run(cls(), cls.Settings(method="bnp", sweeps=150)).data["pooled"]
    assert pooled["step_mean_nm"] == pytest.approx(8.0, abs=1.0)
    # fewer steps fitted to noise: a narrower spread of sizes than AutoStepFinder's
    assert pooled["step_sd_nm"] < greedy["step_sd_nm"]
