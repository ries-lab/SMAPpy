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


def test_both_colours_of_a_molecule_step_in_colour_0s_frame_by_default():
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
    result = session.run(cls(), cls.Settings(min_locs=20))     # two colours: auto
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
    pooled = session.run(cls(), plugins.settings_from(cls.Settings, {"method": "bnp", "bnp.sweeps": 150})).data["pooled"]
    assert pooled["step_mean_nm"] == pytest.approx(8.0, abs=1.0)
    # fewer steps fitted to noise: a narrower spread of sizes than AutoStepFinder's
    assert pooled["step_sd_nm"] < greedy["step_sd_nm"]


def stuck_then_walking(seed=8, angle_deg=30.0):
    """A motor stuck for 1 s, wandering slowly across its track, then walking
    0.5 s at 400 nm/s along ``angle_deg``: 3 ms between localizations."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 1.5, 3e-3)
    a = np.radians(angle_deg)
    along = np.where(t < 1.0, 0.0, 400.0 * (t - 1.0))
    wander = np.where(t < 1.0, 25.0 * np.sin(np.pi * t), 0.0)   # slower than it walks
    x = along * np.cos(a) - wander * np.sin(a) + rng.normal(0, 4, len(t))
    y = along * np.sin(a) + wander * np.cos(a) + rng.normal(0, 4, len(t))
    return t, x, y


def test_the_track_axis_comes_from_where_the_motor_moves():
    from smappy import traces as T
    t, x, y = stuck_then_walking()
    angle, share = T.moving_direction(t, x, y, window=0.1, min_speed=100.0)
    assert angle == pytest.approx(30.0, abs=4.0)
    assert 0.2 < share < 0.5                          # a third of the time it walked
    # the long axis of all the localizations is pulled across by the wander
    r = T.runs(np.zeros(len(t), np.int64), t)
    long_axis = T.axes(r, x, y)["angle_deg"][0]
    off = abs((long_axis - 30.0 + 180) % 360 - 180)
    assert off > 5 and off > abs(angle - 30.0)


def test_in_an_roi_find_steps_proposes_its_moving_direction_as_the_angle():
    t, x, y = stuck_then_walking(seed=9, angle_deg=-50.0)
    locs = Localizations({"x_nm": x.astype(np.float32), "y_nm": y.astype(np.float32),
                          "time_s": t, "frame": np.arange(len(t)),
                          "tid": np.zeros(len(t), np.int64)}, {})
    ctx = plugins.Context(locs=locs, region_id="roi-1")
    cls = plugins.get(PLUGIN)
    result = cls().run(ctx, cls.Settings(min_locs=20))
    assert result.proposals["angle"] == pytest.approx(310.0, abs=4.0)
    assert result.values["angle_deg"] == pytest.approx(310.0, abs=4.0)


def walking_pair(apart: bool, seed=3):
    """One molecule's two colours over 3 s: walking at 200 nm/s along 20°,
    the second colour a few nm off -- or, ``apart``, walking the other way
    from the same start, two molecules that met."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 3, 1500))
    colour = rng.integers(0, 2, 1500)
    along = 200 * t * np.where((colour == 1) & apart, -1.0, 1.0)
    a = np.radians(20)
    x = along * np.cos(a) + rng.normal(0, 3, 1500) + 4.0 * colour
    y = along * np.sin(a) + rng.normal(0, 3, 1500)
    return t, x, y, colour


def test_co_movement_tells_one_walking_molecule_from_two_that_met():
    from smappy.dual_colour import co_movement
    t, x, y, c = walking_pair(apart=False)
    one = co_movement(t, x, y, c, window=0.5, min_speed=100.0)
    assert one.share == 1.0 and one.windows == 6
    assert one.angle_deg < 5 and one.speed_ratio < 1.1 and one.distance_nm < 10
    t, x, y, c = walking_pair(apart=True)
    two = co_movement(t, x, y, c, window=0.5, min_speed=100.0)
    assert two.share == 0.0 and two.angle_deg > 170
    # the whole overlap as one window: the two lines compared
    assert co_movement(t, x, y, c, window=0.0, min_speed=100.0).windows == 1
    # a stuck molecule says nothing about moving together
    still = co_movement(t, np.random.default_rng(1).normal(0, 3, len(t)), y * 0, c,
                        window=0.5, min_speed=100.0)
    assert np.isnan(still.share) and still.windows == 0
    # the second colour seen for 40 ms, 8 nm at this speed: too short to judge
    t, x, y, c = walking_pair(apart=True)
    short = (c == 0) | ((t > 1.0) & (t < 1.04))
    assert co_movement(t[short], x[short], y[short], c[short], window=0.5,
                       min_speed=100.0).windows == 0


@pytest.mark.parametrize("apart, word", [(False, "dual"), (True, "apart")])
def test_in_an_roi_a_two_colour_molecule_is_proposed_both_or_apart(apart, word):
    t, x, y, c = walking_pair(apart)
    locs = Localizations({"x_nm": x.astype(np.float32), "y_nm": y.astype(np.float32),
                          "time_s": t, "frame": np.arange(len(t)), "channel": c,
                          "pair_id": np.zeros(len(t), np.int64),
                          "tid": c.astype(np.int64)}, {})
    ctx = plugins.Context(locs=locs, region_id="roi-1")
    cls = plugins.get(PLUGIN)
    result = cls().run(ctx, cls.Settings(min_locs=20))
    assert result.proposals["keyword:colours"] == word
    assert result.values["comove_share"] == (0.0 if apart else 1.0)


def test_the_smoothed_line_keeps_a_steps_levels_and_the_traces_length():
    x = np.r_[np.zeros(30), np.full(30, 8.0)] + np.random.default_rng(0).normal(0, 1, 60)
    for stat in ("median", "mean"):
        sm = S.smoothed(x, 9, stat)
        assert len(sm) == len(x)
        assert np.median(sm[:20]) == pytest.approx(0, abs=0.8)
        assert np.median(sm[-20:]) == pytest.approx(8, abs=0.8)
        assert np.std(np.diff(sm[:25])) < np.std(np.diff(x[:25]))
    assert np.array_equal(S.smoothed(x, 1), x)


def test_only_the_chosen_methods_part_is_shown():
    cls = plugins.get(PLUGIN)
    state = cls().describe(None, cls.Settings())
    assert state.shown["autostepfinder"] and not state.shown["bnp"]
    state = cls().describe(None, cls.Settings(method="bnp"))
    assert state.shown["bnp"] and not state.shown["autostepfinder"]
    assert "smooth" not in cls.specs()        # the figure's tick, not a setting


def test_the_figures_ticks_hide_the_steps_and_show_the_smoothed_line():
    from matplotlib.figure import Figure
    session = Session(motors())
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    plot = session.run(cls(), cls.Settings()).plot
    assert plot.switches == {"steps": True, "smoothed": False}

    def lines(switches):
        figure = Figure()
        plot.draw_into(figure, switches)
        ax = figure.axes[0]
        return ([ln.get_alpha() for ln in ax.lines if ln.get_drawstyle() == "default"],
                sum(ln.get_drawstyle() == "steps-post" for ln in ax.lines))

    raw, n_steps = lines(None)                      # as declared
    assert n_steps >= 1 and all(a in (None, 1.0) for a in raw)
    raw, n_steps = lines({"steps": False, "smoothed": True})
    assert n_steps == 0
    assert 0.25 in raw and None in raw              # the localizations faint, the smoothed not


def test_the_track_axis_options_are_set_in_under_it_and_a_ruler_is_drawn():
    pytest.importorskip("PySide6")
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.params import SUB_INDENT, SettingsForm
    cls = plugins.get(PLUGIN)
    form = SettingsForm(cls.Settings)
    assert form.labels["window_s"].indent() == SUB_INDENT
    assert form.labels["min_speed"].indent() == SUB_INDENT
    assert form.labels["axis"].indent() <= 0
    session = Session(motors())
    session.show_grouped(0, False)
    for guide, ruled in ((16.0, True), (0.0, False)):
        plot = session.run(cls(), cls.Settings(guide_nm=guide)).plot
        figure = Figure()
        plot.draw_into(figure)
        greys = [ln for ln in figure.axes[0].lines if ln.get_color() == "0.82"]
        assert bool(greys) == ruled
        if ruled:
            ys = np.array([ln.get_ydata()[0] for ln in greys])
            assert np.allclose(np.diff(ys), 16.0)
