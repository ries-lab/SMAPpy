"""Analysis/Measure/Time Course: the rate, columns and image tags against time."""
import numpy as np
import pytest

from smappy import plugins
from smappy.frametags import KEY
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.time_course import (COUNT, TAG, frame_windows, rate,
                                        window_edges, window_statistic)
from smappy.session import Session

PLUGIN = "Analysis/Measure/Time Course"


def bleaching(n_frames=5000, tau=2000.0, seed=0, **metadata):
    """Localizations that thin out as exp(-frame/tau), their photons falling
    linearly from 2000 to 1000 -- the trends the plugin has to find."""
    rng = np.random.default_rng(seed)
    frames = np.repeat(np.arange(n_frames), 8)
    keep = rng.random(len(frames)) < np.exp(-frames / tau)
    frames = frames[keep]
    n = len(frames)
    photons = 2000.0 - 1000.0 * frames / n_frames + rng.normal(0, 50, n)
    return Localizations({"x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n),
                          "frame": frames, "photons": photons,
                          "xy_err_nm": np.full(n, 10.0)}, dict(metadata))


def run(ctx_or_locs, **settings):
    plugin = plugins.get(PLUGIN)
    ctx = ctx_or_locs if isinstance(ctx_or_locs, Context) else Context(locs=ctx_or_locs)
    return plugin().run(ctx, plugin.Settings(**settings))


def test_the_rate_and_a_columns_median_follow_the_trends_they_were_made_with():
    result = run(bleaching(), series=[COUNT, "photons"], windows=50)
    count, photons = result.data["lines"]
    x = count.x
    truth = 8 * np.exp(-x / 2000.0)                      # per frame
    assert np.allclose(count.y, truth, rtol=0.1, atol=0.15)
    assert np.allclose(photons.y, 2000 - 1000 * x / 5000, atol=40)
    assert "photons" in result.text and "at the end (-" in result.text


def test_a_window_with_too_few_localizations_gives_no_point():
    t = np.array([0.5, 1.5, 1.6, 1.7] + [5.5] * 20)
    v = np.arange(len(t), dtype=float)
    out = window_statistic(t, v, window_edges(0, 6, 6), min_count=10)
    assert np.isnan(out[:5]).all() and np.isfinite(out[5])


def test_the_rate_is_per_frame_whatever_the_axis_is_in():
    frames = np.repeat(np.arange(100), 3).astype(float)
    assert np.allclose(rate(frames, window_edges(0, 100, 10)), 3.0)
    # windows of whole frames, the last one counting only the frames there are
    edges, covered = frame_windows(0, 99, 7)
    assert np.allclose(np.diff(edges)[:-1], 15) and covered[-1] == 10
    assert np.allclose(rate(frames, edges, covered), 3.0)


def test_seconds_come_from_the_exposure_the_fit_recorded():
    locs = bleaching(camera={"exposure_ms": 20.0})
    result = run(locs, series=[COUNT], x="seconds", windows=10)
    assert result.data["edges"][-1] == pytest.approx(5000 * 0.02)
    line = result.data["lines"][0]
    assert line.x[0] == pytest.approx(250 * 0.02)          # the first window's middle
    assert np.isclose(line.y[0], 8 * np.exp(-250 / 2000), rtol=0.1)
    with pytest.raises(ValueError, match="frame time"):
        run(bleaching(), series=[COUNT], x="seconds")
    assert run(bleaching(), series=[COUNT], x="seconds", frame_time_ms=10.0,
               windows=10).data["edges"][-1] == pytest.approx(50.0)


def test_per_file_one_point_each():
    locs = bleaching(n_frames=300)
    locs.columns["filenumber"] = (np.asarray(locs["frame"]) // 100 + 1).astype(np.int32)
    result = run(locs, series=["photons"], x="file")
    assert len(result.data["lines"][0].y) == 3 and result.text.startswith("3 files")


def test_image_tags_are_series_drawn_once_numbers_averaged_states_as_steps():
    frames = np.arange(5000)
    tags = {"frame": frames, "Piezo-Position": 10.0 + frames * 0.001,
            "Laser": np.array(["off"] * 2500 + ["on"] * 2500, dtype=object)}
    locs = bleaching(**{KEY: tags})
    session = Session(locs)
    session.show_grouped(0, False)
    second = session.add_layer(like=0)
    second.set_bound("photons", 1500.0, None)
    result = run(session.context(), series=[COUNT, TAG + "Piezo-Position", TAG + "Laser"],
                 windows=10)
    lines = result.data["lines"]
    assert [ln.series for ln in lines].count(COUNT) == 2          # a line per layer
    piezo = next(ln for ln in lines if ln.series == TAG + "Piezo-Position")
    assert np.allclose(piezo.y, 10.0 + piezo.x * 0.001, atol=0.01)
    laser = next(ln for ln in lines if ln.series == TAG + "Laser")
    assert laser.states == ["off", "on"]
    assert "Laser: off -> on" in result.text


def test_a_layer_of_merged_blinks_counts_blinks():
    session = Session(bleaching(n_frames=500))
    assert session.layers[0].grouped
    line = run(session.context(), series=[COUNT]).data["lines"][0]
    assert line.label == "blinks per frame"


def test_a_series_not_in_the_table_is_named():
    with pytest.raises(ValueError, match="not in the table: prob"):
        run(bleaching(n_frames=200), series=["prob"])


def test_the_list_offers_the_rate_the_columns_and_the_tags():
    frames = np.arange(10)
    session = Session(bleaching(n_frames=200, **{KEY: {"frame": frames,
                                                        "Piezo": frames * 1.0}}))
    plugin = plugins.get(PLUGIN)
    state = plugin().describe(session.context(), plugin.Settings())
    offered = state.choices["series"]
    assert offered[0] == COUNT and "photons" in offered and TAG + "Piezo" in offered
    assert "frame" not in offered
    assert state.active["frame_time_ms"] is False
