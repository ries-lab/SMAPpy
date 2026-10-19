"""Stability, the vibrations approach: simulated beads with known vibrations."""
from __future__ import annotations

import numpy as np
import pytest

from smappy import stability as st
from smappy.locs import Localizations
from smappy.plugins import Context, Selection
from smappy.plugins.stability import Stability
from smappy.simulate import simulate_beads

DT_MS = 1.0


def shaking(n_frames, f_hz, amp_nm, axis=0, dt_ms=DT_MS, phase=0.3):
    """(n_frames, 3) of a sinusoid on one axis, as ``drift_nm``."""
    t = np.arange(n_frames) * dt_ms / 1000.0
    out = np.zeros((n_frames, 3))
    out[:, axis] = amp_nm * np.sin(2 * np.pi * f_hz * t + phase)
    return out


def beads(drift, n_beads=1, p_missed=0.0, seed=0, **kw):
    """Simulated beads moving by ``drift``, with a ``track_id`` (1, 2, ...)
    and the frame time."""
    n_frames = len(drift)
    locs = simulate_beads(n_frames, drift, n_beads=n_beads, p_missed=p_missed,
                          seed=seed, **kw)
    columns = dict(locs.columns)
    columns["track_id"] = (np.asarray(locs["bead_true"]) + 1).astype(np.int32)
    meta = dict(locs.metadata)
    meta["files"] = [{"frame_time_ms": DT_MS}]
    return Localizations(columns, meta)


def test_a_bead_shaking_at_a_known_frequency_gives_its_peak_amplitude_and_rms():
    # 37.3 Hz falls between bins; 8 nm on x, localization noise of 3 nm
    locs = beads(shaking(8000, 37.3, 8.0), precision_nm=3.0)
    settings = Stability.Settings(use_z=False)
    settings.vibration.highpass_hz = 2.0
    result = Stability().run(Context(locs=locs), settings)
    v = result.values
    assert v["x_peak_hz"] == pytest.approx(37.3, abs=0.3)
    assert v["x_peak_nm"] == pytest.approx(8.0, rel=0.05)
    # the vibration's RMS is A / sqrt(2), the SD's excess over the precision
    assert v["x_vibration_nm"] == pytest.approx(8.0 / np.sqrt(2), rel=0.05)
    assert v["x_precision_nm"] == pytest.approx(3.0, rel=0.01)
    assert v["x_sd_nm"] == pytest.approx(np.hypot(8.0 / np.sqrt(2), 3.0), rel=0.03)
    # y does not shake: its SD is the precision, and nothing is left over
    assert v["y_sd_nm"] == pytest.approx(3.0, rel=0.05)
    assert v["y_vibration_nm"] < 1.0
    assert v["y_peak_nm"] != v["y_peak_nm"] or v["y_peak_nm"] < 0.5
    assert "z_sd_nm" not in v
    # the figures, each a tick
    assert set(result.plots) == {"whole trace", "amplitude spectrum", "cumulative RMS",
                                 "peaks"}
    assert result.plot is not None
    # the curves: the cumulative RMS at the top is the high-passed SD
    assert result.arrays["x_cumulative_rms_nm"][-1] == pytest.approx(v["x_sd_nm"],
                                                                     rel=0.03)


def test_the_spectrum_reads_a_sinusoids_amplitude_wherever_its_frequency_falls():
    rng = np.random.default_rng(1)
    t = np.arange(16384) * 1e-3
    for f in (50.0, 50.25, 50.5):        # on a bin, a quarter and half between
        x = 5.0 * np.sin(2 * np.pi * f * t) + rng.normal(0, 2.0, len(t))
        s = st.spectrum(x, 1e-3, 1024)
        (found, amp), = st.peaks(s, 1)
        assert found == pytest.approx(f, abs=0.3)
        assert amp == pytest.approx(5.0, rel=0.04)
        assert st.cumulative_rms(s)[-1] == pytest.approx(np.std(x), rel=0.03)


def test_the_differential_trace_cancels_a_vibration_common_to_two_beads():
    locs = beads(shaking(6000, 23.0, 10.0, axis=1), n_beads=2, seed=2)
    each = Stability().run(Context(locs=locs), Stability.Settings(use_z=False))
    assert each.values["y_peak_hz"] == pytest.approx(23.0, abs=0.3)
    assert each.values["y_vibration_nm"] > 6.0
    for how in ("pair", "others"):
        diff = Stability().run(Context(locs=locs),
                               Stability.Settings(use_z=False, traces=how))
        v = diff.values
        # what is left is the two beads' localization noise: sqrt(2) x 3 nm
        assert v["y_sd_nm"] == pytest.approx(3.0 * np.sqrt(2), rel=0.06)
        assert v["y_precision_nm"] == pytest.approx(3.0 * np.sqrt(2), rel=0.01)
        assert v["y_vibration_nm"] < 1.5
        assert not v["y_peak_nm"] > 0.5


def test_the_mean_over_beads_keeps_the_common_vibration_and_averages_the_noise():
    locs = beads(shaking(6000, 23.0, 10.0), n_beads=4, seed=3)
    r = Stability().run(Context(locs=locs), Stability.Settings(use_z=False,
                                                              traces="mean"))
    v = r.values
    assert v["n_traces"] == 1
    assert v["x_peak_nm"] == pytest.approx(10.0, rel=0.05)
    assert v["y_sd_nm"] == pytest.approx(3.0 / 2, rel=0.08)
    assert v["y_precision_nm"] == pytest.approx(1.5, rel=0.01)


def test_missing_frames_are_filled_by_interpolation_and_counted():
    t = np.array([0, 1, 2, 5, 6, 9], float)
    x = np.array([0, 1, 2, 5, 6, 9], float)
    grid, v, filled = st.resample(t, x, 1.0)
    assert grid.tolist() == list(range(10))
    np.testing.assert_allclose(v[:, 0], np.arange(10))
    assert np.flatnonzero(filled).tolist() == [3, 4, 7, 8]

    # through the plugin: a bead missed in 5 % of frames and one long gap
    locs = beads(shaking(8000, 37.3, 8.0), p_missed=0.05,
                 gaps=[(0, 3000, 3099)], seed=4)
    frames = np.asarray(locs["frame"])
    missing = 8000 - len(np.unique(frames)) - frames.min() - (7999 - frames.max())
    r = Stability().run(Context(locs=locs), Stability.Settings(use_z=False))
    assert r.values["n_filled"] == missing
    assert "filled" in r.text
    # the peak survives the gaps
    assert r.values["x_peak_hz"] == pytest.approx(37.3, abs=0.3)
    assert r.values["x_peak_nm"] == pytest.approx(8.0, rel=0.1)


def test_an_irregular_clock_is_put_on_a_regular_grid():
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.uniform(0.8e-3, 1.2e-3, 10000))
    x = 6.0 * np.sin(2 * np.pi * 40.0 * t) + rng.normal(0, 2.0, len(t))
    locs = Localizations({"x_nm": x, "y_nm": rng.normal(0, 2.0, len(t)),
                          "frame": np.arange(len(t)), "time_s": t,
                          "tid": np.ones(len(t), np.int64)}, {})
    r = Stability().run(Context(locs=locs), Stability.Settings(id_column="tid"))
    assert r.values["x_peak_hz"] == pytest.approx(40.0, abs=0.5)
    assert r.values["x_peak_nm"] == pytest.approx(6.0, rel=0.1)
    assert "time_s" in r.text


def test_each_bead_is_a_trace_and_the_selection_narrows_them():
    locs = beads(shaking(3000, 23.0, 5.0), n_beads=3, seed=6)
    keep = np.asarray(locs["track_id"]) != 2
    r = Stability().run(Context(locs=locs, selection=Selection(keep)),
                        Stability.Settings())
    assert r.values["n_traces"] == 2
    assert [t.name for t in r.data["traces"]] == ["1", "3"]
    assert "z_sd_nm" in r.values


def test_the_plugin_says_how_to_get_beads():
    locs = simulate_beads(100, n_beads=1)
    with pytest.raises(ValueError, match="Link tracks"):
        Stability().run(Context(locs=locs), Stability.Settings())


def test_the_form_shows_the_vibrations_part_only_while_it_is_ticked():
    s = Stability.Settings(vibrations=False)
    assert Stability().describe(None, s).shown["vibration"] is False
    assert Stability().describe(None, Stability.Settings()).shown["vibration"] is True


def test_each_bead_is_kept_as_a_row_and_the_shaking_one_stands_out():
    """With each bead, over the view, the numbers per bead go to a region
    table of the id column, current as a block's would be."""
    from smappy.region_results import Evaluation, is_current, rows
    from smappy.session import Session
    drift = shaking(3000, 23.0, 5.0)
    locs = beads(np.zeros_like(drift), n_beads=3, seed=6)
    still = np.asarray(locs["track_id"]) != 2
    shaken = np.asarray(locs["x_nm"], float).copy()
    frames = np.asarray(locs["frame"], int)
    shaken[~still] += drift[frames[~still], 0]          # bead 2 alone shakes
    locs = Localizations({**locs.columns, "x_nm": shaken}, dict(locs.metadata))
    session = Session(locs)
    session.show_grouped(0, False)
    for layer in session.layers:
        layer.remove_bound("z_nm")
    session.run(Stability(), Stability.Settings())
    ev = Evaluation("track_id", "Stability")
    ctx = session.context()
    assert is_current(ctx, ev)
    got = {r["roi_id"]: r for r in rows(ctx, ev)}
    assert set(got) == {"1", "2", "3"}
    # a 5 nm sinusoid is 3.5 nm RMS; the still beads read their noise
    assert got["2"]["x_vibration_nm"] == pytest.approx(5 / np.sqrt(2), rel=0.15)
    assert got["2"]["x_vibration_nm"] > 2.5 * max(got["1"]["x_vibration_nm"],
                                                  got["3"]["x_vibration_nm"])
    assert got["2"]["x_peak_hz"] == pytest.approx(23.0, abs=0.5)
    table = session.id_regions["track_id"]["Stability"]
    assert table.groups["x"]["vibration"] == "x_vibration_nm"
