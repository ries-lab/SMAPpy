"""The tracks render mode: each track drawn as antialiased lines through its
localizations in time order, culled whole, offered by the Render tab."""
import dataclasses

import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.render import (FieldOfView, RenderSettings, render_locs, render_segments,
                           track_segments)


def tracks_table(xy, ids, frames):
    xy = np.asarray(xy, float)
    return Localizations({"x_nm": xy[:, 0].astype(np.float32),
                          "y_nm": xy[:, 1].astype(np.float32),
                          "frame": np.asarray(frames, np.int64),
                          "track_id": np.asarray(ids, np.int64),
                          "xy_err_nm": np.full(len(xy), 10, np.float32)}, {"units": "nm"})


TRACKS = RenderSettings(mode="tracks", line_width=1.0)


def test_a_known_track_lights_its_segments_and_nothing_else():
    # an L through pixel centres: (5.5, 5.5) -> (35.5, 5.5) -> (35.5, 25.5),
    # pixels of 1 nm; given out of order
    locs = tracks_table([(35.5, 25.5), (5.5, 5.5), (35.5, 5.5)], [1, 1, 1], [2, 0, 1])
    fov = FieldOfView(0, 0, 1.0, 40, 30)
    image = render_locs(locs, fov, TRACKS).weight
    assert image[5, 10:30].min() == pytest.approx(1.0)       # along the first leg
    assert image[10:20, 35].min() == pytest.approx(1.0)      # along the second
    assert image[15, 10] == 0 and image[25, 5] == 0          # not the hypotenuse
    # a line one pixel wide: its weight is its length, plus the round ends
    assert image.sum() == pytest.approx(50.0, abs=3.0)


def test_tracks_are_not_joined_to_each_other_and_id_0_is_no_track():
    locs = tracks_table([(5, 5), (35, 5), (5, 25), (35, 25), (20, 15)],
                        [1, 1, 2, 2, 0], [0, 1, 0, 1, 2])
    a, b = track_segments(locs, TRACKS)
    assert sorted(zip(a.tolist(), b.tolist())) == [(0, 1), (2, 3)]
    image = render_locs(locs, FieldOfView(0, 0, 1.0, 40, 30), TRACKS).weight
    assert image[15, 20] == 0                                # no line between them


def test_a_track_crossing_the_view_with_both_ends_outside_is_drawn():
    from smappy.session import Session
    locs = tracks_table([(-5000, 50.5), (5000, 50.5)] + [(9000 + i, 9000) for i in range(3)],
                        [1, 1, 2, 2, 2], [0, 1, 0, 1, 2])
    session = Session(locs)
    layer = session.layers[0]
    layer.show_grouped(False)
    layer.state.settings = dataclasses.replace(layer.state.settings, mode="tracks")
    fov = FieldOfView(0, 0, 1.0, 100, 100)
    image = layer.state.snapshot().render(fov).weight
    assert image[50, :].min() == pytest.approx(1.0)


def test_the_extension_draws_what_the_reference_draws():
    rng = np.random.default_rng(0)
    xa, ya, xb, yb = rng.uniform(-50, 550, (4, 40))
    colours = rng.uniform(0, 1, (40, 3))
    weights = rng.uniform(0.5, 2.0, 40)
    fov = FieldOfView(0, 0, 10.0, 50, 40)
    for width in (1.0, 3.0):
        fast = render_segments(xa, ya, xb, yb, fov, width, weights, colours)
        slow = render_segments(xa, ya, xb, yb, fov, width, weights, colours,
                               use_extension=False)
        assert np.allclose(fast.weight, slow.weight, atol=1e-4)
        assert np.allclose(fast.color, slow.color, atol=1e-4)


def test_an_extension_built_before_the_tracks_mode_still_draws_tracks(monkeypatch):
    import types
    from smappy import render
    monkeypatch.setattr(render, "_render", types.SimpleNamespace())
    fov = FieldOfView(0, 0, 1.0, 100, 100)
    image = render_segments([10.0], [50.5], [90.0], [50.5], fov).weight
    assert image[50, 20:80].min() == pytest.approx(1.0)


def test_a_wider_line_weighs_more_and_colour_follows_the_column():
    locs = tracks_table([(5, 15), (35, 15)], [1, 1], [0, 10])
    fov = FieldOfView(0, 0, 1.0, 40, 30)
    thin = render_locs(locs, fov, TRACKS).weight.sum()
    wide = render_locs(locs, fov, dataclasses.replace(TRACKS, line_width=3.0)).weight.sum()
    assert wide == pytest.approx(3 * thin, rel=0.15)
    coloured = render_locs(locs, fov, dataclasses.replace(TRACKS, color_field="frame"))
    assert coloured.color is not None and coloured.color[15, 20].sum() > 0


def test_a_table_without_tracks_says_how_to_get_them():
    locs = tracks_table([(5, 5), (35, 5)], [1, 1], [0, 1])
    with pytest.raises(ValueError, match="Link tracks"):
        render_locs(locs, FieldOfView(0, 0, 1.0, 40, 30),
                    dataclasses.replace(TRACKS, track_column="tid"))
    bare = Localizations({c: locs[c] for c in ("x_nm", "y_nm", "frame")}, {})
    with pytest.raises(ValueError, match="none of track_id, tid, track_id_true"):
        render_locs(bare, FieldOfView(0, 0, 1.0, 40, 30), TRACKS)


def test_auto_draws_a_simulation_and_a_minflux_trace_without_a_choice():
    from smappy.simulate import simulate_tracks
    simulated = simulate_tracks(n_frames=50, seed=1)
    assert "track_id" not in simulated
    fov = FieldOfView(0, 0, 50.0, 400, 400)
    assert render_locs(simulated, fov, TRACKS).weight.sum() > 0
    minflux = tracks_table([(5.5, 5.5), (35.5, 5.5)], [0, 0], [0, 1])
    minflux = Localizations({**{c: minflux[c] for c in ("x_nm", "y_nm", "frame")},
                             "tid": np.array([7, 7])}, {})
    image = render_locs(minflux, FieldOfView(0, 0, 1.0, 40, 30), TRACKS).weight
    assert image[5, 10:30].min() == pytest.approx(1.0)


def test_the_3d_view_draws_a_tracks_layer_as_points():
    from smappy.view3d import projected_settings
    locs = tracks_table([(5, 5), (35, 5)], [1, 1], [0, 1])
    settings, _ = projected_settings(locs, TRACKS)
    assert settings.mode == "precision"


def test_the_render_tab_shows_the_track_controls_only_for_tracks():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.render_tab import RenderTab
    from smappy.session import Session
    locs = tracks_table([(5, 5), (35, 5)], [1, 1], [0, 1])
    session = Session(locs)
    tab = RenderTab(session)
    form = tab._display_form
    assert not form.isRowVisible(tab.track_column)
    tab.set_mode("tracks")
    assert session.layers[0].state.settings.mode == "tracks"
    assert form.isRowVisible(tab.track_column) and form.isRowVisible(tab.line_width)
    tab.line_width.setValue(2.5)
    assert session.layers[0].state.settings.line_width == 2.5
    assert tab.track_column.currentText() == "auto (track_id)"
    assert session.layers[0].state.settings.track_column == ""
    tab.track_column.setCurrentIndex(tab.track_column.findData("frame"))
    assert session.layers[0].state.settings.track_column == "frame"


def test_the_view_says_why_a_picture_could_not_be_drawn():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.render_view import RenderView
    from smappy.session import Session
    view = RenderView(Session(tracks_table([(5, 5), (35, 5)], [1, 1], [0, 1])))
    view._on_render_failed("ValueError: no column 'tid' to take the tracks from", 0)
    assert not view.error_hint.isHidden()
    assert view.error_hint.text().endswith("no column 'tid' to take the tracks from")
    assert "ValueError" not in view.error_hint.text()
    view._busy = True
    view._on_rendered([], -1)                # any answer from the renderer
    assert view.error_hint.isHidden()


def test_merged_blinks_keep_their_track_so_no_line_crosses_the_field():
    # merging used to average track_id_true: one blink of two particles
    # passing close took a third track's id, and was joined to it
    from smappy.session import Session
    from smappy.simulate import simulate_tracks
    session = Session(simulate_tracks(n_frames=300, seed=0))
    layer = session.layers[0]
    assert layer.grouped
    snap = layer.state.snapshot()
    locs = snap.locs
    assert locs["track_id_true"].dtype.kind == "i"
    a, b = track_segments(locs, TRACKS, snap.indices)
    jump = np.hypot(locs["x_nm"][a] - locs["x_nm"][b], locs["y_nm"][a] - locs["y_nm"][b])
    assert jump.max() < 2000


def test_choosing_tracks_unticks_merge_blinks_once():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.render_tab import RenderTab
    from smappy.session import Session
    session = Session(tracks_table([(5, 5), (35, 5)], [1, 1], [0, 1]))
    tab = RenderTab(session)
    assert session.layers[0].grouped and tab.grouped.isChecked()
    tab.set_mode("tracks")
    assert not session.layers[0].grouped and not tab.grouped.isChecked()
    # ticked again by hand, it stays while other track settings change
    tab.grouped.setChecked(True)
    tab.line_width.setValue(2.0)
    assert session.layers[0].grouped


def test_a_line_is_as_bright_however_finely_it_is_cut_into_segments():
    # with round ends every short step added a whole line: zoomed out, a
    # slow track was a blob that set the quantile and dimmed every other track
    fov = FieldOfView(0, 0, 10.0, 100, 100)
    for n in (1, 2, 10, 100, 1000):
        xs = np.linspace(100, 900, n + 1)
        ys = 503.0 + np.linspace(0, 30, n + 1)
        for fast in (True, False):
            image = render_segments(xs[:-1], ys[:-1], xs[1:], ys[1:], fov, 1.0,
                                    use_extension=fast).weight
            across = image[:, 20:80].sum(axis=0)
            assert across == pytest.approx(1.0, abs=0.01)


def test_crossing_tracks_are_no_brighter_than_a_line_unless_lines_add_up():
    # two tracks crossing at (20.5, 15.5), and a joint of one with itself
    locs = tracks_table([(5.5, 15.5), (35.5, 15.5), (20.5, 2.5), (20.5, 28.5)],
                        [1, 1, 2, 2], [0, 1, 0, 1])
    fov = FieldOfView(0, 0, 1.0, 40, 30)
    covering = render_locs(locs, fov, TRACKS).weight
    adding = render_locs(locs, fov, dataclasses.replace(TRACKS, lines_add=True)).weight
    assert covering.max() == pytest.approx(1.0)
    assert covering[15, 20] == pytest.approx(1.0)
    assert adding[15, 20] == pytest.approx(2.0)
    # away from the crossing the two are the same picture
    assert covering[15, 10] == pytest.approx(adding[15, 10]) == pytest.approx(1.0)


def test_covering_lines_keep_their_colour_where_they_overlap():
    locs = tracks_table([(5.5, 15.5), (35.5, 15.5), (20.5, 2.5), (20.5, 28.5)],
                        [1, 1, 2, 2], [0, 1, 0, 1])
    fov = FieldOfView(0, 0, 1.0, 40, 30)
    settings = dataclasses.replace(TRACKS, color_field="track_id", color_range=(1, 2))
    image = render_locs(locs, fov, settings)
    # the colour plane is colour times weight: capped together, the mean
    # colour of the two lines stays where they cross
    mean = image.color[15, 20] / image.weight[15, 20]
    one, two = (image.color[15, 10] / image.weight[15, 10],
                image.color[5, 20] / image.weight[5, 20])
    assert np.allclose(mean, 0.5 * (one + two), atol=1e-5)


def test_tracks_are_coloured_by_their_id_on_turbo_the_first_time_only():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.render_tab import RenderTab
    from smappy.session import Session
    session = Session(tracks_table([(5, 5), (35, 5), (5, 25), (35, 25)],
                                   [1, 1, 2, 2], [0, 1, 0, 1]))
    tab = RenderTab(session)
    state = session.layers[0].state
    assert not tab._more_form.isRowVisible(tab.lines_add)
    tab.set_mode("tracks")
    assert state.settings.color_field == "track_id"
    assert state.display.lut == "turbo"
    assert tab._more_form.isRowVisible(tab.lines_add)
    tab.lines_add.setChecked(True)
    assert state.settings.lines_add
    # one colour chosen by hand stays through a switch away and back
    tab.color.setCurrentIndex(tab.color.findData(None))
    tab.set_mode("precision")
    tab.set_mode("tracks")
    assert state.settings.color_field is None
