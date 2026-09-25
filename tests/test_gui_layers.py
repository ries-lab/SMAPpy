"""Layers in the Render tab: what a new one inherits, and what the strip shows.

A second layer is nearly always the first one with one thing changed, so
`add_layer` copies rather than starting from the defaults; and the strip has
two things to say at once -- which layer is drawn, and which one the controls
below edit -- which is why they use different cues.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy import lut as luts                                         # noqa: E402
from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(seed, n=4000):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "frame": rng.integers(0, 200, n).astype(np.int64),
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 25, n).astype(np.float32),
    }, {"units": "nm"})


def two_files():
    session = Session()
    session.add_file(table(0), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    session.add_file(table(1), FileInfo("b.hdf5", "/b.hdf5", "smappy"))
    return session


def test_a_new_layer_is_a_copy_of_the_one_it_was_added_from():
    session = two_files()
    first = session.layers[0]
    first.set_files([1])
    first.set_bound("photons", 200.0, 800.0)
    first.set_display(dataclasses.replace(first.get_display(), lut="viridis",
                                          invert=True, gamma=0.7))
    first.state.settings = dataclasses.replace(first.state.settings, mode="gauss",
                                               sigma=12.0)

    second = session.add_layer(like=0)

    assert second.files == [1]                      # including the file it shows
    assert second.state.sets["ungrouped"].filter.ranges["photons"] == (200.0, 800.0)
    assert (second.get_display().lut, second.get_display().invert) == ("viridis", True)
    assert second.get_display().gamma == pytest.approx(0.7)
    assert (second.state.settings.mode, second.state.settings.sigma) == ("gauss", 12.0)
    assert second.grouped == first.grouped
    # copies, not the same objects: editing one layer must not move the other
    assert second.get_display() is not first.get_display()
    assert second.state.settings is not first.state.settings


def test_without_a_template_a_new_layer_still_starts_empty_and_default():
    session = two_files()
    session.layers[0].set_bound("photons", 200.0, 800.0)
    fresh = session.add_layer()
    assert fresh.files == []                        # pick the file(s) it shows
    assert fresh.state.sets["ungrouped"].filter.ranges.get("photons") != (200.0, 800.0)


def test_the_add_button_copies_the_layer_the_tab_is_on_and_moves_to_it(app):
    from smappy.gui.render_tab import RenderTab
    session = two_files()
    tab = RenderTab(session)
    session.layers[0].set_bound("photons", 300.0, 700.0)

    tab.strip._add()

    assert tab.strip.current == 1 and tab.filter.layer_index == 1
    assert session.layers[1].state.sets["ungrouped"].filter.ranges["photons"] == (300.0, 700.0)


def test_the_strip_says_visible_and_selected_in_different_ways(app):
    from smappy.gui.render_tab import RenderTab
    session = two_files()
    tab = RenderTab(session)
    tab.strip._add()                                # two layers, on the second
    chosen, other = tab.strip.buttons[1], tab.strip.buttons[0]
    assert "palette(highlight)" in chosen.styleSheet()
    assert "palette(highlight)" not in other.styleSheet()
    assert "bold" in chosen.styleSheet() and "line-through" not in chosen.styleSheet()

    tab.strip._toggle_visible(1)                    # hide the selected layer

    assert "line-through" in chosen.styleSheet()    # not drawn any more
    assert "palette(highlight)" in chosen.styleSheet()   # but still the one edited
    assert tab.strip.current == 1


def test_invert_reaches_the_layer_and_the_form_reads_it_back(app):
    from smappy.gui.render_tab import RenderTab
    session = two_files()
    tab = RenderTab(session)
    tab.invert.setChecked(True)
    assert session.layers[0].get_display().invert
    tab.strip._add()                                # the copy has it too
    assert tab.invert.isChecked() and session.layers[1].get_display().invert
    tab._bind_layer(0)
    assert tab.invert.isChecked()


def test_white_background_turns_the_picture_over_once(app):
    """White paper is a property of the picture: every layer takes the setting,
    and the sum is turned over once -- a white ground added to a white ground
    would swallow everything drawn on either of them."""
    from smappy.gui.render_tab import RenderTab
    from smappy.gui.render_view import RenderView

    session = two_files()
    view = RenderView(session)
    tab = RenderTab(session, view)
    session.add_layer(like=0)
    fov = view.current_fov(200, 160)

    dark, _ = view.composite(fov)
    assert dark[0, 0].max() < 0.05                      # the ground is black

    tab.white.setChecked(True)
    assert all(l.get_display().white_background for l in session.layers)
    light, _ = view.composite(fov)
    assert light[0, 0].min() > 0.95                     # ...and now white
    assert (light.min(axis=2) < 0.6).sum() > 100        # with the ink still on it
    # the same picture, turned over: brightness flipped, hue kept
    assert np.allclose(light, luts.on_white(dark), atol=1e-6)

    tab.white.setChecked(False)
    assert not any(l.get_display().white_background for l in session.layers)


def test_the_editable_fields_are_marked_and_the_read_only_ones_are_not():
    """"Which of these can I change?" should be answerable at a glance: an
    editable field gets the base colour and a border, a read-only one keeps
    the window's."""
    from PySide6.QtWidgets import QApplication

    from smappy.gui.widgets import EDITABLE_STYLE, apply_style

    app = QApplication.instance() or QApplication([])
    before = app.styleSheet() or ""
    try:
        apply_style(app)
        assert EDITABLE_STYLE in app.styleSheet()
        apply_style(app)                     # twice does not stack it
        assert app.styleSheet().count(EDITABLE_STYLE) == 1
        assert 'QPlainTextEdit[readOnly="true"]' in EDITABLE_STYLE
        assert "palette(base)" in EDITABLE_STYLE       # not a hard-coded white
    finally:
        app.setStyleSheet(before)


def test_copying_settings_from_another_layer_takes_only_what_is_asked_for():
    """What makes two layers two -- the filter, which file each shows -- must
    not ride along with "make this one look like that one"."""
    import dataclasses

    from smappy.session import Session

    session = Session(_layer_table())
    session.add_layer(like=0)
    first, second = session.layers
    first.set_display(dataclasses.replace(first.get_display(), lut="hot"))
    first.state.settings = dataclasses.replace(first.state.settings, sigma=7.0)
    second.set_bound("photons", 700, None)

    session.copy_layer(0, 1)
    assert second.get_display().lut == "hot"
    assert second.state.settings.sigma == 7.0
    assert second.state.sets["ungrouped"].filter.ranges["photons"] == (700, None)

    # and the bounds when they are asked for: the other layer's, not both
    session.copy_layer(0, 1, display=False, bounds=True)
    assert "photons" not in second.state.sets["ungrouped"].filter.ranges
    assert (second.state.sets["ungrouped"].filter.ranges
            == first.state.sets["ungrouped"].filter.ranges)


def test_copying_a_layer_onto_itself_or_out_of_range_does_nothing():
    from smappy.session import Session

    session = Session(_layer_table())
    session.add_layer(like=0)
    before = dict(session.layers[1].state.sets["ungrouped"].filter.ranges)
    session.copy_layer(1, 1, bounds=True)
    session.copy_layer(5, 1, bounds=True)
    assert dict(session.layers[1].state.sets["ungrouped"].filter.ranges) == before


def _layer_table(n=2000):
    import numpy as np

    from smappy.locs import Localizations

    rng = np.random.default_rng(0)
    return Localizations({"x_nm": rng.uniform(0, 1000, n),
                          "y_nm": rng.uniform(0, 1000, n),
                          "frame": (np.arange(n) % 50).astype(np.int64),
                          "photons": rng.exponential(500, n),
                          "xy_err_nm": rng.uniform(5, 40, n)}, {})


def test_the_invert_tick_and_the_inversion_are_one_setting():
    """Which inversion is a choice, not a decision made in the render tab, and
    both live in the one `invert` field so nothing downstream grows a second
    one to carry through the GPU cache key and the saved workspace."""
    import dataclasses

    from PySide6.QtWidgets import QApplication

    from smappy import lut as luts
    from smappy.gui.render_tab import RenderTab
    from smappy.session import Session

    QApplication.instance() or QApplication([])
    session = Session()
    tab = RenderTab(session)
    session.set_locs(_layer_table())
    layer = session.layers[0]

    assert layer.get_display().invert is False and not tab.invert_mode.isEnabled()
    tab.invert.setChecked(True)
    assert layer.get_display().invert == luts.DEFAULT_INVERSION
    assert tab.invert_mode.isEnabled()
    tab.invert_mode.setCurrentIndex(tab.invert_mode.findData("complement"))
    assert layer.get_display().invert == "complement"
    tab.invert.setChecked(False)
    assert layer.get_display().invert is False

    # a workspace written before there was a choice carries a plain True
    layer.set_display(dataclasses.replace(layer.get_display(), invert=True))
    tab._bind_layer(0)
    assert tab.invert.isChecked()
    assert tab.invert_mode.currentData() == luts.DEFAULT_INVERSION
