"""A saved file brings its session back: *File > Open and restore*
(`handoff.session_block`, `gui.restore`)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from smappy import handoff
from smappy.images import ImageData
from smappy.io.hdf5 import load_session_block, save_session_block
from smappy.locs import Localizations
from smappy.session import Session
from smappy.view3d import Projection, Slab


def table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 5000, n), "y_nm": rng.uniform(0, 4000, n),
        "z_nm": rng.uniform(-300, 300, n),
        "frame": np.sort(rng.integers(0, 500, n)),
        "photons": rng.uniform(100, 3000, n),
        "xy_err_nm": rng.uniform(3, 20, n),
    })


def arranged(path):
    """A session saved with two layers, an image layer between them, a
    3D view turned and a slab: what a restore should bring back."""
    session = Session()
    session.add_file(table(), _info(path))
    session.add_layer()
    first, second = (l for l in session.layers if not l.is_image)
    first.name, second.name = "bright", "all"
    first.set_bound("photons", 1000.0, None)
    second.visible = False
    second.state.settings.color_field = "z_nm"
    display = second.get_display()
    display.lut = "rainbow"
    second.set_display(display)
    image = ImageData(np.arange(400, dtype=np.float32).reshape(20, 20), 250.0,
                      name="widefield")
    layer = session.add_image(image)
    layer.display.lut = "red hot"
    layer.visible = False
    session.layers.insert(1, session.layers.pop())        # between the two
    session.set_projection(Projection(azimuth=30.0, elevation=40.0, zoom=4.0))
    session.set_slab(Slab([2500, 2000, 0], [1000, 500, 600], angle=15.0))
    session.save(path)
    return session


def _info(path):
    from smappy.io.formats import FileInfo
    return FileInfo(name=path.name, path=str(path), format="smappy", n=3000)


def summary(session):
    out = []
    for l in session.layers:
        if l.is_image:
            out.append(("image", l.name, l.visible, l.display.lut, l.image.name))
        else:
            out.append(("locs", l.name, l.visible, dict(l.ranges).get("photons"),
                        l.state.settings.color_field, l.get_display().lut))
    return out


def test_a_saved_session_block_brings_back_the_layers_and_the_3d_view(tmp_path):
    path = tmp_path / "saved.h5"
    saved = arranged(path)
    assert load_session_block(path)["version"] == handoff.BLOCK_VERSION
    opened = Session()
    opened.load(path)
    assert summary(opened) != summary(saved)               # Open alone does not
    handoff.apply_session_block(opened, load_session_block(path))
    assert summary(opened) == summary(saved)
    assert opened.projection.azimuth == 30.0 and opened.projection.zoom == 4.0
    assert np.allclose(opened.slab.size, [1000, 500, 600]) and opened.slab.angle == 15.0
    assert opened.slab_follows_roi is False


def test_a_file_without_the_block_still_opens_and_restores_nothing(tmp_path):
    path = tmp_path / "old.h5"
    arranged(path)
    save_session_block(path, None)                         # as an older file
    assert load_session_block(path) is None
    opened = Session()
    opened.load(path)
    before = summary(opened)
    handoff.apply_session_block(opened, load_session_block(path))
    assert summary(opened) == before


def test_without_the_gui_state_no_block_is_saved(tmp_path):
    path = tmp_path / "plain.h5"
    session = Session()
    session.add_file(table(), _info(path))
    session.save(path, gui_state=False)
    assert load_session_block(path) is None


def test_a_window_off_every_screen_is_put_on_the_nearest_and_fits_it():
    from smappy.gui.restore import clamp_rect
    laptop = [(0, 0, 1440, 900)]
    # saved on a second monitor to the right, which is not there now
    assert clamp_rect((2000, 100, 800, 600), laptop) == (640, 100, 800, 600)
    # taller than the screen: shrunk, below the title bar
    assert clamp_rect((100, 0, 600, 2000), laptop, top=28) == (100, 28, 600, 872)
    two = [(0, 0, 1440, 900), (1440, 0, 2560, 1440)]
    # on the second screen, where it was
    assert clamp_rect((2000, 100, 800, 600), two) == (2000, 100, 800, 600)
    # straddling: the screen with more of it
    assert clamp_rect((1300, 100, 800, 600), two)[0] == 1440


# ------------------------------------------------------------------ the GUI
@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_for(condition, app, seconds=20.0):
    end = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        app.processEvents()
        time.sleep(0.01)


def test_open_and_restore_brings_back_the_layers_view_and_windows(app, tmp_path):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.gui.identity import uninstall_identity
    STATISTICS = "Analysis/Measure/Localization Statistics"
    path = tmp_path / "gui.h5"
    windows = []
    try:
        session = arranged(tmp_path / "first.h5")
        render = RenderWindow(session)
        control = ControlWindow(session, render)
        windows += [control, render]
        control.show()
        render.show()
        render.setGeometry(300, 200, 700, 500)
        render.view.view.setRange(xRange=(1000, 2000), yRange=(500, 1500), padding=0)
        plugin = control.open_plugin_window(STATISTICS)
        windows.append(plugin)
        values = plugin.panel.form.value()
        plugin.setGeometry(150, 120, 420, 600)
        control.show_3d()
        windows.append(control.view3d_window)
        control.view3d_window.view.projection.elevation = 55.0
        app.processEvents()
        session.save(path)
        saved_layers, saved_view = summary(session), render.view.view.viewRange()
        assert "plugin:" + STATISTICS in load_session_block(path)["windows"]

        other = Session()
        other_render = RenderWindow(other)
        other_control = ControlWindow(other, other_render)
        windows += [other_control, other_render]
        other_control.show()
        other_render.show()
        other_control.open_and_restore(path)
        wait_for(lambda: other_control._task is None and len(other.locs) == 3000
                 and other_control._restore_pending is None, app)
        app.processEvents()
        assert summary(other) == saved_layers
        assert np.allclose(other_render.view.view.viewRange(), saved_view, atol=1.0)
        window = other_control._plugin_windows[STATISTICS]
        windows.append(window)
        assert window.isVisible()
        assert window.panel.form.value() == values
        view3d = other_control.view3d_window
        windows.append(view3d)
        assert view3d is not None and view3d.isVisible()
        assert view3d.view.projection.elevation == 55.0
        assert other.slab.angle == 15.0
        g = other_render.geometry()
        assert (g.width(), g.height()) == (700, 500)
    finally:
        for w in windows:
            w.close()
        uninstall_identity(app)


# ----------------------------------------------- Edit > Copy / Paste settings
def test_layers_pasted_onto_another_dataset_draw_it_as_the_first(tmp_path):
    source = arranged(tmp_path / "a.h5")
    text = handoff.view_settings_text(source)
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    widefield = other.add_image(ImageData(np.zeros((10, 10), np.float32), 100.0,
                                          name="its own image"))
    handoff.apply_view_settings(other, handoff.parse_view_settings(text))
    locs = [s for s in summary(other) if s[0] == "locs"]
    assert locs == [s for s in summary(source) if s[0] == "locs"]
    # its own image layer stays; the copied one's image is not carried over
    images = [l for l in other.layers if l.is_image]
    assert images == [widefield]
    assert len(other.locs) == 3000                         # the data its own


def test_a_layer_restricted_to_a_file_is_shown_whole_where_the_files_differ(tmp_path):
    source = Session()
    source.add_file(table(), _info(tmp_path / "a.h5"))
    source.add_file(table(seed=1), _info(tmp_path / "a2.h5"), append=True)
    source.layers[0].set_files([1])
    settings = handoff.view_settings(source)
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    handoff.apply_view_settings(other, settings)
    assert other.layers[0].files is None
    # between sessions with as many files, the restriction is kept
    again = Session()
    again.add_file(table(seed=8), _info(tmp_path / "c.h5"))
    again.add_file(table(seed=9), _info(tmp_path / "c2.h5"), append=True)
    handoff.apply_view_settings(again, settings)
    assert list(again.layers[0].files) == [1]


def test_text_that_is_not_copied_settings_is_not_taken_for_them():
    assert handoff.parse_view_settings("") is None
    assert handoff.parse_view_settings("some text someone copied") is None
    assert handoff.parse_view_settings('{"layers": []}') is None
    assert handoff.parse_view_settings("[1, 2]") is None


def _windows(app, *sessions, tabs=None):
    """A control and render window for each session, each on a workspace of
    a Render tab and ``tabs`` (copied, so each window has its own)."""
    import copy
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.workspace import Tab, Workspace
    out = []
    for session in sessions:
        render = RenderWindow(session)
        control = ControlWindow(session, render)
        control.workspace = Workspace(tabs=[Tab(name="Render", kind="render")]
                                      + copy.deepcopy(list(tabs or [])))
        control.build_tabs()
        out += [control, render]
    return out


def _trigger(control, text):
    from PySide6.QtGui import QAction
    next(a for a in control.findChildren(QAction) if a.text() == text).trigger()


def _close(app, windows):
    from smappy.gui.identity import uninstall_identity
    for w in windows:
        w.close()
    uninstall_identity(app)


def test_the_edit_menu_copies_the_layers_with_the_render_tab_in_front(app, tmp_path):
    source = arranged(tmp_path / "a.h5")
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    windows = _windows(app, source, other)
    try:
        windows[0].tabs.setCurrentIndex(0)
        _trigger(windows[0], "Copy settings")
        assert windows[0].statusBar().currentMessage().startswith("copied: 2 layers")
        windows[2].tabs.setCurrentIndex(0)
        _trigger(windows[2], "Paste settings")
        assert [s for s in summary(other) if s[0] == "locs"] == \
            [s for s in summary(source) if s[0] == "locs"]
    finally:
        _close(app, windows)


COMET = "Analysis/Drift/COMET"


def test_a_plugins_open_section_is_copied_without_the_layers_and_pasted_into_that_plugin(
        app, tmp_path):
    from smappy.workspace import Instance, Tab
    source = arranged(tmp_path / "a.h5")
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    before = summary(other)
    drift = Tab(name="Drift", instances=[Instance(plugin="Analysis/Drift/RCC"),
                                         Instance(plugin=COMET, id="comet")])
    windows = _windows(app, source, other, tabs=[drift])
    try:
        control, target = windows[0], windows[2]
        control.tabs.setCurrentIndex(1)
        plugin_tab = control.tabs.currentWidget()
        plugin_tab.open_section("comet")
        plugin_tab.slots["comet"].panel.form.set_values({"segmentation_var": 777})
        said = control.copy_settings()
        assert said == "copied: COMET drift correction, from the Drift tab"
        # the other window has the Render tab in front and COMET never opened
        target.tabs.setCurrentIndex(0)
        said = target.paste_settings()
        assert said == "pasted: COMET drift correction into the Drift tab"
        assert summary(other) == before                      # its layers untouched
        instance = target.workspace.tabs[1].instances[1]
        assert instance.values["segmentation_var"] == 777
        panel = target.tabs.widget(1).slots["comet"].ensure()
        assert panel.form.values()["segmentation_var"] == 777
        # and into its window, where one is open
        window = target.open_plugin_window(COMET)
        window.show()
        assert target.paste_settings().endswith("into the Drift tab and its window")
        assert window.panel.form.values()["segmentation_var"] == 777
    finally:
        _close(app, windows)


def test_a_plugin_this_window_does_not_have_is_named_when_nothing_is_pasted(app, tmp_path):
    from smappy.workspace import Instance, Tab
    source = arranged(tmp_path / "a.h5")
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    windows = _windows(app, source, other)
    try:
        text = handoff.settings_text(handoff.plugin_settings(
            [(COMET, {"segmentation_var": 5}, "")]))
        with pytest.raises(ValueError, match="COMET drift correction is pinned to no tab"):
            windows[2].paste_settings(text)
    finally:
        _close(app, windows)


def test_nothing_is_copied_from_a_tab_with_every_section_closed(app, tmp_path):
    from smappy.workspace import Instance, Tab
    source = arranged(tmp_path / "a.h5")
    windows = _windows(app, source, tabs=[Tab(name="Drift",
                                              instances=[Instance(plugin=COMET)])])
    try:
        windows[0].tabs.setCurrentIndex(1)
        with pytest.raises(ValueError, match="nothing to copy"):
            windows[0].copy_settings()
    finally:
        _close(app, windows)


def test_settings_kept_in_a_text_file_paste_as_they_were_copied(tmp_path):
    source = arranged(tmp_path / "a.h5")
    kept = tmp_path / "our view.txt"
    kept.write_text("\n" + handoff.view_settings_text(source) + "\n\n")
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    handoff.apply_view_settings(other, handoff.parse_view_settings(kept.read_text()))
    assert [s for s in summary(other) if s[0] == "locs"] == \
        [s for s in summary(source) if s[0] == "locs"]


def _plugin_window_and_tab(app, tmp_path):
    """Two windows; the first with a Drift tab whose COMET section is open
    and set, and an RCC window of its own, also set."""
    from smappy.workspace import Instance, Tab
    source = arranged(tmp_path / "a.h5")
    other = Session()
    other.add_file(table(seed=7), _info(tmp_path / "b.h5"))
    drift = Tab(name="Drift", instances=[Instance(plugin=COMET, id="comet"),
                                         Instance(plugin=RCC, id="rcc")])
    windows = _windows(app, source, other, tabs=[drift])
    control = windows[0]
    control.tabs.setCurrentIndex(1)
    control.tabs.currentWidget().open_section("comet")
    window = control.open_plugin_window(RCC)
    window.panel.form.set_values({"n_timepoints": 33})
    return windows, window


RCC = "Analysis/Drift/RCC"


def test_the_edit_menu_opened_by_a_click_from_a_plugin_window_copies_that_plugin(
        app, tmp_path):
    windows, window = _plugin_window_and_tab(app, tmp_path)
    control = windows[0]
    try:
        # the click on the menu bar: the control window comes to the front,
        # the menu opens at once, the item is chosen
        control._note_front(window)
        control._note_front(control)
        control._edit_menu_shown()
        control._edit_menu_hidden()
        _trigger(control, "Copy settings")
        assert control.statusBar().currentMessage() == \
            "copied: RCC drift correction, from its window"
        # a shortcut pressed later, in the control window: its tab
        _trigger(control, "Copy settings")
        assert control.statusBar().currentMessage() == \
            "copied: COMET drift correction, from the Drift tab"
    finally:
        _close(app, windows)


def test_the_edit_menu_opened_after_working_in_the_control_window_copies_its_tab(
        app, tmp_path):
    windows, window = _plugin_window_and_tab(app, tmp_path)
    control = windows[0]
    try:
        control._note_front(window)
        control._note_front(control)
        control._front_since -= 5            # five seconds in the tab since
        control._edit_menu_shown()
        control._edit_menu_hidden()
        _trigger(control, "Copy settings")
        assert control.statusBar().currentMessage() == \
            "copied: COMET drift correction, from the Drift tab"
    finally:
        _close(app, windows)


def test_a_plugin_windows_right_click_menu_copies_it_and_pastes_into_another(
        app, tmp_path):
    windows, window = _plugin_window_and_tab(app, tmp_path)
    try:
        target = windows[2]
        other_window = target.open_plugin_window(RCC)
        copy, paste = window.settings_actions
        copy.trigger()
        assert windows[0].statusBar().currentMessage() == \
            "copied: RCC drift correction, from its window"
        other_window.settings_actions[1].trigger()
        assert other_window.panel.form.values()["n_timepoints"] == 33
        assert target.workspace.tabs[1].instances[1].values["n_timepoints"] == 33
    finally:
        _close(app, windows)
