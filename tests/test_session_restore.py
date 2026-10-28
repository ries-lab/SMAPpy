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
    display.lut = "jet"
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
