"""The Render tab's player: a window of a coordinate moved through the data,
narrowing the pictures and nothing else."""
import numpy as np
import pytest

from smappy import scan as sc
from smappy.layer_state import snapshot_layers
from smappy.locs import Localizations
from smappy.regions import Region
from smappy.session import Session


def table(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 4000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 4000, n).astype(np.float32),
        "z_nm": rng.uniform(-400, 400, n).astype(np.float32),
        "frame": rng.integers(0, 1000, n).astype(np.int64),
        "photons": rng.exponential(1000, n).astype(np.float32),
        "xy_err_nm": np.full(n, 10.0, np.float32),
    }, {})


def test_along_and_across_a_line_are_measured_from_its_middle():
    line = Region.line((1000, 1000), (3000, 1000), 200.0)
    locs = Localizations({"x_nm": np.array([2000.0, 2500.0, 1000.0]),
                          "y_nm": np.array([1000.0, 1000.0, 1300.0])}, {})
    np.testing.assert_allclose(sc.values(locs, sc.ALONG, line), [0, 500, -1000])
    np.testing.assert_allclose(sc.values(locs, sc.ACROSS, line), [0, 0, 300])
    with pytest.raises(ValueError, match="line ROI"):
        sc.values(locs, sc.ALONG, Region.rect(0, 0, 1, 1))


def test_the_steps_overlap_and_reach_the_last_frame():
    steps = sc.scan_steps(0, 999, window=100, step=25, integer=True)
    assert (steps[0].lo, steps[0].hi) == (0, 100)
    assert steps[1].lo == 25
    assert steps[-1].hi >= 1000 > steps[-2].hi
    window, step = sc.auto_window(0, 999, integer=True)
    assert (window, step) == (50.0, 12.0)
    with pytest.raises(ValueError):
        sc.scan_steps(0, 1, 0, 1)


def test_a_scan_narrows_each_layer_on_top_of_its_filter():
    session = Session(table())
    session.layers[0].set_bound("photons", 500, None)
    layers = snapshot_layers(session.layers)
    before = np.asarray(layers[0].mask, bool)
    assert 0 < before.sum() < len(before)
    shown = sc.scanned(layers, sc.Scan("frame", 100, 200))
    frames = np.asarray(layers[0].locs["frame"])
    expected = before & (frames >= 100) & (frames < 200)
    np.testing.assert_array_equal(np.asarray(shown[0].mask, bool), expected)
    # a coordinate a layer lacks leaves it as it is
    assert sc.scanned(layers, sc.Scan("nothing", 0, 1))[0] is layers[0]
    assert sc.scanned(layers, None) == tuple(layers)


def test_a_scan_of_some_layers_leaves_the_others_whole():
    session = Session(table())
    session.add_layer(like=0)
    session.layers[1].set_bound("frame", None, 500)
    layers = snapshot_layers(session.layers)
    shown = sc.scanned(layers, sc.Scan("frame", 100, 200, layers=(1,)))
    assert shown[0] is layers[0]
    frames = np.asarray(layers[1].locs["frame"])[np.asarray(shown[1].mask, bool)]
    assert len(frames) and frames.min() >= 100 and frames.max() < 200
    # the range runs over the scanned layers only
    assert sc.shown_range(layers, "frame", which=(1,))[1] <= 500
    assert sc.shown_range(layers, "frame")[1] >= 900


def test_the_session_narrows_the_pictures_and_nothing_a_plugin_reads():
    session = Session(table())
    seen = []
    session.on_change(seen.append)
    n = len(session.context().selection.apply(session.context().locs))
    session.set_scan(sc.Scan("z_nm", 0, 100))
    assert "scan" in seen
    shown = session.scanned(snapshot_layers(session.layers))
    z = np.asarray(shown[0].locs["z_nm"])[np.asarray(shown[0].mask, bool)]
    assert len(z) and z.min() >= 0 and z.max() < 100
    assert len(session.context().selection.apply(session.context().locs)) == n
    session.set_scan(None)
    assert session.scanned("as is") == "as is"


def test_recorded_pictures_are_one_per_step_and_show_the_step():
    from smappy.picture import ViewFrame
    session = Session(table())
    layers = snapshot_layers(session.layers)
    frame = ViewFrame((0.0, 4000.0), (0.0, 4000.0), 40.0)
    steps = sc.scan_steps(0, 999, 250, 250, integer=True)
    images = list(sc.scan_pictures(layers, frame, "frame", steps))
    assert len(images) == 4 and images[0].shape == (100, 100, 3)
    assert not np.array_equal(images[0], images[1])
    assert sc.step_text("frame", steps[1]) == "frames 250-499"
    assert sc.step_text("frame", steps[1], 20.0) == "5.0-10.0 s"
    assert sc.step_text(sc.ACROSS, sc.Step(-10, 10)) == "across the line -10 to 10 nm"


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_player_steps_bounces_and_stops(app):
    from smappy.gui.scan_player import ScanPlayer
    session = Session(table())
    player = ScanPlayer(session)
    player.coordinate.setCurrentText("frame")
    player.window_.setValue(200)
    player.step.setValue(100)
    player.show_step(0)
    assert (session.scan.lo, session.scan.hi) == (0, 200)
    n = len(player.steps)
    assert player.slider.maximum() == n - 1
    player.slider.setValue(n - 1)
    assert session.scan.lo == 100 * (n - 1)
    player.ends.setCurrentIndex(player.ends.findData("bounce"))
    player._tick()
    assert player.slider.value() == n - 2
    player.ends.setCurrentIndex(player.ends.findData("loop"))
    player.slider.setValue(n - 1)
    player._direction = 1
    player._tick()
    assert player.slider.value() == 0
    player.stop()
    assert session.scan is None


def test_the_player_scans_across_a_line_within_its_width(app):
    from smappy.gui.scan_player import ScanPlayer
    session = Session(table())
    player = ScanPlayer(session)
    player.coordinate.setCurrentText(sc.ACROSS)
    player.show_step(0)
    assert session.scan is None and "line ROI" in player.where.text()
    session.set_roi(Region.line((1000, 2000), (3000, 2000), 400.0))
    player.show_step(0)
    assert session.scan.lo == pytest.approx(-200.0)
    assert player.steps[-1].hi >= 200.0


def test_the_player_records_what_the_2d_window_shows(app, tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    from smappy.gui import scan_player
    from smappy.gui.render_view import RenderView
    session = Session(table())
    view = RenderView(session)
    view.resize(200, 200)
    player = scan_player.ScanPlayer(session)
    player.coordinate.setCurrentText("frame")
    player.window_.setValue(500)
    player.step.setValue(250)
    path = tmp_path / "scan.tif"
    monkeypatch.setattr(scan_player.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(path), ""))
    player._record()
    with tifffile.TiffFile(path) as tif:
        stack = tif.asarray()
    assert stack.shape[0] == len(player.steps) == 3       # 0-499, 250-749, 500-999
    assert stack.dtype == np.uint8


def test_the_player_scans_only_the_layers_ticked(app):
    from smappy.gui.scan_player import ScanPlayer
    session = Session(table())
    session.add_layer(like=0)
    player = ScanPlayer(session)
    player.coordinate.setCurrentText("frame")
    player.show_step(0)
    assert session.scan.layers is None and player.layer_choice.text() == "all"
    player.set_scanned(0, False)
    assert session.scan.layers == (1,)
    player._fill_layers()
    assert [a.isChecked() for a in player.layer_choice.menu().actions()] == [False, True]
    session.add_layer(like=0)             # positions mean others now: all again
    assert player.scanned_layers() is None


def test_the_view_path_alone_plays_its_keys_in_the_3d_view(app):
    from smappy.gui.scan_player import NONE, ScanPlayer
    from smappy.gui.view3d import View3DWindow
    session = Session(table())
    window = View3DWindow(session)
    window.show()
    app.processEvents()
    player = ScanPlayer(session)
    assert player.coordinate.currentText() == "frame"      # not the view path alone
    window.view.projection.opacity = 0.4                     # the look, not the viewpoint
    window.view.refresh_frame()                              # as a render does
    player.add_view()
    window.view.projection.rotate_view(80.0, 0.0)
    window.view.projection.zoom *= 0.25
    window.view.refresh_frame()
    end = window.view.projection.matrix.copy()
    player.add_view()                                        # 2 s after the first
    player.ease.setChecked(False)
    player.coordinate.setCurrentText(NONE)
    player.speed.setValue(5)
    steps = player.prepare()
    assert len(steps) == 11 and not player.window_.isEnabled()
    player.show_step(5)
    assert session.scan is None
    proj = window.view.projection
    np.testing.assert_allclose(proj.matrix, player.path.at(0.5).rotation, atol=1e-9)
    assert proj.opacity == 0.4
    player.show_step(10)
    np.testing.assert_allclose(proj.matrix, end, atol=1e-9)
    # a turn adds a key every 90 degrees from the last key
    player.turn_degrees.setValue(180.0)
    player.add_turn_keys()
    assert len(player.path.keys) == 4
    window.close()


def test_the_view_path_follows_the_scan_and_is_recorded_from_the_3d_view(app, tmp_path,
                                                                      monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    from smappy.gui import scan_player
    from smappy.gui.view3d import View3DWindow
    session = Session(table())
    window = View3DWindow(session)
    window.resize(160, 160)
    window.show()
    app.processEvents()
    window.view.refresh_frame()
    player = scan_player.ScanPlayer(session)
    player.turn_degrees.setValue(90.0)
    player.add_turn_keys()                                   # the view now, then 90 degrees
    player.coordinate.setCurrentText("frame")
    player.window_.setValue(500)
    player.step.setValue(250)
    player.show_step(len(player.prepare()) - 1)              # the scan's end: the path's
    np.testing.assert_allclose(window.view.projection.matrix,
                               player.path.keys[-1].view.rotation, atol=1e-9)
    player.record_from.setCurrentIndex(player.record_from.findData("3D"))
    path = tmp_path / "turn.tif"
    monkeypatch.setattr(scan_player.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(path), ""))
    player._record()
    with tifffile.TiffFile(path) as tif:
        stack = tif.asarray()
    assert stack.shape[0] == 3
    assert not np.array_equal(stack[0], stack[-1])
    window.close()


def test_the_simple_player_folds_the_layers_and_the_view_path_away(app):
    from smappy.gui.scan_player import ScanPlayer
    from smappy.gui.widgets import CollapsibleSection
    player = ScanPlayer(Session(table()))
    more, = [c for c in player.findChildren(CollapsibleSection) if c.title == "more"]
    assert more.frame.isHidden()
    assert player.layer_choice.parent() is not player and more.isAncestorOf(player.keys)


def test_the_view_path_moves_the_box_its_keys_keep(app):
    from smappy.gui.scan_player import NONE, ScanPlayer
    from smappy.gui.view3d import View3DWindow
    from smappy.view3d import Slab
    session = Session(table())
    window = View3DWindow(session)
    window.show()
    app.processEvents()
    window.view.refresh_frame()
    player = ScanPlayer(session)
    session.set_slab(Slab([2000, 2000, 0], [4000, 4000, 800]))
    player.add_view()
    # the front face moved back by 600 nm, the back face kept
    session.set_slab(Slab([2000, 2000, -300], [4000, 4000, 200]))
    player.add_view()
    assert player.keys.item(1).text().endswith("box 4000 x 4000 x 200")
    player.ease.setChecked(False)
    player.coordinate.setCurrentText(NONE)
    steps = player.prepare()
    player.show_step(len(steps) // 2)
    assert session.slab.size[2] == pytest.approx(500.0)
    assert session.slab.center[2] - session.slab.size[2] / 2 == pytest.approx(-400.0)
    assert not session.slab_follows_roi
    window.close()


def test_stop_redraws_the_picture_even_while_a_step_is_drawing(app):
    """The scan cleared while a step's tile was in the render thread: that
    tile came back, was taken as current, and the render waiting behind it
    found the view covered -- the step stayed on screen after stop."""
    from smappy.gui.render_view import RenderView
    session = Session(table())
    view = RenderView(session)
    view.resize(200, 200)
    view.render_requested.disconnect()
    jobs = []
    view.render_requested.connect(lambda job, generation: jobs.append((job, generation)))
    session.set_scan(sc.Scan("frame", 0, 100))
    assert len(jobs) == 1                      # the step, drawing
    session.set_scan(None)                     # stop, while it draws
    job, generation = jobs[0]
    view._on_rendered([np.zeros((job.fov.ny, job.fov.nx, 3), np.float32)], generation)
    assert len(jobs) == 2                      # drawn again, whole
    job, generation = jobs[1]
    view._on_rendered([np.zeros((job.fov.ny, job.fov.nx, 3), np.float32)], generation)
    assert view.tile_valid and not view.frame_source.drawing


def test_space_plays_whichever_was_used_last_the_player_or_the_3d_rock(app):
    from smappy.gui.animation import animations
    from smappy.gui.scan_player import ScanPlayer
    from smappy.gui.view3d import View3DWindow
    session = Session(table())
    window = View3DWindow(session)
    window.show()
    app.processEvents()
    record = animations(session)
    motion = window.view.motion
    # no player used yet: Space in the 3D window rocks, as it always did
    record.space(motion)
    assert motion.active
    record.space(motion)
    assert not motion.active
    player = ScanPlayer(session)
    record.set_player(player)
    player.coordinate.setCurrentText("frame")
    player.used.emit()                          # its section opened
    record.space(motion)                        # now the player, in the 3D window too
    assert player.play.isChecked() and not motion.active
    record.space()                              # and in the render window
    assert not player.play.isChecked()
    motion.start()                              # the rock used last again
    player.timer.stop()
    record.space()
    assert not motion.active and not player.play.isChecked()
    motion.halt()
    window.close()


def test_alt_wheel_steps_the_player_and_alt_drag_scrubs_it_over_the_picture(app):
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    from smappy.gui.render_view import RenderView
    from smappy.gui.scan_player import ScanPlayer
    session = Session(table())
    view = RenderView(session)
    view.resize(400, 400)
    player = ScanPlayer(session)
    view.scrubber = player
    player.coordinate.setCurrentText("frame")
    player.window_.setValue(100)
    player.step.setValue(100)
    n = len(player.prepare())
    viewport = view.graphics.viewport()
    range_before = view.view.viewRange()

    def wheel(dy, modifiers=Qt.AltModifier):
        event = QWheelEvent(QPointF(200, 200), QPointF(200, 200), QPoint(0, 0), QPoint(0, dy),
                            Qt.NoButton, modifiers, Qt.NoScrollPhase, False)
        return view.eventFilter(viewport, event)

    assert wheel(-120)                          # a notch down: the first step, from stopped
    assert (session.scan.lo, session.scan.hi) == (0, 100)
    assert wheel(-240) and player.slider.value() == 2
    assert wheel(-60) and player.slider.value() == 2     # half a notch: not yet
    assert wheel(-60) and player.slider.value() == 3
    assert wheel(120) and player.slider.value() == 2     # up: back
    assert view.view.viewRange() == range_before          # no zoom meanwhile
    assert not wheel(-120, Qt.NoModifier)                 # the plain wheel is pyqtgraph's

    def mouse(kind, x, modifiers=Qt.AltModifier, button=Qt.LeftButton):
        buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        event = QMouseEvent(kind, QPointF(x, 200), QPointF(x, 200), button, buttons, modifiers)
        return view.eventFilter(viewport, event)

    width = viewport.width()
    assert mouse(QEvent.MouseButtonPress, 0)
    assert mouse(QEvent.MouseMove, width / 2)
    assert player.slider.value() == int(round(2 + (n - 1) / 2))
    assert mouse(QEvent.MouseMove, 10 * width)            # past the end: the last step
    assert player.slider.value() == n - 1
    assert mouse(QEvent.MouseButtonRelease, 10 * width)
    assert not mouse(QEvent.MouseMove, 0, Qt.NoModifier)  # released: moves are pyqtgraph's
    view.scrubber = None                                  # a picture without a player
    assert not wheel(-120)
