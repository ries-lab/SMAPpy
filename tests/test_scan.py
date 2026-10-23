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
