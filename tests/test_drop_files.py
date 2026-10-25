"""Files dropped on the main windows: a table opens, a stack goes to a fitter.

The drops are real `QDropEvent`s sent to the windows, so what is tested is
the path a file from the desktop takes -- the event, the deferred open, the
sorting -- and not only the method at the end of it.
"""
import time
import warnings

import numpy as np
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def windows(app):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    yield session, control, render
    control.stop_loading()
    for window in (control, render):
        window.close()
        window.deleteLater()


def mime_for(*paths):
    from PySide6.QtCore import QMimeData, QUrl
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def drop(app, window, *paths):
    """Drag ``paths`` over ``window`` and let go; True if it took the drag."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    mime = mime_for(*paths)
    centre = window.rect().center()
    with warnings.catch_warnings():      # PySide marks the QPoint form deprecated
        warnings.simplefilter("ignore", DeprecationWarning)
        enter = QDragEnterEvent(centre, Qt.CopyAction, mime, Qt.LeftButton,
                                Qt.NoModifier)
    QApplication.sendEvent(window, enter)
    if not enter.isAccepted():
        return False
    event = QDropEvent(QPointF(centre), Qt.CopyAction, mime, Qt.LeftButton,
                       Qt.NoModifier)
    QApplication.sendEvent(window, event)
    app.processEvents()                  # the open is deferred past the drop
    return True


def wait_for_loads(app, control, timeout=60.0):
    deadline = time.monotonic() + timeout
    while (control._task is not None or control._queue) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert control._task is None, "the load did not finish"


def test_an_hdf5_dropped_on_the_picture_opens_as_file_open_would(app, windows, tmp_path):
    from smappy.io.formats import save
    from smappy.simulate import simulate
    session, control, render = windows
    locs = simulate(n_frames=500, seed=2)
    path = save(locs, tmp_path / "simulated.hdf5")

    assert drop(app, render, path)
    wait_for_loads(app, control)

    assert len(session.locs) == len(locs)
    assert session.file_names() == ["simulated.hdf5"]
    assert np.allclose(np.sort(np.asarray(session.locs["x_nm"])),
                       np.sort(np.asarray(locs["x_nm"])))


@pytest.mark.parametrize("answer, names", [
    (False, ["a.hdf5", "b.hdf5"]),
    (True, ["old.hdf5", "a.hdf5", "b.hdf5"])])
def test_tables_dropped_on_open_ones_replace_them_or_join_them_as_answered(
        app, windows, tmp_path, monkeypatch, answer, names):
    from smappy.io.formats import save
    from smappy.simulate import simulate
    session, control, _ = windows
    a = save(simulate(n_frames=300, seed=3), tmp_path / "a.hdf5")
    b = save(simulate(n_frames=300, seed=4), tmp_path / "b.hdf5")
    old = save(simulate(n_frames=300, seed=5), tmp_path / "old.hdf5")
    control.load_paths([str(old)])
    wait_for_loads(app, control)
    asked = []
    monkeypatch.setattr(control, "replace_or_add",
                        lambda paths: asked.append(paths) or answer)

    assert drop(app, control, a, b)
    wait_for_loads(app, control)

    assert len(asked) == 1 and [p.name for p in asked[0]] == ["a.hdf5", "b.hdf5"]
    assert session.file_names() == names


def test_nothing_is_asked_when_no_table_is_open(app, windows, tmp_path, monkeypatch):
    from smappy.io.formats import save
    from smappy.simulate import simulate
    session, control, _ = windows
    assert not len(session.locs)
    path = save(simulate(n_frames=300, seed=3), tmp_path / "a.hdf5")
    assert control.replace_or_add([path]) is False


def _write_tif(path, frames=60):
    import tifffile
    tifffile.imwrite(path, np.full((frames, 32, 32), 100, np.uint16),
                     photometric="minisblack")
    return path


def _write_recipe(path):
    from smappy.simulate import SimulationSettings
    from smappy.simulate.source import write_recipe
    return write_recipe(SimulationSettings(output="camera", n_frames=20), path)


@pytest.mark.parametrize("make, name", [(_write_recipe, "run.sim.yaml"),
                                        (_write_tif, "stack.tif")])
def test_a_dropped_stack_lands_in_the_fitters_file_on_the_localize_tab(
        app, windows, tmp_path, make, name):
    from smappy.plugins.fit import default_output_path
    session, control, _ = windows
    path = make(tmp_path / name)

    assert drop(app, control, path)

    assert control.tabs.tabText(control.tabs.currentIndex()) == "Localize"
    tab = control.tabs.currentWidget()
    opened = tab.tab.instances[tab.tab.index_of(tab.open_instance())]
    assert opened.plugin == "Localize/Gaussian 2D"
    panel = tab.slots[opened.id].panel
    values = panel.form.values()
    assert values["source.path"] == str(path)
    # the fitter heard of it, as from its own ... button: `react` set the output
    assert values["output.path"] == str(default_output_path(path))
    assert not len(session.locs)                     # and nothing was fitted
    assert control.statusBar().currentMessage() == \
        f"Gaussian 2D: file set to {name}; Run fits it"


def test_a_stack_goes_to_the_fitter_open_on_the_localize_tab(app, windows, tmp_path):
    session, control, _ = windows
    tab = next(control.tabs.widget(i) for i in range(control.tabs.count())
               if control.tabs.tabText(i) == "Localize")
    spline = next(i for i in tab.tab.instances if i.plugin == "Localize/Spline 3D")
    tab.slots[spline.id].ensure()
    tab.open_section(spline.id)
    path = _write_recipe(tmp_path / "beads.sim.yaml")

    assert drop(app, control, path)

    assert tab.open_instance() == spline.id
    assert tab.slots[spline.id].panel.form.values()["source.path"] == str(path)


def test_a_short_tiff_is_an_image_to_show_and_a_long_one_data_to_fit(tmp_path):
    """Raw SMLM data runs to thousands of frames; a few frames are a
    widefield picture or some channels to render with the localizations."""
    from smappy.gui import drops
    assert drops.kind_of(_write_tif(tmp_path / "wf.tif", frames=1)) == drops.IMAGE
    assert drops.kind_of(_write_tif(tmp_path / "ch.tif",
                                    frames=drops.MAX_IMAGE_FRAMES)) == drops.IMAGE
    assert drops.kind_of(_write_tif(tmp_path / "raw.tif",
                                    frames=drops.MAX_IMAGE_FRAMES + 1)) == drops.STACK


def test_a_dropped_image_is_opened_as_a_layer_after_the_tables(app, windows,
                                                                 tmp_path):
    session, control, _ = windows
    opened = []
    control.open_image_path = lambda path: opened.append(
        (path.name, list(session.file_names())))
    from smappy.io.formats import save
    from smappy.simulate import simulate
    table = save(simulate(n_frames=300, seed=3), tmp_path / "a.hdf5")
    image = _write_tif(tmp_path / "wf.tif", frames=1)

    assert drop(app, control, image, table)
    wait_for_loads(app, control)

    # opened once the table was in, so the table did not replace it
    assert opened == [("wf.tif", ["a.hdf5"])]


def test_an_unknown_file_is_refused_in_the_status_bar(app, windows, tmp_path):
    session, control, render = windows
    path = tmp_path / "notes.docx"
    path.write_text("not data")

    assert drop(app, render, path)

    assert control.statusBar().currentMessage() == \
        "cannot open notes.docx: not a localization file or an image stack"
    assert control._task is None and not len(session.locs)


def test_only_local_files_are_taken_from_a_drag():
    from PySide6.QtCore import QMimeData, QUrl
    from smappy.gui import drops
    web = QMimeData()
    web.setUrls([QUrl("https://example.org/data.hdf5")])
    text = QMimeData()
    text.setText("/some/path.hdf5")

    assert drops.accepts(mime_for(__file__))
    assert not drops.accepts(web)
    assert not drops.accepts(text)


def test_fit_this_writes_the_simulation_as_set_and_fills_the_fitters_file(
        app, windows, tmp_path):
    """No Run first, and never a recipe an earlier run left: Fit this writes
    the settings as they are and hands the file to the fitter."""
    from smappy.simulate.source import read_recipe
    session, control, _ = windows
    tab = next(control.tabs.widget(i) for i in range(control.tabs.count())
               if control.tabs.tabText(i) == "File")
    simulate = next(i for i in tab.tab.instances
                    if i.plugin == "File/Simulate/Simulate")
    tab.open_section(simulate.id)
    panel = tab.slots[simulate.id].panel
    path = tmp_path / "run.sim.yaml"
    path.write_text("# left by an earlier run\nsmappy: simulation\nsettings: {n_frames: 5}\n")
    panel.form.set_values({"output": "camera", "n_frames": 20,
                           "camera.path": str(path)})
    panel.refresh_derived()
    assert panel.notice_button.isVisibleTo(panel)
    assert panel.notice_button.text() == "Fit this"

    panel.notice_button.click()                 # the fitter's file, as a drop sets it
    assert read_recipe(path).n_frames == 20                 # these settings
    assert control.tabs.tabText(control.tabs.currentIndex()) == "Localize"
    localize = control.tabs.currentWidget()
    opened = localize.tab.instances[localize.tab.index_of(localize.open_instance())]
    assert localize.slots[opened.id].panel.form.values()["source.path"] == str(path)
    assert not len(session.locs)                            # set, not fitted


def test_fit_this_gives_a_split_camera_to_the_two_colour_fitter_with_its_map(
        app, windows, tmp_path):
    """A simulated split camera goes to Gaussian 2D 2C, whichever fitter is
    open, and the fitter takes the transformation written beside the recipe."""
    from smappy.plugins import Context
    from smappy.plugins.simulate import Simulate
    session, control, _ = windows
    settings = Simulate.Settings(output="camera", n_frames=20)
    settings.camera.split = True
    settings.camera.path = str(tmp_path / "two.sim.yaml")
    made = Simulate().run(Context(), settings)
    assert control.fit_file(made.written)
    localize = control.tabs.currentWidget()
    opened = localize.tab.instances[localize.tab.index_of(localize.open_instance())]
    assert opened.plugin == "Localize/Gaussian 2D 2C"
    values = localize.slots[opened.id].panel.form.values()
    assert values["source.path"] == made.written
    assert values["transform.path"] == str(tmp_path / "two_2ct.h5")


def test_fit_this_gives_a_simulation_drawn_with_a_calibration_to_the_spline_fitter(
        app, windows, tmp_path):
    """Frames drawn with a bead calibration are fitted with it: Spline 3D,
    the calibration filled in, whichever fitter is open."""
    from smappy.plugins.simulate import Simulate
    from smappy.simulate.source import write_recipe
    session, control, _ = windows
    settings = Simulate.Settings(output="camera", n_frames=20)
    settings.optics.calibration = str(tmp_path / "beads_3dcal.mat")
    path = write_recipe(settings, tmp_path / "spline.sim.yaml")
    assert control.fit_file(path)
    localize = control.tabs.currentWidget()
    opened = localize.tab.instances[localize.tab.index_of(localize.open_instance())]
    assert opened.plugin == "Localize/Spline 3D"
    values = localize.slots[opened.id].panel.form.values()
    assert values["source.path"] == str(path)
    assert values["model.calibration"] == str(tmp_path / "beads_3dcal.mat")


def test_fit_this_gives_astigmatic_frames_to_gaussian_2d_elliptical_and_names_astigmatic_z(
        app, windows, tmp_path):
    """Astigmatic frames on one camera: Gaussian 2D with *elliptical* ticked, whichever
    fitter is open, and the status line says what to type into Astigmatic z."""
    from smappy.plugins.simulate import Simulate
    session, control, _ = windows
    settings = Simulate.Settings(output="camera", n_frames=20)
    settings.optics.astigmatism = True
    settings.camera.path = str(tmp_path / "astig.sim.yaml")
    assert control.fit_simulation(settings)
    localize = control.tabs.currentWidget()
    opened = localize.tab.instances[localize.tab.index_of(localize.open_instance())]
    assert opened.plugin == "Localize/Gaussian 2D"
    values = localize.slots[opened.id].panel.form.values()
    assert values["source.path"] == str(tmp_path / "astig.sim.yaml")
    assert values["model.elliptical"] is True
    assert "Astigmatic z" in control.statusBar().currentMessage()


def test_fit_this_gives_a_split_camera_drawn_with_a_dual_calibration_to_spline_3d_2c(
        app, windows, tmp_path):
    from test_simulate_plugin import _dual_calibration
    from smappy.plugins.simulate import Simulate
    from smappy.simulate.source import write_recipe
    session, control, _ = windows
    calibration = _dual_calibration(tmp_path / "dual.h5")
    settings = Simulate.Settings(output="camera", n_frames=20)
    settings.camera.split = True
    settings.optics.calibration = str(calibration)
    path = write_recipe(settings, tmp_path / "split.sim.yaml")
    assert control.fit_file(path)
    localize = control.tabs.currentWidget()
    opened = localize.tab.instances[localize.tab.index_of(localize.open_instance())]
    assert opened.plugin == "Localize/Spline 3D 2C"
    values = localize.slots[opened.id].panel.form.values()
    assert values["model.calibration"] == str(calibration)
