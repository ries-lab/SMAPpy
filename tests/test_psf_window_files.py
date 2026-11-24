"""What the PSF calibration window does with a file of the wrong kind: says so at once, offers the method."""
import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from smappy.calibrate import input as calibrate_input       # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _tiff(path, frames):
    with tifffile.TiffWriter(path) as w:
        for _ in range(frames):
            w.write(np.full((8, 8), 100, np.uint16), contiguous=False)
    return str(path)


def test_micro_managers_summary_says_movie_or_z_stack(tmp_path, monkeypatch):
    from smappy.io import tiff
    path = _tiff(tmp_path / "a_MMStack_Pos0.ome.tif", 3)
    opened = tiff.open_stack(path)
    for summary, kind in (({"Slices": 101, "Frames": 1}, "stack"), ({"Slices": 1, "Frames": 100000}, "movie"),
                          ({}, None)):
        opened.summary = summary
        monkeypatch.setattr(tiff, "open_stack", lambda p, s=opened: s)
        assert calibrate_input.acquisition_kind(path)[0] == kind
    # no summary: a long one is a movie
    monkeypatch.setattr(calibrate_input, "MOVIE_FRAMES", 2)
    assert calibrate_input.acquisition_kind(path) == ("movie", 3)


@pytest.mark.parametrize("answer", ["switch", "remove", "keep"])
def test_a_movie_added_for_a_bead_method_is_said_and_in_situ_offered(app, tmp_path, monkeypatch, answer):
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    monkeypatch.setattr(calibrate_input, "MOVIE_FRAMES", 50)
    movie = _tiff(tmp_path / "movie_MMStack_Pos0.ome.tif", 60)
    window = PSFCalibrationWindow()
    window.method_box.setCurrentText("bead average")     # a bead method that is not the default
    width = window.width()
    asked = []

    def click(box):
        asked.append(box.text())
        pick = {"switch": "Use blinking molecules", "remove": "Remove", "keep": "Keep"}[answer]
        box.buttons()                       # the box as shown
        next(b for b in box.buttons() if b.text().startswith(pick)).click()
    monkeypatch.setattr(window, "_ask", click)
    window.add_paths([movie])
    assert len(asked) == 1 and "blinking movie" in asked[0] and "bead average" in asked[0]
    if answer == "switch":
        assert window.method.key == "insitu" and window.paths == [movie]
    elif answer == "remove":
        assert window.method.key == "average" and window.paths == []
    else:
        # kept under the bead method: never read whole, said in a line that does not widen the window
        assert window.paths == [movie]
        assert "not a bead z-stack" in window.stack_browser.title.text()
        assert window.width() == width
    window.close()


def test_a_bead_stack_added_for_a_bead_method_asks_nothing(app, tmp_path, monkeypatch):
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    window = PSFCalibrationWindow()
    monkeypatch.setattr(window, "_ask", lambda box: pytest.fail(box.text()))
    window.add_paths([_tiff(tmp_path / "beads_MMStack_Pos0.ome.tif", 21)])
    window.close()


def test_the_movie_browser_steps_through_every_frame_then_the_frames_with_molecules_only(app, tmp_path):
    from smappy.calibrate.stack_browser import StackBrowser
    movie = _tiff(tmp_path / "movie_MMStack_Pos0.ome.tif", 40)
    browser = StackBrowser()
    browser.set_movie(True)
    browser.set_paths([movie])
    # before detection: every frame, to look through the file before anything is found in it
    assert browser.plane_slider.isEnabled() and browser.plane_slider.maximum() == 39
    browser.plane_slider.setValue(5)
    assert browser.plane_label.text() == "frame 6 of 40"
    browser.set_marks([(0, 3.0, 3.0, 0, 7), (0, 4.0, 4.0, 1, 7), (0, 2.0, 5.0, 2, 31)])
    assert browser.plane_slider.isEnabled() and browser.plane_slider.maximum() == 1
    assert browser.plane_label.text() == "1/2: frame 8 of 40" and "2 molecules ringed" in browser.title.text()
    browser.plane_slider.setValue(1)
    assert browser.plane_label.text() == "2/2: frame 32 of 40"


def test_a_stack_loaded_first_can_be_scrolled_after_choosing_molecules(app, tmp_path):
    """Files first, then what they are: the Stacks page scrolls whichever sample is chosen."""
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    from smappy.calibrate.psf_methods import SAMPLES
    window = PSFCalibrationWindow()
    window._ask = lambda box: None
    window.add_paths([_tiff(tmp_path / "sweep_MMStack_Pos0.ome.tif", 21)])
    molecules = next(s for s in SAMPLES if "molecule" in s)
    window.sample_box.setCurrentText(molecules)
    slider = window.stack_browser.plane_slider
    assert slider.isEnabled() and slider.maximum() == 20
    window.sample_box.setCurrentText(SAMPLES[0])
    assert slider.isEnabled() and slider.maximum() == 20
    window.close()


def test_a_z_sweep_shows_each_frames_stage_z_and_sweep_from_its_metadata(app, tmp_path):
    import json
    from smappy.calibrate.molecule_stack import frame_z_um
    from smappy.calibrate.stack_browser import StackBrowser
    # two sweeps up of five 0.1 µm steps, as Micro-Manager tags each page
    z = [1.0 + 0.1 * (i % 5) for i in range(10)]
    mm = tmp_path / "sweeps_MMStack_Pos0.ome.tif"
    with tifffile.TiffWriter(mm) as w:
        for zi in z:
            md = json.dumps({"ZPositionUm": zi})
            w.write(np.full((8, 8), 100, np.uint16), contiguous=False,
                    extratags=[(51123, "s", len(md) + 1, md, False)])
    np.testing.assert_allclose(frame_z_um(str(mm)), z)
    # and as an OME-TIFF's planes
    ome = tmp_path / "sweeps.ome.tif"
    tifffile.imwrite(ome, np.full((10, 1, 1, 8, 8), 100, np.uint16), ome=True,
                     metadata={"axes": "TZCYX", "Plane": {"PositionZ": z, "PositionZUnit": ["µm"] * 10}})
    np.testing.assert_allclose(frame_z_um(str(ome)), z)
    browser = StackBrowser()
    browser.set_movie(True, sweeps=True)
    browser.set_paths([str(mm)])
    browser.plane_slider.setValue(6)
    assert browser.plane_label.text() == "frame 7 of 10, sweep 2, stage z 1.100 µm"
    browser.set_movie(True)                       # in situ's movies: no z looked for
    assert browser.plane_label.text() == "frame 7 of 10"


def test_delete_removes_the_selected_file_and_selects_the_next_and_clear_empties_the_list(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    window = PSFCalibrationWindow()
    window.paths[:] = ["a.tif", "b.tif", "c.tif"]
    window._refresh_files()
    window.show()
    window.activateWindow()
    window.file_list.setFocus()
    app.processEvents()
    window.file_list.setCurrentRow(1)
    QTest.keyClick(window.file_list, Qt.Key_Delete)
    assert window.paths == ["a.tif", "c.tif"]
    assert window.file_list.currentItem().text() == "c.tif"
    QTest.keyClick(window.file_list, Qt.Key_Backspace)      # at the end: the last one
    assert window.paths == ["a.tif"] and window.file_list.currentItem().text() == "a.tif"
    QTest.keyClick(window.file_list, Qt.Key_Delete)
    assert window.paths == [] and window.file_list.count() == 0
    window.paths[:] = ["a.tif", "b.tif"]
    window._refresh_files()
    window.clear_paths()
    assert window.paths == [] and window.file_list.count() == 0
    window.close()


def test_every_job_of_the_window_hands_back_what_the_allocators_kept(monkeypatch):
    """A calibration left 3-4 GB in Apple's GPU cache, and the next run swapped: the worker releases after each
    job, whether it ended or failed."""
    from smappy.calibrate import psf_window
    released, done = [], []
    monkeypatch.setattr(psf_window, "release_memory", lambda: released.append(1))
    for function in (lambda progress: 42, lambda progress: 1 / 0):
        worker = psf_window._Worker("run", function)
        worker.done.connect(lambda kind, value: done.append(kind))
        worker.run()
    assert released == [1, 1] and done == ["run", "error"]


def test_the_stacks_own_z_steps_are_read_from_each_acquisition_and_files_without_one_add_nothing(monkeypatch):
    from smappy.calibrate import psf_methods
    from smappy.calibrate.input import BeadStack
    stacks = {"a": [BeadStack(np.zeros((4, 2, 2)), np.arange(4) * 50.0)],
              "b": [BeadStack(np.zeros((4, 2, 2)), np.arange(4) * 50.04),
                    BeadStack(np.zeros((4, 2, 2)), np.arange(4) * 20.0)]}

    def read(path, dz_nm=None):
        if path == "c":
            raise ValueError("c: no usable z spacing; supply dz_nm explicitly")
        return stacks[path]
    monkeypatch.setattr(calibrate_input, "discover_acquisitions", lambda paths: list(paths))
    monkeypatch.setattr(calibrate_input, "read_bead_stacks", read)
    monkeypatch.setattr(calibrate_input, "is_fourpi_scan", lambda path: False)
    assert psf_methods.file_z_steps(["a", "b", "c"]) == (20.0, 50.0)


def test_z_step_on_auto_says_the_detected_stacks_step_and_forgets_it_with_the_detection(app):
    from types import SimpleNamespace
    from smappy.calibrate.psf_methods import Detected
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    window = PSFCalibrationWindow()
    window.method_box.setCurrentText("bead average")
    assert window._steps_wanted()
    field = window.detection_form.fields["dz_nm"]
    window._show_detected(Detected(found=SimpleNamespace(records=[], sources=[]), text="3 beads", z_steps_nm=(50.0,)))
    assert field.widget.text() == "50" and window.detection_form.value().dz_nm is None
    window._undetect("the files changed")
    assert field.widget.text() == ""
    window.close()


def test_the_buttons_pass_no_clicked_flag_as_an_argument(app, monkeypatch):
    """*Open calibration...* took clicked's False as its path: "argument should be a str ... not 'bool'"."""
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    window = PSFCalibrationWindow()
    calls = []
    monkeypatch.setattr(window, "open_calibration", lambda *a, **k: calls.append(("open", a, k)))
    monkeypatch.setattr(window, "halves", lambda *a, **k: calls.append(("halves", a, k)))
    window.open_button.setEnabled(True)
    window.halves_button.setEnabled(True)
    window.open_button.click()
    window.halves_button.click()
    assert calls == [("open", (), {}), ("halves", (), {})]
    window.close()


def test_same_contrast_in_every_plane_scales_every_plane_by_the_stacks_brightest(app, monkeypatch):
    from smappy.calibrate.input import BeadStack
    from smappy.calibrate.stack_browser import StackBrowser
    browser = StackBrowser()
    images = np.stack([np.full((16, 16), 100.0) for _ in range(5)])
    for z, peak in enumerate((200, 600, 1000, 600, 200)):
        images[z, 8, 8] = peak
    browser.acquisitions = ["a.tif"]
    browser._cache["a.tif"] = [BeadStack(images, np.arange(5) * 50.0)]
    assert not browser.global_box.isChecked()          # off by default
    browser.draw()                                     # the first draw puts the slider in the middle
    clims = []
    for ticked in (False, True):
        browser.global_box.setChecked(ticked)
        browser.plane_slider.setValue(0)
        browser.draw()
        clims.append(browser.figure.axes[0].images[0].get_clim()[1])
    alone, together = clims
    assert alone < 200 and together > 800              # the dim plane on the in-focus plane's range
