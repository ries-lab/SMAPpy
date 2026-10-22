"""The PSF calibration: `pupil.run_calibration`, its window and its plugin, on simulated bead stacks."""
import time

import numpy as np
import pytest
import tifffile
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, model_path, run_calibration
from smappy.metadata import CameraMetadata

pytest.importorskip("PySide6")

CAMERA = CameraMetadata(conversion=2.0, offset=100.0, pixelsize_um=0.11, read_noise_e=1.6)
OPTICS = OpticsSettings(NA=1.43, wavelength_nm=660.0, n_immersion=1.518)
DETECTION = CalibrationSettings(roi_size=17, min_distance_px=20, dz_nm=50.0)
MODEL = BeadSettings(zernike_order=4)


@pytest.fixture(scope="module")
def stack_file(tmp_path_factory):
    """Ten astigmatic beads through ±600 nm in 50 nm steps, one uint16 TIFF, in a folder of its own."""

    from uipsf_tests.test_pupil_spline import ROI, _draw, _truth
    model, shared = _truth()
    rng = np.random.default_rng(0)
    z = -(np.arange(25) * 50.0 - 600.0)
    image = np.full((len(z), 110, 230), 20.0)
    h = ROI // 2
    for i in range(10):
        x, y = 30 + 40 * (i % 5), 30 + 45 * (i // 5)
        offset = np.r_[rng.uniform(-30, 30, 2), rng.uniform(-50, 50)]
        image[:, y - h:y + h + 1, x - h:x + h + 1] += _draw(model, shared, np.c_[np.tile(offset[:2], (len(z), 1)),
                                                                                  z + offset[2]], 30000)
    adu = np.round(rng.poisson(image) / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16)
    folder = tmp_path_factory.mktemp("psf") / "beads_M2"
    folder.mkdir()
    path = folder / "beads.tif"
    tifffile.imwrite(path, adu)
    return path


def test_the_microscope_has_no_default_for_its_na_or_wavelength():
    with pytest.raises(ValueError, match="NA and emission"):
        OpticsSettings().optics(CAMERA)
    two = OpticsSettings(NA=1.4, wavelength_nm=600.0, wavelength2_nm=680.0, pixel_x_nm=100.0).optics(CAMERA, 2)
    assert [o.wavelength_nm for o in two] == [600.0, 680.0] and two[0].pixel_size_nm == (100.0, 100.0)


@pytest.mark.slow
def test_a_run_learns_saves_beside_the_stacks_and_recalculates_into_the_same_files(stack_file):
    run = run_calibration([stack_file], False, OPTICS, MODEL, DETECTION, camera=CAMERA)
    assert len(run.cal.beads.rows) == 10 and run.saved.parent == stack_file.parent.parent
    assert run.saved.exists() and model_path(run.saved).exists()
    assert run.calibration.parameters["method"] == "uipsf2_pupil"
    again = run_calibration([stack_file], False, OPTICS, MODEL, DETECTION, previous=run, decisions={0: "exclude"})
    assert again.saved == run.saved and again.cal.beads.rows[0].state == "excluded"
    off = run_calibration([stack_file], False, OPTICS, MODEL, DETECTION, camera=CAMERA,
                          run=RunSettings(save_model=False))
    assert off.saved is None and off.calibration is None
    started = run_calibration([stack_file], False, OPTICS, BeadSettings(zernike_order=6), DETECTION, camera=CAMERA,
                              run=RunSettings(start_from=str(run.saved), save_model=False))
    assert started.start is not None and started.cal.first is not None          # its pupil started stage 1


def _window(stack_file, monkeypatch, method):
    from PySide6.QtWidgets import QApplication
    from smappy.calibrate import pupil
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(pupil, "camera_for", lambda path, settings=None: CAMERA)
    window = PSFCalibrationWindow([str(stack_file)])
    window.method_box.setCurrentText(method)
    window.optics_form.set(OPTICS)
    window.model_form.set(MODEL)
    window.detection_form.set(DETECTION)
    handed = []
    window.calibrated.connect(lambda path, dual: handed.append((path, dual)))

    def wait():
        end = time.time() + 240
        while window.busy and time.time() < end:
            app.processEvents()
            time.sleep(0.02)
        assert not window.busy

    return window, handed, wait


def _titles(window):
    return [window.tabs.tabText(i) for i in range(window.tabs.count())]


def _every_page_draws(window):
    for i in range(1, window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        figure, _, page = window.pages[i - 1]
        if page.table is not None:
            assert window.tabs.widget(i).table.rowCount() > 0, window.tabs.tabText(i)
        else:
            assert figure.axes, window.tabs.tabText(i)


def test_the_window_shows_each_method_its_own_sections_only(stack_file, monkeypatch):
    window, _, _ = _window(stack_file, monkeypatch, "bead average")
    # the camera and the pixel for every method; the optics only for a model
    assert not window.sections["microscope"].isHidden() and window.sections["model"].isHidden()
    assert not window.optics_form._shown("NA") and window.optics_form._shown("pixel_x_nm")
    assert not window.sections["beads"].isHidden() and window.browse_button.isVisibleTo(window)
    window.method_box.setCurrentText("pupil model (uiPSF)")
    assert not window.sections["microscope"].isHidden() and not window.sections["model"].isHidden()
    assert window.optics_form._shown("NA")
    assert window.start_form.isVisibleTo(window)                              # start from: in the model
    assert not window.run_form.fields["start_from"].isVisibleTo(window)
    assert not window.install_button.isVisibleTo(window)                      # torch is here and recent
    assert _titles(window) == ["Stacks"] and window.stack_browser.acquisitions == [str(stack_file)]
    window.close()


def test_detect_beads_rings_them_on_the_stacks_page_and_estimates_the_learning(stack_file, monkeypatch):
    window, _, wait = _window(stack_file, monkeypatch, "pupil model (uiPSF)")
    window.detect()
    wait()
    detected = window.detected
    assert detected is not None, window.status_bar.currentMessage()
    assert len({p[3] for p in detected.positions}) == 10 and 0 < detected.seconds < 600
    from smappy.calibrate.pupil import estimate_seconds
    started = time.perf_counter()
    estimate_seconds(detected.found, OPTICS, MODEL)
    assert time.perf_counter() - started < 1.0                                  # coarse, and quick
    assert "10 beads" in window.found_label.text() and "learning takes about" in window.found_label.text()
    assert window.run_result is None                                          # nothing learnt
    window.stack_browser.draw()
    rings = [p for p in window.stack_browser.figure.axes[0].patches if hasattr(p, "pair_ids")]
    assert len(rings) == 10
    assert window.bead_table.rowCount() == 10 and window.bead_table.item(0, 4).text() == "found"
    window.bead_table.selectRow(4)
    window.toggle_beads()                                                    # left out of the first run
    assert window.marks == {4: "exclude"} and "exclude on Calibrate" in window.bead_table.item(4, 4).text()
    distance = window.detection_form.fields["min_distance_px"].widget          # typed in, as by hand
    distance.setText("22")
    distance.editingFinished.emit()
    assert window.detected is None                                            # a detection setting changed
    window.close()


@pytest.mark.slow
def test_the_bead_average_calibrates_in_the_window_as_in_the_bead_calibration(stack_file, monkeypatch):
    window, handed, wait = _window(stack_file, monkeypatch, "bead average")
    window.run()
    wait()
    run = window.run_result
    assert run is not None, window.status_bar.currentMessage()
    assert _titles(window) == ["Stacks", "Overview", "Model vs data", "Fit quality", "Bead diagnostics"]
    assert run.saved.exists() and handed == [(str(run.saved), False)]
    assert window.bead_table.rowCount() == 10 and window.bead_table.columnCount() == 8
    assert {"quality", "views"} <= set(run.extra)                             # without a button
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Model vs data"
    _every_page_draws(window)
    window.browse_stack()
    assert window.stack_viewer.isVisible()
    viewer = window.stack_viewer
    assert [kind for _, _, kind in viewer.panels()] == ["data", "data"]          # data and model
    viewer.residual_box.setChecked(True)
    assert [kind for _, _, kind in viewer.panels()] == ["data", "data", "residual"]
    assert viewer.bead is None and not viewer.bead_slider.isEnabled()          # the average
    viewer.average_box.setChecked(False)
    viewer.bead_slider.setValue(1)
    assert viewer.bead == sorted(run.extra["views"].beads)[1]
    window.bead_table.selectRow(2)
    window.toggle_beads()
    window.run()                                                              # the same settings: warm
    wait()
    assert window.run_result.result.reasons[2] == "manual exclusion" and window.run_result.saved == run.saved
    assert window.marks == {2: "exclude"}                                       # kept for the next one
    # a setting changed: Calibrate detects again with it, rather than reusing the last run's beads
    distance = window.detection_form.fields["min_distance_px"].widget
    distance.setText("22")
    distance.editingFinished.emit()
    assert window.detected is None
    window.run()
    wait()
    assert window.run_result.extra["inputs"]["detection"].min_distance_px == 22
    assert window.detected is not None and window.marks == {}                  # new beads: the marks went
    window.stack_viewer.close()
    window.close()


@pytest.mark.slow
def test_the_pupil_model_learns_in_the_window_shows_its_figures_and_hands_the_calibration_on(
        stack_file, monkeypatch):
    window, handed, wait = _window(stack_file, monkeypatch, "pupil model (uiPSF)")
    window.run()
    wait()
    assert window.run_result is not None, window.status_bar.currentMessage()
    assert {"Stacks", "Pupil and Zernikes", "Residual atlas", "Zernike table", "Model vs data", "Fit quality",
            "Beads"} <= set(_titles(window))
    assert window.detected is not None                                        # Calibrate detected first
    _every_page_draws(window)
    assert window.bead_table.rowCount() == 10
    saved = window.run_result.saved
    assert handed == [(str(saved), False)] and window.use_button.isEnabled()
    views = window.run_result.extra["views"]
    data, model = views.data[0], views.model[0]
    used = [b for b, state in views.states.items() if state == "used"]
    assert all(np.isfinite(views.beads[b][0][0]).any(axis=(1, 2)).all() for b in used)   # no end plane only some reach
    from smappy.calibrate.pupil import covered_z_nm
    low, high = covered_z_nm(window.run_result.result.cal, window.run_result.result.rois.batch)
    assert low <= views.z_nm.min() and views.z_nm.max() <= high
    assert len(views.z_nm) < 25                                                  # the ±600 nm stack, its ends cut
    assert np.nanmax(np.abs(data - model)) < 0.05 * np.nanmax(data)          # the model fits its beads
    yy, xx = np.indices(model.shape[1:])
    weight = np.nansum(model, 0)
    assert abs((weight * xx).sum() / weight.sum() - (model.shape[2] - 1) / 2) < 0.5   # aligned on the centre
    window.tabs.setCurrentIndex(_titles(window).index("Fit quality"))
    quality = window.run_result.extra["quality"]
    used = quality["state"] == "used"
    assert np.nanmedian(np.abs(quality["centred_error_nm"][used])) < 15        # the spline fits its own beads
    curves = [line for ax in window.pages[window.tabs.currentIndex() - 1][0].axes for line in ax.get_lines()
              if hasattr(line, "pair_ids")]
    assert len(curves) == 2 * 10                                               # every bead, used or not
    window.pick_pair(type("Pick", (), {"artist": curves[3], "ind": [0]})())    # a click on bead 3's curve
    assert window.selected_beads() == [3]
    window.bead_table.selectRow(0)
    window.toggle_beads()
    assert window.marks == {0: "exclude"} and "exclude on Calibrate" in window.bead_table.item(0, 4).text()
    window.run()                                                              # the same settings: warm
    wait()
    assert window.run_result.result.cal.beads.rows[0].state == "excluded" and window.run_result.saved == saved
    assert window.marks == {0: "exclude"}                                       # kept for the next one
    window.bead_table.clearSelection()
    window.bead_table.selectRow(1)
    window.toggle_beads()
    window.run()                                                              # the same settings: warm
    wait()
    rows = window.run_result.result.cal.beads.rows
    assert rows[0].state == rows[1].state == "excluded"                          # the first stays out
    # Uncertainty from halves: on the calibration, its pages, and saved into its model file
    assert not window.halves_button.isHidden() and window.halves_button.isEnabled()
    window.halves(splits=2)
    wait()
    halves = window.run_result.result.cal.halves
    assert halves is not None and halves.n_splits == 2
    used = window.run_result.result.cal.beads.used
    assert all(sorted(a + b) == sorted(used) for a, b in halves.splits)
    table = next(page for _, _, page in window.pages if page.title == "Zernike table")
    columns, rows, *_ = table.table()
    assert "± halves (rad)" in columns and np.isfinite([r[columns.index("± halves (rad)")] for r in rows]).any()
    _every_page_draws(window)
    from uipsf.export import load_model
    assert load_model(model_path(window.run_result.saved), device="cpu").halves.n_splits == 2
    window.close()


def test_the_plugin_opens_the_window_and_refuses_to_run_without_it():
    from smappy.plugins.psf_calibration import PSFCalibration
    assert PSFCalibration.window == "psf calibration" and PSFCalibration.changes_locs is False
    with pytest.raises(ValueError, match="run_calibration"):
        PSFCalibration().run(None, PSFCalibration.Settings())


def test_several_stack_folders_dropped_together_are_each_added(tmp_path):
    from pathlib import Path

    from PySide6.QtWidgets import QApplication
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    QApplication.instance() or QApplication([])
    stacks = []
    for name in ("bead_z1", "bead_z2", "bead_z3"):
        folder = tmp_path / name
        folder.mkdir()
        tifffile.imwrite(folder / f"{name}.tif", np.zeros((5, 16, 16), np.uint16))
        stacks.append(folder)
    window = PSFCalibrationWindow()
    window.add_dropped(stacks[:2])                       # two folders at once, each a stack
    assert len(window.paths) == 2
    assert all(Path(p).parent in stacks[:2] or Path(p) in stacks[:2] for p in window.paths)
    window.subdirectories.setChecked(True)
    window.add_dropped([tmp_path])                       # their parent, searched: the third is added
    assert len(window.paths) == 3
    window.close()


def test_add_files_takes_the_macs_files_and_folders_or_falls_back_to_qts_dialog(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    from smappy.gui import mac_panel
    QApplication.instance() or QApplication([])
    folder = tmp_path / "bead_z1"
    folder.mkdir()
    tifffile.imwrite(folder / "bead_z1.tif", np.zeros((5, 16, 16), np.uint16))
    single = tmp_path / "bead_z2.tif"
    tifffile.imwrite(single, np.zeros((5, 16, 16), np.uint16))
    window = PSFCalibrationWindow()
    monkeypatch.setattr(mac_panel, "choose_files_and_folders", lambda title="", start="": [str(folder), str(single)])
    window.add_files()                                   # a folder and a file in one choice
    assert len(window.paths) == 2
    asked = []
    monkeypatch.setattr(mac_panel, "choose_files_and_folders", lambda title="", start="": None)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *args, **kwargs: asked.append(args) or ([str(tmp_path / "other.tif")], ""))
    window.add_files()                                   # no panel: Qt's dialog, as before
    assert len(asked) == 1 and window.paths[-1].endswith("other.tif")
    window.close()
