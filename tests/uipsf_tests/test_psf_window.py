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
    assert started.start is not None and started.cal.first is None              # its pupil replaced stage 1


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
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        assert window.pages[i][0].axes, window.tabs.tabText(i)


def test_the_window_shows_each_method_its_own_sections_only(stack_file, monkeypatch):
    window, _, _ = _window(stack_file, monkeypatch, "bead average")
    assert window.sections["microscope"].isHidden() and window.sections["model"].isHidden()
    assert not window.sections["beads"].isHidden() and window.quality_button.isVisibleTo(window)
    window.method_box.setCurrentText("pupil model (uiPSF)")
    assert not window.sections["microscope"].isHidden() and not window.sections["model"].isHidden()
    assert window.quality_button.isVisibleTo(window) and not window.browse_button.isVisibleTo(window)
    assert not window.install_button.isVisibleTo(window)                      # torch is here and recent
    window.close()


@pytest.mark.slow
def test_the_bead_average_calibrates_in_the_window_as_in_the_bead_calibration(stack_file, monkeypatch):
    window, handed, wait = _window(stack_file, monkeypatch, "bead average")
    window.run()
    wait()
    run = window.run_result
    assert run is not None, window.status_bar.currentMessage()
    assert _titles(window) == ["Overview", "Fit quality", "Bead diagnostics"]
    assert run.saved.exists() and handed == [(str(run.saved), False)]
    assert window.bead_table.rowCount() == 10 and window.bead_table.columnCount() == 8
    window.fit_quality()
    wait()
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Fit quality" and "quality" in run.extra
    _every_page_draws(window)
    window.browse_stack()
    assert window.stack_viewer.isVisible()
    window.bead_table.selectRow(2)
    window.toggle_beads()
    window.rebuild()
    wait()
    assert window.run_result.result.reasons[2] == "manual exclusion" and window.run_result.saved == run.saved
    assert window.marks == {2: "exclude"}                                       # kept for the next one
    window.stack_viewer.close()
    window.close()


@pytest.mark.slow
def test_the_pupil_model_learns_in_the_window_shows_its_figures_and_hands_the_calibration_on(
        stack_file, monkeypatch):
    window, handed, wait = _window(stack_file, monkeypatch, "pupil model (uiPSF)")
    window.run()
    wait()
    assert window.run_result is not None, window.status_bar.currentMessage()
    assert {"Pupil", "Zernikes", "Residual atlas", "Beads"} <= set(_titles(window))
    _every_page_draws(window)
    assert window.bead_table.rowCount() == 10
    saved = window.run_result.saved
    assert handed == [(str(saved), False)] and window.use_button.isEnabled()
    window.fit_quality()
    wait()
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Fit quality"
    quality = window.run_result.extra["quality"]
    used = quality["state"] == "used"
    assert np.nanmedian(np.abs(quality["centred_error_nm"][used])) < 15        # the spline fits its own beads
    curves = [line for ax in window.pages[window.tabs.currentIndex()][0].axes for line in ax.get_lines()
              if hasattr(line, "pair_ids")]
    assert len(curves) == 2 * 10                                               # every bead, used or not
    window.pick_pair(type("Pick", (), {"artist": curves[3], "ind": [0]})())    # a click on bead 3's curve
    assert window.selected_beads() == [3]
    window.bead_table.selectRow(0)
    window.toggle_beads()
    assert window.marks == {0: "exclude"} and "exclude on Recalculate" in window.bead_table.item(0, 4).text()
    window.rebuild()
    wait()
    assert window.run_result.result.cal.beads.rows[0].state == "excluded" and window.run_result.saved == saved
    window.close()


def test_the_plugin_opens_the_window_and_refuses_to_run_without_it():
    from smappy.plugins.psf_calibration import PSFCalibration
    assert PSFCalibration.window == "psf calibration" and PSFCalibration.changes_locs is False
    with pytest.raises(ValueError, match="run_calibration"):
        PSFCalibration().run(None, PSFCalibration.Settings())
