"""*4Pi beads* of the PSF calibration window: a DCIMG bead scan read through the channel layer, its beads
found and registered, the model's export fitted back to it, and the window's own section."""
import numpy as np
import pytest
from smappy import fourpi
from smappy.calibrate.fourpi_beads import FourPiBeadSettings, bead_fits, scan_beads, scan_step_nm
from smappy.channels import from_corners
from smappy.metadata import CameraMetadata

CAMERA = CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1)


def _dcimg_scan(folder, cal, layout, beads_xy, step_nm=20.0, n=31, seed=0):
    """One DCIMG file per slice, the four channels on one chip, a ``Cell01.txt`` with the step, as the
    4Pi-SMS example's bead scans are written."""
    from tests.test_fourpi import _write_new_dcimg
    z = -(np.arange(n) - (n - 1) // 2) * step_nm
    frame = np.repeat(np.arange(n), len(beads_xy))
    frames = fourpi.render_frames(cal, layout, frame, np.tile(beads_xy[:, 0], n), np.tile(beads_xy[:, 1], n),
                                  np.repeat(z, len(beads_xy)), 30000, phase0=0.4, n_frames=n, background=5,
                                  seed=seed, frame_shape=(48, 200))
    adu = (frames / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16)
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        _write_new_dcimg(folder / f"Cell01_642_000_{k:03d}.dcimg", adu[k:k + 1], adu[k:k + 1, 23, :4])
    (folder / "Cell01.txt").write_text(f"10/22/2018\n# of frames: 1\n Step Size (um): {step_nm / 1000:.6f}\n"
                                       f"# of Steps: {n}\n", encoding="utf-8")
    return folder / "Cell01_642_000_000.dcimg", z


def _settings():
    return FourPiBeadSettings(layout="typed", x_corners="4, 54, 104, 154", y_corners="2, 2, 2, 2",
                              width=44, height=44, mirrored="3, 4", roi_size=13)


def test_a_dcimg_bead_scan_gives_its_beads_its_registration_and_its_z_step(tmp_path):
    cal = fourpi.model_calibration()
    shift = np.eye(3)
    shift[:2, 2] = [0.6, -0.3]
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1]).with_maps(
        [np.eye(3), shift, np.eye(3), shift])
    xy = np.array([[12.0, 12.0], [31.0, 14.0], [14.0, 31.0], [30.0, 30.0]])
    path, z = _dcimg_scan(tmp_path / "beads", cal, layout, xy)
    assert scan_step_nm(path) == pytest.approx(20.0)
    beads, found, registration = scan_beads([path], _settings(), CAMERA)
    assert beads.data.shape[:4] == (4, 4, len(z), 1)
    assert np.allclose(beads.z_nm, z)
    assert registration.kind[1] == "affine"
    assert np.allclose(found.to_channel(1, [[22.0, 22.0]]), layout.to_channel(1, [[22.0, 22.0]]), atol=0.1)


def test_beads_fitted_back_with_their_own_model_follow_the_stage(tmp_path):
    """The model that drew the beads, fitted back to them: z on the stage, slope one."""
    cal = fourpi.model_calibration()
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1])
    xy = np.array([[12.0, 12.0], [31.0, 14.0], [14.0, 31.0], [30.0, 30.0]])
    path, _ = _dcimg_scan(tmp_path / "beads", cal, layout, xy)
    beads, _, _ = scan_beads([path], _settings(), CAMERA)
    fits = bead_fits(beads, cal, box=13)
    assert all(abs(s - 1) < 0.05 for s in fits["slope"].values())
    assert all(abs(s - 1) < 0.02 for s in fits["phase_slope"].values())


def test_4pi_is_a_mode_whose_methods_are_the_pupil_model_and_the_bead_average():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    QApplication.instance() or QApplication([])
    window = PSFCalibrationWindow([])
    window.show()
    window.mode_box.setCurrentText("4Pi")
    assert window.method.key == "fourpi_average"                      # the bead average, in 4Pi
    assert not window.sections["4Pi"].isHidden() and window.sections["beads"].isHidden()
    assert window.use_button.text() == "Use in Spline 4Pi"
    window.method_box.setCurrentText("pupil model (uiPSF)")
    assert window.method.key == "fourpi_pupil"
    assert not window.fourpi_form.fields["x_corners"].isVisibleTo(window)
    window.fourpi_form.set(FourPiBeadSettings(layout="typed"))
    window._show_fourpi()
    assert window.fourpi_form.fields["x_corners"].isVisibleTo(window)
    # in situ has no 4Pi mode yet: choosing it goes back to one channel
    window.method_box.setCurrentText("in situ (blinking molecules)")
    assert window.mode_box.currentText() == "Single channel" and window.sections["4Pi"].isHidden()
    # and the 4Pi mode, chosen with in situ, takes the pupil model
    window.mode_box.setCurrentText("4Pi")
    assert window.method.key == "fourpi_pupil"
    assert window.method_box.currentText() == "pupil model (uiPSF)"
    window.close()


def test_the_4pi_bead_average_gives_a_model_that_fits_its_beads(tmp_path):
    """The model-free 4Pi model from a DCIMG scan of one phase per slice, fitted back to the beads."""
    from smappy.calibrate.fourpi_beads import average_model
    cal = fourpi.model_calibration()
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1])
    xy = np.array([[12.3, 12.6], [31.4, 14.2], [14.7, 31.1], [30.2, 29.8]])
    path, _ = _dcimg_scan(tmp_path / "beads", cal, layout, xy, n=41)
    beads, found, registration = scan_beads([path], _settings(), CAMERA)
    model = average_model({"beads": beads, "layout": found, "registration": registration})
    assert model.period_nm == pytest.approx(cal.period_nm, rel=0.02)
    fits = bead_fits(beads, model, box=13)
    assert all(abs(s - 1) < 0.1 for s in fits["slope"].values())


def test_a_beads_stack_is_its_scan_in_the_order_the_stacks_page_lists_them(tmp_path):
    """Scans added out of sorted order: each bead's ``stack`` is still its file's place on the Stacks page
    (`input.discover_acquisitions`), so its ring falls on its own scan.  In the order added, the 4Pi
    example's bead12_ beads were ringed on bead13_."""
    from smappy.calibrate.input import discover_acquisitions
    cal = fourpi.model_calibration()
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1])
    first = np.array([[12.0, 12.0], [31.0, 30.0]])
    second = np.array([[30.0, 13.0], [13.0, 31.0]])
    a, _ = _dcimg_scan(tmp_path / "a", cal, layout, first, n=11)
    b, _ = _dcimg_scan(tmp_path / "b", cal, layout, second, n=11, seed=1)
    beads, _, _ = scan_beads([b, a], _settings(), CAMERA)
    listed = [str(p) for p in discover_acquisitions([b, a])]
    for record in beads.records:
        expected = first if "/a/" in record["path"] else second
        assert listed[record["stack"]] == str(discover_acquisitions([record["path"]])[0])
        assert np.min(np.hypot(*(np.asarray(record["exact_xy"][0]) - expected).T)) < 1.0
    assert {r["stack"] for r in beads.records if "/a/" in r["path"]} == {0}
