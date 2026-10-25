"""The coverslip in SMAPpy's PSF calibration: its index and thicknesses into uipsf's `Optics`, and its tilt
learnt from the window (`docs/uipsf/microscopes.md`, "The coverslip")."""
import math
import time
from dataclasses import replace

import numpy as np
import pytest
import tifffile
import torch
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.pupil import BeadSettings, OpticsSettings, learns_tilt, model_path
from smappy.metadata import CameraMetadata

from uipsf.export import load_model, sample_psf
from uipsf.model import BeadStackGeometry, InSituGeometry
from uipsf.workflows.calibrate import bead_model, insitu_model

CAMERA = CameraMetadata(conversion=2.0, offset=100.0, pixelsize_um=0.11, read_noise_e=1.6)
OIL = OpticsSettings(NA=1.43, wavelength_nm=660.0, n_immersion=1.518)
WATER = OpticsSettings(NA=1.2, wavelength_nm=660.0, n_immersion=1.333, n_coverslip=1.52)
ROI = 17
TILT = (0.012, -0.008)                       # 0.69° and -0.46°


def test_the_coverslip_reaches_uipsfs_optics_with_the_design_thickness_by_default():
    oil, = OIL.optics(CAMERA)
    assert oil.n_cov == oil.n_imm == 1.518 and not oil.coverslip_matters
    water, = WATER.optics(CAMERA)
    assert (water.n_cov, water.d_cov_nm, water.d_cov_design_nm) == (1.52, 170_000.0, 170_000.0)
    assert water.coverslip_matters and not water.layered
    thin, = replace(WATER, coverslip_um=160.0).optics(CAMERA)
    assert thin.d_cov_nm == 160_000.0 and thin.layered


def test_a_bead_run_learns_the_tilt_only_through_a_coverslip_of_another_index():
    tilted = BeadSettings(zernike_order=4, fit_tilt=True)
    assert not learns_tilt(tilted, OIL.optics(CAMERA)).fit_tilt         # beads under oil see none
    kept = learns_tilt(tilted, WATER.optics(CAMERA))
    assert kept.fit_tilt
    model = bead_model(WATER.optics(CAMERA), kept, (ROI, ROI), device="cpu")
    assert isinstance(model.geometry, BeadStackGeometry) and model.geometry.fit_tilt
    assert not bead_model(WATER.optics(CAMERA), BeadSettings(zernike_order=4), (ROI, ROI),
                          device="cpu").geometry.fit_tilt
    situ = insitu_model(OIL.optics(CAMERA)[0], BeadSettings(zernike_order=4), (ROI, ROI), 500.0, device="cpu",
                        fit_tilt=True)
    assert isinstance(situ.geometry, InSituGeometry) and situ.geometry.fit_tilt


@pytest.fixture(scope="module")
def tilted_stack(tmp_path_factory):
    """Ten astigmatic beads through ±600 nm in 50 nm steps under a water objective, the coverslip tilted by
    `TILT`: one uint16 TIFF in a folder of its own."""
    optics = WATER.optics(CAMERA)
    model = bead_model(optics, BeadSettings(zernike_order=4, magnitude="fixed", fit_tilt=True), (ROI, ROI),
                       z_max_nm=1200, device="cpu")
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    phase[:4] = torch.tensor([0.8, -0.2, 0.15, 0.1])                   # astigmatic, for z
    shared = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0]), "geometry.tilt": torch.tensor(TILT),
              "pupil.magnitude": torch.zeros(len(model.psf.pupil.magnitude_modes))}
    rng = np.random.default_rng(0)
    z = -(np.arange(25) * 50.0 - 600.0)
    image = np.full((len(z), 110, 230), 20.0)
    h = ROI // 2
    for i in range(10):
        x, y = 30 + 40 * (i % 5), 30 + 45 * (i // 5)
        offset = np.r_[rng.uniform(-30, 30, 2), rng.uniform(-50, 50)]
        xyz = np.c_[np.tile(offset[:2], (len(z), 1)), z + offset[2]]
        image[:, y - h:y + h + 1, x - h:x + h + 1] += 30000 * sample_psf(model, shared, xyz, roi_px=ROI,
                                                                          upsample=8).double().numpy()
    adu = np.round(rng.poisson(image) / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16)
    folder = tmp_path_factory.mktemp("tilt") / "beads_water"
    folder.mkdir()
    path = folder / "beads.tif"
    tifffile.imwrite(path, adu)
    return path


def _window(path, monkeypatch, method, optics):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.calibrate import pupil
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(pupil, "camera_for", lambda path, settings=None: CAMERA)
    window = PSFCalibrationWindow([str(path)])
    window.method_box.setCurrentText(method)
    window.optics_form.set(optics)
    return app, window


def _shown(form, name):
    return form.row_layouts[name].isRowVisible(form.fields[name])


def test_the_window_offers_the_tilt_in_situ_and_for_beads_only_through_another_glass(tilted_stack, monkeypatch):
    _, window = _window(tilted_stack, monkeypatch, "pupil model (uiPSF)", OIL)
    window._show_optics()
    assert not _shown(window.model_form, "fit_tilt") and not window.optics_form._shown("coverslip_um")
    window.optics_form.set(WATER)
    window._show_optics()
    assert _shown(window.model_form, "fit_tilt") and window.optics_form._shown("coverslip_um")
    window.optics_form.set(OIL)
    window.method_box.setCurrentText("in situ (blinking molecules)")
    assert _shown(window.model_form, "fit_tilt") and not window.optics_form._shown("coverslip_um")
    window.close()


@pytest.mark.long
def test_the_window_learns_the_coverslips_tilt_from_water_beads_and_shows_it(tilted_stack, monkeypatch):
    # The tilted plate's coma through a water objective, apart from the pupil's own up to n = 4 (its secondary
    # coma, n = 5, among what tells them apart): `test_tilt.py` for the model alone.
    app, window = _window(tilted_stack, monkeypatch, "pupil model (uiPSF)", WATER)
    window.model_form.set(BeadSettings(zernike_order=4, fit_tilt=True))
    window.detection_form.set(CalibrationSettings(roi_size=ROI, min_distance_px=20, dz_nm=50.0))
    window.run()
    end = time.time() + 280
    while window.busy and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert not window.busy and window.run_result is not None, window.status_bar.currentMessage()
    cal = window.run_result.result.cal
    from uipsf.diagnostics import coverslip_tilt
    degrees, crlb = coverslip_tilt(cal)
    assert np.all(crlb < 0.01), crlb
    # within 0.05° (measured: 0.024° and 0.008°), not 4 CRLBs of 0.003°: the stack integrates each pixel, the
    # model takes it at its centre (`pupil.UPSAMPLE`), and the learnt blur absorbs only most of that
    assert np.all(np.abs(degrees - np.degrees(TILT)) < 0.05), (degrees, crlb)
    assert "coverslip tilt x" in window.status_bar.currentMessage()
    pupil_page = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "Pupil and Zernikes")
    window.tabs.setCurrentIndex(pupil_page)
    figure = window.pages[pupil_page - 1][0]
    assert "coverslip tilt x" in figure.get_suptitle()
    record = window.run_result.result.calibration.parameters
    assert np.allclose([v for v, _ in record["coverslip_tilt_deg"]], degrees)
    saved = load_model(model_path(window.run_result.saved), device="cpu")
    assert np.allclose(np.degrees(saved.shared["geometry.tilt"].numpy()), degrees, atol=1e-6)
    assert math.isclose(saved.model.channels[0].psf.grid.optics.n_cov, 1.52)
    window.close()
