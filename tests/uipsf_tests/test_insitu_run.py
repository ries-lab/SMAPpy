"""In situ in the PSF calibration window: movies to a saved calibration (M5, step 6)."""
import numpy as np
import pytest
import tifffile
import torch
from smappy.calibrate.insitu import InSituSettings
from smappy.calibrate.insitu_run import InSituModelSettings, block_order, insitu_spline, movie_rois
from smappy.calibrate.pupil import BeadSettings, OpticsSettings
from smappy.metadata import CameraMetadata

from test_insitu_rois import FOCUS, OPTICS, model_and_shared

f64 = torch.float64
CAMERA = CameraMetadata(conversion=2.0, offset=100.0, pixelsize_um=(0.127, 0.116), read_noise_e=1.6)
OPTICS_SETTINGS = OpticsSettings(NA=1.43, wavelength_nm=680.0, n_immersion=1.516, n_medium=1.335, pixel_x_nm=127.0,
                                 pixel_y_nm=116.0)


@pytest.fixture(scope="module")
def movie_file(tmp_path_factory):
    """100 molecules 0–1400 nm deep, a quarter on the coverslip, 150 frames of 64 × 64, as a uint16 TIFF."""
    from smappy.simulate import BlinkingSettings

    from uipsf.movie import MovieSettings, simulate_movie
    model, shared = model_and_shared(P=24)
    s = MovieSettings(size_px=(64, 64), n_frames=150, fluorophores=100, heights_nm=(0.0, 1400.0), on_coverslip=0.25,
                      illumination_fwhm_px=100.0, seed=1,
                      blinking=BlinkingSettings(photons=6000.0, photons_std=3000.0, on_time=1.5, blinks=3.0))
    movie = simulate_movie(model, shared, s)
    folder = tmp_path_factory.mktemp("insitu") / "cells_M2"
    folder.mkdir()
    path = folder / "movie.tif"
    tifffile.imwrite(path, np.round(movie.frames / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16))
    return path, movie


SELECTION = InSituSettings(block_frames=150, molecules=63)          # 3 a bin


def small_model(optics, settings, roi_shape, focus_nm, z_max_nm=2000.0, device=None, dtype=torch.float32):
    """`insitu_model` with a pupil of 16 samples on the CPU instead of 64, for seconds rather than a minute."""
    import uipsf
    from uipsf.workflows.calibrate import magnitude_modes
    grid = uipsf.PupilGrid.build(optics, 16, device="cpu", dtype=dtype)
    pupil = uipsf.ZernikePupil(grid, n_max=settings.order, magnitude_modes=magnitude_modes(settings.order,
                                                                                        settings.magnitude))
    psf = uipsf.VectorPSF(grid, pupil, tuple(roi_shape), sigma_nm=(settings.blur_nm,) * 2)
    return uipsf.ForwardModel(psf, geometry=uipsf.model.InSituGeometry(focus_nm),
                              intensity=uipsf.model.Bleaching(order=0))


def test_blocks_are_read_coarse_to_fine_over_the_whole_movie_and_a_short_one_whole():
    assert block_order(150, 200) == [0]
    starts = block_order(10000, 200)
    assert sorted(starts) == list(range(0, 10000, 200))               # every block, once
    # wherever reading stops, it has been spread over the movie: the first ten span it
    assert starts[:2] == [0, 5000] and max(starts[:10]) - min(starts[:10]) > 8000


def test_molecules_are_found_in_a_movie_through_the_camera(movie_file):
    path, movie = movie_file
    rois = movie_rois([path], SELECTION, CAMERA)
    used = rois.ids("used")
    assert len(used) > 25 and rois.sources[0]["read"] == [0]
    # every unit is in the movie's own frames and on a molecule on in that frame
    t, (px, py) = movie.truth, OPTICS.pixel_size_nm
    near = []
    for i in used:
        on = t.frame == rois.units.frames[i][0]
        near.append(np.hypot(t.x_nm[on] / px - rois.units.x_px[i], t.y_nm[on] / py - rois.units.y_px[i]).min() < 3)
    assert np.mean(near) > 0.9 and rois.read_noise_var == pytest.approx(CAMERA.readout_variance)


def test_the_in_situ_geometry_survives_the_model_file(tmp_path):
    import uipsf
    from uipsf.export import build, describe
    model, _ = model_and_shared()
    again = build(describe(model), device="cpu", dtype=f64)
    assert isinstance(again.geometry, uipsf.model.InSituGeometry)
    assert (again.geometry.focus_nm, again.geometry.focus_prior_nm) == (600.0, 500.0)


def test_a_sampled_height_is_the_models_psf_at_that_depth():
    from uipsf.export import sample_heights
    model, shared = model_and_shared()
    h = torch.tensor([0.0, 400.0, 900.0], dtype=f64)
    sampled = sample_heights(model, shared, h, 15)
    pos = torch.zeros(3, 1, 3, dtype=f64)
    pos[..., 2] = -FOCUS
    direct = model.psf(shared, pos, h[:, None])[:, 0]
    assert torch.allclose(sampled.double(), direct, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("z", ["focus", "height"])
def test_the_spline_z_is_the_height_above_focus_or_above_the_coverslip(z):
    model, shared = model_and_shared()
    cal = insitu_spline(model, shared, (100.0, 1300.0), 20.0, 15, z)
    sharpest = cal.parameters["in_focus_height_nm"]
    assert 0 <= cal.z0 <= cal.coeff.shape[1]
    planes = np.arange(cal.psf.shape[0])
    heights = cal.parameters["sampling"]["heights_nm"][1] - planes * 20.0
    reference = 0.0 if z == "height" else sharpest
    assert np.allclose(cal.z_index_to_nm(planes), heights - reference)
    if z == "height":
        assert heights.min() <= 0.0                          # reaches the coverslip, where z0 is
    assert np.isclose(cal.psf.sum(axis=(1, 2)).max(), 1.0)


def test_without_a_start_model_the_astigmatism_typed_in_starts_the_pupil():
    import math

    from smappy.calibrate.insitu_run import start_shared
    model, _ = model_and_shared()
    start, previous = start_shared(model, "", 54.1, 680.0)
    phase = start["pupil.phase"]
    assert previous is None and float(phase[model.psf.pupil.phase_modes.index(6)]) == pytest.approx(
        2 * math.pi * 54.1 / 680.0)
    assert float(phase.abs().sum()) == pytest.approx(float(phase.abs().max()))


def test_a_movie_becomes_a_saved_in_situ_calibration(movie_file, tmp_path, monkeypatch):
    # The plumbing, on a small model: a TIFF through the camera, one round, saved and read back.  How well the
    # rounds learn is test_insitu_rounds', the window's full-size model the benchmark's
    # (benchmarks/uipsf/synthetic/insitu_movie.py).
    from smappy.calibrate.insitu_run import run_insitu
    from smappy.calibrate.pupil import model_path

    import uipsf
    import uipsf.workflows
    import uipsf.workflows.insitu as insitu_module
    monkeypatch.setattr(uipsf.workflows, "insitu_model", small_model)
    monkeypatch.setattr(insitu_module, "HEIGHT_STEP_NM", 300.0)          # 7 starting heights, not 13
    path, _ = movie_file
    result = run_insitu([path], OPTICS_SETTINGS, BeadSettings(zernike_order=2, blur_nm=30.0),
                        InSituModelSettings(focus_nm=500.0, astigmatism_nm=0.4 * 680 / (2 * np.pi), max_rounds=1),
                        SELECTION, camera=CAMERA, save_to=tmp_path / "insitu_3dcal.h5")
    assert result.saved.exists() and model_path(result.saved).exists()
    assert result.calibration.parameters["method"] == "uipsf2_insitu"
    learned = uipsf.load_model(model_path(result.saved), device="cpu")
    assert learned.model.geometry.focus_nm == pytest.approx(result.model.geometry.focus_nm)
    assert float(learned.shared["geometry.focus"]) == pytest.approx(float(result.cal.shared()["geometry.focus"]))


def test_the_focus_from_the_data_moves_a_guess_toward_the_truth_before_any_learning(movie_file, monkeypatch):
    import uipsf.workflows.insitu_rounds as rounds
    from uipsf.noise import ShiftedPoisson
    from uipsf.workflows.insitu_rounds import focus_from_data
    monkeypatch.setattr(rounds, "FOCUS_SEARCH_STEP_NM", 500.0)          # 7 starting heights over 0–3 µm
    path, _ = movie_file
    rois = movie_rois([path], SELECTION, CAMERA)
    o = OPTICS_SETTINGS.optics(CAMERA, 1)[0]
    model = small_model(o, BeadSettings(zernike_order=3, blur_nm=30.0), (15, 15), 1000.0)
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    phase[model.psf.pupil.phase_modes.index(6)] = 0.4
    focus, edge = focus_from_data(model, rois, {"pupil.phase": phase}, ShiftedPoisson(), 1000.0, molecules=30)
    assert abs(focus - FOCUS) < abs(1000.0 - FOCUS) / 2 and np.isfinite(edge.sd_nm)


def test_the_window_offers_in_situ_with_its_own_section_and_finds_molecules_in_a_movie(movie_file, monkeypatch):
    pytest.importorskip("PySide6")
    import time

    import smappy.calibrate.insitu_run as run_module
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    from smappy.gui import make_app
    app = make_app()
    path, _ = movie_file
    original = run_module.movie_rois
    monkeypatch.setattr(run_module, "movie_rois", lambda paths, s, camera=None, progress=None:
                        original(paths, s, CAMERA, progress))
    window = PSFCalibrationWindow([path])
    window.method_box.setCurrentText("in situ (blinking molecules)")
    shown = {k: v.isVisibleTo(window) for k, v in window.sections.items()}
    assert shown["in situ"] and not shown["beads"] and window.detect_button.text() == "Detect molecules"
    assert window.files_section.button.text() == "movies" and not window.mode_box.isEnabled()
    window.optics_form.set_values(dict(NA=1.43, wavelength_nm=680.0, n_immersion=1.516, n_medium=1.335))
    window.insitu_detection_form.set_values(dict(molecules=63))
    window.detect()
    start = time.time()
    while window.busy and time.time() - start < 60:
        app.processEvents()
        time.sleep(0.02)
    states = [window.bead_table.item(r, 0).text() for r in range(window.bead_table.rowCount())]
    assert states[0] == "used" and any(s.startswith("rejected") for s in states)
    assert window.stack_browser.movie and window.stack_browser.positions
    window.close()
