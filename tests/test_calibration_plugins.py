"""The bead and PSF calibration plugins run without their windows: from settings, as the assistant's
`run_plugin`, a chain or a script runs them (before 2026-11 both only refused)."""
import importlib.util

import numpy as np
import pytest
import tifffile

from smappy.plugins import Context, get, settings_from


@pytest.fixture(scope="module")
def beads(tmp_path_factory):
    """Two simulated bead z-stacks, 20 nm steps, the pixel in their ImageJ tags."""
    from smappy.simulate.camera import bead_stacks
    folder = tmp_path_factory.mktemp("cal") / "beads"
    folder.mkdir()
    stacks, _ = bead_stacks(n_stacks=2, z_range_nm=(-600.0, 600.0))
    for i, stack in enumerate(stacks):
        tifffile.imwrite(folder / f"beads_{i}.tif", stack, imagej=True, resolution=(10, 10),
                         metadata={"unit": "um", "axes": "ZYX"})
    return folder


def run(path, values):
    cls = get(path)
    return cls().run(Context(), settings_from(cls.Settings, values))


def _drawn(result):
    from matplotlib.figure import Figure
    for name, plot in result.plots.items():
        figure = Figure()
        plot.draw_into(figure)
        assert figure.axes, f"{name} drew nothing"


def test_the_bead_calibration_runs_without_its_window_and_saves_beside_the_stacks(beads):
    from smappy import calibration_list
    result = run("Localize/Bead calibration", {"stacks": str(beads), "beads.dz_nm": 20})
    assert result.data["beads"] >= 4 and 0 < result.data["used"] <= result.data["beads"]
    saved = result.data["saved"]
    assert saved.endswith(".h5") and str(beads.parent) in saved
    assert calibration_list.describe(saved) is not None      # a calibration the fitters read
    assert result.data["fitter"] == "Localize/Spline 3D"
    assert list(result.plots) == ["Overview", "Bead diagnostics", "CRLB"]
    _drawn(result)


def test_save_as_puts_the_bead_calibration_where_it_is_asked(beads, tmp_path):
    result = run("Localize/Bead calibration", {"stacks": str(beads), "beads.dz_nm": 20,
                                                "save_as": str(tmp_path / "mine")})
    assert result.data["saved"] == str(tmp_path / "mine.h5") and (tmp_path / "mine.h5").exists()


def test_a_calibration_without_stacks_says_what_to_give(tmp_path):
    with pytest.raises(ValueError, match="give the stacks"):
        run("Localize/Bead calibration", {})
    with pytest.raises(ValueError, match="no file or folder"):
        run("Localize/PSF calibration", {"stacks": str(tmp_path / "nowhere")})


def test_the_psf_calibrations_bead_average_runs_without_its_window_and_is_listed(beads, tmp_path):
    from smappy import calibration_list
    result = run("Localize/PSF calibration", {"stacks": str(beads), "method": "average",
                                               "beads.dz_nm": 20, "save_as": str(tmp_path / "avg.h5")})
    assert result.data["saved"] == str(tmp_path / "avg.h5") and (tmp_path / "avg.h5").exists()
    assert "beads used" in result.data["summary"] and result.data["fitter"] == "Localize/Spline 3D"
    assert any(e.path == str((tmp_path / "avg.h5").resolve()) for e in calibration_list.entries())
    assert {"Overview", "Model vs data"} <= set(result.plots)
    _drawn(result)


def test_a_pupil_model_without_the_microscopes_optics_is_refused_before_any_work(beads, tmp_path, monkeypatch):
    # a configuration of its own: no microscope another test used last stands in for these stacks
    from smappy import config
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(config, "_cache", {})
    with pytest.raises(ValueError, match="give the microscope's NA"):
        run("Localize/PSF calibration", {"stacks": str(beads), "method": "pupil", "beads.dz_nm": 20})
    with pytest.raises(ValueError, match="has no mode"):
        run("Localize/PSF calibration", {"stacks": str(beads), "method": "insitu", "mode": "Dual colour"})


def test_a_typed_pixel_reaches_the_camera_the_beads_are_cut_with():
    """Typed in the window's *microscope*, the pixel was dropped from the camera: a stack whose file
    and database gave none could not be calibrated with it."""
    from smappy.calibrate.pupil import OpticsSettings
    assert OpticsSettings(pixel_x_nm=100, pixel_y_nm=110).camera_overrides() == {
        "pixelsize_um": 0.1, "pixelsize_y_um": 0.11}
    assert OpticsSettings().camera_overrides() == {}


def test_the_panels_show_the_launcher_alone_and_the_assistant_lists_the_settings():
    from smappy.assistant.tools import Assistant
    for path in ("Localize/Bead calibration", "Localize/PSF calibration"):
        cls = get(path)
        assert cls.form_in_window and cls.window
        described = Assistant().call("describe_plugin", {"path": path})
        assert "stacks" in str(described)


@pytest.mark.long
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="the pupil model needs PyTorch")
def test_the_pupil_model_learns_from_the_beads_without_the_window(beads, tmp_path):
    result = run("Localize/PSF calibration", {
        "stacks": str(beads), "method": "pupil", "beads.dz_nm": 20, "save_as": str(tmp_path / "pupil.h5"),
        "microscope.NA": 1.43, "microscope.n_immersion": 1.518, "microscope.wavelength_nm": 670,
        "microscope.pixel_x_nm": 100, "microscope.conversion": 0.5, "microscope.offset": 100})
    assert (tmp_path / "pupil.h5").exists() and "Zernike" in str(list(result.data))
    assert {"Pupil and Zernikes", "Model vs data"} <= set(result.plots)
    assert np.isfinite(len(result.text))
