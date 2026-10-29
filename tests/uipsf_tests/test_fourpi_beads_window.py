"""*4Pi beads* of the PSF calibration window: a DCIMG bead scan read through the channel layer, its beads
found and registered, the model's export fitted back to it, and the window's own section."""
from pathlib import Path

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
        expected = first if Path(record["path"]).parent.name == "a" else second
        assert listed[record["stack"]] == str(discover_acquisitions([record["path"]])[0])
        assert np.min(np.hypot(*(np.asarray(record["exact_xy"][0]) - expected).T)) < 1.0
    assert {r["stack"] for r in beads.records if Path(r["path"]).parent.name == "a"} == {0}


def test_4pi_scans_added_in_one_channel_switch_the_window_to_4pi_and_are_refused_as_tiffs(tmp_path):
    """Quadrant .mat scans added in the single-channel mode: the window turns to the 4Pi mode, the only one
    that reads them; read as one channel's bead stack they are refused by name, not as a broken TIFF."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.calibrate.input import read_bead_stacks
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    from tests.test_fourpi_channels import _write_quadrant_series
    paths = _write_quadrant_series(tmp_path, [np.zeros((4, 12, 12), np.uint16)] * 4, files=2)
    with pytest.raises(ValueError, match="4Pi scan"):
        read_bead_stacks(paths[0])
    QApplication.instance() or QApplication([])
    window = PSFCalibrationWindow([])
    assert window.mode_box.currentText() == "Single channel"
    window.add_paths([str(tmp_path)])
    assert window.mode_box.currentText() == "4Pi"
    window.close()


def test_the_pupils_page_draws_both_objectives_phase_and_magnitude():
    """The *Pupils* page: each objective's phase and magnitude over the aperture, and the coefficients."""
    import types

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from smappy.calibrate.fourpi_beads import draw_pupils

    import uipsf
    from uipsf.model.fourpi import FourPiModel
    from uipsf.params import Params
    optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
    model = FourPiModel(uipsf.PupilGrid.build(optics, 32, device="cpu"), (13, 13), n_max=4, magnitude_n_max=2)
    batch = uipsf.Batch(data=torch.zeros(1, 4, 9, 13, 13), z_nm=torch.zeros(9), roi_xy=torch.zeros(1, 4, 2))
    start = {"xyz": torch.zeros(1, 3), "photons": torch.ones(1), "background": torch.zeros(1, 4)}
    values = Params(model.specs(batch, start), 1).physical()
    values["pupil.magnitude"] = torch.linspace(-0.1, 0.1, values["pupil.magnitude"].numel())
    shared = {k: v for k, v in values.items() if not k.startswith("emitter.")}
    figure = plt.figure()
    text = draw_pupils(figure, types.SimpleNamespace(model=model, shared=lambda: shared))
    titles = [ax.get_title() for ax in figure.axes]
    assert "upper: magnitude" in titles and "lower: phase (nm)" in titles and "magnitude" in text
    assert "upper + lower(−k)" in titles and "upper − lower(−k)" in titles
    plt.close(figure)


def test_a_lower_pupil_that_is_the_upper_ones_twin_makes_the_same_psf_and_no_mismatch():
    """The pupils page's third column: a lower pupil that is the upper one's twin (phase −(−1)^m c, magnitude
    (−1)^m c) gives both arms the same image at every height, and `arms_mismatch` is zero; the same
    coefficients in both arms are not compensated: astigmatism and spherical add up there, coma does not."""
    import torch
    from smappy.calibrate.fourpi_beads import arms_mismatch

    import uipsf
    from uipsf.model.fourpi import FourPiModel
    from uipsf.optics.zernike import noll_to_nm
    from uipsf.params import Params
    optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
    model = FourPiModel(uipsf.PupilGrid.build(optics, 32, device="cpu"), (13, 13), n_max=4, magnitude_n_max=2,
                        kind="scalar")
    batch = uipsf.Batch(data=torch.zeros(1, 4, 9, 13, 13), z_nm=torch.zeros(9), roi_xy=torch.zeros(1, 4, 2))
    start = {"xyz": torch.zeros(1, 3), "photons": torch.ones(1), "background": torch.zeros(1, 4)}
    p = Params(model.specs(batch, start), 1).physical()
    up = model.pupils[0]
    rng = np.random.default_rng(3)
    for which, sign in (("phase", -1.0), ("magnitude", 1.0)):
        modes = getattr(up, f"{which}_modes")
        c = torch.tensor(rng.normal(0, 0.3 if which == "phase" else 0.05, len(modes)), dtype=p["fourpi.offset"].dtype)
        parity = torch.tensor([(-1.0) ** abs(noll_to_nm(j)[1]) for j in modes], dtype=c.dtype)
        p[f"pupil.{which}"], p[f"lo.pupil.{which}"] = c, sign * parity * c
    p["fourpi.offset"] = torch.zeros(3, dtype=p["fourpi.offset"].dtype)
    mismatch = arms_mismatch(model, p)
    assert np.nanmax(np.abs(mismatch["phase"])) < 1e-9 and np.nanmax(np.abs(mismatch["magnitude"])) < 1e-9
    z = torch.linspace(-600, 600, 7, dtype=p["fourpi.offset"].dtype)
    xyz = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1)[None]
    with torch.no_grad():
        parts = model.images(p, xyz)
    np.testing.assert_allclose(parts[0, 1].numpy(), parts[0, 0].numpy(), atol=1e-6 * float(parts[0, 0].max()))

    p["lo.pupil.phase"] = p["pupil.phase"].clone()
    c = arms_mismatch(model, p)["phase_coefficients"]
    for j, v in c.items():
        expected = 2 * float(p["pupil.phase"][up.phase_modes.index(j)]) if noll_to_nm(j)[1] % 2 == 0 else 0.0
        assert v == pytest.approx(expected, abs=1e-12)


def test_the_beads_own_fringe_period_is_measured_without_any_optics(tmp_path):
    """Beads drawn with a 260 nm fringe give 260 nm back from their scan alone (`beads_period_nm`)."""
    from smappy.calibrate.fourpi_beads import beads_period_nm
    cal = fourpi.model_calibration(period_nm=260.0)
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1])
    xy = np.array([[12.0, 12.0], [31.0, 14.0], [14.0, 31.0], [30.0, 30.0]])
    path, _ = _dcimg_scan(tmp_path / "beads", cal, layout, xy)
    beads, _, _ = scan_beads([path], _settings(), CAMERA)
    assert beads_period_nm(beads) == pytest.approx(260.0, rel=0.02)


def test_a_pupil_model_whose_fringe_is_not_its_beads_is_said_to_have_the_wrong_optics(tmp_path):
    """The check after a pupil model's learning: within 8 % nothing is said -- uiPSF's 4Pi example learnt
    with its own optics came to 264.1 nm against its beads' 251; uiPSF's 4Pi example learnt
    with the 4Pi-SMS optics (279.5 nm against its beads' 251) names the period, the beads' and the emission,
    and the 4Pi methods' finish puts it where the window shows it (`check_period`) (`Run.extra["warning"]`)."""
    from smappy.calibrate.fourpi_beads import period_mismatch
    from smappy.calibrate.psf_methods import FourPiPupil, Run
    assert period_mismatch(255.0, 251.0) is None
    assert period_mismatch(264.1, 251.0) is None
    said = period_mismatch(279.5, 251.0, 660.0)
    assert "279.5 nm" in said and "11% longer" in said and "251.0 nm" in said and "660 nm" in said
    assert "shorter" in period_mismatch(230.0, 251.0)
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1])
    xy = np.array([[12.0, 12.0], [31.0, 14.0], [14.0, 31.0], [30.0, 30.0]])
    path, _ = _dcimg_scan(tmp_path / "beads", fourpi.model_calibration(period_nm=260.0), layout, xy)
    beads, found, _ = scan_beads([path], _settings(), CAMERA)
    # a model as a pupil model with the wrong emission would come out: its fringe 12 % long
    run = Run({"fit": object(), "found": {"beads": beads, "layout": found}, "excluded": [],
               "settings": _settings(), "wavelength_nm": 660.0,
               "calibration": fourpi.model_calibration(period_nm=291.0)}, False)
    assert "12% longer" in FourPiPupil().check_period(run) == run.extra["warning"]
    run.result["calibration"] = fourpi.model_calibration(period_nm=262.0)
    assert FourPiPupil().check_period(run) is None
