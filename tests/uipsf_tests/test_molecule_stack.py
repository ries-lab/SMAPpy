"""Molecules on the coverslip through z sweeps: reading, the width curve, detection and selection
(`smappy.calibrate.molecule_stack`), and a whole run that recovers the pupil it was simulated with
(`molecule_stack_run`)."""
import numpy as np
import pytest
from smappy.calibrate.molecule_stack import MoleculeStackSettings, molecule_rois, read_file, split_sweeps

import uipsf
from uipsf_tests.molecule_sweeps import CAMERA, PHASE_NM, write_sweeps

OPTICS = uipsf.Optics(NA=1.45, wavelength_nm=680.0, n_imm=1.518, n_med=1.334, pixel_size_nm=(110.0, 110.0))


def test_sweeps_are_split_where_the_stage_jumps_back_or_turns():
    up = np.tile(np.arange(5) * 0.05, 3)
    assert split_sweeps(up).tolist() == [0] * 5 + [1] * 5 + [2] * 5
    up_down = np.r_[np.arange(5), np.arange(5)[::-1][1:], np.arange(5)[1:]] * 0.05
    assert split_sweeps(up_down).tolist() == [0] * 5 + [1] * 4 + [2] * 4


def test_the_stage_z_of_each_frame_is_read_from_an_ome_tiffs_planes(tmp_path):
    truth = write_sweeps(tmp_path, OPTICS, files=1, sweeps=2, per_frame=1, step_nm=200.0, focus_um=(40.0,))
    f = read_file(truth["paths"][0], CAMERA)
    assert f.frames.shape == (22, 128, 128)
    assert np.allclose(f.z_um[:11], 40.0 + np.arange(-1.0, 1.01, 0.2))
    assert f.sweep.tolist() == [0] * 11 + [1] * 11


def test_a_file_without_a_stage_z_needs_the_step_typed(tmp_path):
    import tifffile
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.full((6, 32, 32), 100, np.uint16))
    with pytest.raises(ValueError, match="z step"):
        read_file(path, CAMERA)
    f = read_file(path, CAMERA, MoleculeStackSettings(z_step_nm=50.0, sweep_frames=3))
    assert np.allclose(f.z_um, [0, 0.05, 0.1, 0, 0.05, 0.1]) and f.sweep.tolist() == [0, 0, 0, 1, 1, 1]


def test_the_molecules_are_found_at_each_files_focus_and_drawn_evenly_over_the_defocus(tmp_path):
    truth = write_sweeps(tmp_path, OPTICS, files=2, sweeps=2, focus_um=(50.0, 50.3))
    rois = molecule_rois(truth["paths"], MoleculeStackSettings(molecules=150), CAMERA)
    # the files' foci 300 nm apart, their difference found within tens of nm: a start for the sweeps' foci,
    # under a 300 nm prior (the width curve of 84 frames a file: 2-47 nm over three seeds)
    true = 0.3 + (truth["foci_nm"][2:].mean() - truth["foci_nm"][:2].mean()) / 1000
    assert abs(np.diff(rois.curve.focus_um)[0] - true) < 0.06
    assert rois.curve.shape == "astigmatic" and rois.sweeps == 4
    used = rois.ids("used")
    assert len(used) == 150
    counts = np.histogram(rois.defocus_nm[used], np.arange(-1050, 1051, 100))[0]
    # evenly over the defocus the molecules pass at: at the far end the curve's 15-pixel fits lose the spots
    filled = counts[counts > 0]
    inner = filled[1:-1]                          # the outermost bins only partly inside the sweeps' span
    assert len(filled) >= 14 and inner.max() - inner.min() <= 2
    assert len(rois.ids("not drawn")) > 0
    batch, _ = rois.batch()
    assert tuple(batch.data.shape[1:]) == (1, 1, 25, 25) and batch.stack_id.max() == 3
    assert np.allclose(batch.z_nm[:, 0].numpy(), -rois.defocus_nm[used], atol=1e-3)


@pytest.mark.slow
def test_a_run_recovers_the_pupil_the_height_and_the_sweeps_foci_and_saves_a_spline(tmp_path):
    from smappy.calibrate.molecule_stack_run import run_molecule_stack
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, PSFModelSettings, RunSettings
    from smappy.io.calibration import load_spline_calibration
    truth = write_sweeps(tmp_path, OPTICS, files=1, sweeps=3, per_frame=4, step_nm=100.0, span_nm=700.0,
                         focus_um=(50.0,))
    optics = OpticsSettings(NA=1.45, wavelength_nm=680.0, n_immersion=1.518, n_medium=1.334, pixel_x_nm=110.0,
                            pixel_y_nm=110.0)
    model = BeadSettings(psf=PSFModelSettings(zernike_order=4))
    out = tmp_path / "cal.h5"
    result = run_molecule_stack(truth["paths"], optics, model,
                                MoleculeStackSettings(molecules=150, roi_px=17, max_rounds=2),
                                run=RunSettings(save_model=True), camera=CAMERA, save_to=out, device="cpu")
    shared = result.cal.shared()
    pupil = result.cal.model.psf.pupil
    nm = {int(j): float(v) * OPTICS.wavelength_nm / (2 * np.pi)
          for j, v in zip(pupil.phase_modes, shared["pupil.phase"].detach().cpu(), strict=True)}
    for j, v in PHASE_NM.items():
        assert abs(nm[j] - v) < 4.0, (j, nm[j], v)
    assert abs(float(shared["pupil.gamma"]) - truth["gamma"]) < 0.15
    # the height is told by the SAF's brightness alone, ±9 nm from 150 molecules: within three of those
    sd = float(result.cal.result.std["geometry.height"])
    assert abs(result.height_nm() - truth["height_nm"]) < 3 * sd and sd < 15.0
    foci = shared["geometry.focus"].detach().cpu().double().numpy()
    true = truth["foci_nm"]
    assert np.abs((foci - foci.mean()) - (true - true.mean())).max() < 10.0
    # the bend of 150 molecules is their medians' noise: 7.4 nm held at the true height, 7.8 learnt, of 11.6 nm
    assert result.check.scatter < 1.6 and result.check.bend_rms(600) < 0.75 * np.median(result.check.crlb_nm)
    cal = load_spline_calibration(result.saved)
    assert cal.parameters["sample"] == "molecules on the coverslip in a z stack"
    assert abs(cal.parameters["height_nm"] - result.height_nm()) < 1e-6
    assert (result.saved.parent / "cal.uipsf.h5").exists()


def test_the_window_offers_molecule_stacks_as_a_sample_and_detects_their_molecules(tmp_path, monkeypatch):
    import time

    from smappy.calibrate import molecule_stack as module
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    from smappy.gui import make_app
    app = make_app()
    truth = write_sweeps(tmp_path, OPTICS, files=1, sweeps=2, per_frame=4, focus_um=(50.0,))
    original = module.molecule_rois
    monkeypatch.setattr(module, "molecule_rois", lambda paths, s, camera=None, progress=None:
                        original(paths, s, CAMERA, progress))
    window = PSFCalibrationWindow(truth["paths"])
    window.sample_box.setCurrentText("molecules on the coverslip in a z stack")
    methods = [window.method_box.itemText(i) for i in range(window.method_box.count())]
    assert methods == ["pupil model (uiPSF)", "elliptical Gaussian"]
    assert [window.mode_box.itemText(i) for i in range(window.mode_box.count())] == ["Single channel", "Dual colour"]
    shown = {k: v.isVisibleTo(window) for k, v in window.sections.items()}
    assert shown["molecules"] and shown["model"] and not shown["beads"] and not shown["in situ"]
    assert window.detect_button.text() == "Detect molecules"
    form = window.model_form
    assert not form.row_layouts["bead_diameter_nm"].isRowVisible(form.fields["bead_diameter_nm"])
    window.optics_form.set_values(dict(NA=1.45, wavelength_nm=680.0, n_immersion=1.518, n_medium=1.334,
                                       pixel_x_nm=110.0, pixel_y_nm=110.0))
    window.molecule_form.set_values(dict(molecules=60))
    window.detect()
    start = time.time()
    while window.busy and time.time() - start < 60:
        app.processEvents()
        time.sleep(0.02)
    states = [window.bead_table.item(r, 0).text() for r in range(window.bead_table.rowCount())]
    assert states[0] == "used" and "learning takes about" in window.found_label.text()
    assert window.stack_browser.movie and window.stack_browser.positions
    window.close()


@pytest.mark.slow
def test_dual_colour_pairs_the_halves_molecules_and_learns_both_channels_with_their_map(tmp_path):
    from smappy.calibrate.dual import load_dual_color_calibration
    from smappy.calibrate.molecule_dual import dual_molecule_rois
    from smappy.calibrate.molecule_stack_run import run_molecule_stack
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, PSFModelSettings, RunSettings

    from uipsf_tests.molecule_sweeps import DUAL_SHIFT_PX, write_dual_sweeps
    truth = write_dual_sweeps(tmp_path, OPTICS)
    settings = MoleculeStackSettings(molecules=50, roi_px=15, max_rounds=1)
    rois = dual_molecule_rois(truth["paths"], settings, CAMERA, transform=truth["transform"])
    used = rois.ids("used")
    assert rois.n_channels == 2 and len(used) == 50
    # each pair's secondary region where the map puts its main one
    gap = rois.roi_xy[used, 1] - rois.roi_xy[used, 0]
    assert np.allclose(np.median(gap, 0), (128 + DUAL_SHIFT_PX[0], DUAL_SHIFT_PX[1]), atol=1)
    optics = OpticsSettings(NA=1.45, wavelength_nm=680.0, wavelength2_nm=680.0, n_immersion=1.518, n_medium=1.334,
                            pixel_x_nm=110.0, pixel_y_nm=110.0)
    result = run_molecule_stack(truth["paths"], optics, BeadSettings(psf=PSFModelSettings(zernike_order=4)),
                                settings, run=RunSettings(save_model=True), camera=CAMERA, rois=rois,
                                save_to=tmp_path / "dual.h5", device="cpu")
    shared = result.cal.shared()
    for prefix in ("", "ch1."):
        pupil = result.cal.model.channels[0 if not prefix else 1].psf.pupil
        nm = {int(j): float(v) * OPTICS.wavelength_nm / (2 * np.pi)
              for j, v in zip(pupil.phase_modes, shared[prefix + "pupil.phase"].detach().cpu(), strict=True)}
        assert all(abs(nm[j] - v) < 6.0 for j, v in PHASE_NM.items()), (prefix, nm)
    cal = load_dual_color_calibration(result.saved)
    assert cal.main.parameters["sample"] == "molecules on the coverslip in a z stack"


def test_the_elliptical_gaussian_from_molecules_gives_back_their_z(tmp_path):
    from smappy.astigmatic_z import z_from_widths
    from smappy.calibrate.elliptical import EllipticalCalibrationSettings, calibrate_widths
    from smappy.calibrate.molecule_stack import molecule_widths
    truth = write_sweeps(tmp_path, OPTICS, files=1, sweeps=3, per_frame=6, step_nm=100.0, span_nm=600.0,
                         focus_um=(50.0,))
    rois = molecule_rois(truth["paths"], MoleculeStackSettings(molecules=200), CAMERA)
    widths = molecule_widths(rois, (110.0, 110.0), EllipticalCalibrationSettings())
    cal = calibrate_widths(widths, higher_orders=True)
    z = widths.z_nm[:, 0]
    found = z_from_widths(widths.sx_nm[:, 0], widths.sy_nm[:, 0], cal, widths.sx_err_nm[:, 0],
                          widths.sy_err_nm[:, 0])
    found = found.z_nm
    inside = np.abs(z) < 500
    # each molecule's z back from its widths, about the stage's (the sweeps' own foci scatter ±20 nm)
    assert np.nanmedian(np.abs(np.asarray(found)[inside] - z[inside])) < 40
