"""Molecules on the coverslip through z sweeps: reading, the width curve, detection and selection
(`smappy.calibrate.molecule_stack`), and a whole run that recovers the pupil it was simulated with
(`molecule_stack_run`)."""
import numpy as np
import pytest

import uipsf
from smappy.calibrate.molecule_stack import MoleculeStackSettings, molecule_rois, read_file, split_sweeps
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


@pytest.mark.slow
def test_the_molecules_are_found_at_each_files_focus_and_drawn_evenly_over_the_defocus(tmp_path):
    truth = write_sweeps(tmp_path, OPTICS, files=2, sweeps=2, focus_um=(50.0, 50.3))
    rois = molecule_rois(truth["paths"], MoleculeStackSettings(molecules=150), CAMERA)
    # the files' foci 300 nm apart, each found within a few tens of nm (the sweeps' own foci scatter 20 nm)
    assert abs(np.diff(rois.curve.focus_um)[0] - 0.3) < 0.04
    assert rois.curve.shape == "astigmatic" and rois.sweeps == 4
    used = rois.ids("used")
    assert len(used) == 150
    counts = np.histogram(rois.defocus_nm[used], np.arange(-1050, 1051, 100))[0]
    # evenly over the defocus the molecules pass at: at the far end the curve's 15-pixel fits lose the spots
    filled = counts[counts > 0]
    assert len(filled) >= 14 and filled.max() - filled.min() <= 2
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
    assert abs(result.height_nm() - truth["height_nm"]) < 10.0
    foci = shared["geometry.focus"].detach().cpu().double().numpy()
    true = truth["foci_nm"]
    assert np.abs((foci - foci.mean()) - (true - true.mean())).max() < 10.0
    assert result.check.scatter < 1.6 and result.check.bend_rms(600) < 0.5 * np.median(result.check.crlb_nm)
    cal = load_spline_calibration(result.saved)
    assert cal.parameters["sample"] == "molecules on the coverslip in a z stack"
    assert abs(cal.parameters["height_nm"] - result.height_nm()) < 1e-6
    assert (result.saved.parent / "cal.uipsf.h5").exists()
