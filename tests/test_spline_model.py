"""Spline 3D's model settings: the calibration file, the EM gain it was taken
with, the ROI it can hold, and where z starts."""
import numpy as np
import pytest

from smappy.calibrate.core import CalibrationSettings, build_calibration, collect_beads
from smappy.calibrate.input import BeadStack
from smappy.io.calibration import load_spline_calibration, save_spline_calibration
from smappy.metadata import CameraMetadata
from smappy.plugins import Context
from smappy.plugins.fit import SplineFit, SplineFitSettings, SplineModelSettings


@pytest.fixture(scope="module")
def calibration():
    from smappy.simulate import bead_stacks
    stacks, z = bead_stacks(1, seed=0)
    beads = collect_beads([BeadStack(s.astype(np.float32), z, source="b")
                           for s in stacks], CalibrationSettings(roi_size=21))
    return build_calibration(beads).calibration


def test_the_calibration_dialog_offers_the_bead_tools_own_files():
    info = SplineModelSettings.__dataclass_fields__["calibration"].metadata["smappy"]
    assert "*.h5" in info.file_filter and "*_3dcal.mat" in info.file_filter


def test_a_saved_calibration_keeps_the_em_gain_it_was_taken_with(tmp_path, calibration):
    """It was dropped on saving, so the mismatch warning never fired for a
    calibration made here -- only for SMAP's."""
    for em_on in (True, False, None):
        calibration.em_on = em_on
        path = tmp_path / f"cal_{em_on}.h5"
        save_spline_calibration(path, calibration)
        assert load_spline_calibration(path).em_on is em_on


def test_beads_in_memory_say_nothing_about_em_gain():
    from smappy.simulate import bead_stacks
    stacks, z = bead_stacks(1, seed=1)
    beads = collect_beads([BeadStack(stacks[0].astype(np.float32), z)],
                          CalibrationSettings(roi_size=21))
    assert beads.em_on is None


def test_an_em_mismatch_is_reported_where_the_user_reads_it(tmp_path, calibration):
    """A Python warning goes to a console a GUI user never sees."""
    calibration.em_on = True
    path = tmp_path / "em.h5"
    save_spline_calibration(path, calibration)
    told = []
    settings = SplineFitSettings(model=SplineModelSettings(calibration=str(path)))
    with pytest.warns(UserWarning, match="EM gain mismatch"):
        SplineFit()._model_telling(Context(progress=told.append), settings,
                                   CameraMetadata(em_on=False))
    assert any("EM gain mismatch" in line for line in told)


def test_a_roi_larger_than_the_calibration_is_refused(tmp_path, calibration):
    """The model used to repeat its edge beyond the grid, without a word."""
    calibration.em_on = None
    path = tmp_path / "small.h5"
    save_spline_calibration(path, calibration)
    lateral = min(calibration.shape[1:])
    model = SplineModelSettings(calibration=str(path))
    model.model(roisize=lateral - 2)
    with pytest.raises(ValueError, match="smaller than the calibration"):
        model.model(roisize=lateral - 1)


def test_the_z_start_is_a_setting(tmp_path, calibration):
    path = tmp_path / "z.h5"
    save_spline_calibration(path, calibration)
    psf = SplineModelSettings(calibration=str(path), z_start_nm=-250.0).model()
    assert psf.z_start_nm == -250.0
