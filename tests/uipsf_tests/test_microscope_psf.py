"""The microscope's PSF (`smappy.calibrate.microscope_psf`): a pupil kept with a microscope's entry, drawn again
from its description (`uipsf.export.from_description`) into a spline Spline 3D fits with, and drawn at depth by
its *focus* as a bead model is."""
import json
import os
import sys

import numpy as np
import pytest
import torch
from smappy import camera_db, microscope_db
from smappy.calibrate import depth
from smappy.calibrate import microscope_psf as mpsf

import uipsf
from uipsf.export import LearnedPSF, describe, from_description
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_microscope_db import M2_BEADS  # noqa: E402
from test_microscope_manager import a_tiff  # noqa: E402

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(127.0, 116.0))
ROI = 17


def _model(magnitude: str) -> LearnedPSF:
    """An astigmatic bead model, as a bead calibration learns it, with every value set."""
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude=magnitude)),
                       (ROI, ROI), z_max_nm=1000, device="cpu", dtype=torch.float64)
    pupil = model.psf.pupil
    phase = torch.zeros(len(pupil.phase_modes), dtype=torch.float64)
    phase[:4] = torch.tensor([0.8, -0.2, 0.15, 0.1])
    shared = {spec.name: torch.as_tensor(spec.init, dtype=torch.float64).expand(spec.shape).clone()
              for spec in model.psf.specs()}
    shared.update({"pupil.phase": phase, "psf.sigma": torch.tensor([40.0], dtype=torch.float64)})
    if magnitude == "minimal":
        shared["pupil.gamma"] = torch.tensor(0.3, dtype=torch.float64)
    else:
        shared["pupil.magnitude"] = 0.05 * torch.ones(len(pupil.magnitude_modes), dtype=torch.float64)
    return LearnedPSF(model, shared, {}, describe(model))


@pytest.mark.parametrize("magnitude", ["minimal", "symmetric"])
def test_a_pupils_description_draws_the_psf_it_describes(magnitude):
    learned = _model(magnitude)
    psf = learned.model.channels[0].psf
    description = psf.describe(learned.shared)
    again = from_description(description, OPTICS, roi_px=ROI, upsample=psf.upsample, pupil_size=psf.grid.size)
    xyz = np.array([[0.0, 0.0, -500.0], [20.0, -10.0, 0.0], [0.0, 30.0, 600.0]])
    # to the rounding of the description (the phase to a thousandth of a nm)
    expected = learned.psf(xyz, ROI)
    assert float((again.psf(xyz, ROI) - expected).abs().max()) < 1e-4 * float(expected.max())
    # and it describes itself as it was described: what is kept is what is drawn
    back = again.model.channels[0].psf.describe(again.shared)
    assert back["phase_nm"] == description["phase_nm"] and back.get("gamma") == description.get("gamma")
    # drawn at another emission, the phase is the same wavefront in nm, not the same radians
    red = from_description(description, uipsf.Optics(**{**OPTICS.__dict__, "wavelength_nm": 600.0}), ROI)
    j = red.model.channels[0].psf.pupil.phase_modes.index(5)
    assert float(red.shared["pupil.phase"][j]) == pytest.approx(0.8 * 680 / 600, rel=1e-3)


@pytest.fixture
def config(tmp_path, monkeypatch):
    from smappy import config as settings
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    settings.load(reload=True)
    camera_db.database(reload=True)
    microscope_db.database(reload=True)
    yield tmp_path / "config"
    monkeypatch.undo()
    # a fit's file recognised here is the "last used" microscope: read the session's again, so that a later
    # test of an unrecognised file is not given M2 as a stand-in
    settings.load(reload=True)
    camera_db.database(reload=True)
    microscope_db.database(reload=True)


def _keep(tmp_path):
    """M2's pupil kept from the minimal model, and a file of its beads' state."""
    learned = _model("minimal")
    pupil = dict(learned.model.channels[0].psf.describe(learned.shared), source="beads_3dcal.h5",
                 date="2026-08-29")
    data = a_tiff(tmp_path / "movie_MMStack.ome.tif", M2_BEADS)
    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import CameraSettings
    scope = CameraSettings().resolve(open_stack(data), require=False).microscope
    microscope_db.keep_pupil(scope, pupil)
    return data, scope


def test_the_microscopes_pupil_is_a_calibration_drawn_once_and_drawn_again_at_depth(tmp_path, config):
    data, scope = _keep(tmp_path)
    path = mpsf.calibration_path(scope, em_on=True)
    assert path.parent == config / mpsf.MICROSCOPE_FOLDER and path.name.startswith("M2_3DA_on")
    written = path.stat().st_mtime_ns
    assert mpsf.calibration_path(scope, em_on=True).stat().st_mtime_ns == written     # drawn once
    from smappy.io.calibration import load_spline_calibration
    cal = load_spline_calibration(path)
    record = cal.parameters["pupil"]
    # the file's emission (its 700/100) and the pixel of its lens state, the beads' medium
    assert record["optics"]["wavelength_nm"] == 700.0 and record["optics"]["pixel_size_nm"] == [127.0, 116.0]
    assert record["key"] == "3DA: on · far red" and cal.parameters["microscope"]["name"] == "M2"
    assert cal.dz == mpsf.DZ_NM and cal.em_on is True
    # *focus* draws it at depth as it draws a bead model: z the height above the coverslip
    assert depth.has_model(cal) and depth.model_file(cal) is None
    drawn = depth.at_depth(cal, 1000.0)
    assert 600 < depth.in_focus(drawn) < 1000                  # the plane in focus below the focus, in water


def test_spline_3d_offers_the_microscopes_pupil_and_fits_with_its_file(tmp_path, config):
    from smappy.plugins.fit import SplineFit
    data, _ = _keep(tmp_path)
    plugin = SplineFit()
    settings = plugin.Settings()
    settings.source.path = data
    state = plugin.describe(None, settings)
    choice = dict(state.choices["model.calibration"])
    assert mpsf.MICROSCOPE in choice and "M2's pupil, 3DA: on" in choice[mpsf.MICROSCOPE]
    settings.model.calibration = mpsf.MICROSCOPE
    state = plugin.describe(None, settings)
    assert state.shown["model.focus"]                         # it has a model to draw at depth
    said = []
    resolved = plugin.with_calibration(settings, type("Ctx", (), {"report": said.append})())
    assert resolved.model.calibration.endswith("_3dcal.h5") and "M2's pupil" in said[0]
    # with no listed calibration, auto says the pupil is there
    settings.model.calibration = ""
    with pytest.raises(ValueError, match="the microscope's pupil"):
        plugin.with_calibration(settings)


def test_simulate_offers_every_kept_pupil_and_draws_it_at_its_pixel(tmp_path, config):
    from smappy.plugins.simulate import Simulate
    _keep(tmp_path)
    value, label = mpsf.choices()[0]
    assert value == "microscope:M2:3DA: on · far red" and "M2's pupil" in label
    plugin = Simulate()
    settings = plugin.Settings()
    settings.optics.pixelsize_nm = 110.0
    settings.optics.calibration = value
    path = plugin.react("optics.calibration", settings)["optics.calibration"]
    from smappy.io.calibration import load_spline_calibration
    record = load_spline_calibration(path).parameters["pupil"]
    assert record["optics"]["pixel_size_nm"] == [110.0, 110.0]
    assert json.loads(json.dumps(record["pupil"]))["gamma"] == pytest.approx(0.3)
