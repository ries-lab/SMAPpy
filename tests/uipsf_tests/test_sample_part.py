"""The sample part (`smappy.calibrate.depth.SampleSettings`, `for_sample`): a pupil (Zernike) calibration drawn
for a focus in the sample wherever a spline is taken besides Spline 3D -- the DECODE training, *Simulate*,
Spline 3D 2C -- with Spline 3D's choices and meanings, and shown only for a calibration that carries a model."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from smappy.calibrate import depth
from smappy.calibrate.depth import SampleSettings
from smappy.calibrate.pupil import spline_calibration
from smappy.io.calibration import load_spline_calibration, save_spline_calibration

import uipsf
from uipsf.export import LearnedPSF, describe, sample_psf
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=660, n_imm=1.518, n_med=1.33, pixel_size_nm=(110.0, 110.0))
ROI, BOX = 17, 13


@pytest.fixture(scope="module")
def learned():
    """An astigmatic pupil as a bead calibration would have learnt it (`test_sample_depth`'s)."""
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude="fixed", blur="free",
                                                                 pixel_blur=False)),
                       (ROI, ROI), z_max_nm=1500, device="cpu", dtype=torch.float64)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=torch.float64)
    phase[:4] = torch.tensor([0.8, -0.2, 0.15, 0.1])
    shared = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0], dtype=torch.float64),
              "pupil.magnitude": torch.zeros(len(model.psf.pupil.magnitude_modes), dtype=torch.float64)}
    return LearnedPSF(model, shared, {}, describe(model))


@pytest.fixture
def files(tmp_path, learned, monkeypatch):
    """A bead average (no model) and a pupil calibration whose model file is the fixture's."""
    beads = spline_calibration(learned, 20.0, 800.0, ROI)
    average = tmp_path / "average_3dcal.h5"
    save_spline_calibration(average, replace(beads, parameters={"method": "bead average"}))
    pupil = tmp_path / "pupil_3dcal.h5"
    save_spline_calibration(pupil, replace(beads, parameters=dict(beads.parameters, model_file="pupil_3dcal.uipsf.h5")))
    (tmp_path / "pupil_3dcal.uipsf.h5").write_bytes(b"")
    monkeypatch.setattr(depth, "load", lambda path: learned)
    depth._DRAWN.clear()
    return load_spline_calibration(average), load_spline_calibration(pupil)


def _spots(learned, focus_nm, heights, photons, rng, background=15.0):
    xyz = np.c_[rng.uniform(-50, 50, (len(heights), 2)), np.full(len(heights), -float(focus_nm))]
    psf = sample_psf(learned.model, learned.shared, xyz, BOX, depth_nm=heights).double().numpy()
    return rng.poisson(psf * np.asarray(photons, float)[:, None, None] + background).astype(np.float32)


def test_the_sample_part_is_shown_only_for_a_calibration_that_carries_a_model(files):
    average, pupil = files
    shown, _, _ = depth.sample_state(average, SampleSettings())
    assert not any(shown.values())
    shown, active, hints = depth.sample_state(pupil, SampleSettings())
    assert shown == {"sample.focus": True, "sample.focus_nm": True, "sample.sample_index": True}
    assert not active["sample.focus_nm"] and hints["sample.focus_nm"].startswith("auto")
    shown, active, _ = depth.sample_state(pupil, SampleSettings(focus="calibrated"))
    assert shown["sample.focus"] and not shown["sample.focus_nm"]
    assert depth.sample_state(pupil, SampleSettings(focus="typed"))[1]["sample.focus_nm"]


def test_a_bead_average_or_as_calibrated_is_the_calibration_as_it_is(files):
    average, pupil = files
    assert depth.for_sample(average, SampleSettings(focus="typed", focus_nm=500.0)) == (average, None)
    assert depth.for_sample(pupil, SampleSettings(focus="calibrated")) == (pupil, None)


def test_a_typed_focus_draws_the_spline_there_and_says_how(files):
    _, pupil = files
    spline, found = depth.for_sample(pupil, SampleSettings(focus="typed", focus_nm=600.0))
    record = spline.parameters["sample_depth"]
    assert found is None and record["method"] == "typed" and record["focus_nm"] == 600.0
    low, high = depth.z_bounds(spline)
    # z the height above the coverslip: the spline reaches into the glass, no further than BELOW_NM
    assert -depth.BELOW_NM - spline.dz <= low < 0 < depth.in_focus(spline) < high
    assert "600 nm into the sample (typed)" in depth.describe_sample(spline)
    # the kept drawing is not changed by the record: an auto focus at the same depth says auto
    again, _ = depth.for_sample(pupil, SampleSettings(focus="typed", focus_nm=600.0))
    assert again.parameters["sample_depth"]["method"] == "typed"
    assert "method" not in depth.at_depth(pupil, 600.0).parameters["sample_depth"]


def test_auto_finds_the_focus_from_the_lowest_spots_and_draws_it_there(files, learned):
    _, pupil = files
    rng = np.random.default_rng(3)
    h = np.where(rng.random(1500) < 0.3, 0.0, rng.uniform(0, 1000, 1500))
    images = _spots(learned, 700.0, h, np.full(len(h), 8000.0), rng) + 1.0
    spline, found = depth.for_sample(pupil, SampleSettings(focus="auto"), images=images)
    assert found.focus_nm == pytest.approx(700.0, abs=60)
    assert spline.parameters["sample_depth"]["method"] == "auto"


def test_spots_taken_with_the_other_em_gain_are_mirrored_into_the_beads_orientation(files):
    _, pupil = files
    images = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    beads = replace(pupil, em_on=False)
    assert depth.in_beads_orientation(images, beads, False) is images
    assert depth.in_beads_orientation(images, beads, None) is images
    assert np.array_equal(depth.in_beads_orientation(images, beads, True), images[..., ::-1])


def test_a_simulation_draws_its_frames_at_the_typed_focus_and_fit_this_hands_it_to_spline_3d(files, tmp_path):
    from smappy.plugins.fit import SplineFit
    from smappy.simulate.camera import spline_for
    from smappy.simulate.settings import SimulationSettings
    from smappy.simulate.source import write_recipe
    _, pupil = files
    settings = SimulationSettings(output="camera")
    optics = replace(settings.optics, calibration=str(pupil.source))
    assert spline_for(optics).parameters.get("sample_depth") is None            # as calibrated, the default
    optics = replace(optics, sample=SampleSettings(focus="typed", focus_nm=400.0))
    assert spline_for(optics).parameters["sample_depth"]["focus_nm"] == 400.0
    with pytest.raises(ValueError, match="type it"):
        spline_for(replace(optics, sample=SampleSettings(focus="auto")))
    recipe = write_recipe(replace(settings, optics=optics), tmp_path / "scene.sim.yaml")
    plugin = SplineFit()
    s = plugin.Settings()
    s = replace(s, source=replace(s.source, path=str(recipe)))
    updates = plugin.react("source.path", s)
    assert updates["model.calibration"] == str(pupil.source)
    assert updates["model.focus"] == "typed" and updates["model.focus_nm"] == 400.0


def test_a_dual_pupil_calibration_drawn_at_depth_keeps_the_share_and_its_map(tmp_path, monkeypatch):
    # `test_smappy_dual`'s ratiometric pair: two channels, their own aberrations, a share and a defocus
    from smappy.psf import GlobalSplinePSF
    from tests.uipsf_tests.test_smappy_dual import SHARE, _dual_calibration, _model
    model, shared = _model()
    dual = _dual_calibration(model, shared, tmp_path)
    monkeypatch.setattr(depth, "model_of", lambda cal, channels=1: SimpleNamespace(model=model, shared=shared))
    drawn = depth.dual_depth_calibration(dual, 500.0)
    assert drawn.secondary.parameters["photon_normalization"] == pytest.approx(SHARE, abs=0.03)
    assert np.array_equal(drawn.transformation, dual.transformation) and drawn.geometry == dual.geometry
    for spline in (drawn.main, drawn.secondary):
        assert spline.parameters["sample_depth"]["focus_nm"] == 500.0
        assert spline.parameters["z_reference"].startswith("the coverslip")
    psf = GlobalSplinePSF(drawn.layer_splines(), z_start_nm=depth.in_focus(drawn.main))    # one z grid
    assert psf.z_start_nm > 0


def test_a_decode_job_learns_the_psf_drawn_in_the_sample_and_records_where(files, learned, tmp_path):
    """The training window's work without the window: the prefit fits with the spline drawn at the typed focus,
    the estimates' z are heights above the coverslip, and the job's calibration and record are that spline's."""
    import tifffile
    from smappy.gui.train_window import PrepareSettings, default_folder, run_prefit, write_job
    from smappy.io.calibration import load_spline_calibration
    from smappy.plugins.fit import CameraSettings
    from smappy.training import job as jobfile
    _, pupil = files
    rng = np.random.default_rng(5)
    centres = [(12 + 20 * i, 12 + 20 * j) for i in range(4) for j in range(4)]
    n_frames = 60
    h = rng.uniform(0, 800, n_frames * len(centres))
    spots = _spots(learned, 600.0, h, rng.lognormal(np.log(5000), 0.2, len(h)), rng, background=0.0)
    frames = rng.poisson(np.full((n_frames, 92, 92), 15.0)).astype(np.float32)
    half = BOX // 2
    for k, spot in enumerate(spots):
        f, (x, y) = k // len(centres), centres[k % len(centres)]
        frames[f, y - half:y + half + 1, x - half:x + half + 1] += spot
    movie = tmp_path / "cells.tif"
    tifffile.imwrite(movie, (frames + 100).astype(np.uint16))
    settings = PrepareSettings(movie=str(movie), calibration=str(pupil.source), blocks=2, block_frames=30,
                               camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.11, em_on=False),
                               sample=SampleSettings(focus="typed", focus_nm=600.0))
    prepared = run_prefit(settings)
    # heights, not z about the beads' focus: the molecules were drawn 0 to 800 nm above the coverslip
    z = np.asarray(prepared.localizations["z_nm"])
    assert np.quantile(z, 0.05) == pytest.approx(40.0, abs=40) and np.quantile(z, 0.95) == pytest.approx(760.0, abs=40)
    low, high = depth.z_bounds(prepared.calibration)
    assert low <= prepared.estimates.z_nm[0] < prepared.estimates.z_nm[1] <= high
    folder = write_job(prepared, prepared.estimates, default_folder(movie, pupil.source))
    job, _ = jobfile.read(folder)
    assert job.calibration["sample_depth"]["focus_nm"] == 600.0
    written = load_spline_calibration(folder / job.calibration["file"])
    assert written.parameters["sample_depth"]["method"] == "typed"
