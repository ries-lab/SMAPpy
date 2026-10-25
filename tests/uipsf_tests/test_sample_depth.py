"""Spline 3D's *sample depth* (`smappy.calibrate.depth`): a bead-learnt pupil drawn as the spline of a sample
whose focus is some way above the coverslip, z the height above it, and the focus found from the lowest
localisations."""
import numpy as np
import pytest
import torch
from smappy.calibrate import depth
from smappy.calibrate.pupil import spline_calibration

import uipsf
from uipsf.export import LearnedPSF, describe, sample_psf
from uipsf.workflows import BeadSettings, bead_model

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=660, n_imm=1.518, n_med=1.33, pixel_size_nm=(110.0, 110.0))
ROI, BOX = 17, 13


@pytest.fixture(scope="module")
def learned():
    """An astigmatic pupil as a bead calibration would have learnt it, wrapped as its model file loads."""
    model = bead_model(OPTICS, BeadSettings(zernike_order=4, magnitude="fixed"), (ROI, ROI), z_max_nm=1500,
                       device="cpu", dtype=torch.float64)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=torch.float64)
    phase[:4] = torch.tensor([0.8, -0.2, 0.15, 0.1])
    shared = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0], dtype=torch.float64),
              "pupil.magnitude": torch.zeros(len(model.psf.pupil.magnitude_modes), dtype=torch.float64)}
    return LearnedPSF(model, shared, {}, describe(model))


@pytest.fixture(scope="module")
def beads(learned):
    """The bead spline: the model at the coverslip, z the objective's movement."""
    return spline_calibration(learned, 20.0, 800.0, ROI)


def _spots(learned, focus_nm, heights, photons, rng, background=15.0):
    xyz = np.c_[rng.uniform(-50, 50, (len(heights), 2)), np.full(len(heights), -float(focus_nm))]
    psf = sample_psf(learned.model, learned.shared, xyz, BOX, depth_nm=heights).double().numpy()
    return rng.poisson(psf * np.asarray(photons, float)[:, None, None] + background).astype(np.float32)


def test_a_spline_drawn_at_depth_gives_the_height_above_the_coverslip(learned, beads):
    rng = np.random.default_rng(0)
    h = rng.uniform(100, 1000, 400)
    images = _spots(learned, 600.0, h, np.full(len(h), 20000.0), rng)
    from smappy.psf import SplinePSF
    cal = depth.depth_spline(beads, 600.0, learned=learned)
    m = SplinePSF(cal, z_start_nm=depth.in_focus(cal))
    z = m.unpack(m.fit(images, iterations=50))["z_nm"]
    ok = np.isfinite(z)
    assert ok.sum() > 380
    # the drawn spline's z is the height: no scale, no offset
    assert np.median(np.abs(z[ok] - h[ok])) < 15
    # the bead spline's z is the objective's movement: off the heights' scale by the index mismatch
    b = SplinePSF(beads)
    zb = b.unpack(b.fit(images, iterations=50))["z_nm"]
    inside = np.isfinite(zb) & (np.abs(zb) < 600)
    slope = np.polyfit(h[inside], zb[inside], 1)[0]
    assert abs(slope - 1) > 0.1


def test_the_spline_reaches_below_the_coverslip_into_the_glass_and_runs_on_at_zero(learned, beads):
    cal = depth.depth_spline(beads, 0.0, learned=learned)
    low, high = cal.parameters["sample_depth"]["heights_nm"]
    assert low == pytest.approx(-depth.BELOW_NM, abs=cal.dz)
    assert cal.parameters["z_reference"].startswith("the coverslip")
    psf = depth.draw(learned.model, learned.shared, 300.0, [-1.0, 0.0, 1.0], ROI)
    # an emitter just in the glass and one just in the sample are nearly the same: no step at the coverslip
    assert np.abs(psf[0] - psf[1]).sum() < 0.05 * psf[1].sum()


def test_a_drawn_spline_with_another_sample_index_moves_the_plane_in_focus(learned):
    model, shared = depth.with_medium(learned, 1.45)
    water = depth.in_focus_height(learned.model, learned.shared, 1000.0, 800.0)
    glycerol = depth.in_focus_height(model, shared, 1000.0, 800.0)
    assert glycerol > water + 50              # a better matched sample: the focus moves nearly as the objective


@pytest.mark.slow
def test_the_auto_depth_finds_the_focus_from_the_molecules_on_the_coverslip(learned, beads):
    rng = np.random.default_rng(1)
    n = 2500
    h = np.where(rng.random(n) < 0.25, 0.0, rng.uniform(0, 1200, n))
    images = _spots(learned, 700.0, h, rng.lognormal(np.log(3000), 0.4, n), rng)
    found = depth.auto_focus(beads, images, learned=learned)
    assert found.focus_nm == pytest.approx(700.0, abs=40)
    assert found.spike > 0 and not found.messages


@pytest.mark.slow
def test_the_auto_depth_without_molecules_on_the_coverslip_stays_short_of_the_focus_and_says_so(learned, beads):
    rng = np.random.default_rng(2)
    n = 2500
    h = rng.uniform(250, 1200, n)                  # the lowest 250 nm above the coverslip
    images = _spots(learned, 700.0, h, rng.lognormal(np.log(3000), 0.4, n), rng)
    found = depth.auto_focus(beads, images, learned=learned)
    assert 200 < found.focus_nm < 700
    assert found.messages or found.notes



@pytest.mark.slow
def test_spline_3d_on_auto_finds_the_focus_once_draws_it_and_names_it(tmp_path, learned, beads, monkeypatch):
    """The whole plugin: a TIFF of molecules through 700 nm of sample, a bead calibration with its model
    beside it; *sample depth* on auto writes the depth it found into the settings, and z is the height."""
    from dataclasses import replace

    import tifffile
    from smappy.io.calibration import save_spline_calibration
    from smappy.pipeline import FitSettings
    from smappy.plugins import Context
    from smappy.plugins.fit import (
        CameraSettings,
        DetectionSettings,
        OutputSettings,
        SourceSettings,
        SplineFit,
        SplineFitSettings,
        SplineModelSettings,
    )
    rng = np.random.default_rng(4)
    centres = [(12 + 20 * i, 12 + 20 * j) for i in range(5) for j in range(5)]
    n_frames = 120
    h = np.where(rng.random(n_frames * len(centres)) < 0.25, 0.0, rng.uniform(0, 1000, n_frames * len(centres)))
    spots = _spots(learned, 700.0, h, rng.lognormal(np.log(4000), 0.3, len(h)), rng, background=0.0)
    frames = rng.poisson(np.full((n_frames, 112, 112), 15.0)).astype(np.float32)
    half = BOX // 2
    for k, spot in enumerate(spots):
        f, (x, y) = divmod(k, len(centres))[0], centres[k % len(centres)]
        frames[f, y - half:y + half + 1, x - half:x + half + 1] += spot
    stack = tmp_path / "sample.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    calibration = tmp_path / "beads_3dcal.h5"
    save_spline_calibration(calibration, replace(beads, parameters=dict(beads.parameters,
                                                                         model_file="beads_3dcal.uipsf.h5")))
    (tmp_path / "beads_3dcal.uipsf.h5").write_bytes(b"")          # the model is the fixture's, below
    monkeypatch.setattr(depth, "load", lambda path: learned)
    settings = SplineFitSettings(
        source=SourceSettings(path=str(stack)),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.11),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=10.0),
        model=SplineModelSettings(calibration=str(calibration)),
        fit=FitSettings(roisize=BOX), output=OutputSettings(save=False))
    plugin = SplineFit()
    state = plugin.describe(None, settings)
    assert state.shown["model.focus"] and not state.active["model.focus_nm"]   # auto, the default
    preview = plugin.preview(Context(), settings)
    assert "focus" in preview.plots and not preview.data["error"]
    assert plugin.describe(None, settings).hints["model.focus_nm"].endswith("found before")
    told = []
    result = plugin.run(Context(progress=told.append), settings)
    assert any("as found before" in line for line in told)             # the Preview's, not found again
    used = result.settings.model
    assert used.focus == "typed" and used.focus_nm == pytest.approx(700.0, abs=50)
    z = np.asarray(result.locs["z_nm"])
    assert np.quantile(z, 0.1) == pytest.approx(0.0, abs=60)          # the lowest on the coverslip
