"""A learnt pupil model as smappy's spline calibration (`smappy.calibrate.pupil`, plan §6 and §7.5): Spline 3D
fits frames of the true PSF with it, z the bead average's way up, the photons as the bead average counts them.

The acceptance of plan §9 M6 is here too: simulated beads -> uipsf -> spline -> Spline 3D, z slope 1.00 ± 0.01
and the bead average's photons; Spline 3D and uipsf's own localisation agree on the same ROIs (§6.5)."""
import numpy as np
import pytest
import torch
from smappy.calibrate.core import CalibrationSettings, calibrate
from smappy.calibrate.input import BeadStack
from smappy.calibrate.pupil import model_path, save, single_rois, spline_calibration, z_grid
from smappy.io.calibration import load_spline_calibration
from smappy.metadata import CameraMetadata
from smappy.psf import SplinePSF

import uipsf
from uipsf.export import sample_psf
from uipsf.workflows import BeadSettings, bead_model, calibrate_beads
from uipsf.workflows.scoring import localise, slice_fits

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=660, n_imm=1.518, n_med=1.33, pixel_size_nm=(110.0, 110.0))
CAMERA = CameraMetadata(conversion=2.0, offset=100.0, pixelsize_um=0.11, read_noise_e=1.6)
ROI = 17


def _truth():
    model = bead_model(OPTICS, BeadSettings(zernike_order=4, magnitude="fixed"), (ROI, ROI), z_max_nm=1200,
                       device="cpu")
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    phase[:4] = torch.tensor([0.8, -0.2, 0.15, 0.1])                   # astigmatic, for z
    return model, {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0])}


def _draw(model, shared, xyz, photons, background=0.0):
    """Expected photons of emitters at ``xyz`` [n, 3], one frame each: [n, ROI, ROI], each pixel integrated over
    8 × 8 sub-samples, as a camera records it -- not the model's own pixel centres, which the learning fits
    with (`UPSAMPLE` in `smappy.calibrate.pupil` is why that matters)."""
    shared = dict(shared, **{"pupil.magnitude": torch.zeros(len(model.psf.pupil.magnitude_modes))})
    psf = sample_psf(model, shared, np.asarray(xyz, float), roi_px=ROI, upsample=8).double().numpy()
    return float(photons) * psf + float(background)


@pytest.fixture(scope="module")
def learnt():
    return _learn()


def _learn():
    """Twelve beads through ±1.2 µm in 50 nm steps, drawn into one ADU stack, learnt through smappy's adapter;
    the same stack's bead average beside it."""
    model, shared = _truth()
    rng = np.random.default_rng(0)
    centres = [(30 + 40 * (i % 4), 30 + 40 * (i // 4)) for i in range(12)]
    objective = np.arange(49) * 50.0
    z = -(objective - objective.mean())
    image = np.full((len(z), 150, 190), 20.0)
    h = ROI // 2
    for x, y in centres:
        offset = np.r_[rng.uniform(-40, 40, 2), rng.uniform(-60, 60)]
        spots = _draw(model, shared, np.c_[np.tile(offset[:2], (len(z), 1)), z + offset[2]], 40000)
        image[:, y - h:y + h + 1, x - h:x + h + 1] += spots
    adu = np.round(rng.poisson(image) / CAMERA.adu_to_photons + CAMERA.offset).astype(np.uint16)
    stack = BeadStack(adu, objective, "sim", {}, (0, 0, 190, 150))
    settings = CalibrationSettings(roi_size=ROI, min_distance_px=20)
    rois = single_rois([stack], CAMERA, settings, slices=slice(None))
    cal = calibrate_beads(rois.batch, OPTICS, BeadSettings(zernike_order=4), noise=rois.noise, device="cpu")
    average = calibrate([stack], CalibrationSettings(roi_size=ROI, min_distance_px=20, smooth_z_nm=0.0))
    return model, shared, cal, average.calibration, rois


def _fit(calibration, frames, z_start=0.0):
    psf = SplinePSF(calibration, z_start_nm=z_start)
    return psf.unpack(psf.fit(np.ascontiguousarray(frames, np.float32), iterations=100))


def test_the_z_grid_runs_down_from_the_highest_emitter_with_focus_at_z0():
    z, z0 = z_grid(50.0, 120.0)
    assert z0 == 3.0 and z.tolist() == [150.0, 100.0, 50.0, 0.0, -50.0, -100.0, -150.0]


@pytest.mark.slow
def test_spline_3d_with_the_learnt_model_finds_the_true_z_and_the_bead_average_agrees_on_its_sign(learnt):
    model, shared, cal, average, _ = learnt
    spline = spline_calibration(cal, 25.0, 900.0, roi_size=ROI, camera=CAMERA)
    assert spline.parameters["method"] == "uipsf2_pupil" and spline.em_on is False
    assert spline.psf.sum(axis=(1, 2)).max() == pytest.approx(1.0, rel=1e-5)
    rng = np.random.default_rng(1)
    truth = np.c_[rng.uniform(-30, 30, (120, 2)), rng.uniform(-600, 600, 120)]
    frames = rng.poisson(_draw(model, shared, truth, 5000, background=10.0)).astype(float)
    frames = frames[:, 2:-2, 2:-2]                                    # a 13 px fit box in the 17 px spline
    ours, theirs = _fit(spline, frames), _fit(average, frames)
    error = ours["z_nm"] - truth[:, 2]
    assert abs(np.median(error)) < 10 and np.std(error) < 40
    assert np.corrcoef(ours["z_nm"], theirs["z_nm"])[0, 1] > 0.98     # the same way up as the bead average
    x = (ours["x_roi"] - (frames.shape[-1] - 1) / 2) * 110.0
    assert abs(np.median(x - truth[:, 0])) < 5
    # photons as the bead average counts them: both normalise the brightest plane to one
    assert np.median(ours["photons"] / theirs["photons"]) == pytest.approx(1.0, abs=0.05)


@pytest.mark.slow
def test_the_calibration_and_its_model_file_are_saved_side_by_side(learnt, tmp_path):
    _, _, cal, _, _ = learnt
    spline = spline_calibration(cal, 50.0, 600.0, roi_size=ROI)
    path, beside = save(tmp_path / "beads_3dcal.h5", cal, spline)
    assert beside == model_path(path) == tmp_path / "beads_3dcal.uipsf.h5" and beside.exists()
    loaded = load_spline_calibration(path)
    assert loaded.parameters["model_file"] == "beads_3dcal.uipsf.h5"
    assert loaded.parameters["zernikes"][0]["phase"][0][0] == 5                # Noll 5 first, with its CRLB
    assert np.allclose(loaded.coeff, spline.coeff)
    again = spline_calibration(uipsf.load_model(beside, device="cpu"), 50.0, 600.0, roi_size=ROI)
    assert np.allclose(again.psf, spline.psf, atol=1e-6)               # re-sampled from the file, not relearnt
    with pytest.raises(FileExistsError):
        save(path, cal, spline)


@pytest.mark.slow
def test_the_whole_chain_gives_spline_3d_a_z_slope_of_one_and_the_bead_averages_photons(learnt):
    """Plan §9 M6.  Measured: slope 0.9986, the photons 0.2 % above the bead average's; the spline sampled with
    its pixels integrated once more (4 × 4, before October 2026) gave 0.992 and 1.2 %."""
    model, shared, cal, average, _ = learnt
    spline = spline_calibration(cal, 25.0, 900.0, roi_size=ROI, camera=CAMERA)
    rng = np.random.default_rng(1)
    levels = np.linspace(-600, 600, 25)
    truth = np.c_[rng.uniform(-30, 30, (40 * len(levels), 2)), np.repeat(levels, 40)]
    frames = rng.poisson(_draw(model, shared, truth, 5000, background=10.0)).astype(float)[:, 2:-2, 2:-2]
    ours, theirs = _fit(spline, frames), _fit(average, frames)
    assert np.isfinite(ours["z_nm"]).all()
    slope, intercept = np.polyfit(truth[:, 2], ours["z_nm"], 1)
    assert slope == pytest.approx(1.0, abs=0.01) and abs(intercept) < 5
    assert np.median(ours["photons"] / theirs["photons"]) == pytest.approx(1.0, abs=0.03)


@pytest.mark.slow
def test_spline_3d_and_uipsfs_own_localisation_agree_on_the_same_bead_rois(learnt):
    """Plan §6.5: every slice of every bead localised by the pupil model (`scoring.slice_fits`, from the beads' stack
    fits) and by Spline 3D on the exported spline, the same pixels (the spline 4 px larger than the ROI, for the
    spot's shift).  Over ±600 nm, measured: z 0.07 nm apart in the median, 2.8 nm rms against a CRLB of 3.4, the
    photons in a constant ratio (the spline's photons are the light its box holds) to 0.6 %.  With the pixels
    integrated twice (`UPSAMPLE`) z was 6.1 nm rms apart and bent by ±7 nm: the sampling was wrong."""
    _, _, cal, _, rois = learnt
    batch = rois.batch
    fit, _ = localise(cal.model, cal.shared(), batch, rois.noise)
    found, crlb, stack, photons = slice_fits(cal.model, cal.shared(), batch, rois.noise, fit.params, photons=True)
    E, K = found.shape[:2]
    spline = spline_calibration(cal, 25.0, 1300.0, roi_size=ROI + 4)
    assert spline.parameters["sampling"]["upsample"] == cal.model.psf.upsample
    data = np.maximum(batch.data[:, 0].double().numpy() + float(batch.s), 0).reshape(E * K, ROI, ROI)
    best = None
    for z in (-600, -300, 0, 300, 600):
        theirs = SplinePSF(spline, z_start_nm=z).fit(np.ascontiguousarray(data, np.float32), iterations=100)
        if best is None:
            best = theirs
            continue
        keep = np.isfinite(theirs.logl) & (~np.isfinite(best.logl) | (theirs.logl > best.logl))
        for key in ("theta", "logl"):
            getattr(best, key)[keep] = getattr(theirs, key)[keep]
    z = spline.z_index_to_nm(best.theta[:, 4]).reshape(E, K)
    central = np.abs(stack[..., 2]) < 600
    dz = (z - found[..., 2])[central]
    assert abs(np.median(dz)) < 1.0
    assert np.sqrt(np.mean(dz ** 2)) < np.median(crlb[..., 2][central])
    ratio = (best.theta[:, 2].reshape(E, K) / photons)[central]
    assert np.diff(np.percentile(ratio, [16, 84]))[0] / 2 < 0.01 * np.median(ratio)
