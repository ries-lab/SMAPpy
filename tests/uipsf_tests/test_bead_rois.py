"""smappy.calibrate.pupil: bead files to uipsf's batch, with smappy's reader, camera, detection and pairing."""
import numpy as np
import pytest
import torch
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.dual import DualColorSettings
from smappy.calibrate.input import BeadStack
from smappy.calibrate.pupil import dual_rois, single_rois
from smappy.metadata import CameraMetadata

CAMERA = CameraMetadata(conversion=2.0, offset=100.0, pixelsize_um=0.11, read_noise_e=1.6)


def _adu(photons, camera=CAMERA):
    return np.round(photons / camera.adu_to_photons + camera.offset).astype(np.uint16)


def _stack(spots, shape=(120, 140), K=7, seed=0, origin=(10, 20)):
    """Gaussian spots that sharpen towards the middle slice, Poisson noise on a background of 20 photons, in
    ADU; the camera ROI at ``origin``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.full((K, *shape), 20.0)
    for k in range(K):
        sd = 1.5 + 0.4 * abs(k - K // 2)
        for x, y, n in spots:
            img[k] += n / (2 * np.pi * sd ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sd ** 2))
    return BeadStack(_adu(rng.poisson(img).astype(float)), np.arange(K) * 50.0 + 1000.0, f"stack{seed}", {},
                     (origin[0], origin[1], shape[1], shape[0]))


def test_isolated_beads_are_found_and_crowded_or_cut_off_ones_are_not():
    spots = [(40, 40, 5000), (100, 60, 3000),          # isolated
             (60, 95, 4000), (68, 97, 4000),           # a pair closer than the minimum distance: neither kept
             (4, 70, 5000)]                            # its ROI would leave the image
    rois = single_rois([_stack(spots)], CAMERA, CalibrationSettings(roi_size=15))
    found = sorted((r["x_px"], r["y_px"]) for r in rois.records)
    assert found == [(40, 40), (100, 60)]


def test_rois_are_centred_photons_with_z_noise_saturation_and_chip_position():
    st = _stack([(40, 40, 5000)])
    st.images[3, 40, 40] = np.iinfo(np.uint16).max                 # one saturated pixel
    other = _stack([(50, 50, 5000)], seed=1, origin=(0, 0))
    settings = CalibrationSettings(roi_size=15)
    assert single_rois([st, other], CAMERA, settings).batch.data.shape[2] == 6     # the first slice dropped
    rois = single_rois([st, other], CAMERA, settings, slices=slice(None))
    batch = rois.batch
    assert batch.data.shape == (2, 1, 7, 15, 15)
    assert torch.argmax(batch.data[0, 0, 3]) == 7 * 15 + 7          # the spot's peak on the centre pixel
    expected = (st.images[2, 33:48, 33:48].astype(np.float32) - 100.0) * 2.0
    assert np.allclose(batch.data[0, 0, 2].numpy(), expected)       # photons, by smappy's conversion
    assert torch.allclose(batch.z_nm, -(torch.arange(7) * 50.0 - 150.0))   # smappy's sign, about the middle
    assert not batch.mask[0, 0, 3, 7, 7] and batch.mask.sum() == batch.mask.numel() - 1
    assert batch.field_xy.tolist() == [[50.0, 60.0], [50.0, 50.0]] and batch.stack_id.tolist() == [0, 1]
    assert batch.s == pytest.approx(1.6 ** 2) and rois.noise.excess == 1.0
    assert rois.pixel_size_nm == (110.0, 110.0)


def test_an_em_camera_has_no_read_noise_and_doubles_the_variance():
    em = CameraMetadata(conversion=10.0, offset=100.0, em_on=True, emgain=100.0, pixelsize_um=0.1)
    rois = single_rois([_stack([(40, 40, 5000)])], em, CalibrationSettings(roi_size=15))
    assert rois.batch.s == 0.0 and rois.noise.excess == 2.0


H_SMALL = np.array([[0.998, 0.012, 2.4], [-0.011, 1.001, 1.3], [1e-6, -2e-6, 1.0]])
SHIFT = np.array([[1.0, 0, 0], [0, 1, -256], [0, 0, 1]])            # the lower half onto the upper one
MIRROR = np.array([[1.0, 0, 0], [0, -1, 511], [0, 0, 1]])           # mirrored across the split


def _apply(H, xy):
    q = np.c_[xy, np.ones(len(xy))] @ H.T
    return q[:, :2] / q[:, 2:]


def _chip(H, seed, n=14, shape=(512, 256), origin=(100, 0)):
    """One split-chip stack in ADU: n beads in the upper half, their partners in the lower one."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    while True:                                     # isolated beads, away from the edges
        main = rng.uniform([30, 30], [shape[1] - 30, shape[0] / 2 - 30], (n, 2)) + origin
        d = np.linalg.norm(main[:, None] - main[None], axis=-1) + 1e3 * np.eye(n)
        if d.min() > 30:
            break
    second = _apply(np.linalg.inv(H), main)
    K = 7
    img = np.full((K, *shape), 10.0)
    for k in range(K):
        w = 1.5 + 0.3 * abs(k - K // 2)
        for x, y in np.r_[main, second] - origin:
            img[k] += 4000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * w ** 2))
    stack = BeadStack(_adu(rng.poisson(img).astype(float)), np.arange(K) * 100.0, f"chip{seed}", {},
                      (origin[0], origin[1], shape[1], shape[0]))
    return stack, main, second


@pytest.mark.parametrize("mirrored", [False, True])
def test_pairs_their_rois_and_a_starting_transform_come_back_from_a_simulated_chip(mirrored):
    H = H_SMALL @ MIRROR if mirrored else H_SMALL @ SHIFT
    stacks, mains, seconds = zip(*[_chip(H, seed) for seed in range(3)], strict=True)
    rois = dual_rois(list(stacks), CAMERA, DualColorSettings(roi_size=15), slices=slice(None))
    assert rois.geometry["layout"] == ("up-down mirrored" if mirrored else "up-down")
    assert rois.geometry["split_position"] == 256
    batch = rois.batch
    assert batch.data.shape == (42, 2, 7, 15, 15)
    truth_main, truth_second = np.concatenate(mains), np.concatenate(seconds)
    main_xy, second_xy = batch.roi_xy[:, 0].double().numpy(), batch.roi_xy[:, 1].double().numpy()
    nearest = np.abs(main_xy[:, None] - truth_main[None]).max(-1).argmin(1)
    assert len(set(nearest)) == 42                                         # every true bead, once
    assert np.abs(main_xy - truth_main[nearest]).max() < 1.0          # each ROI on its bead's pixel
    assert np.abs(second_xy - truth_second[nearest]).max() < 1.0      # and on its partner's
    assert np.abs(_apply(rois.transform, truth_second) - truth_main).max() < 0.3
    assert rois.transform[2, 2] == 1.0
    assert torch.equal(batch.field_xy, batch.roi_xy[:, 0])


def test_aberrations_come_back_through_smappys_reading_and_uipsfs_learning():
    """End to end: beads drawn by uipsf's model into ADU stacks on a chip, read, converted and cut by smappy,
    learnt by `calibrate_beads`."""
    import uipsf
    from uipsf.simulate import parameters, template
    from uipsf.workflows import BeadSettings, bead_model, calibrate_beads
    optics = uipsf.Optics(NA=1.43, wavelength_nm=660, n_imm=1.518, n_med=1.33, pixel_size_nm=(110.0, 110.0))
    roi, K, n = 15, 21, 8
    model = bead_model(optics, BeadSettings(zernike_order=4, magnitude="fixed"), (roi, roi), z_max_nm=1000,
                       device="cpu")
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    phase[:4] = torch.tensor([0.6, -0.3, 0.2, 0.15])
    centres = [(30 + 40 * (i % 4), 30 + 40 * (i // 4)) for i in range(n)]       # whole pixels, 40 px apart
    rng = np.random.default_rng(0)
    xyz = torch.tensor(np.c_[rng.uniform(-50, 50, (n, 2)), rng.uniform(-100, 100, n)], dtype=torch.float32)
    objective = np.arange(K) * 100.0
    b0 = template(n, uipsf.Batch.z_from_objective(torch.tensor(objective - objective.mean(), dtype=torch.float32)),
                  (roi, roi), device="cpu")
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([40.0, 40.0]), "emitter.xyz": xyz,
              "emitter.photons": torch.full((n,), 30000.0), "emitter.background": torch.zeros(n)}
    with torch.no_grad():
        mu = model(parameters(model, b0, values, device="cpu").physical(), b0)[:, 0].numpy()   # [n, K, Y, X]
    image = np.full((K, 110, 190), 20.0)
    h = roi // 2
    for (x, y), spot in zip(centres, mu, strict=True):
        image[:, y - h:y + h + 1, x - h:x + h + 1] += spot
    stack = BeadStack(_adu(rng.poisson(image).astype(float)), objective, "sim", {}, (0, 0, 190, 110))
    rois = single_rois([stack], CAMERA, CalibrationSettings(roi_size=roi, min_distance_px=20),
                       slices=slice(None))
    found = np.array(sorted((r["x_px"], r["y_px"]) for r in rois.records))
    assert len(found) == n and np.abs(found - np.array(sorted(centres))).max() <= 1   # an aberrated spot
    cal = calibrate_beads(rois.batch, optics, BeadSettings(zernike_order=4), noise=rois.noise, device="cpu")
    assert torch.allclose(cal.shared()["pupil.phase"][:4].cpu(), phase[:4], atol=0.02)
