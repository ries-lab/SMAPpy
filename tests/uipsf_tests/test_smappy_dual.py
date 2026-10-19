"""A two-channel uipsf model as SMAPpy's dual calibration, fitted by SMAPpy's global fitter (plan §6.4, §7.1).

What SMAPpy will do with a two-channel model file, written out here with SMAPpy's own functions, so
that the file is known to carry everything its global fitter (`dualfit.DualChannelEngine`) needs: each
channel sampled on one z grid (`LearnedPSF.psf(…, channel=c)`), normalised jointly by
`positive_pair_models`, the transform and the split's geometry from the file, saved and loaded as a
`DualColorCalibration`.  Split frames drawn from uipsf's own forward model are then fitted: position,
z, photons and the colour ratio come back.  Skips without SMAPpy.
"""
import numpy as np
import pytest
import torch

import uipsf
from uipsf.model import Channel
from uipsf.model.channels import inverse_map

pytest.importorskip("smappy")

PIXEL = 110.0
SHAPE = (96, 48)                    # (height, width): the upper half main, the lower half mirrored
SPLIT = 48
ANGLE = np.deg2rad(0.5)
# secondary → main chip pixels: y mirrored across the split, a small turn and shift
TRANSFORM = (np.array([[np.cos(ANGLE), -np.sin(ANGLE), 0.6], [np.sin(ANGLE), np.cos(ANGLE), -0.4], [0, 0, 1]])
             @ np.array([[1.0, 0, 0], [0, -1, 95], [0, 0, 1]]))
GEOMETRY = {"layout": "up-down mirrored", "main_channel": "upper", "split_position": SPLIT,
            "image_shape": list(SHAPE), "coordinate_system": "camera-chip", "mirror_axis_xy": 1,
            "sources": [{"path": None, "axes": None, "roi": [0, 0, SHAPE[1], SHAPE[0]], "shape": [1, *SHAPE],
                         "z_nm": [0.0]}]}
SHARE, DZ = 0.3, 40.0


def _model():
    """A ratiometric pair with its own aberrations per channel, the parameters known."""
    channels = []
    for c, wavelength in enumerate((670.0, 700.0)):
        optics = uipsf.Optics(NA=1.43, wavelength_nm=wavelength, n_imm=1.516, n_med=1.335,
                              pixel_size_nm=(PIXEL, PIXEL))
        grid = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)    # float64: not on MPS
        psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), (21, 21), upsample=3)
        channels.append(Channel(psf, transform=TRANSFORM if c else None))
    model = uipsf.ForwardModel(channels)
    shared = {"pupil.phase": torch.zeros(len(channels[0].psf.pupil.phase_modes), dtype=torch.float64),
              "ch1.pupil.phase": torch.zeros(len(channels[1].psf.pupil.phase_modes), dtype=torch.float64)}
    shared["pupil.phase"][1] = 0.9                                    # astigmatism, for z
    shared["ch1.pupil.phase"][1] = -0.8
    shared["ch1.pupil.phase"][2] = 0.15                               # and some coma in the secondary
    for pre in ("", "ch1."):
        shared[pre + "pupil.magnitude"] = torch.zeros(len(channels[0].psf.pupil.magnitude_modes), dtype=torch.float64)
        shared[pre + "psf.sigma"] = torch.tensor([30.0, 30.0], dtype=torch.float64)
    shared["ch1.transform"] = torch.tensor((TRANSFORM / TRANSFORM[2, 2]).ravel()[:8])
    shared["ch1.share"] = torch.tensor(SHARE, dtype=torch.float64)
    shared["ch1.dz"] = torch.tensor(DZ, dtype=torch.float64)
    return model, shared


def _dual_calibration(model, shared, tmp_path):
    """SMAPpy's `DualColorCalibration` from the model, the way SMAPpy will make it."""
    from smappy.calibrate.core import CalibrationSettings, positive_pair_models
    from smappy.calibrate.dual import DualColorCalibration, load_dual_color_calibration, save_dual_color_calibration

    from uipsf.export import sample_psf
    dz, half, size = 20.0, 40, 21
    z = -(np.arange(2 * half + 1) - half) * dz                          # SMAPpy: z = −(k − z0)·dz
    xyz = np.c_[np.zeros((len(z), 2)), z]
    raw = np.stack([sample_psf(model, shared, xyz, roi_px=size, upsample=4, channel=c).double().numpy()
                    for c in range(2)])
    settings = CalibrationSettings(roi_size=size, smooth_z_nm=0.0, smooth_xy_px=0.0)
    (main, secondary), _ = positive_pair_models(raw, dz, settings)
    H = np.asarray(TRANSFORM) / TRANSFORM[2, 2]
    cal = DualColorCalibration(main, secondary, H, GEOMETRY, {"method": "uipsf2_pupil"})
    path = tmp_path / "pair_3dcal.h5"
    save_dual_color_calibration(path, cal)
    return load_dual_color_calibration(path)                           # SMAPpy's own checks of the format


def _frames(model, shared, emitters, n, rng, background=20.0):
    """Split frames drawn from uipsf's forward model: each emitter in both halves where the transform puts
    it, its photons split by the shares."""
    from uipsf.export import sample_psf
    H = torch.tensor(TRANSFORM / TRANSFORM[2, 2])
    mean = np.full(SHAPE, background)
    for x, y, z, photons in emitters:
        second = inverse_map(H, torch.tensor([[x, y]], dtype=torch.float64))[0].numpy()
        for c, (cx, cy) in enumerate(((x, y), tuple(second))):
            ix, iy = int(round(cx)), int(round(cy))
            local = [[(cx - ix) * PIXEL, (cy - iy) * PIXEL, z]]
            patch = photons * sample_psf(model, shared, local, roi_px=21, upsample=4, channel=c)[0].double().numpy()
            mean[iy - 10:iy + 11, ix - 10:ix + 11] += patch
    return rng.poisson(np.broadcast_to(mean, (n, *SHAPE))).astype(np.float32)


def test_smappys_global_fitter_reads_a_two_channel_model_as_its_own_dual_calibration(tmp_path):
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
    from smappy.dualfit import LINK_XYZ, DualChannelEngine
    from smappy.locs import in_pixels
    from smappy.metadata import CameraMetadata
    from smappy.pipeline import FitSettings
    from smappy.psf import GlobalSplinePSF
    model, shared = _model()
    cal = _dual_calibration(model, shared, tmp_path)
    # The joint normalisation hands back the shares: what SMAPpy's fitter weighs the channels by.
    assert cal.secondary.parameters["photon_normalization"] == pytest.approx(SHARE, abs=0.01)
    # SMAPpy's photons are the light its spline's box holds around focus (`positive_pair_models`), as for
    # its own bead calibrations: the emitter's total times the box's share of it.
    in_box = cal.main.parameters["common_normalization"]
    emitters = [(14.0, 15.0, 150.0, 20000.0), (31.0, 30.0, -250.0, 20000.0)]
    n = 20
    frames = _frames(model, shared, emitters, n, np.random.default_rng(0))
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=PIXEL / 1000, em_on=False)
    engine = DualChannelEngine(camera, PeakFinder(DoGFilter(1.2), AbsoluteCutoff(20.0)),
                               GlobalSplinePSF((cal.main, cal.secondary), LINK_XYZ), cal, FitSettings(roisize=13))
    engine.push(frames)
    locs = engine.flush()
    assert engine.stats["matched"] >= 0.9 * 2 * n           # both halves detected, paired by the transform
    x, y = in_pixels(locs, "x_pix", "y_pix")
    for x0, y0, z0, photons in emitters:
        mine = np.hypot(x - x0, y - y0) < 3
        assert mine.sum() >= n
        assert np.median(x[mine]) == pytest.approx(x0, abs=0.1)
        assert np.median(y[mine]) == pytest.approx(y0, abs=0.1)
        assert np.median(locs["z_nm"][mine]) == pytest.approx(z0, abs=15.0)
        assert np.median(locs["ratio"][mine]) == pytest.approx(SHARE, abs=0.03)
        assert np.median(locs["photons"][mine]) == pytest.approx(photons * in_box, rel=0.03)
