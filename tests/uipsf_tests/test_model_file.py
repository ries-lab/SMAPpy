import h5py
import numpy as np
import pytest
import torch

import uipsf
from uipsf.export import build, describe, load_model, sample_psf, save_model
from uipsf.model import PerSliceFactor, Planar
from uipsf.model.geometry import BeadStackGeometry
from uipsf.workflows import learn_beads
from uipsf_tests.conftest import small_beads


@pytest.fixture(scope="module")
def calibration():
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 100))
    model, _, batch = small_beads(optics, n=8, dtype=torch.float32)
    return learn_beads(model, batch, heldout=0.25, first_order=None), batch


def test_a_saved_model_file_loads_to_the_same_psf_and_record(calibration, tmp_path):
    cal, _ = calibration
    path = tmp_path / "psf.uipsf.h5"
    save_model(cal, path)
    with h5py.File(path) as f:
        assert f.attrs["format"] == "uipsf-model" and f.attrs["version"] == 2
    m = load_model(path)
    for n, v in cal.shared().items():
        assert torch.equal(m.shared[n].cpu(), v.detach().cpu()), n
    xyz = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 250.0], [-50.0, 10.0, -400.0]])
    assert torch.allclose(m.psf(xyz), cal.psf(xyz), atol=1e-7)
    phase = m.zernikes("phase")
    assert [j for j, _, _ in phase] == cal.model.psf.pupil.phase_modes
    std = cal.result.std["pupil.phase"].double().numpy()
    assert np.allclose([s for _, _, s in phase], std)
    assert m.fit["stop_reason"] == cal.result.stop_reason and m.fit["iterations"] == cal.result.iterations
    assert [b["state"] for b in m.beads] == [r.state for r in cal.beads.rows]
    assert m.diagnostics["atlas"]["mean_deviance"] == pytest.approx(cal.atlas.mean_deviance)
    assert m.diagnostics["heldout"]["summary"]["beads"] == 2
    assert m.emitters["emitter.xyz"].shape == (len(cal.beads.used), 3)


def test_the_description_builds_the_same_model_whatever_its_components(calibration):
    cal, batch = calibration
    psf = cal.model.psf
    for model in (cal.model,
                  uipsf.ForwardModel(psf, BeadStackGeometry(drift=True, z_scale=True), PerSliceFactor(2, 0.1),
                                     Planar(psf.grid.optics.pixel_size_nm))):
        again = build(describe(model), device=batch.data.device)
        assert describe(again) == describe(model)
        assert [s.name for s in again.specs(batch)] == [s.name for s in model.specs(batch)]
        p = uipsf.Params(model.specs(batch), batch.n_emitters, values=cal.shared()).physical()
        assert torch.allclose(again(p, batch), model(p, batch), rtol=1e-5)


def test_the_sampled_psf_is_the_models_and_integrates_to_one_photon_in_focus(calibration):
    cal, _ = calibration
    model, shared = cal.model, cal.shared()
    xyz = torch.tensor([[12.0, -7.0, 100.0], [0.0, 0.0, -200.0]])
    own = model.psf(shared, xyz[None])[0]
    assert torch.allclose(cal.psf(xyz), own, atol=1e-8)
    # A wide region holds (nearly) all the light: photons stay physical (psf.py, normalisation).
    # The test model's 24-sample pupil aliases a region this wide; the same Zernikes on 64 samples do not.
    config = describe(model)
    config["channels"][0]["pupil_size"] = 64
    wide = sample_psf(build(config), shared, torch.zeros(1, 3), roi_px=61)[0]
    assert 0.9 < float(wide.sum()) <= 1.0
    # x along the last axis, y along the one before (docs/uipsf/conventions.md): one pixel is one column.
    px, py = model.pixel_size_nm
    a, b = sample_psf(build(config), shared, torch.tensor([[0.0, 0.0, 0.0], [px, py, 0.0]]), roi_px=15)
    assert (a[:-1, :-1] - b[1:, 1:]).abs().max() < 3e-3 * a.max()


def test_sampling_at_more_points_per_pixel_converges(calibration):
    cal, _ = calibration
    xyz = torch.tensor([[20.0, 10.0, 150.0]])
    fine = cal.psf(xyz, upsample=6)
    assert (cal.psf(xyz, upsample=3) - fine).abs().max() < (cal.psf(xyz, upsample=1) - fine).abs().max()


def test_a_bead_is_left_out_of_the_sampled_psf_unless_asked(calibration):
    cal, _ = calibration
    import copy
    beaded = copy.copy(cal)
    beaded.model = uipsf.ForwardModel(copy.copy(cal.model.psf))
    beaded.model.psf.bead_radius_nm = 100.0
    z = torch.linspace(-400, 400, 17)
    xyz = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1)
    assert torch.allclose(beaded.psf(xyz), cal.psf(xyz))
    big = beaded.psf(xyz, include_bead=True)
    assert big[8].max() < cal.psf(xyz)[8].max()          # a 200 nm bead blurs the focus
    with pytest.raises(ValueError, match="equal z steps"):
        sample_psf(beaded.model, cal.shared(), xyz[[0, 1, 3]], include_bead=True)


def test_a_file_of_another_kind_or_a_newer_version_is_refused(tmp_path):
    with h5py.File(tmp_path / "a.h5", "w") as f:
        f.attrs["format"] = "something"
    with pytest.raises(ValueError, match="not a uipsf model"):
        load_model(tmp_path / "a.h5")
    with h5py.File(tmp_path / "b.h5", "w") as f:
        f.attrs.update(format="uipsf-model", version=99)
    with pytest.raises(ValueError, match="version 99"):
        load_model(tmp_path / "b.h5")


def test_a_version_1_file_still_loads(calibration, tmp_path):
    # Version 1 (before M4) had one channel, its optics, pupil and PSF at the top of the description.
    import json
    cal, _ = calibration
    path = tmp_path / "v1.uipsf.h5"
    save_model(cal, path)
    with h5py.File(path, "r+") as f:
        config = json.loads(f["model"].attrs["config"])
        old = dict(config["channels"][0], **{k: v for k, v in config.items() if k != "channels"})
        for k in ("name", "transform", "share", "dz_nm", "fit_transform"):
            old.pop(k)
        f["model"].attrs["config"] = json.dumps(old)
        f.attrs["version"] = 1
    m = load_model(path)
    xyz = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 250.0]])
    assert torch.allclose(m.psf(xyz), cal.psf(xyz), atol=1e-7)
    assert [j for j, _, _ in m.zernikes("phase")] == cal.model.psf.pupil.phase_modes


def test_a_two_channel_model_file_keeps_each_channels_psf_transform_and_share(optics, tmp_path):
    from uipsf_tests.conftest import two_channel_beads
    model, _, batch = two_channel_beads(optics, n=6, dtype=torch.float32, mirrored=True)
    geometry = {"layout": "up-down mirrored", "main_channel": "upper", "split_position": 256,
                "mirror_axis_xy": 1, "image_shape": [512, 512], "coordinate_system": "camera-chip"}
    cal = learn_beads(model, batch, first_order=None, geometry=geometry)
    path = tmp_path / "pair.uipsf.h5"
    save_model(cal, path)
    m = load_model(path)
    p = cal.shared()
    assert m.n_channels == 2 and sum(m.shares()) == pytest.approx(1.0)
    assert m.geometry == geometry                     # SMAPpy's DualColorCalibration.geometry
    assert m.shares()[1] == pytest.approx(float(p["ch1.share"]))
    assert m.dz_nm(1) == pytest.approx(float(p["ch1.dz"]))
    H = m.transform(1)
    assert H[2, 2] == 1.0 and np.allclose(H.ravel()[:8], p["ch1.transform"].double().numpy())
    assert [j for j, _, _ in m.zernikes("phase", channel=1)] == model.channels[1].psf.pupil.phase_modes
    xyz = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 250.0]])
    for c in (0, 1):
        assert torch.allclose(m.psf(xyz, channel=c), cal.psf(xyz, channel=c), atol=1e-7)
    # Channel 1 is sampled at the main channel's z: its own PSF at z + δz, times its share.
    psf1 = model.channels[1].psf
    own = psf1(uipsf.model.channels.view(p, 1), (xyz + torch.tensor([0.0, 0.0, float(p["ch1.dz"])]))[None])[0]
    assert torch.allclose(m.psf(xyz, channel=1), float(p["ch1.share"]) * own, atol=1e-7)
