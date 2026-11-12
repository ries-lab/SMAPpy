import h5py
import numpy as np
import pytest
import torch

import uipsf
from uipsf.export import build, describe, load_model, sample_psf, save_model
from uipsf.export.model_file import VERSION
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
        assert f.attrs["format"] == "uipsf-model" and f.attrs["version"] == VERSION
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
                  uipsf.ForwardModel(psf, BeadStackGeometry(drift="free", drift_axes="xyz", z_scale=True),
                                     PerSliceFactor(2, 0.1),
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
    geometry = {"layout": "up-down mirrored", "main_channel": "upper", "image_shape": [512, 512],
                "coordinate_system": "camera-chip",
                "channels": {"name": "", "regions": [[0, 0, 512, 256], [0, 256, 512, 256]],
                             "flips": [[False, False], [False, True]]}}
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


def test_a_model_written_into_a_group_of_another_file_loads_the_same_and_leaves_the_rest(calibration, tmp_path):
    # how SMAPpy keeps a calibration's model inside its spline file (`pupil.save`)
    from uipsf.export.model_file import EMBEDDED, has_model
    cal, _ = calibration
    path = tmp_path / "beads_3dcal.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "smappy-spline-calibration"
        f["coeff"] = np.arange(4.0)
    assert not has_model(path)
    save_model(cal, path, group=EMBEDDED)
    save_model(cal, path, group=EMBEDDED)                     # written again: replaced, not refused
    with h5py.File(path) as f:
        assert f.attrs["format"] == "smappy-spline-calibration" and np.array_equal(f["coeff"], np.arange(4.0))
        assert f[EMBEDDED].attrs["format"] == "uipsf-model"
    assert has_model(path) and not has_model(tmp_path / "missing.h5")
    xyz = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 250.0]])
    assert torch.allclose(load_model(path).psf(xyz), cal.psf(xyz), atol=1e-7)


def test_a_calibration_saved_with_its_model_is_found_by_the_focus_and_by_start_from(calibration, tmp_path):
    from smappy.calibrate import depth
    from smappy.calibrate.pupil import carries_model, load_start, model_path, save, spline_calibration
    from smappy.io.calibration import load_spline_calibration
    cal, _ = calibration
    path = save(tmp_path / "beads_3dcal.h5", cal, spline_calibration(cal, 50.0, 400.0, roi_size=15))
    assert carries_model(path) and not model_path(path).exists()
    loaded = load_spline_calibration(path)
    assert depth.model_file(loaded) == path and depth.has_model(loaded)
    xyz = torch.tensor([[0.0, 0.0, 100.0]])
    assert torch.allclose(depth.model_of(loaded).psf(xyz.double()).float(), cal.psf(xyz), atol=1e-6)
    assert torch.allclose(load_start(path).psf(xyz), cal.psf(xyz), atol=1e-7)


def test_an_older_calibration_with_its_model_file_beside_it_is_still_read(calibration, tmp_path):
    from smappy.calibrate import depth
    from smappy.calibrate.pupil import load_start, model_path, spline_calibration
    from smappy.io.calibration import load_spline_calibration, save_spline_calibration
    cal, _ = calibration
    spline = spline_calibration(cal, 50.0, 400.0, roi_size=15)
    path = tmp_path / "old_3dcal.h5"
    spline.parameters["model_file"] = model_path(path).name           # as saved until November 2026
    save_spline_calibration(path, spline)
    save_model(cal, model_path(path))
    loaded = load_spline_calibration(path)
    assert depth.model_file(loaded) == model_path(path)
    xyz = torch.tensor([[0.0, 0.0, 100.0]])
    assert torch.allclose(load_start(path).psf(xyz), cal.psf(xyz), atol=1e-7)
    model_path(path).unlink()
    assert depth.model_file(loaded) is None and not depth.has_model(loaded)
