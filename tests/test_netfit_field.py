"""netfit with a field-dependent PSF (`netfit.field`, docs/decode/fd-plan.md): spots drawn with the node
splines blended where they are on the chip, as Spline 3D's `CSplineField` fits them, and u, v maps of where
each pixel is on the chip -- the same at inference for any camera ROI."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smappy import _fit3d
from smappy.calibrate.core import spline_coefficients
from smappy.field_splines import FieldSplines, bilinear, blend
from smappy.io.calibration import SplineCalibration, save_spline_calibration
from smappy.netfit import network, simulate
from smappy.netfit.field import FieldRecord, resolve
from smappy.netfit.render import SplineRenderer
from smappy.netfit.settings import CameraModel, SimulationSettings, TrainingSettings
from smappy.simulate.camera import _draw_spline

NZ, BOX = 41, 15
XS, YS = np.array([100.0, 300.0, 500.0]), np.array([50.0, 250.0])


def astigmatic(e: float) -> np.ndarray:
    """A Gaussian stack whose focal lines part with ``e``: astigmatism that changes over the field."""
    z = np.linspace(-1, 1, NZ)[:, None, None]
    y, x = np.mgrid[:BOX, :BOX] - (BOX - 1) / 2
    sx, sy = 1.3 + 0.8 * (z - e) ** 2, 1.3 + 0.8 * (z + e) ** 2
    return (np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)).astype(np.float32)


def calibration() -> SplineCalibration:
    """Node splines over (100..500, 50..250) chip pixels, astigmatism growing along x and y; the centre's
    spline as ``coeff``."""
    e = lambda x, y: 0.05 + 0.5 * (x - 100) / 400 + 0.2 * (y - 50) / 200
    field = FieldSplines(np.stack([spline_coefficients(astigmatic(e(x, y))) for y in YS for x in XS]), XS, YS)
    return SplineCalibration(blend(field, (300.0, 150.0)), 50.0, NZ // 2, x0=(BOX - 1) / 2, em_on=False,
                             field_splines=field)


def t(a, d=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=d)


def test_the_renderers_bilinear_nodes_and_weights_are_the_fitters():
    cal = calibration()
    r = SplineRenderer(cal, field=True)
    rng = np.random.default_rng(0)
    # inside, on nodes, beyond every edge
    chip = np.r_[rng.uniform(0, 650, (200, 2)), [[100, 50], [500, 250], [300, 250], [-40, 900]]]
    nodes, w = bilinear(cal.field_splines, chip)
    tn, tw = r.bilinear(t(chip))
    # the same weight on each node: compare the node-by-node sums, the order of equal-weight terms aside
    dense = lambda nn, ww: np.stack([np.bincount(a, b, minlength=len(XS) * len(YS)) for a, b in zip(nn, ww)])
    assert np.allclose(dense(tn.numpy(), tw.numpy()), dense(nodes, w), atol=1e-6)


def test_a_field_spot_is_the_blended_spline_drawn_as_smappys_simulation_draws_it():
    cal = calibration()
    rng = np.random.default_rng(1)
    n = 60
    x, y = rng.uniform(4, 28, n), rng.uniform(4, 28, n)
    z, photons = rng.uniform(-900, 900, n), rng.uniform(100, 1000, n)
    chip = rng.uniform(0, 650, (n, 2))
    out = torch.zeros((n, 32, 32))
    SplineRenderer(cal, field=True).draw(out, t(np.arange(n), torch.long), t(x), t(y), t(z), t(photons),
                                         chip=t(chip))
    ref = np.zeros((n, 32, 32))
    for k in range(n):     # each spot with the spline blended at its place, through SMAPpy's own simulation
        one = SplineCalibration(blend(cal.field_splines, chip[k]), cal.dz, cal.z0, x0=cal.x0)
        _draw_spline(ref[k:k + 1], np.zeros(1, int), np.c_[x[k], y[k]], z[k:k + 1], photons[k:k + 1], one)
    assert np.abs(out.numpy() - ref).max() < 2e-5 * np.abs(ref).max()
    # without a chip position, or without the field, the centre's spline as before
    plain, centre = torch.zeros((n, 32, 32)), torch.zeros((n, 32, 32))
    SplineRenderer(cal, field=True).draw(plain, t(np.arange(n), torch.long), t(x), t(y), t(z), t(photons))
    SplineRenderer(cal).draw(centre, t(np.arange(n), torch.long), t(x), t(y), t(z), t(photons), chip=t(chip))
    assert torch.equal(plain, centre)
    assert np.abs(plain.numpy() - ref).max() > 0.05 * np.abs(ref).max()


def test_spline_3d_with_the_node_splines_fits_a_noise_free_field_spot_where_it_was_drawn():
    """The renderer's field spot is the model `CSplineField` fits: noise-free, the fit returns the truth
    exactly, and the centre's spline does not."""
    cal = calibration()
    r = SplineRenderer(cal, field=True)
    rng = np.random.default_rng(2)
    n = 40
    x, y = 20 + rng.uniform(-0.5, 0.5, n), 20 + rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(-600, 600, n)
    chip = np.c_[rng.choice([110.0, 490.0], n), rng.choice([60.0, 240.0], n)]         # the corners
    spots, ox, oy = r.spots(t(x), t(y), t(z), chip=t(chip))
    rois = (5000.0 * spots + 10.0).numpy().astype(np.float32)
    nodes, w = bilinear(cal.field_splines, chip)
    z_start = float(cal.z_nm_to_index(0.0))
    fit = _fit3d.fit_cspline_field(rois, cal.field_splines.coeff, nodes, w, z_start, 100, 1)[0]
    centre = _fit3d.fit_cspline(rois, cal.coeff, z_start, 100, 1)[0]
    true = np.c_[x - ox.numpy(), y - oy.numpy(), cal.z_nm_to_index(z)]
    assert np.abs(fit[:, [0, 1, 4]] - true).max() < 2e-3
    assert np.abs(fit[:, 2] / 5000.0 - 1).max() < 1e-3
    assert np.abs(centre[:, 4] - true[:, 2]).max() * cal.dz > 20.0          # nm: the field matters here


def test_the_maps_are_u_and_v_of_each_pixels_chip_place_under_a_camera_roi():
    rec = FieldRecord.from_splines(calibration().field_splines, (110.0, 120.0))
    assert rec.centre_px == (300.0, 150.0) and rec.grid == (2, 3)
    whole = rec.maps((0, 0), 400, 700)
    roi = rec.maps((230, 90), 48, 64)
    assert np.array_equal(roi, whole[:, 90:138, 230:294])
    # u = (chip x - centre) pixel / 50 um, v likewise in rows; clipped to the nodes' span
    assert whole[0, 0, 300] == 0 and whole[0, 0, 400] == pytest.approx(100 * 110 / 50000)
    assert whole[1, 250, 0] == pytest.approx(100 * 120 / 50000)
    assert whole[0, 0, 650] == whole[0, 0, 500] and whole[1, 10, 0] == whole[1, 50, 0]
    corners = torch.tensor([[230.0, 90.0], [0.0, 0.0]])
    assert np.allclose(rec.maps_torch(corners, 32)[0].numpy(), whole[:, 90:122, 230:262])
    # a crop's corners: its centre over the span widened by half a crop each side
    g = torch.Generator().manual_seed(0)
    c = rec.corners(5000, 32, g, "cpu").numpy() + 15.5
    assert c[:, 0].min() == pytest.approx(84, abs=1) and c[:, 0].max() == pytest.approx(516, abs=1)
    assert c[:, 1].min() == pytest.approx(34, abs=1) and c[:, 1].max() == pytest.approx(266, abs=1)
    assert rec.beyond((100, 50), 200, 400) == 0 and rec.beyond((480, 50), 100, 100) == 79
    assert FieldRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec


def test_a_field_batch_draws_each_movie_at_its_own_place_with_its_maps():
    cal = calibration()
    rec = FieldRecord.from_splines(cal.field_splines, (100.0, 100.0))
    sim = SimulationSettings(frame_px=24, density_per_40px=10.0, photons_mean=3000.0, photons_sd=0.0)
    b = simulate.draw_batch(SplineRenderer(cal, field=True), sim, CameraModel(pixel_nm=(100.0, 100.0)), 8,
                            torch.Generator().manual_seed(3), field=rec)
    assert b.aux.shape == (8, 2, 24, 24) and b.origin.shape == (8, 2)
    for k in range(8):
        assert np.allclose(b.aux[k].numpy(), rec.maps(b.origin[k].numpy(), 24, 24))
    assert len(torch.unique(b.origin[:, 0])) > 1


def test_whether_a_training_draws_the_field_follows_the_calibration_on_auto():
    cal = calibration()
    plain = SplineCalibration(cal.coeff, cal.dz, cal.z0, x0=cal.x0)
    assert resolve("auto", cal) and not resolve("auto", plain) and not resolve("off", cal)
    with pytest.raises(ValueError, match="no field"):
        resolve("on", plain)


def test_a_one_channel_network_with_maps_reads_them_and_a_model_without_is_built_as_before():
    s = TrainingSettings()
    s.parts.backbone.settings["features"] = 8
    torch.manual_seed(0)
    plain = network.build(s.parts, 3)
    torch.manual_seed(0)
    fd = network.build(s.parts, 3, field=True)
    assert plain.aux == 0 and fd.aux == 2
    assert network.parameters(fd) > network.parameters(plain)
    for head in fd.heads.values():             # the heads start at zero output, whatever they read
        torch.nn.init.normal_(head.net[-1].weight, std=0.1)
    x = torch.randn(2, 3, 16, 16)
    maps = torch.zeros(2, 2, 16, 16)
    a = fd(x, aux=maps)
    b = fd(x, aux=maps + 0.5)
    assert a.shape == plain(x).shape and not torch.allclose(a, b)
    with pytest.raises(ValueError, match="input maps"):
        fd(x)


def test_a_field_training_records_the_field_and_its_model_reads_maps_made_from_the_camera_roi(tmp_path):
    from smappy.netfit import train
    from smappy.netfit.predict import Predictor
    from smappy.training import model_file
    from smappy.training.predictors import field_dependent
    cal_path = tmp_path / "fd_3dcal.h5"
    save_spline_calibration(cal_path, calibration())
    s = TrainingSettings(calibration=str(cal_path))
    s.camera = CameraModel(em_gain=0.0, pixel_nm=(100.0, 100.0))
    s.simulation = SimulationSettings(frame_px=16, density_per_40px=15.0, field="on")
    s.parts.backbone.settings["features"] = 8
    s.parts.schedule.settings.update(samples_per_epoch=32, batch=16, test_samples=16, epochs=1,
                                      precision="fp32", compile=False, snapshots=[])
    assert train.train(tmp_path / "run", s, "cpu", epochs=1, log=lambda *a: None) == "finished"
    meta = model_file.read_metadata(tmp_path / "run" / "model.safetensors")
    assert field_dependent(meta) and meta["calibration_field"]
    assert meta["field"]["centre_px"] == [300.0, 150.0] and meta["field"]["pixel_nm"] == [100.0, 100.0]
    rng = np.random.default_rng(4)
    frames = rng.poisson(60, (5, 19, 22)).astype(np.float32)
    at = lambda corner: Predictor(tmp_path / "run" / "model.safetensors", "cpu", threshold=0.0, corner=corner)
    near, far = at((100, 50))(frames), at((480, 230))(frames)
    assert len(near["x_pix"]) and not np.allclose(near["z_nm"][:20], far["z_nm"][:20])
    with pytest.raises(ValueError, match="EM register"):
        Predictor(tmp_path / "run" / "model.safetensors", "cpu", frames_mirrored=True)
    # the same calibration with the field off: no maps, but the record says the calibration has a field
    s.simulation = SimulationSettings(frame_px=16, density_per_40px=15.0, field="off")
    train.train(tmp_path / "centre", s, "cpu", epochs=1, log=lambda *a: None)
    meta = model_file.read_metadata(tmp_path / "centre" / "model.safetensors")
    assert not field_dependent(meta) and meta["calibration_field"]
    from smappy.metadata import CameraMetadata
    from smappy.plugins.fit_decode import field_warnings
    assert "centre PSF everywhere" in field_warnings(meta, CameraMetadata())[0]
    fd_meta = model_file.read_metadata(tmp_path / "run" / "model.safetensors")
    assert "not known" in field_warnings(fd_meta, CameraMetadata())[0]
    assert "beyond the beads" in field_warnings(fd_meta, CameraMetadata(roi=(450, 100, 200, 100)))[0]
    assert field_warnings(fd_meta, CameraMetadata(roi=(150, 60, 100, 100))) == []
