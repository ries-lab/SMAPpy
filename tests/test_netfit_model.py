"""netfit's network, loss, extraction and predictor: shapes, the loss's terms against their
definitions, detections where the maps put them, a model file read back."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smappy.netfit import extract, loss, network
from smappy.netfit.network import Scaling
from smappy.netfit.simulate import Batch

SCALING = Scaling(input_offset=100.0, input_scale=100.0, phot_max=10000.0, z_max=1000.0, bg_max=200.0)


def _maps(h=8, w=8):
    out = torch.zeros(1, 10, h, w)
    out[:, 0] = 1e-4
    out[:, 5:9] = 0.5
    return out


@pytest.mark.parametrize("style", [{}, {"upsample_kernel": 1, "pool_act": False, "out_conv": True, "init": "kaiming", "p_clamp": 8.0}])
def test_the_network_gives_ten_maps_the_size_of_its_frames(style):
    model = network.TwoStageUNet(features=8, **style)
    assert model(torch.zeros(2, 3, 16, 24)).shape == (2, 10, 16, 24)


def test_on_a_finer_grid_the_maps_have_two_cells_a_pixel_and_their_centres_a_quarter_pixel_off():
    model = network.TwoStageUNet(features=8, out_scale=2)
    assert model(torch.zeros(2, 3, 16, 24)).shape == (2, 10, 32, 48)
    assert network.centres(4, 2).tolist() == [-0.25, 0.25, 0.75, 1.25]
    # p starts at the same molecules a pixel, spread over four cells
    p = model(torch.zeros(1, 3, 8, 8))[0, 0]
    assert p.sum().item() == pytest.approx(network.P0 * 64, rel=1e-3)


def test_on_a_finer_grid_a_detection_takes_its_cells_centre_and_the_p_of_its_square():
    out = torch.zeros(1, 10, 16, 16)
    out[:, 5:9] = 0.5
    out[0, 0, 7, 10], out[0, 0, 8, 11] = 0.45, 0.4   # a molecule at a cell's corner: p on a diagonal
    out[0, 2, 7, 10] = 0.1                          # 0.1 pixel right of the cell's centre
    e = extract.extract(out, SCALING, 0.5, scale=2)
    assert len(e["frame"]) == 1
    assert e["x"].item() == pytest.approx(10 / 2 + 0.25 - 0.5 + 0.1)
    assert e["y"].item() == pytest.approx(7 / 2 + 0.25 - 0.5)
    assert e["prob"].item() == pytest.approx(0.85)
    # DECODE's plus misses the diagonal: the same p is no detection
    assert len(extract.extract(out, SCALING, 0.5, scale=2, summed_over="plus")["frame"]) == 0


def test_the_mixture_term_is_the_molecules_likelihood_under_the_pixels_mixture():
    out = _maps()
    out[0, 0, 3, 5] = 0.9                       # one confident pixel: column 5, row 3
    out[0, 1, 3, 5] = 0.2                       # 2000 photons
    out[0, 2, 3, 5], out[0, 3, 3, 5], out[0, 4, 3, 5] = 0.1, -0.2, 0.3
    batch = Batch(frames=None, xyz=torch.tensor([[[5.1, 2.8, 300.0]]]), photons=torch.tensor([[2000.0]]),
                  mask=torch.tensor([[True]]), background=torch.tensor([0.0]))
    t = loss.terms(out, batch, SCALING)
    # the confident pixel's component, at its mean: weight p / sum p, a Gaussian of sigma 0.5 in four dims
    p = out[0, 0].flatten()
    w = 0.9 / p.sum()
    expected = -(math.log(w) - 2 * math.log(2 * math.pi) - 4 * math.log(0.5))
    assert t["mixture"].item() == pytest.approx(expected, rel=1e-3)
    mean, var = p.sum(), (p * (1 - p)).sum()
    count = 1 * (0.5 * (1 - mean) ** 2 / var + 0.5 * torch.log(2 * math.pi * var))
    assert t["count"].item() == pytest.approx(count.item(), rel=1e-5)


def test_a_detection_is_where_p_peaks_and_takes_its_pixels_position():
    out = _maps()
    out[0, 0, 3, 5] = 0.7
    out[0, 0, 3, 6] = 0.2                       # the molecule near the edge: p split with the next column
    out[0, 2, 3, 5] = 0.4
    e = extract.extract(out, SCALING, 0.5)
    assert len(e["frame"]) == 1
    assert e["x"].item() == pytest.approx(5.4) and e["y"].item() == pytest.approx(3.0)
    assert e["prob"].item() == pytest.approx(0.9 + 3e-4, abs=1e-3)   # p summed over the plus


def test_a_model_file_is_read_back_by_the_predictor_with_the_same_outputs(tmp_path):
    from smappy.netfit.predict import Predictor
    from smappy.netfit.settings import TrainingSettings
    from smappy.training import model_file
    s = TrainingSettings()
    s.parts.backbone.settings["features"] = 8
    model = network.build(s.parts, 3).eval()
    meta = {"settings": s.to_dict(), "scaling": SCALING.__dict__, "window": 3}
    model_file.write(tmp_path / "m.safetensors", {k: v.numpy() for k, v in model.state_dict().items()}, meta)
    p = Predictor(tmp_path / "m.safetensors", "cpu", threshold=0.0)
    frames = np.random.default_rng(0).uniform(50, 150, (5, 18, 22)).astype(np.float32)
    cols = p(frames, first_frame=10)
    # 3 frames localized (5 less the window's 2), cropped to a multiple of 4: 16 x 20, centred
    assert set(np.unique(cols["frame"])) <= {10, 11, 12}
    assert cols["x_pix"].min() >= 1 - 1.5 and cols["x_pix"].max() <= 20 + 1.5
    with torch.no_grad():
        x = (torch.as_tensor(frames[0:3, 1:17, 1:21])[None] - 100.0) / 100.0
        direct = model(x)
    e = extract.extract(direct, SCALING, 0.0)
    assert len(e["frame"]) == int(np.sum(cols["frame"] == 10))


def test_a_molecule_below_100_photons_counts_neither_as_missed_nor_found_as_false():
    from smappy.netfit.metrics import evaluate
    xyz = np.array([[5.0, 5.0, 0.0], [20.0, 20.0, 0.0], [30.0, 8.0, 0.0], [12.0, 30.0, 0.0]])
    truth = {"frame": np.zeros(4, int), "xyz": xyz, "photons": np.array([2000.0, 50.0, 60.0, 3000.0]), "crlb": None}
    # found: the bright one, the dim one at 20, 20, and a false one; the dim one at 30, 8 and the
    # bright one at 12, 30 are missed
    found = {"frame": np.zeros(3, int), "xyz": np.array([[5.1, 5.0, 0.0], [20.0, 20.1, 0.0], [40.0, 40.0, 0.0]]),
             "err": np.ones((3, 3))}
    r = evaluate(truth, found, (100.0, 100.0))
    assert r["recall"] == 0.5 and r["precision"] == 0.5           # 1 of the 2 bright; 1 of 2 counted fits
    assert evaluate(truth, found, (100.0, 100.0), photons_min=None)["recall"] == 0.5   # 2 of 4


def test_a_new_training_gets_decodes_network_and_a_model_saved_without_its_details_version_1s():
    from smappy.netfit.settings import Part, Parts
    new = network.build(Parts(), 3)
    assert new.p_clamp == 8.0 and isinstance(new.frame.out, torch.nn.Conv2d)
    old = Parts(backbone=Part("unet2", {"features": 48, "depth_frame": 2, "depth_union": 2}))
    v1 = network.build(old, 3)
    assert v1.p_clamp is None and isinstance(v1.frame.out, torch.nn.Identity)
    assert sum(p.numel() for p in new.parameters()) != sum(p.numel() for p in v1.parameters())


def test_the_pixel_modulation_reads_a_cosines_amplitude_and_uniform_positions_as_none():
    from smappy.training.reference import pixel_modulation
    rng = np.random.default_rng(0)
    assert pixel_modulation(rng.uniform(0, 50, 400_000)) < 0.01
    for a in (0.05, 0.2):
        u = rng.uniform(size=400_000)
        kept = u[rng.uniform(size=u.size) < (1 + a * np.cos(2 * np.pi * u)) / (1 + a)]
        assert pixel_modulation(kept + rng.integers(0, 50, kept.size)) == pytest.approx(a, abs=0.006)
    # a finer grid's two cells: the second harmonic counts the same
    u = rng.uniform(size=400_000)
    kept = u[rng.uniform(size=u.size) < (1 + 0.1 * np.cos(4 * np.pi * u)) / 1.1]
    assert pixel_modulation(kept) == pytest.approx(0.1, abs=0.006)
