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


def test_blur_keeps_a_constant_and_step_4s_backbones_give_the_same_maps():
    x = torch.full((1, 4, 8, 8), 3.0)
    assert torch.allclose(network.Blur(4)(x), x)
    for style in ({"blur": True}, {"block": "residual"}, {"block": "convnext"}, {"block": "dilated"}, {"offset_range": 2.0}):
        model = network.TwoStageUNet(features=8, **style)
        assert model(torch.zeros(2, 3, 16, 24)).shape == (2, 10, 16, 24)


def test_the_offsets_reach_as_many_cells_as_their_range():
    model = network.TwoStageUNet(features=8, offset_range=2.0)
    last = model.heads["phot_xyz"].net[-1]
    torch.nn.init.constant_(last.bias, 50.0)              # tanh saturated
    assert model(torch.zeros(1, 3, 8, 8))[0, 2].max().item() == pytest.approx(2.0)


def test_detections_are_matched_to_their_mutually_nearest_truth_within_the_gates():
    t = torch.tensor([[[5.0, 5.0, 0.0], [20.0, 20.0, 0.0], [30.0, 8.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    # two detections near the first truth (the nearer wins), one near the second, one far, one on a masked truth
    f = torch.tensor([[5.2, 5.0, 0.0], [5.5, 5.0, 0.0], [20.0, 20.5, 100.0], [40.0, 40.0, 0.0], [30.0, 8.0, 0.0]])
    d, k = loss.match_nearest(t, mask, torch.zeros(5, dtype=torch.long), f, (100.0, 100.0), 250.0, 500.0)
    assert d.tolist() == [0, 2] and k.tolist() == [0, 1]


def test_the_reported_sigma_is_trained_by_the_reported_errors_and_moves_nothing_else():
    torch.manual_seed(0)
    model = network.TwoStageUNet(features=8, sigma_rep=True)
    with torch.no_grad():
        model.heads["p"].net[-1].bias.fill_(-8.0)
    x = torch.zeros(1, 3, 16, 16)
    out2 = model(x)
    assert out2.shape == (1, 13, 16, 16)
    p = torch.zeros_like(out2[:, :1])
    p[0, 0, 7, 5] = 0.99                        # one detection, at row 7, column 5; its molecule at 5.3
    out3 = torch.cat([p, out2[:, 1:]], 1)
    batch = Batch(frames=x, xyz=torch.tensor([[[5.3, 7.0, 0.0]]]), photons=torch.tensor([[2000.0]]),
                  mask=torch.tensor([[True]]), background=torch.tensor([0.0]))
    r = loss.reported_term(out3, batch, SCALING, (100.0, 100.0), 0.5, "lookahead")
    r.backward()
    grads = {n: (q.grad is not None and q.grad.abs().sum().item() > 0) for n, q in model.named_parameters()}
    assert any(g for n, g in grads.items() if n.startswith("heads.sigma_rep"))
    assert not any(g for n, g in grads.items() if not n.startswith("heads.sigma_rep"))
    e = extract.extract(out3.detach(), SCALING, 0.5)
    assert torch.equal(e["x_err"], out3[0, 10, 7, 5].reshape(1)) and "x_err_mix" in e


def test_the_local_count_is_the_exact_poisson_binomial_of_each_windows_true_count():
    import itertools
    g = torch.Generator().manual_seed(3)
    p = torch.rand(1, 4, 5, generator=g) * 0.8
    counts = torch.zeros(1, 4, 5)
    counts[0, 1, 2], counts[0, 1, 3], counts[0, 3, 0] = 1, 1, 1
    got = loss.window_counts(p, counts, 3).item()
    want = 0.0
    for r in range(4):
        for c in range(5):
            cells = [(i, j) for i in range(r - 1, r + 2) for j in range(c - 1, c + 2) if 0 <= i < 4 and 0 <= j < 5]
            n = int(sum(counts[0, i, j] for i, j in cells))
            q = [p[0, i, j].item() for i, j in cells]
            prob = sum(math.prod(qi if on else 1 - qi for qi, on in zip(q, ons))
                       for ons in itertools.product((0, 1), repeat=len(q)) if sum(ons) == n)
            want -= math.log(prob)
    assert got == pytest.approx(want / 9, rel=1e-4)


def test_a_molecule_is_counted_in_the_cell_whose_centre_is_nearest_on_either_grid():
    batch = Batch(frames=None, xyz=torch.tensor([[[2.4, 1.6, 0.0], [2.6, 1.4, 0.0], [-0.7, 0.0, 0.0]]]),
                  photons=torch.ones(1, 3), mask=torch.tensor([[True, True, True]]), background=torch.zeros(1))
    c = loss.cell_counts(batch, 4, 5)
    assert c[0, 2, 2] == 1 and c[0, 1, 3] == 1 and c.sum() == 2           # the third is outside the frame
    fine = loss.cell_counts(batch, 8, 10, scale=2)
    assert fine[0, 4, 5] == 1 and fine[0, 3, 6] == 1                       # 2.4 -> cell 5 (centre 2.25)


def test_the_local_count_asks_for_more_probability_where_two_molecules_are_close():
    out = _maps()
    out[0, 0, 3, 4], out[0, 0, 3, 5] = 0.5, 0.5             # the map's mass: one molecule's worth
    out.requires_grad_(True)
    batch = Batch(frames=None, xyz=torch.tensor([[[4.0, 3.0, 0.0], [5.0, 3.0, 0.0]]]), photons=torch.ones(1, 2),
                  mask=torch.tensor([[True, True]]), background=torch.zeros(1))
    t = loss.terms(out, batch, SCALING, names=("local_count",))
    t["local_count"].sum().backward()
    assert out.grad[0, 0, 3, 4] < 0 and out.grad[0, 0, 3, 5] < 0      # raising p there lowers the term
    single = Batch(frames=None, xyz=torch.tensor([[[4.0, 3.0, 0.0]]]), photons=torch.ones(1, 1),
                   mask=torch.tensor([[True]]), background=torch.zeros(1))
    out.grad = None
    loss.terms(out, single, SCALING, names=("local_count",))["local_count"].sum().backward()
    assert out.grad[0, 0, 3, 5] > 0                                    # one molecule: the second half goes


def test_a_detection_gathering_two_molecules_worth_of_p_is_split_at_its_cells_two_places():
    out = _maps(12, 12)
    # two molecules 1.4 pixels apart: p spread over three cells, one peak, the next cell's p below SPLIT_P
    for col, p, dx in ((5, 0.7, -0.2), (6, 0.55, 0.2), (7, 0.25, -0.8)):
        out[0, 0, 6, col], out[0, 2, 6, col] = p, dx
    out[0, 0, 2, 2] = 0.95                                             # and a single molecule elsewhere
    plain = extract.extract(out, SCALING, 0.5, "mixture_peak")
    assert len(plain["frame"]) == 2                                    # one for the pair
    e = extract.extract(out, SCALING, 0.5, "mixture_peak", split=1.4)
    assert len(e["frame"]) == 3 and sorted(e["split"].tolist()) == [0, 1, 2]
    two = e["split"] > 0
    assert sorted(e["x"][two].tolist()) == pytest.approx([4.8, 6.2], abs=0.05)     # each half at its cells
    assert e["x"][~two].item() == pytest.approx(2.0)                               # the single one untouched


def _slot_maps(k=2, h=8, w=8):
    """K slots' maps, nine channels each and the background (`network`'s layout), p 1e-4 everywhere."""
    out = torch.zeros(1, 9 * k + 1, h, w)
    for s in range(k):
        out[:, 9 * s] = 1e-4
        out[:, 9 * s + 5:9 * s + 9] = 0.5
    return out


def _two_in_a_cell():
    """Two molecules 0.3 pixels apart in cell (row 3, column 5) and the batch that holds them."""
    batch = Batch(frames=None, xyz=torch.tensor([[[4.9, 3.0, -300.0], [5.2, 3.0, 300.0]]]),
                  photons=torch.tensor([[2000.0, 2000.0]]), mask=torch.tensor([[True, True]]), background=torch.tensor([0.0]))
    return batch


def test_a_network_with_two_slots_gives_nine_maps_a_slot_and_the_background_and_one_slot_is_unchanged():
    model = network.TwoStageUNet(features=8, slots=2)
    out = model(torch.zeros(2, 3, 16, 24))
    assert out.shape == (2, 19, 16, 24) and network.slots_of(out) == 2
    # p starts at the same molecules a pixel, shared by the slots
    assert out[0, [0, 9]].sum().item() == pytest.approx(network.P0 * 16 * 24, rel=1e-3)
    assert network.slots_of(network.TwoStageUNet(features=8)(torch.zeros(1, 3, 8, 8))) == 1
    with pytest.raises(ValueError):
        network.TwoStageUNet(features=8, slots=2, sigma_rep=True)


def test_the_matched_loss_gives_two_molecules_in_one_cell_a_slot_each_and_charges_every_other_slot():
    batch = _two_in_a_cell()
    out = _slot_maps()
    for s, (dx, z) in enumerate(((-0.1, -0.3), (0.2, 0.3))):           # slot 0 the left one, slot 1 the right
        out[0, 9 * s, 3, 5], out[0, 9 * s + 1, 3, 5] = 0.9, 0.2
        out[0, 9 * s + 2, 3, 5], out[0, 9 * s + 4, 3, 5] = dx, z
    t = loss.terms(out, batch, SCALING, names=("matched",))
    # each molecule exactly at its slot: -log N = sum log sigma + 2 log 2 pi, then -log p; the rest -log (1 - p)
    nll = 4 * math.log(0.5) + 2 * math.log(2 * math.pi)
    others = (2 * 64 - 2) * -math.log(1 - 1e-4)
    assert t["matched"].item() == pytest.approx(others + 2 * (nll - math.log(0.9)), rel=1e-4)
    assert t["unmatched"].item() == 0


def test_the_matched_loss_charges_a_duplicate_and_a_merge_where_the_mixture_does_not():
    batch = _two_in_a_cell()
    one = Batch(frames=None, xyz=batch.xyz[:, :1], photons=batch.photons[:, :1], mask=batch.mask[:, :1], background=batch.background)
    good = _slot_maps()
    good[0, 0, 3, 5], good[0, 1, 3, 5], good[0, 2, 3, 5], good[0, 4, 3, 5] = 0.9, 0.2, -0.1, -0.3
    dup = good.clone()                                            # a second slot on the same molecule
    dup[0, 9:13, 3, 5] = good[0, 0:4, 3, 5]
    dup[0, 13, 3, 5] = good[0, 4, 3, 5]
    m = lambda o, b: loss.terms(o, b, SCALING, names=("matched", "mixture"))
    assert m(dup, one)["matched"].item() > m(good, one)["matched"].item() + 2.0
    # the mixture barely notices: the faint cells' share of the weights moves a little
    assert m(dup, one)["mixture"].item() == pytest.approx(m(good, one)["mixture"].item(), abs=0.02)
    # two molecules, one slot: the second must take a slot with p 1e-4
    assert m(good, batch)["matched"].item() > m(dup, batch)["matched"].item()


def test_the_threshold_extraction_reports_each_slot_whose_own_p_passes_at_its_own_place():
    out = _slot_maps()
    out[0, 0, 3, 5], out[0, 2, 3, 5] = 0.8, -0.1
    out[0, 9, 3, 5], out[0, 11, 3, 5], out[0, 13, 3, 5] = 0.7, 0.2, 0.3
    out[0, 9, 6, 1] = 0.4                                         # below the threshold
    e = extract.extract(out, SCALING, 0.5, "threshold")
    order = torch.argsort(e["slot"])
    assert e["x"][order].tolist() == pytest.approx([4.9, 5.2])
    assert e["y"][order].tolist() == pytest.approx([3.0, 3.0])
    assert e["z_nm"][order].tolist() == pytest.approx([0.0, 300.0])
    assert e["prob"][order].tolist() == pytest.approx([0.8, 0.7])


def test_shots_sinkhorn_assignment_reaches_the_exact_one_and_gives_two_molecules_in_a_cell_a_slot_each():
    batch = _two_in_a_cell()
    out = _slot_maps()
    for s, (dx, z) in enumerate(((-0.1, -0.3), (0.2, 0.3))):
        out[0, 9 * s, 3, 5], out[0, 9 * s + 1, 3, 5] = 0.9, 0.2
        out[0, 9 * s + 2, 3, 5], out[0, 9 * s + 4, 3, 5] = dx, z
    exact = loss.terms(out, batch, SCALING, names=("matched",))["matched"].item()
    near = loss.terms(out, batch, SCALING, names=("matched",), assign={"how": "sinkhorn", "eps": 1e-2})["matched"].item()
    assert near == pytest.approx(exact, rel=1e-3)
    # softer, the coupling spreads each molecule over the slots in reach: never below the exact optimum
    soft = loss.terms(out, batch, SCALING, names=("matched",), assign={"how": "sinkhorn", "eps": 0.3})["matched"].item()
    assert soft > exact


@pytest.mark.parametrize("time_model, extra", [("attention", {}), ("convgru", {}), ("convgru", {"skip": True}),
                                              ("convgru", {"norm": True})])
def test_a_time_model_gives_the_same_maps_and_reads_the_neighbouring_frames(time_model, extra):
    from smappy.netfit.settings import Part, Parts
    torch.manual_seed(0)
    parts = Parts(time_model=Part(time_model, {"frames": 5, **extra}))
    parts.backbone.settings["features"] = 8
    model = network.build(parts, 5).eval()
    for head in model.heads.values():                            # the heads start at a constant output
        torch.nn.init.normal_(head.net[-1].weight, std=0.1)
    x = torch.randn(1, 5, 16, 16)
    out = model(x)
    assert out.shape == (1, 13, 16, 16)
    y = x.clone()
    y[:, 0] += 1.0                                               # the first frame changes the middle's maps
    assert not torch.allclose(model(y), out)


def test_the_convgrus_skip_hands_on_the_middle_frames_features_beside_the_two_states():
    gru = network.TimeGRU(8, 5, skip=True)
    f = torch.randn(2, 5, 8, 4, 4)
    out = gru(f)
    assert gru.out_channels == 24 and out.shape == (2, 24, 4, 4)
    assert torch.equal(out[:, :8], f[:, 2])
    assert network.TimeGRU(8, 5).out_channels == 16                 # a model saved before is built as before


def test_the_attentions_temporal_dropout_acts_only_in_training_and_never_hides_the_middle_frame():
    att = network.TimeAttention(8, 5, heads=2, drop=1.0)
    f = torch.randn(2, 5, 8, 4, 4)
    att.train()
    alone = att(f)                                               # every neighbour hidden: the middle frame alone
    g = f.clone()
    g[:, [0, 1, 3, 4]] = torch.randn(2, 4, 8, 4, 4)
    assert torch.allclose(att(g), alone, atol=1e-5)
    att.eval()
    assert not torch.allclose(att(g), att(f))
