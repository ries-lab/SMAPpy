"""The simulation model: each stage gives back the parameters it was given."""
from dataclasses import replace

import numpy as np
import pytest

from smappy.simulate import (BlinkingSettings, LabellingSettings,
                             LocalizationOutputSettings, SimulationSettings,
                             StructureSettings, blink, emission, expected_blinks,
                             ground_truth, label, load_structure, localizations,
                             mortensen, off_time_for, presets, simulate)
from smappy.simulate.structure import Labels


def _labels(n, rng=None):
    rng = rng or np.random.default_rng(0)
    return Labels(rng.uniform(0, 1e4, (n, 3)), np.ones(n, np.int32), np.zeros(n, np.int32))


# ----------------------------------------------------------------- structure
def test_a_line_and_a_circle_are_labelled_at_their_density_along_them():
    s = load_structure({"elements": [
        {"line": [[0, 0, 0], [3000, 4000, 0]], "density": 0.1},    # 5 um
        {"circle": 1000, "density": 0.2, "centre": [0, 0, 500], "dye": 2}]})
    counts = [np.bincount(s.sample(np.random.default_rng(k)).dye, minlength=3)[1:]
              for k in range(40)]
    mean = np.mean(counts, axis=0)
    assert mean[0] == pytest.approx(500, rel=0.03)
    assert mean[1] == pytest.approx(0.2 * 2 * np.pi * 1000, rel=0.03)
    ring = s.sample(np.random.default_rng(1))
    ring = ring.xyz[ring.dye == 2]
    np.testing.assert_allclose(np.hypot(ring[:, 0], ring[:, 1]), 1000, atol=1e-6)
    assert np.all(ring[:, 2] == 500)


def test_a_polygon_is_filled_evenly_at_its_density_per_area():
    s = load_structure({"elements": [{"polygon": [[0, 0], [2000, 0], [0, 2000]],
                                      "density": 1e-4, "z": [-50, 50]}]})
    pts = np.vstack([s.sample(np.random.default_rng(k)).xyz for k in range(20)])
    assert len(pts) / 20 == pytest.approx(1e-4 * 2e6, rel=0.03)
    assert np.all(pts[:, 0] + pts[:, 1] <= 2000) and np.all(np.abs(pts[:, 2]) <= 50)
    assert np.mean(pts[:, 0]) == pytest.approx(2000 / 3, rel=0.05)     # the centroid


def test_label_positions_come_from_a_list_or_a_file_and_are_the_same_in_every_copy(tmp_path):
    (tmp_path / "pts.csv").write_text("x,y,z,dye\n0,0,0,1\n10,0,5,2\n")
    (tmp_path / "s.yaml").write_text(
        "elements:\n  - file: pts.csv\n  - points: [[0, 20]]\n"
        "copies:\n  n: 4\n  placement: grid\n  field: [0, 0, 2000, 2000]\n")
    labels = load_structure(tmp_path / "s.yaml").sample(np.random.default_rng(0))
    assert len(labels) == 12 and set(labels.copy) == {0, 1, 2, 3}
    first = labels.xyz[labels.copy == 0]
    for c in range(1, 4):
        offset = labels.xyz[labels.copy == c] - first
        np.testing.assert_allclose(offset, offset[:1].repeat(3, 0))   # a pure shift
    assert list(labels.dye[labels.copy == 0]) == [1, 2, 1]
    np.testing.assert_allclose(first[1] - first[0], [10, 0, 5])
    centres = np.array([labels.xyz[labels.copy == c][0] for c in range(4)])
    np.testing.assert_allclose(sorted(set(centres[:, 0])), [500, 1500])   # a 2 x 2 grid


def test_random_copies_keep_their_distance_and_turn():
    s = load_structure({"elements": [{"points": [[100, 0], [-100, 0]]}],
                        "copies": {"n": 200, "field": [0, 0, 1e4, 1e4],
                                   "min_distance": 400, "rotation": "random"}})
    labels = s.sample(np.random.default_rng(0))
    pairs = labels.xyz[:, :2].reshape(200, 2, 2)
    centres = pairs.mean(axis=1)
    from scipy.spatial.distance import pdist
    assert pdist(centres).min() >= 400
    along = pairs[:, 0] - pairs[:, 1]
    np.testing.assert_allclose(np.hypot(*along.T), 200)     # turned, not bent
    angle = np.arctan2(along[:, 1], along[:, 0])
    assert np.histogram(angle, bins=4, range=(-np.pi, np.pi))[0].min() > 30


def test_an_image_scales_the_density_by_its_grey_value(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    img = np.zeros((20, 40), np.float32)
    img[:, 20:] = 1.0
    img[:, :20] = 0.25
    tifffile.imwrite(tmp_path / "grey.tif", img)
    s = load_structure({"elements": [{"image": str(tmp_path / "grey.tif"),
                                      "pixelsize": 10, "density": 0.01}]})
    pts = np.vstack([s.sample(np.random.default_rng(k)).xyz for k in range(10)])
    right, left = np.sum(pts[:, 0] >= 195), np.sum(pts[:, 0] < 195)
    assert right / 10 == pytest.approx(0.01 * 100 * 400, rel=0.05)
    assert left / right == pytest.approx(0.25, rel=0.1)


def test_every_preset_loads_and_the_pores_have_32_labels_each():
    assert {"demo", "npc", "filaments", "pie"} <= set(presets())
    for name in presets():
        assert len(load_structure(name).sample(np.random.default_rng(0))) > 100
    pores = load_structure("npc").sample(np.random.default_rng(0))
    assert set(np.bincount(pores.copy)) == {32}
    assert len(set(pores.copy)) == pytest.approx(2 * 81, rel=0.2)    # 2 per um^2


def test_the_pie_doubles_its_density_from_segment_to_segment():
    labels = load_structure("pie").sample(np.random.default_rng(0))
    xy = labels.xyz[:, :2] - 5000
    segment = (np.arctan2(xy[:, 1], xy[:, 0]) % (2 * np.pi)) // (np.pi / 4)
    per_um2 = np.bincount(segment.astype(int), minlength=8) / (np.pi * 4.0 ** 2 / 8)
    np.testing.assert_allclose(per_um2[2:], 25 * 2.0 ** np.arange(2, 8), rtol=0.1)
    assert np.all(np.diff(per_um2) > 0)


def test_a_malformed_structure_says_what_is_wrong():
    with pytest.raises(ValueError, match="exactly one of"):
        load_structure({"elements": [{"line": [[0, 0], [1, 1]], "circle": 3}]})
    with pytest.raises(ValueError, match="density"):
        load_structure({"elements": [{"line": [[0, 0], [1, 1]]}]})
    with pytest.raises(FileNotFoundError, match="demo"):
        load_structure("no such structure")


# ----------------------------------------------------------------- labelling
def test_labelling_keeps_labels_at_the_efficiency_with_one_fluorophore_each():
    rng = np.random.default_rng(0)
    fl = label(_labels(20000), LabellingSettings(efficiency=0.6), rng)
    assert len(fl) / 20000 == pytest.approx(0.6, abs=0.01)
    assert np.all(np.bincount(fl.label) <= 1)


def test_a_poisson_number_of_fluorophores_with_a_linkage_error():
    rng = np.random.default_rng(1)
    labels = _labels(20000)
    fl = label(labels, LabellingSettings(fluorophores=2.0, poisson=True, linkage_nm=8.0), rng)
    per_label = np.bincount(fl.label, minlength=20000)
    assert per_label.mean() == pytest.approx(2.0, rel=0.02)
    assert per_label.var() == pytest.approx(2.0, rel=0.05)        # Poisson
    offset = fl.xyz - labels.xyz[fl.label]
    np.testing.assert_allclose(offset.std(axis=0), 8.0, rtol=0.03)


# ------------------------------------------------------------------ blinking
@pytest.mark.parametrize("activation", ["constant", "decay"])
def test_the_off_time_gives_the_number_of_blinks_asked_for(activation):
    settings = BlinkingSettings(on_time=1.5, blinks=3.0, bleaching=0.1, activation=activation)
    b = blink(20000, 5000, settings, np.random.default_rng(0))
    assert len(b) / 20000 == pytest.approx(3.0, rel=0.03)
    assert expected_blinks(b.off_time, 1.5, 0.1, 5000) == pytest.approx(3.0, rel=1e-3)


def test_on_times_are_exponential_and_a_blink_emits_its_photons():
    settings = BlinkingSettings(on_time=2.0, blinks=2.0, photons=4000, photons_std=0)
    b = blink(20000, 100000, settings, np.random.default_rng(2))
    whole = b.end < 100000
    duration = (b.end - b.start)[whole]
    assert duration.mean() == pytest.approx(2.0, rel=0.02)
    assert duration.std() == pytest.approx(2.0, rel=0.03)          # exponential
    assert (b.rate * (b.end - b.start))[whole].mean() == pytest.approx(4000, rel=0.02)
    # the frames share out a blink's photons without losing any
    em = emission(b, 100000)
    assert em.photons.sum() == pytest.approx((b.rate * (b.end - b.start)).sum(), rel=1e-9)
    assert np.all(np.diff(em.frame) >= 0)
    key = em.frame * 20000 + em.owner
    assert len(np.unique(key)) == len(key)          # one row per fluorophore and frame


def test_photons_per_blink_have_the_mean_and_spread_asked_for():
    settings = BlinkingSettings(photons=5000, photons_std=2500)
    b = blink(20000, 10000, settings, np.random.default_rng(3))
    total = b.rate * settings.on_time
    assert total.mean() == pytest.approx(5000, rel=0.02)
    assert total.std() == pytest.approx(2500, rel=0.03)


def test_constant_activation_is_flat_and_decaying_activation_falls():
    counts = {}
    for activation in ("constant", "decay"):
        settings = BlinkingSettings(blinks=4.0, bleaching=0.2, activation=activation)
        b = blink(20000, 10000, settings, np.random.default_rng(4))
        counts[activation] = np.histogram(b.start, bins=5, range=(0, 10000))[0]
    flat = counts["constant"]
    assert flat.max() / flat.min() < 1.05
    falling = counts["decay"]
    assert np.all(np.diff(falling) < 0) and falling[0] > 2 * falling[-1]


def test_more_blinks_than_bleaching_allows_are_refused():
    with pytest.raises(ValueError, match="1/p"):
        off_time_for(5.0, 1.5, 0.25, 1000)


# ------------------------------------------------------------- localizations
def test_the_mortensen_precision_at_a_known_point():
    # 1000 photons, no background, sigma 100 on 100 nm pixels:
    # sa^2 = 100^2 + 100^2/12, and with no background var = sa^2 / N
    sa2 = 100 ** 2 + 100 ** 2 / 12
    assert mortensen(1000, 0, 100, 100) == pytest.approx(np.sqrt(sa2 / 1000))
    assert mortensen(1000, 10, 100, 100) > mortensen(1000, 0, 100, 100)
    assert mortensen(1000, 10, 100, 100, emccd=True) == pytest.approx(
        np.sqrt(2) * mortensen(1000, 10, 100, 100))
    # vectorized, and the same as one at a time
    many = mortensen(np.array([200.0, 1000.0]), np.array([2.0, 10.0]), 130, 100)
    assert many == pytest.approx([mortensen(200, 2, 130, 100),
                                  mortensen(1000, 10, 130, 100)])


def test_the_mortensen_precision_is_what_the_fitter_achieves():
    """The maximum-likelihood precision (Mortensen's eq. 5), not the
    least-squares one (eq. 6, 16/9 at no background) that SMAP's
    MortensenCRLB uses and that this was until then: 10-23% pessimistic."""
    from scipy.special import erf
    from smappy.psf import GaussianPSF
    rng = np.random.default_rng(0)
    size, pixel, sigma = 13, 100.0, 130.0
    k = np.arange(size)
    for photons, background in ((1000, 2), (5000, 10), (300, 20)):
        x0 = size // 2 + rng.uniform(-0.5, 0.5, 3000)
        y0 = size // 2 + rng.uniform(-0.5, 0.5, 3000)
        s = sigma / pixel

        def edges(c):
            return 0.5 * (erf((k[None] - c[:, None] + 0.5) / (np.sqrt(2) * s))
                          - erf((k[None] - c[:, None] - 0.5) / (np.sqrt(2) * s)))
        rois = rng.poisson(background + photons * edges(y0)[:, :, None]
                           * edges(x0)[:, None, :]).astype(np.float32)
        found = GaussianPSF(sigma=1.2).unpack(GaussianPSF(sigma=1.2).fit(rois))
        ok = np.isfinite(found["x_roi"])
        scatter = np.std(found["x_roi"][ok] - x0[ok]) * pixel
        assert mortensen(photons, background, sigma, pixel) == pytest.approx(
            scatter, rel=0.05), (photons, background)


def test_the_localization_errors_are_the_precision_the_table_claims():
    settings = SimulationSettings(n_frames=4000, seed=5)
    truth = ground_truth(settings)
    locs = localizations(truth)
    true = truth.fluorophores.xyz[locs["emitter"]]
    for k, (axis, err) in enumerate((("x_nm", "xy_err_nm"), ("y_nm", "xy_err_nm"),
                                     ("z_nm", "z_err_nm"))):
        pull = (locs[axis] - true[:, k]) / locs[err]
        assert pull.std() == pytest.approx(1.0, abs=0.03), axis
    assert locs["photons"].min() >= settings.localizations.min_photons
    assert locs.metadata["n_emitters"] == len(truth.fluorophores)


def test_close_emitters_are_removed_or_averaged():
    base = SimulationSettings(n_frames=200, seed=6)      # dense on purpose
    from scipy.spatial import cKDTree
    removed = simulate(base)
    for f in np.unique(removed["frame"])[:40]:
        # bright ones only: a 10-photon localization can be put anywhere
        m = (removed["frame"] == f) & (removed["photons"] > 500)
        # the true positions were 250 nm apart; the noise moves them a little
        xy = np.column_stack([removed["x_nm"][m], removed["y_nm"][m]])
        assert not cKDTree(xy).query_pairs(200)
    averaged = simulate(replace(base, localizations=LocalizationOutputSettings(close="average")))
    assert averaged.metadata["n_close"] == removed.metadata["n_close"] > 0
    merged = averaged["n_merged"]
    assert merged.max() > 1
    assert len(averaged) == len(removed) + np.sum(merged > 1)
    assert averaged["photons"][merged > 1].mean() > averaged["photons"][merged == 1].mean()


def test_a_zero_spread_is_exact():
    s = SimulationSettings(n_frames=500, seed=7, background=0.0,
                           blinking=BlinkingSettings(photons=3000, photons_std=0))
    truth = ground_truth(s)
    assert np.allclose(truth.blinks.rate, 3000 / 1.5)
    assert np.all(localizations(truth)["background"] == 0)


def test_the_simulation_keeps_its_settings_and_the_seed_repeats_it():
    a = simulate(n_frames=300, seed=8, drift=True)
    b = simulate(n_frames=300, seed=8, drift=True)
    np.testing.assert_array_equal(a["x_nm"], b["x_nm"])
    assert a.metadata["simulation_settings"]["blinking"]["on_time"] == 1.5
    assert len(a.metadata["drift_truth"]) == 300


def test_a_structure_file_drives_the_simulation(tmp_path):
    (tmp_path / "one.yaml").write_text("elements:\n  - points: [[5000, 5000, 0]]\n")
    locs = simulate(n_frames=1000, seed=9,
                    structure=StructureSettings(file=str(tmp_path / "one.yaml")),
                    labelling=LabellingSettings(fluorophores=50))
    assert locs.metadata["n_emitters"] == 50
    assert abs(np.median(locs["x_nm"]) - 5000) < 5


# ------------------------------------------------------------ camera source
def test_a_simulation_file_opens_as_an_acquisition_with_its_camera(tmp_path):
    from smappy.io.tiff import camera_metadata, open_stack
    from smappy.simulate import render
    from smappy.simulate.source import SimulatedSource, write_recipe
    settings = SimulationSettings(output="camera", n_frames=60, seed=10,
                                  labelling=LabellingSettings(efficiency=0.05))
    path = write_recipe(settings, tmp_path / "run.sim.yaml")
    source = open_stack(path)
    assert isinstance(source, SimulatedSource) and len(source) == 60
    blocks = np.concatenate([b for _, b in source.frames(chunk=7)])
    np.testing.assert_array_equal(blocks, render(ground_truth(settings)))
    np.testing.assert_array_equal(source.frame(33), blocks[33])
    camera = camera_metadata(source)
    assert (camera.conversion, camera.offset) == (0.5, 100.0)
    assert np.allclose(camera.pixelsize_um, 0.1)
    assert len(source.truth()) == len(ground_truth(settings).emission)
    # and the camera window can tell it was the simulation that said so
    from smappy.io.tiff import SIMULATED, resolve_camera
    origin = resolve_camera(source).sources["conversion"]
    assert (origin.kind, origin.detail) == SIMULATED


def test_the_plugin_writes_a_simulation_a_fitter_can_open(tmp_path):
    from smappy import plugins
    from smappy.plugins import Context
    plugin = plugins.get("File/Simulate/Blinking Structure")()
    settings = plugin.Settings(output="camera", n_frames=20)
    with pytest.raises(ValueError, match="where to write"):
        plugin.run(Context(), settings)
    settings.camera.path = str(tmp_path / "x.sim.yaml")
    settings.camera.tiff = True
    result = plugin.run(Context(), settings)
    tifffile = pytest.importorskip("tifffile")
    assert tifffile.imread(result.data["tiff"]).shape == (20, 100, 100)
    from smappy.plugins.fit import default_output_path
    assert default_output_path(result.data["path"]).name == "x_locs.hdf5"


# ------------------------------------------------- after comparing with SMAP
def test_a_free_linkage_error_moves_each_blink_and_a_fixed_one_the_fluorophore(tmp_path):
    """SMAP's two linkage errors: fixed per fluorophore, free per blink."""
    (tmp_path / "one.yaml").write_text("elements:\n  - points: [[5000, 5000, 0]]\n")
    for fixed, free in ((12.0, 0.0), (0.0, 12.0)):
        truth = ground_truth(SimulationSettings(
            n_frames=20000, seed=11, structure=StructureSettings(file=str(tmp_path / "one.yaml")),
            labelling=LabellingSettings(fluorophores=300, linkage_nm=fixed,
                                        linkage_free_nm=free),
            blinking=BlinkingSettings(activation="all", blinks=20)))
        b = truth.blinks
        at = truth.fluorophores.xyz[b.owner, 0] + b.offset[:, 0]
        # one fluorophore's blinks scatter about their mean by the free error
        # alone, and the fluorophores about the label by the fixed one
        n = np.bincount(b.owner)
        mean = np.bincount(b.owner, weights=at) / n
        within = np.sum((at - mean[b.owner]) ** 2) / np.sum(n - 1)
        assert np.sqrt(within) == pytest.approx(free, abs=0.5)
        assert (truth.fluorophores.xyz[:, 0] - 5000).std() == pytest.approx(fixed, abs=1.2)


def test_every_blink_until_bleached_is_the_dyes_number_whatever_the_frames():
    """SMAP's "Dye": geometric blinks with mean `blinks`, all of them shown,
    spread evenly, each fluorophore's in order and never on twice at once."""
    settings = BlinkingSettings(activation="all", blinks=4.0, on_time=1.5)
    for n_frames in (500, 50000):
        b = blink(20000, n_frames, settings, np.random.default_rng(12))
        per = np.bincount(b.owner, minlength=20000)
        assert per.mean() == pytest.approx(4.0, rel=0.03)
        assert np.mean(per == 1) == pytest.approx(0.25, abs=0.01)   # geometric, p = 1/4
        same = b.owner[1:] == b.owner[:-1]
        assert np.all(b.start[1:][same] >= b.end[:-1][same])
        counts = np.histogram(b.start, bins=5, range=(0, n_frames))[0]
        assert counts.max() / counts.min() < 1.05


def test_copies_are_tilted_jittered_and_their_poses_recorded():
    from scipy.spatial.transform import Rotation
    shape = [[100, 0, 0], [0, 50, 0], [0, 0, 20]]
    s = load_structure({"elements": [{"points": shape}],
                        "copies": {"n": 300, "rotation": "random", "tilt": 15,
                                   "jitter": [20, 5], "placement": "grid",
                                   "field": [0, 0, 1e5, 1e5]}})
    labels = s.sample(np.random.default_rng(13))
    p = {k: np.asarray(v) for k, v in labels.poses.items()}
    assert p["beta_deg"].max() <= 15 and p["beta_deg"].max() > 13
    # a grid of 18 x 17 over 100 um, each copy up to 20 nm off its node
    nodes = (np.arange(18) + 0.5) * 1e5 / 18
    off = p["x_nm"] - nodes[np.abs(p["x_nm"][:, None] - nodes).argmin(axis=1)]
    assert np.abs(off).max() <= 20 and np.abs(off).max() > 18
    assert np.abs(p["z_nm"]).max() <= 5 and np.abs(p["z_nm"]).max() > 4
    for k in range(0, 300, 37):
        r = Rotation.from_euler("ZYZ", np.deg2rad([p["alpha_deg"][k], p["beta_deg"][k],
                                                   p["gamma_deg"][k]])).as_matrix()
        centre = [p["x_nm"][k], p["y_nm"][k], p["z_nm"][k]]
        np.testing.assert_allclose(labels.xyz[labels.copy == k] - centre,
                                   np.array(shape, float) @ r.T, atol=0.05)


def test_the_truth_carries_each_copys_pose():
    locs = simulate(n_frames=300, seed=14, structure=StructureSettings(preset="npc"))
    poses = locs.metadata["copies"]
    assert len(poses["x_nm"]) == locs["copy"].max() + 1
    k = int(np.bincount(locs["copy"]).argmax())
    mine = locs["copy"] == k
    assert np.hypot(np.median(locs["x_nm"][mine]) - poses["x_nm"][k],
                    np.median(locs["y_nm"][mine]) - poses["y_nm"][k]) < 30


def test_an_emccd_doubles_the_variance_and_the_gain_divides_out():
    from smappy.simulate import CameraOutputSettings, render
    # the conversions keep an ADU step small against a photon, so rounding
    # to whole ADU adds no variance to speak of
    for gain, conversion, excess in ((0.0, 0.05, 1.0), (100.0, 5.0, 2.0)):
        s = SimulationSettings(output="camera", n_frames=20, seed=15, background=20.0,
                               labelling=LabellingSettings(efficiency=0.0),
                               camera=CameraOutputSettings(em_gain=gain, conversion=conversion,
                                                           read_noise=0.0, offset=100.0))
        frames = render(ground_truth(s)).astype(float)
        photons = (frames - 100.0) * conversion / (gain or 1.0)
        assert photons.mean() == pytest.approx(20.0, rel=0.02)
        assert photons.var() == pytest.approx(excess * 20.0, rel=0.05)


def test_the_camera_draws_every_spot_however_dim():
    """Camera frames keep what the localizations would drop: a sliver of a
    blink with a few photons is drawn, and the frames hold all the light."""
    from smappy.simulate import CameraOutputSettings, camera_truth, render
    s = SimulationSettings(output="camera", n_frames=200, seed=16, background=0.0,
                           labelling=LabellingSettings(efficiency=0.2),
                           camera=CameraOutputSettings(read_noise=0.0, offset=0.0,
                                                       conversion=1.0, size_px=120))
    truth = ground_truth(s)
    table = camera_truth(truth)
    assert np.sum(table["photons"] < 10) > 20
    frames = render(truth).astype(float)
    assert frames.sum() == pytest.approx(table["photons"].sum(), rel=0.01)
