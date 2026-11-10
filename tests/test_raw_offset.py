"""Raw channel offset: the bead map's error in the sample, from crosstalk too dim to detect.

A split camera simulated frame by frame: the main half's dye leaks a known fraction of its light
into the other half, displaced by a known amount from where the map puts it; the other dye leaks
nothing.  The table is the molecules themselves, as a perfect fit would give them, so what is
tested is the summing and the fit, and the numbers are asserted back.
"""
import numpy as np
import pytest
from scipy.special import erf

from smappy.calibrate.dual import ChannelTransform, split_geometry
from smappy.calibrate.transform import load_transform, save_channel_transform
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.raw_offset import PatchSum, fit_correction, fit_spot

SHAPE = (96, 64)          # rows, columns: the upper half the main channel, the lower the other
SPLIT = 48
PIXEL = 100.0
SIGMA = 1.2               # PSF, px


def registration(dx=0.8, dy=-0.5):
    """Secondary -> main in chip pixels: the lower half, shifted."""
    forward = np.eye(3)                                      # main -> secondary
    forward[:2, 2] = [dx, SPLIT + dy]
    return ChannelTransform(np.linalg.inv(forward),
                            split_geometry("up-down", "upper", SHAPE, split_position=SPLIT), {})


def spot(frame, x, y, photons):
    """Add a pixel-integrated Gaussian to ``frame`` in place."""
    r = 6
    cx, cy = int(round(x)), int(round(y))
    xs = np.arange(max(cx - r, 0), min(cx + r + 1, frame.shape[1]))
    ys = np.arange(max(cy - r, 0), min(cy + r + 1, frame.shape[0]))
    s = np.sqrt(2) * SIGMA
    ex = 0.5 * (erf((xs + 0.5 - x) / s) - erf((xs - 0.5 - x) / s))
    ey = 0.5 * (erf((ys + 0.5 - y) / s) - erf((ys - 0.5 - y) / s))
    frame[np.ix_(ys, xs)] += photons * np.outer(ey, ex)


def movie(path, t, offset_px=(0.12, -0.08), leak=0.08, n_frames=400, per=(6, 3), seed=1):
    """Write the movie and return its table.  The main dye's leak sits where the map would put a
    partner that is ``offset_px`` (main-half px) away: ``to_secondary(p - offset)``, so the map
    is off by ``offset`` -- what the plugin reports as main position less mapped secondary."""
    import tifffile
    rng = np.random.default_rng(seed)
    frames, rows = [], []
    for f in range(n_frames):
        img = np.full(SHAPE, 10.0)
        for _ in range(per[0]):                              # the main dye, upper half
            p = np.array([rng.uniform(8, SHAPE[1] - 8), rng.uniform(8, SPLIT - 8)])
            spot(img, p[0], p[1], 3000 * (1 - leak))
            q = t.to_secondary((p - offset_px)[None, :])[0]
            spot(img, q[0], q[1], 3000 * leak)
            rows.append((f, p[0], p[1], 0))
        for _ in range(per[1]):                              # the other dye: no crosstalk
            q = np.array([rng.uniform(8, SHAPE[1] - 8), rng.uniform(SPLIT + 8, SHAPE[0] - 8)])
            spot(img, q[0], q[1], 3000)
            rows.append((f, q[0], q[1], 1))
        frames.append(rng.poisson(img) + 100)
    tifffile.imwrite(path, np.stack(frames).astype(np.uint16))
    a = np.array(rows)
    noise = rng.normal(0, 0.02, (len(a), 2))
    return Localizations({"frame": a[:, 0].astype(np.int64),
                          "x_nm": (a[:, 1] + noise[:, 0]) * PIXEL,
                          "y_nm": (a[:, 2] + noise[:, 1]) * PIXEL,
                          "photons": np.full(len(a), 3000.0),
                          "sigma_nm": np.full(len(a), SIGMA * PIXEL),
                          "camera_channel": a[:, 3].astype(np.int8)},
                         {"pixelsize_nm": PIXEL, "source": str(path),
                          "channels": {"regions": [[0, 0, SHAPE[1], SPLIT],
                                                   [0, SPLIT, SHAPE[1], SHAPE[0] - SPLIT]]}})


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    folder = tmp_path_factory.mktemp("raw_offset")
    t = registration()
    map_path = folder / "beads_2ct.h5"
    save_channel_transform(map_path, t)
    locs = movie(folder / "sample.tif", t)
    return folder, t, map_path, locs


def settings(**changes):
    from smappy.plugins.raw_offset import RawOffsetSettings
    base = dict(frames=400, lag=5, grid=2)
    base.update(changes)
    return RawOffsetSettings(**base)


def test_the_maps_offset_in_the_sample_is_found_from_crosstalk_too_dim_to_detect(sample):
    from smappy.plugins.raw_offset import RawChannelOffset
    folder, t, map_path, locs = sample
    result = RawChannelOffset().run(Context(locs=locs), settings(registration=str(map_path)))
    main = result.data["directions"]["main -> secondary"]
    np.testing.assert_allclose(main["residual_nm"], [12.0, -8.0], atol=max(
        3 * max(main["error_nm"]), 1.5))
    assert max(main["error_nm"]) < 2.0
    assert abs(main["fraction"] - 0.08 / 0.92) < 0.01          # leaked over what stayed
    # the other dye leaks nothing, and says so rather than reporting noise
    assert result.data["directions"]["secondary -> main"] is None
    assert "no measurable crosstalk" in result.text


def test_the_correction_is_kept_for_transform_and_folded_into_a_saved_map(sample):
    from smappy.plugins.raw_offset import RawChannelOffset
    folder, t, map_path, locs = sample
    result = RawChannelOffset().run(Context(locs=locs),
                                    settings(registration=str(map_path), save=True))
    kept = result.transformations["raw channel offset"]
    x, y, _ = kept.apply(np.array([3000.0]), np.array([2000.0]))
    np.testing.assert_allclose([x[0] - 3000, y[0] - 2000], [12.0, -8.0], atol=2.0)
    corrected = load_transform(folder / "beads_raw_2ct.h5")
    q = t.to_secondary(np.array([[30.0, 20.0]]))
    moved = corrected.to_reference(q) - t.to_reference(q)
    np.testing.assert_allclose(moved[0] * PIXEL, [12.0, -8.0], atol=2.0)


def test_an_overlaid_table_is_measured_with_the_map_it_was_overlaid_with(sample):
    from smappy.channel_overlay import overlay_table
    from smappy.plugins.raw_offset import RawChannelOffset
    folder, t, map_path, locs = sample
    overlaid = overlay_table(locs, t, path=map_path)
    result = RawChannelOffset().run(Context(locs=overlaid), settings())
    main = result.data["directions"]["main -> secondary"]
    np.testing.assert_allclose(main["residual_nm"], [12.0, -8.0], atol=max(
        3 * max(main["error_nm"]), 1.5))
    assert "beads_2ct.h5" in result.text


def test_the_raw_frames_are_found_near_the_table_when_their_recorded_path_is_gone(tmp_path):
    from smappy.plugins.raw_offset import find_raw
    (tmp_path / "acquisition").mkdir()
    raw = tmp_path / "acquisition" / "movie_MMStack_Pos0.ome.tif"
    raw.write_bytes(b"")
    locs = Localizations({"x_nm": np.zeros(1)},
                         {"source": r"D:\DataSSD\acquisition\movie_MMStack_Pos0.ome.tif"})
    assert find_raw(locs, str(tmp_path / "movie_locs.h5")) == raw
    assert find_raw(locs, None) is None


def test_mean_positions_keep_the_sum_unbiased_when_every_molecule_sits_alike_in_its_pixel():
    """Bin centres move a 10 nm offset by up to half a bin when the predicted positions all lie
    at one place within their pixels; each bin at its pixels' mean offset does not."""
    rng = np.random.default_rng(2)
    r, n, true = 5, 40000, np.array([0.1, -0.04])
    delta = np.tile([0.2, 0.1], (n, 1))
    k = np.arange(-r, r + 1)
    s = np.sqrt(2) * SIGMA
    cx, cy = delta[:, 0] + true[0], delta[:, 1] + true[1]
    ex = 0.5 * (erf((k[None, :] + 0.5 - cx[:, None]) / s) - erf((k[None, :] - 0.5 - cx[:, None]) / s))
    ey = 0.5 * (erf((k[None, :] + 0.5 - cy[:, None]) / s) - erf((k[None, :] - 0.5 - cy[:, None]) / s))
    signal = rng.poisson(300 * ey[:, :, None] * ex[:, None, :] + 20.0) - 20.0
    summed = PatchSum(r)
    summed.add(signal, np.zeros_like(signal), delta[:, 0], delta[:, 1])
    found = fit_spot(summed)
    np.testing.assert_allclose([found.x, found.y], true, atol=0.01)
    # the same sum fitted at the bins' centres is half a bin off
    summed.su[:] = 0
    summed.sv[:] = 0
    centres = PatchSum(r)
    centres += summed
    centres.n = summed.n.copy()
    u, v = np.meshgrid(*(2 * [(np.arange(summed.n.shape[0]) + 0.5) / summed.fine - r - 0.5]))
    centres.su, centres.sv = u * summed.n, v * summed.n
    assert abs(fit_spot(centres).x - true[0]) > 0.05


def test_a_shift_and_an_affine_correction_are_fitted_exactly_from_exact_offsets():
    rng = np.random.default_rng(0)
    place = rng.uniform(0, 60, (9, 2))
    a, b = np.array([[0.002, -0.001], [0.001, -0.003]]), np.array([0.1, -0.2])
    residual = place @ a.T + b
    matrix, after = fit_correction(place, residual, np.full((9, 2), 0.01), "affine")
    np.testing.assert_allclose(matrix[:2, :2], np.eye(2) + a, atol=1e-9)
    np.testing.assert_allclose(np.abs(after).max(), 0, atol=1e-9)
    shift, _ = fit_correction(place, np.tile(b, (9, 1)), np.full((9, 2), 0.01), "shift")
    np.testing.assert_allclose(shift[:2, 2], b, atol=1e-12)
