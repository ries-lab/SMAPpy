"""Detection on each frame less the temporal median of the frames about it
(Picasso's `TemporalMedianMovie`): `detect.TemporalMedian`, the fitters'
*temporal median*, fitted always on the photons."""
import numpy as np
import pytest
import tifffile
from scipy import ndimage
from scipy.special import erf

from smappy.detect import (DoGFilter, DynamicCutoff, PeakFinder, TemporalMedian,
                           median_bounds)
from smappy.plugins import Context
from smappy.plugins.fit import (CameraSettings, DetectionSettings, GaussianFit,
                                GaussianFitSettings, OutputSettings, SourceSettings)


def test_the_tiles_are_picassos_shifted_inwards_at_the_ends():
    # stride = window: a tile's median is over the tile itself
    assert median_bounds(0, 51, 51, 0, 1000) == (0, 51)
    assert median_bounds(60, 51, 51, 0, 1000) == (51, 102)
    # the last tile, cut short by the movie's end, takes the last 51 frames
    assert median_bounds(990, 51, 51, 0, 1000) == (949, 1000)
    # stride 1: centred on the frame, shifted at the start
    assert median_bounds(100, 51, 1, 0, 1000) == (75, 126)
    assert median_bounds(3, 51, 1, 0, 1000) == (0, 51)
    # shorter than a window: what there is
    assert median_bounds(5, 51, 51, 0, 20) == (0, 20)


def test_a_static_uneven_background_goes_and_a_short_spot_stays():
    rng = np.random.default_rng(0)
    background = rng.uniform(10, 500, (16, 16)).astype(np.float32)
    frames = np.repeat(background[None], 120, axis=0)
    frames[40:44, 8, 8] += 300.0               # on for four frames
    median = TemporalMedian(51)
    out = np.concatenate([median(frames[i:i + 30], i) for i in range(0, 120, 30)])
    assert np.allclose(out[40:44, 8, 8], 300.0)
    out[40:44, 8, 8] = 0
    assert np.allclose(out, 0.0)


def test_in_blocks_the_median_is_the_whole_movies_but_at_the_end_of_a_block():
    """A tile cut by a block's end takes the frames up to it, and the next
    block, which has the rest, its own window again."""
    rng = np.random.default_rng(1)
    frames = rng.poisson(50.0, (400, 8, 8)).astype(np.float32)
    whole = TemporalMedian(51)(frames, 0)
    median = TemporalMedian(51)
    blocks = np.concatenate([median(frames[i:i + 100], i) for i in range(0, 400, 100)])
    differs = np.flatnonzero(np.any(blocks != whole, axis=(1, 2)))
    # the tiles 51-101, 153-203, 255-305 straddle 100, 200, 300: their
    # frames before the block's end
    assert set(differs) <= set(range(51, 100)) | set(range(153, 200)) | set(range(255, 300))
    assert len(differs) > 0


def test_frames_already_held_are_answered_from_them_and_a_jump_starts_again():
    rng = np.random.default_rng(2)
    frames = rng.poisson(50.0, (51, 4, 4)).astype(np.float32)
    median = TemporalMedian(51)
    window = median(frames, 100)
    assert np.array_equal(median(frames[[7]], 107), window[[7]])      # a Preview's frame
    alone = median(frames[[7]], 500)                                   # no frames about it
    assert np.allclose(alone, 0.0)


# ------------------------------------------------- does it detect better?

def scene(n_frames=500, size=64, amp=150.0, base=30.0, photons=800.0,
          per_frame=8, sigma=1.3, seed=0):
    """Spots of a few frames each (``photons`` a frame) on a background of
    ``base`` photons plus up to ``amp`` in patches about two pixels across,
    under a broad hump, all of it swelling and fading by 40% over the
    movie: what a difference of Gaussians cannot remove.  Returns the
    photons and the truth, ``(frame, x, y)``."""
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.normal(size=(size, size)), 2.0)
    field = (field - field.min()) / (field.max() - field.min())
    yy, xx = np.mgrid[:size, :size]
    hump = np.exp(-((xx - 0.3 * size) ** 2 + (yy - 0.6 * size) ** 2) / (2 * (0.3 * size) ** 2))
    slow = 1 + 0.4 * np.cos(2 * np.pi * np.arange(n_frames) / n_frames)
    image = base + amp * (field * hump)[None] * slow[:, None, None]
    n_events = per_frame * n_frames // 4
    start = rng.integers(0, n_frames, n_events)
    length = rng.geometric(1 / 4, n_events)           # four frames on average
    x, y = rng.uniform(4, size - 4, n_events), rng.uniform(4, size - 4, n_events)
    event = np.repeat(np.arange(n_events), length)
    frame = start[event] + np.concatenate([np.arange(n) for n in length])
    event, frame = event[frame < n_frames], frame[frame < n_frames]
    edges, s = np.arange(size + 1) - 0.5, np.sqrt(2) * sigma
    for e, f in zip(event, frame):
        ex = np.diff(erf((edges - x[e]) / s)) / 2
        ey = np.diff(erf((edges - y[e]) / s)) / 2
        image[f] += photons * np.outer(ey, ex)
    return rng.poisson(image).astype(np.float32), (frame, x[event], y[event])


def score(candidates, truth, radius=1.5):
    """``(found, false, missed)``: candidates matched one to one to a true
    spot of their frame within ``radius`` pixels, nearest first."""
    frame, x, y = truth
    found = 0
    for f in np.unique(frame):
        tx, ty = x[frame == f], y[frame == f]
        mine = candidates.frame == f
        cx, cy = candidates.x[mine], candidates.y[mine]
        if not len(cx):
            continue
        d = np.hypot(cx[:, None] - tx[None], cy[:, None] - ty[None])
        taken = set()
        for i in np.argsort(d.min(1)):
            j = next((j for j in np.argsort(d[i]) if d[i, j] <= radius and j not in taken), None)
            if j is not None:
                taken.add(j)
                found += 1
    return found, len(candidates) - found, len(frame) - found


@pytest.mark.parametrize("amp", [150.0, 400.0])
def test_under_an_uneven_background_the_median_finds_what_the_filter_alone_misses(amp):
    photons, truth = scene(amp=amp)
    finder = PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7))
    plain, _ = finder(photons)
    median = TemporalMedian(51)
    detect = np.concatenate([median(photons[i:i + 200], i) for i in range(0, len(photons), 200)])
    better, _ = finder(detect)
    found0, false0, missed0 = score(plain, truth)
    found1, false1, missed1 = score(better, truth)
    n = len(truth[0])
    # the patches set the dynamic cutoff, which then misses most spots;
    # less their median, the frames have only the spots on a flat floor
    assert missed0 > 0.4 * n and missed1 < 0.06 * n
    assert false1 + missed1 < 0.25 * (false0 + missed0)
    assert false1 < 0.02 * n


def test_on_a_flat_background_the_median_changes_little():
    photons, truth = scene(amp=0.0)
    finder = PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7))
    median = TemporalMedian(51)
    plain = score(finder(photons)[0], truth)
    better = score(finder(median(photons, 0))[0], truth)
    assert abs(better[0] - plain[0]) < 0.01 * len(truth[0])


# ------------------------------------------------------------ in the fitter

@pytest.fixture
def movie(tmp_path):
    """300 frames of the scene in ADU (offset 100, 1 photon a count)."""
    photons, truth = scene(n_frames=300, amp=300.0, per_frame=4)
    path = tmp_path / "paint.tif"
    tifffile.imwrite(path, (photons + 100).astype(np.uint16))
    return path, truth


def settings_for(path, tmp_path, on: bool):
    return GaussianFitSettings(
        source=SourceSettings(path=str(path), chunk=100),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1, em_on=False),
        detection=DetectionSettings(temporal_median=on),
        output=OutputSettings(path=str(tmp_path / f"out_{on}.hdf5")))


def test_the_fitter_detects_on_the_median_and_fits_the_photons(movie, tmp_path):
    path, truth = movie
    plain = GaussianFit().run(Context(), settings_for(path, tmp_path, False)).locs
    locs = GaussianFit().run(Context(), settings_for(path, tmp_path, True)).locs
    assert len(locs) > 1.5 * len(plain)
    # fitted on the photons, background included: away from the patches it
    # is the 30 photons there, not the 0 of the detection image, and the
    # spots have their 800 photons
    x, y = np.asarray(locs["x_nm"]) / 100, np.asarray(locs["y_nm"]) / 100
    flat = np.hypot(x - 0.3 * 64, y - 0.6 * 64) > 30
    assert flat.sum() > 100
    assert 20 < np.median(np.asarray(locs["background"])[flat]) < 70
    assert abs(np.median(np.asarray(locs["photons"])[flat]) / 800 - 1) < 0.15
    from smappy.io import formats
    saved, _ = formats.load(tmp_path / "out_True.hdf5")
    assert saved.metadata["detection"]["background"] == "temporal median of 51 frames, every 51"


def test_the_preview_detects_on_the_frames_median_as_the_run_does(movie, tmp_path):
    path, truth = movie
    index = 120
    frame, x, y = truth
    n_true = int((frame == index).sum())
    settings = settings_for(path, tmp_path, True)
    result = GaussianFit().preview(Context(), settings, frame=index)
    assert result.data["error"] == ""
    assert abs(result.data["candidates"] - n_true) <= 1
    locs = result.data["locs"]
    assert set(np.asarray(locs["frame"])) <= {index} and len(locs) >= n_true - 1


def test_the_window_settings_are_greyed_while_the_median_is_off(movie, tmp_path):
    path, _ = movie
    state = GaussianFit().describe(None, settings_for(path, tmp_path, False))
    assert state.active["detection.median_window"] is False
    state = GaussianFit().describe(None, settings_for(path, tmp_path, True))
    assert state.active["detection.median_stride"] is True


def test_a_split_chip_keeps_a_median_per_channel_in_the_run_and_the_preview(movie, tmp_path):
    """On *two channels* each half is detected by an engine of its own, with
    its own median; the median is pixel by pixel, so the Preview's, of the
    frame before it is cut, is the same."""
    path, truth = movie
    settings = settings_for(path, tmp_path, True)
    settings.channels.split = "two channels"
    settings.channels.layout, settings.channels.main_channel = "up-down", "upper"
    settings.output.save = False
    locs = GaussianFit().run(Context(), settings).locs
    assert set(np.asarray(locs["camera_channel"])) == {0, 1}
    index = 120
    result = GaussianFit().preview(Context(), settings, frame=index)
    assert result.data["error"] == ""
    run = int((np.asarray(locs["frame"]) == index).sum())
    assert abs(len(result.data["locs"]) - run) <= 1
