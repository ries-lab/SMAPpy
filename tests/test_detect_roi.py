"""Detection and ROI cutting on synthetic images with known peak positions."""
import numpy as np
import pytest

from smappy.detect import (AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter,
                            PeakFinder, find_candidates, local_maxima)
from smappy.roi import cut_rois


def _spots(positions, shape=(64, 60), amplitude=100.0, sigma=1.2, background=10.0):
    """An image with Gaussian spots at (x, y) positions; note x is the column."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, background, np.float32)
    for x, y in positions:
        img += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img


@pytest.mark.parametrize("use_extension", [True, False])
def test_local_maxima_are_strict(use_extension):
    img = np.zeros((1, 9, 9), np.float32)
    img[0, 4, 4] = 1.0
    f, y, x, v = local_maxima(img, use_extension=use_extension)
    assert y.tolist() == [4] and x.tolist() == [4] and v.tolist() == [1.0]
    # a plateau has no strict maximum, as in SMAP's maximumfind2.c
    img[0, 4, 5] = 1.0
    assert local_maxima(img, use_extension=use_extension)[0].size == 0


def test_extension_and_numpy_maxima_agree():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (4, 40, 50)).astype(np.float32)
    fast = local_maxima(img, use_extension=True)
    reference = local_maxima(img, use_extension=False)
    for a, b in zip(fast, reference):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_dog_backends_agree():
    """The C++ filter must match the scipy reference to float precision."""
    rng = np.random.default_rng(1)
    img = rng.uniform(50, 500, (3, 64, 60)).astype(np.float32)
    fast = DoGFilter(1.2, use_extension=True)(img)
    reference = DoGFilter(1.2, use_extension=False)(img)
    assert np.abs(fast - reference).max() < 1e-3 * np.abs(reference).max()


def test_candidate_coordinates_are_x_column_y_row():
    """A single off-centre spot must come back with x = column, y = row."""
    img = _spots([(40, 12)])
    cands = find_candidates(GaussFilter(1.2)(img[None]), AbsoluteCutoff(20.0))
    assert len(cands) == 1
    assert cands.x[0] == 40 and cands.y[0] == 12


def test_border_candidates_are_dropped_not_clamped():
    # the second spot is detected, but a 13x13 ROI around it would not fit
    img = _spots([(30, 30), (3, 30)])
    finder = PeakFinder(GaussFilter(1.2), AbsoluteCutoff(20.0))
    cands, _ = finder(img[None])
    rois = cut_rois(img[None], cands, roisize=13)
    assert len(cands) == 2
    assert len(rois) == 1  # the border one is gone, not shifted inward
    assert rois.candidates.x[0] == 30


def test_rois_are_centred_on_the_candidate():
    positions = [(20, 15), (40, 33), (10, 50)]
    img = _spots(positions)
    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(10.0))
    cands, _ = finder(img[None])
    rois = cut_rois(img[None], cands, roisize=13)
    centre = rois.images[:, 6, 6]
    assert np.all(centre > rois.images[:, 0, 0])
    # the fitted-position mapping must return the candidate for a centred fit
    x_back = rois.to_image_x(np.full(len(rois), 6.0))
    assert np.allclose(x_back, rois.candidates.x)


def test_mirror_flips_and_maps_back():
    img = _spots([(30, 30)])
    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(10.0))
    cands, _ = finder(img[None])
    plain = cut_rois(img[None], cands, roisize=13)
    flipped = cut_rois(img[None], cands, roisize=13, mirror=True)
    assert np.allclose(flipped.images, plain.images[:, :, ::-1])
    # a fit at ROI column 4 in the flipped stack is column 8 in the image
    assert np.allclose(flipped.to_image_x([4.0]), plain.to_image_x([8.0]))


def test_frames_are_indexed_absolutely():
    img = np.stack([_spots([(20, 20)]), _spots([(35, 25)])])
    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(10.0))
    cands, _ = finder(img, first_frame=1000)
    assert set(cands.frame.tolist()) == {1000, 1001}
    rois = cut_rois(img, cands, roisize=13, first_frame=1000)
    assert len(rois) == 2
    with pytest.raises(ValueError, match="do not fit"):
        cut_rois(img, cands, roisize=13, first_frame=0)


def test_dog_kernel_removes_constant_background():
    flat = np.full((1, 40, 40), 500.0, np.float32)
    assert np.abs(DoGFilter(1.2)(flat)).max() < 1e-3


def test_dynamic_cutoff_scales_with_the_distribution():
    rng = np.random.default_rng(0)
    weak = DynamicCutoff(1.7)(rng.normal(10, 1, 1000))
    strong = DynamicCutoff(1.7)(rng.normal(100, 10, 1000))
    assert strong > weak
    assert DynamicCutoff(3.0)(rng.normal(10, 1, 1000)) > weak


def test_the_dynamic_cutoff_takes_smaps_nearest_rank_quantiles():
    """20 maxima: the 4th, 10th and 16th smallest, as SMAP's myquantilefast
    picks them.  np.quantile would interpolate to 0.88, 1.35 and 2.42."""
    values = np.array([0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                       1.6, 1.8, 2.0, 2.3, 2.9, 3.5, 41.0, 95.0], np.float32)
    q20, q50, q80 = 0.8, 1.3, 2.3
    expected = q50 + 1.7 * (q80 - q20) / 0.6
    assert DynamicCutoff(1.7)(values) == pytest.approx(expected, abs=1e-5)
    starts = np.array([0, values.size])
    assert DynamicCutoff(1.7).thresholds(values, starts)[0] == pytest.approx(expected,
                                                                              abs=1e-5)


@pytest.mark.parametrize("extension", [True, False])
def test_the_cutoff_of_a_block_is_the_cutoff_of_each_frame_bit_for_bit(
        extension, monkeypatch):
    """Empty frames, frames under ten maxima, ties, and values far from 1."""
    if not extension:
        monkeypatch.setattr("smappy.detect._fit3d", None)
    rng = np.random.default_rng(3)
    cutoff = DynamicCutoff(1.7)
    for trial in range(60):
        counts = rng.integers(0, 2000 if trial % 2 else 25, rng.integers(1, 40))
        starts = np.concatenate([[0], np.cumsum(counts)])
        values = (rng.gamma(2, 3, starts[-1]) * 10.0 ** rng.integers(-3, 4))
        if trial % 3 == 0:
            values = np.round(values, 1)
        values = values.astype(np.float32)
        block = cutoff.thresholds(values, starts)
        for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            if b > a:
                assert block[i] == np.float32(cutoff(values[a:b]))


@pytest.mark.parametrize("split", [None, (1, 30), (0, 17)])
def test_detection_keeps_what_the_per_frame_loop_keeps(split):
    """A cutoff that is not a DynamicCutoff still takes the loop: the reference."""
    rng = np.random.default_rng(4)
    images = rng.normal(20, 3, (20, 48, 60)).astype(np.float32)
    for frame in images:
        frame += _spots(rng.uniform(5, 55, (4, 2)), shape=(48, 60), background=0.0)
    images[3] = 0.0                                  # a frame with no maxima
    images[5] = _spots([(20, 20), (40, 30)], shape=(48, 60))  # one with two
    filtered = DoGFilter(1.2)(images)
    cutoff = DynamicCutoff(1.7)
    fast = find_candidates(filtered, cutoff, split=split)
    loop = find_candidates(filtered, lambda v: cutoff(v), split=split)
    assert len(fast) > 0
    for name in ("frame", "x", "y", "value"):
        assert np.array_equal(getattr(fast, name), getattr(loop, name))


@pytest.mark.parametrize("n_threads", [1, 2, 8])
def test_threading_does_not_change_detection(n_threads):
    """Threaded filtering and peak finding must be bit-identical to serial."""
    rng = np.random.default_rng(7)
    images = rng.uniform(50, 500, (12, 64, 60)).astype(np.float32)
    for x, y in [(10, 12), (30, 40), (50, 20)]:
        images += _spots([(x, y)], shape=(64, 60), amplitude=300.0)[None]

    serial = DoGFilter(1.2)(images, n_threads=1)
    threaded = DoGFilter(1.2)(images, n_threads=n_threads)
    assert np.array_equal(serial, threaded)

    a = local_maxima(serial, n_threads=1)
    b = local_maxima(serial, n_threads=n_threads)
    for one, many in zip(a, b):
        assert np.array_equal(np.asarray(one), np.asarray(many))


def test_maxima_stay_sorted_by_frame_when_threaded():
    """The per-frame cutoff relies on maxima being grouped by frame."""
    rng = np.random.default_rng(8)
    images = rng.normal(0, 1, (16, 50, 50)).astype(np.float32)
    frames = local_maxima(images, n_threads=8)[0]
    assert np.all(np.diff(frames) >= 0)


@pytest.mark.parametrize("n_threads", [1, 4])
def test_candidates_are_thread_independent(n_threads):
    rng = np.random.default_rng(9)
    images = rng.uniform(10, 100, (8, 48, 48)).astype(np.float32)
    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(5.0))
    a = finder(images, n_threads=1)[0]
    b = finder(images, n_threads=n_threads)[0]
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert np.array_equal(a.frame, b.frame)
