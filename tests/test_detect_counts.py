"""Peak finding on the raw counts (`detect.Counts`): the frames filtered as the camera wrote them, the
offset cancelling in the difference of Gaussians and the gain scaling the result, and only the boxes
converted to photons -- the candidates of the photons, the same boxes to the last bit, the same table.
"""
import numpy as np
import pytest

import smappy.pipeline
from smappy.detect import (
    AbsoluteCutoff,
    Counts,
    DoGFilter,
    DynamicCutoff,
    GaussFilter,
    PeakFinder,
)
from smappy.fit_rois import cut_rois
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings, LocalizationEngine, raw_counts
from smappy.psf import GaussianPSF


def spots(n_frames=30, shape=(64, 72), per=12, seed=0, offset=100.0, gain=0.5):
    """Raw uint16 frames of Gaussian spots on a flat background: counts = photons / gain + offset."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    frames = []
    for _ in range(n_frames):
        image = np.full(shape, 20.0)
        for x, y, n in zip(rng.uniform(4, shape[1] - 4, per), rng.uniform(4, shape[0] - 4, per),
                           rng.uniform(300, 3000, per)):
            image += n / (2 * np.pi * 1.3 ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3 ** 2))
        frames.append(rng.poisson(image) / gain + offset)
    return np.round(frames).astype(np.uint16)


def keys(c):
    return list(zip(c.frame.tolist(), c.x.tolist(), c.y.tolist()))


@pytest.mark.parametrize("finder", [
    PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7)),
    PeakFinder(DoGFilter(1.2), AbsoluteCutoff(8.0)),
    PeakFinder(GaussFilter(1.2), DynamicCutoff(1.7)),
    PeakFinder(GaussFilter(1.2), AbsoluteCutoff(28.0)),
    PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7), split=(1, 36)),
], ids=str)
@pytest.mark.parametrize("excess", [1.0, np.sqrt(2.0)])
def test_the_counts_give_the_photons_candidates_with_their_values_in_photons(finder, excess):
    raw = spots()
    counts = Counts(100.0, 0.5, excess)
    on_photons, filtered = finder(counts.photons(raw), first_frame=7)
    on_counts, none = finder(raw, first_frame=7, counts=counts)
    assert none is None and filtered is not None
    assert len(on_photons) > 200 and keys(on_counts) == keys(on_photons)
    assert on_counts.value.dtype == np.float32
    assert np.allclose(on_counts.value, on_photons.value, rtol=1e-4, atol=1e-3)


def test_a_filter_that_does_not_say_what_it_makes_of_an_offset_is_given_photons():
    class Smooth:
        def __call__(self, images, n_threads=0):
            return GaussFilter(1.0)(images, n_threads)
    finder = PeakFinder(Smooth(), AbsoluteCutoff(28.0))
    raw = spots(5)
    counts = Counts(100.0, 0.5)
    assert keys(finder(raw, counts=counts)[0]) == keys(finder(counts.photons(raw))[0])


def test_boxes_cut_from_the_counts_are_the_photons_boxes_to_the_last_bit():
    raw = spots(8)
    counts = Counts(100.3, 0.4917, 1.3)
    candidates, _ = PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7))(raw, first_frame=3, counts=counts)
    a = cut_rois(counts.photons(raw), candidates, 9, first_frame=3)
    b = cut_rois(raw, candidates, 9, first_frame=3, counts=counts)
    assert b.images.dtype == np.float32 and len(a) > 20
    assert np.array_equal(a.images.view(np.uint32), b.images.view(np.uint32))


def test_only_one_offset_for_all_pixels_is_searched_in_counts():
    camera = CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1)
    raw = spots(2)
    assert raw_counts(raw, camera) == Counts(100.0, 0.5, 1.0)
    assert raw_counts(raw, camera, noise=object()) is None           # a noise map's offsets
    per_pixel = CameraMetadata(conversion=0.5, offset=np.full(raw.shape[1:], 100.0), pixelsize_um=0.1)
    assert raw_counts(raw, per_pixel) is None


def test_the_engine_on_the_counts_fits_the_table_it_fitted_on_the_photons(monkeypatch):
    camera = CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1)
    raw = spots(40)

    def table():
        engine = LocalizationEngine(camera, PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7)),
                                    GaussianPSF(sigma=1.3), FitSettings(roisize=9))
        for i in range(0, len(raw), 10):
            engine.detect(raw[i:i + 10], i)
        return engine.flush()
    counted = table()
    monkeypatch.setattr(smappy.pipeline, "raw_counts", lambda *a, **k: None)
    photons = table()
    assert len(photons) > 300 and sorted(counted.columns) == sorted(photons.columns)
    for name in photons.columns:
        assert np.array_equal(np.asarray(counted[name]), np.asarray(photons[name]), equal_nan=True), name


def test_the_paired_two_colour_engine_finds_and_fits_the_same_on_the_counts(monkeypatch):
    """Gaussian 2D 2C's engine: each channel searched in its counts, the boxes of both converted."""
    from test_dual_gauss_fit import movie, paired_engine, transform

    from smappy.psf import GlobalGaussianPSF
    rng = np.random.default_rng(4)
    t = transform()
    frames, _ = movie(30, rng, t)
    raw = (frames + 100).astype(np.uint16)

    def table():
        engine = paired_engine(PeakFinder(DoGFilter(1.2), DynamicCutoff(1.7)), GlobalGaussianPSF(sigma=1.2),
                               t, FitSettings(roisize=13))
        engine.push(raw, first_frame=0)
        return engine.flush()
    counted = table()
    monkeypatch.setattr(smappy.pipeline, "raw_counts", lambda *a, **k: None)
    photons = table()
    assert len(photons) > 60 and len(counted) == len(photons)
    for name in photons.columns:
        assert np.array_equal(np.asarray(counted[name]), np.asarray(photons[name]), equal_nan=True), name
