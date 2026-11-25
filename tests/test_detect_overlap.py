"""The loop around the fit made faster without moving a number: the photon conversion and the 4Pi
channel sum in threaded C++ (`csrc/photons.hpp`), and each block fitted in a worker while the next is
found (`LocalizationEngine.overlap`, `SplitEngine.overlap`) -- the same photons to the last bit, the
same table in the same order."""
from dataclasses import replace

import numpy as np
import pytest

from smappy.camera import convert, fit_photons
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings, LocalizationEngine, drive
from smappy.plugins.fit import DetectionSettings
from smappy.psf import GaussianPSF
from smappy.simulate import scmos_frames, scmos_maps

CONVERSION = 0.5


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def numpy_photons(frames, offset, gain, excess):
    """The conversion as it was written in numpy before the kernel."""
    out = np.asarray(frames, np.float32) - offset
    out *= np.float32(gain)
    return out / excess


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8, np.int16, np.int32, np.uint32,
                                   np.float32, np.float64])
def test_the_threaded_conversion_is_numpys_to_the_last_bit(dtype):
    rng = np.random.default_rng(1)
    frames = (rng.random((7, 33, 29)) * 250).astype(dtype)
    for excess in (1.0, float(np.sqrt(2.0)), 1.3):
        got = convert(frames, 100.3, 0.4917, excess)
        assert got.dtype == np.float32 and got.shape == frames.shape
        assert np.array_equal(bits(got), bits(numpy_photons(frames, np.float32(100.3), 0.4917, excess)))
    per_pixel = (rng.random((33, 29)) * 20 + 90).astype(np.float32)
    assert np.array_equal(bits(convert(frames, per_pixel, 0.4917)),
                          bits(numpy_photons(frames, per_pixel, 0.4917, 1.0)))
    assert np.array_equal(bits(convert(frames[0], 100.3, 0.4917, 1.3)),
                          bits(numpy_photons(frames[0], np.float32(100.3), 0.4917, 1.3)))


def test_a_layouts_offset_map_converts_each_channel_by_its_own_pixels():
    rng = np.random.default_rng(2)
    frames = rng.integers(90, 400, (5, 4, 16, 18)).astype(np.uint16)
    offset = (rng.random((4, 16, 18)) * 10 + 95).astype(np.float32)
    assert np.array_equal(bits(convert(frames, offset, 0.7, n_threads=3)),
                          bits(numpy_photons(frames, offset, 0.7, 1.0)))
    # a type the kernel does not read goes through numpy, the same way
    assert np.array_equal(bits(convert(frames.astype(np.int64), offset, 0.7)),
                          bits(numpy_photons(frames, offset, 0.7, 1.0)))


def test_an_em_cameras_photons_are_divided_by_the_excess_noise_as_before():
    em = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1, em_on=True, emgain=100.0)
    assert em.excess_noise != 1.0
    frames = np.random.default_rng(3).integers(80, 2000, (3, 20, 20)).astype(np.uint16)
    before = numpy_photons(frames, np.float32(em.offset), em.adu_to_photons, em.excess_noise)
    assert np.array_equal(bits(fit_photons(frames, em)), bits(before))


def test_the_threaded_channel_sum_is_the_loops_to_the_last_bit():
    """`multichannel.detection_image`: a rotated, scaled and a moved channel, read by nearest pixel,
    and one on channel 0's pixels, summed in C++ as the numpy loop summed them."""
    from smappy.channels import ChannelLayout, detection_lookup, onto_reference
    from smappy.multichannel import _channel_sum_kernel, detection_image
    h = w = 40
    angle = np.deg2rad(1.5)
    turn = np.array([[np.cos(angle), -np.sin(angle), 1.2], [np.sin(angle), np.cos(angle), -0.7], [0, 0, 1]])
    maps = [np.eye(3), turn, np.diag([1.01, 0.99, 1.0]) + np.array([[0, 0, 3.4], [0, 0, 0], [0, 0, 0]]),
            np.eye(3)]
    layout = ChannelLayout([(w * c, 0, w, h) for c in range(4)], maps=maps)
    channels = np.random.default_rng(4).random((6, 4, h, w)).astype(np.float32) * 50
    lookups = [None] + [detection_lookup(layout, c, (h, w)) for c in range(1, 4)]
    assert _channel_sum_kernel(channels, layout, lookups) is not None
    loop = np.array(channels[:, 0], np.float32)
    for c in range(1, 4):
        loop += channels[:, c] if c == 3 else onto_reference(channels[:, c], lookups[c])
    assert np.array_equal(bits(detection_image(channels, layout, lookups, n_threads=4)), bits(loop))
    # bilinear has no kernel: the loop, as before
    assert _channel_sum_kernel(channels, layout,
                               [None] + [detection_lookup(layout, c, (h, w), "bilinear")
                                         for c in range(1, 4)]) is None


def movie(n_frames=120):
    offset, noise = scmos_maps((48, 48), seed=1, conversion=CONVERSION, hot_fraction=0.0)
    frames, _ = scmos_frames(offset, noise, seed=5, n_frames=n_frames, photons=1500, background=8,
                             conversion=CONVERSION)
    return frames, offset, noise


def run(engine, frames, block=10):
    parts = []
    drive(engine, ((i, frames[i:i + block]) for i in range(0, len(frames), block)),
          sink=parts.append, read_ahead=0)
    return parts


def same_tables(plain, overlapped):
    assert len(plain) > 2 and len(overlapped) == len(plain)
    for a, b in zip(plain, overlapped):
        assert sorted(a.columns) == sorted(b.columns)
        for name in a.columns:
            assert np.array_equal(np.asarray(a[name]), np.asarray(b[name]), equal_nan=True), name


@pytest.mark.parametrize("per_pixel", [False, True])
def test_fitting_while_the_next_block_is_found_changes_nothing(tmp_path, per_pixel):
    """`LocalizationEngine.overlap`: each block fitted in a worker while the next is converted,
    searched and cut, its localizations handed back a block later and the last by `end` -- the same
    table, in the same order, with and without an sCMOS noise map."""
    from smappy import scmos
    frames, offset, noise = movie()
    camera = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1)
    if per_pixel:
        path = scmos.NoiseMap(offset, noise ** 2, region=(0, 0, 48, 48)).save(tmp_path / "noise.h5")
        camera = replace(camera, noise_map=str(path))
    tables = []
    for overlap in (False, True):
        settings = FitSettings(max_block_rois=60, per_pixel_noise=per_pixel)   # a fit every few blocks
        engine = LocalizationEngine(camera, DetectionSettings().finder(), GaussianPSF(sigma=1.3),
                                    settings, overlap=overlap)
        tables.append(run(engine, frames))
        assert engine.stats["localizations"] == sum(len(p) for p in tables[-1])
    same_tables(*tables)


def test_a_field_dependent_spline_fitted_while_the_next_block_is_found_changes_nothing():
    """Each ROI's place on the chip goes with its batch into the worker."""
    from smappy.calibrate.core import spline_coefficients
    from smappy.field_splines import FieldSplines, blend
    from smappy.io.calibration import SplineCalibration
    from smappy.psf import SplinePSF
    nz, box = 21, 15
    z = np.linspace(-1, 1, nz)[:, None, None]
    y, x = np.mgrid[:box, :box] - (box - 1) / 2

    def stack(e):
        sx, sy = 1.3 + 0.8 * (z - e) ** 2, 1.3 + 0.8 * (z + e) ** 2
        return (np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)).astype(np.float32)

    xs = ys = np.array([0.0, 48.0])
    field = FieldSplines(np.stack([spline_coefficients(stack(0.1 + 0.3 * xx / 48)) for yy in ys for xx in xs]),
                         xs, ys)
    cal = SplineCalibration(blend(field, (24, 24)), 100.0, nz // 2, x0=(box - 1) / 2, field_splines=field)
    model = SplinePSF(cal)
    assert model.field_dependent
    frames, _, _ = movie(60)
    camera = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1, roi=(4, 6, 48, 48))
    tables = [run(LocalizationEngine(camera, DetectionSettings().finder(), model,
                                     FitSettings(roisize=11, max_block_rois=60), overlap=overlap), frames)
              for overlap in (False, True)]
    same_tables(*tables)


def test_a_split_cameras_channels_fitted_while_the_next_block_is_found_change_nothing():
    """`SplitEngine.overlap`: the channels' fits of one push go to the worker together, so the
    joined blocks -- sorted by frame within each -- are the ones it made before."""
    from smappy.channels import ChannelLayout
    from smappy.multichannel import SplitEngine
    frames, _, _ = movie()
    camera = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1)
    layout = ChannelLayout([(0, 0, 24, 48), (24, 0, 24, 48)])
    tables = []
    for overlap in (False, True):
        finder = DetectionSettings().finder()
        engine = SplitEngine(camera, [finder, finder], GaussianPSF(sigma=1.3), layout,
                             FitSettings(max_block_rois=40), overlap=overlap)
        tables.append(run(engine, frames))
    same_tables(*tables)


def test_an_overlapped_engine_hands_back_every_block_when_flushed_at_the_end():
    frames, _, _ = movie(30)
    camera = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1)
    engine = LocalizationEngine(camera, DetectionSettings().finder(), GaussianPSF(sigma=1.3),
                                FitSettings(), overlap=True)
    assert engine.push(frames, 0) is None              # buffered, nothing fitted yet
    assert engine.flush() is None                      # its fit started in the worker
    got = engine.end()
    assert len(got) and engine.end() is None
