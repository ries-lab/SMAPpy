"""sCMOS noise maps: the calibration from dark frames, the map cut to a file,
the per-pixel likelihood in the fitter, and the map's place in the camera
database (`smappy.scmos`)."""
from dataclasses import replace

import numpy as np
import pytest

from smappy import _fit3d, camera_db, scmos
from smappy.camera_db import Camera, Parameter, State
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings, LocalizationEngine
from smappy.plugins.fit import DetectionSettings
from smappy.psf import GaussianPSF
from smappy.simulate import dark_frames, scmos_frames, scmos_maps

CONVERSION = 0.5


def spots(n=4000, sz=13, photons=1000.0, background=10.0, seed=1):
    """Gaussian spots near the centre of their ROIs, Poisson noise, in photons."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:sz, :sz]
    x0 = sz // 2 + rng.uniform(-1, 1, n)
    y0 = sz // 2 + rng.uniform(-1, 1, n)
    mu = background + photons / (2 * np.pi * 1.3 ** 2) * np.exp(
        -((xx - x0[:, None, None]) ** 2 + (yy - y0[:, None, None]) ** 2) / (2 * 1.3 ** 2))
    return rng.poisson(mu).astype(np.float32), x0, y0


def a_map(tmp_path, shape=(64, 64), hot_fraction=0.03, region=(0, 0)):
    offset, noise = scmos_maps(shape, seed=1, conversion=CONVERSION, hot_fraction=hot_fraction)
    measured = scmos.dark_map([dark_frames(offset, noise, 1000, seed=2)], region=region)
    return offset, noise, measured.save(tmp_path / "noise.h5")


# ---------------------------------------------------------------- calibration
def test_moments_combined_by_blocks_are_the_moments_of_the_whole_movie():
    frames = np.random.default_rng(0).normal(100, 3, (257, 5, 7))
    moments = scmos.Moments()
    for start, stop in ((0, 1), (1, 100), (100, 103), (103, 257)):
        moments.add(frames[start:stop])
    assert moments.n == 257
    assert np.allclose(moments.mean, frames.mean(0))
    assert np.allclose(moments.variance(), frames.var(0, ddof=1))


def test_a_dark_movie_gives_each_pixels_offset_and_read_noise():
    offset, noise = scmos_maps((40, 50), seed=3, conversion=CONVERSION)
    frames = dark_frames(offset, noise, 1000, seed=4)
    measured = scmos.dark_map(frames[i:i + 128] for i in range(0, 1000, 128))
    assert measured.n_frames == 1000 and measured.shape == (40, 50)
    # the offset to its standard error (noise / sqrt(N)) and the rounding to whole counts
    assert np.all(np.abs(measured.offset - offset) < 5 * noise / np.sqrt(1000) + 0.5)
    # the variance to sqrt(2 / (N - 1)) = 4.5 %, where the rounding is small beside it
    ratio = measured.variance / (noise ** 2 + 1 / 12)
    assert abs(np.median(ratio) - 1) < 0.01
    assert 0.035 < np.std(ratio) < 0.055
    found = scmos.summary(measured, CONVERSION)
    assert found["read_noise_median"] == pytest.approx(1.0, rel=0.05)
    assert found["hot_fraction"] == pytest.approx(0.01, abs=0.004)


def test_too_few_dark_frames_are_refused():
    offset, noise = scmos_maps((8, 8))
    with pytest.raises(ValueError, match="at least 100"):
        scmos.dark_map([dark_frames(offset, noise, 50)])


def test_a_map_survives_its_file(tmp_path):
    offset, noise = scmos_maps((8, 12))
    measured = scmos.dark_map([dark_frames(offset, noise, 200)], region=(16, 32),
                              info={"camera": "cam", "exposure_ms": 10.0})
    back = scmos.NoiseMap.load(measured.save(tmp_path / "m.h5"))
    assert back.region == (16, 32, 12, 8) and back.n_frames == 200
    assert back.info == {"camera": "cam", "exposure_ms": 10.0}
    assert np.array_equal(back.variance, measured.variance)


# -------------------------------------------------------------------- cutting
def test_a_map_is_cut_at_the_files_camera_roi():
    rng = np.random.default_rng(0)
    noise = scmos.NoiseMap(offset=rng.normal(100, 2, (60, 80)),
                           variance=rng.uniform(1, 9, (60, 80)), region=(100, 200))
    part = noise.crop((110, 230, 20, 10), (10, 20))
    assert part.region == (110, 230, 20, 10)
    assert np.array_equal(part.variance, noise.variance[30:40, 10:30])
    # a file of the map's size that records no ROI is the map's ROI
    assert noise.crop(None, (60, 80)).region == (100, 200, 80, 60)
    with pytest.raises(ValueError, match="only x 100-179"):
        noise.crop((170, 200, 20, 10), (10, 20))
    with pytest.raises(ValueError):
        noise.crop(None, (10, 20))           # taken at the chip's corner: outside


def test_the_variance_is_converted_to_the_fits_units():
    noise = scmos.NoiseMap(offset=np.zeros((2, 2)), variance=np.full((2, 2), 4.0))
    assert np.allclose(noise.in_photons(0.5), 1.0)        # 2 ADU rms = 1 e-


def test_a_fit_without_a_map_or_with_em_gain_says_why(tmp_path):
    with pytest.raises(ValueError, match="no noise map"):
        scmos.for_frames(CameraMetadata(conversion=0.5, offset=100), (10, 10))
    _, _, path = a_map(tmp_path, shape=(16, 16))
    em = CameraMetadata(conversion=0.5, offset=100, em_on=True, emgain=100,
                        noise_map=str(path))
    with pytest.raises(ValueError, match="EM gain"):
        scmos.for_frames(em, (16, 16))


# ------------------------------------------------------------------ the fitter
@pytest.mark.parametrize("name", ["fit_gauss_free", "fit_gauss_xy"])
def test_a_map_of_one_variance_fits_as_the_chip_wide_variance_does(name):
    """The per-pixel likelihood with the same variance everywhere is the
    chip-wide one, whose background took the variance instead."""
    rois, _, _ = spots()
    v = 2.25
    fit = getattr(_fit3d, name)
    chip = fit(rois + v, 1.3, 50, 1)
    pixels = fit(rois + v, 1.3, 50, 1, np.full(rois.shape, v, np.float32))
    assert np.allclose(chip[0][:, :3], pixels[0][:, :3], atol=1e-3, rtol=1e-3)
    assert np.allclose(chip[0][:, 3] - v, pixels[0][:, 3], atol=1e-3)
    assert np.allclose(chip[1][:, :3], pixels[1][:, :3], rtol=1e-3)


def test_a_variance_of_another_shape_is_refused():
    rois, _, _ = spots(n=4)
    with pytest.raises(ValueError, match="shape"):
        _fit3d.fit_gauss_free(rois, 1.3, 50, 1, np.ones((4, 5, 5), np.float32))


def test_a_noisy_pixel_counts_less_and_the_precision_says_so():
    """A pixel with 20 e- of read noise beside each spot: the map weights it
    down, so the fit scatters less, and its Cramer-Rao bound grows to what
    the fit actually reaches."""
    rois, x0, _ = spots(n=4000, photons=800, background=2, seed=5)
    rng = np.random.default_rng(6)
    variance = np.full(rois.shape, 1.0, np.float32)
    variance[:, 6, 8] = 400.0
    noisy = (rois + rng.normal(0, 1, rois.shape) * np.sqrt(variance)).astype(np.float32)
    chip = _fit3d.fit_gauss_free(noisy + 1.0, 1.3, 50, 1)
    pixels = _fit3d.fit_gauss_free(noisy + variance, 1.3, 50, 1, variance)
    spread = {k: 1.4826 * np.median(np.abs(f[0][:, 0] - x0)) for k, f in
              (("chip", chip), ("pixels", pixels))}
    assert spread["pixels"] < 0.8 * spread["chip"]
    bound = np.sqrt(np.median(pixels[1][:, 0]))
    assert bound == pytest.approx(spread["pixels"], rel=0.15)


def test_with_per_pixel_noise_off_a_cameras_map_is_not_read(tmp_path):
    offset, noise, path = a_map(tmp_path, shape=(32, 32))
    frames, _ = scmos_frames(offset, noise, seed=3, n_frames=100, conversion=CONVERSION)
    finder = DetectionSettings().finder(0)
    plain = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1)
    results = []
    for camera in (plain, replace(plain, noise_map=str(path))):
        engine = LocalizationEngine(camera, finder, GaussianPSF(sigma=1.3), FitSettings())
        engine.push(frames, 0)
        results.append(engine.flush())
    assert np.array_equal(results[0]["x_nm"], results[1]["x_nm"])


def test_the_map_keeps_fits_near_hot_pixels_on_their_spots(tmp_path):
    """A simulated sCMOS with 3 % hot pixels: with the map, fewer fits land
    far from the truth and the background is no longer pulled up."""
    from scipy.spatial import cKDTree
    from smappy.simulate.camera import ISOLATED_NM, neighbour_distance
    offset, noise, path = a_map(tmp_path)
    frames, truth = scmos_frames(offset, noise, seed=3, n_frames=1500, photons=1000,
                                 background=5, conversion=CONVERSION)
    isolated = neighbour_distance(truth["x_nm"], truth["y_nm"], truth["frame"]) > ISOLATED_NM
    camera = CameraMetadata(conversion=CONVERSION, offset=100.0, pixelsize_um=0.1,
                            noise_map=str(path))
    far, background = {}, {}
    for on in (False, True):
        engine = LocalizationEngine(camera, DetectionSettings().finder(0),
                                    GaussianPSF(sigma=1.3), FitSettings(per_pixel_noise=on))
        parts = [engine.push(frames[i:i + 500], i) for i in range(0, 1500, 500)]
        parts = [p for p in parts + [engine.flush()] if p is not None]
        got = {k: np.concatenate([p[k] for p in parts])
               for k in ("x_nm", "y_nm", "frame", "photons", "background")}
        errors = []
        for f in np.unique(got["frame"]):
            a, b = got["frame"] == f, truth["frame"] == f
            if not b.any():
                continue
            tx, ty = truth["x_nm"][b], truth["y_nm"][b]
            d, j = cKDTree(np.c_[tx, ty]).query(np.c_[got["x_nm"][a], got["y_nm"][a]])
            keep = (d < 200) & isolated[b][j] & (got["photons"][a] > 300)
            errors.append(np.hypot(got["x_nm"][a] - tx[j], got["y_nm"][a] - ty[j])[keep])
        errors = np.concatenate(errors)
        far[on] = np.mean(errors > 20)
        background[on] = np.median(got["background"])
        if on:
            assert engine.stats["noise_map"] == str(path)
    assert far[True] < 0.85 * far[False]
    assert abs(background[True] - 5) < abs(background[False] - 5)


# ------------------------------------------------------------ several channels
def test_each_channels_pixel_keeps_its_own_noise_through_the_cut_and_the_mirror(tmp_path):
    """A map whose every pixel is its own number, cut into a split chip's two
    channels -- the second mirrored, the camera ROI inside the map -- lands on
    the same pixels as the frames do."""
    from smappy.channels import ChannelLayout
    chip = np.arange(80 * 60, dtype=np.float32).reshape(80, 60)
    noise = scmos.NoiseMap(offset=chip, variance=chip, region=(10, 20))
    path = noise.save(tmp_path / "chip.h5")
    roi = (14, 25, 40, 50)                                   # x, y, width, height on the chip
    frame = chip[roi[1] - 20:roi[1] - 20 + 50, roi[0] - 10:roi[0] - 10 + 40]
    layout = ChannelLayout([(0, 0, 40, 25), (0, 25, 40, 22)], flips=[(False, False), (False, True)])
    camera = CameraMetadata(conversion=1.0, offset=0.0, roi=roi, noise_map=str(path))
    cut = scmos.for_layout(camera, layout, frame.shape)
    assert np.array_equal(cut.offset_adu, layout.cut(frame[None])[0])
    assert np.array_equal(cut.variance, layout.cut(frame[None])[0])
    one = scmos.channel(cut, 1, layout.sizes[1])
    assert one.variance.shape == (22, 40)
    assert np.array_equal(one.variance[0], frame[46])        # the mirror's first row is its last


@pytest.mark.parametrize("name", ["fit_gauss_global", "fit_cspline4pi_global"])
def test_a_global_fit_with_a_map_of_one_variance_is_the_chip_wide_one(name):
    """The global fits through the same switch: two Gaussians with x and y
    shared, and the 4Pi IAB model's four phase channels."""
    if name == "fit_gauss_global":
        from smappy.psf import GlobalGaussianPSF
        r1, _, _ = spots(n=1000, seed=1)
        r2, _, _ = spots(n=1000, seed=1, photons=600)
        rois = np.stack([r1, r2], 1)
        model = GlobalGaussianPSF(sigma=1.3)
        link = np.zeros((len(rois), 2, 2, 5), np.float32)
        link[:, 1] = 1
    else:
        from smappy import fourpi
        from smappy.psf import Global4PiPSF
        cal = fourpi.model_calibration()
        rois, _ = fourpi.simulate_rois(cal, 200, photons=4000, background=10, seed=1)
        model, link = Global4PiPSF(cal), None
    v = 2.0
    chip = model.fit(rois + v, link)
    pixels = model.fit(rois + v, link, variance=np.full(rois.shape, v, np.float32))
    assert np.allclose(chip.theta[:, :2], pixels.theta[:, :2], atol=2e-3)
    assert np.allclose(chip.logl, pixels.logl, rtol=1e-4)


def test_the_4pi_fit_weights_a_noisy_pixel_down():
    """A pixel of 20 e- read noise beside each emitter in every phase channel
    (a map cut through the 4Pi layout puts each channel's own there): with the
    map, x and z scatter less."""
    from smappy import fourpi
    from smappy.psf import Global4PiPSF
    cal = fourpi.model_calibration()
    rois, truth = fourpi.simulate_rois(cal, 1000, photons=2000, background=2, seed=3)
    rng = np.random.default_rng(4)
    variance = np.ones(rois.shape, np.float32)
    variance[:, :, 6, 7] = 400.0
    noisy = (rois + rng.normal(0, 1, rois.shape) * np.sqrt(variance)).astype(np.float32)
    model = Global4PiPSF(cal)
    chip = model.unpack(model.fit(noisy + 1.0))
    pixels = model.unpack(model.fit(noisy + variance, variance=variance))
    for fitted, true in (("x_roi", "x"), ("z_nm", "z_nm")):
        spread = {k: 1.4826 * np.median(np.abs(np.asarray(v[fitted]) - np.asarray(truth[true])))
                  for k, v in (("chip", chip), ("pixels", pixels))}
        assert spread["pixels"] < 0.92 * spread["chip"], fitted


def test_the_two_colour_fit_with_a_map_finds_its_molecules_closer(tmp_path):
    """Gaussian 2D 2C's engine on a split-frame movie from a camera with 3 %
    hot pixels: the map cut into both channels, the lower one through its
    registration, and the fit lands closer to the truth."""
    from scipy.spatial import cKDTree
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
    from smappy.multichannel import MultiChannelEngine
    from smappy.psf import GlobalGaussianPSF
    from test_dual_gauss_fit import SHAPE, camera, movie, transform
    rng = np.random.default_rng(4)
    t = transform()
    frames, truth = movie(300, rng, t, ratio=0.35, photons=1500., background=3., per=3)
    offset, noise = scmos_maps(SHAPE, seed=1, conversion=1.0, hot_fraction=0.03)
    path = scmos.dark_map([dark_frames(offset, noise, 1000, seed=2)]).save(tmp_path / "n.h5")
    raw = np.round(frames + offset + np.random.default_rng(9).normal(0, 1, frames.shape) * noise)
    raw = np.clip(raw, 0, 65535).astype(np.uint16)
    layout = t.layout()
    spread = {}
    for on in (False, True):
        engine = MultiChannelEngine(replace(camera(), noise_map=str(path)),
                                    PeakFinder(DoGFilter(1.2), AbsoluteCutoff(10.0)),
                                    GlobalGaussianPSF(sigma=1.2), layout,
                                    FitSettings(roisize=13, per_pixel_noise=on),
                                    detection="pairs", frame_shape=SHAPE)
        parts = [engine.push(layout.cut(raw[i:i + 50]), i) for i in range(0, len(raw), 50)]
        parts = [p for p in parts + [engine.flush()] if p is not None]
        got = {k: np.concatenate([p[k] for p in parts]) for k in ("x_nm", "y_nm", "frame")}
        errors = []
        for f in np.unique(got["frame"]):
            a = got["frame"] == f
            d, _ = cKDTree(truth[f]).query(np.c_[got["x_nm"][a], got["y_nm"][a]] / 100)
            errors.append(d[d < 2] * 100)
        spread[on] = np.median(np.concatenate(errors))
        assert ("noise_map" in engine.stats) == on
    assert spread[True] < 0.9 * spread[False]


def test_a_split_chip_fitted_channel_by_channel_reads_each_channels_part():
    """*Two channels* on a 1C fitter: each channel's engine gets its own part of
    the map, mirrored as its pixels are."""
    from smappy.multichannel import SplitEngine
    from smappy.channels import ChannelLayout
    import tempfile, os
    chip = np.arange(40 * 30, dtype=np.float32).reshape(40, 30)
    path = scmos.NoiseMap(offset=chip, variance=chip).save(
        os.path.join(tempfile.mkdtemp(), "m.h5"))
    layout = ChannelLayout([(0, 0, 30, 20), (0, 20, 30, 20)], flips=[(False, False), (False, True)])
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1, noise_map=str(path))
    finder = DetectionSettings().finder(0)
    engine = SplitEngine(camera, [finder, finder], GaussianPSF(sigma=1.3), layout,
                         FitSettings(per_pixel_noise=True))
    engine.push(np.full((1, 40, 30), 10, np.uint16), 0)
    assert np.array_equal(engine.engines[0].noise.offset_adu, chip[:20])
    assert np.array_equal(engine.engines[1].noise.offset_adu, chip[20:][::-1])
    assert engine.stats["noise_map"] == str(path)


# ------------------------------------------------------------ camera database
def a_camera(**kwargs) -> Camera:
    defaults = dict(
        name="cmos", prefix="Cam",
        identify=Parameter(tag="{prefix}-SerialNumber", read="exact", argument="7"),
        parameters={"conversion": Parameter(state=True), "offset": Parameter(state=True)},
        states=[State(match={"{prefix}-Mode": "quiet"}, values={"conversion": 0.2, "offset": 100}),
                State(match={"{prefix}-Mode": "fast"}, values={"conversion": 0.5, "offset": 100})])
    defaults.update(kwargs)
    return Camera(**defaults)


QUIET = {"Cam-SerialNumber": "7", "Cam-Mode": "quiet"}
FAST = {"Cam-SerialNumber": "7", "Cam-Mode": "fast"}


def test_a_map_belongs_to_the_readout_mode_it_was_measured_in():
    camera = camera_db.with_noise_map(a_camera(), QUIET, "/maps/quiet.h5")
    assert camera.resolve(QUIET).camera_metadata().noise_map == "/maps/quiet.h5"
    assert camera.resolve(FAST).camera_metadata().noise_map is None
    camera = camera_db.with_noise_map(camera, FAST, "/maps/fast.h5")
    assert camera.resolve(QUIET).camera_metadata().noise_map == "/maps/quiet.h5"
    assert camera.resolve(FAST).camera_metadata().noise_map == "/maps/fast.h5"
    # and survives the database's file
    again = Camera.from_dict(camera.to_dict())
    assert again.resolve(FAST).camera_metadata().noise_map == "/maps/fast.h5"


def test_a_map_of_an_unknown_readout_mode_is_not_entered():
    with pytest.raises(ValueError, match="readout mode is not in the camera database"):
        camera_db.with_noise_map(a_camera(), {"Cam-SerialNumber": "7", "Cam-Mode": "x"}, "m.h5")


def test_a_camera_without_modes_keeps_one_map():
    camera = camera_db.with_noise_map(a_camera(states=[], parameters={
        "conversion": Parameter(fixed=0.5), "offset": Parameter(fixed=100)}), QUIET, "m.h5")
    assert camera.resolve(FAST).camera_metadata().noise_map == "m.h5"


# --------------------------------------------------------------------- plugin
def test_the_calibration_saves_a_map_and_says_what_it_measured(tmp_path):
    import tifffile
    from smappy.plugins import Context
    from smappy.plugins.camera_noise import CameraNoiseCalibration, CameraNoiseSettings
    offset, noise = scmos_maps((32, 32), seed=1)
    tifffile.imwrite(tmp_path / "dark.tif", dark_frames(offset, noise, 300))
    warnings = []
    ctx = Context()
    ctx.warn = warnings.append
    result = CameraNoiseCalibration().run(ctx, CameraNoiseSettings(
        path=str(tmp_path / "dark.tif"), output=str(tmp_path / "map.h5")))
    measured = scmos.NoiseMap.load(tmp_path / "map.h5")
    assert measured.n_frames == 300 and measured.shape == (32, 32)
    assert "300 dark frames" in result.text
    # 300 frames: said to be few; no camera in the database: not entered
    assert any("1000 or more" in w for w in warnings)
    assert any("not entered" in w for w in warnings)


def test_the_spline_fit_takes_a_map_as_the_gaussian_does():
    """Spline 3D through the same switch: a map of one variance fits as the
    chip-wide variance does."""
    from smappy.calibrate.core import CalibrationSettings, build_calibration, collect_beads
    from smappy.calibrate.input import BeadStack
    from smappy.psf import SplinePSF
    from smappy.simulate import bead_stacks
    stacks, z = bead_stacks(2, seed=0)
    beads = collect_beads([BeadStack(s.astype(np.float32), z, source=f"b{i}")
                           for i, s in enumerate(stacks)], CalibrationSettings())
    psf = SplinePSF(build_calibration(beads).calibration)
    rois, _, _ = spots(n=500, photons=3000)
    v = 2.0
    chip = psf.fit(rois + v, iterations=50)
    pixels = psf.fit(rois + v, iterations=50, variance=np.full(rois.shape, v, np.float32))
    # to the fit's convergence: an extra iteration moves z by a hundredth of a step
    assert np.allclose(chip.theta[:, :2], pixels.theta[:, :2], atol=1e-3)
    assert np.allclose(chip.theta[:, 4], pixels.theta[:, 4], atol=0.05)
    assert np.allclose(chip.theta[:, 3] - v, pixels.theta[:, 3], atol=2e-3)
