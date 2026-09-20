"""The 2D two-colour workflow end to end: register, then fit.

A synthetic split-frame movie with a known transformation between the halves
and a known photon split, taken the way the plugin takes it -- no PSF model,
no transformation given, the registration measured from the movie itself.
"""
import numpy as np
import pytest
from scipy.special import erf

from smappy.calibrate.dual import ChannelTransform
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings
from smappy.plugins import Context
from smappy.psf import GlobalGaussianPSF

SPLIT = 48            # a 96 x 64 frame: upper half reference, lower secondary
SHAPE = (96, 64)
WIDTHS = (1.1, 1.4)   # the two halves see different wavelengths


def transform(dx=0.37, dy=-0.21):
    """Secondary -> reference, the direction `calibrate.dual` builds."""
    matrix = np.array([[1., 0., -dx], [0., 1., -SPLIT - dy], [0., 0., 1.]])
    geometry = {"layout": "up-down", "main_channel": "upper",
                "split_position": SPLIT, "image_shape": list(SHAPE),
                "coordinate_system": "camera-chip", "sources": [],
                "mirror_axis_xy": None}
    return ChannelTransform(matrix, geometry, {})


def draw(frame, x, y, photons, sigma):
    """One pixel-integrated Gaussian added into the frame in place."""
    ix = np.arange(SHAPE[1])
    iy = np.arange(SHAPE[0])
    norm = np.sqrt(1 / 2 / sigma ** 2)

    def axis(ii, centre):
        return .5 * (erf((ii - centre + .5) * norm) - erf((ii - centre - .5) * norm))
    frame += photons * np.outer(axis(iy, y), axis(ix, x))


def split_frame(positions, photons, ratio, background, rng, t):
    """One frame with each emitter drawn in both halves, correctly registered."""
    frame = np.full(SHAPE, float(background))
    forward = np.linalg.inv(t.transformation)
    for x, y in positions:
        partner = (forward @ np.array([x, y, 1.0]))[:2]
        draw(frame, x, y, photons * (1 - ratio), WIDTHS[0])
        draw(frame, partner[0], partner[1], photons * ratio, WIDTHS[1])
    return rng.poisson(np.clip(frame, 0, None)).astype(np.float32)


def movie(n_frames, rng, t, ratio=0.35, photons=6000., background=15., per=3):
    """A stack, and the true positions in it, one list per frame."""
    frames, truth = [], []
    for _ in range(n_frames):
        positions = np.c_[rng.uniform(6, SHAPE[1] - 6, per),
                          rng.uniform(6, SPLIT - 6, per)]
        frames.append(split_frame(positions, photons, ratio, background, rng, t))
        truth.append(positions)
    return np.stack(frames), truth


def camera():
    return CameraMetadata(conversion=1.0, offset=100.0, pixelsize_um=0.1,
                          em_on=False, roi=(0, 0, SHAPE[1], SHAPE[0]))


# ------------------------------------------------------------- the engine
def test_the_paired_gaussian_fit_finds_the_position_and_the_colour():
    """`DualChannelEngine` with a Gaussian model instead of a spline: the same
    pairing, the same link, a width per channel instead of a z."""
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
    from smappy.dualfit import DualChannelEngine

    rng = np.random.default_rng(4)
    t = transform()
    ratio = 0.35
    frames, truth = movie(40, rng, t, ratio=ratio)
    raw = (frames + 100).astype(np.uint16)

    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(25.0))
    model = GlobalGaussianPSF(sigma=1.2)
    engine = DualChannelEngine(camera(), finder, model, t,
                               FitSettings(roisize=13, output_unit="pixel"))
    engine.push(raw, first_frame=0)
    locs = engine.flush()

    assert locs is not None and len(locs) > 80
    # every fit lands on a true emitter
    want = np.vstack(truth)
    got = np.c_[locs["x_pix"], locs["y_pix"]]
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(want).query(got)
    assert np.median(distance) < 0.1

    assert abs(np.median(locs["ratio"]) - ratio) < 0.02
    # the widths were free, so each half found its own
    assert abs(np.median(locs["sigma_pix_ch0"]) - WIDTHS[0]) < 0.08
    assert abs(np.median(locs["sigma_pix_ch1"]) - WIDTHS[1]) < 0.08
    assert "z_nm" not in locs


def test_the_table_has_no_half_converted_widths():
    """`sigma_pix_ch1` has to become nm when `sigma_pix` does, or the table
    would carry two units under one name."""
    from smappy.locs import Localizations, to_nm

    table = Localizations({"x_pix": np.zeros(3), "y_pix": np.zeros(3),
                           "sigma_pix": np.ones(3), "sigma_pix_ch0": np.ones(3),
                           "sigma_pix_ch1": np.full(3, 2.0),
                           "photons_ch1": np.ones(3)}, {})
    nm = to_nm(table, 100.0)
    assert "sigma_nm_ch1" in nm and "sigma_pix_ch1" not in nm
    np.testing.assert_allclose(nm["sigma_nm_ch1"], 200.0)
    assert "photons_ch1" in nm          # not a length, left alone


# -------------------------------------------------------------- the plugin
def test_the_2c2d_plugin_calibrates_on_the_movie_and_then_fits_it(tmp_path):
    """The whole workflow as the Localize tab drives it: a split-frame TIFF and
    nothing else -- the registration is measured from the movie's own pairs."""
    import tifffile
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DetectionSettings, DualGaussianFit,
                                    DualGaussianFitSettings,
                                    DualGaussianModelSettings, OutputSettings,
                                    SourceSettings)

    rng = np.random.default_rng(7)
    t = transform(dx=0.8, dy=-0.5)
    frames, _ = movie(300, rng, t, ratio=0.35, per=4)
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))

    settings = DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=50),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        model=DualGaussianModelSettings(sigma=1.2),
        transform=ChannelTransformSettings(calibrate=True, calibrate_frames=150),
        fit=FitSettings(roisize=13, output_unit="pixel+nm"),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))

    said = []
    result = DualGaussianFit().run(Context(progress=said.append), settings)

    assert result.locs is not None and len(result.locs) > 500
    assert abs(np.median(result.locs["ratio"]) - 0.35) < 0.03
    assert "sigma_pix_ch1" in result.locs and "z_nm" not in result.locs
    assert (tmp_path / "out.hdf5").exists()
    # the measured transformation was written beside the output
    saved = tmp_path / "out_2ct.h5"
    assert saved.exists()
    assert any("registering" in line for line in said)

    # and it really is the transformation that was simulated
    from smappy.calibrate.transform import load_transform
    loaded = load_transform(saved)
    probe = np.c_[[20.0, 40.0], [60.0, 80.0]]
    np.testing.assert_allclose(loaded.transform(probe),
                               t.transform(probe), atol=0.3)


def test_a_saved_transformation_is_used_when_one_is_given(tmp_path):
    import tifffile
    from smappy.calibrate.transform import save_channel_transform
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DetectionSettings, DualGaussianFit,
                                    DualGaussianFitSettings,
                                    DualGaussianModelSettings, OutputSettings,
                                    SourceSettings)

    rng = np.random.default_rng(8)
    t = transform()
    frames, _ = movie(30, rng, t)
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    path = tmp_path / "given_2ct.h5"
    save_channel_transform(path, t)

    settings = DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=10),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        model=DualGaussianModelSettings(sigma=1.2),
        transform=ChannelTransformSettings(path=str(path)),
        fit=FitSettings(roisize=13, output_unit="pixel"),
        output=OutputSettings(save=False))

    plugin = DualGaussianFit()
    result = plugin.run(Context(), settings)
    assert result.locs is not None and len(result.locs) > 50
    assert abs(np.median(result.locs["ratio"]) - 0.35) < 0.03

    # the preview draws the secondary peaks back-projected, which is the
    # cheapest check of a transformation there is
    preview = plugin.preview(Context(), settings, frame=0)
    assert preview.plot is not None and preview.data["candidates"] > 2


def test_it_says_so_when_there_is_no_transformation_at_all(tmp_path):
    from smappy.plugins.fit import (DualGaussianFit, DualGaussianFitSettings,
                                    SourceSettings)
    settings = DualGaussianFitSettings(source=SourceSettings(path=str(tmp_path / "x.tif")))
    assert "calibrate from this movie" in DualGaussianFit().preflight(Context(), settings)
    with pytest.raises(ValueError, match="needs a transformation"):
        settings.transform.load()
