"""The 2D two-colour workflow end to end: register, then fit.

A synthetic split-frame movie with a known transformation between the halves
and a known photon split, taken the way the plugin takes it -- no PSF model,
no transformation given, the registration measured from the movie itself.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erf

from smappy.calibrate.dual import ChannelTransform
from smappy.locs import in_pixels
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings
from smappy.plugins import Context
from smappy.psf import GlobalGaussianPSF

SPLIT = 48            # a 96 x 64 frame: upper half reference, lower secondary
SHAPE = (96, 64)
WIDTHS = (1.1, 1.4)   # the two halves see different wavelengths


def chosen():
    """The split these movies have, chosen as for a microscope without an entry: their files name
    none, and the arrangement is never detected."""
    from smappy.calibrate.dual import RegisterSettings
    return RegisterSettings(layout="up-down", main_channel="upper")


def split_chosen(settings):
    """``settings`` -- a two-colour fitter's -- with the split chosen in its *channels* part."""
    settings.channels.layout, settings.channels.main_channel = "up-down", "upper"
    return settings


def transform(dx=0.37, dy=-0.21):
    """Secondary -> reference, the direction `calibrate.dual` builds."""
    matrix = np.array([[1., 0., -dx], [0., 1., -SPLIT - dy], [0., 0., 1.]])
    from smappy.calibrate.dual import split_geometry
    return ChannelTransform(matrix, split_geometry("up-down", "upper", SHAPE, split_position=SPLIT), {})


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


def paired_engine(finder, model, t, settings, photon_factors=None):
    """The fit as Gaussian 2D 2C runs it: `MultiChannelEngine` on the
    registration's two channels, the peaks of each paired, fed whole split
    frames cut into the channels first, as `ChannelSource` does."""
    from smappy.multichannel import MultiChannelEngine
    layout = t.layout()
    engine = MultiChannelEngine(camera(), finder, model, layout, settings,
                                detection="pairs", photon_factors=photon_factors)
    push = engine.push
    engine.push = lambda frames, first_frame=0: push(layout.cut(np.asarray(frames)), first_frame)
    return engine


# ------------------------------------------------------------- the engine
def test_the_paired_gaussian_fit_finds_the_position_and_the_colour():
    """The paired fit with a Gaussian model instead of a spline: the same
    pairing, the same link, a width per channel instead of a z."""
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder

    rng = np.random.default_rng(4)
    t = transform()
    ratio = 0.35
    frames, truth = movie(40, rng, t, ratio=ratio)
    raw = (frames + 100).astype(np.uint16)

    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(25.0))
    model = GlobalGaussianPSF(sigma=1.2)
    engine = paired_engine(finder, model, t, FitSettings(roisize=13))
    engine.push(raw, first_frame=0)
    locs = engine.flush()

    assert locs is not None and len(locs) > 80
    # every fit lands on a true emitter
    want = np.vstack(truth)
    got = np.c_[in_pixels(locs, "x_pix", "y_pix")]
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(want).query(got)
    assert np.median(distance) < 0.1

    assert abs(np.median(locs["ratio"]) - ratio) < 0.02
    # the widths were free, so each half found its own
    assert abs(np.median(in_pixels(locs, "sigma_pix_ch0")) - WIDTHS[0]) < 0.08
    assert abs(np.median(in_pixels(locs, "sigma_pix_ch1")) - WIDTHS[1]) < 0.08
    assert "z_nm" not in locs


def test_the_table_has_no_half_converted_widths():
    """`sigma_pix_ch1` gets its nm column when `sigma_pix` does, or the nm
    columns would be incomplete."""
    from smappy.locs import Localizations, to_nm

    table = Localizations({"x_pix": np.zeros(3), "y_pix": np.zeros(3),
                           "sigma_pix": np.ones(3), "sigma_pix_ch0": np.ones(3),
                           "sigma_pix_ch1": np.full(3, 2.0),
                           "photons_ch1": np.ones(3)}, {})
    nm = to_nm(table, 100.0)
    assert "sigma_nm_ch1" in nm and "sigma_pix_ch1" in nm
    np.testing.assert_allclose(nm["sigma_nm_ch1"], 200.0)
    assert "photons_ch1" in nm          # not a length, left alone


def test_an_unlinked_fit_keeps_how_far_the_partner_is_from_the_link():
    """Unlinking x and y is how to check the transformation, and the partner's
    position used to be dropped: the table had the reference's alone."""
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder

    rng = np.random.default_rng(6)
    frames, _ = movie(30, rng, transform())
    raw = (frames + 100).astype(np.uint16)
    finder = PeakFinder(DoGFilter(1.2), AbsoluteCutoff(25.0))

    def fit(t, shared):
        model = GlobalGaussianPSF(sigma=1.2, shared=shared)
        engine = paired_engine(finder, model, t, FitSettings(roisize=13))
        engine.push(raw, first_frame=0)
        return engine.flush()

    free = (False, False, False, False, False)
    right = fit(transform(), free)
    assert "dx_nm_ch1" in right and "dy_nm_ch1" in right
    assert abs(np.median(right["dx_nm_ch1"])) < 5          # 0.05 px
    assert abs(np.median(right["dy_nm_ch1"])) < 5
    # a transformation that puts the partner a third of a pixel too far right
    # says so: the partner is found that far left of where the link expects
    wrong = fit(transform(dx=0.37 + 0.33), free)
    assert np.median(wrong["dx_nm_ch1"]) == pytest.approx(-33.0, abs=5)
    assert abs(np.median(wrong["dy_nm_ch1"])) < 5
    # linked, there is nothing to compare
    assert "dx_nm_ch1" not in fit(transform(), (True, True, False, False, False))
    # and every width and its error is in nm, with no pixel copy beside it
    assert "sigma_nm_ch1" in right and "sigma_err_nm_ch1" in right
    assert [n for n in right.keys() if "_pix" in n] == ["peak_x_pix", "peak_y_pix"]


def test_linked_photons_are_the_emitters_total():
    """One photon number for both halves, the split typed in: the number
    reported is the total over the halves, not the main half's share."""
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder

    rng = np.random.default_rng(8)
    frames, _ = movie(40, rng, transform(), ratio=0.35, photons=6000.)
    engine = paired_engine(
        PeakFinder(DoGFilter(1.2), AbsoluteCutoff(25.0)),
        GlobalGaussianPSF(sigma=1.2, shared=(True, True, True, False, False)),
        transform(), FitSettings(roisize=13), photon_factors=(1.0, 0.35 / 0.65))
    engine.push((frames + 100).astype(np.uint16), first_frame=0)
    locs = engine.flush()
    assert np.median(locs["photons"]) == pytest.approx(6000, rel=0.05)
    assert np.all(locs["ratio"] == 0)


def test_frames_from_another_camera_roi_are_refused():
    """A registration in the bead image's own pixels cannot be placed on a
    movie of another size: the channels are not cut at all."""
    from smappy.plugins.fit import DualGaussianFit

    t = transform()
    t.geometry["coordinate_system"] = "roi-local"
    plugin = DualGaussianFit()
    plugin.registration = lambda settings: t
    with pytest.raises(ValueError, match="original bead image"):
        plugin.channel_layout(None, np.zeros((2, SHAPE[0] + 8, SHAPE[1])), camera())


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

    settings = split_chosen(DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=50),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        model=DualGaussianModelSettings(sigma=1.2),
        transform=ChannelTransformSettings(calibrate=True, calibrate_frames=150),
        fit=FitSettings(roisize=13),
        output=OutputSettings(path=str(tmp_path / "out.hdf5"))))

    said = []
    result = DualGaussianFit().run(Context(progress=said.append), settings)

    assert result.locs is not None and len(result.locs) > 500
    assert abs(np.median(result.locs["ratio"]) - 0.35) < 0.03
    assert "sigma_nm_ch1" in result.locs and "z_nm" not in result.locs
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


def test_the_preview_shows_the_registration_it_measured_and_the_run_reuses_it(tmp_path, monkeypatch):
    """*calibrate from this movie* is checked before the fit: Preview measures the registration and
    draws it as a tab of its own; a second preview and the run then take that measurement rather
    than fitting the calibration frames again -- until a setting it depends on changes."""
    import tifffile
    from smappy.plugins import fit as module
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
    settings = split_chosen(DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=50),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        model=DualGaussianModelSettings(sigma=1.2),
        transform=ChannelTransformSettings(calibrate=True, calibrate_frames=150),
        fit=FitSettings(roisize=13),
        output=OutputSettings(path=str(tmp_path / "out.hdf5"))))
    measured = []
    original = DualGaussianFit._measure_registration
    monkeypatch.setattr(DualGaussianFit, "_measure_registration",
                        lambda self, ctx, s: measured.append(1) or original(self, ctx, s))
    module._MEASURED.clear()

    preview = DualGaussianFit().preview(Context(), settings, frame=10)
    assert "channel registration" in preview.plots
    assert "measured from the movie" in preview.text and "pairs kept" in preview.text
    probe = np.c_[[20.0, 40.0], [60.0, 80.0]]
    found = preview.data["registration"].transform
    np.testing.assert_allclose(found.transform(probe), t.transform(probe), atol=0.3)
    assert preview.data["error"] == "" and len(preview.data["locs"]) > 0     # fitted on it
    assert not (tmp_path / "out_2ct.h5").exists()                            # a preview writes nothing

    DualGaussianFit().preview(Context(), settings, frame=20)                 # another frame
    result = DualGaussianFit().run(Context(), settings)
    assert measured == [1]
    assert "channel registration" in result.plots
    assert (tmp_path / "out_2ct.h5").exists()                                # the run saves it

    settings.transform.calibrate_frames = 120                                # what it depends on
    DualGaussianFit().preview(Context(), settings, frame=10)
    assert measured == [1, 1]
    settings.output.path = str(tmp_path / "elsewhere.hdf5")                  # what it does not
    DualGaussianFit().preview(Context(), settings, frame=10)
    assert measured == [1, 1]


def test_without_a_microscope_the_split_is_chosen_never_detected(tmp_path):
    """A file no microscope entry recognises: measuring the transformation from it asks for the
    layout rather than guessing it, and says so in the preview."""
    import tifffile
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DetectionSettings, DualGaussianFit,
                                    DualGaussianFitSettings, SourceSettings)
    frames, _ = movie(30, np.random.default_rng(2), transform(), per=3)
    stack = tmp_path / "unknown.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    settings = DualGaussianFitSettings(
        source=SourceSettings(path=str(stack)),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        transform=ChannelTransformSettings(calibrate=True, calibrate_skip=0))
    preview = DualGaussianFit().preview(Context(), settings, frame=0)
    assert "choose the layout" in preview.text
    with pytest.raises(ValueError, match="choose the layout"):
        DualGaussianFit().run(Context(), settings)


def test_a_failed_measurement_is_said_in_the_preview_not_raised(tmp_path):
    import tifffile
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DetectionSettings, DualGaussianFit,
                                    DualGaussianFitSettings, SourceSettings)
    frames, _ = movie(20, np.random.default_rng(1), transform(), per=1)
    stack = tmp_path / "dark.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    settings = split_chosen(DualGaussianFitSettings(
        source=SourceSettings(path=str(stack)),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=1e6),    # nothing found
        transform=ChannelTransformSettings(calibrate=True, calibrate_skip=0)))
    preview = DualGaussianFit().preview(Context(), settings, frame=0)
    assert "measuring the transformation failed" in preview.text
    assert "channel registration" not in preview.plots


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

    settings = split_chosen(DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=10),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        model=DualGaussianModelSettings(sigma=1.2),
        transform=ChannelTransformSettings(path=str(path)),
        fit=FitSettings(roisize=13),
        output=OutputSettings(save=False)))

    plugin = DualGaussianFit()
    result = plugin.run(Context(), settings)
    assert result.locs is not None and len(result.locs) > 50
    assert abs(np.median(result.locs["ratio"]) - 0.35) < 0.03

    # the preview draws the secondary peaks back-projected, which is the
    # cheapest check of a transformation there is
    preview = plugin.preview(Context(), settings, frame=0)
    assert preview.plot is not None and preview.data["candidates"] > 2


def test_a_2d_2c_run_reads_its_transformation_once_and_a_measured_one_not_at_all(
        tmp_path, monkeypatch):
    """The two-channel fitters share the reading (`_TwoChannelFit`): a file
    once per run, and none when the movie is registered on itself, even with
    a file chosen."""
    import tifffile
    from smappy.calibrate import transform as module
    from smappy.calibrate.transform import save_channel_transform
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DetectionSettings, DualGaussianFit,
                                    DualGaussianFitSettings, OutputSettings,
                                    SourceSettings)

    rng = np.random.default_rng(8)
    t = transform()
    frames, _ = movie(12, rng, t)
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    path = tmp_path / "given_2ct.h5"
    save_channel_transform(path, t)
    reads = []
    load = module.load_transform
    monkeypatch.setattr(module, "load_transform",
                        lambda p: reads.append(p) or load(p))
    settings = split_chosen(DualGaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=4),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        transform=ChannelTransformSettings(path=str(path)),
        fit=FitSettings(roisize=13), output=OutputSettings(save=False)))
    plugin = DualGaussianFit()
    assert len(plugin.run(Context(), settings).locs) > 10
    assert len(reads) == 1

    reads.clear()
    monkeypatch.setattr(DualGaussianFit, "_calibrate",
                        lambda self, ctx, s, save=True: SimpleNamespace(transform=t))
    settings.transform.calibrate = True
    assert len(plugin.run(Context(), settings).locs) > 10
    assert reads == []


def test_it_says_so_when_there_is_no_transformation_at_all(tmp_path):
    from smappy.plugins.fit import (DualGaussianFit, DualGaussianFitSettings,
                                    SourceSettings)
    settings = DualGaussianFitSettings(source=SourceSettings(path=str(tmp_path / "x.tif")))
    assert "this movie" in DualGaussianFit().preflight(Context(), settings).text
    with pytest.raises(ValueError, match="needs a transformation"):
        settings.transform.load()


def test_without_a_transformation_the_question_offers_calibrating_not_a_doomed_run():
    """"Start anyway" always failed: the channels cannot be paired without a
    transformation.  The one way on is to measure it from the movie."""
    from smappy.plugins import PreflightQuestion
    from smappy.plugins.fit import DualGaussianFit, DualGaussianFitSettings
    settings = DualGaussianFitSettings()
    question = DualGaussianFit().preflight(None, settings)
    assert isinstance(question, PreflightQuestion) and question.run_label == ""
    (choice,) = question.choices
    assert choice.settings.transform.calibrate
    assert not settings.transform.calibrate                  # the form's own untouched
    settings.transform.calibrate = True
    assert DualGaussianFit().preflight(None, settings) is None


def test_a_fit_writes_nm_and_the_peak_pixels_and_registers_from_them(tmp_path):
    """A fit's table has no pixel copies of its nm columns -- only the peak
    finder's candidates stay in chip pixels -- and the registration, which
    lives in chip pixels, recovers them from the pixel size the table
    records, along each axis: these pixels are not square.  The table is a
    two-channel fit's, which records the channels it read for the registration."""
    import tifffile
    from smappy.plugins.fit import (CameraSettings, ChannelsSettings, DetectionSettings,
                                    GaussianFit, GaussianFitSettings,
                                    OutputSettings, SourceSettings)
    from smappy.plugins.registration import (CalibrateChannelTransform,
                                             RegisterLocsSettings)

    rng = np.random.default_rng(11)
    t = transform(dx=0.8, dy=-0.5)
    frames, _ = movie(200, rng, t, ratio=0.4, per=4)
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack)),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1,
                              pixelsize_y_um=0.12),
        channels=ChannelsSettings(split="two channels", layout="up-down", main_channel="upper"),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=25.0),
        fit=FitSettings(roisize=13),
        output=OutputSettings(save=False))
    locs = GaussianFit().run(Context(), settings).locs
    assert locs.metadata["channels"]["regions"] == [[0, 0, SHAPE[1], SPLIT],
                                                    [0, SPLIT, SHAPE[1], SHAPE[0] - SPLIT]]

    assert {"x_nm", "y_nm", "xy_err_nm", "sigma_nm",
            "peak_x_pix", "peak_y_pix"} <= set(locs.keys())
    assert not {"x_pix", "y_pix", "x_err_pix", "xy_err_pix",
                "sigma_pix"} & set(locs.keys())
    assert locs.metadata["units"] == "nm"
    assert locs.metadata["pixelsize_nm_xy"] == [100.0, 120.0]
    x, y = in_pixels(locs, "x_pix", "y_pix")
    assert np.median(np.abs(x - locs["peak_x_pix"])) < 0.5
    assert np.median(np.abs(y - locs["peak_y_pix"])) < 0.5

    found = CalibrateChannelTransform().preview(
        Context(locs=locs), RegisterLocsSettings(save=False, registration=chosen())).data["transform"]
    probe = np.c_[[20.0, 40.0], [60.0, 80.0]]
    np.testing.assert_allclose(found.transform(probe), t.transform(probe), atol=0.3)


def test_a_1c_fit_reads_two_channels_on_request_each_on_its_own_and_on_the_chip(tmp_path):
    """Gaussian 2D with *channels* on *two channels*: each half detected and fitted on its own --
    the dim half too, at its own threshold -- every localization where its channel saw it on the
    chip, as a whole-chip fit puts it, and which channel in ``camera_channel``."""
    import tifffile
    from smappy.plugins.fit import (CameraSettings, DetectionSettings, GaussianFit,
                                    GaussianFitSettings, OutputSettings, SourceSettings)
    rng = np.random.default_rng(11)
    t = transform(dx=0.8, dy=-0.5)
    frames, truth = movie(40, rng, t, ratio=0.15, per=3)        # the lower half a sixth as bright
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))

    def fit(split):
        settings = GaussianFitSettings(
            source=SourceSettings(path=str(stack)),
            camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
            detection=DetectionSettings(cutoff_mode="dynamic"),
            fit=FitSettings(roisize=11), output=OutputSettings(save=False))
        settings.channels.split = split
        settings.channels.layout, settings.channels.main_channel = "up-down", "upper"
        return GaussianFit().run(Context(), settings).locs

    two, whole = fit("two channels"), fit("whole chip")
    channel = np.asarray(two["camera_channel"])
    y = np.asarray(two["y_nm"]) / 100
    assert set(channel) == {0, 1}
    assert (y[channel == 0] < SPLIT).all() and (y[channel == 1] >= SPLIT).all()
    # the main channel's molecules where the truth is, on the chip
    from scipy.spatial import cKDTree
    main = np.concatenate([np.c_[np.full(len(p), f), p] for f, p in enumerate(truth)])
    xy0 = np.c_[two["frame"], two["x_nm"] / 100, two["y_nm"] / 100][channel == 0]
    d, _ = cKDTree(main).query(xy0)
    assert np.median(d) < 0.1
    # and where a whole-chip fit of the same movie puts them
    d, _ = cKDTree(np.c_[whole["frame"], whole["x_nm"] / 100, whole["y_nm"] / 100]).query(
        np.c_[two["frame"], two["x_nm"] / 100, two["y_nm"] / 100])
    assert np.median(d) < 0.05
    # each half thresholded on its own finds the dim half's molecules
    assert (channel == 1).sum() >= (np.asarray(whole["y_nm"]) / 100 >= SPLIT).sum()
