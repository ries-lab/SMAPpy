"""Two channels on one chip, fitted as one emitter, end to end.

A synthetic split frame with a known transformation between the halves and a
known photon split: the pipeline has to find the peaks, pair them, cut both
ROIs, fit them jointly and give back the true position, z and colour.
"""
import numpy as np
import pytest

from smappy.calibrate.core import spline_coefficients
from smappy.calibrate.dual import DualColorCalibration
from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
from smappy.dualfit import (DualChannelEngine, LINK_XYZ, combine_peaks,
                            cut_paired_rois, which_channel)
from smappy.io.calibration import SplineCalibration, evaluate_spline
from smappy.locs import in_pixels
from smappy.metadata import CameraMetadata
from smappy.pipeline import FitSettings
from smappy.plugins import Context
from smappy.psf import GlobalSplinePSF

SPLIT = 32           # the frame is 64 x 64, upper half main, lower half secondary
SHAPE = (64, 64)


def psf_volume(nz=41, size=17, astigmatism=0.6):
    z, y, x = np.mgrid[:nz, :size, :size].astype(float)
    t = (z - (nz - 1) / 2) / 12
    sx = 1.2 * np.sqrt(1 + (t - astigmatism) ** 2)
    sy = 1.2 * np.sqrt(1 + (t + astigmatism) ** 2)
    return np.exp(-.5 * (((x - size // 2) / sx) ** 2 + ((y - size // 2) / sy) ** 2)) \
        / (2 * np.pi * sx * sy)


def spline(astigmatism=0.6):
    volume = psf_volume(astigmatism=astigmatism)
    return SplineCalibration(spline_coefficients(volume), 10., 20., x0=8.0,
                             psf=volume, parameters={})


def dual_calibration(dx=0.37, dy=-0.21, ratio=0.45):
    """Main in the upper half, secondary in the lower, offset by (dx, dy).

    The transformation maps secondary -> main, as `calibrate.dual` builds it.
    """
    transformation = np.array([[1., 0., -dx],
                               [0., 1., -SPLIT - dy],
                               [0., 0., 1.]])
    geometry = {"layout": "up-down", "main_channel": "upper",
                "split_position": SPLIT, "image_shape": list(SHAPE),
                "coordinate_system": "camera-chip"}
    return DualColorCalibration(spline(0.6), spline(-0.6), transformation, geometry,
                                {"secondary_main_brightness_ratio": ratio / (1 - ratio)})


def split_frame(positions, z_index, total_photons, ratio, background, rng,
                calibration, size=15):
    """One frame with each emitter drawn in both halves, correctly registered."""
    frame = np.full(SHAPE, float(background))
    forward = np.linalg.inv(calibration.transformation)
    for x, y in positions:
        secondary = (forward @ np.array([x, y, 1.0]))[:2]
        for (cx, cy), cal, photons in (
                ((x, y), calibration.main, total_photons * (1 - ratio)),
                (tuple(secondary), calibration.secondary, total_photons * ratio)):
            x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
            patch = evaluate_spline(cal, cx - x0, cy - y0, z_index, size) * photons
            frame[y0:y0 + size, x0:x0 + size] += patch
    return rng.poisson(np.clip(frame, 0, None)).astype(np.float32)


def camera():
    return CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1, em_on=False)


def finder():
    return PeakFinder(DoGFilter(1.2), AbsoluteCutoff(20.0))


# ------------------------------------------------------------------ geometry
def test_a_position_is_assigned_to_the_half_of_the_chip_it_is_on():
    geometry = dual_calibration().geometry
    x = np.array([10., 10., 50., 50.])
    y = np.array([10., 50., 10., 50.])
    np.testing.assert_array_equal(which_channel(x, y, geometry),
                                  [True, False, True, False])

    flipped = dict(geometry, main_channel="lower")
    np.testing.assert_array_equal(which_channel(x, y, flipped),
                                  [False, True, False, True])

    sideways = dict(geometry, layout="right-left", main_channel="left")
    np.testing.assert_array_equal(which_channel(x, y, sideways),
                                  [True, True, False, False])


# ----------------------------------------------------------------- combining
def test_a_molecule_both_channels_saw_becomes_one_candidate_with_two_rois():
    rng = np.random.default_rng(0)
    cal = dual_calibration()
    truth = [(20.0, 12.0), (44.0, 20.0)]
    frame = split_frame(truth, 20.0, 6000., 0.45, 30., rng, cal)

    candidates, _ = finder()(frame[None])
    reference, secondary, residual, counts = combine_peaks(candidates, cal, SHAPE, 13)

    assert len(reference) == len(truth)            # merged, not duplicated
    assert counts == {"matched": len(truth), "dropped": 0}
    order = np.argsort(reference.x)
    for got, (x, y) in zip(order, sorted(truth)):
        assert abs(reference.x[got] - x) <= 1 and abs(reference.y[got] - y) <= 1
    # every reference ROI sits in the main half and every partner in the other
    assert which_channel(reference.x, reference.y, cal.geometry).all()
    assert not which_channel(secondary.x, secondary.y, cal.geometry).any()
    # the reference channel carries no sub-pixel offset; the partner does
    assert np.allclose(residual[:, 0], 0)
    assert np.abs(residual[:, 1]).max() > 0.05


def test_a_molecule_only_one_channel_saw_is_still_fitted():
    """The union, not the intersection: SMAP keeps the lone peak and cuts its
    partner ROI where the transformation says, which is how a molecule of one
    colour gets a photon ratio at all."""
    rng = np.random.default_rng(1)
    cal = dual_calibration()
    frame = np.full(SHAPE, 30.0)
    x, y = 24.0, 14.0
    patch = evaluate_spline(cal.main, x - 17, y - 7, 20.0, 15) * 8000.
    frame[7:22, 17:32] += patch                    # main channel only
    frame = rng.poisson(frame).astype(np.float32)

    candidates, _ = finder()(frame[None])
    reference, secondary, _, counts = combine_peaks(candidates, cal, SHAPE, 13)

    assert len(reference) == 1 and counts["matched"] == 0      # nothing to pair with
    assert abs(reference.x[0] - x) <= 1 and abs(reference.y[0] - y) <= 1
    assert not which_channel(secondary.x, secondary.y, cal.geometry).any()


def test_candidates_whose_partner_would_fall_off_the_frame_are_dropped():
    cal = dual_calibration()
    from smappy.detect import Candidates
    edge = Candidates(np.zeros(3, np.int64), np.array([2, 30, 62], np.int32),
                      np.array([2, 14, 30], np.int32), np.ones(3, np.float32))
    reference, secondary, _, counts = combine_peaks(edge, cal, SHAPE, 13)
    # only the middle one leaves room for a 13-pixel ROI in both halves
    assert len(reference) == 1 and reference.x[0] == 30
    assert counts["dropped"] == 2


# ---------------------------------------------------------------------- fits
def test_the_joint_fit_recovers_position_z_and_the_photon_ratio():
    rng = np.random.default_rng(2)
    cal = dual_calibration(ratio=0.35)
    model = GlobalSplinePSF((cal.main, cal.secondary), LINK_XYZ)
    settings = FitSettings(roisize=13)
    engine = DualChannelEngine(camera(), finder(), model, cal, settings)

    truth, z_index, total = (25.0, 15.0), 26.0, 12000.
    frames = np.stack([split_frame([truth], z_index, total, 0.35, 30., rng, cal)
                       for _ in range(60)])
    engine.push(frames, 0)
    locs = engine.flush()

    assert len(locs) > 40
    x, y = in_pixels(locs, "x_pix", "y_pix")
    assert np.median(x) == pytest.approx(truth[0], abs=0.15)
    assert np.median(y) == pytest.approx(truth[1], abs=0.15)
    assert np.median(locs["z_nm"]) == pytest.approx(cal.main.z_index_to_nm(z_index),
                                                    abs=25.0)
    assert np.median(locs["photons"]) == pytest.approx(total, rel=0.12)
    assert np.median(locs["ratio"]) == pytest.approx(0.35, abs=0.05)


def test_the_two_colours_come_apart_in_the_ratio():
    """What the workflow is for: the same structure in two dyes separates on
    the photon ratio, which only a free photon parameter can measure."""
    rng = np.random.default_rng(3)
    cal = dual_calibration()
    model = GlobalSplinePSF((cal.main, cal.secondary), LINK_XYZ)
    engine = DualChannelEngine(camera(), finder(), model, cal,
                              FitSettings(roisize=13))

    ratios = []
    for fraction in (0.2, 0.8):
        frames = np.stack([split_frame([(25.0, 15.0)], 22.0, 10000., fraction,
                                       30., rng, cal) for _ in range(40)])
        engine.push(frames, 0)
        locs = engine.flush()
        ratios.append(np.median(locs["ratio"]))

    assert ratios[0] == pytest.approx(0.2, abs=0.06)
    assert ratios[1] == pytest.approx(0.8, abs=0.06)


def test_linking_the_photons_makes_the_ratio_meaningless_but_z_tighter():
    """Both of GlobLoc's options, and why the default is the one it is."""
    rng = np.random.default_rng(4)
    cal = dual_calibration()
    truth, z_index = (25.0, 15.0), 27.0
    frames = np.stack([split_frame([truth], z_index, 4000., 0.5, 25., rng, cal)
                       for _ in range(200)])

    spread = {}
    for name, shared in (("xyz", LINK_XYZ), ("all", (True,) * 5)):
        engine = DualChannelEngine(
            camera(), finder(), GlobalSplinePSF((cal.main, cal.secondary), shared),
            cal, FitSettings(roisize=13))
        engine.push(frames, 0)
        locs = engine.flush()
        spread[name] = np.std(locs["z_nm"])
        if name == "all":
            assert np.all(locs["ratio"] == 0)      # nothing left to measure
    assert spread["all"] <= spread["xyz"]


def test_an_unlinked_fit_keeps_how_far_the_partner_is_in_x_y_and_z():
    """With x, y and z free, the secondary half's values used to be dropped;
    the differences are what says whether the calibration holds."""
    rng = np.random.default_rng(5)
    cal = dual_calibration()
    frames = np.stack([split_frame([(25.0, 15.0)], 22.0, 12000., 0.5, 30., rng, cal)
                       for _ in range(60)])
    engine = DualChannelEngine(
        camera(), finder(), GlobalSplinePSF((cal.main, cal.secondary), (False,) * 5),
        cal, FitSettings(roisize=13))
    engine.push(frames, 0)
    locs = engine.flush()
    for name in ("dx_nm_ch1", "dy_nm_ch1", "dz_nm_ch1"):
        assert name in locs
    assert abs(np.median(in_pixels(locs, "dx_pix_ch1"))) < 0.05
    assert abs(np.median(in_pixels(locs, "dy_pix_ch1"))) < 0.05
    assert abs(np.median(locs["dz_nm_ch1"])) < 15.0
    assert np.std(locs["dz_nm_ch1"]) > 0            # two fits, not one copied


def test_a_dual_calibration_keeps_the_em_gain_and_a_mismatch_is_reported(tmp_path):
    """Dual calibrations never recorded it, so the warning could not fire."""
    from smappy.calibrate.dual import load_dual_color_calibration
    from smappy.plugins.fit import (DualModelSettings, DualSplineFit,
                                    DualSplineFitSettings)
    cal = dual_calibration()
    cal.main.em_on = cal.secondary.em_on = True
    path = _dual_calibration_file(tmp_path / "em_3dcal.h5", cal)
    loaded = load_dual_color_calibration(path)
    assert loaded.main.em_on is True and loaded.secondary.em_on is True

    told = []
    settings = DualSplineFitSettings(model=DualModelSettings(calibration=str(path)))
    with pytest.warns(UserWarning, match="EM gain mismatch"):
        DualSplineFit()._model_telling(Context(progress=told.append), settings,
                                       CameraMetadata(em_on=False))
    assert any("EM gain mismatch" in line for line in told)


def test_a_split_frame_runs_through_fit_stack_style_blocks():
    """Blocks are buffered and flushed like the single-channel engine's."""
    rng = np.random.default_rng(5)
    cal = dual_calibration()
    engine = DualChannelEngine(
        camera(), finder(), GlobalSplinePSF((cal.main, cal.secondary), LINK_XYZ),
        cal, FitSettings(roisize=13))

    total = 0
    for block in range(3):
        frames = np.stack([split_frame([(25.0, 15.0), (42.0, 22.0)], 20.0, 8000.,
                                       0.4, 30., rng, cal) for _ in range(10)])
        out = engine.push(frames, block * 10)
        total += 0 if out is None else len(out)
    rest = engine.flush()
    total += 0 if rest is None else len(rest)

    assert total > 40
    assert engine.stats["frames"] == 30
    # two emitters per frame, each seen in both halves and merged into one pair
    assert engine.stats["matched"] == 60 and engine.stats["dropped_at_border"] == 0
    assert "x_nm" in (rest.keys() if rest is not None else [])


# ------------------------------------------------------------ the 3D-2C plugin
def _dual_calibration_file(path, cal):
    from smappy.calibrate.dual import save_dual_color_calibration
    save_dual_color_calibration(path, cal)
    return path


def test_the_3d_2c_workflow_runs_from_a_tiff_and_a_saved_calibration(tmp_path):
    """The whole plugin, as the Localize tab drives it: a split-frame TIFF, a
    dual-colour calibration off disk, localizations with x, y, z and a ratio."""
    import tifffile
    from smappy.plugins.fit import (CameraSettings, DetectionSettings,
                                    DualModelSettings, DualSplineFit,
                                    DualSplineFitSettings, OutputSettings,
                                    SourceSettings)
    from smappy.pipeline import FitSettings as PipelineFitSettings

    rng = np.random.default_rng(6)
    cal = dual_calibration(ratio=0.3)
    frames = np.stack([split_frame([(25.0, 15.0), (42.0, 21.0)], 24.0, 9000., 0.3,
                                   30., rng, cal) for _ in range(12)])
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    calibration_file = _dual_calibration_file(tmp_path / "dual_3dcal.h5", cal)

    settings = DualSplineFitSettings(
        source=SourceSettings(path=str(stack), chunk=6),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=20.0),
        model=DualModelSettings(calibration=str(calibration_file)),
        fit=PipelineFitSettings(roisize=13),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))

    result = DualSplineFit().run(Context(), settings)
    locs = result.locs

    assert len(locs) > 15
    assert {"x_nm", "y_nm", "z_nm", "ratio", "photons"} <= set(locs.keys())
    assert np.median(locs["ratio"]) == pytest.approx(0.3, abs=0.06)
    assert (tmp_path / "out.hdf5").exists()
    # both emitters found, at 100 nm pixels
    xs = np.sort(np.unique(np.rint(locs["x_nm"] / 100.0)))
    assert set(xs.astype(int)) <= {25, 42}


def test_a_3d_2c_run_and_a_preview_each_read_the_calibration_once(
        tmp_path, monkeypatch):
    """The finder, the model, the engine and the back-projection all need it;
    each used to read the file, three times a run and four a preview."""
    import tifffile
    from smappy.calibrate import dual
    from smappy.plugins.fit import (CameraSettings, DetectionSettings,
                                    DualModelSettings, DualSplineFit,
                                    DualSplineFitSettings, OutputSettings,
                                    SourceSettings)

    rng = np.random.default_rng(9)
    cal = dual_calibration()
    frames = np.stack([split_frame([(25.0, 15.0)], 20.0, 9000., 0.3, 30., rng,
                                   cal) for _ in range(6)])
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    settings = DualSplineFitSettings(
        source=SourceSettings(path=str(stack), chunk=2),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=20.0),
        model=DualModelSettings(calibration=str(
            _dual_calibration_file(tmp_path / "dual_3dcal.h5", cal))),
        fit=FitSettings(roisize=13), output=OutputSettings(save=False))

    reads = []
    load = dual.load_dual_color_calibration
    monkeypatch.setattr(dual, "load_dual_color_calibration",
                        lambda path: reads.append(path) or load(path))
    plugin = DualSplineFit()
    assert len(plugin.run(Context(), settings).locs) >= 3
    assert len(reads) == 1
    reads.clear()
    assert plugin.preview(Context(), settings, frame=1).data["locs"] is not None
    assert len(reads) == 1


def test_the_link_flags_reach_the_model():
    from smappy.plugins.fit import DualModelSettings
    default = DualModelSettings()
    assert default.shared() == LINK_XYZ
    everything = DualModelSettings(link_photons=True, link_background=True)
    assert everything.shared() == (True, True, True, True, True)
    unlinked = DualModelSettings(link_xy=False, link_z=False)
    assert unlinked.shared() == (False, False, False, False, False)


@pytest.mark.parametrize("pixelsize", [0.1, None])
def test_a_2c_preview_shows_each_channels_peaks_and_the_fits_where_the_channels_have_them(
        tmp_path, pixelsize):
    """Peaks are found in each channel; the second channel's are put on the first by the
    registration, which is the cheapest check of it there is (an x beside each o); the fits are
    drawn in both.  The movie's camera ROI starts 20 px down the chip, and the positions are counted
    from the main half's corner on the chip whatever ROI the movie had -- the preview once drew fits
    one ROI offset away whenever it fell back to pixel units."""
    import tifffile
    from dataclasses import replace
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from smappy.plugins.fit import DualSplineFit

    oy = 20                                  # a chip ROI that starts 20 px down
    rng = np.random.default_rng(6)
    cal = dual_calibration(ratio=0.3)
    frame = split_frame([(25.0, 15.0), (42.0, 21.0)], 24.0, 9000., 0.3, 30., rng, cal)
    tifffile.imwrite(tmp_path / "split.tif", (frame[None] + 100).astype(np.uint16))
    # chip coordinates: the beads covered the chip from its corner to the movie's far edge
    cal.geometry = dict(cal.geometry, split_position=SPLIT + oy, image_shape=[SHAPE[0] + oy, SHAPE[1]])
    calibration_file = _dual_calibration_file(tmp_path / "dual.h5", cal)

    plugin = DualSplineFit()
    settings = plugin.Settings()
    settings.source.path = str(tmp_path / "split.tif")
    settings.camera.conversion, settings.camera.offset = 1.0, 100.0
    settings.camera.pixelsize_um = pixelsize
    settings.detection.cutoff_mode, settings.detection.cutoff = "absolute", 20.0
    settings.model.calibration = str(calibration_file)
    settings.fit.roisize = 13
    read = plugin._camera
    plugin._camera = lambda s: replace(read(s), roi=(0, oy, 64, 64))

    result = plugin.preview(Context(), settings, frame=0)
    assert result.data["error"] == "" and len(result.data["locs"]) == 2
    px = (pixelsize or 0.1) * 1000
    np.testing.assert_allclose(np.sort(np.asarray(result.data["locs"]["y_nm"]) / px), [15 + oy, 21 + oy],
                               atol=0.2)

    figure = plt.figure()
    result.plot.draw_into(figure)
    panels = {a.get_title(): a for a in figure.axes if a.get_title()}
    first, second = panels["channel 1"], panels["channel 2"]

    def marked(panel, marker):
        line = next(l for l in panel.lines if l.get_marker() == marker)
        return np.c_[line.get_xdata(), line.get_ydata()]

    # in each channel's own pixels: the movie's part of the main half starts at its row 20
    found = marked(first, "o")
    np.testing.assert_allclose(np.sort(found[:, 1]), [15, 21], atol=1)
    projected = marked(first, "x")
    nearest = np.min(np.linalg.norm(projected[:, None] - found[None], axis=2), axis=1)
    assert len(projected) == 2 and np.all(nearest < 1.0)
    np.testing.assert_allclose(np.sort(marked(first, "+")[:, 1]), [15, 21], atol=0.5)
    np.testing.assert_allclose(np.sort(marked(second, "+")[:, 1]), np.sort(marked(second, "o")[:, 1]),
                               atol=1)
    plt.close(figure)


def _mirrored_calibration(dx=0.37, dy=-0.21, ratio=0.3):
    """Main upper, secondary lower *and flipped in y* -- an image splitter
    with a mirror in one path.  The secondary spline is stored as the camera
    sees it, flipped too, the way `calibrate.dual` saves it."""
    transformation = np.array([[1., 0., -dx],
                               [0., -1., 2 * SPLIT - 1 + dy],
                               [0., 0., 1.]])
    volume = np.flip(psf_volume(astigmatism=-0.6), axis=1).copy()
    secondary = SplineCalibration(spline_coefficients(volume), 10., 20., x0=8.0,
                                  psf=volume, parameters={})
    geometry = {"layout": "up-down mirrored", "main_channel": "upper",
                "split_position": SPLIT, "image_shape": list(SHAPE),
                "coordinate_system": "camera-chip", "mirror_axis_xy": 1}
    return DualColorCalibration(spline(0.6), secondary, transformation, geometry,
                                {"secondary_main_brightness_ratio": ratio / (1 - ratio)})


@pytest.mark.parametrize("mirrored", [False, True])
def test_a_mirrored_splitter_keeps_its_colour(mirrored):
    """The photon split survives a mirror in the secondary path.

    The link scales the partner channel's coordinates by the transformation's
    local factor, -1 along a mirrored axis.  Anchored at the ROI corner instead
    of its centre, that put the partner channel's emitter outside its ROI: x, y
    and z still came out right, from the main channel alone, while the
    secondary photons fitted to nothing and every ratio read zero.
    """
    cal = _mirrored_calibration() if mirrored else dual_calibration(ratio=0.3)
    rng = np.random.default_rng(1)
    truth = [(25.0, 12.0), (42.0, 20.0)]
    frames = np.stack([split_frame(truth, 24.0, 9000., 0.3, 30., rng, cal)
                       for _ in range(30)])
    model = GlobalSplinePSF((cal.main, cal.secondary), LINK_XYZ)
    engine = DualChannelEngine(camera(), finder(), model, cal,
                               FitSettings(roisize=13))
    engine.push(frames)
    locs = engine.flush()

    assert engine.stats["matched"] == 60                 # every emitter paired
    assert len(locs) == 60
    assert np.median(locs["ratio"]) == pytest.approx(0.3, abs=0.03)
    assert np.median(locs["photons_ch1"]) > 1000
    # the per-channel errors travel with the counts, for a colour assignment
    # to weigh the split by how well each half was measured
    assert np.all(locs["photons_err_ch0"] > 0)
    assert np.median(locs["photons_err_ch1"]) < np.median(locs["photons_ch1"])
    assert np.median(locs["z_nm"]) == pytest.approx(-40, abs=5)
    x, y = in_pixels(locs, "x_pix", "y_pix")
    for x0, y0 in truth:
        mine = np.hypot(x - x0, y - y0) < 3
        assert mine.sum() == 30
        assert np.median(np.abs(x[mine] - x0)) < 0.1
        assert np.median(np.abs(y[mine] - y0)) < 0.1


def test_a_split_preview_reports_each_channels_own_cutoff(tmp_path):
    """Each channel is searched alone, and the preview says both cutoffs: the dim channel's is its
    own, far below the bright one's."""
    import tifffile
    from smappy.camera import to_photons
    from smappy.detect import DynamicCutoff, local_maxima
    from smappy.plugins.fit import DualSplineFit

    rng = np.random.default_rng(6)
    cal = dual_calibration(ratio=0.1)
    frame = split_frame([(25.0, 15.0), (42.0, 21.0)], 24.0, 9000., 0.1, 30., rng, cal)
    frame[SPLIT:] *= 0.2                          # a much dimmer lower half
    tifffile.imwrite(tmp_path / "split.tif", (frame[None] + 100).astype(np.uint16))
    plugin = DualSplineFit()
    settings = plugin.Settings()
    settings.source.path = str(tmp_path / "split.tif")
    settings.camera.conversion, settings.camera.offset = 1.0, 100.0
    settings.camera.pixelsize_um = 0.1
    settings.model.calibration = str(_dual_calibration_file(tmp_path / "dual.h5", cal))
    settings.fit.roisize = 13

    result = plugin.preview(Context(), settings, frame=0)
    first, second = result.data["cutoff"]

    # computed here, from the maxima of each channel
    camera = plugin._camera(settings)
    finder = plugin.finder(settings, camera)
    channels = plugin._reading_once(settings, lambda: plugin.open_source(settings)).frame(0)
    photons = to_photons(channels, camera)
    cutoff = DynamicCutoff(settings.detection.cutoff)
    for c, expected in ((0, first), (1, second)):
        filtered = finder.filter(np.ascontiguousarray(photons[c][None], np.float32))
        _, _, _, values = local_maxima(filtered, use_extension=False)
        assert expected == pytest.approx(cutoff(values), rel=1e-6)
    assert second < 0.5 * first
    assert f"(cutoff {first:.3g})" in result.text and f"(cutoff {second:.3g})" in result.text


def test_a_fit_region_drawn_on_one_half_applies_to_the_other_too():
    from smappy.dualfit import region_counterparts
    from smappy.plugins.fit import in_region
    cal = dual_calibration()
    box = {"kind": "exclude", "points": [[10, 10], [20, 10], [20, 20], [10, 20]]}
    mirrored = region_counterparts([box], cal)
    # the main half's box, where the secondary half sees the same sample
    np.testing.assert_allclose(mirrored[0]["points"][0], [10 + 0.37, 10 + SPLIT - 0.21])
    assert mirrored[0]["kind"] == "exclude"
    # and one drawn on the secondary half comes back onto the main one
    back = region_counterparts(mirrored, cal)
    np.testing.assert_allclose(back[0]["points"], box["points"], atol=1e-9)
    # a bead excluded on the main half is excluded in both: neither peak is fitted
    x, y = np.array([15.0, 15.37, 40.0]), np.array([15.0, 15.0 + SPLIT - 0.21, 40.0])
    assert in_region(x, y, [box]).tolist() == [False, True, True]
    assert in_region(x, y, [box] + mirrored).tolist() == [False, False, True]
    # with an offset camera ROI, shapes in the frame's pixels still land right
    shifted = region_counterparts([box], cal, origin=(4.0, 6.0))
    np.testing.assert_allclose(shifted[0]["points"], mirrored[0]["points"])


def test_spline_2c_keeps_the_calibrations_psfs_with_a_registration_measured_from_the_movie(tmp_path, monkeypatch):
    """The splitter moves between bead calibrations, the PSFs hardly: *measure from this movie* measures
    with the calibration's layout and main channel, so each PSF stays on its half, and fits with both."""
    import json

    import h5py
    import tifffile
    from smappy.calibrate.dual import ChannelTransform, with_registration
    from smappy.plugins.fit import (CameraSettings, DetectionSettings, DualModelSettings, DualSplineFit,
                                    DualSplineFitSettings, OutputSettings, SourceSettings)
    from smappy.pipeline import FitSettings as PipelineFitSettings

    rng = np.random.default_rng(7)
    moved = dual_calibration(dx=1.4, dy=0.8, ratio=0.3)           # the splitter, since the beads
    frames = np.stack([split_frame([(25.0, 15.0), (42.0, 21.0)], 24.0, 9000., 0.3, 30., rng, moved)
                       for _ in range(12)])
    stack = tmp_path / "split.tif"
    tifffile.imwrite(stack, (frames + 100).astype(np.uint16))
    beads = dual_calibration(ratio=0.3)                            # the beads' registration, now off
    calibration_file = _dual_calibration_file(tmp_path / "dual_2Ccal.h5", beads)
    asked = []

    def measured(self, ctx, settings):
        asked.append(settings.transform.registration)
        return ChannelTransform(moved.transformation, dict(moved.geometry), {"n_inliers": 100})
    monkeypatch.setattr(DualSplineFit, "_calibrate", measured)
    settings = DualSplineFitSettings(
        source=SourceSettings(path=str(stack), chunk=6),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=20.0),
        model=DualModelSettings(calibration=str(calibration_file)),
        fit=PipelineFitSettings(roisize=13),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))
    settings.transform.calibrate = True
    locs = DualSplineFit().run(Context(), settings).locs
    assert (asked[0].layout, asked[0].main_channel) == ("up-down", "upper")
    assert len(locs) > 15 and np.median(locs["ratio"]) == pytest.approx(0.3, abs=0.06)
    with h5py.File(tmp_path / "out.hdf5") as f:                     # the file says what it was fitted with
        assert json.loads(f.attrs["metadata"])["calibration"]["file"] == str(calibration_file)
    # a registration with the other half as the main one would put each PSF on the wrong light
    flipped = ChannelTransform(moved.transformation, dict(moved.geometry, main_channel="lower"), {})
    with pytest.raises(ValueError, match="wrong halves"):
        with_registration(beads, flipped)


# ------------------------------------------------------------- the channel layer
def _plugin_on(tmp_path, frames, cal, roi, name):
    """Spline 3D 2C on ``frames`` written as a TIFF, its camera ROI ``roi`` (x, y, w, h) on the chip."""
    import tifffile
    from dataclasses import replace
    from smappy.plugins.fit import DualSplineFit
    tifffile.imwrite(tmp_path / f"{name}.tif", (frames + 100).astype(np.uint16))
    plugin = DualSplineFit()
    settings = plugin.Settings()
    settings.source.path = str(tmp_path / f"{name}.tif")
    settings.camera.conversion, settings.camera.offset = 1.0, 100.0
    settings.camera.pixelsize_um = 0.1
    settings.detection.cutoff_mode, settings.detection.cutoff = "absolute", 20.0
    settings.model.calibration = str(_dual_calibration_file(tmp_path / f"{name}_dual.h5", cal))
    settings.fit.roisize = 13
    settings.output.save = False
    read = plugin._camera
    plugin._camera = lambda s: replace(read(s), roi=roi)
    return plugin, settings


def test_a_movie_from_a_smaller_camera_roi_is_fitted_where_the_calibrations_halves_have_it(tmp_path):
    """The calibration's halves are on the chip; a movie whose camera ROI cuts into both is cut where
    its pixels lie in them, and its molecules come out where the whole frame's do."""
    rng = np.random.default_rng(3)
    cal = dual_calibration(ratio=0.3)
    truth = [(22.0, 13.0), (40.0, 18.5)]
    frames = np.stack([split_frame(truth, 24.0, 9000., 0.3, 30., rng, cal) for _ in range(6)])
    results = {}
    for name, (x0, y0, x1, y1) in (("whole", (0, 0, 64, 64)), ("cut", (6, 4, 58, 60))):
        plugin, settings = _plugin_on(tmp_path, frames[:, y0:y1, x0:x1], cal, (x0, y0, x1 - x0, y1 - y0), name)
        results[name] = plugin.run(Context(), settings).locs
    whole, cut = results["whole"], results["cut"]
    assert len(whole) == len(cut) == 12
    for key in ("x_nm", "y_nm", "z_nm", "ratio"):
        np.testing.assert_allclose(np.sort(cut[key]), np.sort(whole[key]), rtol=1e-5, atol=1e-3)


def test_a_fit_region_drawn_on_channel_1_picks_among_the_pairs():
    """The region is drawn where positions are reported, on channel 1, and acts on the pairs: a
    molecule excluded there is fitted in neither channel."""
    from smappy.multichannel import MultiChannelEngine
    from smappy.plugins.fit import in_region
    rng = np.random.default_rng(4)
    cal = dual_calibration(ratio=0.3)
    frames = np.stack([split_frame([(20.0, 14.0), (44.0, 18.0)], 24.0, 9000., 0.3, 30., rng, cal)
                       for _ in range(4)])
    box = {"kind": "exclude", "points": [[14, 8], [26, 8], [26, 20], [14, 20]]}
    lay = cal.layout()
    engine = MultiChannelEngine(camera(), finder(), GlobalSplinePSF(cal.layer_splines(), LINK_XYZ), lay,
                                FitSettings(roisize=13), detection="pairs", photon_factors=(1.0, 1.0),
                                select=lambda x, y: in_region(x, y, [box]))
    engine.push(lay.cut(frames))
    locs = engine.flush()
    assert engine.stats["matched"] == 8 and len(locs) == 4
    np.testing.assert_allclose(np.asarray(locs["x_nm"]) / 100, 44.0, atol=0.2)
