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
                             psf=volume, em_mirror=False, parameters={})


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
    settings = FitSettings(roisize=13, output_unit="pixel")
    engine = DualChannelEngine(camera(), finder(), model, cal, settings)

    truth, z_index, total = (25.0, 15.0), 26.0, 12000.
    frames = np.stack([split_frame([truth], z_index, total, 0.35, 30., rng, cal)
                       for _ in range(60)])
    engine.push(frames, 0)
    locs = engine.flush()

    assert len(locs) > 40
    assert np.median(locs["x_pix"]) == pytest.approx(truth[0], abs=0.15)
    assert np.median(locs["y_pix"]) == pytest.approx(truth[1], abs=0.15)
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
                              FitSettings(roisize=13, output_unit="pixel"))

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
            cal, FitSettings(roisize=13, output_unit="pixel"))
        engine.push(frames, 0)
        locs = engine.flush()
        spread[name] = np.std(locs["z_nm"])
        if name == "all":
            assert np.all(locs["ratio"] == 0)      # nothing left to measure
    assert spread["all"] <= spread["xyz"]


def test_a_split_frame_runs_through_fit_stack_style_blocks():
    """Blocks are buffered and flushed like the single-channel engine's."""
    rng = np.random.default_rng(5)
    cal = dual_calibration()
    engine = DualChannelEngine(
        camera(), finder(), GlobalSplinePSF((cal.main, cal.secondary), LINK_XYZ),
        cal, FitSettings(roisize=13, output_unit="pixel+nm"))

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
        fit=PipelineFitSettings(roisize=13, output_unit="nm"),
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


def test_the_link_flags_reach_the_model():
    from smappy.plugins.fit import DualModelSettings
    default = DualModelSettings()
    assert default.shared() == LINK_XYZ
    everything = DualModelSettings(link_photons=True, link_background=True)
    assert everything.shared() == (True, True, True, True, True)
    unlinked = DualModelSettings(link_xy=False, link_z=False)
    assert unlinked.shared() == (False, False, False, False, False)
