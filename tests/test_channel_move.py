"""The channel move (`smappy.channel_move`, gui-plan §4b): how far a split camera's second channel lies
from where the beads' registration puts it, measured from the movie with the model -- the vote of the
peaks for the start, the fit with x and y free per channel for the answer -- and applied by the
two-channel fitters through the channel layer.

The movies are `test_dual_fit`'s split frames, the secondary drawn where the registration puts it plus a
known move: what is measured must be that move, and the fit with it must pair what the beads' map alone
cannot.
"""
import numpy as np
import pytest
import tifffile
from test_dual_fit import (
    _dual_calibration_file,
    _mirrored_calibration,
    dual_calibration,
    split_frame,
)

from smappy import channel_move as cm
from smappy.detect import Candidates
from smappy.plugins import Context


def _moved(cal, columns, rows):
    """``cal`` with the secondary's image moved by (columns, rows) on the chip: what the movie shows."""
    from smappy.netfit.fit_two import moved_calibration
    return moved_calibration(cal, (columns, rows))


def _movie(tmp_path, cal, truth, n=30, per_frame=4, seed=1):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        xy = np.c_[rng.uniform(10, 46, per_frame), rng.uniform(11, 20, per_frame)]
        frames.append(split_frame(list(map(tuple, xy)), float(rng.uniform(14, 30)), 9000.0, 0.4, 20.0, rng, truth))
    path = tmp_path / "moved.tif"
    tifffile.imwrite(path, (np.stack(frames) + 100).astype(np.uint16))
    return path, _dual_calibration_file(tmp_path / "beads_2ct.h5", cal)


def _settings(movie, calibration, **move):
    from smappy.pipeline import FitSettings
    from smappy.plugins.fit import (
        CameraSettings,
        ChannelMoveSettings,
        DetectionSettings,
        DualModelSettings,
        DualSplineFitSettings,
        OutputSettings,
        SourceSettings,
    )
    return DualSplineFitSettings(
        source=SourceSettings(path=str(movie), chunk=10),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=20.0),
        model=DualModelSettings(calibration=str(calibration), link_photons=False,
                                move=ChannelMoveSettings(**{"blocks": 3, "block_frames": 10, **move})),
        fit=FitSettings(roisize=13), output=OutputSettings(path=""))


def test_the_vote_finds_a_move_of_pixels_from_the_peaks_alone_mirrored_or_not():
    for cal in (dual_calibration(), _mirrored_calibration()):
        layout = cal.layout()
        rng = np.random.default_rng(0)
        ref = np.c_[rng.uniform(5, 59, 400), rng.uniform(5, 27, 400)]
        frame = np.repeat(np.arange(40), 10)
        true = np.array([3.4, -2.6])                                 # columns, rows on the chip
        layer = cm.to_chip(true, layout)                             # its own inverse: a mirror flips back
        ch1 = layout.to_channel(1, ref) + layer + rng.normal(0, 0.3, ref.shape)
        noise = np.c_[rng.uniform(0, 64, 400), rng.uniform(0, 32, 400)]        # peaks with no partner
        a = Candidates(frame, np.rint(ref[:, 0]).astype(np.int32), np.rint(ref[:, 1]).astype(np.int32),
                       np.ones(400, np.float32))
        b = Candidates(np.r_[frame, frame], np.rint(np.r_[ch1[:, 0], noise[:, 0]]).astype(np.int32),
                       np.rint(np.r_[ch1[:, 1], noise[:, 1]]).astype(np.int32), np.ones(800, np.float32))
        found, strength = cm.vote(a, b, layout)
        assert strength > cm.VOTE_MIN_STRENGTH
        np.testing.assert_allclose(found, true, atol=0.6)


def test_the_robust_mean_and_the_curve_recover_the_shift_with_a_tenth_of_the_pairs_wrong():
    rng = np.random.default_rng(2)
    frame = rng.uniform(0, 1000, 4000)
    truth = np.c_[0.3 + 0.4 * frame / 1000, -1.2 + 0 * frame]
    shift = truth + rng.normal(0, 0.05, truth.shape)
    shift[:400] += rng.uniform(-3, 3, (400, 2))                       # wrong pairs
    err = np.full(len(frame), 0.05)
    mean, se, kept = cm.constant(shift, err)
    np.testing.assert_allclose(mean, truth.mean(0), atol=0.01)
    assert kept > 3400 and np.all(se < 0.005)
    grid, curve, _ = cm.over_time(frame, shift, err, 0.0, 1000.0)
    np.testing.assert_allclose(curve, np.c_[0.3 + 0.4 * grid / 1000, np.full(len(grid), -1.2)], atol=0.02)
    move = cm.ChannelMove(curve, grid)
    assert move.over_time and move.at([500.0])[0] == pytest.approx([0.5, -1.2], abs=0.02)


@pytest.mark.parametrize("mirrored", [False, True])
def test_spline_2c_measures_a_move_beyond_the_pairing_and_fits_with_it(tmp_path, mirrored):
    """Moved 3 columns and -2.5 rows since the beads: measured to a tenth of a pixel, and the fit with it
    finds the photon split the beads' map alone cannot pair."""
    from smappy.plugins.fit import DualSplineFit
    cal = _mirrored_calibration() if mirrored else dual_calibration()
    truth = np.array([3.0, -2.5])
    movie, calibration = _movie(tmp_path, cal, _moved(cal, *truth))
    plugin = DualSplineFit()
    move, data = plugin.measure_move(_settings(movie, calibration, move="measured"))
    np.testing.assert_allclose(move.mean, truth, atol=0.1)
    assert move.record["pairs"] >= cm.MIN_PAIRS and len(data["frame"]) == len(data["shift"])
    measured = plugin.run(Context(), _settings(movie, calibration, move="measured")).locs
    assert np.median(measured["ratio"]) == pytest.approx(0.4, abs=0.05)
    # kept beside the movie: the next run reads it, measuring nothing
    assert cm.cache_path(movie).exists()
    DualSplineFit._moves.clear()
    again, _ = DualSplineFit().channel_move(_settings(movie, calibration, move="measured"))
    np.testing.assert_allclose(again.mean, move.mean)
    # typed, the same; none, the pairs fall apart
    typed = plugin.run(Context(), _settings(movie, calibration, move="typed", columns=3.0, rows=-2.5)).locs
    assert np.median(typed["ratio"]) == pytest.approx(0.4, abs=0.05)
    none = plugin.run(Context(), _settings(movie, calibration, move="none")).locs
    assert abs(np.median(none["ratio"]) - 0.4) > 0.1 or len(none) < 0.7 * len(measured)


def test_a_preview_measures_the_move_and_draws_it(tmp_path):
    from smappy.plugins.fit import DualSplineFit
    cal = dual_calibration()
    movie, calibration = _movie(tmp_path, cal, _moved(cal, 2.0, 1.5), n=20)
    DualSplineFit._moves.clear()
    result = DualSplineFit().preview(Context(), _settings(movie, calibration, move="measured"), 3)
    assert "channel move" in result.plots and "channel 2 moved" in result.text
    np.testing.assert_allclose(result.data["channel_move"].mean, [2.0, 1.5], atol=0.15)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    result.plots["channel move"].draw(Figure())


def test_rotation_and_scale_are_learnt_from_the_pairs_beside_the_shift(tmp_path):
    """The splitter turned 10 mrad and scaled by half a percent about the second channel's centre, and
    moved: *rotation and scale* recovers both, and the shift is still the shift."""
    from smappy.plugins.fit import DualSplineFit
    cal = dual_calibration()
    rot, scale = 0.01, 0.005
    L = (1 + scale) * np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) - np.eye(2)
    turned = cm.ChannelMove(np.zeros((1, 2)), None, {}, L, np.array([32.0, 48.0]))
    truth = _moved(cm.with_linear(cal, turned), 1.5, -1.0)
    movie, calibration = _movie(tmp_path, cal, truth, n=40, per_frame=6)
    move, _ = DualSplineFit().measure_move(_settings(movie, calibration, move="measured", affine=True,
                                                     blocks=4))
    got_rot, got_scale = cm.rotation_scale(move.linear)
    assert got_rot == pytest.approx(rot, abs=0.003) and got_scale == pytest.approx(scale, abs=0.003)
    # the shift at the centre the linear part turns about
    shift_at = move.mean + move.linear @ (np.array([32.0, 48.0]) - move.centre)
    np.testing.assert_allclose(shift_at, [1.5, -1.0], atol=0.15)
    assert "turned" in move.line()


def test_a_move_that_cannot_be_measured_leaves_the_fit_to_go_on_and_says_why(tmp_path):
    from smappy.plugins.fit import DualSplineFit
    cal = dual_calibration()
    movie, calibration = _movie(tmp_path, cal, cal, n=3, per_frame=1)
    said = []
    DualSplineFit._moves.clear()
    move, data = DualSplineFit().channel_move(_settings(movie, calibration, move="measured", blocks=2,
                                                        block_frames=2), said.append)
    assert move is None and "failed" in data
    assert any("not measured" in s for s in said)
    assert not cm.cache_path(movie).exists()


def test_a_constant_move_that_changes_between_blocks_says_so():
    move = cm.ChannelMove(np.array([[0.2, 1.0]]), None, {"how": "measured", "drift_px": 0.4})
    assert "changes by 0.40 px between the blocks: tick over time" in move.line()
