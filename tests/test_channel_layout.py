"""The channel layer for channels of different sizes and camera ROIs that cut into a channel.

A dual-colour splitter's seam is where it was measured, not always in the middle, and a movie may be
taken with a smaller camera ROI than the calibration: the layer cuts what there is and keeps where it lies
in the whole rectangle, so the maps -- between whole rectangles -- keep their meaning.
"""
import numpy as np
import pytest

from smappy.channels import ChannelLayout


def _chip(shape=(100, 120)):
    return np.arange(shape[0] * shape[1], dtype=float).reshape(shape)


def _map(dx=3.3, dy=-1.7, scale=1.002, rot=0.004):
    c, s = scale * np.cos(rot), scale * np.sin(rot)
    return np.array([[c, -s, dx], [s, c, dy], [0, 0, 1.0]])


def test_channels_of_different_sizes_are_cut_each_to_its_own_and_padded_beyond():
    layout = ChannelLayout([(0, 0, 120, 46), (0, 46, 120, 54)])
    assert layout.sizes == [(120, 46), (120, 54)] and layout.size == (120, 54)
    assert not layout.same_size
    chip = _chip()
    block = layout.cut(chip[None])
    assert block.shape == (1, 2, 54, 120)
    np.testing.assert_array_equal(block[0, 0, :46], chip[:46])
    assert not block[0, 0, 46:].any()                       # the smaller channel's padding
    np.testing.assert_array_equal(block[0, 1], chip[46:])


@pytest.mark.parametrize("flip", [(False, False), (False, True), (True, False)])
def test_a_camera_roi_cutting_into_the_channels_keeps_their_pixels_where_the_whole_rectangle_has_them(flip):
    """What a ROI leaves of each channel is the same pixels as the whole channel's, at the channel's
    ``origins``; the map read through them gives the same positions in both."""
    whole = ChannelLayout([(0, 0, 60, 100), (60, 0, 60, 100)], [(False, False), flip],
                          [np.eye(3), _map()])
    chip = _chip()
    roi = (9, 13)                                           # corner on the chip
    frame = chip[13:13 + 70, 9:9 + 95]                      # the far side cuts channel 1 too
    cut = whole.on_camera(roi, shape=frame.shape, cut_into=True)
    full, part = whole.cut(chip[None])[0], cut.cut(frame[None])[0]
    for c in range(2):
        (ox, oy), (w, h) = cut.origins[c], cut.sizes[c]
        np.testing.assert_array_equal(part[c, :h, :w], full[c, oy:oy + h, ox:ox + w])
    p0 = np.array([[20.0, 30.0], [41.5, 50.25]])            # channel 0's pixels in what is left of it
    o0, o1 = np.asarray(cut.origins[0]), np.asarray(cut.origins[1])
    np.testing.assert_allclose(cut.to_channel(1, p0), whole.to_channel(1, p0 + o0) - o1, atol=1e-12)
    np.testing.assert_allclose(cut.to_reference(1, cut.to_channel(1, p0)), p0, atol=1e-9)


def test_without_cut_into_a_roi_that_cuts_a_channel_is_refused_as_before():
    whole = ChannelLayout([(0, 0, 60, 100), (60, 0, 60, 100)])
    with pytest.raises(ValueError, match="cuts off a channel"):
        whole.on_camera((5, 0))
    moved = ChannelLayout([(10, 0, 50, 100), (60, 0, 50, 100)]).on_camera((5, 0))
    assert moved.regions[0] == (5, 0, 50, 100) and moved.origins == [(0, 0), (0, 0)]


def test_origins_travel_with_the_layout_in_its_record():
    whole = ChannelLayout([(0, 0, 60, 100), (60, 0, 60, 100)], maps=[np.eye(3), _map()])
    cut = whole.on_camera((7, 3), shape=(90, 100), cut_into=True)
    again = ChannelLayout.from_dict(cut.to_dict())
    assert again.origins == cut.origins and again.regions == cut.regions
    assert "origins" not in whole.to_dict()                 # a whole layout's record is as it was


# ------------------------------------------------------------- dual colour on the layer
def _geometry(shape, layout, main, split, mirror=None, roi=None):
    return {"layout": layout + (" mirrored" if mirror is not None else ""), "main_channel": main,
            "split_position": split, "image_shape": list(shape), "coordinate_system": "camera-chip",
            "sources": [{"roi": roi}] if roi else [], "mirror_axis_xy": mirror}


@pytest.mark.parametrize("layout, main, split, mirror, roi", [
    ("up-down", "upper", 48, None, None),
    ("up-down", "lower", 46, 1, None),
    ("right-left", "right", 52, 0, (10, 20, 110, 80)),
])
def test_a_dual_colour_registration_reads_as_two_channels_with_the_same_map(layout, main, split, mirror, roi):
    """`ChannelTransform.layout`: the main half channel 0, the other channel 1 flipped back along its
    mirror; a position in the main half, read into channel 1 and put back on the chip, is where the
    chip's map puts it."""
    from smappy.calibrate.dual import ChannelTransform, chip_halves
    shape = (80, 110) if layout == "right-left" else (100, 80)
    if roi:
        split += roi[0]
    g = _geometry(shape, layout, main, split, mirror, roi)
    H = _map(dx=0.4, dy=-0.3) @ np.diag([1.0, -1.0 if mirror == 1 else 1.0, 1.0])
    H[:2, 2] += (5.0, 300.0)
    t = ChannelTransform(H, g, {})
    lay = t.layout()
    (mx, my, mw, mh), (sx, sy, sw, sh) = chip_halves(g)
    assert lay.regions == [(mx, my, mw, mh), (sx, sy, sw, sh)]
    assert lay.flips[1] == (mirror == 0, mirror == 1)
    local = np.array([[10.0, 12.0], [30.5, 20.25]])
    there = lay.to_channel(1, local)
    fx, fy = lay.flips[1]
    chip = np.c_[sx + (sw - 1 - there[:, 0] if fx else there[:, 0]),
                 sy + (sh - 1 - there[:, 1] if fy else there[:, 1])]
    np.testing.assert_allclose(chip, t.to_secondary(local + (mx, my)), atol=1e-9)
    np.testing.assert_allclose(lay.to_reference(1, there), local, atol=1e-9)


def test_a_mirrored_spline_is_the_spline_of_the_flipped_bead_stack():
    from smappy.calibrate.core import spline_coefficients
    from smappy.calibrate.dual import mirrored_spline
    from smappy.io.calibration import SplineCalibration
    rng = np.random.default_rng(0)
    stack = rng.random((9, 11, 13))
    cal = SplineCalibration(spline_coefficients(stack), 10.0, 4.0, psf=stack)
    for axis in (1, 2):
        flipped = mirrored_spline(cal, axis)
        expected = spline_coefficients(np.flip(stack, axis))
        np.testing.assert_allclose(flipped.coeff, expected, atol=1e-5 * np.abs(expected).max())
        np.testing.assert_array_equal(flipped.psf, np.flip(stack, axis))


def test_a_polynomial_registration_rides_on_the_layout_exactly():
    from smappy.calibrate.dual import ChannelTransform
    from smappy.calibrate.transform import fit_polynomial
    g = _geometry((100, 80), "up-down", "upper", 50)
    rng = np.random.default_rng(1)
    secondary = np.c_[rng.uniform(5, 75, 200), rng.uniform(55, 95, 200)]
    bend = lambda p: np.c_[p[:, 0] + 0.3 + 1e-4 * (p[:, 1] - 75) ** 2, p[:, 1] - 50.2 + 2e-4 * p[:, 0] ** 2]
    poly = fit_polynomial(secondary, bend(secondary))
    t = ChannelTransform(np.eye(3), g, {}, poly)
    lay = t.layout()
    local = np.array([[20.0, 20.0], [60.5, 30.25]])
    np.testing.assert_allclose(lay.to_channel(1, local) + (0, 50), t.to_secondary(local), atol=1e-9)
