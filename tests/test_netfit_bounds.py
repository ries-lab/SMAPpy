"""netfit's plug-in Cramer-Rao bound (`smappy.netfit.bounds`): each localization's bound from its own
estimates on two channels, and over the frames a fit reads.

The dual calibration is the channel tests' synthetic one: an up-down splitter with a mirror, a small
rotation and scale."""
import numpy as np
import pytest

from smappy.calibrate.core import spline_coefficients
from smappy.calibrate.dual import DualColorCalibration, split_geometry
from smappy.io.calibration import SplineCalibration
from smappy.netfit.bounds import links, plugin_crlb
from smappy.netfit.channels import TwoChannels

SHAPE, SPLIT, PIXEL = (160, 96), 80, (100.0, 100.0)


def _spline(astigmatism, flip=False):
    z, y, x = np.mgrid[:21, :17, :17].astype(float)
    t = (z - 10) / 6
    sx, sy = 1.3 * np.sqrt(1 + (t - astigmatism) ** 2), 1.3 * np.sqrt(1 + (t + astigmatism) ** 2)
    v = np.exp(-.5 * (((x - 8.5) / sx) ** 2 + ((y - 8.5) / sy) ** 2)) / (2 * np.pi * sx * sy)
    v = np.flip(v, 1).copy() if flip else v
    return SplineCalibration(spline_coefficients(v), 50., 10., x0=8.0, psf=v, parameters={})


@pytest.fixture(scope="module")
def two():
    a, s = 0.01, 1.01
    t = np.array([[s * np.cos(a), -s * np.sin(a), 3.4], [-s * np.sin(a), -s * np.cos(a), 2 * SPLIT - 1 - 5.7],
                  [0., 0., 1.]])
    cal = DualColorCalibration(_spline(0.5), _spline(-0.5, flip=True), t,
                               split_geometry("up-down mirrored", "upper", SHAPE, split_position=SPLIT), {})
    return cal, TwoChannels.from_calibration(cal)


def _bound(cal, two, kind, frames, photons, background=20.0, window=1, xy=(40.3, 30.6)):
    n = len(frames)
    xy = np.tile(np.asarray(xy, float), (n, 1))
    ph = np.full(n, photons) if kind == "biplane" else np.tile(photons, (n, 1))
    return plugin_crlb(two, cal, kind, np.asarray(frames), xy, np.zeros(n), ph, np.full((n, 2), background),
                       PIXEL, em=False, window=window)


@pytest.mark.parametrize("kind", ["biplane", "colour"])
def test_a_molecule_on_in_three_frames_alike_has_its_frame_s_bound_over_the_root_of_three(two, kind):
    cal, tw = two
    ph = 3000.0 if kind == "biplane" else np.array([1000.0, 2000.0])
    one = _bound(cal, tw, kind, [5], ph)[0]
    three = _bound(cal, tw, kind, [4, 5, 6], ph, window=3)[1]
    assert np.all(np.isfinite(one)) and np.all(one > 0)
    np.testing.assert_allclose(three, one / np.sqrt(3), rtol=1e-6)


def test_four_times_the_photons_without_background_halve_the_bound(two):
    cal, tw = two
    a = _bound(cal, tw, "colour", [0], np.array([1000.0, 1000.0]), background=0.0)[0]
    b = _bound(cal, tw, "colour", [0], np.array([4000.0, 4000.0]), background=0.0)[0]
    np.testing.assert_allclose(b, a / 2, rtol=1e-3)


def test_both_channels_bound_a_molecule_tighter_than_either_alone(two):
    cal, tw = two
    both = _bound(cal, tw, "colour", [0], np.array([2000.0, 2000.0]))[0]
    only0 = _bound(cal, tw, "colour", [0], np.array([2000.0, 1e-3]))[0]
    only1 = _bound(cal, tw, "colour", [0], np.array([1e-3, 2000.0]))[0]
    assert np.all(both < only0) and np.all(both < only1)


def test_a_spot_further_than_the_link_in_the_next_frame_is_another_molecule():
    frame = np.array([0, 1, 1])
    xy = np.array([[10.0, 10.0], [10.4, 10.2], [13.0, 10.0]])
    rows, partners = links(frame, xy, 1, link_px=1.0)[1]          # offset +1
    assert dict(zip(rows.tolist(), partners.tolist())) == {0: 1}
