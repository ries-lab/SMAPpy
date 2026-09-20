"""The global (multi-channel) spline fit: one emitter, several channels, some
parameters fitted once for all of them.

The port's own claim is that it is `lm.hpp` over a linked parameter vector, so
the tests hold it to that: with one channel and everything shared it must be
the single-channel fitter to the last bit, and with two channels it must beat
either channel alone on the parameters they share.
"""
import numpy as np
import pytest

from smappy import _fit3d
from smappy.calibrate.core import spline_coefficients
from smappy.io.calibration import SplineCalibration, evaluate_spline

# (x, y, photons, background, z) -- the fitter's order everywhere in smappy
X, Y, N, BG, Z = range(5)


def psf_volume(nz=41, size=17, astigmatism=0.6):
    """An astigmatic Gaussian PSF stack, normalised per plane."""
    z, y, x = np.mgrid[:nz, :size, :size].astype(float)
    t = (z - (nz - 1) / 2) / 12
    sx = 1.2 * np.sqrt(1 + (t - astigmatism) ** 2)
    sy = 1.2 * np.sqrt(1 + (t + astigmatism) ** 2)
    return np.exp(-.5 * (((x - size // 2) / sx) ** 2 + ((y - size // 2) / sy) ** 2)) \
        / (2 * np.pi * sx * sy)


def calibration(astigmatism=0.6):
    volume = psf_volume(astigmatism=astigmatism)
    return SplineCalibration(spline_coefficients(volume), 10., 20., psf=volume,
                             em_mirror=False, parameters={})


def render(cal, x, y, z_index, photons, background, sz, rng):
    """One Poisson ROI of an emitter at (x, y) in the ROI, z in spline units."""
    image = evaluate_spline(cal, x, y, z_index, sz) * photons + background
    return rng.poisson(np.clip(image, 0, None)).astype(np.float32)


def link_array(n, channels, offsets=None, factors=None):
    """The (n, 2, C, 5) link: offsets then factors, per channel."""
    link = np.zeros((n, 2, channels, 5), np.float32)
    link[:, 1] = 1.0
    if offsets is not None:
        link[:, 0] = offsets
    if factors is not None:
        link[:, 1] = factors
    return link


def coeff_stack(*cals):
    return np.ascontiguousarray(np.stack([c.coeff for c in cals]), np.float32)


# --------------------------------------------------------------- one channel
def test_one_channel_all_shared_is_the_single_channel_fitter():
    """Nothing is linked when there is nothing to link to, so the global loop
    must reduce to `lm.hpp` exactly -- same arithmetic, same order."""
    rng = np.random.default_rng(0)
    cal, sz = calibration(), 13
    rois = np.stack([render(cal, 6.3, 6.8, 22.0, 2000., 20., sz, rng)
                     for _ in range(40)])

    plain = _fit3d.fit_cspline(rois, cal.coeff, 20.0, 50, 1)
    joint = _fit3d.fit_cspline_global(rois[:, None], coeff_stack(cal),
                                      link_array(len(rois), 1),
                                      np.ones(5, np.int32), 20.0, 50, 1)

    for a, b, name in zip(plain, joint, ("theta", "crlb", "logl", "iterations")):
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_the_parameter_count_follows_which_parameters_are_shared():
    rng = np.random.default_rng(1)
    cal, sz = calibration(), 13
    rois = np.stack([render(cal, 6.5, 6.5, 20.0, 1500., 15., sz, rng) for _ in range(6)])
    pair = np.stack([rois, rois], axis=1)
    coeff = coeff_stack(cal, cal)

    for shared, nv in (([1, 1, 1, 1, 1], 5),      # everything linked
                       ([1, 1, 0, 0, 1], 7),      # x, y, z linked; N and bg free
                       ([0, 0, 0, 0, 0], 10)):    # two independent fits
        theta, crlb, logl, _ = _fit3d.fit_cspline_global(
            pair, coeff, link_array(len(rois), 2), np.array(shared, np.int32),
            20.0, 50, 1)
        assert theta.shape == (len(rois), nv) and crlb.shape == (len(rois), nv)
        assert np.isfinite(logl).all()


def test_unlinking_everything_gives_each_channel_its_own_single_channel_fit():
    """With no parameter shared the two channels never meet, so each half of
    the answer has to be what fitting that channel alone gives."""
    rng = np.random.default_rng(2)
    cal_a, cal_b, sz = calibration(0.6), calibration(-0.4), 13
    a = np.stack([render(cal_a, 6.2, 6.9, 24.0, 3000., 20., sz, rng) for _ in range(20)])
    b = np.stack([render(cal_b, 6.7, 6.1, 24.0, 2000., 30., sz, rng) for _ in range(20)])

    separate = [_fit3d.fit_cspline(r, c.coeff, 20.0, 50, 1)[0]
                for r, c in ((a, cal_a), (b, cal_b))]
    theta, _, _, _ = _fit3d.fit_cspline_global(
        np.stack([a, b], axis=1), coeff_stack(cal_a, cal_b),
        link_array(len(a), 2), np.zeros(5, np.int32), 20.0, 50, 1)

    for p in (X, Y, N, BG, Z):                     # free: two slots per parameter
        for channel in (0, 1):
            np.testing.assert_allclose(theta[:, 2 * p + channel],
                                       separate[channel][:, p], rtol=1e-5, atol=1e-5)


# -------------------------------------------------------------- two channels
def test_sharing_xyz_across_two_channels_beats_either_channel_alone():
    """The point of the global fit: twice the photons on the shared
    parameters, so z -- the expensive one -- comes out better."""
    rng = np.random.default_rng(3)
    cal_a, cal_b, sz = calibration(0.6), calibration(-0.6), 13
    true_z, photons = 26.0, 800.
    a = np.stack([render(cal_a, 6.4, 6.6, true_z, photons, 15., sz, rng) for _ in range(300)])
    b = np.stack([render(cal_b, 6.4, 6.6, true_z, photons, 15., sz, rng) for _ in range(300)])

    alone = [_fit3d.fit_cspline(r, c.coeff, 20.0, 60, 0)[0][:, Z]
             for r, c in ((a, cal_a), (b, cal_b))]
    theta, crlb, _, _ = _fit3d.fit_cspline_global(
        np.stack([a, b], axis=1), coeff_stack(cal_a, cal_b), link_array(len(a), 2),
        np.array([1, 1, 0, 0, 1], np.int32), 20.0, 60, 0)
    # x, y, z shared (slots 0, 1); N and bg free (2, 3 and 4, 5); z at slot 6
    joint_z = theta[:, 6]

    assert abs(np.median(joint_z) - true_z) < 0.4          # and it is unbiased
    spread = np.std(joint_z)
    assert spread < min(np.std(z) for z in alone) * 0.85   # ~sqrt(2) better
    # the reported bound agrees with what the scatter actually is
    assert np.median(np.sqrt(np.clip(crlb[:, 6], 0, None))) == pytest.approx(spread, rel=0.25)


def test_an_offset_between_the_channels_is_taken_out_by_the_link():
    """The second ROI is cut a little off centre, as it always is once a
    registration has been applied; the offset in the link puts it back."""
    rng = np.random.default_rng(4)
    cal, sz = calibration(), 13
    dx, dy = 0.35, -0.42
    a = np.stack([render(cal, 6.5, 6.5, 22.0, 4000., 10., sz, rng) for _ in range(120)])
    b = np.stack([render(cal, 6.5 + dx, 6.5 + dy, 22.0, 4000., 10., sz, rng)
                  for _ in range(120)])

    offsets = np.zeros((2, 5), np.float32)
    offsets[1, X], offsets[1, Y] = dx, dy
    theta, _, _, _ = _fit3d.fit_cspline_global(
        np.stack([a, b], axis=1), coeff_stack(cal, cal),
        link_array(len(a), 2, offsets=offsets), np.array([1, 1, 0, 0, 1], np.int32),
        20.0, 60, 0)

    assert np.median(theta[:, X]) == pytest.approx(6.5, abs=0.05)
    assert np.median(theta[:, Y]) == pytest.approx(6.5, abs=0.05)

    ignored = _fit3d.fit_cspline_global(
        np.stack([a, b], axis=1), coeff_stack(cal, cal), link_array(len(a), 2),
        np.array([1, 1, 0, 0, 1], np.int32), 20.0, 60, 0)[0]
    assert abs(np.median(ignored[:, X]) - 6.5) > 0.1       # without it, biased


def test_a_photon_factor_links_a_ratiometric_split():
    """A splitter that sends a fixed fraction to each channel: the photon
    number is one parameter with a factor per channel, and the fit returns the
    total as seen through channel 0."""
    rng = np.random.default_rng(5)
    cal, sz = calibration(), 13
    total, fraction = 5000., 0.3
    a = np.stack([render(cal, 6.5, 6.5, 21.0, total * (1 - fraction), 12., sz, rng)
                  for _ in range(150)])
    b = np.stack([render(cal, 6.5, 6.5, 21.0, total * fraction, 12., sz, rng)
                  for _ in range(150)])

    factors = np.ones((2, 5), np.float32)
    factors[1, N] = fraction / (1 - fraction)
    theta, _, _, _ = _fit3d.fit_cspline_global(
        np.stack([a, b], axis=1), coeff_stack(cal, cal),
        link_array(len(a), 2, factors=factors), np.array([1, 1, 1, 0, 1], np.int32),
        20.0, 60, 0)

    # N shared at slot 2, in channel-0 units: the photons that channel sees
    assert np.median(theta[:, 2]) == pytest.approx(total * (1 - fraction), rel=0.05)


def test_a_bad_channel_poisons_the_pair_rather_than_the_process():
    """Empty or negative ROIs happen at the edge of a split frame.  GlobLoc
    does not guard the model against going negative and neither does this
    port, so such a pair comes back non-finite -- which is right, since the
    two channels are one measurement -- and `locs.valid` drops it.  What the
    fitter must not do is crash or return the wrong shape."""
    cal, sz = calibration(), 13
    rng = np.random.default_rng(6)
    good = render(cal, 6.5, 6.5, 20.0, 2000., 10., sz, rng)
    pair = np.ascontiguousarray(np.stack([
        np.stack([good, np.zeros((sz, sz), np.float32)]),
        np.stack([good, np.full((sz, sz), -5.0, np.float32)])]))

    theta, crlb, logl, iters = _fit3d.fit_cspline_global(
        pair, coeff_stack(cal, cal), link_array(2, 2), np.array([1, 1, 0, 0, 1], np.int32),
        20.0, 50, 1)
    assert theta.shape == (2, 7) and crlb.shape == (2, 7) and iters.shape == (2,)
    assert not np.isfinite(theta).any()      # and the pair is discardable


def test_the_shapes_it_refuses():
    cal = calibration()
    rois = np.zeros((3, 2, 13, 13), np.float32)
    coeff, link, shared = coeff_stack(cal, cal), link_array(3, 2), np.ones(5, np.int32)
    with pytest.raises(ValueError, match="channels, sz, sz"):
        _fit3d.fit_cspline_global(rois[0], coeff, link, shared, 20.0)
    with pytest.raises(ValueError, match="channels, 64"):
        _fit3d.fit_cspline_global(rois, coeff[:1], link, shared, 20.0)
    with pytest.raises(ValueError, match="link must have shape"):
        _fit3d.fit_cspline_global(rois, coeff, link[:, :1], shared, 20.0)
    with pytest.raises(ValueError, match="5 flags"):
        _fit3d.fit_cspline_global(rois, coeff, link, shared[:4], 20.0)
