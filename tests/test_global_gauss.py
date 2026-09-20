"""The global Gaussian fit: the 2D two-colour kernel.

`fit_gauss_global` is `fit_cspline_global`'s twin -- the same `global_fit`
template over a linked parameter vector, with `GaussFree` in place of the
spline.  So the tests are the spline's, asked of the Gaussian: with one channel
and everything shared it must be the single-channel fitter bit for bit, and
with two channels linked through a registration it must recover the position
and the photon split that carries the colour.
"""
import numpy as np
import pytest
from scipy.special import erf

from smappy import _fit3d
from smappy.psf import GlobalGaussianPSF

# (x, y, photons, background, sigma) -- the fitter's order, with sigma where
# the spline keeps z
X, Y, N, BG, S = range(5)


def render(x, y, photons, background, sigma, sz, rng=None):
    """One ROI of an emitter at (x, y), pixel-integrated as the model is."""
    ii = np.arange(sz)
    def axis(centre):
        norm = np.sqrt(1 / 2 / sigma ** 2)
        return .5 * (erf((ii - centre + .5) * norm) - erf((ii - centre - .5) * norm))
    image = background + photons * np.outer(axis(y), axis(x))
    if rng is None:
        return image.astype(np.float32)
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


def sigmas(*values):
    return np.array(values, np.float32)


# --------------------------------------------------------------- one channel
def test_one_channel_all_shared_is_the_single_channel_fitter():
    """Nothing is linked when there is nothing to link to, so the global loop
    must reduce to `lm.hpp` exactly -- same arithmetic, same order."""
    rng = np.random.default_rng(0)
    sz = 13
    rois = np.stack([render(6.3, 6.8, 2000., 20., 1.3, sz, rng) for _ in range(40)])

    plain = _fit3d.fit_gauss_free(rois, 1.2, 50, 1)
    joint = _fit3d.fit_gauss_global(rois[:, None], sigmas(1.2),
                                    link_array(len(rois), 1),
                                    np.ones(5, np.int32), 50, 1)

    for a, b, name in zip(plain, joint, ("theta", "crlb", "logl", "iterations")):
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_the_parameter_count_follows_which_parameters_are_shared():
    rng = np.random.default_rng(1)
    sz = 13
    rois = np.stack([render(6.5, 6.5, 1500., 15., 1.2, sz, rng) for _ in range(6)])
    pair = np.stack([rois, rois], axis=1)

    for shared, nv in (([1, 1, 1, 1, 1], 5),      # everything linked
                       ([1, 1, 0, 0, 0], 8),      # the two-colour default
                       ([0, 0, 0, 0, 0], 10)):    # two independent fits
        theta, crlb, logl, _ = _fit3d.fit_gauss_global(
            pair, sigmas(1.2, 1.2), link_array(len(rois), 2),
            np.array(shared, np.int32), 50, 1)
        assert theta.shape == (len(rois), nv) and crlb.shape == (len(rois), nv)
        assert np.isfinite(logl).all()


# -------------------------------------------------------------- two channels
def two_colour(n, ratio=0.35, shift=(0.4, -0.25), widths=(1.1, 1.4), sz=13,
               photons=4000., background=20., seed=2):
    """A stack of paired ROIs, a splitter's two halves of the same emitter.

    Channel 1 sits a sub-pixel `shift` away and keeps `ratio` of the photons --
    the registration and the colour, exactly what the link carries.
    """
    rng = np.random.default_rng(seed)
    x, y = 6.4, 6.55
    rois = np.stack([
        np.stack([render(x, y, photons * (1 - ratio), background, widths[0], sz, rng),
                  render(x + shift[0], y + shift[1], photons * ratio, background,
                         widths[1], sz, rng)])
        for _ in range(n)])
    link = link_array(n, 2)
    link[:, 0, 1, X] = shift[0]          # the sub-pixel remainder of the
    link[:, 0, 1, Y] = shift[1]          # transformed position, as `build_link`
    return np.ascontiguousarray(rois, np.float32), link, (x, y)


def test_a_linked_pair_recovers_the_position_and_the_photon_split():
    """The two-colour default: x and y shared through the registration, the
    photons free, which is what makes the split readable as a colour."""
    rois, link, (x, y) = two_colour(300)
    model = GlobalGaussianPSF(sigma=(1.2, 1.2))
    fitted = model.fit(rois, link, iterations=60, n_threads=1)
    p = model.unpack(fitted)

    assert abs(np.median(p["x_roi"]) - x) < 0.02
    assert abs(np.median(p["y_roi"]) - y) < 0.02
    assert abs(np.median(p["ratio"]) - 0.35) < 0.01
    # the widths were free, so each channel must have found its own
    assert abs(np.median(p["sigma_pix_ch0"]) - 1.1) < 0.03
    assert abs(np.median(p["sigma_pix_ch1"]) - 1.4) < 0.03
    # and the reference channel's is what carries no suffix
    np.testing.assert_array_equal(p["sigma_pix"], p["sigma_pix_ch0"])


def test_the_link_offset_is_what_puts_the_second_roi_in_the_right_place():
    """Drop the sub-pixel shift from the link and the fit is pulled off: the
    offset is the registration, not decoration."""
    rois, link, (x, y) = two_colour(200, shift=(0.45, 0.45))
    model = GlobalGaussianPSF(sigma=(1.2, 1.2))

    with_offset = model.unpack(model.fit(rois, link, iterations=60, n_threads=1))
    without = link.copy()
    without[:, 0, 1, [X, Y]] = 0.0
    blind = model.unpack(model.fit(rois, without, iterations=60, n_threads=1))

    good = abs(np.median(with_offset["x_roi"]) - x)
    bad = abs(np.median(blind["x_roi"]) - x)
    assert good < 0.02 < bad


def test_sharing_x_and_y_beats_either_channel_alone():
    """The point of linking: both channels' photons measure one position."""
    rois, link, (x, y) = two_colour(400, ratio=0.5, widths=(1.2, 1.2))
    joint = GlobalGaussianPSF(sigma=(1.2, 1.2))
    together = joint.unpack(joint.fit(rois, link, iterations=60, n_threads=1))

    alone = _fit3d.fit_gauss_free(np.ascontiguousarray(rois[:, 0]), 1.2, 60, 1)
    spread_alone = np.std(alone[0][:, X])
    spread_together = np.std(together["x_roi"])
    # half the photons each, so one channel alone is sqrt(2) worse; allow slack
    assert spread_together < spread_alone / 1.25


def test_a_photon_factor_makes_the_shared_number_the_total():
    """With the photons shared, the link's factor is the splitter's ratio and
    the one fitted number is the total as channel 0 sees it."""
    ratio = 0.4
    rois, link, _ = two_colour(200, ratio=ratio, widths=(1.2, 1.2))
    link[:, 1, 0, N] = 1 - ratio
    link[:, 1, 1, N] = ratio
    model = GlobalGaussianPSF(sigma=(1.2, 1.2),
                              shared=(True, True, True, False, False))
    p = model.unpack(model.fit(rois, link, iterations=60, n_threads=1))

    assert abs(np.median(p["photons"]) - 4000.) < 150.
    # nothing to divide, so no colour: `ratio` is zero by construction
    assert np.all(p["ratio"] == 0)


# ------------------------------------------------------------------- refusals
def test_the_shapes_it_refuses():
    rois = np.zeros((3, 2, 13, 13), np.float32)
    link, shared = link_array(3, 2), np.ones(5, np.int32)
    with pytest.raises(ValueError, match="channels, sz, sz"):
        _fit3d.fit_gauss_global(rois[0], sigmas(1.2, 1.2), link, shared)
    with pytest.raises(ValueError, match="one starting width per channel"):
        _fit3d.fit_gauss_global(rois, sigmas(1.2), link, shared)
    with pytest.raises(ValueError, match="link must have shape"):
        _fit3d.fit_gauss_global(rois, sigmas(1.2, 1.2), link[:, :1], shared)
    with pytest.raises(ValueError, match="5 flags"):
        _fit3d.fit_gauss_global(rois, sigmas(1.2, 1.2), link, shared[:4])


def test_the_model_refuses_a_width_it_cannot_place():
    with pytest.raises(ValueError, match="one starting width per channel"):
        GlobalGaussianPSF(sigma=(1.1, 1.2, 1.3), channels=2)
    with pytest.raises(ValueError, match="must be positive"):
        GlobalGaussianPSF(sigma=(1.1, 0.0))
    with pytest.raises(ValueError, match="link"):
        GlobalGaussianPSF().fit(np.zeros((2, 2, 13, 13), np.float32))


def test_a_scalar_width_is_used_for_every_channel():
    model = GlobalGaussianPSF(sigma=1.3, channels=3)
    assert model.sigma == (1.3, 1.3, 1.3) and model.n_channels == 3
    assert model.n_values == 2 + 3 * 3       # x, y shared; N, bg, sigma free
