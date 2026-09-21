"""Registering two channels from localizations alone.

The claim the module makes is that it needs no initial guess: not the shift,
not the magnification, and above all not where the frame is split.  So the
tests give it geometries that a guess would get wrong -- all four layouts, a
seam nowhere near the middle, a rotation the vote cannot represent -- and ask
for the transformation back to a fraction of a pixel.
"""
import warnings

import numpy as np
import pytest

from smappy.calibrate.dual import map_points
from smappy.calibrate.transform import (RegisterSettings, load_channel_transform,
                                        load_transform, register_channels,
                                        save_channel_transform)

ROI = (177, 0, 207, 512)          # the strip the example dataset was taken on
SHAPE = (512, 207)


def simulate(n_frames=200, per_frame=6, angle_deg=1.5, scale=1.008,
             shift=(2.5, 11.0), mirrored=False, lone=0.3, stray=0.15,
             precision=0.02, seed=3, roi=ROI, split=256, axis=1, barrel=0.0):
    """A ratiometric splitter: every molecule imaged twice in the same frame.

    ``shift`` is the residual misalignment on top of the nominal half-frame
    step, ``lone`` the fraction of partners that blink in only one channel,
    and ``stray`` the rate of localizations belonging to no pair -- noise, and
    molecules whose partner fell off the chip.  ``axis`` is which coordinate
    the chip is split along, 0 for x and 1 for y.
    """
    rng = np.random.default_rng(seed)
    x0, y0, w, h = roi
    origin = (x0, y0)[axis]
    t = np.deg2rad(angle_deg)
    half = [w, h]
    half[axis] = split
    centre = np.array([x0 + half[0] / 2, y0 + half[1] / 2])
    rotation = scale * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    step = np.zeros(2)
    step[axis] = split

    def forward(p):
        """Where the reference half's coordinates land in the other half."""
        q = np.asarray(p, float) - centre
        if barrel:
            q = q * (1 + barrel * (q ** 2).sum(axis=1, keepdims=True) / 128.0 ** 2)
        q = q @ rotation.T + centre
        if mirrored:
            q = q.copy()
            q[:, axis] = 2 * (origin + split) - 1 - q[:, axis]
        else:
            q = q + step
        return q + np.asarray(shift)

    lo = np.array([x0, y0], float)
    hi = np.array([x0 + w, y0 + h], float)
    hi[axis] = origin + split
    xs, ys, fs = [], [], []
    for f in range(n_frames):
        k = rng.poisson(per_frame)
        ref = rng.uniform(lo, hi, (k, 2))
        sec = forward(ref) if k else np.empty((0, 2))
        keep = np.ones(k, bool)
        if k:
            keep = ((sec[:, 0] >= x0) & (sec[:, 0] < x0 + w)
                    & (sec[:, 1] >= y0) & (sec[:, 1] < y0 + h)
                    & (sec[:, axis] >= origin + split))
        ref, sec = ref[keep], sec[keep]
        ref = ref[rng.random(len(ref)) > lone / 2]
        sec = sec[rng.random(len(sec)) > lone / 2]
        n = rng.poisson(per_frame * stray)
        noise = np.c_[rng.uniform(x0, x0 + w, n), rng.uniform(y0, y0 + h, n)]
        points = np.vstack([ref, sec, noise])
        if not len(points):
            continue
        points = points + rng.normal(0, precision, points.shape)
        xs.append(points[:, 0])
        ys.append(points[:, 1])
        fs.append(np.full(len(points), f))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(fs), forward,
            lo, hi)


def grid_error(result, forward, lo, hi, n=9):
    """How far a point makes it there and back, over the reference half.

    The honest measure of a registration: not the residual of the pairs it
    chose to keep, which it fitted, but whether the map is right everywhere.
    """
    gx, gy = np.meshgrid(np.linspace(lo[0] + 3, hi[0] - 3, n),
                         np.linspace(lo[1] + 3, hi[1] - 3, n))
    reference = np.c_[gx.ravel(), gy.ravel()]
    back = result.transform.to_reference(forward(reference))
    return float(np.max(np.linalg.norm(back - reference, axis=1)))


def register(**kwargs):
    settings = kwargs.pop("settings", None)
    x, y, f, forward, lo, hi = simulate(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = register_channels(x, y, f, SHAPE, ROI, settings=settings)
    return result, forward, lo, hi


# ------------------------------------------------------------------ the map
def test_it_recovers_a_known_transformation():
    result, forward, lo, hi = register()
    assert grid_error(result, forward, lo, hi) < 0.05
    assert result.counts["inliers"] > 400


def test_a_rotation_and_a_scale_the_link_cannot_represent_are_still_fitted():
    """The vote only ever finds a translation; the rest is the refit's job."""
    result, forward, lo, hi = register(angle_deg=3.0, scale=1.02)
    assert grid_error(result, forward, lo, hi) < 0.05


@pytest.mark.parametrize("axis,mirrored,layout,split", [
    (1, False, "up-down", 256),
    (1, True, "up-down mirrored", 256),
    (0, False, "right-left", 103),
    (0, True, "right-left mirrored", 103),
])
def test_every_layout_is_recognised_without_being_told(axis, mirrored, layout, split):
    result, forward, lo, hi = register(axis=axis, mirrored=mirrored, split=split,
                                       shift=(6.0, 2.0) if axis == 0 else (2.5, 11.0))
    assert result.geometry["layout"] == layout
    assert grid_error(result, forward, lo, hi) < 0.05


def test_the_seam_is_measured_and_not_assumed_to_be_the_middle():
    """The whole reason for voting on pairs rather than on a rendered image."""
    result, _, _, _ = register(split=180)
    assert abs(result.geometry["split_position"] - 180) < 8
    assert result.geometry["image_shape"] == [512, 207]


def test_a_declared_layout_narrows_the_search_rather_than_being_ignored():
    x, y, f, *_ = simulate()
    settings = RegisterSettings(layout="right-left mirrored")
    with pytest.raises(ValueError):
        register_channels(x, y, f, SHAPE, ROI, settings=settings)


def test_a_given_split_wins_over_the_measured_one():
    result, _, _, _ = register(settings=RegisterSettings(split_position=300))
    assert result.geometry["split_position"] == 300
    assert result.transform.parameters["split_measured_px"] != 300


def test_the_main_channel_can_be_the_other_half():
    """Naming the lower half the reference turns the transformation round."""
    upper, forward, lo, hi = register()
    lower, _, _, _ = register(settings=RegisterSettings(main_channel="lower"))
    assert lower.geometry["main_channel"] == "lower"
    identity = lower.transformation @ upper.transformation
    identity = identity / identity[2, 2]
    np.testing.assert_allclose(identity, np.eye(3), atol=0.02)


# --------------------------------------------------------------- the stages
def test_the_tight_round_is_what_makes_it_accurate():
    """Coarse alone pairs everything but fits the outliers too; SMAP's second
    pass over clean matches only is what the accuracy comes from."""
    coarse, forward, lo, hi = register(
        angle_deg=3.0, scale=1.02, settings=RegisterSettings(fine_tolerance_px=0))
    staged, _, _, _ = register(
        angle_deg=3.0, scale=1.02, settings=RegisterSettings())
    assert grid_error(staged, forward, lo, hi) < grid_error(coarse, forward, lo, hi) / 3


def test_a_seam_the_pairs_cannot_place_says_so():
    """Halves so unequal that a wide band pairs with nothing: the split is
    then bracketed, not known, and that has to be said out loud."""
    x, y, f, *_ = simulate(split=330)
    with pytest.warns(UserWarning, match="no pairs in it"):
        result = register_channels(x, y, f, SHAPE, ROI)
    low, high = result.transform.parameters["split_interval_px"]
    assert high - low > 50


# ------------------------------------------------------------------ refusals
def test_it_refuses_what_it_cannot_register():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 10)
    with pytest.raises(ValueError, match="too few"):
        register_channels(x, x, np.arange(10), SHAPE, ROI)
    x, y, f, *_ = simulate(n_frames=40, per_frame=1, lone=0.9)
    with pytest.raises(ValueError):
        register_channels(x, y, f, SHAPE, ROI)


def test_the_settings_refuse_nonsense():
    for bad in (dict(layout="diagonal"), dict(coarse_tolerance_px=0),
                dict(coarse_tolerance_px=1.0, fine_tolerance_px=5.0),
                dict(vote_bin_px=0), dict(min_pairs=2),
                dict(max_pairs=1), dict(main_channel="upper", layout="right-left")):
        with pytest.raises(ValueError):
            RegisterSettings(**bad).validate()


# ------------------------------------------------------------------- storage
def test_a_saved_transformation_comes_back_the_same(tmp_path):
    result, forward, lo, hi = register()
    path = tmp_path / "reg_2ct.h5"
    result.save(path)
    loaded = load_channel_transform(path)
    np.testing.assert_allclose(loaded.transformation, result.transformation)
    assert loaded.geometry == result.geometry
    assert loaded.parameters["n_inliers"] == result.counts["inliers"]
    with pytest.raises(FileExistsError):
        result.save(path)
    result.save(path, overwrite=True)

    # and it is the same object the generic loader hands back
    again = load_transform(path)
    np.testing.assert_allclose(again.transformation, result.transformation)
    points = np.c_[[200.0, 250.0], [300.0, 400.0]]
    np.testing.assert_allclose(again.transform(points), loaded.transform(points))


def test_the_loader_also_takes_a_bead_calibration(tmp_path):
    """A 2D two-colour fit needs only the registration, so a dual-colour bead
    calibration serves it as well as one measured from localizations."""
    from smappy.calibrate.dual import save_dual_color_calibration
    from test_dual_fit import dual_calibration

    cal = dual_calibration(ratio=0.3)
    path = tmp_path / "beads_2c.h5"
    save_dual_color_calibration(path, cal)
    loaded = load_transform(path)
    np.testing.assert_allclose(loaded.transformation, cal.transformation)
    assert loaded.geometry["layout"] == cal.geometry["layout"]


# ---------------------------------------------------------------- polynomial
def test_a_quadratic_could_not_help_and_a_cubic_can():
    """The equation says the leading distortion term is cubic, so the model
    has to reach third order; this is that statement, measured."""
    from smappy.calibrate.transform import fit_polynomial

    rng = np.random.default_rng(0)
    source = rng.uniform(-100, 100, (3000, 2))
    r2 = (source ** 2).sum(axis=1, keepdims=True) / 100.0 ** 2
    target = source * (1 + 0.05 * r2) + (3.0, -2.0)

    quadratic = fit_polynomial(source, target, order=2)
    cubic = fit_polynomial(source, target, order=3)
    assert np.abs(quadratic.forward(source) - target).max() > 1.0
    assert np.abs(cubic.forward(source) - target).max() < 1e-6


def test_the_inverse_is_an_inverse_and_not_merely_a_backward_fit():
    """The backward cubic is only the seed: it is half a pixel out on a badly
    distorted field, and that error would land in the partner ROI."""
    from smappy.calibrate.transform import fit_polynomial

    rng = np.random.default_rng(1)
    source = rng.uniform(-100, 100, (3000, 2))
    r2 = (source ** 2).sum(axis=1, keepdims=True) / 100.0 ** 2
    target = source * (1 + 0.05 * r2) + (3.0, -2.0)
    poly = fit_polynomial(source, target)

    seed = np.abs(poly._seed(poly.forward(source)) - source).max()
    refined = np.abs(poly.inverse(poly.forward(source)) - source).max()
    assert seed > 0.1                      # a cubic is not the inverse of a cubic
    assert refined < 1e-6 < seed


def test_a_distorted_field_is_described_far_better_by_the_polynomial():
    x, y, f, forward, lo, hi = simulate(barrel=0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = register_channels(x, y, f, SHAPE, ROI,
                                 settings=RegisterSettings(model="projective"))
        bent = register_channels(x, y, f, SHAPE, ROI,
                                 settings=RegisterSettings(model="polynomial"))
    assert flat.transform.model == "projective"
    assert bent.transform.model == "polynomial"
    assert grid_error(bent, forward, lo, hi) < grid_error(flat, forward, lo, hi) / 5


def test_a_clean_field_is_not_made_worse_by_it():
    """The cost of twenty coefficients where eight would do: a fraction of a
    nanometre, which is the reason it is offered at all."""
    x, y, f, forward, lo, hi = simulate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bent = register_channels(x, y, f, SHAPE, ROI,
                                 settings=RegisterSettings(model="polynomial"))
    assert grid_error(bent, forward, lo, hi) < 0.05


def test_it_refuses_a_polynomial_the_pairs_cannot_support():
    """Twenty coefficients fitted over a strip do whatever they like off it,
    and nothing in the residual would say so.

    The reference channel keeps its whole field; only its partners are taken
    away, so the pairs speak for a strip while the localizations span
    everything -- which is the situation the gate exists for, and the one a
    real dataset with one dim channel actually lands in.
    """
    x, y, f, forward, lo, hi = simulate()
    band = (y < 256) | ((y > 256 + 40) & (y < 256 + 110))
    with pytest.warns(UserWarning, match="keeping the projective fit"):
        result = register_channels(x[band], y[band], f[band], SHAPE, ROI,
                                   settings=RegisterSettings(model="polynomial",
                                                             split_position=256))
    assert result.transform.model == "projective"


def test_it_refuses_a_polynomial_too_few_pairs_can_support():
    x, y, f, forward, lo, hi = simulate(n_frames=15)
    with pytest.warns(UserWarning, match="too few for a polynomial"):
        result = register_channels(x, y, f, SHAPE, ROI,
                                   settings=RegisterSettings(model="polynomial"))
    assert result.transform.model == "projective"


def test_a_saved_polynomial_comes_back_exactly(tmp_path):
    x, y, f, forward, lo, hi = simulate(barrel=0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = register_channels(x, y, f, SHAPE, ROI,
                                   settings=RegisterSettings(model="polynomial"))
    path = tmp_path / "poly_2ct.h5"
    result.save(path)
    loaded = load_transform(path)
    assert loaded.model == "polynomial" and loaded.polynomial.order == 3
    probe = np.c_[[200.0, 250.0, 300.0], [30.0, 120.0, 200.0]]
    np.testing.assert_allclose(loaded.to_reference(probe),
                               result.transform.to_reference(probe))
    np.testing.assert_allclose(loaded.to_secondary(probe),
                               result.transform.to_secondary(probe))
    # the projective fit is written alongside, so a reader that knows nothing
    # of polynomials still gets a usable transformation
    np.testing.assert_allclose(loaded.transformation, result.transformation)


# ------------------------------------------------- weighting by precision
def simulate_mixed(bright=3, dim=12, sigma_bright=0.03, sigma_dim=0.35,
                   n_frames=150, seed=5):
    """Well-localized pairs, plus the poor ones a lower threshold adds.

    Returns the precision of each localization alongside it, which is what
    the fit needs in order not to be led by its worst pairs.
    """
    rng = np.random.default_rng(seed)
    x0, y0, w, h = ROI
    split = 256
    t = np.deg2rad(2.0)
    centre = np.array([x0 + w / 2, y0 + split / 2])
    rotation = 1.006 * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def forward(p):
        return ((np.asarray(p, float) - centre) @ rotation.T + centre
                + (0, split) + (3.0, 9.0))

    xs, ys, fs, ps = [], [], [], []
    for f in range(n_frames):
        for rate, sigma in ((bright, sigma_bright), (dim, sigma_dim)):
            n = rng.poisson(rate)
            if not n:
                continue
            ref = np.c_[rng.uniform(x0 + 6, x0 + w - 6, n),
                        rng.uniform(y0 + 6, split - 6, n)]
            sec = forward(ref)
            ok = ((sec[:, 0] >= x0 + 1) & (sec[:, 0] < x0 + w - 1)
                  & (sec[:, 1] >= split + 1) & (sec[:, 1] < y0 + h - 1))
            ref, sec = ref[ok], sec[ok]
            pts = np.vstack([ref, sec]) + rng.normal(0, sigma, (2 * len(ref), 2))
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
            fs.append(np.full(len(pts), f))
            ps.append(np.full(len(pts), sigma))
    order = np.argsort(np.concatenate(fs), kind="stable")
    return (np.concatenate(xs)[order], np.concatenate(ys)[order],
            np.concatenate(fs)[order], np.concatenate(ps)[order], forward)


def test_dim_pairs_are_believed_only_as_far_as_they_deserve():
    """A lower detection threshold buys dim pairs, and dim pairs are what a
    two-colour dataset has to lower the threshold for.  Unweighted they drag
    the fit toward their own noise; weighted they cost nothing."""
    x, y, f, precision, forward = simulate_mixed(bright=3, dim=12)
    lo = np.array([ROI[0], ROI[1]], float)
    hi = np.array([ROI[0] + ROI[2], 256.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = register_channels(x, y, f, (512, ROI[2]), ROI)
        weighted = register_channels(x, y, f, (512, ROI[2]), ROI,
                                     precision=precision)
    # ~2.4x at this simulation size, and much more at a realistic one -- the
    # bar is where it is so the test does not ride the edge of its own noise
    assert (grid_error(weighted, forward, lo, hi)
            < grid_error(flat, forward, lo, hi) / 1.5)


def test_equal_pairs_are_unaffected_by_weighting():
    """Weighting must not be doing anything when there is nothing to weigh."""
    x, y, f, precision, forward = simulate_mixed(bright=6, dim=0)
    lo = np.array([ROI[0], ROI[1]], float)
    hi = np.array([ROI[0] + ROI[2], 256.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = register_channels(x, y, f, (512, ROI[2]), ROI)
        weighted = register_channels(x, y, f, (512, ROI[2]), ROI,
                                     precision=precision)
    a = grid_error(flat, forward, lo, hi)
    b = grid_error(weighted, forward, lo, hi)
    assert abs(a - b) < 0.3 * max(a, b, 1e-6)


def test_the_weights_are_the_inverse_variance_of_the_pair():
    from smappy.calibrate.transform import pair_weights

    sigma = np.array([0.1, 0.1, 0.2, np.nan])
    w = pair_weights(sigma, np.array([0, 2]), np.array([1, 2]))
    # 1/(0.01+0.01) against 1/(0.04+0.04): a factor of four, before normalising
    assert np.isclose(w[0] / w[1], 4.0)
    assert np.isclose(w.mean(), 1.0)
    assert pair_weights(None, np.array([0]), np.array([1])) is None
