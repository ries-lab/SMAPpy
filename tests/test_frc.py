"""The FRC, against pictures whose resolution is known by construction."""
import numpy as np
import pytest

from smappy.frc import (blur_envelope, envelope_resolution, fpc_resolution,
                        frc_curve, frc_resolution, resolution_from_curve,
                        taper, time_blocks)


def picture(sigma=8.0, n_emitters=20000, repeats=20, extent=10000.0,
            n_frames=2000, seed=1, drift=0.0):
    """A fixed random structure, localized `repeats` times per molecule.

    The emitters are the structure both halves of the split have to agree
    about; the localization error is what stops them agreeing at high
    frequency.  The frames are drawn per localization, so every molecule is
    seen throughout the acquisition -- a structure that appears only in the
    first half of the movie is a structure the two halves do not share, and no
    FRC of it means anything.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, extent, (n_emitters, 2))
    which = np.repeat(np.arange(n_emitters), repeats)
    frame = rng.integers(0, n_frames, len(which))
    xy = truth[which] + rng.normal(0, sigma, (len(which), 2))
    if drift:
        xy = xy + drift * frame[:, None] * np.array([1.0, 0.0])
    return xy[:, 0], xy[:, 1], frame


def test_the_resolution_follows_the_precision_it_was_given():
    # with the sampling held, doubling the localization error doubles what the
    # picture resolves -- the curve is reading the blur and nothing else
    fine = frc_resolution(*picture(sigma=6.0), repeats=3)
    coarse = frc_resolution(*picture(sigma=12.0), repeats=3)
    assert fine.ok and coarse.ok
    assert coarse.resolution == pytest.approx(2 * fine.resolution, rel=0.12)


def test_denser_sampling_resolves_better_than_the_blur_envelope_alone():
    # 4.5 sigma is where the blur alone crosses 1/7, and it is not a floor:
    # averaging many localizations per molecule beats it
    sparse = frc_resolution(*picture(sigma=8.0, repeats=5), repeats=3)
    dense = frc_resolution(*picture(sigma=8.0, repeats=50), repeats=3)
    assert dense.resolution < sparse.resolution
    assert dense.resolution < envelope_resolution(8.0)
    # and the prediction of the module docstring, S phi = N / 6, holds
    predicted = 2 * np.pi * 8.0 / np.sqrt(np.log(6 * 50))
    assert dense.resolution == pytest.approx(predicted, rel=0.15)


def test_drift_costs_resolution_that_the_precision_does_not_explain():
    # 100 nm of drift over the acquisition, which no localization precision
    # knows about: both halves are smeared by it, so the structure they agree
    # about is the smeared one
    still = frc_resolution(*picture(sigma=8.0), repeats=3)
    moving = frc_resolution(*picture(sigma=8.0, drift=0.05), repeats=3)
    assert moving.resolution > still.resolution * 1.5


def blinking_picture(sigma=8.0, blink_error=0.0, n_emitters=20000, blinks=8,
                     on_time=3, extent=10000.0, n_frames=2000, seed=1):
    """As `picture`, but the localizations come in blinks of consecutive frames.

    ``blink_error`` is one displacement per blink, shared by all its frames:
    the background, the neighbour or the PSF residual that a molecule sees for
    as long as it is on.  It is a real error of that molecule's position, and
    telling whether the FRC counts it is the point of the split.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, extent, (n_emitters, 2))
    emitter = np.repeat(np.arange(n_emitters), blinks)
    start = rng.integers(0, n_frames - on_time, len(emitter))
    which = np.repeat(emitter, on_time)
    frame = np.repeat(start, on_time) + np.tile(np.arange(on_time), len(emitter))
    xy = truth[which] + rng.normal(0, sigma, (len(which), 2))
    if blink_error:
        xy = xy + np.repeat(rng.normal(0, blink_error, (len(emitter), 2)),
                            on_time, axis=0)
    return xy[:, 0], xy[:, 1], frame


def test_independent_errors_do_not_care_how_the_split_is_made():
    """The usual justification, tested: on its own it does not hold."""
    x, y, frame = blinking_picture(sigma=8.0)
    toss = np.random.default_rng(0).integers(0, 2, len(x))
    by_time = frc_resolution(x, y, frame, pixelsize=5.0, repeats=1)
    by_localization = frc_resolution(x, y, toss, n_blocks=2, pixelsize=5.0,
                                     repeats=1)
    assert by_localization.resolution == pytest.approx(by_time.resolution,
                                                       rel=0.08)


def test_an_error_a_blink_shares_is_what_the_time_split_protects_against():
    # 15 nm of error per blink -- one background, one neighbour, one PSF
    # residual for as long as the molecule is on.  Split by time it is seen for
    # what it is; split by localization both halves inherit it and agree
    x, y, frame = blinking_picture(sigma=8.0, blink_error=15.0)
    toss = np.random.default_rng(0).integers(0, 2, len(x))
    by_time = frc_resolution(x, y, frame, pixelsize=5.0, repeats=1)
    by_localization = frc_resolution(x, y, toss, n_blocks=2, pixelsize=5.0,
                                     repeats=1)
    assert by_localization.resolution < 0.8 * by_time.resolution


def test_the_blocks_keep_a_blink_on_one_side_of_the_split():
    frame = np.repeat(np.arange(100), 3)
    side = time_blocks(frame, n_blocks=10, assignment="alternating")
    # a frame is never split between the halves
    for value in np.unique(frame):
        assert len(np.unique(side[frame == value])) == 1
    assert 0.4 < side.mean() < 0.6


def test_random_and_alternating_deal_the_same_blocks_differently():
    frame = np.arange(1000)
    alternating = time_blocks(frame, 10, "alternating")
    random = time_blocks(frame, 10, "random", seed=3)
    assert not np.array_equal(alternating, random)
    assert set(np.unique(alternating)) == {False, True}


def test_the_spread_over_splits_is_the_error_bar():
    out = frc_resolution(*picture(sigma=8.0), repeats=5)
    assert out.repeats == 5 and len(out.per_repeat) == 5
    assert out.error == pytest.approx(np.std(out.per_repeat, ddof=1))
    # the analytic one is kept, and is the smaller of the two
    assert out.analytic_error < out.error


def test_a_curve_that_never_falls_through_the_threshold_says_so():
    curve = np.ones(64)
    counts = np.arange(1, 65) * 8
    out = resolution_from_curve(curve, counts, 128, 5.0)
    assert not out.ok and "1/7" in out.message


def test_too_few_localizations_is_a_message_and_not_a_number():
    out = frc_resolution(np.zeros(10), np.zeros(10), np.arange(10))
    assert not out.ok and "too few" in out.message


def test_the_taper_leaves_the_middle_alone_and_closes_the_edge():
    image = np.ones((64, 64))
    tapered = taper(image)
    assert tapered[32, 32] == pytest.approx(1.0)
    assert tapered[0, 32] == pytest.approx(0.0, abs=1e-9)


def test_two_unrelated_pictures_do_not_correlate():
    rng = np.random.default_rng(2)
    first = rng.poisson(1.0, (256, 256)).astype(float)
    second = rng.poisson(1.0, (256, 256)).astype(float)
    curve, counts = frc_curve(first, second)
    # the lowest rings are the mean and the taper, which both images share
    assert np.abs(curve[20:]).max() < 0.25
    assert counts.sum() == pytest.approx(256 * 256, rel=0.25)


def test_the_blur_envelope_crosses_the_threshold_where_it_should():
    sigma = 10.0
    q = 1.0 / envelope_resolution(sigma)
    assert blur_envelope(q, sigma) == pytest.approx(1 / 7)


# ------------------------------------------------ the planes, for 3D anisotropy

def volume_picture(sigma=8.0, sigma_z=None, n_emitters=20000, repeats=20,
                   extent=4000.0, depth=600.0, n_frames=2000, seed=1):
    """`picture` with a third dimension, and its own error along it."""
    rng = np.random.default_rng(seed)
    truth = np.column_stack([rng.uniform(0, extent, n_emitters),
                             rng.uniform(0, extent, n_emitters),
                             rng.uniform(0, depth, n_emitters)])
    which = np.repeat(np.arange(n_emitters), repeats)
    frame = rng.integers(0, n_frames, len(which))
    spread = np.array([sigma, sigma, sigma_z if sigma_z else sigma])
    xyz = truth[which] + rng.normal(0, 1, (len(which), 3)) * spread
    return xyz[:, 0], xyz[:, 1], xyz[:, 2], frame


def test_the_planes_find_the_same_resolution_along_every_axis_when_it_is_isotropic():
    out = fpc_resolution(*volume_picture(sigma=8.0), repeats=2)
    assert out.ok
    resolutions = [out.axes[name].resolution for name in "xyz"]
    assert max(resolutions) < 1.25 * min(resolutions)
    assert out.anisotropy == pytest.approx(1.0, abs=0.25)


def test_the_planes_measure_the_axial_resolution_separately_from_the_lateral():
    # the axial error is three times the lateral, and that is what comes back:
    # a shell would have averaged the two into a number describing neither
    out = fpc_resolution(*volume_picture(sigma=8.0, sigma_z=24.0), repeats=2)
    assert out.anisotropy == pytest.approx(3.0, rel=0.25)
    assert out.axes["x"].resolution == pytest.approx(out.axes["y"].resolution,
                                                     rel=0.1)
    assert out.axes["z"].resolution > 2 * out.axes["x"].resolution


def test_the_axial_sampling_follows_the_axial_resolution_and_not_the_lateral():
    # left to 2.5 x the lateral voxel, a sharply resolved z would be measured
    # at its own sampling; the coarse pass is there to stop that
    out = fpc_resolution(*volume_picture(sigma=8.0), repeats=1)
    assert out.axes["z"].resolution > 3 * out.voxel[2]
    assert not out.axes["z"].message


def test_a_volume_with_no_depth_says_so_rather_than_dividing_by_zero():
    x, y, z, frame = volume_picture()
    out = fpc_resolution(x, y, np.zeros_like(z), frame, repeats=1)
    assert not out.ok and "extent" in out.message


def test_the_per_axis_resolution_does_not_depend_on_how_finely_it_was_sampled():
    """The regression that matters: a plane must sum over a band, not a grid.

    Summing a plane over the whole grid includes every transverse frequency,
    most of which hold only noise once the sampling is fine -- and the answer
    then follows the voxel instead of the data.  Untreated, this same dataset
    read 89 nm axially on a coarse lateral voxel and 417 nm on a fine one.
    """
    x, y, z, frame = volume_picture(sigma=8.0, sigma_z=24.0)
    fine = fpc_resolution(x, y, z, frame, pixelsize=4.0, z_pixelsize=10.0,
                          repeats=1)
    coarse = fpc_resolution(x, y, z, frame, pixelsize=10.0, z_pixelsize=25.0,
                            repeats=1)
    for name in "xyz":
        assert (fine.axes[name].resolution
                == pytest.approx(coarse.axes[name].resolution, rel=0.15))


def test_the_band_a_plane_sums_over_barely_changes_the_answer():
    # the band is set from the resolution already measured along the other
    # axes; widening it by half again must not move the number much, or the
    # choice of band would be doing the measuring
    x, y, z, frame = volume_picture(sigma=8.0, sigma_z=24.0)
    from smappy.frc import band_masks, plane_sums, _volume, _fold
    from smappy.frc import resolution_from_curve

    shape, voxel = (256, 256, 48), (6.0, 6.0, 15.0)
    low = np.array([x.min(), y.min(), z.min()])
    side = frame % 2 == 0
    halves = [_volume(x[m], y[m], z[m], low, voxel, shape) for m in (~side, side)]
    found = {}
    for width in (1.0, 1.5):
        masks = band_masks(shape, voxel, 30.0 * width, 90.0 * width)
        sums = plane_sums(halves[0], halves[1], -1, masks)
        num, pa, pb = sums["z"]
        curve = _fold(np.divide(num, np.sqrt(pa * pb), out=np.zeros(len(num)),
                                where=pa * pb > 0))
        counts = np.full(len(curve), float(masks["z"].sum()))
        found[width] = resolution_from_curve(curve, counts, shape[2],
                                             voxel[2]).resolution
    assert found[1.5] == pytest.approx(found[1.0], rel=0.1)


def test_tiles_give_the_same_answer_as_one_transform():
    x, y, frame = picture(sigma=8.0, extent=6000.0)
    whole = frc_resolution(x, y, frame, pixelsize=6.0, tile_pixels=4096, repeats=1)
    tiled = frc_resolution(x, y, frame, pixelsize=6.0, tile_pixels=256, repeats=1)
    assert tiled.tiles > 4 and whole.tiles == 1
    assert tiled.resolution == pytest.approx(whole.resolution, rel=0.03)


def test_empty_tiles_are_skipped_rather_than_transformed():
    # localizations in one corner of a wide field: the rest is empty space and
    # must not cost a transform each
    x, y, frame = picture(sigma=8.0, extent=2000.0)
    wide = frc_resolution(np.append(x, 20000.0), np.append(y, 20000.0),
                          np.append(frame, 0), pixelsize=6.0, tile_pixels=256,
                          repeats=1)
    assert wide.tiles <= 9                      # not the 170 the grid holds
    assert wide.ok
