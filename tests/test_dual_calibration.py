"""Scientific invariants for split-frame geometry, projective fitting and joint PSFs."""
from dataclasses import replace
import numpy as np
import pytest

from smappy.calibrate.dual import (DualColorSettings, LAYOUTS, map_points, fit_projective,
    robust_projective, calibrate_dual, build_dual_calibration, load_dual_color_calibration,
    fit_dual_transform, collect_dual_beads)
from smappy.calibrate.input import BeadStack
from smappy.calibrate.core import positive_pair_models, estimate_pair_shift
from smappy.io.calibration import load_spline_calibration, evaluate_spline


def test_projective_recovery_outliers_and_degeneracy():
    rng = np.random.default_rng(41)
    p = rng.uniform(20, 400, (40, 2))
    truth = np.array([[1.02, .03, 12], [-.02, .98, -7], [.0001, -.0002, 1]])
    q = map_points(truth, p)
    q[-5:] += 30
    h, good, errors = robust_projective(p, q, .2, 8)
    assert good.sum() == 35
    np.testing.assert_allclose(map_points(h, p[:-5]), q[:-5], atol=1e-8)
    with pytest.raises(ValueError, match='collinear'):
        fit_projective(np.c_[np.arange(8), np.arange(8)], np.c_[np.arange(8), np.arange(8)])
    with pytest.raises(ValueError):
        DualColorSettings(min_pairs=3).validate()


def test_two_round_componentwise_screen_and_geometric_weights():
    rng = np.random.default_rng(713)
    source = rng.uniform(0, 400, (103, 2))
    truth = np.array([[1.01, .02, 17], [-.01, -.98, 430], [.0001, -.0001, 1.]])
    target = map_points(truth, source)+rng.normal(0, .005, source.shape)
    target[-3:] += [[.4, 0], [0, -.4], [.13, .13]]
    s = DualColorSettings()
    fitted = fit_dual_transform(source, target, s)
    assert fitted.round1_inliers.all()  # inside the coarse 2-pixel radius
    assert not fitted.accepted[-3:-1].any()
    assert fitted.accepted[-1]  # inside each axis limit, outside a .15-pixel circle
    np.testing.assert_array_equal(fitted.accepted, np.all(np.abs(fitted.round1_dxdy) <= .15, axis=1))
    assert np.linalg.norm(fitted.round1_dxdy[-1]) > .15
    assert (fitted.weights_xy[-3:-1] == 0).all()
    assert (fitted.weights_xy[-1] < .8).all()
    assert np.median(np.linalg.norm(fitted.dxdy[:-3], axis=1)) < .01
    again = fit_dual_transform(source, target, s)
    np.testing.assert_array_equal(again.transformation, fitted.transformation)
    # Coordinate-origin changes must not change predicted physical positions.
    origin = np.array([12000., 8000.])
    translated = fit_dual_transform(source+origin, target+origin, s)
    np.testing.assert_allclose(translated.dxdy, fitted.dxdy, atol=1e-6)
    with pytest.raises(ValueError, match='dx/dy limit'):
        fit_dual_transform(source, target, replace(s, transform_axis_limit_px=1e-7))
    with pytest.raises(ValueError, match='dx/dy limit'):
        replace(s, transform_axis_limit_px=np.nan).validate()


def test_shape_outliers_do_not_refit_or_reduce_transformation_support(tmp_path):
    from scipy.ndimage import gaussian_filter
    stack, settings = synthetic('up-down', 'upper')
    beads = collect_dual_beads([stack], replace(settings, rejection_mad=1.5))
    baseline = build_dual_calibration(beads)
    # Alter shapes without changing their measured lateral coordinates.
    for channel in beads.channels:
        for i in (0, 8):
            channel.volumes[i] = gaussian_filter(channel.volumes[i], (0, 3, 0))
    result = build_dual_calibration(beads)
    assert result.transform_accepted.all()
    assert not result.accepted[[0, 8]].any()
    assert result.accepted.sum() < settings.min_pairs  # PSF minimum is independent
    np.testing.assert_array_equal(result.calibration.transformation, baseline.calibration.transformation)
    coverage = result.calibration.parameters['coverage_area_px2']
    assert coverage['eligible'] == coverage['transformation']
    assert coverage['psf'] < coverage['transformation']
    rebuilt = build_dual_calibration(beads, excluded=[0])
    assert not rebuilt.transform_accepted[0] and not rebuilt.accepted[0]
    assert rebuilt.reasons[0] == 'manual exclusion'
    path = tmp_path/'separate_selections.h5'
    result.save(path)
    import h5py
    with h5py.File(path) as f:
        d = f['diagnostics']
        np.testing.assert_array_equal(d['transform_accepted'], result.transform_accepted)
        np.testing.assert_array_equal(d['accepted'], result.accepted)
        np.testing.assert_allclose(d['round1_dxdy_px'], result.transform_fit.round1_dxdy)
        np.testing.assert_allclose(d['transform_weights_xy'], result.transform_fit.weights_xy)


def test_the_pair_is_clipped_at_zero_and_normalised_together():
    """Clipped rather than lifted: a constant added to every pixel was the
    worst noise dip of the volume, 11-40% of the signal, and the fit's
    background read low by it."""
    z, y, x = np.mgrid[-4:5, -4:5, -4:5]
    raw = np.exp(-x*x-y*y-z*z/4)-.02
    settings = DualColorSettings(roi_size=9, smooth_z_nm=0)
    models, _ = positive_pair_models(np.stack((raw, raw*3)), 20, settings)
    assert models[0].psf.min() == 0
    np.testing.assert_allclose(models[1].psf, models[0].psf*3, rtol=2e-5, atol=1e-8)
    # the tails are zero, not lifted, and what dips below between knots is
    # recorded -- a small fraction of the peak
    assert 'positivity_offset_before_normalization' not in models[0].parameters
    assert -0.05 < models[0].parameters['spline_minimum'] <= 0
    lowest = min(evaluate_spline(models[0], 2.3, 1.8, zpos, 5).min()
                 for zpos in np.linspace(.05, 7.95, 41))
    assert lowest >= models[0].parameters['spline_minimum']*models[0].psf.max() - 1e-7


def synthetic(layout, main):
    horizontal = 'right-left' in layout
    mirrored = 'mirrored' in layout
    shape = (220, 440) if horizontal else (440, 220)
    images = np.full((41, *shape), 50., dtype=np.float32)
    z, y, x = np.mgrid[:41, -11:12, -11:12]
    for i, (cy, cx) in enumerate(( (y, x) for y in (42, 108, 175) for x in (42, 108, 175))):
        height = (i%3-1)*1.5
        for ch in range(2):
            sigma_x = 1.4+.002*(z-20-height-3*ch)**2
            sigma_y = 1.6+.001*(z-20-height+2-3*ch)**2
            volume = (1+ch)*1200*np.exp(-.5*((x/sigma_x)**2+(y/sigma_y)**2)) / (sigma_x*sigma_y)
            px, py = cx, cy
            if ch:
                if horizontal:
                    px = 439-cx if mirrored else cx+220
                else:
                    py = 439-cy if mirrored else cy+220
            images[:, py-11:py+12, px-11:px+12] += volume.astype(np.float32)
    s = DualColorSettings(layout=layout, main_channel=main, roi_size=15, padding=4,
            max_xy_shift_px=2, max_z_shift_nm=100, smooth_z_nm=0,
            rejection_mad=10, registration_iterations=1, brightness_range=None)
    return BeadStack(images, np.arange(41)*20., roi=(91, 0, shape[1], shape[0])), s


@pytest.mark.parametrize('layout', LAYOUTS)
def test_all_layouts_shared_z_ratio_rebuild_and_roundtrip(layout, tmp_path):
    main = 'right' if 'right-left' in layout else 'lower'
    stack, settings = synthetic(layout, main)
    result = calibrate_dual([stack], settings)
    assert len(result.beads.records) == 9
    assert result.accepted.sum() == 9
    assert result.transform_accepted.sum() == 9
    assert result.calibration.geometry['split_position'] == 220
    assert result.calibration.main.z0 == result.calibration.secondary.z0
    assert result.calibration.main.shape == result.calibration.secondary.shape
    np.testing.assert_array_equal(result.registration.beads.channel_offsets[:, :, 0], 0)
    from smappy.calibrate.dual import channel_result
    np.testing.assert_array_equal(channel_result(result, 0).shifts[:, 0],
                                  channel_result(result, 1).shifts[:, 0])
    assert np.max(result.transform_residuals) < .05
    # The secondary's intrinsic axial displacement must survive joint alignment.
    def width_min(psf):
        xx = np.arange(psf.shape[-1])-(psf.shape[-1]-1)/2
        return np.argmin((psf*xx[None, None, :]**2).sum((1, 2))/psf.sum((1, 2)))
    assert width_min(result.calibration.main.psf)-width_min(result.calibration.secondary.psf) >= 2
    rebuilt = build_dual_calibration(result.beads, excluded=[0])
    assert rebuilt.accepted.sum() == 8
    assert rebuilt.transform_accepted.sum() == 8
    assert rebuilt.reasons[0] == 'manual exclusion'
    np.testing.assert_array_equal(rebuilt.initial_transformation, result.initial_transformation)
    path = tmp_path/'dual.h5'
    rebuilt.save(path)
    loaded = load_dual_color_calibration(path)
    np.testing.assert_array_equal(loaded.main.coeff, rebuilt.calibration.main.coeff)
    np.testing.assert_allclose(loaded.transform(result.beads.secondary_points),
                               result.beads.main_points, atol=.05)
    loaded.validate_image(stack.images.shape, stack.roi)
    with pytest.raises(ValueError, match='load_dual_color'):
        load_spline_calibration(path)
    with pytest.raises(FileExistsError):
        rebuilt.save(path)


def test_roi_fallback_warning_shape_error_and_unequal_split():
    stack, settings = synthetic('up-down', 'upper')
    stack = replace(stack, roi=None)
    settings = replace(settings, split_position=219)
    with pytest.warns(UserWarning, match='Missing camera ROI'):
        result = calibrate_dual([stack], settings)
    assert result.calibration.geometry['coordinate_system'] == 'roi-local'
    with pytest.warns(UserWarning, match='cannot verify'):
        result.calibration.validate_image(stack.images.shape)
    with pytest.raises(ValueError, match='dimensions'):
        result.calibration.validate_image((440, 221))


def test_pair_registration_recovers_one_shared_shift():
    from scipy.ndimage import shift
    z, y, x = np.mgrid[-15:16, -7:8, -7:8]
    a = np.exp(-((z/5)**2+(x/2)**2+(y/2)**2))
    pair = np.stack((a, 3*np.exp(-(((z-4)/6)**2+(x/2)**2+(y/2)**2))))
    moving = np.stack([shift(p, (2.2, .35, -.4), order=3) for p in pair])
    delta = estimate_pair_shift(pair, moving, limits=(5, 2, 2), lateral_window=13)
    np.testing.assert_allclose(delta, [-2.2, -.35, .4], atol=.15)


def test_saturation_rejects_whole_pair_and_acquisitions_cannot_cross_match():
    stack, settings = synthetic('up-down', 'upper')
    # Saturate one secondary bead, leaving exactly eight good pairs in this acquisition.
    stack.images[20, 262, 42] = 4095
    second = replace(stack, images=stack.images.copy(), source='second acquisition', axes={'position': 1})
    result = calibrate_dual([stack, second], replace(settings, saturation_adu=4095))
    assert len(result.beads.records) == 18
    assert result.accepted.sum() == 16
    assert result.transform_accepted.sum() == 16
    for rec in result.beads.records:
        assert rec['channel_records'][0]['stack'] == rec['channel_records'][1]['stack']
    assert sum('saturated' in reason for reason in result.reasons) == 2


def test_a_tiny_axis_limit_is_reported_as_such_not_as_an_optimizer_failure():
    """The screen's own message must survive a pathological limit.

    Half the axis limit is the soft-L1 scale, so a minuscule limit used to make
    the round-one refinement diverge and hide why nothing was accepted.
    """
    rng = np.random.default_rng(5)
    source = rng.uniform(0, 400, (60, 2))
    truth = np.array([[1.0, .01, 5], [-.01, 1.0, -3], [0, 0, 1.]])
    target = map_points(truth, source) + rng.normal(0, .004, source.shape)
    settings = DualColorSettings()
    for limit in (1e-7, 1e-5, 1e-4):
        with pytest.raises(ValueError, match='dx/dy limit'):
            fit_dual_transform(source, target, replace(settings,
                                                       transform_axis_limit_px=limit))
    # the floor does not touch a normal limit: the fit is what it was
    reference = fit_dual_transform(source, target, settings)
    assert reference.accepted.sum() >= settings.min_pairs
    from smappy.calibrate.dual import MIN_LOSS_SCALE_PX
    assert MIN_LOSS_SCALE_PX < settings.transform_axis_limit_px / 2


def test_the_two_channel_fitter_refits_the_calibration_beads():
    """Fit quality for 2C goes through the workflow the data will meet.

    One z per bead *pair*, from the global fit, not one per channel -- and the
    single-channel entry point says so instead of failing on an attribute.
    """
    from smappy.calibrate.validation import (fit_bead_diagnostics,
                                             fit_paired_bead_diagnostics)
    stack, settings = synthetic('up-down mirrored', 'upper')
    result = calibrate_dual([stack], settings)

    fitted = fit_paired_bead_diagnostics(result)
    beads = fitted['bead_id'] >= 0
    assert set(np.unique(fitted['bead_id'])) == set(range(9)) | {-1}
    good = fitted['fit_valid'] & beads
    assert good.sum() > .8*beads.sum()
    # a shared z is the point of the global fit: it must land on the plane
    assert np.nanstd(fitted['centered_error_nm'][good]) < 20
    # channel 1 is twice as bright, and the ratio is the photons' true
    # fraction: the secondary spline's share of the beads is read back as
    # photons (`GlobalSplinePSF.photon_scales`), not left in bead units
    assert abs(np.nanmedian(fitted['ratio'][good])-2/3) < .07

    with pytest.raises(TypeError, match='fit_paired_bead_diagnostics'):
        fit_bead_diagnostics(result)


@pytest.fixture(scope="module")
def dim_secondary():
    """A dual calibration from beads that send a fifth of their light to the
    secondary half, as `smappy.simulate` draws them."""
    import warnings
    from smappy.calibrate.dual import DualColorSettings
    from smappy.calibrate.input import BeadStack
    from smappy.simulate import dual_bead_stacks
    stacks, z = dual_bead_stacks(1, seed=0, dz_nm=50.0, z_range_nm=(-700.0, 700.0),
                                 secondary_share=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # ROI-local: no camera ROI
        return calibrate_dual([BeadStack(stacks[0].astype(np.float32), z)],
                              DualColorSettings(layout="up-down", main_channel="upper",
                                                dz_nm=50.0, roi_size=17,
                                                smooth_z_nm=40.0)).calibration


def _molecules(share, n=400, photons=2000.0, background=10.0, seed=1):
    """ROI pairs of one astigmatic molecule each, cut on the spot, drawn from
    the PSF the beads were drawn with, a fraction ``share`` in the secondary."""
    from scipy.special import erf
    from smappy.simulate import ASTIGMATISM, astigmatic_sigmas
    rng = np.random.default_rng(seed)
    z = rng.choice([-200.0, 0.0, 200.0], n)
    x0, y0 = 6 + rng.uniform(-0.5, 0.5, (2, n))
    k = np.arange(13)[None]

    def psf(sigma_nm):
        sx, sy = (w / 100.0 * np.sqrt(2.0)
                  for w in astigmatic_sigmas(z, sigma_nm, *ASTIGMATISM))
        ex = .5*(erf((k+.5-x0[:, None])/sx[:, None]) - erf((k-.5-x0[:, None])/sx[:, None]))
        ey = .5*(erf((k+.5-y0[:, None])/sy[:, None]) - erf((k-.5-y0[:, None])/sy[:, None]))
        return ey[:, :, None]*ex[:, None, :]
    return np.stack([rng.poisson(background + photons*part*psf(sigma))
                     for part, sigma in ((1-share, 130.0), (share, 145.0))],
                    axis=1).astype(np.float32)


def test_the_ratio_and_the_photons_are_the_dyes_not_the_beads(dim_secondary):
    """The secondary spline carries the beads' split, and the fit read its
    photons in bead units: a dye with a quarter of its light in the secondary
    came out at 0.57 and the total 3.4x the truth."""
    from smappy.psf import GlobalSplinePSF
    cal = dim_secondary
    model = GlobalSplinePSF((cal.main, cal.secondary))
    link = np.zeros((400, 2, 2, 5), np.float32)
    link[:, 1] = 1.0
    values = model.unpack(model.fit(_molecules(0.25), link))
    assert np.median(values['ratio']) == pytest.approx(0.25, abs=0.015)
    assert np.median(values['photons']) == pytest.approx(2000, rel=0.05)
    assert np.median(values['photons_ch1']) == pytest.approx(500, rel=0.08)


def test_linked_photons_are_the_total_with_the_split_from_the_beads(dim_secondary):
    """Biplane: one photon number, the split taken from the beads.  The PSFs
    are normalised together, so that number is the emitter's total -- and
    `build_link` once multiplied the secondary by the beads' ratio on top of
    a spline that already carried it."""
    from smappy.dualfit import build_link
    from smappy.psf import GlobalSplinePSF
    cal = dim_secondary
    model = GlobalSplinePSF((cal.main, cal.secondary), (True,)*5)
    class spot:                          # where the candidates were: any
        x = y = np.full(400, 40)

        def __len__(self):
            return 400
    spot = spot()
    link = build_link(spot, spot, np.zeros((400, 2, 2)), cal, roisize=13)
    link[:, 0] = 0.0                     # the ROIs are cut on the spot
    link[:, 1, :, :2] = 1.0
    assert np.allclose(link[:, 1, 1, 2], 1.0)       # the split is in the PSFs
    values = model.unpack(model.fit(_molecules(0.2), link), link)
    assert np.median(values['photons']) == pytest.approx(2000, rel=0.05)
    assert np.all(values['ratio'] == 0)


def test_the_two_psfs_are_normalised_together_and_keep_their_shares(dim_secondary):
    """One factor for both, their signal at focus summing to one, and each
    channel's share stored with its PSF for a fit with a photon number each."""
    from smappy.psf import signal_integral
    cal = dim_secondary
    shares = [signal_integral(c) for c in (cal.main, cal.secondary)]
    assert sum(shares) == pytest.approx(1.0, abs=1e-6)
    assert shares[1] == pytest.approx(0.2, abs=0.02)     # the beads' split
    for c, share in zip((cal.main, cal.secondary), shares):
        centre = (c.psf.shape[0]-1)//2
        assert c.psf[centre].sum() == pytest.approx(share, rel=0.03)


def test_the_shares_survive_a_save(dim_secondary, tmp_path):
    from smappy.calibrate.dual import (load_dual_color_calibration,
                                       save_dual_color_calibration)
    from smappy.psf import signal_integral
    path = tmp_path / "dual.h5"
    save_dual_color_calibration(path, dim_secondary)
    loaded = load_dual_color_calibration(path)
    for before, after in ((dim_secondary.main, loaded.main),
                          (dim_secondary.secondary, loaded.secondary)):
        assert signal_integral(after) == pytest.approx(signal_integral(before))


def test_the_fitted_background_is_the_background(dim_secondary):
    """The PSFs were lifted by the worst noise dip of the volume, which the
    background absorbed: it read low by the photons times that constant, and
    at a photon per pixel every fit of a bright molecule sat on the floor."""
    from smappy.psf import GlobalSplinePSF
    cal = dim_secondary
    model = GlobalSplinePSF((cal.main, cal.secondary))
    link = np.zeros((400, 2, 2, 5), np.float32)
    link[:, 1] = 1.0
    fit = model.fit(_molecules(0.25, photons=10000, background=1.0), link)
    values = model.unpack(fit, link)
    assert np.isfinite(fit.logl).all()
    for channel in (0, 1):
        background = values[f'background_ch{channel}']
        assert np.median(background) == pytest.approx(1.0, abs=0.15)
        assert np.mean(background < 0.02) < 0.05          # not on the floor
    assert np.median(values['photons']) == pytest.approx(10000, rel=0.03)
