"""The Gaussian PSF as a spline calibration (`calibrate.gaussian_psf`): what DECODE trains on without beads."""
import numpy as np
import pytest

from smappy.calibrate import gaussian_psf as gp
from smappy.io.calibration import (
    evaluate_spline,
    load_spline_calibration,
    save_spline_calibration,
)


def test_the_spline_draws_the_pixel_integrated_gaussian_at_any_position_and_width():
    # exact on the pixels' centres; between them a cubic on the pixel grid follows a Gaussian 1.2 pixels
    # wide to 1 % of its peak, the spline format's limit, a bead spline's too
    cal = gp.gaussian_calibration((100, 180), (100, 110))
    rec = gp.record(cal)
    size = 15
    for sigma, dx, dy in ((100.0, 0.0, 0.0), (137.0, 0.3, -0.21), (163.5, -0.45, 0.4), (180.0, 0.11, 0.37), (120.0, 0.5, 0.5)):
        z_index = cal.z_nm_to_index(gp.z_of(sigma, rec))
        x, y = 7 + dx, 7 + dy
        spot = evaluate_spline(cal, x, y, float(z_index), size)
        ox = np.arange(size) - x
        oy = np.arange(size) - y
        want = (gp._pixel_profile(oy, [sigma / 110])[0][:, None] * gp._pixel_profile(ox, [sigma / 100])[0][None])
        assert np.abs(spot - want).max() < (1e-6 if dx == dy == 0 else 1.2e-2) * want.max()
        assert spot.sum() == pytest.approx(1.0, abs=2e-3)


def test_the_planes_run_from_the_widest_and_z_is_the_width_less_the_middle():
    cal = gp.gaussian_calibration((120, 160), 100, step_nm=5)
    rec = gp.record(cal)
    assert rec["sigma_ref_nm"] == 140
    # plane k's width is sigma_ref + its z: the first plane the widest, the margin beyond each end
    z = cal.z_index_to_nm(np.arange(cal.psf.shape[0]))
    widths = gp.width_of(z, rec)
    assert widths[0] == pytest.approx(160 + gp.MARGIN * 5) and widths[-1] == pytest.approx(120 - gp.MARGIN * 5)
    # the drawn planes are those widths: the second moment of each plane's x profile
    profile = cal.psf.sum(axis=1)
    centre = (cal.psf.shape[2] - 1) / 2
    var = (profile * (np.arange(cal.psf.shape[2]) - centre) ** 2).sum(axis=1) - 1 / 12
    assert np.sqrt(var) * 100 == pytest.approx(widths, rel=2e-3)


def test_a_fixed_width_still_makes_a_spline_and_a_too_narrow_range_is_refused():
    cal = gp.gaussian_calibration((130, 130), 100)
    assert cal.psf.shape[0] == 2 * gp.MARGIN + 1
    with pytest.raises(ValueError, match="below"):
        gp.gaussian_calibration((35, 60), 100)


def test_the_record_survives_the_file(tmp_path):
    cal = gp.gaussian_calibration((110, 170), 107)
    save_spline_calibration(tmp_path / "g.h5", cal)
    back = load_spline_calibration(tmp_path / "g.h5")
    assert gp.record(back) == gp.record(cal)
    assert gp.record_of_file(tmp_path / "g.h5")["sigma_nm"] == [110, 170]
    assert np.array_equal(back.coeff, cal.coeff)


def test_gaussian_2d_finds_the_width_and_photons_the_spline_was_drawn_with():
    from smappy.psf import GaussianPSF
    cal = gp.gaussian_calibration((100, 200), 100)
    rec = gp.record(cal)
    rng = np.random.default_rng(1)
    n, size = 400, 13
    sigma = rng.uniform(110, 190, n)
    pos = 6 + rng.uniform(-0.5, 0.5, (n, 2))
    z_index = cal.z_nm_to_index(gp.z_of(sigma, rec))
    model = np.stack([evaluate_spline(cal, x, y, float(z), size) for (x, y), z in zip(pos, z_index)])
    rois = rng.poisson(20000 * np.clip(model, 0, None) + 5).astype(np.float32)
    out = GaussianPSF(sigma=1.2).unpack(GaussianPSF(sigma=1.2).fit(rois))
    assert np.median(np.abs(out["sigma_pix"] * 100 - sigma)) < 1.0
    assert np.median(out["photons"] / 20000) == pytest.approx(1.0, abs=0.01)
    assert np.median(np.abs(out["x_roi"] - pos[:, 0])) < 0.01


def test_as_widths_turns_a_gaussian_models_z_into_sigma_and_leaves_a_psf_models_alone():
    rec = {"sigma_ref_nm": 140.0, "sigma_nm": [110, 170]}
    cols = {"x_pix": np.ones(2), "z_nm": np.array([-10.0, 20.0]), "z_err_nm": np.array([3.0, 4.0])}
    out = gp.as_widths(cols, rec)
    assert "z_nm" not in out and "z_err_nm" not in out
    assert np.array_equal(out["sigma_nm"], [130.0, 160.0]) and np.array_equal(out["sigma_err_nm"], [3.0, 4.0])
    assert gp.as_widths(cols, None) is cols


def test_the_width_range_covers_the_fits_and_is_never_too_narrow():
    rng = np.random.default_rng(0)
    s = rng.uniform(120, 160, 5000)
    lo, hi = gp.sigma_range(s)
    assert lo < 120 and hi > 160 and hi - lo == pytest.approx(40 * 1.3, rel=0.05)
    lo, hi = gp.sigma_range(np.full(100, 140.0))
    assert (lo, hi) == pytest.approx((140 * 0.85, 140 * 1.15))
    with pytest.raises(ValueError, match="type it"):
        gp.sigma_range([np.nan])
