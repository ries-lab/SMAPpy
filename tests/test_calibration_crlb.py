"""The CRLB a calibration allows against z (`calibrate.crlb`), and the page that draws it."""
import numpy as np
import pytest

from smappy.calibrate.core import spline_coefficients
from smappy.calibrate.crlb import dual_crlb, spline_crlb, z_range_nm
from smappy.io.calibration import SplineCalibration, evaluate_spline
from smappy.psf import SplinePSF

NZ, BOX = 41, 17


def astigmatic(ellipticity=0.4) -> SplineCalibration:
    z = np.linspace(-1, 1, NZ)[:, None, None]
    y, x = np.mgrid[:BOX, :BOX] - (BOX - 1) / 2
    sx = 1.3 + 0.8 * (z - ellipticity) ** 2
    sy = 1.3 + 0.8 * (z + ellipticity) ** 2
    psf = np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)
    return SplineCalibration(spline_coefficients(psf.astype(np.float32)), 50.0, NZ // 2, x0=(BOX - 1) / 2)


@pytest.mark.parametrize("z_nm", [-600.0, 0.0, 300.0])
def test_the_bound_is_the_fitters_crlb_and_the_scatter_of_noisy_fits(z_nm):
    cal = astigmatic()
    bound = spline_crlb(cal, z_nm, photons=2000, background=20, roisize=13)
    mu = 2000 * evaluate_spline(cal, 6, 6, float(cal.z_nm_to_index(z_nm)), 13) + 20
    model = SplinePSF(cal, z_nm)
    exact = model.unpack(model.fit(mu[None].astype(np.float32)))
    for ours, fitter in (("x_px", "x_err_pix"), ("y_px", "y_err_pix"), ("z_nm", "z_err_nm"),
                         ("photons", "photons_err")):
        assert bound[ours][0] == pytest.approx(exact[fitter][0], rel=0.01), ours
    rois = np.random.default_rng(1).poisson(np.repeat(mu[None], 2000, 0)).astype(np.float32)
    fitted = model.unpack(model.fit(rois))
    assert np.std(fitted["x_roi"]) == pytest.approx(bound["x_px"][0], rel=0.1)
    assert np.std(fitted["z_nm"]) == pytest.approx(bound["z_nm"][0], rel=0.1)


def test_the_bound_falls_as_one_over_the_root_of_the_photons_without_background():
    cal = astigmatic()
    few, many = (spline_crlb(cal, 0.0, photons=n, background=0) for n in (500, 2000))
    assert few["z_nm"][0] / many["z_nm"][0] == pytest.approx(2.0, rel=0.01)


def test_astigmatism_trades_x_for_y_on_either_side_of_focus():
    cal = astigmatic()
    below, above = spline_crlb(cal, -300.0), spline_crlb(cal, 300.0)
    assert below["x_px"][0] == pytest.approx(above["y_px"][0], rel=0.02)
    assert below["x_px"][0] < below["y_px"][0]


def test_two_channels_fitted_together_beat_either_alone_with_its_share():
    from dataclasses import replace
    main = replace(astigmatic(0.4), parameters={"photon_normalization": 0.5})
    secondary = replace(astigmatic(-0.4), parameters={"photon_normalization": 0.5})
    main.coeff, secondary.coeff = main.coeff * 0.5, secondary.coeff * 0.5   # the pair sums to one
    z = z_range_nm(main)
    joint = dual_crlb(main, secondary, z)
    half = spline_crlb(main, z, photons=1000)     # one half's share of the 2000
    assert np.all(joint["z_nm"] < half["z_nm"])
    # both halves alike but mirrored in astigmatism: x and y are equally well known at focus
    at = int(np.argmin(np.abs(z)))
    assert joint["x_px"][at] == pytest.approx(joint["y_px"][at], rel=0.01)


def test_the_page_draws_a_built_calibration_with_its_values_at_focus():
    from matplotlib.figure import Figure
    from test_bead_calibration import synthetic_collection

    from smappy.calibrate.core import build_calibration
    from smappy.calibrate.plots import draw_crlb
    result = build_calibration(synthetic_collection())
    figure = Figure()
    line = draw_crlb(figure, result)
    assert len(figure.axes) == 1                      # x, y and z on one axis, all in nm
    assert "100 nm assumed" in figure.axes[0].get_title()   # no camera: the pixel size assumed
    assert line.startswith("CRLB at focus") and "ASSUMED" in line and "nan" not in line
    z = figure.axes[0].lines[2].get_ydata()
    assert np.all(np.isfinite(z)) and np.all(z > 0)
    x100 = figure.axes[0].lines[0].get_ydata()
    draw_crlb(figure, result, pixel_nm=50.0)
    assert "assumed" not in figure.axes[0].get_title()
    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), x100 / 2)
