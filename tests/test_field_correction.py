"""The bias of a fit with one spline over a field-dependent PSF, subtracted afterwards
(`smappy.field_correction`, *Field correction*; docs/uipsf/field-dependence.md §11)."""
import numpy as np
import pytest
from test_field_splines import BOX, NZ, astigmatic, field_of

from smappy.field_correction import corrected, has_field, measure
from smappy.field_splines import blend
from smappy.io.calibration import SplineCalibration, save_spline_calibration
from smappy.locs import Localizations
from smappy.psf import SplinePSF

ROI = 11
XS = YS = np.linspace(0, 1000, 5)


def ellipticity(x, y):
    """Astigmatism growing across the chip in x: a centre spline is wrong at both edges."""
    return 0.1 + 0.3 * x / 1000


def calibration(ell=ellipticity):
    field = field_of(ell, XS, YS)
    return SplineCalibration(blend(field, (500, 500)), 50.0, NZ // 2, x0=(BOX - 1) / 2, field_splines=field)


def centre_fit(cal, x, photons=3000.0, background=10.0):
    """Noise-free spots drawn at chip x (y 300) over the planes 10..30, fitted with the centre spline:
    (chip x, z fitted, z true)."""
    k = np.arange(10, 31)
    s = (BOX - ROI) // 2
    rois = (astigmatic(ellipticity(x, 300.0))[k, s:s + ROI, s:s + ROI] * photons + background).astype(np.float32)
    plain = SplineCalibration(cal.coeff, 50.0, NZ // 2, x0=(BOX - 1) / 2)
    fit = SplinePSF(plain)
    out = fit.unpack(fit.fit(rois, 50, 1))
    return x + out["x_roi"] - (ROI - 1) / 2, out["z_nm"], (NZ // 2 - k) * 50.0


def test_the_correction_takes_the_z_bias_of_the_centre_spline_away_across_the_field():
    cal = calibration()
    correction = measure(cal, 3000.0, 10.0, roi=ROI)
    assert correction.residual_nm < 5 and not correction.failed
    for x in (950.0, 600.0, 130.0):
        chip_x, z, truth = centre_fit(cal, x)
        d, outside = correction(chip_x, np.full(len(z), 300.0), z)
        raw = np.sqrt(np.mean((z - truth) ** 2))
        fixed = np.sqrt(np.mean((z - d[:, 2] - truth) ** 2))
        assert not outside.any()
        assert raw > 20 and fixed < 0.15 * raw, (x, raw, fixed)


def test_a_field_that_does_not_vary_needs_no_correction():
    correction = measure(calibration(lambda x, y: 0.25), 3000.0, 10.0, roi=ROI)
    x, y = np.meshgrid(XS, YS)
    d, _ = correction(x.ravel(), y.ravel(), np.zeros(x.size))
    assert np.abs(d).max() < 1.0


def test_outside_what_was_measured_a_localization_takes_the_edges_value():
    correction = measure(calibration(), 3000.0, 10.0, roi=ROI)
    z_hi = correction.z_range[1]
    d, outside = correction([1000.0, 1500.0, 1000.0], [300.0, 300.0, 300.0], [0.0, 0.0, z_hi + 400])
    edge, _ = correction([1000.0], [300.0], [z_hi])
    assert outside.tolist() == [False, True, True]
    assert np.allclose(d[1], d[0]) and np.allclose(d[2], edge[0])


def test_the_corrected_positions_are_the_fitted_ones_less_the_bias_in_nm():
    correction = measure(calibration(), 3000.0, 10.0, roi=ROI)
    x, y, z = np.array([900.0 * 100]), np.array([300.0 * 100]), np.array([200.0])
    cx, cy, cz, _ = corrected(x, y, z, correction, (100.0, 100.0))
    d, _ = correction([900.0], [300.0], [200.0])
    assert np.allclose([x[0] - cx[0], y[0] - cy[0], z[0] - cz[0]], [d[0, 0] * 100, d[0, 1] * 100, d[0, 2]])


def fitted_table(path, n=200):
    rng = np.random.default_rng(3)
    metadata = {"pixelsize_nm": 100.0, "calibration": {"file": str(path), "fit_box_px": ROI}}
    return Localizations({"x_nm": rng.uniform(0, 100000, n).astype(np.float32),
                          "y_nm": rng.uniform(0, 100000, n).astype(np.float32),
                          "z_nm": rng.uniform(-500, 500, n).astype(np.float32),
                          "photons": np.full(n, 3000.0, np.float32), "background": np.full(n, 10.0, np.float32),
                          "frame": np.arange(n)}, metadata)


def test_the_plugin_corrects_the_table_once_by_the_calibration_it_was_fitted_with(tmp_path):
    from smappy.plugins import Context
    from smappy.plugins.field_correction import CorrectField, FieldCorrectionSettings, can_correct
    path = tmp_path / "field.h5"
    save_spline_calibration(path, calibration())
    locs = fitted_table(path)
    assert has_field(path) and can_correct(locs.metadata)
    result = CorrectField().run(Context(locs=locs), FieldCorrectionSettings())
    out = result.locs
    assert out.metadata["field_correction"]["calibration"] == str(path)
    assert not np.allclose(out["z_nm"], locs["z_nm"])
    with pytest.raises(ValueError, match="already"):
        CorrectField().run(Context(locs=out), FieldCorrectionSettings())


def test_after_the_fit_offers_the_field_correction_only_with_a_field(tmp_path):
    from smappy.plugins.after_fit import AfterFitSettings, finish_localizations, shown, table_can
    path = tmp_path / "field.h5"
    save_spline_calibration(path, calibration())
    locs = fitted_table(path)
    can = table_can(locs, AfterFitSettings().colors)
    assert can["field"] is True
    assert shown(AfterFitSettings(), **{**can, "field": False})["field_correction"] is False
    assert shown(AfterFitSettings(field_correction=True), **can)["correction"] is True
    done = finish_localizations(locs, AfterFitSettings(field_correction=True, assign_colors=False))
    assert done.changed and "field_correction" in done.locs.metadata
    plain = Localizations(dict(locs.columns), {"pixelsize_nm": 100.0})
    assert table_can(plain, AfterFitSettings().colors)["field"] is False
    assert not finish_localizations(plain, AfterFitSettings(field_correction=True, assign_colors=False)).changed


def test_spline_3d_with_the_step_on_fits_with_the_centre_spline(tmp_path):
    from dataclasses import replace

    from smappy.plugins.fit import SplineFit, SplineFitSettings
    path = tmp_path / "field.h5"
    save_spline_calibration(path, calibration())
    settings = SplineFitSettings()
    settings = replace(settings, model=replace(settings.model, calibration=str(path), focus="calibrated"),
                       fit=replace(settings.fit, roisize=ROI))
    fitter = SplineFit()
    assert fitter.field_dependent(settings)
    assert fitter.model(settings, None).field_dependent
    settings.finish.field_correction = True
    assert not fitter.model(settings, None).field_dependent
    assert fitter.describe(None, settings).shown["finish.correction"] is True
