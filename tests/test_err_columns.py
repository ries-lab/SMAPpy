"""Precisions follow one rule, ``<quantity>_err_<unit>``.

The lateral precision is ``xy_err_nm``, the RMS of ``x_err_nm`` and
``y_err_nm``, and the axial one is ``z_err_nm`` -- the name SMAPpy's own 3D
fit writes, so its z precision reaches Statistics and the render.
"""
import numpy as np

from smappy.columns import add_xy_err
from smappy.locs import Localizations, to_nm

from test_batch_foundations import blinks


# ------------------------------------------------------------- xy_err itself
def test_the_lateral_precision_is_the_rms_of_the_two_errors():
    columns = add_xy_err({"x_err_nm": np.array([3.0, 6.0]),
                          "y_err_nm": np.array([4.0, 8.0])})
    np.testing.assert_allclose(columns["xy_err_nm"],
                               np.sqrt([(9 + 16) / 2, (36 + 64) / 2]), rtol=1e-6)


def test_it_stays_the_rms_after_a_conversion_with_pixels_that_are_not_square():
    x_err, y_err = np.array([0.1, 0.2]), np.array([0.3, 0.1])
    locs = Localizations(add_xy_err({"x_pix": np.zeros(2), "y_pix": np.zeros(2),
                                     "x_err_pix": x_err, "y_err_pix": y_err}),
                         {"units": "pixel"})
    nm = to_nm(locs, (100.0, 120.0))
    np.testing.assert_allclose(
        nm["xy_err_nm"], np.sqrt(((100 * x_err) ** 2 + (120 * y_err) ** 2) / 2),
        rtol=1e-6)


def test_after_grouping_it_is_derived_from_the_combined_errors():
    from smappy.group import GroupSettings, group
    rng = np.random.default_rng(0)
    locs = blinks()
    n = len(locs)
    x_err = rng.uniform(4, 20, n)
    y_err = rng.uniform(4, 20, n)
    locs.columns.update(add_xy_err({"x_err_nm": x_err, "y_err_nm": y_err}))
    grouped, _ = group(locs, GroupSettings())
    assert len(grouped) == 40
    np.testing.assert_allclose(
        grouped["xy_err_nm"],
        np.sqrt((grouped["x_err_nm"].astype(float) ** 2
                 + grouped["y_err_nm"].astype(float) ** 2) / 2), rtol=1e-5)
    # which is not what the precision rule applied to it would have given
    first = locs["group_id"] == 1
    by_rule = 1 / np.sqrt(np.sum(1 / locs["xy_err_nm"][first].astype(float) ** 2))
    assert abs(grouped["xy_err_nm"][0] - by_rule) > 1e-4


def test_a_thunderstorm_csv_brings_both_uncertainties(tmp_path):
    from smappy.io.formats import load
    path = tmp_path / "ts.csv"
    path.write_text('"id","frame","x [nm]","y [nm]","z [nm]","uncertainty [nm]",'
                    '"uncertainty_z [nm]"\n1,1,100,200,10,7.5,21\n2,2,300,400,-5,9,30\n')
    locs, _ = load(path)
    np.testing.assert_allclose(locs["xy_err_nm"], [7.5, 9])
    np.testing.assert_allclose(locs["z_err_nm"], [21, 30])


# -------------------------------------------- a SMAPpy 3D fit, end to end
def test_a_spline_3d_fits_z_precision_reaches_statistics_and_the_render():
    from test_dual_fit import camera, finder, spline

    from smappy.io.calibration import evaluate_spline
    from smappy.pipeline import FitSettings, fit_stack
    from smappy.plugins.statistics import statistics
    from smappy.psf import SplinePSF
    from smappy.render import (FieldOfView, RenderAxes, RenderSettings,
                               precision_for, render_sigmas)

    cal = spline(0.6)
    rng = np.random.default_rng(3)
    size, n_frames = 15, 40
    frames = []
    for _ in range(n_frames):
        frame = np.full((48, 48), 10.0)
        for cx, cy in ((12.3, 12.6), (34.7, 13.2), (23.1, 34.4)):
            x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
            z = rng.uniform(12, 28)
            frame[y0:y0 + size, x0:x0 + size] += evaluate_spline(
                cal, cx - x0, cy - y0, z, size) * 3000
        frames.append(rng.poisson(frame).astype(np.float32))
    frames = np.stack(frames)
    locs, _ = fit_stack([(0, frames)], camera(), finder(), SplinePSF(cal),
                        FitSettings(roisize=13, output_unit="nm"))

    assert len(locs) >= 0.9 * 3 * n_frames
    assert {"z_nm", "z_err_nm", "xy_err_nm"} <= set(locs.keys())
    assert not any(n.startswith("loc_precision") for n in locs.keys())
    np.testing.assert_allclose(
        locs["xy_err_nm"],
        np.sqrt((locs["x_err_nm"] ** 2 + locs["y_err_nm"] ** 2) / 2), rtol=1e-5)
    z_err = np.median(locs["z_err_nm"])
    assert 1 < z_err < 100                       # nm, a real axial precision

    found = {d.key: d for d in statistics(locs)}
    assert "precision_z" in found and "precision" in found

    # the renderer blurs z by the axial precision, not the lateral one
    assert precision_for(locs, "z_nm") == "z_err_nm"
    # a side view: x across, z up, 1 nm pixels so that no floor applies
    settings = RenderSettings(mode="precision", axes=RenderAxes(x="x_nm", y="z_nm"))
    fov = FieldOfView(0.0, -500.0, 1.0, 4800, 1000)
    sigma_x, sigma_z = render_sigmas(locs, settings, fov)
    assert sigma_z is not None
    np.testing.assert_allclose(sigma_x, settings.sigma_settings.apply(locs["xy_err_nm"], 1.0))
    np.testing.assert_allclose(sigma_z, settings.sigma_settings.apply(locs["z_err_nm"], 1.0))
