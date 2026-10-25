"""Astigmatic z: z from the widths of an elliptical Gaussian fit, and its calibration from beads."""
from dataclasses import replace

import numpy as np
import pytest

from smappy.astigmatic_z import (AstigmaticCalibration, Defocus, fit_calibration, is_calibration,
                                 simulator_calibration, z_from_widths)
from smappy.locs import Localizations
from smappy.plugins import Context, get


def _widths(z, cal, noise_nm, seed=0):
    rng = np.random.default_rng(seed)
    sx, sy = cal.widths(z)
    return sx + rng.normal(0, noise_nm, len(z)), sy + rng.normal(0, noise_nm, len(z))


def test_z_comes_back_from_both_widths_with_an_error_the_size_of_its_scatter():
    cal = simulator_calibration()
    z = np.random.default_rng(1).uniform(-700, 700, 200_000)
    sx, sy = _widths(z, cal, 5.0)
    found = z_from_widths(sx, sy, cal, np.full(len(z), 5.0), np.full(len(z), 5.0))
    dz = found.z_nm - z
    assert abs(np.median(dz)) < 0.5
    pull = dz / found.z_err_nm
    # Birge's ratio makes the error a little conservative, never optimistic
    assert 0.85 < np.std(pull) < 1.0
    assert not found.edge.any()


def test_the_lookups_start_is_refined_to_the_least_squares_z_of_a_curve_with_higher_orders():
    """With A and B, sx^2 - sy^2 is no longer linear in z, and is one-to-one only near focus;
    noise-free widths must still give z back exactly, inside the range."""
    cal = AstigmaticCalibration(Defocus(125, 280, 380, 0.08, 0.02), Defocus(135, -320, 420, -0.05, 0.01),
                                (-700.0, 700.0))
    z = np.linspace(-650, 650, 1301)
    sx, sy = cal.widths(z)
    found = z_from_widths(sx, sy, cal, np.ones_like(z), np.ones_like(z))
    np.testing.assert_allclose(found.z_nm, z, atol=0.05)


def test_widths_that_match_no_height_get_a_large_error():
    cal = simulator_calibration()
    sx, sy = cal.widths(np.array([0.0, 0.0]))
    # the second spot is round but half again as wide as any round spot: two molecules overlapping
    found = z_from_widths(np.r_[sx[0], 1.5 * sx[1]], np.r_[sy[0], 1.5 * sy[1]], cal,
                          np.full(2, 3.0), np.full(2, 3.0))
    assert found.chi2[1] > 100 * max(found.chi2[0], 1.0)
    assert found.z_err_nm[1] > 10 * found.z_err_nm[0]


def test_a_z_beyond_the_calibrated_range_has_the_whole_range_as_its_error():
    cal = simulator_calibration(z_range_nm=500.0)
    sx, sy = cal.x.width(np.r_[0.0, 900.0]), cal.y.width(np.r_[0.0, 900.0])
    found = z_from_widths(np.r_[sx, np.nan], np.r_[sy, 150.0], cal, np.full(3, 2.0), np.full(3, 2.0))
    assert found.z_nm[1] == pytest.approx(500.0, abs=1e-3)
    assert found.edge.tolist() == [False, True, True]
    assert found.z_err_nm[1] == found.z_err_nm[2] == 1000.0
    assert np.isnan(found.z_nm[2])                       # no widths, no z
    assert found.z_err_nm[0] < 20


def test_without_width_errors_one_error_is_estimated_from_the_residuals():
    cal = simulator_calibration()
    z = np.random.default_rng(2).uniform(-600, 600, 50_000)
    sx, sy = _widths(z, cal, 4.0, seed=3)
    found = z_from_widths(sx, sy, cal)
    assert found.width_err_nm == pytest.approx(4.0, rel=0.03)
    assert 0.85 < np.std((found.z_nm - z) / found.z_err_nm) < 1.0


def test_the_curves_come_back_from_their_own_widths_with_z_zero_where_the_spot_is_round():
    true = AstigmaticCalibration(Defocus(128, 260, 370, 0.06, 0.01), Defocus(132, -340, 410, -0.04, 0.0),
                                 (-800.0, 800.0))
    z = np.repeat(np.arange(-800.0, 801.0, 20.0), 6)
    sx, sy = _widths(z, true, 2.0, seed=4)
    shift = 37.0                                          # the stage's zero is not the round spot
    cal = fit_calibration(z + shift, sx, sy, np.full(len(z), 2.0), np.full(len(z), 2.0))
    round_z = cal.z_range[0] + np.argmin(np.abs(np.subtract(*cal.widths(
        np.linspace(*cal.z_range, 16001))))) * 0.1
    assert abs(round_z) < 1.0
    # the same curves, moved by where the true ones are round
    zero = np.linspace(-800, 800, 16001)[np.argmin(np.abs(np.subtract(*true.widths(np.linspace(-800, 800, 16001)))))]
    grid = np.linspace(-600, 600, 61)
    for fitted, axis in ((cal.x, true.x), (cal.y, true.y)):
        np.testing.assert_allclose(fitted.width(grid - zero), axis.width(grid), atol=1.5)
    assert cal.residual_nm == pytest.approx((2.0, 2.0), rel=0.1)
    plain = fit_calibration(z, sx, sy, higher_orders=False)
    assert plain.x.a == plain.x.b == 0.0
    assert plain.residual_nm[0] > cal.residual_nm[0]      # the plain beam misses these widths


def test_a_calibration_is_saved_and_read_back_and_told_from_other_files(tmp_path):
    import h5py
    cal = replace(simulator_calibration(), residual_nm=(2.5, 3.0), beads=7)
    path = cal.save(tmp_path / "beads_astigz.h5")
    assert is_calibration(path)
    assert AstigmaticCalibration.load(path) == replace(cal, source=cal.source)
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f["x"] = [1]
    assert not is_calibration(other)
    assert not is_calibration(tmp_path / "missing.h5")
    with pytest.raises(ValueError, match="elliptical Gaussian"):
        AstigmaticCalibration.load(other)


def _beads(n_stacks=1):
    from smappy.calibrate.core import CalibrationSettings, collect_beads
    from smappy.calibrate.input import BeadStack
    from smappy.simulate.camera import bead_stacks
    stacks, z = bead_stacks(n_stacks=n_stacks)
    return [BeadStack(s, z) for s in stacks], CalibrationSettings()


CAMERA = dict(conversion=0.5, offset=100.0, pixelsize_um=0.1)


def test_beads_through_focus_calibrate_the_curves_they_were_drawn_with():
    """`bead_stacks` draws the simulator's astigmatic Gaussian at -z of the objective: the curves
    come back in the sample's z, x in focus above the round spot, as the simulator puts it."""
    from smappy.calibrate.core import collect_beads
    from smappy.calibrate.elliptical import bead_widths, calibrate_widths
    from smappy.metadata import CameraMetadata
    stacks, settings = _beads()
    widths = bead_widths(collect_beads(stacks, settings), CameraMetadata(**CAMERA), (100.0, 100.0))
    for higher in (True, False):
        cal = calibrate_widths(widths, higher_orders=higher)
        assert cal.x.s0 == pytest.approx(130, abs=3) and cal.y.s0 == pytest.approx(130, abs=3)
        assert cal.x.c == pytest.approx(300, abs=10) and cal.y.c == pytest.approx(-300, abs=10)
        assert cal.x.d == pytest.approx(400, abs=15) and cal.y.d == pytest.approx(400, abs=15)
    assert np.all(np.abs(widths.z0_nm) < 15)              # the beads lie on one plane


def test_the_psf_windows_elliptical_method_saves_a_calibration_astigmatic_z_reads(tmp_path):
    from smappy.calibrate.elliptical import EllipticalCalibrationSettings
    from smappy.calibrate.psf_methods import METHODS
    from smappy.calibrate.pupil import RunSettings
    from smappy.metadata import CameraMetadata
    method = next(m for m in METHODS if m.key == "elliptical")
    stacks, settings = _beads()
    inputs = {"detection": settings, "run": RunSettings(save_model=False),
              "camera": CameraMetadata(**CAMERA), "elliptical": EllipticalCalibrationSettings()}
    detected = method.detect(stacks, False, inputs)
    run = method.run(stacks, False, inputs, detected=detected, marks={0: "exclude"})
    rows = method.rows(run, method.initial_marks(run))       # the marks stay after the run
    assert rows[0][0][-1] == "manual exclusion" and rows[0][1]
    assert run.result["calibration"].beads == sum(r[0][-1] == "used" for r in rows)
    assert "beads used" in method.summary(run)
    saved = method.save_as(run, tmp_path / "beads_astigz.h5")
    assert is_calibration(saved)
    with pytest.raises(ValueError, match="one channel"):
        method.detect(stacks, True, inputs)


def test_simulated_astigmatic_frames_fitted_elliptical_give_their_true_z(tmp_path):
    """The whole path: frames -> Gaussian 2D, elliptical -> Astigmatic z on the simulator's curves.
    Isolated spots only: a neighbour in the fit box makes a spot rounder, z nearer focus (NOTES.md,
    "Crowding compresses z")."""
    tifffile = pytest.importorskip("tifffile")
    from scipy.spatial import cKDTree
    from smappy.plugins.astigmatic_z import AstigmaticZSettings
    from smappy.plugins.fit import (CameraSettings, GaussianFit, GaussianFitSettings,
                                    GaussianModelSettings, OutputSettings, SourceSettings)
    from smappy.simulate import ISOLATED_NM, LabellingSettings, camera_frames
    frames, truth = camera_frames(300, seed=2, labelling=LabellingSettings(efficiency=0.1),
                                  astigmatism=(300.0, 400.0))
    tifffile.imwrite(tmp_path / "frames.tif", frames)
    fit = GaussianFitSettings(source=SourceSettings(path=str(tmp_path / "frames.tif")),
                              camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
                              model=GaussianModelSettings(elliptical=True), output=OutputSettings(path=""))
    locs = GaussianFit().run(Context(), fit).locs
    assert {"sigma_x_err_nm", "sigma_y_err_nm"} <= set(locs.keys())
    result = get("Analysis/After fit/Astigmatic z")().run(
        Context(locs=locs), AstigmaticZSettings(calibration="typed"))
    z, err = np.asarray(result.locs["z_nm"]), np.asarray(result.locs["z_err_nm"])
    clean = (np.asarray(truth["photons"]) > 500) & (np.asarray(truth["neighbour_nm"]) > ISOLATED_NM)
    found, true, errors = [], [], []
    for f in np.unique(locs["frame"]):
        here = np.flatnonzero(np.asarray(locs["frame"]) == f)
        there = np.flatnonzero((np.asarray(truth["frame"]) == f) & clean)
        if len(there):
            d, i = cKDTree(np.c_[truth["x_nm"][there], truth["y_nm"][there]]).query(
                np.c_[locs["x_nm"][here], locs["y_nm"][here]])
            found += list(z[here[d < 50]])
            errors += list(err[here[d < 50]])
            true += list(np.asarray(truth["z_nm"])[there[i[d < 50]]])
    found, true, errors = map(np.asarray, (found, true, errors))
    assert len(found) > 500
    slope, offset = np.polyfit(true, found, 1)
    assert slope == pytest.approx(1.0, abs=0.03) and abs(offset) < 8
    spread = 1.4826 * np.median(np.abs((found - true) / errors))
    assert 0.65 < spread < 1.1


def test_the_plugin_writes_z_scaled_and_names_the_columns_it_needs():
    from smappy.plugins.astigmatic_z import AstigmaticZSettings, TypedCalibration
    cal = simulator_calibration()
    z = np.linspace(-500, 500, 101)
    sx, sy = cal.widths(z)
    locs = Localizations({"x_nm": np.zeros_like(z), "y_nm": np.zeros_like(z),
                          "sigma_x_nm": sx, "sigma_y_nm": sy, "frame": np.arange(len(z))})
    plugin = get("Analysis/After fit/Astigmatic z")()
    settings = AstigmaticZSettings(calibration="typed", typed=TypedCalibration(), z_scale=0.8)
    result = plugin.run(Context(locs=locs), settings)
    np.testing.assert_allclose(result.locs["z_nm"], 0.8 * z, atol=0.05)
    assert "no width errors" in result.text
    assert "z_nm" not in locs                             # a new table, the old one untouched
    with pytest.raises(ValueError, match="elliptical"):
        plugin.run(Context(locs=Localizations({"x_nm": z, "y_nm": z, "sigma_nm": sx})), settings)
    with pytest.raises(ValueError, match="calibration file"):
        plugin.run(Context(locs=locs), AstigmaticZSettings())
    state = plugin.describe(None, settings)
    assert state.shown == {"path": False, "typed": True}


def test_fit_this_on_astigmatic_frames_points_to_astigmatic_z_with_the_simulators_curves():
    from smappy.plugins.simulate import astigmatic_note
    settings = get("File/Simulate/Simulate").Settings()
    settings.output = "camera"
    assert astigmatic_note(settings) is None
    settings.optics.astigmatism = True
    settings.optics.focal_offset_nm = 250.0
    note = astigmatic_note(settings)
    assert "Astigmatic z" in note and "250" in note and "-250" in note
    settings.optics.calibration = "beads_3dcal.mat"        # drawn with a spline: Spline 3D's
    assert astigmatic_note(settings) is None
