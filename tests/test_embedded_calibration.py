"""The calibration a table was fitted with, kept in its localization file (`smappy.io.embedded`)."""
import io
import json

import h5py
import numpy as np
import pytest

from smappy.calibrate.core import spline_coefficients
from smappy.io import embedded
from smappy.io.calibration import (SplineCalibration, load_spline_calibration, save_spline_calibration,
                                   spline_knots)
from smappy.io.hdf5 import load_localizations, save_localizations
from smappy.locs import Localizations

from test_dual_fit import dual_calibration


def _knots(nz=12, n=11, seed=0):
    """A smooth PSF on a grid: the knots of a spline made from them."""
    z, y, x = np.meshgrid(np.arange(nz), np.arange(n), np.arange(n), indexing="ij")
    rng = np.random.default_rng(seed)
    sigma = 1.5 + 0.1 * (z - nz / 2) ** 2 / nz + 0.05 * rng.random()
    return (np.exp(-((x - n // 2) ** 2 + (y - n // 2) ** 2) / (2 * sigma ** 2)) / sigma ** 2).astype(np.float32)


def _calibration(path, **parameters):
    psf = _knots()
    cal = SplineCalibration(spline_coefficients(psf), 50.0, 5.0, x0=5.0, psf=psf, em_on=False,
                            parameters=dict(parameters))
    save_spline_calibration(path, cal)
    return cal


def _fitted(path, carried, metadata=None):
    """A localization file carrying ``carried`` (an image), as a fit writes it."""
    locs = Localizations({"x_nm": np.arange(3.0), "y_nm": np.zeros(3), "frame": np.arange(3)},
                         dict(metadata or {}, **{embedded.KEY: carried}))
    save_localizations(path, locs)
    return path


def test_the_knots_stand_for_the_spline_and_the_coefficients_come_back(tmp_path):
    cal = _calibration(tmp_path / "beads_3dcal.h5", method="bead average")
    # the knots are read off the spline, to float32 rounding on the far faces (a sum of coefficients)
    assert np.allclose(spline_knots(cal.coeff), cal.psf, rtol=0, atol=1e-6 * cal.psf.max())
    data = embedded.image(tmp_path / "beads_3dcal.h5")
    with h5py.File(tmp_path / "a.h5", "w") as f:
        embedded.write(f, data)
        # the copy keeps the knots and not the coefficients, 64 times their size
        assert "coeff" not in f["calibration"] and f["calibration"].attrs["coeff_from"] == "psf"
    fitted = _fitted(tmp_path / "cells_locs.h5", data)
    back = load_spline_calibration(fitted)
    assert np.allclose(back.coeff, cal.coeff, rtol=0, atol=1e-6 * np.abs(cal.coeff).max())
    assert back.parameters == {"method": "bead average"} and back.dz == 50.0 and back.z0 == 5.0
    assert back.em_on is False and back.source == fitted


def test_a_spline_made_otherwise_keeps_its_coefficients(tmp_path):
    psf = _knots()
    coeff = spline_coefficients(psf)
    coeff[0] *= 1.05                     # not the knots' spline: SMAP's, or one smoothed
    save_spline_calibration(tmp_path / "smap_3dcal.h5", SplineCalibration(coeff, 50.0, 5.0, psf=psf))
    back = load_spline_calibration(_fitted(tmp_path / "f.h5", embedded.image(tmp_path / "smap_3dcal.h5")))
    assert np.array_equal(back.coeff, coeff)


def test_the_pupil_model_comes_with_it_and_is_found_in_the_fitted_file(tmp_path):
    from smappy.calibrate import depth, pupil
    _calibration(tmp_path / "pupil_3dcal.h5")
    with h5py.File(tmp_path / "pupil_3dcal.h5", "a") as f:       # a model as uipsf saves one, in brief
        g = f.create_group("uipsf")
        g.attrs["format"] = embedded.MODEL_FORMAT
        g.create_dataset("model/params/zernike", data=np.arange(5.0))
    fitted = _fitted(tmp_path / "cells_locs.h5", embedded.image(tmp_path / "pupil_3dcal.h5"))
    assert embedded.model_group(fitted) == "calibration/uipsf"
    assert embedded.model_group(tmp_path / "pupil_3dcal.h5") == "uipsf"
    assert pupil.carries_model(fitted)
    with h5py.File(fitted, "r") as f:
        assert np.array_equal(f["calibration/uipsf/model/params/zernike"][...], np.arange(5.0))
    # what draws a focus in the sample finds the model in the localization file
    assert depth.model_file(load_spline_calibration(fitted)) == fitted
    assert "with its pupil model" in embedded.describe(embedded.image(fitted))


def test_a_dual_colour_calibration_is_kept_and_read_back(tmp_path):
    from smappy.calibrate.dual import load_dual_color_calibration, save_dual_color_calibration
    cal = dual_calibration()
    save_dual_color_calibration(tmp_path / "dual_2ct.h5", cal)
    fitted = _fitted(tmp_path / "cells_locs.h5", embedded.image(tmp_path / "dual_2ct.h5"))
    assert embedded.carried_format(fitted) == "smappy-dual-color-calibration"
    back = load_dual_color_calibration(fitted)
    for a, b in ((back.main, cal.main), (back.secondary, cal.secondary)):
        assert np.allclose(a.coeff, b.coeff, rtol=0, atol=1e-5 * np.abs(b.coeff).max())
    assert np.array_equal(back.transformation, cal.transformation) and back.geometry == cal.geometry


def test_a_4pi_model_is_kept_and_read_back(tmp_path):
    from smappy import fourpi
    cal = fourpi.model_calibration(size=15, z_range_nm=300.0)
    fourpi.save_fourpi_calibration(tmp_path / "model_4pi.h5", cal)
    fitted = _fitted(tmp_path / "cells_locs.h5", embedded.image(tmp_path / "model_4pi.h5"))
    assert fourpi.is_fourpi_calibration(fitted)
    back = fourpi.load_fourpi_calibration(fitted)
    assert np.array_equal(back.volumes, cal.volumes) and back.period_nm == cal.period_nm


def test_field_dependent_node_splines_are_kept_as_their_knots(tmp_path):
    from smappy.field_splines import FieldSplines
    cal = _calibration(tmp_path / "plain.h5")
    nodes = np.stack([spline_coefficients(_knots(seed=s)) for s in range(4)])
    cal.field_splines = FieldSplines(nodes, np.array([10.0, 50.0]), np.array([10.0, 50.0]))
    save_spline_calibration(tmp_path / "field_3dcal.h5", cal)       # half precision on disk
    fitted = _fitted(tmp_path / "cells_locs.h5", embedded.image(tmp_path / "field_3dcal.h5"))
    with h5py.File(fitted, "r") as f:
        assert "knots" in f["calibration/field"] and "coeff" not in f["calibration/field"]
    stored = load_spline_calibration(tmp_path / "field_3dcal.h5").field_splines
    back = load_spline_calibration(fitted).field_splines
    # the half-precision coefficients moved the spline by ~4e-4 of its peak; its knots' spline is as close
    centre = embedded._centre_values
    for a, b in zip(back.coeff, stored.coeff):
        assert np.abs(centre(a) - centre(b)).max() <= 2e-3 * np.abs(centre(b)).max()
    assert np.array_equal(back.xs, stored.xs)


def test_a_file_fitted_before_says_what_it_was_fitted_with(tmp_path):
    save_localizations(tmp_path / "old_locs.h5", Localizations(
        {"x_nm": np.zeros(2)}, {"calibration": {"file": "/somewhere/beads_3dcal.h5"}}))
    assert embedded.carried_format(tmp_path / "old_locs.h5") is None
    with pytest.raises(ValueError, match="carries no calibration: it was fitted with /somewhere/beads_3dcal.h5"):
        load_spline_calibration(tmp_path / "old_locs.h5")


def test_it_survives_loading_and_saving_and_is_kept_once_for_the_first_file(tmp_path):
    _calibration(tmp_path / "beads_3dcal.h5")
    data = embedded.image(tmp_path / "beads_3dcal.h5")
    other = embedded.image(_calibration(tmp_path / "other_3dcal.h5", method="other") and
                           tmp_path / "other_3dcal.h5")
    from smappy.io.formats import FileInfo
    files = [FileInfo("a", "a.h5", "smappy", metadata={embedded.KEY: data}).to_dict(),
             FileInfo("b", "b.h5", "smappy", metadata={embedded.KEY: other}).to_dict()]
    fitted = _fitted(tmp_path / "cells_locs.h5", data, {"files": files})
    with h5py.File(fitted, "r") as f:
        written = json.loads(f["files"][()])
        assert "calibration_h5" not in json.loads(f.attrs["metadata"])       # never in the JSON
    # the first file's is the table's, written once; the second's in the list
    assert written[0]["metadata"][embedded.KEY] == "the table's"
    assert "base64" in written[1]["metadata"][embedded.KEY]
    locs = load_localizations(fitted)
    infos = [FileInfo.from_dict(d) for d in locs.metadata["files"]]
    # an image read back is not the same bytes, but the same calibration
    assert infos[0].metadata[embedded.KEY] is locs.metadata[embedded.KEY]
    assert _parameters(locs.metadata[embedded.KEY]) == {}
    assert _parameters(infos[1].metadata[embedded.KEY]) == {"method": "other"}
    save_localizations(tmp_path / "again.h5", locs)                         # saved again: still there
    assert load_spline_calibration(tmp_path / "again.h5").psf is not None


def _parameters(data):
    with h5py.File(io.BytesIO(data), "r") as f:
        return json.loads(f["parameters_json"][()])


def test_a_handed_over_table_keeps_it(tmp_path):
    from smappy import handoff
    data = embedded.image(_calibration(tmp_path / "beads_3dcal.h5") and tmp_path / "beads_3dcal.h5")
    out = handoff._carried_out({embedded.KEY: data, "camera": {}})
    json.dumps(out)                                                         # JSON-able
    assert handoff._carried_in(json.loads(json.dumps(out)))[embedded.KEY] == data


def test_a_spline_fit_keeps_its_calibration_and_its_file_fits_a_new_data_set(tmp_path):
    """The whole round: beads -> calibration -> fit; the calibration file gone, the fitted file is the
    calibration of the next fit, which keeps it in turn."""
    tifffile = pytest.importorskip("tifffile")
    from smappy.calibrate.core import CalibrationSettings, build_calibration, collect_beads
    from smappy.calibrate.input import BeadStack
    from smappy.plugins import Context
    from smappy.plugins.fit import (CameraSettings, OutputSettings, SourceSettings, SplineFit,
                                    SplineFitSettings, SplineModelSettings)
    from smappy.simulate import ASTIGMATISM, LabellingSettings, bead_stacks, camera_frames
    stacks, z = bead_stacks(2, seed=0)
    beads = collect_beads([BeadStack(s.astype(np.float32), z, source=f"b{i}")
                           for i, s in enumerate(stacks)], CalibrationSettings())
    calibration = tmp_path / "beads_3dcal.h5"
    build_calibration(beads).save(calibration)
    original = load_spline_calibration(calibration)
    frames, _ = camera_frames(60, seed=3, astigmatism=ASTIGMATISM, labelling=LabellingSettings(efficiency=0.1))
    tifffile.imwrite(tmp_path / "frames.tif", frames)

    def fit(cal, out):
        return SplineFit().run(Context(), SplineFitSettings(
            source=SourceSettings(path=str(tmp_path / "frames.tif")),
            camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
            model=SplineModelSettings(calibration=str(cal), focus="calibrated"),
            output=OutputSettings(path=str(out))))

    first = fit(calibration, tmp_path / "first.hdf5")
    assert embedded.KEY in first.locs.metadata
    kept = load_spline_calibration(tmp_path / "first.hdf5")
    assert np.allclose(kept.coeff, original.coeff, rtol=0, atol=1e-5 * np.abs(original.coeff).max())
    calibration.unlink()
    second = fit(tmp_path / "first.hdf5", tmp_path / "second.hdf5")
    assert np.allclose(np.sort(second.locs["z_nm"]), np.sort(first.locs["z_nm"]), atol=1e-2)
    assert embedded.carried_format(tmp_path / "second.hdf5") == "smappy-spline-calibration"
