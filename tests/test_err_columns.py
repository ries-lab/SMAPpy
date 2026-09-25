"""Precisions follow one rule, ``<quantity>_err_<unit>``, and what was saved
under the old names still reads.

``loc_precision_nm`` is now ``xy_err_nm``, the RMS of ``x_err_nm`` and
``y_err_nm``, and ``loc_precision_z_nm`` is ``z_err_nm`` -- the name SMAPpy's
own 3D fit always wrote, so its z precision now reaches Statistics and the
render.
"""
import numpy as np
import pytest

from smappy import chain
from smappy.chain import chain_class
from smappy.columns import RENAMED, add_xy_err, current, current_name
from smappy.locs import Localizations, to_nm
from smappy.session import Session

from test_batch_foundations import blinks


# ------------------------------------------------------------------ the map
def test_an_old_name_is_renamed_as_a_word_in_keys_and_expressions():
    assert current_name("loc_precision_nm") == "xy_err_nm"
    assert current_name("loc_precision_z_nm") == "z_err_nm"
    assert current_name("loc_precision_pix_ch1") == "xy_err_pix_ch1"
    assert (current_name("(loc_precision_nm < 25) & (loc_precision_z_nm < 60)")
            == "(xy_err_nm < 25) & (z_err_nm < 60)")
    # a name that only contains an old one is somebody else's column
    assert current_name("my_loc_precision_nm") == "my_loc_precision_nm"
    assert current_name("loc_precision_nm2") == "loc_precision_nm2"


def test_nested_state_is_renamed_and_a_new_name_already_there_wins():
    old = {"bounds": {"loc_precision_nm": [None, 25.0], "photons": [100, None]},
           "steps": [{"values": {"field": "loc_precision_z_nm"}}],
           "n": 3, "array": np.arange(3)}
    new = current(old)
    assert new["bounds"] == {"xy_err_nm": [None, 25.0], "photons": [100, None]}
    assert new["steps"][0]["values"]["field"] == "z_err_nm"
    assert new["n"] == 3 and new["array"] is old["array"]
    both = current({"loc_precision_nm": 1, "xy_err_nm": 2})
    assert both == {"xy_err_nm": 2}
    assert set(RENAMED.values()) == {"xy_err_nm", "xy_err_pix", "z_err_nm"}


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


# --------------------------------------------------------------- old files
def old_table(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations(
        {"x_nm": rng.uniform(0, 1000, n), "y_nm": rng.uniform(0, 1000, n),
         "z_nm": rng.uniform(-300, 300, n), "frame": np.arange(n, dtype=np.int64),
         "photons": rng.uniform(500, 5000, n),
         "loc_precision_nm": rng.uniform(3, 30, n),
         "loc_precision_z_nm": rng.uniform(10, 60, n)},
        {"units": "nm",
         "derived": [{"field": "sharp", "expression": "loc_precision_nm < 10",
                      "grouped": "recompute"}],
         "roi_project": {"filters": {"loc_precision_nm": [None, 25.0]}}})


def test_an_hdf5_file_with_the_old_names_opens_under_the_new_ones(tmp_path):
    from smappy.io.formats import load
    from smappy.io.hdf5 import (load_gui_state, load_localizations,
                                load_results, save_gui_state,
                                save_localizations, save_results)
    written = old_table()
    path = save_localizations(tmp_path / "old.hdf5", written, written.metadata)
    save_gui_state(path, {"tabs": [{"name": "Analysis", "instances": [
        {"values": {"expression": "loc_precision_nm < 20"}}]}]})
    save_results(path, {"Analysis/Measure/X": {"settings": {"field": "loc_precision_nm"}}})
    # the file really does carry the old names; only reading renames them
    assert "loc_precision_nm" in load_localizations(path, renamed=False)

    locs, _ = load(path)
    assert "loc_precision_nm" not in locs and "loc_precision_z_nm" not in locs
    np.testing.assert_array_equal(locs["xy_err_nm"], written["loc_precision_nm"])
    np.testing.assert_array_equal(locs["z_err_nm"], written["loc_precision_z_nm"])
    assert locs.metadata["derived"][0]["expression"] == "xy_err_nm < 10"
    assert locs.metadata["roi_project"]["filters"] == {"xy_err_nm": [None, 25.0]}
    state = load_gui_state(path)
    assert state["tabs"][0]["instances"][0]["values"]["expression"] == "xy_err_nm < 20"
    assert load_results(path)["Analysis/Measure/X"]["settings"]["field"] == "xy_err_nm"

    # and a session opened on it filters by the new name out of the box
    session = Session()
    session.load(path)
    assert "xy_err_nm" in session.layers[0].filter.ranges
    # saving writes only the new names
    session.save(tmp_path / "new.hdf5")
    again = load_localizations(tmp_path / "new.hdf5", renamed=False)
    assert "xy_err_nm" in again and "loc_precision_nm" not in again


OLD_CHAIN = """\
schema: smappy-chain-v1
steps:
  - plugin: Chain/Layers
    label: filter
    version: "1"
    values:
      layers:
        - grouped: false
          start: empty
          bounds:
            - {field: loc_precision_nm, hi: 20}
      remove: true
  - plugin: Analysis/Process/Math Parser
    label: flag
    values: {field: sharp, expression: "loc_precision_nm < 10"}
"""


def test_a_chain_saved_with_the_old_names_reads_and_runs(tmp_path):
    path = tmp_path / "old.chain.yaml"
    path.write_text(OLD_CHAIN)
    spec = chain.read(path)
    assert spec.steps[0].values["layers"][0]["bounds"][0]["field"] == "xy_err_nm"
    session = Session(blinks())
    cls = chain_class(spec)
    session.run(cls(), cls.Settings())
    kept = blinks()["xy_err_nm"] <= 20
    assert len(session.locs) == int(kept.sum())
    assert session.layers[0].filter.ranges == {"xy_err_nm": (None, 20.0)}
    np.testing.assert_array_equal(session.locs["sharp"] != 0,
                                  session.locs["xy_err_nm"] < 10)


def test_a_batch_job_and_a_workspace_with_the_old_names_read(tmp_path):
    from pathlib import Path

    import yaml

    from smappy import batch, workspace
    examples = Path(__file__).parent.parent / "docs" / "examples"
    for name in ("cells.batch.yaml", "filter_drift_statistics.chain.yaml"):
        text = (examples / name).read_text().replace("xy_err_nm", "loc_precision_nm")
        assert "loc_precision_nm" in text
        (tmp_path / name).write_text(text)
    job = batch.read_job(tmp_path / "cells.batch.yaml")
    assert "loc_precision_nm" not in repr(job) and "xy_err_nm" in repr(job)

    saved = workspace.Workspace.default().to_dict()
    saved["tabs"][0]["instances"][0]["values"] = {"expression": "loc_precision_nm < 5"}
    (tmp_path / "ws.yaml").write_text(yaml.safe_dump(saved))
    loaded = workspace.load(tmp_path / "ws.yaml").to_dict()
    assert loaded["tabs"][0]["instances"][0]["values"] == {"expression": "xy_err_nm < 5"}


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
