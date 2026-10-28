"""Edit columns, Collapse by id and Save's columns: `smappy.table_edit`."""
import re

import h5py
import numpy as np
import pytest

from smappy import plugins
from smappy.group import GroupSettings, group
from smappy.locs import KEPT_BOUNDS, Localizations
from smappy.mathparse import apply_recipes, recipes, remember, rename_in
from smappy.render import FieldOfView, RenderSettings, render_locs
from smappy.session import Session
from smappy.table_edit import (MERGED, collapse, keep_columns, remove_columns,
                               rename_column)

EDIT = "Analysis/Process/Edit columns"
COLLAPSE = "Analysis/Process/Collapse by id"


def table(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 5e3, n).astype(np.float32),
        "y_nm": rng.uniform(0, 5e3, n).astype(np.float32),
        "frame": np.arange(n, dtype=np.int64),
        "photons": rng.uniform(500, 5000, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 20, n).astype(np.float32),
        "cluster_id": np.repeat(np.arange(0, 30), n // 30).astype(np.int64)})


def test_a_rename_in_an_expression_touches_the_column_only():
    assert rename_in("photons * (photons_err < max(photons))", "photons", "N") \
        == "N * (photons_err < max(N))"
    assert rename_in("max(x) + sqrt( max )", "max", "m") == "max(x) + sqrt( m )"


def test_a_rename_keeps_the_values_and_carries_recipes_and_kept_bounds():
    locs = table()
    locs.columns["bright"] = np.asarray(locs["photons"]) * 2
    remember(locs, "bright", "photons * 2", grouped="sum")
    locs.metadata[KEPT_BOUNDS] = {"photons": [1000, None]}
    out, notes = rename_column(locs, "photons", "N")
    assert "photons" not in out and np.array_equal(out["N"], locs["photons"])
    assert list(out.keys()).index("N") == list(locs.keys()).index("photons")
    assert recipes(out)[0] == {"field": "bright", "expression": "N * 2", "grouped": "sum"}
    assert out.metadata[KEPT_BOUNDS] == {"N": [1000, None]}
    assert len(notes) == 2
    # the recipe still runs on the renamed table, and the old one is untouched
    assert apply_recipes(out) == ["bright"]
    assert recipes(locs)[0]["expression"] == "photons * 2" and "photons" in locs


def test_a_rename_of_a_derived_field_renames_its_recipe():
    locs = table()
    remember(locs, "bright", "photons * 2")
    apply_recipes(locs)
    out, _ = rename_column(locs, "bright", "twice")
    assert [r["field"] for r in recipes(out)] == ["twice"]


def test_structural_columns_and_taken_names_are_refused():
    locs = table()
    for name in ("x_nm", "frame", "xy_err_nm"):
        with pytest.raises(ValueError, match="cannot be renamed"):
            rename_column(locs, name, "other")
    with pytest.raises(ValueError, match="cannot be renamed"):
        remove_columns(locs, ["frame"])
    with pytest.raises(ValueError, match="already has"):
        rename_column(locs, "photons", "cluster_id")
    with pytest.raises(ValueError, match="function or a constant"):
        rename_column(locs, "photons", "max")
    with pytest.raises(ValueError, match="no column"):
        remove_columns(locs, ["nothing"])


def test_a_removal_forgets_the_recipes_that_read_it_and_says_so():
    locs = table()
    remember(locs, "bright", "photons * 2", grouped="sum")
    remember(locs, "precise", "xy_err_nm < 10")
    apply_recipes(locs)
    locs.metadata[KEPT_BOUNDS] = {"photons": [1000, None], "precise": [0.5, None]}
    out, notes = remove_columns(locs, ["photons"])
    assert "photons" not in out and np.array_equal(out["bright"], locs["bright"])
    assert [r["field"] for r in recipes(out)] == ["precise"]
    assert out.metadata[KEPT_BOUNDS] == {"precise": [0.5, None]}
    assert any("bright" in n for n in notes) and any("photons" in n for n in notes)


def test_the_picture_is_drawn_without_a_colour_column_the_table_lacks():
    locs = table()
    fov = FieldOfView(0.0, 0.0, 50.0, 100, 100)
    settings = RenderSettings(color_field="gone", weight_field="also_gone")
    plain = render_locs(locs, fov, RenderSettings())
    image = render_locs(locs, fov, settings)
    assert image.color is None and np.allclose(image.weight, plain.weight)


def test_the_plugin_renames_and_the_layers_follow_and_an_undo_brings_it_back():
    session = Session(table())
    layer = session.layers[0]
    layer.set_bound("photons", 1000.0, None)
    import dataclasses
    layer.state.settings = dataclasses.replace(layer.state.settings, color_field="photons")
    shown = int(layer.selection(0).mask.sum())
    plugin = plugins.get(EDIT)
    session.run(plugin(), plugin.Settings(operation="rename", column="photons",
                                          new_name="N"))
    assert "N" in session.locs and "photons" not in session.locs
    assert layer.ranges["N"] == (1000.0, None)
    assert layer.state.settings.color_field == "N"
    assert int(layer.selection(0).mask.sum()) == shown
    session.undo()
    assert "photons" in session.locs and "N" not in session.locs


def test_the_plugin_removes_the_ticked_columns_and_warns_of_the_filters_lost():
    session = Session(table())
    session.layers[0].set_bound("photons", 1000.0, None)
    plugin = plugins.get(EDIT)
    result = session.run(plugin(), plugin.Settings(operation="remove",
                                                   remove=["photons", "cluster_id"]))
    assert set(session.locs.keys()) == {"x_nm", "y_nm", "frame", "xy_err_nm"}
    assert "removed photons, cluster_id" in result.text
    assert "filter on photons" in result.text
    with pytest.raises(ValueError, match="tick"):
        plugin().run(plugins.Context(locs=table()), plugin.Settings(operation="remove"))


# ------------------------------------------------------------- collapsing

def test_collapsed_by_group_id_it_is_the_merged_blinks():
    from smappy.simulate import simulate
    locs = simulate(n_frames=400, seed=3)
    grouped, _ = group(locs, GroupSettings(dx=50.0, dt=1))
    out, _ = collapse(locs, "group_id")
    assert len(out) == len(grouped)
    assert "group_id" not in out and "n_in_group" not in out
    order = np.lexsort((np.asarray(grouped["x_nm"]), np.asarray(grouped["frame"])))
    mine = np.lexsort((np.asarray(out["x_nm"]), np.asarray(out["frame"])))
    for name in ("x_nm", "y_nm", "photons", "xy_err_nm", "frame"):
        assert np.allclose(np.asarray(out[name])[mine], np.asarray(grouped[name])[order],
                           rtol=1e-5), name
    assert np.array_equal(np.asarray(out[MERGED])[mine],
                          np.asarray(grouped["n_in_group"])[order])
    assert out.metadata["collapsed"]["by"] == "group_id"


def test_one_row_per_id_per_file_weighted_by_precision_and_none_kept_or_removed():
    n_per, ids = 20, np.arange(1, 6)
    rng = np.random.default_rng(1)
    centre = rng.uniform(0, 5000, (len(ids), 2))
    rows = np.repeat(ids, n_per)
    err = rng.uniform(5, 30, rows.size)
    x = centre[rows - 1, 0] + rng.normal(0, 1, rows.size) * err
    noise = 7                                      # rows in no cluster
    locs = Localizations({
        "x_nm": np.concatenate([x, rng.uniform(0, 5e3, noise)]).astype(np.float32),
        "y_nm": np.concatenate([centre[rows - 1, 1], rng.uniform(0, 5e3, noise)]).astype(np.float32),
        "xy_err_nm": np.concatenate([err, np.full(noise, 10.0)]).astype(np.float32),
        "photons": np.full(rows.size + noise, 100.0, np.float32),
        "frame": np.arange(rows.size + noise, dtype=np.int64),
        "cluster_id": np.concatenate([rows, np.zeros(noise, int)]).astype(np.int64),
        # the second half of each cluster in another file
        "filenumber": np.concatenate([np.tile(np.repeat([0, 1], n_per // 2), len(ids)),
                                      np.zeros(noise, int)]).astype(np.int32)})
    kept, _ = collapse(locs, "cluster_id", "keep")
    removed, _ = collapse(locs, "cluster_id", "remove")
    assert len(kept) == 2 * len(ids) + noise and len(removed) == 2 * len(ids)
    assert sorted(np.asarray(removed[MERGED])) == [n_per // 2] * (2 * len(ids))
    assert np.allclose(removed["photons"], 100.0 * n_per / 2)
    # the inverse-variance mean of each file's half of cluster 3
    for f in (0, 1):
        sel = (np.asarray(locs["cluster_id"]) == 3) & (np.asarray(locs["filenumber"]) == f)
        w = 1 / err[sel[:rows.size]] ** 2
        expected = np.sum(np.asarray(locs["x_nm"], float)[sel] * w) / w.sum()
        row = (np.asarray(removed["cluster_id"]) == 3) & (np.asarray(removed["filenumber"]) == f)
        assert np.isclose(np.asarray(removed["x_nm"])[row][0], expected, rtol=1e-5)
    assert removed["cluster_id"].dtype.kind == "i"
    # the unassigned rows are themselves, one each
    alone = np.asarray(kept["cluster_id"]) == 0
    assert alone.sum() == noise and np.all(np.asarray(kept[MERGED])[alone] == 1)
    assert np.allclose(np.sort(np.asarray(kept["x_nm"])[alone]),
                       np.sort(np.asarray(locs["x_nm"])[-noise:]))


def test_a_recipe_in_the_grouping_columns_is_kept_as_values():
    from smappy.simulate import simulate
    locs = simulate(n_frames=300, seed=4)
    group(locs, GroupSettings())
    remember(locs, "on_ms", "n_in_group * 20")
    apply_recipes(locs)
    out, notes = collapse(locs, "group_id")
    assert np.allclose(out["on_ms"], np.asarray(out[MERGED]) * 20)
    assert "on_ms" not in [r["field"] for r in recipes(out)]
    assert any("on_ms" in n for n in notes)


def test_the_collapse_plugin_is_one_undo_step():
    session = Session(table())
    plugin = plugins.get(COLLAPSE)
    session.run(plugin(), plugin.Settings(column="cluster_id", unassigned="remove"))
    assert len(session.locs) == 29
    session.undo()
    assert len(session.locs) == 600


# ------------------------------------------------------------- saving

def test_keep_columns_keeps_the_positions_and_drops_the_recipes_of_the_rest():
    locs = table()
    remember(locs, "bright", "photons * 2")
    apply_recipes(locs)
    out = keep_columns(locs, ["photons"])
    assert set(out.keys()) == {"x_nm", "y_nm", "photons"}
    assert recipes(out) == []
    assert keep_columns(locs, list(locs.keys())) is locs


def test_save_writes_the_chosen_columns_and_the_session_keeps_all(tmp_path):
    locs = table()
    remember(locs, "bright", "photons * 2")
    apply_recipes(locs)
    session = Session(locs)
    session.set_unsaved(True)
    plugin = plugins.get("File/Save/smappy HDF5")
    path = tmp_path / "few.hdf5"
    result = session.run(plugin(), plugin.Settings(path=str(path), gui_state=False,
                                                   columns=["photons", "frame"]))
    with h5py.File(path, "r") as f:
        assert set(f["locs"]) == {"x_nm", "y_nm", "photons", "frame"}
    assert re.search(r"4 of \d+ columns", result.text)
    assert "bright" in session.locs and session.unsaved
    import json
    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["metadata"]).get("derived", []) == []
    session.run(plugin(), plugin.Settings(path=str(tmp_path / "all.hdf5"), gui_state=False))
    with h5py.File(tmp_path / "all.hdf5", "r") as f:
        assert "bright" in f["locs"]
    assert not session.unsaved
