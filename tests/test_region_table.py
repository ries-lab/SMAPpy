"""An evaluation's results in columns: rows set whole, read back as stored."""
import h5py
import numpy as np

from smappy.region_table import RegionTable

STEP = {"plugin": "A/B", "version": "1", "parameters": {"bins": 10}}
OTHER = {"plugin": "A/B", "version": "1", "parameters": {"bins": 20}}


def a_table():
    t = RegionTable("stats")
    t.set("r1", {"values": {"n": 3, "mean": 2.5, "name": "ring", "ok": True},
                 "arrays": {"profile": [1.0, 2.0, 3.0]}, "signature": "s1"}, STEP, "d1")
    t.set("r2", {"values": {"n": 5, "mean": float("nan")},
                 "arrays": {"profile": [4.0]}, "signature": "s2"}, OTHER, "d2")
    t.set("r3", {"error": "ValueError: empty", "signature": "s3"}, STEP)
    return t


def test_a_row_reads_back_as_it_was_stored_and_says_what_made_it():
    t = a_table()
    assert len(t) == 3 and "r2" in t and "r9" not in t
    assert t.entry("r1") == {"values": {"n": 3, "mean": 2.5, "name": "ring", "ok": True},
                             "arrays": {"profile": [1.0, 2.0, 3.0]}, "signature": "s1"}
    # a value a row did not return is missing, not zero
    assert "name" not in t.entry("r2")["values"]
    assert t.entry("r3") == {"error": "ValueError: empty", "signature": "s3"}
    assert t.step("r1") == STEP and t.step("r2") == OTHER and len(t.steps) == 2
    assert t.data_signature("r2") == "d2" and t.entry("nope") is None


def test_setting_a_row_again_replaces_it_whole():
    t = a_table()
    t.set("r1", {"values": {"n": 4}, "signature": "s1b"}, OTHER)
    assert t.entry("r1") == {"values": {"n": 4}, "signature": "s1b"}
    assert len(t) == 3 and t.step("r1") == OTHER


def test_the_columns_are_arrays_with_nan_where_a_row_has_none():
    c = a_table().columns()
    assert np.allclose(c["n"][:2], [3, 5]) and np.isnan(c["n"][2])
    assert c["name"].tolist() == ["ring", "", ""]
    t = RegionTable()
    t.set("a", {"values": {"n": 1}})
    t.set("b", {"values": {"n": 2}})
    assert t.columns()["n"].dtype == np.int64          # ints stay ints while whole


def test_keeping_only_some_regions_drops_the_rest_and_the_steps_they_used():
    t = a_table()
    t.keep_only(["r1", "r3"])
    assert t.ids == ["r1", "r3"] and t.steps == [STEP]
    assert t.entry("r3")["error"] and t.step("r1") == STEP


def test_a_table_survives_its_dict():
    t = a_table()
    back = RegionTable.from_dict(t.to_dict())
    for r in t.ids:
        assert back.entry(r) == t.entry(r) or r == "r2"
        assert back.step(r) == t.step(r)
    assert np.isnan(back.entry("r2")["values"]["mean"])


def test_a_table_survives_hdf5_with_its_kinds_and_its_ragged_arrays(tmp_path):
    t = a_table()
    with h5py.File(tmp_path / "t.h5", "w") as f:
        t.to_hdf5(f.create_group("stats"))
    with h5py.File(tmp_path / "t.h5", "r") as f:
        back = RegionTable.from_hdf5(f["stats"])
    assert back.label == "stats" and back.ids == t.ids
    one = back.entry("r1")
    assert one["values"] == {"n": 3, "mean": 2.5, "name": "ring", "ok": True}
    assert type(one["values"]["n"]) is int and type(one["values"]["ok"]) is bool
    assert one["arrays"] == {"profile": [1.0, 2.0, 3.0]}
    assert back.entry("r2")["arrays"] == {"profile": [4.0]}
    assert back.entry("r3") == t.entry("r3")
    assert back.step("r2") == OTHER and back.data_signature("r1") == "d1"


def test_a_hundred_thousand_rows_cost_what_their_numbers_do(tmp_path):
    t = RegionTable("clusters")
    for i in range(100_000):
        t.set(i, {"values": {"n": i % 50, "mean_photons": 1000.0 + i}})
    with h5py.File(tmp_path / "big.h5", "w") as f:
        t.to_hdf5(f.create_group("clusters"))
    size = (tmp_path / "big.h5").stat().st_size
    assert size < 5_000_000                             # not the 130 MB of JSON records


def test_a_saved_table_with_old_value_names_reads_with_the_new_and_old_names_still_find_them():
    from smappy.region_table import current_name
    from smappy.roi_manager.order import condition, sort_ids
    old = RegionTable("Localization Statistics")
    old.set("1", {"values": {"n_localizations": 7, "median_photons": 300.0}})
    back = RegionTable.from_dict(old.to_dict())
    assert set(back.values) == {"n_locs", "photons_median"}
    assert current_name("Blinking.n_localizations") == "Blinking.n_locs"
    assert current_name("cfr_median") == "cfr_median"
    test, names = condition("n_localizations > 5 and median_photons < 400")
    assert names == ["n_locs", "photons_median"]
    assert test({"n_locs": 7, "photons_median": 300.0})
    assert sort_ids({"a": {"n_locs": 2}, "b": {"n_locs": 9}}, [("n_localizations", True)]) == ["b", "a"]


def test_a_measurement_over_the_rois_takes_several_selected_ones_and_never_a_single_one():
    from test_region_statistics import _session_with_rois
    from smappy.region_runs import ROIRegions
    from smappy.gui.region_steps import run_text
    session, project, rois, _ = _session_with_rois()
    every = [r.id for r in rois]
    project.selected = [rois[0].id]                 # the ROI being looked at
    assert ROIRegions(project).ids == every and run_text(project) == "Measure all"
    project.selected = [rois[0].id, rois[2].id]
    assert ROIRegions(project).ids == [rois[0].id, rois[2].id]
    assert run_text(project) == "Measure 2 selected"
    rois[2].use = False                             # use still decides
    assert ROIRegions(project).ids == [rois[0].id]
