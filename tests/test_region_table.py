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
