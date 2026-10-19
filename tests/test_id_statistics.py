"""Statistics per id (`smappy.id_statistics`, *Id statistics*): every trace,
track or cluster at once, kept as a region table that reads as a block's."""
import time

import h5py
import numpy as np
import pytest

from smappy import plugins
from smappy.id_statistics import dense_codes, id_statistics, stat_name
from smappy.locs import Localizations
from smappy.region_results import Evaluation, available, is_current, rows
from smappy.region_runs import split_ids
from smappy.region_table import RegionTable
from smappy.session import Session

PLUGIN = "Analysis/Measure/Id statistics"
EV = Evaluation("tid", "Id statistics")


def traces(n_traces=400, sigma=(4.0, 6.0, 9.0), seed=0):
    """MINFLUX-like traces: each a molecule localized 5-40 times around its
    own position with a known spread per axis, one after the other in time,
    and a cfr that is NaN where it was not measured."""
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 41, n_traces)
    tid = np.repeat(np.arange(1, n_traces + 1), n)
    # z inside the slab a new session shows (+-500 nm)
    centre = rng.uniform(0, 20000, (n_traces, 3)) * np.array([1, 1, 0.02]) - [0, 0, 200]
    xyz = centre[tid - 1] + rng.normal(size=(len(tid), 3)) * np.array(sigma)
    time_s = np.cumsum(rng.uniform(0.001, 0.003, len(tid)))
    cfr = rng.uniform(0.1, 0.5, len(tid))
    cfr[rng.random(len(tid)) < 0.2] = np.nan
    locs = Localizations({"x_nm": xyz[:, 0], "y_nm": xyz[:, 1], "z_nm": xyz[:, 2],
                          "frame": np.arange(len(tid)), "time_s": time_s,
                          "photons": rng.uniform(500, 2000, len(tid)),
                          "cfr": cfr.astype(np.float32), "tid": tid.astype(np.int64),
                          "filenumber": np.ones(len(tid), np.int32)}, {"units": "nm"})
    return locs, n, centre


def test_the_spread_count_and_duration_of_each_trace_come_back():
    locs, n, centre = traces(n_traces=2000)
    found = id_statistics(locs["tid"], {"x": locs["x_nm"], "y": locs["y_nm"],
                                        "z": locs["z_nm"]},
                          time_s=locs["time_s"], columns={"cfr": locs["cfr"]})
    v = found.values
    assert list(found.ids) == list(range(1, 2001))
    assert np.array_equal(v["n_locs"], n)
    # the pooled sample variance is unbiased (ddof 1): its mean is sigma^2
    for axis, sigma in zip("xyz", (4.0, 6.0, 9.0)):
        assert np.sqrt(np.mean(v[f"std_{axis}_nm"] ** 2)) == pytest.approx(sigma, rel=0.03)
        assert np.abs(v[f"centre_{axis}_nm"] - centre[:, "xyz".index(axis)]).max() \
            < 5 * sigma
    assert np.sqrt(np.mean(v["std_xy_nm"] ** 2)) == pytest.approx(
        np.sqrt((16 + 36) / 2), rel=0.03)
    t = np.asarray(locs["time_s"])
    first = np.r_[0, np.cumsum(n)[:-1]]
    assert np.allclose(v["duration_s"], t[first + n - 1] - t[first])
    assert set(found.groups["cfr"]) == {"mean", "median", "max"}


def test_column_statistics_skip_unmeasured_values_and_match_numpy():
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 300, 20000)
    value = rng.normal(size=20000)
    value[rng.random(20000) < 0.3] = np.nan
    value[ids == 7] = np.nan                    # a region with nothing measured
    found = id_statistics(ids, {"x": value * 0, "y": value * 0},
                          columns={"v": value}, stats=("mean", "median", "std", "min", "max"))
    for stat, ref in (("mean", np.nanmean), ("median", np.nanmedian), ("min", np.nanmin),
                      ("max", np.nanmax), ("std", lambda a: np.nanstd(a, ddof=1))):
        expect = []
        for i in found.ids:
            a = value[ids == i]
            enough = np.isfinite(a).sum() > (1 if stat == "std" else 0)
            expect.append(ref(a) if enough else np.nan)
        assert np.allclose(found.values[f"v_{stat}"], expect, equal_nan=True), stat
    assert 0 not in found.ids                   # 0 is no region
    assert np.isnan(found.values["v_mean"][list(found.ids).index(7)])


def test_the_kept_rows_alone_are_measured_and_a_region_with_none_is_left_out():
    ids = np.array([1, 1, 1, 2, 2, 3])
    x = np.array([0.0, 2.0, 100.0, 5.0, 7.0, 9.0])
    keep = np.array([True, True, False, True, True, False])
    found = id_statistics(ids, {"x": x, "y": x}, keep=keep)
    assert list(found.ids) == [1, 2]
    assert list(found.values["n_locs"]) == [2, 2]
    assert list(found.values["centre_x_nm"]) == [1.0, 6.0]


def test_large_and_float_ids_name_the_same_regions():
    ids = np.array([5, 10 ** 12, 5, 10 ** 12, 0])
    found, codes = dense_codes(ids)
    assert list(found) == [5, 10 ** 12] and list(codes) == [0, 1, 0, 1, -1]
    found, codes = dense_codes(np.array([2.0, np.nan, 2.0, 3.0]))
    assert list(found) == [2, 3] and list(codes) == [0, -1, 0, 1]


def test_a_statistic_goes_before_the_unit():
    assert stat_name("cfr", "mean") == "cfr_mean"
    assert stat_name("xy_err_nm", "median") == "xy_err_median_nm"
    assert stat_name("photons", "max") == "photons_max"


def test_the_plugin_keeps_a_table_that_reads_as_a_blocks_and_goes_stale_with_the_filter():
    locs, n, _ = traces()
    session = Session(locs)
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    result = session.run(cls(), cls.Settings())
    ctx = session.context()
    assert EV in available(ctx, PLUGIN) and is_current(ctx, EV)
    got = {int(r["roi_id"]): r for r in rows(ctx, EV)}
    assert len(got) == len(n) and got[1]["n_locs"] == n[0]
    assert "cfr_median" in got[1] and "std_xy_nm" in got[1]
    assert session.id_regions["tid"]["Id statistics"].groups["cfr"]["median"] == "cfr_median"
    assert "regions of tid" in result.text
    # a bound on a layer changes what the regions hold: the table no longer
    # describes them, as a block's would not
    session.layers[0].set_bound("photons", 1000, None)
    assert not is_current(session.context(), EV)
    # and run again, it is current, measuring only what the filter keeps
    session.run(cls(), cls.Settings())
    ctx = session.context()
    assert is_current(ctx, EV)
    kept = np.asarray(locs["photons"]) >= 1000
    assert sum(r["n_locs"] for r in rows(ctx, EV)) == int(kept.sum())


def test_the_table_survives_a_save_and_is_current_after_loading(tmp_path):
    locs, _, _ = traces(n_traces=50)
    session = Session(locs)
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    session.run(cls(), cls.Settings())
    path = session.save(tmp_path / "traces.h5")
    again = Session()
    again.load(path)
    again.show_grouped(0, False)
    ctx = again.context()
    assert is_current(ctx, EV)
    assert len(rows(ctx, EV)) == 50
    assert again.id_regions["tid"]["Id statistics"].groups["cfr"]["mean"] == "cfr_mean"


def test_a_table_of_a_million_regions_is_small_and_quick_to_write_and_read(tmp_path):
    n = 10 ** 6
    values = {k: np.random.default_rng(0).random(n)
              for k in ("n_locs", "std_x_nm", "std_y_nm", "duration_s")}
    start = time.perf_counter()
    table = RegionTable.from_columns("Id statistics", np.arange(1, n + 1), values,
                                     {"plugin": PLUGIN}, "a" * 64, "b" * 64)
    with h5py.File(tmp_path / "t.h5", "w") as f:
        table.to_hdf5(f.create_group("g"))
    with h5py.File(tmp_path / "t.h5") as f:
        back = RegionTable.from_hdf5(f["g"])
    elapsed = time.perf_counter() - start
    # four values a region and the ids are 40 MB; a signature written per
    # row made it 236 MB
    assert (tmp_path / "t.h5").stat().st_size < 60e6
    assert elapsed < 30
    assert back.ids[-1] == str(n) and back.data[0] == "b" * 64
    assert back.values["std_x_nm"][123] == values["std_x_nm"][123]


def test_each_regions_rows_stay_in_table_order():
    ids, rows_ = split_ids(np.array([2, 0, 1, 2, -1, 1, 2, 1]))
    assert list(ids) == [1, 2]
    assert [list(r) for r in rows_] == [[2, 5, 7], [0, 3, 6]]
