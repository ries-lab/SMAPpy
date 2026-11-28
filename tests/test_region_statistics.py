"""Statistics per region (`smappy.id_statistics`, *Region statistics*):
every ROI, trace, track or cluster at once, kept as a region table that
reads as a block's."""
import time

import h5py
import numpy as np
import pytest

from smappy import plugins
from smappy.id_statistics import dense_codes, id_statistics, member_statistics, stat_name
from smappy.locs import Localizations
from smappy.region_results import Evaluation, available, is_current, rows
from smappy.region_runs import split_ids
from smappy.region_table import RegionTable
from smappy.session import Session

PLUGIN = "Analysis/Measure/Region statistics"
EV = Evaluation("tid", "Region statistics")


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
    ids = rng.integers(-1, 300, 20000)
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
    assert -1 not in found.ids and 0 in found.ids   # -1 is no region, 0 is one
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
    ids = np.array([5, 10 ** 12, 5, 10 ** 12, -1])
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
    assert session.id_regions["tid"]["Region statistics"].groups["cfr"]["median"] == "cfr_median"
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
    assert again.id_regions["tid"]["Region statistics"].groups["cfr"]["mean"] == "cfr_mean"


def test_a_table_of_a_million_regions_is_small_and_quick_to_write_and_read(tmp_path):
    n = 10 ** 6
    values = {k: np.random.default_rng(0).random(n)
              for k in ("n_locs", "std_x_nm", "std_y_nm", "duration_s")}
    start = time.perf_counter()
    table = RegionTable.from_columns("Region statistics", np.arange(1, n + 1), values,
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
    assert list(ids) == [0, 1, 2]                   # 0 is a region, -1 none
    assert [list(r) for r in rows_] == [[1], [2, 5, 7], [0, 3, 6]]


def gaussian_clusters(n_clusters=100, seed=7, three_d=False):
    """Elongated Gaussian clusters, each with its own angle (0-180 degrees
    from x) and standard deviations along and across it, in random order of
    rows, a few singletons and pairs among them."""
    rng = np.random.default_rng(seed)
    n = rng.integers(800, 1200, n_clusters)
    n[:3] = (1, 2, 3)                           # too small for a std or a hull
    angle = rng.uniform(0, 180, n_clusters)
    long = rng.uniform(30, 80, n_clusters)
    short = long / rng.uniform(2.5, 4.0, n_clusters)
    ids = np.repeat(np.arange(n_clusters), n)
    u = rng.normal(size=len(ids)) * long[ids]
    v = rng.normal(size=len(ids)) * short[ids]
    t = np.radians(angle[ids])
    centre = rng.uniform(0, 30000, (n_clusters, 2))
    x = centre[ids, 0] + u * np.cos(t) - v * np.sin(t)
    y = centre[ids, 1] + u * np.sin(t) + v * np.cos(t)
    z = rng.normal(size=len(ids)) * 40.0 if three_d else None
    order = rng.permutation(len(ids))
    return (ids[order], x[order], y[order], None if z is None else z[order],
            angle, long, short)


def test_shape_finds_the_angle_and_the_axes_of_elongated_gaussian_clusters():
    ids, x, y, _, angle, long, short = gaussian_clusters()
    found = id_statistics(ids, {"x": x, "y": y}, shape=True)
    v = found.values
    big = v["n_locs"] >= 800
    # unoriented, in [0, 180): the difference is taken modulo 180
    diff = (v["angle_deg"][big] - angle[big] + 90) % 180 - 90
    assert np.all((v["angle_deg"] >= 0) & (v["angle_deg"] < 180) | np.isnan(v["angle_deg"]))
    assert np.abs(diff).max() < 5
    assert np.abs(v["std_long_nm"][big] / long[big] - 1).max() < 0.1
    assert np.abs(v["std_short_nm"][big] / short[big] - 1).max() < 0.1
    assert np.allclose(v["elongation"][big], v["std_long_nm"][big] / v["std_short_nm"][big])
    # a Gaussian's radius of gyration is the RMS of the two axes (ddof 0)
    assert np.median(v["rg_nm"][big] / np.hypot(long[big], short[big])) == \
        pytest.approx(1, abs=0.03)
    # one point: no std, no hull; two: a std but no hull; three: a triangle
    assert np.isnan(v["std_long_nm"][0]) and v["area_nm2"][0] == 0
    assert np.isfinite(v["std_long_nm"][1]) and v["area_nm2"][1] == 0
    assert v["area_nm2"][2] > 0 and np.isnan(v["density_per_um2"][0])


def test_shape_values_equal_cluster_shapes_for_the_same_clusters():
    from smappy.plugins.cluster_shape import NAMES_2D, NAMES_3D, cluster_shape
    ids, x, y, z, *_ = gaussian_clusters(n_clusters=40, three_d=True)
    x[5] = np.nan                     # left out of the shape, as Cluster shape does
    found = id_statistics(ids, {"x": x, "y": y, "z": z}, shape=True)
    for k, i in enumerate(found.ids):
        rows_ = (ids == i) & np.isfinite(x)
        expect = cluster_shape(x[rows_], y[rows_], z[rows_])
        for name in NAMES_2D + NAMES_3D:
            if name in ("n_locs", "centre_x_nm", "centre_y_nm", "centre_z_nm"):
                continue              # Id statistics' own: per coordinate's finite values
            assert np.array_equal(found.values[name][k], expect[name], equal_nan=True), \
                (i, name)


def test_shape_off_adds_no_columns_and_the_plugin_adds_them_when_ticked():
    from smappy.id_statistics import SHAPE_2D, SHAPE_3D
    ids, x, y, _, *_ = gaussian_clusters(n_clusters=20)
    plain = id_statistics(ids, {"x": x, "y": y})
    assert not set(SHAPE_2D + SHAPE_3D) & set(plain.values)
    shaped = id_statistics(ids, {"x": x, "y": y}, shape=True)
    assert set(shaped.values) == set(plain.values) | set(SHAPE_2D)
    for name in plain.values:
        assert np.array_equal(plain.values[name], shaped.values[name], equal_nan=True)
    locs, _, _ = traces(n_traces=50)
    session = Session(locs)
    session.show_grouped(0, False)
    cls = plugins.get(PLUGIN)
    assert cls.Settings().shape is False
    session.run(cls(), cls.Settings())
    assert "area_nm2" not in session.id_regions["tid"]["Region statistics"].columns()
    result = session.run(cls(), cls.Settings(shape=True))
    columns = session.id_regions["tid"]["Region statistics"].columns()
    assert set(SHAPE_2D + SHAPE_3D) <= set(columns)
    assert "shape:" in result.text


# ------------------------------------------------------------- the ROIs

def test_rois_given_as_rows_give_the_numbers_of_an_id_column_holding_them():
    rng = np.random.default_rng(4)
    ids = rng.integers(0, 60, 5000)
    x, y = rng.normal(size=5000) * 10 + ids * 200, rng.normal(size=5000) * 5
    p = rng.uniform(100, 900, 5000)
    keep = rng.random(5000) > 0.1
    options = dict(columns={"photons": p}, stats=("mean", "median", "std", "min", "max"),
                   shape=True, frame=np.arange(5000), keep=keep)
    by_id = id_statistics(ids, {"x": x, "y": y}, **options)
    members = [(f"r{i}", np.flatnonzero(ids == i)) for i in range(60)]
    by_rows = member_statistics(members, {"x": x, "y": y}, **options)
    assert list(by_rows.ids) == [f"r{i}" for i in by_id.ids]
    for name, value in by_id.values.items():
        assert np.allclose(by_rows.values[name], value, equal_nan=True), name


def test_a_localization_in_two_overlapping_rois_counts_in_both_and_an_empty_roi_is_left_out():
    x = np.arange(10.0)
    members = [("a", np.arange(0, 6)), ("b", np.arange(4, 10)), ("empty", np.arange(0))]
    found = member_statistics(members, {"x": x, "y": x * 0})
    assert list(found.ids) == ["a", "b"]
    assert found.values["n_locs"].tolist() == [6, 6]
    assert found.values["centre_x_nm"].tolist() == [2.5, 6.5]


def _session_with_rois():
    from test_clusters import blobs, table
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    project = session.rois
    file_id = next(iter(project.sources))
    rois = [project.add_roi(file_id, list(xy[truth == k].mean(0))) for k in (1, 2, 3)]
    return session, project, rois, truth


def test_the_rois_are_measured_at_once_and_kept_as_current_roi_results():
    from test_clusters import SIZES
    session, project, rois, truth = _session_with_rois()
    cls = plugins.get(PLUGIN)
    session.run(cls(), cls.Settings(regions="rois"))
    got = {str(r["roi_id"]): r for r in rows(session.context(), Evaluation("rois", "Region statistics"))}
    assert sorted(int(r["n_locs"]) for r in got.values()) == sorted(SIZES)
    # each row current with its own ROI, as a per-region step's would be
    assert all("Region statistics" in project.evaluations(r.id) for r in rois)
    project.move_roi(rois[0].id, np.asarray(rois[0].center) + 5000.0)
    assert "Region statistics" not in project.evaluations(rois[0].id)
    assert "Region statistics" in project.evaluations(rois[1].id)


def test_auto_takes_an_id_column_before_the_rois_and_the_rois_without_one():
    from smappy.plugins.region_statistics import ROIS, regions_source
    session, project, rois, truth = _session_with_rois()
    assert regions_source(session.context(), "") == "cluster_id"
    assert regions_source(session.context(), ROIS) == ROIS
    session.locs.columns.pop("cluster_id")
    assert regions_source(session.context(), "") == ROIS


def test_id_statistics_saved_settings_read_as_region_statistics():
    path, values = plugins.moved("Analysis/Measure/Id statistics",
                                 {"id_column": "tid", "columns": "cfr, efo", "shape": True})
    assert path == PLUGIN
    assert values == {"regions": "tid", "columns": ["cfr", "efo"], "common": False,
                      "shape": True}
    assert plugins.moved("Analysis/Measure/Id statistics", {"columns": ""})[1] == \
        {"columns": [], "common": True}
    assert plugins.get("Analysis/Measure/Id statistics").path == PLUGIN


def test_the_common_attributes_and_the_ticked_ones_are_summarised_each_once():
    from smappy.plugins.region_statistics import offered_columns, summarised_columns
    locs, _, _ = traces(50)
    assert summarised_columns(locs, [], True) == ["cfr", "photons"]
    assert summarised_columns(locs, ["z_nm", "cfr"], True) == ["cfr", "photons", "z_nm"]
    assert summarised_columns(locs, ["cfr"], False) == ["cfr"]
    assert summarised_columns(locs, "cfr, photons", False) == ["cfr", "photons"]
    with pytest.raises(ValueError, match="no attribute efo"):
        summarised_columns(locs, ["efo"], True)
    offered = offered_columns(locs)
    assert "cfr" in offered and "photons" in offered
    assert not {"x_nm", "frame", "time_s", "tid"} & set(offered)
