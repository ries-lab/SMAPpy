"""Clusters as regions: DBSCAN's cluster_id and ROIs, and what was measured
per region written back onto the table."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.mathparse import recipes
from smappy.plugins.clusters import DBSCAN, auto_eps, dbscan, sizes_of
from smappy.plugins.region_column import (ColumnToROIs, RegionsToColumn, column_regions,
                                          hull, roi_numbers)
from smappy.region_runs import values_onto
from smappy.session import Session

SIZES = (60, 40, 25)


def blobs(sizes=SIZES, noise=30, seed=0, spread=15.0):
    """Tight clusters of known sizes far apart, and sparse noise."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal((k + 1) * 3000, spread, (n, 2)) for k, n in enumerate(sizes)]
    parts.append(rng.uniform(0, 12000, (noise, 2)) + [0, 20000])   # well away
    xy = np.vstack(parts)
    truth = np.concatenate([np.full(n, k + 1) for k, n in enumerate(sizes)]
                           + [np.zeros(noise, int)])
    return xy, truth


def table(xy, **extra):
    n = len(xy)
    rng = np.random.default_rng(1)
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(n, dtype=np.int64),
                          "photons": rng.gamma(4, 400, n),
                          "xy_err_nm": rng.uniform(5, 15, n), **extra}, {})


def test_dbscan_finds_the_clusters_numbered_by_size_and_leaves_the_noise():
    xy, truth = blobs()
    labels = dbscan(xy, eps=40, k=5)
    for k in (1, 2, 3):
        assert set(labels[truth == k]) == {k}           # each blob one cluster
    assert (labels[truth == 0] == 0).all()
    assert sizes_of(labels)[truth == 1][0] == 60 and sizes_of(labels)[truth == 0][0] == 0


def test_a_border_point_joins_its_nearest_core_point_and_a_lone_one_is_noise():
    # two dense groups, a point between them nearer the right one, one far off
    left = np.column_stack((np.zeros(6), np.arange(6) * 1.0))
    right = left + [10, 0]
    border = [[6.5, 2.5]]
    lone = [[100, 100]]
    labels = dbscan(np.vstack([left, right, border, lone]), eps=4, k=3)
    assert labels[12] == labels[6] != labels[0]          # nearer the right group
    assert labels[13] == 0


def test_the_automatic_radius_is_smaps():
    from math import gamma, pi
    points = np.random.default_rng(2).uniform(0, 1000, (500, 2))
    volume = np.prod(np.ptp(points, axis=0))
    expected = (volume * 10 * gamma(2) / (500 * pi)) ** 0.5
    assert auto_eps(points, 10) == pytest.approx(expected)


def test_the_plugin_writes_cluster_id_and_its_size_with_a_rule_for_blinks():
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    result = session.run(DBSCAN(), DBSCAN.Settings(radius_nm=40, neighbours=5))
    ids = np.asarray(session.locs["cluster_id"])
    assert np.array_equal(ids, truth)
    assert np.asarray(session.locs["n_in_cluster"])[truth == 2][0] == 40
    rules = {r["field"]: r["grouped"] for r in recipes(session.locs)}
    assert rules["cluster_id"] == "max" and rules["n_in_cluster"] == "max"
    assert result.text.startswith("3 clusters of 125 localizations")


def test_only_the_selected_localizations_are_clustered_and_each_file_alone():
    xy, truth = blobs()
    files = np.zeros(len(xy), dtype=np.int64)
    files[truth == 2] = 1                               # the second blob, another file
    session = Session(table(xy, filenumber=files))
    session.show_grouped(0, False)
    session.layers[0].set_bound("photons", 1500.0, None)
    session.run(DBSCAN(), DBSCAN.Settings(radius_nm=40, neighbours=3))
    ids = np.asarray(session.locs["cluster_id"])
    dim = np.asarray(session.locs["photons"]) < 1500
    assert (ids[dim] == 0).all()                        # filtered out: in none
    assert set(ids[(files == 1) & (ids > 0)]).isdisjoint(set(ids[(files == 0) & (ids > 0)]))


def test_on_a_grouped_layer_the_blinks_are_clustered():
    xy, truth = blobs()
    xy = np.repeat(xy, 2, axis=0)                       # each spot on two frames
    truth = np.repeat(truth, 2)
    locs = table(xy)
    locs.columns["frame"] = np.tile([0, 1], len(xy) // 2) + np.repeat(
        np.arange(len(xy) // 2) * 10, 2)
    session = Session(locs)
    session.show_grouped(0, True)
    session.run(DBSCAN(), DBSCAN.Settings(radius_nm=40, neighbours=5))
    assert np.array_equal(np.asarray(session.locs["cluster_id"]), truth)


def test_an_roi_per_cluster_outlines_it_and_replace_makes_them_again():
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    session.run(DBSCAN(), DBSCAN.Settings(radius_nm=40, neighbours=5))
    result = session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=30))
    project = session.rois
    assert len(project.rois) == 2 and result.data["rois"] == 2   # 25 < 30
    for roi in project.rois.values():
        inside = project.region(roi).mask(xy[:, 0], xy[:, 1])
        k = roi.origin["cluster"]
        assert inside[truth == k].all()                 # the hull holds its cluster
    session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=1, replace=True))
    assert len(project.rois) == 3
    session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=1))
    assert len(project.rois) == 6                       # without replace, beside them


def test_any_id_column_makes_rois_and_a_box_takes_the_projects_shape():
    xy, truth = blobs()
    session = Session(table(xy, mine=truth * 7))
    session.show_grouped(0, False)
    session.rois.set_geometry(400)
    session.run(ColumnToROIs(), ColumnToROIs.Settings(column="mine", shape="box"))
    rois = list(session.rois.rois.values())
    assert sorted(r.origin["cluster"] for r in rois) == [7, 14, 21]
    assert all(r.polygon is None for r in rois)
    first = next(r for r in rois if r.origin["cluster"] == 7)
    assert np.allclose(first.center, xy[truth == 1].mean(axis=0))


def test_column_regions_refuses_a_column_that_is_not_an_id():
    xy, truth = blobs()
    with pytest.raises(ValueError, match="not an id column"):
        column_regions(table(xy), "photons")
    with pytest.raises(ValueError, match="has no 'cluster_id'"):
        column_regions(table(xy), "cluster_id")


def test_rois_cannot_be_made_inside_a_block():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    cls = chain_class(ChainSpec(steps=[Block(regions="cluster_id", steps=[
        Step(plugin="Analysis/Regions/Column to ROIs")])]))
    result = session.run(cls(), cls.Settings())
    table_ = session.id_regions["cluster_id"]["Column to ROIs"]
    assert all("cannot be made inside a block" in e for e in table_.errors)


def test_a_block_repeats_over_the_clusters_dbscan_wrote():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    cls = chain_class(ChainSpec(steps=[
        Step(plugin="Analysis/Regions/Find/DBSCAN", values={"radius_nm": 40,
                                                            "neighbours": 5}),
        Block(regions="cluster_id", steps=[
            Step(plugin="Analysis/Measure/Localization Statistics")])]))
    session.run(cls(), cls.Settings())
    stats = session.id_regions["cluster_id"]["Localization Statistics"]
    assert stats.columns()["n_localizations"].tolist() == list(SIZES)


def test_hull_needs_three_points_off_a_line():
    assert hull(np.array([[0, 0], [1, 1]])) is None
    assert hull(np.array([[0, 0], [1, 1], [2, 2]])) is None
    assert len(hull(np.array([[0, 0], [1, 0], [0, 1], [0.2, 0.2]]))) == 3


def test_region_values_go_onto_their_rows_by_the_overlap_rule():
    members = [("1", np.array([0, 1, 2, 3])), ("2", np.array([2, 3, 4]))]
    values = {"v": {"1": 10.0, "2": 20.0}}
    assert np.allclose(values_onto(6, members, values, "last")["v"][:5], [10, 10, 20, 20, 20])
    assert np.allclose(values_onto(6, members, values, "first")["v"][:5], [10, 10, 10, 10, 20])
    assert np.allclose(values_onto(6, members, values, "mean")["v"][:5], [10, 10, 15, 15, 20])
    assert np.isnan(values_onto(6, members, values)["v"][5])          # in no region
    assert np.isnan(values_onto(6, members, {"v": {"1": 1.0}})["v"][4])  # no value


def test_regions_to_column_writes_the_later_roi_number_where_two_overlap():
    assert roi_numbers(6, [("a", np.array([0, 1, 2, 3])), ("b", np.array([2, 3, 4]))],
                       {"a": 1, "b": 2}).tolist() == [1, 1, 2, 2, 2, 0]
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(400)
    file_id = next(iter(project.sources))
    first = project.add_roi(file_id, [3000.0, 3000.0])
    second = project.add_roi(file_id, [3000.0, 3000.0],
                             polygon=[[3000, 2900], [3200, 2900], [3200, 3100],
                                      [3000, 3100]])
    excluded = project.add_roi(file_id, [6000.0, 6000.0])
    excluded.use = False
    session.run(RegionsToColumn(), RegionsToColumn.Settings())
    ids = np.asarray(session.locs["roi_id"])
    numbers = project.numbers()
    in_second = project.region(second).mask(xy[:, 0], xy[:, 1])
    assert (ids[in_second] == numbers[second.id]).all()
    assert (ids[(truth == 1) & ~in_second] == numbers[first.id]).all()
    assert (ids[truth == 2] == 0).all()                 # its ROI is excluded
    assert {r["field"]: r["grouped"] for r in recipes(session.locs)}["roi_id"] == "max"


def test_histograms_summarise_the_clusters_too():
    from smappy import plugins
    from smappy.chain import Block, ChainSpec, Step, chain_class
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    cls = chain_class(ChainSpec(steps=[Block(regions="cluster_id", steps=[
        Step(plugin="Analysis/Measure/Localization Statistics")])]))
    session.run(cls(), cls.Settings())
    histograms = plugins.get("Analysis/Regions/Summarize/Histograms")
    offered = histograms().choices(session.context(), histograms.Settings())["evaluation"]
    name = "per cluster_id: Localization Statistics"
    assert (name, name) in offered
    result = histograms()(ctx=session.context(), evaluation=name)
    assert result.text.startswith("3 regions")
    assert sorted(result.data["histograms"]["n_localizations"]["values"]) == sorted(SIZES)


def clustered_session():
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    return session, truth


def test_a_cluster_statistic_goes_onto_its_localizations_and_filters_them():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    session, truth = clustered_session()
    cls = chain_class(ChainSpec(steps=[Block(regions="cluster_id", steps=[
        Step(plugin="Analysis/Measure/Localization Statistics")])]))
    session.run(cls(), cls.Settings())
    name = "per cluster_id: Localization Statistics"
    offered = RegionsToColumn().choices(session.context(), RegionsToColumn.Settings())
    assert (name, name) in offered["source"]
    session.run(RegionsToColumn(), RegionsToColumn.Settings(
        source=name, values="n_localizations", prefix="cluster_"))
    column = np.asarray(session.locs["cluster_n_localizations"])
    for k, n in enumerate(SIZES, start=1):
        assert (column[truth == k] == n).all()
    assert np.isnan(column[truth == 0]).all()
    assert {r["field"]: r["grouped"] for r in recipes(session.locs)}[
        "cluster_n_localizations"] == "mean"


def test_a_value_is_never_written_over_a_column_of_the_localizations_own():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    session, truth = clustered_session()
    session.locs.columns["n_localizations"] = np.zeros(len(session.locs))
    cls = chain_class(ChainSpec(steps=[Block(regions="cluster_id", steps=[
        Step(plugin="Analysis/Measure/Localization Statistics")])]))
    session.run(cls(), cls.Settings())
    with pytest.raises(ValueError, match="not written from regions"):
        RegionsToColumn()(ctx=session.context(),
                          source="per cluster_id: Localization Statistics",
                          values="n_localizations")


def test_a_block_writes_the_values_it_measured_and_again_over_them():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    session, truth = clustered_session()
    cls = chain_class(ChainSpec(steps=[Block(
        regions="cluster_id", write_values=["n_localizations"],
        steps=[Step(plugin="Analysis/Measure/Localization Statistics")])]))
    for _ in range(2):                      # the second run writes over the first
        session.run(cls(), cls.Settings())
    column = np.asarray(session.locs["n_localizations"])
    assert (column[truth == 2] == SIZES[1]).all() and np.isnan(column[truth == 0]).all()
    block = cls.spec.steps[0]
    assert Block.from_dict(block.to_dict()).write_values == ["n_localizations"]


def test_a_block_refuses_to_write_a_value_no_step_measured():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    session, truth = clustered_session()
    cls = chain_class(ChainSpec(steps=[Block(
        regions="cluster_id", write_values=["nothing_like_it"],
        steps=[Step(plugin="Analysis/Measure/Localization Statistics")])]))
    with pytest.raises(ValueError, match="no step measured nothing_like_it"):
        cls()(ctx=session.context())


def test_an_roi_evaluation_goes_onto_the_localizations_in_each_roi():
    from smappy.chain import Block, ChainSpec, Step, chain_class
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(400)
    file_id = next(iter(project.sources))
    for k in (1, 2):
        project.add_roi(file_id, [3000.0 * k, 3000.0 * k])
    cls = chain_class(ChainSpec(steps=[Block(steps=[
        Step(plugin="Analysis/Measure/Localization Statistics")])]))
    session.run(cls(), cls.Settings())
    settings = RegionsToColumn.Settings(source="Localization Statistics",
                                        values="n_localizations, roi_id")
    assert ("Localization Statistics",) * 2 in RegionsToColumn().choices(
        session.context(), settings)["source"]
    session.run(RegionsToColumn(), settings)
    column = np.asarray(session.locs["n_localizations"])
    assert (column[truth == 1] == SIZES[0]).all() and (column[truth == 2] == SIZES[1]).all()
    assert np.isnan(column[truth == 3]).all()
    assert set(np.asarray(session.locs["roi_id"])[truth == 2]) == {2}
