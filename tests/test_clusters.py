"""Clusters as regions: Find clusters' cluster_id (DBSCAN, Voronoi, HDBSCAN)
and ROIs, and what was measured per region written back onto the table."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.mathparse import recipes
from smappy.plugins.clusters import (DBSCANSettings, FindClusters, auto_eps, dbscan,
                                     sizes_of)
from smappy.plugins.region_column import (ColumnToROIs, RegionsToColumn, column_regions,
                                          hull)
from smappy.region_runs import roi_numbers, values_onto
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


def by_dbscan(**values):
    """Find clusters' settings for DBSCAN with these values."""
    return FindClusters.Settings(dbscan=DBSCANSettings(**values))


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
    result = session.run(FindClusters(), by_dbscan(radius_nm=40, neighbours=5))
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
    session.run(FindClusters(), by_dbscan(radius_nm=40, neighbours=3))
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
    session.run(FindClusters(), by_dbscan(radius_nm=40, neighbours=5))
    assert np.array_equal(np.asarray(session.locs["cluster_id"]), truth)


def test_an_roi_per_cluster_outlines_it_and_replace_makes_them_again():
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    session.run(FindClusters(), by_dbscan(radius_nm=40, neighbours=5))
    result = session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=30))
    project = session.rois
    assert len(project.rois) == 2 and len(result.rois) == 2   # 25 < 30
    for roi in project.rois.values():
        inside = project.region(roi).mask(xy[:, 0], xy[:, 1])
        k = roi.origin["cluster"]
        assert inside[truth == k].all()                 # the hull holds its cluster
    session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=1))
    assert len(project.rois) == 3                       # replace: the default
    session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=1, replace=False))
    assert len(project.rois) == 6                       # without replace, beside them


def test_any_id_column_makes_rois_and_without_outline_only_the_centre():
    xy, truth = blobs()
    session = Session(table(xy, mine=truth * 7))
    session.show_grouped(0, False)
    session.rois.set_geometry(400)
    session.run(ColumnToROIs(), ColumnToROIs.Settings(column="mine", outline=False))
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
        Step(plugin="Analysis/Regions/Find clusters",
             values={"dbscan.radius_nm": 40, "dbscan.neighbours": 5}),
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
    offered = histograms().describe(session.context(), histograms.Settings()).choices["evaluation"]
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
    offered = RegionsToColumn().describe(session.context(),
                                         RegionsToColumn.Settings()).choices
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
    assert ("Localization Statistics",) * 2 in RegionsToColumn().describe(
        session.context(), settings).choices["source"]
    session.run(RegionsToColumn(), settings)
    column = np.asarray(session.locs["n_localizations"])
    assert (column[truth == 1] == SIZES[0]).all() and (column[truth == 2] == SIZES[1]).all()
    assert np.isnan(column[truth == 3]).all()
    assert set(np.asarray(session.locs["roi_id"])[truth == 2]) == {2}


def test_a_cluster_with_no_hull_still_gets_an_roi_at_its_centre():
    xy = np.column_stack((np.arange(12.0) * 10, np.arange(12.0) * 10))   # a line
    session = Session(table(xy, cluster_id=np.ones(12, dtype=np.int64)))
    session.show_grouped(0, False)
    result = session.run(ColumnToROIs(), ColumnToROIs.Settings())
    (roi,) = session.rois.rois.values()
    assert roi.polygon is None and np.allclose(roi.center, [55, 55])
    assert "1 with no outline" in result.text


def test_dbscan_reports_its_progress_and_a_stop_writes_nothing(monkeypatch):
    import threading
    from smappy.plugins import Stopped
    from smappy.plugins import clusters
    monkeypatch.setattr(clusters, "COUNT_CHUNK", 20)        # many steps
    xy, _ = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    settings = by_dbscan(radius_nm=40, neighbours=5)
    lines = []
    FindClusters().run(session.context(0, progress=lines.append), settings)
    percents = [int(line.rsplit(" ", 1)[1].rstrip("%")) for line in lines]
    assert len(percents) > 5 and percents == sorted(percents) and percents[-1] == 100

    stop = threading.Event()
    ctx = session.context(0, progress=lambda line: stop.set(), stop=stop)
    with pytest.raises(Stopped, match="nothing written"):
        FindClusters().run(ctx, settings)
    assert "cluster_id" not in session.locs


# ------------------------------------------------------- Voronoi and HDBSCAN

def clusters_in_background(n_clusters=25, per=50, spread=20.0, background=3000,
                           side=10_000.0, seed=3):
    """Gaussian clusters on a grid, far apart, in a uniform background."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(np.sqrt(n_clusters)))
    grid = (np.arange(k) + 0.5) * side / k
    centres = np.array([(x, y) for x in grid for y in grid])[:n_clusters]
    xy = np.vstack([rng.normal(c, spread, (per, 2)) for c in centres]
                   + [rng.uniform(0, side, (background, 2))])
    truth = np.concatenate([np.full(per, i + 1) for i in range(n_clusters)]
                           + [np.zeros(background, int)])
    return xy, truth


def agreement(labels, truth):
    """Per true cluster, the share of its points in its commonest found
    cluster; and the share of background points put in any."""
    found = []
    for t in np.unique(truth[truth > 0]):
        mine = labels[truth == t]
        found.append(np.bincount(mine[mine > 0]).max() / len(mine) if (mine > 0).any() else 0)
    return np.array(found), (labels[truth == 0] > 0).mean()


def test_voronoi_finds_simulated_clusters_in_a_background():
    from smappy.plugins.clusters import voronoi_clusters
    xy, truth = clusters_in_background()
    labels, mean = voronoi_clusters(xy, factor=2.0, min_size=10)
    from scipy.spatial import ConvexHull
    assert mean == pytest.approx(len(xy) / (ConvexHull(xy).volume / 1e6))
    assert labels.max() == 25                               # one each, none merged
    caught, stray = agreement(labels, truth)
    assert caught.min() > 0.8 and stray < 0.05
    # each found cluster is mostly one true cluster
    for c in range(1, labels.max() + 1):
        assert np.bincount(truth[labels == c]).max() / (labels == c).sum() > 0.8


def test_voronoi_drops_small_clusters_and_keeps_one_position_together():
    from smappy.plugins.clusters import voronoi_clusters
    xy, truth = clusters_in_background(per=12)
    few, _ = voronoi_clusters(xy, factor=2.0, min_size=20)
    assert few.max() == 0                                  # all below 20
    twice = np.vstack([xy, xy[:5]])                        # five at one position
    labels, _ = voronoi_clusters(twice, factor=2.0, min_size=5)
    assert (labels[:5] == labels[-5:]).all()


def test_the_plugin_runs_voronoi_and_names_the_clusters():
    xy, truth = clusters_in_background()
    session = Session(table(xy))
    session.show_grouped(0, False)
    result = session.run(FindClusters(), FindClusters.Settings(algorithm="voronoi"))
    ids = np.asarray(session.locs["cluster_id"])
    assert ids.max() == 25 and "Voronoi" in result.text
    sizes = np.bincount(ids)[1:]
    assert (np.diff(sizes) <= 0).all()                     # numbered by size
    assert {r["field"]: r["grouped"] for r in recipes(session.locs)}["cluster_id"] == "max"


def test_the_form_shows_only_the_chosen_algorithm():
    plugin = FindClusters()
    for algorithm in ("dbscan", "voronoi", "hdbscan"):
        shown = plugin.describe(None, FindClusters.Settings(algorithm=algorithm)).shown
        assert shown == {"dbscan": algorithm == "dbscan", "voronoi": algorithm == "voronoi",
                         "hdbscan": algorithm == "hdbscan"}


def test_without_scikit_learn_hdbscan_says_how_to_install_it(monkeypatch):
    from smappy.plugins import clusters
    monkeypatch.setattr(clusters, "hdbscan_available", lambda: False)
    settings = FindClusters.Settings(algorithm="hdbscan")
    notice = FindClusters().describe(None, settings).notice
    assert notice is not None and "smappy-smlm[clusters]" in notice.text
    xy, _ = blobs()
    session = Session(table(xy))
    with pytest.raises(ValueError, match=r"smappy-smlm\[clusters\]"):
        session.run(FindClusters(), settings)
    assert "cluster_id" not in session.locs


def _has_hdbscan():
    from smappy.plugins.clusters import hdbscan_available
    return hdbscan_available()


@pytest.mark.skipif(not _has_hdbscan(), reason="scikit-learn >= 1.3 is not installed")
def test_hdbscan_finds_clusters_of_different_densities_without_a_radius():
    from smappy.plugins.clusters import hdbscan_clusters
    rng = np.random.default_rng(4)
    tight = rng.normal((2000, 2000), 15, (80, 2))          # dense
    loose = rng.normal((6000, 2000), 120, (80, 2))         # eight times wider
    noise = rng.uniform(0, 8000, (60, 2)) + [0, 4000]      # well away
    xy = np.vstack([tight, loose, noise])
    truth = np.repeat([1, 2, 0], [80, 80, 60])
    labels = hdbscan_clusters(xy, min_cluster_size=20)
    caught, stray = agreement(labels, truth)
    # both found whole; but HDBSCAN lets scattered points join the cluster
    # they are nearest, here 14 of 60 -- which the page says
    assert labels.max() == 2 and caught.min() > 0.85 and stray < 0.25
    # one radius cannot do both: a DBSCAN radius for the tight one splits
    # the loose one or leaves most of it as noise
    narrow = dbscan(xy, 30.0, 10)
    assert agreement(narrow, truth)[0].min() < 0.85


def test_a_long_hdbscan_or_voronoi_run_asks_first():
    from smappy.plugins import Context
    from smappy.plugins.clusters import expected_seconds
    assert expected_seconds("hdbscan", [1_000_000]) > 3600      # hours
    assert expected_seconds("hdbscan", [20_000]) < 20
    xy, _ = blobs()
    ctx = Context(locs=table(xy))
    for algorithm in ("dbscan", "voronoi", "hdbscan"):
        assert FindClusters().preflight(ctx, FindClusters.Settings(algorithm=algorithm)) is None
    big = Context(locs=table(np.random.default_rng(0).uniform(0, 1e5, (300_000, 2))))
    question = FindClusters().preflight(big, FindClusters.Settings(algorithm="hdbscan"))
    assert "minutes" in question and "square" in question
