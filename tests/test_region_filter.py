"""The region filter (`smappy.region_filter`, *Filter regions*): the
localizations of the regions whose results pass, by one flag every layer
honours -- and a region's measurement that never sees it."""
import numpy as np
import pytest

from smappy import plugins
from smappy.chain import Block, ChainSpec, Step, chain_class
from smappy.locs import Localizations
from smappy.plugins.region_column import RegionsToColumn
from smappy.region_filter import ANNOTATIONS, Condition, evaluate
from smappy.region_results import Evaluation, is_current, rows
from smappy.region_runs import REGION_FILTER
from smappy.region_table import RegionTable
from smappy.session import Session

from test_clusters import SIZES, blobs, table

STATS = "Analysis/Measure/Localization Statistics"
IDSTATS = "Analysis/Measure/Id statistics"
PER_CLUSTER = "per cluster_id: Localization Statistics"


def clusters():
    """Three clusters of 60, 40 and 25 localizations (`SIZES`) and noise,
    Id statistics measured per cluster."""
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    cls = plugins.get(IDSTATS)
    session.run(cls(), cls.Settings(id_column="cluster_id"))
    return session, truth


def shown(session, layer=0) -> np.ndarray:
    return np.asarray(session.selection(layer).mask, dtype=bool)


def by_size(n):
    return Condition("cluster_id", "Id statistics", "n_locs", lo=n)


def test_the_flag_keeps_the_localizations_of_the_clusters_that_pass_and_no_noise():
    session, truth = clusters()
    session.set_region_filter([by_size(30)])          # the clusters of 60 and 40
    keep = np.isin(truth, [1, 2])
    assert np.array_equal(np.asarray(session.locs[REGION_FILTER]) > 0.5, keep)
    assert np.array_equal(shown(session), keep)
    assert session.region_filter_outcome().counts == [(2, 3)]
    # the filter off in a layer -- the Render tab's button -- shows everything
    session.layers[0].remove_bound(REGION_FILTER)
    assert shown(session).all()


def test_a_merged_blink_is_in_if_any_of_its_localizations_is():
    session, truth = clusters()
    session.show_grouped(0, True)
    session.set_region_filter([by_size(30)])
    grouped, sel = session.layers[0].grouped_selection(0)
    assert REGION_FILTER in grouped
    ids = np.asarray(grouped["cluster_id"])
    assert np.array_equal(np.asarray(sel.mask, bool), np.isin(ids, [1, 2]))


def test_clearing_the_conditions_takes_the_flag_and_its_bound_away():
    session, _ = clusters()
    session.set_region_filter([by_size(30)])
    session.set_region_filter([])
    assert REGION_FILTER not in session.locs
    assert REGION_FILTER not in session.layers[0].ranges
    assert shown(session).all()


def test_the_summaries_count_the_regions_the_filter_keeps():
    session, _ = clusters()
    ev = Evaluation("cluster_id", "Id statistics")
    assert len(rows(session.context(), ev)) == 3
    session.set_region_filter([by_size(30)])
    assert sorted(r["n_locs"] for r in rows(session.context(), ev)) == [40, 60]


def test_a_filter_on_a_result_neither_dates_nor_narrows_the_next_measurement():
    """The feedback loop: a bound on a value written from regions, or on the
    flag, is no input of region results, and a block measures every region
    whatever it says -- else a re-run would empty the regions that failed."""
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    block = chain_class(ChainSpec(steps=[Block(regions="cluster_id",
                                               steps=[Step(plugin=STATS)])]))
    session.run(block(), block.Settings())
    session.run(RegionsToColumn(), RegionsToColumn.Settings(
        source=PER_CLUSTER, values="n_localizations", prefix="cluster_"))
    session.layers[0].set_bound("cluster_n_localizations", 30, None)
    session.set_region_filter([
        Condition("cluster_id", "Localization Statistics", "n_localizations", lo=30)])
    ctx = session.context()
    ev = Evaluation.parse(PER_CLUSTER)
    assert is_current(ctx, ev)
    session.run(block(), block.Settings())                 # measured again
    measured = sorted(r["n_localizations"] for r in
                      rows(session.context(), ev))          # the filter keeps two
    assert measured == [40, 60]
    table_ = session.id_regions["cluster_id"]["Localization Statistics"]
    assert len(table_) == 3                                # but all three measured


def overlapping_rois():
    """Two ROIs overlapping on one blob, and a third apart, with made-up
    results: ROI A has the corners, ROI B the diameter, ROI C both."""
    rng = np.random.default_rng(4)
    xy = np.vstack([rng.normal([3000, 3000], 120, (400, 2)),
                    rng.normal([9000, 3000], 60, (100, 2))])
    session = Session(table(xy))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(400)
    file_id = next(iter(project.sources))
    for centre in ([2850, 3000], [3150, 3000], [9000, 3000]):
        project.add_roi(file_id, centre)
    a, b, c = list(project.rois)
    made = RegionTable("Made up")
    for roi_id, corners, diameter in ((a, 8, 50), (b, 3, 110), (c, 8, 108)):
        made.set(roi_id, {"values": {"corners": corners, "diameter_nm": diameter}})
    project.tables["Made up"] = made
    return session, project, (a, b, c), xy


def test_rois_are_judged_whole_and_a_localization_is_in_if_any_passing_roi_holds_it():
    session, project, (a, b, c), xy = overlapping_rois()
    corners = Condition("rois", "Made up", "corners", lo=6)
    diameter = Condition("rois", "Made up", "diameter_nm", lo=100, hi=120)
    in_a, in_b, in_c = (np.isin(np.arange(len(xy)), project.session_rows(project.rois[r]))
                        for r in (a, b, c))
    session.set_region_filter([corners])                  # A and C pass
    assert np.array_equal(shown(session), in_a | in_c)    # the overlap is in, through A
    session.set_region_filter([corners, diameter])        # only C has both
    assert np.array_equal(shown(session), in_c)           # the overlap is out
    assert session.region_filter_outcome().counts == [(2, 3), (2, 3)]


def test_an_annotation_is_a_condition_like_a_result():
    session, project, (a, b, c), xy = overlapping_rois()
    project.rois[b].use = False
    session.changed("rois")
    session.set_region_filter([Condition("rois", ANNOTATIONS, "use", accept=["true"])])
    assert session.region_filter_outcome().passing["rois"] == {a, c}


def test_two_sources_combine_by_and():
    session, project, (a, b, c), xy = overlapping_rois()
    cluster = np.where(xy[:, 0] > 8000, 2, 1)             # blob 1, blob 2
    session.set_locs(Localizations({**session.locs.columns, "cluster_id": cluster},
                                   dict(session.locs.metadata)))
    session.id_regions["cluster_id"] = {"Made up": RegionTable.from_columns(
        "Made up", [1, 2], {"big": np.array([1.0, 0.0])})}
    corners = Condition("rois", "Made up", "corners", lo=6)       # A and C
    big = Condition("cluster_id", "Made up", "big", lo=0.5)       # blob 1
    session.set_region_filter([corners, big])
    in_a = np.isin(np.arange(len(xy)), project.session_rows(project.rois[a]))
    assert np.array_equal(shown(session), in_a & (cluster == 1))


def test_a_drift_correction_keeps_the_filter_and_a_new_clustering_suspends_it():
    session, truth = clusters()
    session.set_region_filter([by_size(30)])
    moved = Localizations({**session.locs.columns,
                           "x_nm": np.asarray(session.locs["x_nm"]) + 5.0},
                          dict(session.locs.metadata))
    session.set_locs(moved)                    # the same ids: stale, still applied
    outcome = session.region_filter_outcome()
    assert outcome.flag is not None and any("out of date" in w for w in outcome.warnings)
    assert np.array_equal(shown(session), np.isin(truth, [1, 2]))
    relabelled = Localizations({**session.locs.columns,
                                "cluster_id": np.where(truth > 0, 4 - truth, 0)},
                               dict(session.locs.metadata))
    session.set_locs(relabelled)               # a new clustering: suspended
    outcome = session.region_filter_outcome()
    assert outcome.flag is None and "cluster_id" in outcome.suspended
    assert any("suspended" in w for w in outcome.warnings)


def test_a_saved_file_opens_filtered(tmp_path):
    session, truth = clusters()
    session.set_region_filter([by_size(30)])
    path = session.save(tmp_path / "filtered.h5")
    again = Session()
    again.load(path)
    again.show_grouped(0, False)
    assert [c.to_dict() for c in again.region_filter] == [by_size(30).to_dict()]
    assert REGION_FILTER in again.layers[0].ranges
    assert np.array_equal(shown(again), np.isin(truth, [1, 2]))


def test_the_plugin_adds_replaces_and_clears_and_a_chain_hands_the_filter_back():
    session, truth = clusters()
    cls = plugins.get("Analysis/Regions/Filter regions")
    ev = "per cluster_id: Id statistics"
    form = cls().describe(session.context(), cls.Settings(evaluation=ev, value="n_locs",
                                                          low=30))
    assert form.notice.text == "keeps 2 of 3 regions"           # before running
    assert (ev, ev) in form.choices["evaluation"] and ("n_locs", "n_locs") in \
        form.choices["value"]
    session.run(cls(), cls.Settings(evaluation=ev, value="n_locs", low=30))
    assert np.array_equal(shown(session), np.isin(truth, [1, 2]))
    session.run(cls(), cls.Settings(evaluation=ev, value="n_locs", high=50))
    assert np.array_equal(shown(session), truth == 2)            # both: 40
    session.run(cls(), cls.Settings(mode="replace", evaluation=ev, value="n_locs",
                                    high=50))
    assert np.array_equal(shown(session), np.isin(truth, [2, 3]))
    session.run(cls(), cls.Settings(mode="clear"))
    assert session.region_filter == [] and shown(session).all()

    before = dict(session.locs.metadata)
    chain = chain_class(ChainSpec(steps=[Step(plugin=IDSTATS,
                                              values={"id_column": "cluster_id"}),
                                         Step(plugin="Analysis/Regions/Filter regions",
                                              values={"evaluation": ev, "value": "n_locs",
                                                      "low": 30})]))
    session.run(chain(), chain.Settings())
    assert len(session.region_filter) == 1
    assert np.array_equal(shown(session), np.isin(truth, [1, 2]))
    assert "region_filter" not in before


def test_a_condition_says_what_it_keeps():
    c = Condition("cluster_id", "Id statistics", "n_locs", lo=20)
    assert c.text() == "per cluster_id › Id statistics › n_locs  20 … ∞"
    assert c.passes(25) and not c.passes(10) and not c.passes(None)
    assert not c.passes(float("nan"))
    k = Condition("rois", ANNOTATIONS, "quality", accept=["good"])
    assert k.passes("good") and not k.passes("bad")
    assert Condition.from_dict(c.to_dict()) == c
