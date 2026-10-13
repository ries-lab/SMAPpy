"""One way to read what was measured per region, on the ROIs or per cluster
(`smappy.region_results`), and the readers that used to branch on the store."""
import numpy as np
import pytest

from smappy import plugins
from smappy.chain import Block, ChainSpec, Step, chain_class
from smappy.locs import Localizations
from smappy.plugins.region_column import RegionsToColumn
from smappy.region_results import Evaluation, available, is_current, rows, steps
from smappy.session import Session

from test_clusters import SIZES, blobs, table

STATS = "Analysis/Measure/Localization Statistics"
PER_CLUSTER = "per cluster_id: Localization Statistics"


def measured_clusters():
    """Three clusters of known sizes, Localization Statistics run in each."""
    xy, truth = blobs()
    session = Session(table(xy, cluster_id=truth))
    session.show_grouped(0, False)
    cls = chain_class(ChainSpec(steps=[Block(regions="cluster_id",
                                             steps=[Step(plugin=STATS)])]))
    session.run(cls(), cls.Settings())
    return session, truth


def measured_rois():
    """The same three blobs with an ROI on each, measured by a block."""
    xy, truth = blobs()
    session = Session(table(xy))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(400)
    file_id = next(iter(project.sources))
    for k in (1, 2, 3):
        project.add_roi(file_id, [3000.0 * k, 3000.0 * k])
    cls = chain_class(ChainSpec(steps=[Block(steps=[Step(plugin=STATS)])]))
    session.run(cls(), cls.Settings())
    return session, truth


def test_an_evaluation_reads_back_from_the_string_results_from_holds():
    for text in ("Corners", "per cluster_id: Corners", "per roi_id: a: b"):
        assert str(Evaluation.parse(text)) == text
    assert Evaluation.parse("Corners") == Evaluation("rois", "Corners")
    assert Evaluation.parse("per cluster_id: Corners") == Evaluation("cluster_id", "Corners")
    assert Evaluation.parse("per cluster_id: a: b").label == "a: b"


def test_both_stores_are_read_the_same_way():
    session, _ = measured_clusters()
    ctx = session.context()
    ev = Evaluation("cluster_id", "Localization Statistics")
    assert available(ctx, STATS) == [ev] and is_current(ctx, ev)
    assert sorted(r["n_localizations"] for r in rows(ctx, ev)) == sorted(SIZES)
    assert [s["plugin"] for s in steps(ctx, ev)] == [STATS]

    session, _ = measured_rois()
    ctx = session.context()
    ev = Evaluation("rois", "Localization Statistics")
    assert available(ctx, STATS) == [ev] and is_current(ctx, ev)
    assert sorted(r["n_localizations"] for r in rows(ctx, ev)) == sorted(SIZES)
    assert [s["plugin"] for s in steps(ctx, ev)] == [STATS]


def test_clusters_relabelled_leave_their_results_out_of_date():
    """Three clusters made one: what was measured in the three no longer
    describes the table, and Histograms says so rather than drawing three."""
    session, truth = measured_clusters()
    histograms = plugins.get("Analysis/Regions/Summarize/Histograms")
    assert histograms()(ctx=session.context(), evaluation=PER_CLUSTER).text.startswith(
        "3 regions")
    one = Localizations({**session.locs.columns,
                         "cluster_id": np.where(truth > 0, 1, 0)}, dict(session.locs.metadata))
    session.set_locs(one)
    ctx = session.context()
    ev = Evaluation.parse(PER_CLUSTER)
    assert not is_current(ctx, ev) and rows(ctx, ev) == [] and available(ctx) == []
    with pytest.raises(ValueError, match="out of date"):
        histograms()(ctx=session.context(), evaluation=PER_CLUSTER)
    offered = histograms().describe(ctx, histograms.Settings()).choices["evaluation"]
    assert (PER_CLUSTER, PER_CLUSTER) not in offered


def test_histograms_asks_nothing_before_reading_the_clusters():
    session, _ = measured_clusters()
    histograms = plugins.get("Analysis/Regions/Summarize/Histograms")
    settings = histograms.Settings(evaluation=PER_CLUSTER)
    assert histograms().preflight(session.context(), settings) is None


def test_a_value_written_onto_the_table_leaves_its_results_current():
    """Writing a region's number onto its localizations changes nothing it was
    measured on: a second value can be written from the same results."""
    session, _ = measured_clusters()
    for value in ("n_localizations", "median_photons"):
        session.run(RegionsToColumn(), RegionsToColumn.Settings(
            source=PER_CLUSTER, values=value, prefix="cluster_"))
    assert {"cluster_n_localizations", "cluster_median_photons"} <= set(session.locs.keys())
    assert is_current(session.context(), Evaluation.parse(PER_CLUSTER))

    session, _ = measured_rois()
    for value in ("n_localizations", "median_photons"):
        session.run(RegionsToColumn(), RegionsToColumn.Settings(
            source="Localization Statistics", values=value, prefix="roi_"))
    assert session.rois.labels() == ["Localization Statistics"]


def test_regions_to_column_says_when_the_cluster_results_are_out_of_date():
    session, truth = measured_clusters()
    session.set_locs(Localizations({**session.locs.columns,
                                    "cluster_id": np.where(truth > 0, 1, 0)},
                                   dict(session.locs.metadata)))
    with pytest.raises(ValueError, match="out of date"):
        RegionsToColumn()(ctx=session.context(), source=PER_CLUSTER,
                          values="n_localizations")


def counted_pores():
    """Simulated pores, each its own cluster (its ``copy``), NPC Corners run
    once per cluster."""
    from test_npc import pores
    locs = pores(0.5, seed=2)
    copy = np.asarray(locs["copy"]).astype(int)
    session = Session(Localizations({**locs.columns,
                                     "cluster_id": np.where(copy >= 0, copy + 1, 0)},
                                    dict(locs.metadata)))
    session.show_grouped(0, True)
    cls = chain_class(ChainSpec(steps=[Block(regions="cluster_id", steps=[
        Step(plugin="Analysis/Measure/NPC Corners")])]))
    session.run(cls(), cls.Settings())
    return session


def test_npc_labeling_efficiency_reads_corners_counted_per_cluster():
    """The corner counts of each cluster are offered and read; the joint
    model, which needs the ROIs' files for the field's precisions, says so."""
    from smappy.plugins.npc import LabelingEfficiencySettings
    session = counted_pores()
    analysis = plugins.get("Analysis/Regions/Summarize/NPC labelling efficiency")
    name = "per cluster_id: NPC Corners"
    offered = analysis().describe(session.context(), analysis.Settings()).choices["evaluation"]
    assert (name, name) in offered
    assert offered[0][1] == f"auto ({name})"          # the only corner count there is
    smap = analysis()(ctx=session.context(),
                      settings=LabelingEfficiencySettings(method="smap"))
    assert 0.3 < smap.data["efficiency"] < 0.7 and "counted by" in smap.text
    with pytest.raises(ValueError, match="ROIs"):
        analysis()(ctx=session.context(), settings=LabelingEfficiencySettings())


def test_a_chain_analyses_the_corners_its_own_block_counted_per_cluster():
    from test_npc import pores
    locs = pores(0.5, seed=2)
    copy = np.asarray(locs["copy"]).astype(int)
    session = Session(Localizations({**locs.columns,
                                     "cluster_id": np.where(copy >= 0, copy + 1, 0)},
                                    dict(locs.metadata)))
    session.show_grouped(0, True)
    cls = chain_class(ChainSpec(steps=[
        Block(regions="cluster_id", steps=[Step(plugin="Analysis/Measure/NPC Corners")]),
        Step(plugin="Analysis/Regions/Summarize/NPC labelling efficiency",
             values={"method": "smap"})]))
    assert cls().preflight(session.context(), cls.Settings()) is None
    result = session.run(cls(), cls.Settings())
    fitted = list(result.data["results"].values())[-1].data
    assert 0.3 < fitted["efficiency"] < 0.7
