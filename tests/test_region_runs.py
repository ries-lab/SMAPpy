"""Running plugins per region of an id column, and the status of a long run."""
import threading

import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.region_runs import (ColumnRegions, Progress, evaluate_regions,
                                progress_text, split_ids)
from smappy.roi_manager.pipeline import Step
from smappy.session import Session


def clusters(sizes=(30, 50, 20), noise=15, seed=0):
    """Clusters of known sizes far apart, and noise with id 0."""
    rng = np.random.default_rng(seed)
    parts, ids = [], []
    for k, n in enumerate(sizes, start=1):
        parts.append(rng.normal(k * 5000, 30, (n, 2)))
        ids.append(np.full(n, k))
    parts.append(rng.uniform(0, 20000, (noise, 2)))
    ids.append(np.zeros(noise, int))
    xy = np.vstack(parts)
    m = len(xy)
    return Localizations({"x_nm": xy[:, 0].astype(np.float32),
                          "y_nm": xy[:, 1].astype(np.float32),
                          "frame": np.arange(m, dtype=np.int64),
                          "photons": rng.gamma(4, 400, m).astype(np.float32),
                          "xy_err_nm": rng.uniform(5, 15, m).astype(np.float32),
                          "cluster_id": np.concatenate(ids).astype(np.int64)}, {})


class Count(Plugin):
    """Counts what it is given, and how often it ran."""
    name = "Count"
    path = "Test/Count"
    per_region = True
    runs = 0

    def run(self, ctx, settings):
        Count.runs += 1
        return Result(values={"n": len(ctx.selection), "all": len(ctx.locs),
                              "per_region": ctx.per_region})


def count_step(label="count"):
    return Step(label=label, path=Count.path, plugin=Count(), settings=None)


def test_an_id_column_names_its_regions_and_zero_is_none():
    ids, rows = split_ids(np.array([2, 0, 1, 2, -1, 1, 2]))
    assert ids.tolist() == [1, 2]
    assert [r.tolist() for r in rows] == [[2, 5], [0, 3, 6]]
    ids, rows = split_ids(np.array([0.0, np.nan, 3.0]))
    assert ids.tolist() == [3.0] and rows[0].tolist() == [2]


def test_the_status_says_how_far_how_fast_and_how_long():
    assert progress_text(0, 1000, 0.0) == "0% (0 of 1000 regions)"
    assert progress_text(370, 1000, 3.7) == \
        "37% (370 of 1000 regions), 100 a second, about 6 s left"
    assert progress_text(1000, 1000, 10.0).endswith("100 a second")
    assert "min left" in progress_text(10, 10_000, 1.0)
    said, ticks = [], []
    status = Progress(3, said.append, lambda d, t: ticks.append(d))
    for done in (1, 2, 3):
        status(done)
    assert ticks == [1, 2, 3] and said[-1].startswith("100% (3 of 3")


def test_each_cluster_is_measured_on_its_own_rows_under_the_layers_filter():
    session = Session(clusters())
    session.show_grouped(0, False)
    session.layers[0].set_bound("photons", 1000.0, None)
    Count.runs = 0
    tables = session.evaluate_regions("cluster_id", [count_step()])
    table = tables["count"]
    assert table.ids == ["1", "2", "3"]                  # the noise is not a region
    columns = table.columns()
    assert columns["all"].tolist() == [30, 50, 20]       # membership: the id alone
    photons = np.asarray(session.locs["photons"])
    ids = np.asarray(session.locs["cluster_id"])
    assert columns["n"].tolist() == [int(((ids == k) & (photons >= 1000)).sum())
                                     for k in (1, 2, 3)]
    assert columns["per_region"].all() and Count.runs == 3
    assert session.id_regions["cluster_id"]["count"] is table


def test_a_current_step_is_not_run_again_and_a_changed_filter_makes_it_stale():
    session = Session(clusters())
    session.show_grouped(0, False)
    Count.runs = 0
    session.evaluate_regions("cluster_id", [count_step()])
    session.evaluate_regions("cluster_id", [count_step()])
    assert Count.runs == 3                               # one signature, still true
    session.layers[0].set_bound("photons", 500.0, None)
    session.evaluate_regions("cluster_id", [count_step()])
    assert Count.runs == 6


def test_a_stopped_run_keeps_the_regions_it_measured():
    stop = threading.Event()
    source = ColumnRegions(locs=clusters(sizes=(10,) * 6), column="cluster_id")
    tables = evaluate_regions(source, [count_step()], stop=stop,
                              tick=lambda done, total: done == 2 and stop.set())
    assert tables["count"].ids == ["1", "2"]


def test_a_failing_step_costs_its_entry_in_that_region_only():
    class Picky(Plugin):
        per_region = True

        def run(self, ctx, settings):
            if len(ctx.locs) < 25:
                raise ValueError("too few")
            return Result(values={"n": len(ctx.locs)})

    source = ColumnRegions(locs=clusters(), column="cluster_id")
    table = evaluate_regions(source, [Step("picky", "T/P", Picky(), None)])["picky"]
    assert table.entry("3") == {"error": "ValueError: too few",
                                "signature": table.entry("3")["signature"]}
    assert table.entry("1")["values"] == {"n": 30}


def test_the_statistics_run_per_cluster_and_an_empty_one_is_a_zero():
    from smappy.roi_manager import pipeline as pipeline_module
    session = Session(clusters())
    session.show_grouped(0, False)
    session.layers[0].set_bound("photons", 1e9, None)     # nothing passes
    steps = pipeline_module.resolve(pipeline_module.default_instances())
    table = session.evaluate_regions("cluster_id", steps)["Localization Statistics"]
    assert table.columns()["n_localizations"].tolist() == [0, 0, 0]


def test_per_cluster_results_survive_a_save_and_can_be_left_out(tmp_path):
    import h5py
    session = Session(clusters())
    session.show_grouped(0, False)
    session.evaluate_regions("cluster_id", [count_step()])
    out = session.save(tmp_path / "c.hdf5", gui_state=False)
    with h5py.File(out, "r") as f:
        assert f["id_regions/0"].attrs["column"] == "cluster_id"
    back = Session()
    back.load(out)
    table = back.id_regions["cluster_id"]["count"]
    assert table.ids == ["1", "2", "3"] and table.columns()["all"].tolist() == [30, 50, 20]
    out = session.save(tmp_path / "d.hdf5", gui_state=False, region_results=False)
    back = Session()
    back.load(out)
    assert back.id_regions == {}


def test_a_cluster_has_its_blinks_through_group_id():
    session = Session(clusters())
    session.show_grouped(0, True)                         # links, writes group_id
    source = ColumnRegions(session, "cluster_id")
    pinned, rows, layer, _ = source.tables(0)
    blinks, _ = pinned[layer]["grouped"]
    want = np.unique(np.asarray(session.locs["group_id"])[rows])
    assert sorted(np.asarray(blinks["group_id"]).tolist()) == want.tolist()


def test_the_roi_run_reports_its_status_and_stops_when_asked():
    from smappy.roi_manager import ROIProject
    from smappy.workspace import Instance
    project = ROIProject()
    source = project.add_source(clusters(sizes=(20,) * 4, noise=0), name="a")
    project.set_filters({})
    for k in range(1, 5):
        project.add_roi(source.id, [k * 5000.0, k * 5000.0], reviewed=True)
    said = []
    calls = []
    run = project.evaluate([Instance(plugin="Analysis/Measure/Localization Statistics",
                                     values={"fit_precision": False})],
                           report=said.append,
                           stop=lambda: len(calls) >= 2,
                           progress=lambda done, total: calls.append(done))
    assert len(run["records"]) == 2 and project.runs[-1]["stopped"]
    assert said and "of 4 ROIs" in said[-1]
