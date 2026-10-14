"""Analysis/Regions/Summarize/Export table: the results table as a csv."""
import csv

import numpy as np
import pytest

from smappy import plugins
from smappy.plugins import Context
from test_roi_order import measured_project

PLUGIN = "Analysis/Regions/Summarize/Export table"


def export(session, path, **settings):
    plugin = plugins.get(PLUGIN)
    return plugin().run(Context(locs=session.locs, session=session),
                        plugin.Settings(path=str(path), **settings))


def read(path, delimiter=","):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def test_a_row_per_included_roi_with_its_place_annotation_and_numbers(tmp_path):
    session, project, ids = measured_project()
    project.rois[ids[2]].comment = "a nice one"
    project.rois[ids[0]].use = False                       # left out
    export(session, tmp_path / "t.csv")
    rows = read(tmp_path / "t.csv")
    assert [int(r["roi_id"]) for r in rows] == [project.rois[i].number for i in ids[1:]]
    assert [int(r["n_localizations"]) for r in rows] == [9, 7, 9]
    assert [r["quality"] for r in rows] == ["good", "", "bad"]
    third = rows[1]
    roi = project.rois[ids[2]]
    assert float(third["x_nm"]) == pytest.approx(roi.center[0])
    assert third["comment"] == "a nice one" and third["file"]
    assert list(rows[0])[:6] == ["roi_id", "file_id", "x_nm", "y_nm", "file", "comment"]


def test_rows_come_in_the_rois_order(tmp_path):
    session, project, ids = measured_project()
    project.order = [ids[3], ids[1], ids[2], ids[0]]
    export(session, tmp_path / "t.csv", positions=False)
    rows = read(tmp_path / "t.csv")
    assert [int(r["roi_id"]) for r in rows] == \
        [project.rois[i].number for i in (ids[3], ids[1], ids[2], ids[0])]
    assert "x_nm" not in rows[0]


def test_only_the_ticked_columns_are_written_and_tabs_for_a_tsv(tmp_path):
    session, project, ids = measured_project()
    export(session, tmp_path / "t.tsv", columns=["roi_id", "n_localizations"])
    header = (tmp_path / "t.tsv").read_text().splitlines()[0]
    assert header == "roi_id\tn_localizations"
    with pytest.raises(ValueError, match="none of the chosen columns"):
        export(session, tmp_path / "u.csv", columns=["nope"])


def test_the_list_offers_the_tables_columns():
    session, project, ids = measured_project()
    plugin = plugins.get(PLUGIN)
    state = plugin().describe(Context(locs=session.locs, session=session),
                              plugin.Settings())
    offered = state.choices["columns"]
    assert {"roi_id", "x_nm", "comment", "n_localizations", "quality"} <= set(offered)


def test_per_cluster_results_are_written_without_positions(tmp_path):
    from smappy.locs import Localizations
    from smappy.roi_manager.pipeline import Step
    from smappy.session import Session
    rng = np.random.default_rng(0)
    centres = np.array([[1000, 1000], [5000, 1000], [3000, 4000]], float)
    sizes = (30, 50, 20)
    xy = np.concatenate([c + rng.normal(0, 15, (n, 2)) for c, n in zip(centres, sizes)])
    locs = Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(len(xy)), "photons": np.full(len(xy), 500.0),
                          "xy_err_nm": np.full(len(xy), 10.0),
                          "cluster_id": np.repeat([1, 2, 3], sizes).astype(np.int32)})
    session = Session(locs)
    session.show_grouped(0, False)
    hist = plugins.get("Analysis/Measure/Histogram")
    session.evaluate_regions("cluster_id", [Step(label="photons", path=hist.path,
                                                 plugin=hist(),
                                                 settings=hist.Settings(column="photons"))])
    export(session, tmp_path / "c.csv", evaluation="per cluster_id: photons")
    rows = read(tmp_path / "c.csv")
    assert [int(r["photons_n"]) for r in rows] == [30, 50, 20]
    assert "x_nm" not in rows[0]


def test_nothing_measured_says_so(tmp_path):
    from test_roi_annotation import project_with_rois
    session, project = project_with_rois(2)
    with pytest.raises(ValueError, match="nothing measured"):
        export(session, tmp_path / "t.csv")
