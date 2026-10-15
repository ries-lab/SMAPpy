"""Sort ROIs and Filter ROIs: by what was measured and what they were called."""
import numpy as np
import pytest

from smappy.plugins import Context
from smappy.roi_manager.order import condition, sort_ids
from test_roi_annotation import project_with_rois


def test_a_sort_puts_missing_values_last_and_breaks_ties_with_the_next_key():
    rows = {"a": {"n": 3, "q": "bad"}, "b": {"n": 8, "q": "good"},
            "c": {"q": "good"}, "d": {"n": 3, "q": "good"}}
    lists = {"q": ["good", "bad"]}
    numbers = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert sort_ids(rows, [("n", True)], lists, numbers) == ["b", "a", "d", "c"]
    assert sort_ids(rows, [("n", False)], lists, numbers) == ["a", "d", "b", "c"]
    # a keyword by its place in its list: good before bad
    assert sort_ids(rows, [("n", False), ("q", False)], lists, numbers) == \
        ["d", "a", "b", "c"]


def test_a_condition_reads_columns_keywords_and_backquoted_names():
    test, names = condition('n >= 6 and quality == "good" or `NPC Corners.n` > 7')
    assert names == ["n", "quality", "NPC Corners.n"]
    assert test({"n": 6, "quality": "good"})
    assert not test({"n": 6, "quality": "bad"})
    assert test({"NPC Corners.n": 8})
    assert not test({})                          # nothing measured: comparisons fail
    assert condition("not n >= 6")[0]({})
    with pytest.raises(ValueError):
        condition("__import__('os')")
    with pytest.raises(ValueError):
        condition("n >=")


def measured_project():
    session, project = project_with_rois(4)
    project.evaluate()
    ids = list(project.rois)
    # the counts the statistics measured differ per ROI only by chance; make
    # them known, as a step's values
    table = next(iter(project.tables.values()))
    for roi_id, n in zip(ids, (5, 9, 7, 9)):
        entry = dict(table.entry(roi_id))
        entry["values"] = {**entry["values"], "n_localizations": n}
        table.set(roi_id, entry, table.step(roi_id))
    project.annotate(ids[3], "quality", "bad")
    project.annotate(ids[1], "quality", "good")
    return session, project, ids


def test_sorting_orders_the_list_and_the_results_and_keeps_the_numbers():
    session, project, ids = measured_project()
    numbers = {i: project.rois[i].number for i in ids}
    from smappy.plugins.roi_order import SortROIs, SortROIsSettings
    result = SortROIs().run(Context(locs=session.locs, session=session),
                            SortROIsSettings(by="n_localizations", descending=True,
                                             then_by="quality", then_descending=False))
    # 9 (good), 9 (bad), 7, 5
    assert project.ordered_ids() == [ids[1], ids[3], ids[2], ids[0]]
    assert [r["roi_id"] for r in project.results()] == [numbers[i] for i in
                                                        (ids[1], ids[3], ids[2], ids[0])]
    assert {i: project.rois[i].number for i in ids} == numbers
    assert "sorted by n_localizations" in result.text
    again = project.to_dict()
    project.from_dict(again)
    assert project.ordered_ids() == [ids[1], ids[3], ids[2], ids[0]]
    SortROIs().run(Context(locs=session.locs, session=session), SortROIsSettings(by=""))
    assert project.ordered_ids() == ids


def test_filtering_sets_use_by_text_and_is_undone():
    session, project, ids = measured_project()
    from smappy.plugins.roi_order import FilterROIs, FilterROIsSettings
    ctx = Context(locs=session.locs, session=session)
    settings = FilterROIsSettings(condition='n_localizations >= 7 and quality != "bad"')
    assert "would use 1 of 4" in FilterROIs().describe(ctx, settings).notice.text
    FilterROIs().run(ctx, settings)
    assert [project.rois[i].use for i in ids] == [False, True, False, False]
    assert [r["roi_id"] for r in project.results()] == [project.rois[ids[1]].number]
    project.undo()
    assert all(project.rois[i].use for i in ids)
    with pytest.raises(ValueError, match="no ROI has 'corners'"):
        FilterROIs().run(ctx, FilterROIsSettings(condition="corners > 2"))


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_click_on_a_column_sorts_the_list(app):
    from smappy.gui.roi_window import ROIManagerWindow
    session, project, ids = measured_project()
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.table.horizontalHeader().sectionClicked.emit(3)    # annotations
    assert window.rows.ids[:2] == [ids[1], ids[3]]           # good, bad, then none
    window.table.horizontalHeader().sectionClicked.emit(0)    # back to the numbers
    assert window.rows.ids == ids
    window.close()


def test_a_column_chosen_from_insert_goes_onto_the_condition():
    from smappy.plugins.roi_order import FilterROIs, FilterROIsSettings
    plugin = FilterROIs()
    out = plugin.react("insert", FilterROIsSettings(condition="n_corners >= 6 and",
                                                    insert="quality"))
    assert out == {"condition": "n_corners >= 6 and quality", "insert": ""}
    out = plugin.react("insert", FilterROIsSettings(insert="NPC Corners.n_corners"))
    assert out["condition"] == "`NPC Corners.n_corners`"
    session, project, ids = measured_project()
    found = plugin.describe(Context(locs=session.locs, session=session),
                            FilterROIsSettings()).choices["insert"]
    assert ("n_localizations", "n_localizations") in found
    assert ("quality", "quality (keywords)") in found


# ------------------------------------------------------- double ROIs

def test_of_two_close_rois_the_one_with_more_localizations_stays():
    from smappy.roi_manager.order import doubles
    centers = [(0, 0), (60, 0), (1000, 0), (1000, 90), (1000, 180), (5000, 0)]
    counts = [10, 30, 50, 5, 50, 1]
    # 0 and 1 are doubles: 1 has more; 2-3-4 a chain: the middle goes
    assert doubles(centers, counts, 100).tolist() == [True, False, False, True, False, False]
    # closer than: 100 apart exactly is not a double
    assert not doubles([(0, 0), (100, 0)], [1, 2], 100).any()
    # other files: never doubles
    assert not doubles([(0, 0), (10, 0)], [1, 2], 100, groups=[1, 2]).any()


def blobs_project(centers, counts):
    """ROIs on blobs of known sizes, one ROI per centre."""
    from smappy.session import Session
    from test_roi_evaluation_plots import table
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for (cx, cy), n in zip(centers, counts):
        xs.append(rng.normal(cx, 10, n))
        ys.append(rng.normal(cy, 10, n))
    from smappy.locs import Localizations
    n = sum(counts)
    locs = Localizations({"x_nm": np.concatenate(xs), "y_nm": np.concatenate(ys),
                          "frame": np.arange(n), "photons": np.full(n, 500.0),
                          "xy_err_nm": np.full(n, 10.0)})
    session = Session(locs)
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(100)
    file_id = next(iter(project.sources))
    ids = [project.add_roi(file_id, list(c)).id for c in centers]
    return session, project, ids


def test_remove_double_rois_excludes_the_dimmer_of_a_pair_as_one_undo_step():
    from smappy.plugins.roi_order import RemoveDoubleROIs, RemoveDoubleSettings
    # ROIs 1 and 2 sit on one blob of 40 (2 a little off it, so it shows
    # fewer); ROI 3 on a blob of 20 far away
    session, project, ids = blobs_project([(0, 0), (40, 0), (2000, 0)], [40, 1, 20])
    ctx = Context(locs=session.locs, session=session)
    shown = [len(project.indices(project.rois[i])) for i in ids]
    assert shown[0] > shown[1]
    settings = RemoveDoubleSettings(distance_nm=100)
    assert "would exclude 1 of 3" in RemoveDoubleROIs().describe(ctx, settings).notice.text
    result = RemoveDoubleROIs().run(ctx, settings)
    assert [project.rois[i].use for i in ids] == [True, False, True]
    assert result.values == {"excluded": 1, "used": 2}
    # an excluded ROI takes no part the next time
    project.rois[ids[0]].use = False
    RemoveDoubleROIs().run(ctx, settings)
    assert [project.rois[i].use for i in ids] == [False, False, True]
    project.rois[ids[0]].use = True
    project.undo()
    assert all(project.rois[i].use for i in ids)


# --------------------------------------------- inside the drawn ROI

def test_the_rois_inside_the_drawn_roi_are_kept_by_a_condition_on_position():
    from smappy.plugins.roi_order import FilterROIs, FilterROIsSettings
    from smappy.regions import Region
    session, project = project_with_rois(4)          # centres at 0, 1000, 2000, 3000
    ids = list(project.rois)
    ctx = Context(locs=session.locs, session=session)
    settings = FilterROIsSettings(condition="in_drawn_roi")
    with pytest.raises(ValueError, match="needs an ROI drawn"):
        FilterROIs().run(ctx, settings)
    session.set_roi(Region.rect(500, -500, 2500, 500))
    FilterROIs().run(ctx, settings)
    assert [project.rois[i].use for i in ids] == [False, True, True, False]
    # the centres are columns too
    FilterROIs().run(ctx, FilterROIsSettings(condition="roi_x_nm < 1500"))
    assert [project.rois[i].use for i in ids] == [False, True, False, False]


def test_an_roi_of_a_file_the_layer_does_not_show_is_not_in_the_drawn_roi():
    from smappy.regions import Region
    from test_roi_session import rois_of
    from test_roi_evaluation_plots import table
    project, first = rois_of(table([(0, 0)]), table([(0, 0)]))
    second = project.source_of(1)
    a = project.add_roi(first.id, [0, 0])
    b = project.add_roi(second.id, [0, 0])
    session = project.session
    session.set_roi(Region.rect(-100, -100, 100, 100))
    assert project.in_drawn_roi() == {a.id: True, b.id: True}
    session.layers[0].state.set_files([1])
    assert project.in_drawn_roi() == {a.id: False, b.id: True}
