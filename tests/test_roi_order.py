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
