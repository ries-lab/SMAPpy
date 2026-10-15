"""Named lines per ROI: drawn, reversed, saved, and read as columns."""
import numpy as np
import pytest

from smappy.roi_manager.core import (ROI, line_angle_length, parse_line_names)
from test_roi_annotation import project_with_rois


def test_a_lines_angle_is_smaps_from_start_to_end_and_its_length_in_nm():
    assert line_angle_length([[0, 0], [10, 0]]) == (0.0, 10.0)
    angle, length = line_angle_length([[0, 0], [0, -30]])
    assert angle == pytest.approx(-90) and length == pytest.approx(30)
    angle, _ = line_angle_length([[1, 1], [0, 0]])
    assert angle == pytest.approx(-135)


def test_line_names_are_column_prefixes_read_from_commas():
    assert parse_line_names("direction, long axis,\naxis, direction,") == [
        "direction", "long_axis", "axis"]


def test_a_line_is_drawn_reversed_undone_and_kept_with_the_project():
    session, project = project_with_rois()
    project.set_line_names(["direction", "axis"])
    first = next(iter(project.rois))
    project.set_line(first, "axis", [[0, 0], [100, 100]])
    assert project.reverse_line(first, "axis")
    assert project.rois[first].lines == {"axis": [[100.0, 100.0], [0.0, 0.0]]}
    project.undo()
    assert project.rois[first].lines["axis"] == [[0.0, 0.0], [100.0, 100.0]]
    assert not project.reverse_line(first, "direction")        # none drawn
    project.move_roi(first, [10, 0])
    assert project.rois[first].lines["axis"] == [[10.0, 0.0], [110.0, 100.0]]
    saved = project.to_dict()
    project.from_dict(saved)
    assert project.line_names == ["direction", "axis"]
    assert project.rois[first].lines == {"axis": [[10.0, 0.0], [110.0, 100.0]]}
    project.set_line(first, "axis", None)
    assert project.rois[first].lines == {}


def test_an_old_files_direction_is_the_line_called_direction():
    roi = ROI.from_dict({"file_id": "f", "center": [0, 0],
                         "direction": [[0, 0], [0, 50]]})
    assert roi.lines == {"direction": [[0.0, 0.0], [0.0, 50.0]]}
    assert ROI.from_dict({"file_id": "f", "center": [0, 0], "direction": None}).lines == {}


def test_the_direction_keeps_the_signature_it_had_before_named_lines():
    """An old project's results stay current: the line called direction is
    the old ``direction`` key, and a ROI with no other line adds nothing."""
    session, project = project_with_rois(1)
    roi = next(iter(project.rois.values()))
    roi.lines = {"direction": [[0, 0], [0, 50]]}
    geometry = project.geometry(roi)
    assert geometry == {"center": roi.center, "shape": project.shape,
                        "size_nm": project.size_nm, "direction": [[0, 0], [0, 50]]}
    roi.lines = {**roi.lines, "axis": [[0, 0], [1, 1]]}
    assert project.geometry(roi)["lines"] == {"axis": [[0, 0], [1, 1]]}


def test_drawing_a_line_puts_it_out_of_date_like_the_direction_did():
    session, project = project_with_rois(2)
    project.evaluate()
    first = next(iter(project.rois))
    project.set_line(first, "direction", [[0, 0], [50, 0]])
    assert project.needs_evaluation()


def test_angle_and_length_are_columns_that_filter_sort_and_results_read():
    session, project = project_with_rois(3)
    project.set_line_names(["direction"])
    ids = list(project.rois)
    for roi_id, angle in zip(ids, (30, -60, 120)):
        c = np.asarray(project.rois[roi_id].center)
        end = c + 80 * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        project.set_line(roi_id, "direction", [c.tolist(), end.tolist()])
    rows = project.sort_rows()
    assert rows[ids[1]]["direction_angle_deg"] == pytest.approx(-60)
    assert rows[ids[1]]["direction_length_nm"] == pytest.approx(80)
    assert "direction_angle_deg" in project.sort_columns()
    assert project.filter_rois("direction_angle_deg > 0") == {
        ids[0]: True, ids[1]: False, ids[2]: True}
    assert project.sort([("direction_angle_deg", False)]) == [ids[1], ids[0], ids[2]]
    project.evaluate()
    measured = {r["roi_id"]: r for r in project.results()}
    assert measured[project.rois[ids[2]].number]["direction_angle_deg"] == pytest.approx(120)


def test_regions_to_column_writes_a_lines_angle_with_no_evaluation():
    from smappy.plugins.region_column import RegionsToColumn
    session, project = project_with_rois(2)
    first, second = list(project.rois)
    project.set_line(first, "direction", [[0, 0], [0, 40]])
    session.run(RegionsToColumn(), RegionsToColumn.Settings(values="direction_angle_deg"))
    angle = np.asarray(session.locs["direction_angle_deg"])
    x = np.asarray(session.locs["x_nm"])
    assert np.allclose(angle[np.abs(x) < 100], 90)
    assert np.all(np.isnan(angle[np.abs(x - 1000) < 100]))   # no line there
    with pytest.raises(ValueError, match="direction_angle_deg"):
        RegionsToColumn()(ctx=session.context(), values="axis_angle_deg")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_lines_are_drawn_chosen_and_reversed_from_the_keys(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from smappy.gui.roi_window import ROIManagerWindow
    session, project = project_with_rois()
    project.set_line_names(["direction", "axis"])
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.show()
    QTest.qWaitForWindowExposed(window)
    window.table.setFocus()
    first = next(iter(project.rois))
    window.select(first)
    assert [window.line_box.itemText(i) for i in range(window.line_box.count())] == [
        "direction", "axis"]
    QTest.keyClick(window, Qt.Key_L, Qt.ShiftModifier)            # the next line
    assert window.line_name == "axis"
    QTest.keyClick(window, Qt.Key_L)                              # draw it
    assert window.drawing == "line"
    window._roi_clicked(-50.0, 0.0)
    window._roi_clicked(50.0, 0.0)
    assert project.rois[first].lines == {"axis": [[-50.0, 0.0], [50.0, 0.0]]}
    assert [t.toPlainText() for t in window.roi_pane.texts] == ["axis"]
    QTest.keyClick(window, Qt.Key_R)
    assert project.rois[first].lines["axis"] == [[50.0, 0.0], [-50.0, 0.0]]
    window.set_line_name("direction")
    window._roi_clicked(0.0, 1.0)                     # on the axis: it is chosen
    assert window.line_name == "axis"
    window.clear_line_button.click()
    assert project.rois[first].lines == {}
    window.annotation.line_names.setText("direction, axis, tip")
    window.annotation.apply_line_names()
    assert project.line_names == ["direction", "axis", "tip"]
    window.close()


def test_a_drafted_roi_takes_its_lines_when_it_is_added(app):
    from smappy.gui.roi_window import ROIManagerWindow
    session, project = project_with_rois(1)
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.draft = np.array([2000.0, 0.0])
    window.redraw()
    window._start_drawing("line")
    window._roi_clicked(2000.0, 0.0)
    window._roi_clicked(2000.0, 60.0)
    window._reverse_line()
    window._add()
    roi = project.rois[window.active]
    assert roi.lines == {"direction": [[2000.0, 60.0], [2000.0, 0.0]]}
    window.close()
