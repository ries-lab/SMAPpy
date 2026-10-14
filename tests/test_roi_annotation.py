"""Annotating ROIs: a keyword from each of the project's lists, by key."""
import numpy as np
import pytest

from smappy.roi_manager.core import (annotation_lists_text, parse_annotation_lists)
from smappy.session import Session
from test_roi_evaluation_plots import table


def project_with_rois(n=3):
    session = Session(table([(i * 1000, 0) for i in range(n)]))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(200)
    file_id = next(iter(project.sources))
    for i in range(n):
        project.add_roi(file_id, [i * 1000, 0])
    return session, project


def test_the_lists_are_read_as_written_one_a_line():
    lists = parse_annotation_lists("quality: good, bad, good\n\nshape: ring,arc\nlater")
    assert lists == {"quality": ["good", "bad"], "shape": ["ring", "arc"], "later": []}
    assert parse_annotation_lists(annotation_lists_text(lists)) == lists


def test_an_annotation_is_undone_and_kept_with_the_project():
    session, project = project_with_rois()
    project.set_annotation_lists({"quality": ["good", "bad"], "shape": ["ring"]})
    first = next(iter(project.rois))
    project.annotate(first, "quality", "bad")
    project.annotate(first, "shape", "ring")
    assert project.rois[first].annotations == {"quality": "bad", "shape": "ring"}
    project.undo()
    assert project.rois[first].annotations == {"quality": "bad"}
    with pytest.raises(ValueError):
        project.annotate(first, "quality", "awful")      # not in the list
    saved = project.to_dict()
    project.from_dict(saved)
    assert project.rois[first].annotations == {"quality": "bad"}
    assert project.annotation_lists == {"quality": ["good", "bad"], "shape": ["ring"]}


def test_the_keywords_are_columns_of_the_results_and_not_of_the_signature():
    session, project = project_with_rois(2)
    project.evaluate()
    first, second = list(project.rois)
    project.annotate(first, "quality", "good")
    assert not project.needs_evaluation()                 # nothing measured changed
    rows = {r["roi_id"]: r for r in project.results()}
    assert rows[project.rois[first].number]["quality"] == "good"
    assert "quality" not in rows[project.rois[second].number]


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_key_per_roi_annotates_and_moves_on(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from smappy.gui.roi_window import ROIManagerWindow
    session, project = project_with_rois()
    project.set_annotation_lists({"quality": ["good", "bad"], "shape": ["ring", "arc"]})
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.show()
    QTest.qWaitForWindowExposed(window)
    window.table.setFocus()
    ids = list(project.rois)
    window.select(ids[0])
    window.annotation.advance.setChecked(True)
    QTest.keyClick(window, Qt.Key_2)                       # bad, then the next ROI
    assert project.rois[ids[0]].annotations == {"quality": "bad"}
    assert window.active == ids[1]
    QTest.keyClick(window, Qt.Key_Tab)                     # the shape list
    QTest.keyClick(window, Qt.Key_1)                       # ring, and on to the third
    assert project.rois[ids[1]].annotations == {"shape": "ring"}
    assert window.rows.data(window.rows.index(1, 3)) == "ring"
    assert window.active == ids[2]
    QTest.keyClick(window, Qt.Key_Up)
    assert window.active == ids[1]
    QTest.keyClick(window, Qt.Key_Space)                   # use is a switch
    assert not project.rois[ids[1]].use
    assert not window.annotation.use.isChecked()
    window.close()


def test_typing_in_the_lists_is_text_and_annotates_nothing(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from smappy.gui.roi_window import ROIManagerWindow
    session, project = project_with_rois()
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.show()
    QTest.qWaitForWindowExposed(window)
    first = next(iter(project.rois))
    window.select(first)
    box = window.annotation.lists
    box.parentWidget().show()
    box.setPlainText("")
    box.setFocus()
    QTest.keyClicks(box, "class: 1x, 2y")
    assert box.toPlainText() == "class: 1x, 2y"
    assert project.rois[first].annotations == {}
    window.annotation.apply_lists()                # what the pause after typing does
    assert project.annotation_lists == {"class": ["1x", "2y"]}
    assert [b.text() for b in window.annotation.rows[0]] == ["1 1x", "2 2y"]
    window.close()
