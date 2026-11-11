"""The suite's guard against a modal dialog nobody answers: the test fails, naming it, and the
run goes on (`conftest.modal_guard`)."""
import pytest

from conftest import modal_guard


def test_a_question_nobody_answers_is_closed_and_named():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.instance() or QApplication([])
    with modal_guard(0.5) as left:
        answer = QMessageBox.question(None, "Stuck", "Nobody answers this")
    assert answer != QMessageBox.Yes
    assert len(left) == 1 and "QMessageBox" in left[0] and "Nobody answers" in left[0]


def test_a_dialog_answered_in_time_is_left_alone():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.instance() or QApplication([])
    box = QMessageBox(QMessageBox.Question, "Answered", "yes", QMessageBox.Yes | QMessageBox.No)
    with modal_guard(5.0) as left:
        QTimer.singleShot(100, lambda: box.done(QMessageBox.Yes))
        assert box.exec() == QMessageBox.Yes
    assert left == []


def test_pyplot_show_returns_on_the_qt_backend():
    """A test that ends in ``plt.show()`` after a GUI test made QtAgg the backend hung its worker."""
    pytest.importorskip("PySide6")
    import matplotlib
    from matplotlib import pyplot as plt
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    before = matplotlib.get_backend()
    plt.switch_backend("QtAgg")                                 # as `smappy.gui.figures` does
    try:
        plt.subplots()
        plt.show()                                              # returns
    finally:
        plt.close("all")
        plt.switch_backend(before)
