"""Every window kept to its screen (`gui.screen`)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _app():
    from smappy.gui import make_app
    return make_app()


def test_a_window_larger_than_the_screen_is_shrunk_and_put_back_on_it():
    from PySide6.QtWidgets import QApplication, QWidget

    from smappy.gui.screen import fit_to_screen
    app = _app()
    room = QApplication.primaryScreen().availableGeometry()
    window = QWidget()
    window.resize(room.width() * 3, room.height() * 3)
    window.move(room.right() + 500, room.top() - 200)
    window.show()
    fit_to_screen(window)
    app.processEvents()
    frame = window.frameGeometry()
    assert window.width() <= room.width() and window.height() <= room.height()
    assert room.contains(frame.topLeft())
    window.close()


def test_a_window_that_fits_is_left_as_it_is():
    from PySide6.QtWidgets import QApplication, QWidget

    from smappy.gui.screen import fit_to_screen
    _app()
    room = QApplication.primaryScreen().availableGeometry()
    window = QWidget()
    window.resize(room.width() // 3, room.height() // 3)
    window.move(room.left() + 10, room.top() + 40)
    window.show()
    before = window.geometry()
    fit_to_screen(window)
    assert window.geometry() == before
    window.close()
