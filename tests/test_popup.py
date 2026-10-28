"""A title-bar popup stays on the screen when what it holds grows.

The 3D window's options open under a button near the window's right edge;
unfolding the slab section inside them widened the popup rightwards from
where it stood, past the screen (the rendering tutorial caught it).
"""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_popup_that_grows_near_the_screens_edge_moves_back_onto_it(app):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from smappy.gui.widgets import Popup

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.addWidget(QLabel("narrow"))
    popup = Popup(content)
    screen = app.primaryScreen().availableGeometry()
    popup.adjustSize()
    popup.move(QPoint(screen.right() - popup.width() - 2, screen.top() + 20))
    popup.show()
    app.processEvents()

    wide = QLabel("x" * 50)                  # a section unfolded inside it
    layout.addWidget(wide)
    popup.adjustSize()
    app.processEvents()

    assert popup.width() > wide.sizeHint().width() - 20
    assert popup.geometry().right() <= screen.right()
    assert popup.geometry().left() >= screen.left()
    popup.close()
