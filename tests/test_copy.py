"""Edit > Copy: the picture of the window in front, or the selected text."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Plot
from smappy.session import Session


def table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 2000, n),
                          "y_nm": rng.uniform(0, 2000, n),
                          "frame": np.arange(n, dtype=np.int64),
                          "photons": np.full(n, 1000.0),
                          "xy_err_nm": np.full(n, 10.0)})


@pytest.fixture
def windows():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.app import ControlWindow, RenderWindow

    app = QApplication.instance() or QApplication([])
    session = Session(table())
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    render.show()
    app.processEvents()
    yield app, control, render
    render.close()
    control.close()


def test_copy_rendered_image_puts_the_view_on_the_clipboard_as_displayed(windows):
    app, control, render = windows
    app.clipboard().clear()
    assert control.copy(render)
    image = app.clipboard().image()
    expected = render.view.qimage()
    assert not image.isNull()
    assert (image.width(), image.height()) == (expected.width(), expected.height())


def test_copy_from_the_menu_means_the_last_picture_not_the_control_window(windows):
    app, control, render = windows
    # the menu makes the control window the active one; Copy still means the
    # picture, and without another one in front, the render window's
    assert control._copy_window() is render


def test_copy_takes_the_figure_being_looked_at(windows):
    app, control, render = windows
    from smappy.gui.figures import ResultWindow

    window = ResultWindow("test")
    window.show_plots([Plot(lambda ax: ax.plot([0, 1], [0, 1]), name="line"),
                       Plot(lambda ax: ax.hist([1, 2, 2, 3]), name="hist")])
    window.show()
    app.processEvents()
    app.clipboard().clear()
    assert control.copy(window)
    image = app.clipboard().image()
    # twice the screen's resolution: a figure of 6.4 x 4.8 in at 100 dpi
    figure = window.figures.panes["line"].figure
    assert image.width() == pytest.approx(2 * figure.get_dpi()
                                          * figure.get_figwidth(), abs=2)
    window.close()


def test_copy_lets_a_text_field_copy_its_own_selection(windows):
    app, control, render = windows
    from PySide6.QtWidgets import QLineEdit
    from smappy.gui.app import _copy_selected_text

    field = QLineEdit("drift correction")
    field.setSelection(0, 5)
    assert _copy_selected_text(field)
    assert app.clipboard().text() == "drift"
    field.deselect()
    assert not _copy_selected_text(field)


def test_a_window_with_nothing_to_show_copies_nothing(windows):
    app, control, render = windows
    from smappy.gui.figures import ResultWindow

    empty = ResultWindow("empty")
    assert not control.copy(empty)
