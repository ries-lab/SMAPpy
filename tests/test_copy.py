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


def test_copy_window_as_image_takes_any_window_its_figure_sharp_else_itself(windows):
    """Edit > Copy window as image: a window that says nothing -- a stack viewer -- gives its figure in
    view, drawn again at twice the screen's resolution, else itself."""
    app, control, render = windows
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QLabel, QMainWindow
    from smappy.gui.canvas import canvas_class

    figure = Figure(figsize=(4, 3))
    figure.subplots().plot([0, 1], [1, 0])
    with_figure = QMainWindow()
    with_figure.setCentralWidget(canvas_class()(figure))
    with_figure.resize(500, 400)
    with_figure.show()
    plain = QMainWindow()
    plain.setCentralWidget(QLabel("a window"))
    plain.resize(300, 200)
    plain.show()
    app.processEvents()
    assert control.copy_window_image(with_figure)
    image = app.clipboard().image()
    assert image.width() == pytest.approx(2 * figure.get_dpi() * figure.get_figwidth(), abs=2)
    assert control.copy_window_image(plain)
    assert not app.clipboard().image().isNull()
    # Cmd+C leaves a window that is not a picture alone: there it is the selected text's
    app.clipboard().setText("kept")
    assert not control.copy(plain)
    assert app.clipboard().text() == "kept"
    with_figure.close()
    plain.close()


def test_the_psf_calibration_window_copies_the_page_in_front_as_an_image_not_on_cmd_c(windows):
    app, control, render = windows
    from smappy.calibrate.psf_window import PSFCalibrationWindow

    window = PSFCalibrationWindow([])
    window.show()
    app.processEvents()
    assert control.copy_window_image(window)     # the Stacks page, as it is shown
    assert not app.clipboard().image().isNull()
    assert not control.copy(window)
    window.close()
