"""What the render view says over the picture: a hint while it is empty,
and the colour scale of a layer coloured by a column."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 5000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 5000, n).astype(np.float32),
        "z_nm": rng.uniform(-300, 300, n).astype(np.float32),
        "frame": rng.integers(0, 1000, n).astype(np.int64),
        "xy_err_nm": rng.uniform(5, 20, n).astype(np.float32),
    }, {"units": "nm"})


def test_an_empty_view_says_how_to_get_data_and_stops_once_a_table_is_open(app):
    from smappy.gui.app import RenderWindow
    session = Session()
    window = RenderWindow(session)
    window.show()
    app.processEvents()
    hint = window.view.empty_hint
    assert hint.isVisible() and "File > Open" in hint.text()
    assert "Simulate" in hint.text() and not window.view.scalebar.isVisible()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    app.processEvents()
    assert not hint.isVisible()
    window.close()


def test_colouring_by_a_column_shows_its_scale_and_intensity_shows_none(app):
    from smappy.gui.app import RenderWindow
    from smappy.gui.render_tab import RenderTab
    session = Session()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    window = RenderWindow(session)
    tab = RenderTab(session, window.view)
    window.show()
    app.processEvents()
    bar = window.view.colour_bar
    assert not bar.isVisible()                       # intensity: no scale
    tab.color.setCurrentIndex(1)                     # colour by a column
    tab.color_field.setCurrentText("z_nm")
    app.processEvents()
    assert bar.isVisible() and bar.field == "z_nm"
    title, top, bottom = bar.labels()
    lo, hi = session.layers[0].state.settings.color_range
    assert title == "z_nm" and top == f"{hi:.4g} nm" and bottom == f"{lo:.4g} nm"
    # the range typed in the tab is the range on the bar
    tab.color_lo.setText("-100")
    tab.color_hi.setText("250")
    tab.color_lo.editingFinished.emit()
    assert bar.labels()[1:] == ("250 nm", "-100 nm")
    # a column that is not a length is counted in itself: no unit
    tab.color_field.setCurrentText("frame")
    assert bar.field == "frame" and not bar.labels()[1].endswith(" frame")
    bar.grab()                                       # paints without error
    tab.color.setCurrentIndex(0)
    assert not bar.isVisible()
    window.close()
