"""What the render view says over the picture: a hint while it is empty."""
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
