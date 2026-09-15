"""The Render tab while a fit is running, and the overview after a load."""
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


def block(n=500, seed=0, z=True):
    rng = np.random.default_rng(seed)
    columns = {
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "frame": rng.integers(0, 100, n).astype(np.int64),
        "photons": rng.uniform(100, 5000, n).astype(np.float32),
        "loc_precision_nm": rng.uniform(5, 25, n).astype(np.float32),
    }
    if z:
        columns["z_nm"] = rng.normal(0, 200, n).astype(np.float32)
    return Localizations(columns, {})


def fields(tab):
    return [tab.color_field.itemText(i) for i in range(tab.color_field.count())]


def test_the_colour_field_list_fills_as_a_fit_produces_columns(app):
    """A fit starts with an empty table: z_nm exists only once the first block
    is in, and a list filled at the start stayed empty for the whole run."""
    from smappy.gui.render_tab import RenderTab

    session = Session()
    tab = RenderTab(session)
    session.begin_live()
    assert fields(tab) == []                      # nothing to colour by yet

    session.append(block())
    assert "z_nm" in fields(tab) and "photons" in fields(tab)
    assert tab.color_field.currentText() == "z_nm"

    # what the user picked survives every block after that
    tab.color_field.setCurrentText("photons")
    for i in range(1, 12):
        session.append(block(seed=i))
    assert tab.color_field.currentText() == "photons"
