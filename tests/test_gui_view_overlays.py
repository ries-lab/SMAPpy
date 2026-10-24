"""What the render view says over the picture: a hint while it is empty,
and the colour scale of a layer coloured by a column."""
import dataclasses

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
    bars = window.view.colour_bars
    assert not bars.isVisible()                      # intensity: no scale
    tab.color.setCurrentIndex(1)                     # colour by a column
    tab.color_field.setCurrentText("z_nm")
    app.processEvents()
    assert bars.isVisible() and len(bars.bars) == 1
    bar = bars.bars[0]
    title, top, bottom = bar.labels()
    lo, hi = session.layers[0].state.settings.color_range
    assert title == "layer 1\nz_nm"           # the layers, the column under them
    # no unit at the ends: the title's column name says it
    from smappy.picture import bar_end
    assert top == bar_end(hi) and bottom == bar_end(lo) and "nm" not in top
    # whole numbers from 100 up, a large count written out, not as 2e+04
    assert bar_end(20000.0) == "20000" and bar_end(-299.3456) == "-299"
    assert bar_end(0.25678) == "0.257"
    # the range typed in the tab is the range on the bar
    tab.color_lo.setText("-100")
    tab.color_hi.setText("250")
    tab.color_lo.editingFinished.emit()
    assert bars.bars[0].labels()[1:] == ("250", "-100")
    # options > colour bar hides it, and the saved picture with it
    window.toolbar.colour_bar_action.setChecked(False)
    assert not bars.isVisible()
    assert window.view.quick_settings().colour_bars is False
    window.toolbar.colour_bar_action.setChecked(True)
    assert bars.isVisible()
    # a column that is not a length is counted in itself: no unit
    tab.color_field.setCurrentText("frame")
    bar = bars.bars[0]
    assert bar.field == "frame" and not bar.labels()[1].endswith(" frame")
    bar.grab()                                       # paints without error
    tab.color.setCurrentIndex(0)
    assert not bars.isVisible()
    window.close()


def colour(layer, field, lut="turbo", color_range=(0.0, 100.0)):
    layer.state.settings = dataclasses.replace(
        layer.state.settings, color_field=field, color_range=color_range)
    layer.state.display = dataclasses.replace(layer.state.display, lut=lut)


def test_each_scale_has_its_bar_and_layers_on_one_scale_share_it(app):
    from smappy.gui.app import RenderWindow
    session = Session()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    session.add_layer(like=0)
    window = RenderWindow(session)
    window.show()
    bars = window.view.colour_bars
    first, second = session.layers[:2]

    colour(first, "z_nm", color_range=(-300.0, 300.0))
    colour(second, "frame", color_range=(0.0, 1000.0))
    session.changed("layer")
    titles = [bar.labels()[0] for bar in bars.bars]
    assert titles == [f"{first.name}\nz_nm", f"{second.name}\nframe"]
    assert bars.isVisible() and all(bar.isVisible() for bar in bars.bars)
    # stacked, not on top of each other
    assert bars.bars[1].y() >= bars.bars[0].y() + bars.bars[0].height()

    colour(second, "z_nm", color_range=(-300.0, 300.0))
    session.changed("layer")
    assert [bar.labels()[0] for bar in bars.bars] == \
        [f"{first.name}, {second.name}\nz_nm"]

    # the same column through another LUT is another scale
    colour(second, "z_nm", lut="jet", color_range=(-300.0, 300.0))
    session.changed("layer")
    assert len(bars.bars) == 2
    window.close()
