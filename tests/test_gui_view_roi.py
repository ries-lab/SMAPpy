"""The drawn region's outline, and how wide a tab is allowed to make the window."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.locs import Localizations                                 # noqa: E402
from smappy.regions import Region                                     # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=2000):
    rng = np.random.default_rng(0)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "frame": rng.integers(0, 100, n).astype(np.int64),
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "loc_precision_nm": rng.uniform(5, 25, n).astype(np.float32),
    }, {"units": "nm"})


@pytest.mark.parametrize("region", [
    Region.rect(0, 0, 1000, 1000),
    Region.line((0, 0), (1000, 1000), 200),
    Region("polygon", [(0, 0), (1000, 0), (500, 900)]),
])
def test_every_region_gets_the_dark_line_under_its_outline(app, region):
    from smappy.gui.render_view import RenderView
    view = RenderView(Session(table()))
    view._show_roi(region)
    assert view.roi_halo is not None
    assert view.roi_halo.path().elementCount() > 2
    assert view.roi_halo.zValue() < view.roi_item.zValue()

    view.roi_item.sigRegionChanged.emit(view.roi_item)   # dragged: it follows
    assert view.roi_halo.path().elementCount() > 2

    view._show_roi(None)
    assert view.roi_halo is None


def test_the_roi_tab_never_widens_the_control_window(app):
    from smappy.gui.roi_tab import ROITab
    from smappy.gui.widgets import CONTROL_WIDTH
    tab = ROITab(Session(table()))
    tab.file_count.setText("1. " + "a_long_experiment_name" * 4 + "_sml.hdf5 - 128 ROIs")
    assert tab.sizeHint().width() <= CONTROL_WIDTH
