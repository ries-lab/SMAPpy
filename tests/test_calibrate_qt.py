"""The Qt calibration window: the widgets are new, the plots are the Tk ones."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from smappy.calibrate.core import CalibrationSettings, build_calibration   # noqa: E402
from smappy.calibrate.dual import DualColorSettings                        # noqa: E402
from smappy.plugins import param_specs                                     # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_settings_forms_cover_both_dataclasses_on_this_python():
    """`int | None` cannot be evaluated before 3.10; the form must not care."""
    from smappy.calibrate.qt_gui import SETTING_INFO
    single = param_specs(CalibrationSettings, SETTING_INFO)
    dual = param_specs(DualColorSettings, SETTING_INFO)
    assert set(single) == {f.name for f in CalibrationSettings.__dataclass_fields__.values()}
    assert dual["split_position"].optional and dual["split_position"].type is int
    assert dual["layout"].info.choices


def test_window_draws_a_real_result_and_offers_it(app, tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_bead_calibration import synthetic_collection
    from smappy.calibrate.qt_gui import CalibrationWindow

    window = CalibrationWindow()
    result = build_calibration(synthetic_collection())
    window._finished("result", result)
    assert window.bead_table.rowCount() == len(result.beads.records)
    assert len(window.pages[0][0].axes) > 0          # the overview drew

    window.bead_table.selectRow(2)
    window.toggle_excluded()
    assert window.excluded == {2}
    window.toggle_excluded()
    assert window.excluded == set()

    # the fit-quality page, through the same path the worker uses
    from smappy.calibrate.validation import aligned_midline_profiles, fit_bead_diagnostics
    window._finished("quality", (fit_bead_diagnostics(result), aligned_midline_profiles(result)))
    assert window.tabs.currentIndex() == 2 and len(window.pages[2][0].axes) > 0

    path = tmp_path / "cal.h5"
    result.save(path)
    seen = []
    window.calibrated.connect(seen.append)
    window.saved_path = str(path)
    window._use_in_fitter()
    assert seen == [str(path)]


def test_mode_switch_keeps_the_shared_settings(app):
    from smappy.calibrate.qt_gui import CalibrationWindow
    window = CalibrationWindow(settings=CalibrationSettings(roi_size=19, smooth_z_nm=15.0))
    assert window.settings().roi_size == 19
    window.mode_box.setCurrentText("Dual colour")
    dual = window.settings()
    assert isinstance(dual, DualColorSettings)
    assert dual.roi_size == 19 and dual.smooth_z_nm == 15.0
    window.mode_box.setCurrentText("Single channel")
    assert window.settings().roi_size == 19
