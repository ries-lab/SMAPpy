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

    # the averaged stack browses in its own window, with the view controls in it
    window.browse_stack()
    assert window.stack_viewer.slice_count() == result.raw_psf.shape[0]
    window.stack_viewer.orientation_box.setCurrentText("XZ")
    assert window.stack_viewer.slice_count() == result.raw_psf.shape[1]

    path = tmp_path / "cal.h5"
    result.save(path)
    seen = []
    window.calibrated.connect(lambda p, dual: seen.append((p, dual)))
    window.saved_path = str(path)
    window._use_in_fitter()
    assert seen == [(str(path), False)]


def test_a_dual_calibration_is_offered_to_the_two_channel_fitter(app, tmp_path):
    from smappy.calibrate.qt_gui import CalibrationWindow
    window = CalibrationWindow()
    window.mode_box.setCurrentText("Dual colour")
    seen = []
    window.calibrated.connect(lambda p, dual: seen.append((p, dual)))
    window.saved_path = str(tmp_path / "cal.h5")
    window._use_in_fitter()
    assert seen == [(str(tmp_path / "cal.h5"), True)]


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


def test_the_dual_fit_quality_page_refits_with_the_two_channel_fitter(app):
    """The crash was that the window took the single-channel path for 2C.

    A dual result's beads have two channels each, so there is no one volume to
    refit; the page now goes through `smappy.dualfit`'s global fitter, which is
    what a two-colour dataset is fitted with.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_dual_calibration import synthetic
    from smappy.calibrate.dual import calibrate_dual
    from smappy.calibrate.qt_gui import CalibrationWindow
    from smappy.calibrate.unified_gui import paired_profiles
    from smappy.calibrate.validation import fit_paired_bead_diagnostics

    stack, settings = synthetic("up-down mirrored", "upper")
    result = calibrate_dual([stack], settings)
    window = CalibrationWindow()
    window.mode_box.setCurrentText("Dual colour")
    window._finished("result", result)
    assert window.bead_table.rowCount() == len(result.beads.records)

    window._finished("quality", (fit_paired_bead_diagnostics(result),
                                 paired_profiles(result)))
    assert window.tabs.currentIndex() == 2
    axes = window.pages[2][0].axes
    assert len(axes) >= 8                      # two rows of four, plus a bar
    assert "two-channel fit" in axes[0].get_title()

    # every page draws for a dual result, not only the two the crash reached
    for page in range(len(window.pages)):
        window.tabs.setCurrentIndex(page)
        window.redraw()
        assert len(window.pages[page][0].axes) > 0, page
    window.tabs.setCurrentIndex(2)
    window.redraw()

    # the averaged stacks browse as a pair, with the mirror known
    window.browse_stack()
    assert len(window.stack_viewer.volumes) == 2
    assert window.stack_viewer.linked_box.isVisible()
    window.stack_viewer.native_box.setChecked(True)
    assert window.stack_viewer.shown_volumes()[1].shape == \
        window.stack_viewer.volumes[1].shape


def test_an_auto_setting_shows_what_the_run_resolved_it_to(app):
    """"auto" says a decision will be made; afterwards it says which one."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_bead_calibration import synthetic_collection
    from smappy.calibrate.qt_gui import CalibrationWindow

    window = CalibrationWindow()
    step = window.form.fields["dz_nm"]
    assert step.auto.isChecked() and step.widget.text() == ""

    result = build_calibration(synthetic_collection())
    window._finished("result", result)
    assert step.auto.isChecked()                  # still auto ...
    assert step.value() is None                   # ... and still not a setting
    assert float(step.widget.text()) == pytest.approx(result.beads.dz_nm)
    assert "gray" in step.widget.styleSheet()


@pytest.fixture
def control(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path/"config"))
    from smappy import config
    config.load(reload=True)
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    return ControlWindow(session, RenderWindow(session))


@pytest.mark.parametrize("dual, wanted", [(False, "Localize/Spline 3D"),
                                          (True, "Localize/Spline 3D 2C")])
def test_a_saved_calibration_reaches_the_fitter_it_belongs_to(control, dual, wanted):
    """Including one whose section has never been opened, so has no panel yet."""
    control.use_calibration("/tmp/beads_cal.h5", dual)
    tab = control.tabs.widget(control.tabs.currentIndex())
    panel = next(slot.panel for instance, slot in
                 ((i, tab.slots[i.id]) for i in tab.tab.instances)
                 if instance.plugin == wanted)
    assert panel is not None
    assert panel.form.values()["model.calibration"] == "/tmp/beads_cal.h5"
    # and the other fitter was left alone
    others = [tab.slots[i.id].panel for i in tab.tab.instances
              if i.plugin.startswith("Localize/Spline") and i.plugin != wanted]
    assert all(p is None or p.form.values()["model.calibration"] == ""
               for p in others)


def test_a_calibration_with_nowhere_to_go_says_so(control):
    for tab in (control.tabs.widget(i) for i in range(control.tabs.count())):
        if getattr(tab, "slots", None):
            tab.tab.instances = [i for i in tab.tab.instances
                                 if not i.plugin.startswith("Localize/Spline")]
    control.use_calibration("/tmp/beads_cal.h5", True)
    assert "no Spline 3D 2C fitter is pinned" in control.statusBar().currentMessage()
