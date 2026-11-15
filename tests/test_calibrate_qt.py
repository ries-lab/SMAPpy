"""A saved calibration handed to its fitter (`ControlWindow.use_calibration`, which the PSF calibration
window's *Save* calls), and the bead settings' forms.  The old bead calibration window these tests
began with was retired in 2026-11; the PSF calibration window's are in `tests/uipsf_tests/test_psf_window.py`."""
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from smappy.calibrate.core import CalibrationSettings                      # noqa: E402
from smappy.calibrate.dual import DualColorSettings                        # noqa: E402
from smappy.plugins import param_specs                                     # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_settings_forms_cover_both_dataclasses_on_this_python():
    """`int | None` in a settings dataclass is read as an optional int."""
    single = param_specs(CalibrationSettings)
    dual = param_specs(DualColorSettings)
    assert set(single) == {f.name for f in CalibrationSettings.__dataclass_fields__.values()}
    geometry = dual["channels"].children
    assert geometry["split_position"].optional and geometry["split_position"].type is int
    assert geometry["layout"].info.choices
    assert dual["roi_size"].info.label == "bead box"     # inherited, labelled



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


def test_an_elliptical_gaussian_calibration_goes_to_the_elliptical_fit_after_the_fit(control, tmp_path):
    """Its curves turn the widths Gaussian 2D measures into z: the fitter gets
    *elliptical* ticked and its *after the fit* astigmatic z on, with the file."""
    from smappy.astigmatic_z import simulator_calibration
    path = simulator_calibration().save(tmp_path / "beads_astigz.h5")
    control.use_calibration(str(path), False)
    tab = control.tabs.widget(control.tabs.currentIndex())
    panel = next(tab.slots[i.id].panel for i in tab.tab.instances
                 if i.plugin == "Localize/Gaussian 2D")
    values = panel.form.values()
    assert values["model.elliptical"] is True and values["finish.astigmatic_z"] is True
    assert values["finish.astigmatic.path"] == str(path)
    assert values["finish.astigmatic.calibration"] == "file"
    assert "Gaussian 2D fitter now uses" in control.statusBar().currentMessage()
