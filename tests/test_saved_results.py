"""What a tool worked out is kept with the file, and can be drawn again.

A drift correction subtracts a curve and the corrected table no longer says
what the curve was.  Opening that file next week and pressing *Plot* has to
show it, which means the curve travels in the file and the panel picks it up.
"""
import numpy as np
import pytest

from smappy.drift import Drift, DriftSettings                          # noqa: E402
from smappy.io.formats import FileInfo                                 # noqa: E402
from smappy.io.hdf5 import load_results, save_results                  # noqa: E402
from smappy.locs import Localizations                                  # noqa: E402
from smappy.plugins import Result                                      # noqa: E402
from smappy.plugins.drift_comet import CometDrift                      # noqa: E402
from smappy.session import Session                                     # noqa: E402

COMET = "Analysis/Drift/COMET"


def table(n=200):
    rng = np.random.default_rng(0)
    return Localizations({"x_nm": rng.uniform(0, 1000, n),
                          "y_nm": rng.uniform(0, 1000, n),
                          "frame": np.arange(n) % 50,
                          "photons": np.full(n, 300.0),
                          "xy_err_nm": np.full(n, 10.0)})


def a_drift(frames=50):
    curve = np.zeros((frames, 3))
    curve[:, 0] = np.linspace(0, 40, frames)
    return Drift(curve, DriftSettings(), n_used=1234)


def test_a_curve_survives_being_written_as_plain_data():
    drift = a_drift()
    back = Drift.from_dict(drift.to_dict())
    assert np.allclose(back.drift, drift.drift)
    assert back.n_used == 1234
    assert back.settings.max_drift_nm == drift.settings.max_drift_nm


def test_a_curve_from_a_newer_version_keeps_its_numbers():
    """A setting this version has never heard of costs that setting, no more."""
    saved = a_drift().to_dict()
    saved["settings"]["something_new"] = 7
    back = Drift.from_dict(saved)
    assert len(back) == 50
    assert back.settings.max_drift_nm == DriftSettings().max_drift_nm

    saved["settings"] = "not a settings dict at all"
    assert Drift.from_dict(saved).settings is None


def test_running_the_plugin_records_the_curve_on_the_session():
    session = Session(table())
    drift = a_drift()
    session.apply(CometDrift(), Result(text=str(drift), data={"drift": drift}))
    assert COMET in session.results
    restored = session.restore_result(CometDrift())
    assert np.allclose(restored.data["drift"].drift, drift.drift)
    assert restored.figures()


def test_a_plugin_that_keeps_nothing_records_nothing():
    session = Session(table())
    session.apply(CometDrift(), Result(text="no drift in this one"))
    assert session.results == {}


def test_the_curve_comes_back_with_the_file(tmp_path):
    session = Session(table())
    session.add_file(table(), FileInfo("a.hdf5", str(tmp_path / "a.hdf5"), "smappy"))
    drift = a_drift()
    session.apply(CometDrift(), Result(text=str(drift), data={"drift": drift}))
    path = session.save(tmp_path / "corrected.hdf5", gui_state=False)

    opened = Session()
    opened.load(path)
    assert COMET in opened.results
    assert np.allclose(opened.restore_result(CometDrift()).data["drift"].drift,
                       drift.drift)


def test_opening_another_file_forgets_the_last_one(tmp_path):
    session = Session(table())
    session.apply(CometDrift(), Result(data={"drift": a_drift()}))
    plain = tmp_path / "plain.hdf5"
    Session(table()).save(plain, gui_state=False)
    session.load(plain)
    assert session.results == {}


def test_a_file_with_nothing_saved_reads_as_nothing(tmp_path):
    path = tmp_path / "plain.hdf5"
    Session(table()).save(path, gui_state=False)
    assert load_results(path) == {}
    assert load_results(tmp_path / "not-a-file.hdf5") == {}


def test_saved_results_can_be_cleared_again(tmp_path):
    path = tmp_path / "plain.hdf5"
    Session(table()).save(path, gui_state=False)
    save_results(path, {"a/b": {"data": {"x": 1}}})
    assert load_results(path) == {"a/b": {"data": {"x": 1}}}
    save_results(path, {})
    assert load_results(path) == {}


def test_the_panel_offers_the_plot_of_a_run_that_is_over(tmp_path):
    """The point of all of it: the Plot button, on a freshly opened file."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.plugin_panel import PluginPanel
    QApplication.instance() or QApplication([])

    session = Session(table())
    session.apply(CometDrift(), Result(text="drifted", data={"drift": a_drift()}))
    path = session.save(tmp_path / "corrected.hdf5", gui_state=False)

    opened = Session()
    opened.load(path)
    panel = PluginPanel(CometDrift, opened)
    assert panel.plot_button.isEnabled()
    assert "ran before" in panel.status.text()
    assert panel.result.figures()


def test_the_panel_picks_up_a_file_opened_after_it_was_built(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.plugin_panel import PluginPanel
    QApplication.instance() or QApplication([])

    session = Session(table())
    session.apply(CometDrift(), Result(text="drifted", data={"drift": a_drift()}))
    path = session.save(tmp_path / "corrected.hdf5", gui_state=False)

    opened = Session()
    panel = PluginPanel(CometDrift, opened)
    assert not panel.plot_button.isEnabled()
    opened.load(path)
    assert panel.plot_button.isEnabled()
