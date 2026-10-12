"""A failure said where one looks: the message in red beside Run, without the
exception's class, and the traceback behind *details…* with a bug report."""
import time
import zipfile

import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.locs import Localizations                                 # noqa: E402
from smappy.plugins import Plugin, Result                             # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def a_table(n=100):
    rng = np.random.default_rng(0)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.arange(n, dtype=np.int64)}, {})


class NeedsALine(Plugin):
    """Fails the way a plugin is meant to: with a message for the reader."""
    path = "Test/Errors/NeedsALine"
    name = "NeedsALine"
    changes_locs = False
    fail = True

    def run(self, ctx, settings):
        if NeedsALine.fail:
            raise ValueError("no ROI: draw a line")
        return Result(text="measured")


def wait_for(app, condition, seconds=10.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert condition(), "timed out"


def test_the_message_is_what_the_error_says_or_its_class_when_it_says_nothing():
    from smappy.gui.errors import message_of
    assert message_of(ValueError("no ROI: draw a line")) == "no ROI: draw a line"
    assert message_of(RuntimeError()) == "RuntimeError"
    # a bare key is said as what is missing, not as "'x_nm'"
    assert message_of(KeyError("x_nm")) == "missing x_nm"
    assert "no position column" in message_of(KeyError("no position column"))


def test_a_failed_run_says_its_message_in_red_beside_run_and_the_traceback_on_request(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
    from smappy.gui.errors import ERROR_STYLE
    from smappy.gui.plugin_panel import PluginPanel
    NeedsALine.fail = True
    session = Session(a_table())
    panel = PluginPanel(NeedsALine, session)
    said = []
    panel.ended.connect(lambda kind, payload: said.append((kind, payload)))
    assert panel.details_button.isHidden()           # nothing to detail yet
    panel.start_run(ask=False)
    wait_for(app, lambda: said)
    app.processEvents()
    panel._thread.wait()

    assert panel.status.text() == "no ROI: draw a line"
    assert "ValueError" not in panel.status.text()
    assert "ValueError" not in panel.output.toPlainText()
    assert panel.status.styleSheet() == ERROR_STYLE
    assert not panel.details_button.isHidden()
    assert said[0][0] == "failed" and "Traceback" in said[0][1]

    panel.details_button.click()
    window = panel._details_window
    assert window.isVisible()
    shown = window.text.toPlainText()
    assert "Traceback" in shown and "ValueError: no ROI: draw a line" in shown
    assert window.report_button.text() == "Save a bug report…"

    # the report button goes through Help > Save a bug report's machinery,
    # with the traceback in the report
    target = tmp_path / "report.zip"
    monkeypatch.setattr(QInputDialog, "getMultiLineText",
                        staticmethod(lambda *a, **k: ("it broke", True)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    window.report_button.click()
    with zipfile.ZipFile(target) as z:
        report = z.read("report.txt").decode()
    assert "it broke" in report and "ValueError: no ROI: draw a line" in report
    window.close()

    # the next run that works takes the red and the button away again
    NeedsALine.fail = False
    said.clear()
    panel.start_run(ask=False)
    wait_for(app, lambda: said)
    app.processEvents()
    panel._thread.wait()
    assert panel.status.text() == "done" and panel.status.styleSheet() == ""
    assert panel.details_button.isHidden()


def test_a_panel_with_a_long_error_and_its_details_button_still_fits_the_column(app):
    from smappy import plugins
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.gui.widgets import CONTROL_WIDTH
    panel = PluginPanel(plugins.get("Localize/Gaussian 2D"), Session())
    narrow = panel.minimumSizeHint().width()
    panel._on_failed("the file has no frames: " + "a very long reason " * 10, "Traceback")
    assert not panel.details_button.isHidden()
    assert panel.minimumSizeHint().width() <= max(narrow, CONTROL_WIDTH - 40)


def test_a_file_that_will_not_open_says_so_in_red_without_a_modal_box(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from smappy.gui.app import ControlWindow, RenderWindow
    boxes = []
    for kind in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, kind,
                            staticmethod(lambda *a, k=kind, **kw: boxes.append(k)))
    broken = tmp_path / "broken.hdf5"
    broken.write_bytes(b"not an HDF5 file at all")
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    assert control.load_error.isHidden()
    control.load_paths([str(broken)])
    wait_for(app, lambda: control._task is None and not control._queue)
    app.processEvents()

    assert not boxes
    assert not control.load_error.isHidden()
    text = control.load_error_text.text()
    assert text.startswith("could not open broken.hdf5: ")
    control.load_error_details.click()
    assert "Traceback" in control._load_error_window.text.toPlainText()
    control._load_error_window.close()
    control.close()
    render.close()
