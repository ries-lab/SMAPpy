"""What a run hands back reaches the windows that show it: the session says
what changed, and the panel's progress line is replaced whole."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.session import Session


def table(n=200):
    rng = np.random.default_rng(1)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.arange(n, dtype=np.int64),
                          "photons": rng.uniform(100, 2000, n)}, {})


class HandsBackAProject(Plugin):
    """As a chain does: the ROI manager's project as its steps left it."""
    path = "Test/Project"
    name = "Project"
    changes_locs = False

    def run(self, ctx, settings):
        return Result(text="done", roi_project=ctx.session.rois.to_dict())


def test_a_project_handed_back_tells_the_windows_the_rois_changed():
    """The Regions tab read "0 ROIs" after the NPC Analysis chain had found
    167: `Session.apply` loaded the project without saying so."""
    session = Session()
    session.set_locs(table())
    heard = []
    session.on_change(heard.append)
    plugin = HandsBackAProject()
    session.apply(plugin, plugin.run(session.context(), None))
    assert "rois" in heard


def test_a_misspelt_request_to_the_session_fails_where_the_result_is_made():
    """What a result asks of the session is a field of `Result`, not a key of
    `data` that `Session.apply` looks up: a key misspelt there did nothing."""
    with pytest.raises(TypeError, match="bound"):
        Result(text="hid some", bound={"use": (0.5, 2.0)})


def test_the_plugins_own_numbers_ask_the_session_nothing():
    """A key of `data` that once named an effect is only the plugin's own."""
    session = Session()
    session.set_locs(table())
    before = dict(session.layers[0].ranges)
    session.apply(Measures(), Result(text="measured",
                                     data={"bounds": {"photons": (1e9, 2e9)},
                                           "layers": [{"bounds": {}}]}))
    assert dict(session.layers[0].ranges) == before
    session.apply(Measures(), Result(text="hid", bounds={"photons": (500.0, 2e9)}))
    assert session.layers[0].ranges["photons"] == (500.0, 2e9)


class Measures(Plugin):
    path = "Test/Measures"
    name = "Measures"
    changes_locs = False

    def run(self, ctx, settings):
        return Result(text="measured")


def test_a_progress_line_that_wraps_is_replaced_whole():
    """A progress line long enough to wrap left its first visual line behind
    ("frame 200 of 2,000 ... 0 frame frame 600 ...")."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(Measures, Session())
    panel.output.setFixedWidth(120)
    panel.show()
    app.processEvents()
    panel.output.appendPlainText("summary")
    long = ", ".join(f"frame {i} of 2,000" for i in range(12))
    panel._on_progress("first " + long)
    app.processEvents()
    panel._on_progress("second " + long)
    app.processEvents()
    assert panel.output.toPlainText() == "summary\nsecond " + long
    panel.close()
