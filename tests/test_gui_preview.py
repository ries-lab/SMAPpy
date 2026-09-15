"""The panel's Preview button, and what an "auto" field shows while it is auto."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt     # noqa: E402

from smappy.gui.params import SettingsForm                            # noqa: E402
from smappy.plugins import param, param_specs                         # noqa: E402
from smappy.plugins.assign_colors import AssignColorSettings          # noqa: E402
from smappy.plugins.fit import GaussianFit, SplineFit                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_only_a_plugin_that_can_preview_grows_the_button(app):
    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(GaussianFit, Session())
    assert panel.preview_button is not None and panel.preview_frame is not None

    from smappy.plugins import Plugin, Result

    class Plain(Plugin):
        Settings = None

        def run(self, ctx, settings):
            return Result()

    assert PluginPanel(Plain, Session()).preview_button is None


def test_a_method_greys_out_the_other_methods_parameters(app):
    """A number that does nothing must not look like a number that does."""
    from smappy import plugins
    from smappy.gui.plugin_panel import PluginPanel

    panel = PluginPanel(plugins.get("Analysis/Dual-Color/AssignColors"), Session())
    form = panel.form
    # the panel greys on the way up, before anything is edited
    assert form.fields["exclusion"].isEnabled()
    assert not form.fields["crosstalk"].isEnabled()
    assert not form.labels["crosstalk"].isEnabled()

    form.fields["mode"].set("probabilistic")
    panel._react("mode")
    assert form.fields["crosstalk"].isEnabled() and form.labels["crosstalk"].isEnabled()
    assert form.fields["tolerance"].isEnabled()
    assert not form.fields["exclusion"].isEnabled()
    # greying is cosmetic: the value is still there and still comes back
    assert form.value().exclusion == AssignColorSettings().exclusion


def test_a_preview_that_is_not_of_a_frame_gets_no_frame_number(app):
    """A colour assignment previews the whole selection; there is no frame."""
    from smappy import plugins
    from smappy.gui.plugin_panel import PluginPanel

    panel = PluginPanel(plugins.get("Analysis/Dual-Color/AssignColors"), Session())
    assert panel.preview_button is not None and panel.preview_frame is None


def test_an_auto_field_shows_what_auto_resolves_to_in_grey(app):
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class Settings:
        offset: Optional[float] = param(None, unit="ADU")

    form = SettingsForm(Settings, param_specs(Settings))
    field = form.fields["offset"]
    assert field.auto.isChecked() and field.value() is None

    form.set_hints({"offset": 500.0})
    assert field.widget.text() == "500"          # the number is on show...
    assert field.value() is None                 # ...but it is still auto
    assert "gray" in field.widget.styleSheet()
    assert "auto: 500" in field.toolTip()

    # taking it off auto keeps the number, now as the user's own
    field.auto.setChecked(False)
    assert field.value() == 500.0
    assert field.widget.styleSheet() == ""

    # a hint for a field that is not there is ignored, not raised
    form.set_hints({"nothing.at.all": 1})


def test_a_preview_draws_without_touching_the_session(app, tmp_path):
    import tifffile
    from smappy.gui.plugin_panel import PluginPanel

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:40, :40]
    frames = np.full((3, 40, 40), 100.0)
    for y, x in ((10, 12), (28, 30)):
        frames += 900*np.exp(-((yy-y)**2+(xx-x)**2)/(2*1.3**2))
    path = tmp_path/"run_MMStack.ome.tif"
    tifffile.imwrite(path, rng.poisson(frames).astype(np.uint16))

    session = Session()
    panel = PluginPanel(GaussianFit, session)
    panel.form.set_values({"source.path": str(path), "camera.conversion": 1.0,
                           "camera.offset": 100.0, "camera.pixelsize_um": 0.1,
                           "camera.em_on": False})
    panel.preview_frame.setValue(1)
    result = panel.plugin.preview(session.context(), panel.form.value(),
                                  frame=panel.preview_frame.value())
    panel._job = "preview"
    panel._on_done(result)
    assert len(session.locs) == 0                # nothing was applied
    assert panel.plot_button.isEnabled()
    assert result.data["frame"] == 1 and result.data["candidates"] == 2
    plt.close("all")
