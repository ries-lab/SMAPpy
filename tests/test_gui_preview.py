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
        changes_locs = False
        Settings = None

        def run(self, ctx, settings):
            return Result()

    assert PluginPanel(Plain, Session()).preview_button is None


def test_a_method_greys_out_the_other_methods_parameters(app):
    """A number that does nothing must not look like a number that does."""
    from smappy import plugins
    from smappy.gui.plugin_panel import PluginPanel

    panel = PluginPanel(plugins.get("Analysis/Dual-colour/Assign colours"), Session())
    form = panel.form
    # the panel greys on the way up, before anything is edited
    assert form.fields["exclusion"].isEnabled()
    assert not form.fields["crosstalk"].isEnabled()
    assert not form.labels["crosstalk"].isEnabled()

    form.fields["mode"].set("probabilistic")
    panel._react("mode")
    assert form.fields["crosstalk"].isEnabled() and form.labels["crosstalk"].isEnabled()
    assert form.fields["tolerance"].isEnabled()
    assert form.fields["keep_tails"].isEnabled()
    assert not form.fields["exclusion"].isEnabled()
    # greying is cosmetic: the value is still there and still comes back
    assert form.value().exclusion == AssignColorSettings().exclusion


def test_a_preview_that_is_not_of_a_frame_gets_no_frame_number(app):
    """A colour assignment previews the whole selection; there is no frame."""
    from smappy import plugins
    from smappy.gui.plugin_panel import PluginPanel

    panel = PluginPanel(plugins.get("Analysis/Dual-colour/Assign colours"), Session())
    assert panel.preview_button is not None and panel.preview_frame is None


def test_an_auto_field_shows_what_auto_resolves_to_in_grey(app):
    from dataclasses import dataclass
    from smappy.gui.widgets import note_colour
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
    assert note_colour() in field.widget.styleSheet()
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
    assert panel.result_button.isEnabled()
    assert result.data["frame"] == 1 and result.data["candidates"] == 2
    plt.close("all")


def test_the_figures_share_one_window_and_draw_when_looked_at(app):
    """Several figures are tabs of one window, and only one of them is drawn.

    Each press used to make a new pyplot window, so previewing a few times
    buried the screen in windows of the same plot; and every figure was drawn
    whether or not anyone was going to look at it.
    """
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins import Plugin, Result

    drawn = []

    class Plotter(Plugin):
        changes_locs = False
        Settings = None

        def run(self, ctx, settings):
            return Result(plot=lambda ax: drawn.append("first"),
                          plots={"second": lambda ax: drawn.append("second")})

    panel = PluginPanel(Plotter, Session())
    before = set(plt.get_fignums())
    panel._on_done(panel.plugin.run(None, None))   # a run opens its figures
    window = panel._window
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "figure", "second", "All"]
    assert drawn == ["first"]                    # the one on screen, no more
    assert not set(plt.get_fignums()) - before   # and pyplot holds none of it

    panel.plot()                                 # the same window, redrawn
    assert panel._window is window
    assert drawn == ["first", "first"]

    window.tabs.setCurrentIndex(1)               # looked at: drawn now
    assert drawn == ["first", "first", "second"]
    window.tabs.setCurrentIndex(0)               # already drawn: not again
    assert len(drawn) == 3


def test_a_tab_can_be_taken_out_into_its_own_window(app):
    """Tabs tidy the screen; tearing one off is how two are compared."""
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins import Plugin, Result

    drawn = []

    class Plotter(Plugin):
        changes_locs = False
        Settings = None

        def run(self, ctx, settings):
            return Result(plot=lambda ax: drawn.append("first"),
                          plots={"second": lambda ax: drawn.append("second")})

    panel = PluginPanel(Plotter, Session())
    panel._on_done(panel.plugin.run(None, None))
    panel.plot()
    window = panel._window

    window.detach(1)
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "figure", "All"]
    assert "second" in window.detached
    assert drawn[-1] == "second"                 # on screen, so drawn at once

    drawn.clear()
    panel.plot()                                 # both are on screen now
    assert sorted(drawn) == ["first", "second"]

    window.detached["second"].close()            # closed: the tab comes back
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "figure", "second", "All"]


# A plugin with a part per algorithm, of which only the chosen one is read.

from dataclasses import dataclass, field                              # noqa: E402

from smappy.plugins import FormState, Plugin, Result, register        # noqa: E402


@dataclass
class _DBSCANPart:
    radius_nm: float = param(50.0, label="radius")


@dataclass
class _VoronoiPart:
    cut: float = param(2.0, label="cut")


@dataclass
class _AlgorithmSettings:
    algorithm: str = param("dbscan", label="algorithm",
                           choices=(("dbscan", "DBSCAN"), ("voronoi", "Voronoi")))
    dbscan: _DBSCANPart = param(default_factory=_DBSCANPart, label="DBSCAN")
    voronoi: _VoronoiPart = param(default_factory=_VoronoiPart, label="Voronoi")
    min_size: int = param(10, label="at least")
    seed: int = param(0, label="seed", advanced=True)


@register("Test/Form/Algorithms")
class _Algorithms(Plugin):
    changes_locs = False
    Settings = _AlgorithmSettings

    def describe(self, ctx, settings):
        dbscan = settings.algorithm == "dbscan"
        return FormState(shown={"dbscan": dbscan, "voronoi": not dbscan,
                                "min_size": dbscan, "seed": dbscan})

    def run(self, ctx, settings):
        return Result()


def test_only_the_chosen_algorithms_settings_are_shown(app):
    """The parts of the algorithms not chosen are hidden, not greyed, and
    "more" goes with its last row; hiding keeps every value."""
    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(_Algorithms, Session())
    form = panel.form
    # isHidden, not isVisible: the panel is never shown on screen here
    assert not form.sections["dbscan"].isHidden()
    assert form.sections["voronoi"].isHidden()
    assert form.row_layouts["min_size"].isRowVisible(form.fields["min_size"])

    form.fields["algorithm"].set("voronoi")
    panel._react("algorithm")
    assert form.sections["dbscan"].isHidden()
    assert not form.sections["voronoi"].isHidden()
    assert not form.row_layouts["min_size"].isRowVisible(form.fields["min_size"])
    assert form.more_section.isHidden()
    assert form.value().dbscan.radius_nm == 50.0 and form.value().min_size == 10

    form.fields["algorithm"].set("dbscan")
    panel._react("algorithm")
    assert not form.sections["dbscan"].isHidden()
    assert not form.more_section.isHidden()


def test_a_chain_step_hides_what_its_plugin_hides(app):
    from smappy.chain import ChainSpec, Step, chain_class
    from smappy.gui.plugin_panel import PluginPanel
    cls = chain_class(ChainSpec(steps=[Step(plugin="Test/Form/Algorithms",
                                            values={"algorithm": "voronoi"})]))
    panel = PluginPanel(cls, Session())
    step = panel.form.fields["algorithms"]          # the step, by its title
    assert step.sections["dbscan"].isHidden()
    assert not step.sections["voronoi"].isHidden()
