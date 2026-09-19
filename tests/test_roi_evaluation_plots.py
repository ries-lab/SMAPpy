"""Walking down the ROI list and seeing what the evaluators drew.

An evaluator's numbers land in a row of the site table; what it *draws* had
nowhere to go, and a figure per ROI would bury the screen after five clicks.
So each plot gets one window, which the next ROI redraws -- which is what makes
two sites comparable at all.
"""
import numpy as np
import pytest

from smappy.locs import Localizations                                  # noqa: E402
from smappy.plugins import Plugin, Result, param                       # noqa: E402
from smappy.roi_manager import ROIProject                              # noqa: E402
from smappy.roi_manager import pipeline as pipeline_module             # noqa: E402
from smappy.session import Session                                     # noqa: E402

from dataclasses import dataclass                                      # noqa: E402


@dataclass
class DrawingSettings:
    bins: int = param(4, label="bins", min=1)


class Drawing(Plugin):
    """An evaluator with a figure, which the shipped ones do not have."""
    name = "Drawing"
    path = "ROIManager/Evaluate/Drawing"
    scope = "site"
    Settings = DrawingSettings

    def run(self, ctx, settings):
        x = np.asarray(ctx.locs["x_nm"])
        return Result(text=f"{len(x)} localizations",
                      plot=lambda ax: ax.hist(x, bins=settings.bins),
                      plots={"positions": lambda ax: ax.plot(x, x)},
                      data={"n": len(x)}, settings=settings)


class Other(Plugin):
    """A second evaluator with a figure, to watch what the first one costs."""
    name = "Other"
    path = "ROIManager/Evaluate/Other"
    scope = "site"
    Settings = DrawingSettings

    def run(self, ctx, settings):
        n = len(ctx.locs)
        return Result(text=f"{n}", plot=lambda ax: ax.bar([0], [n]), data={"n": n})


class Failing(Plugin):
    name = "Failing"
    path = "ROIManager/Evaluate/Failing"
    scope = "site"
    Settings = DrawingSettings

    def run(self, ctx, settings):
        raise ValueError("not this site")


def table(centers, per=10, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(cx, 20, per) for cx, _ in centers])
    y = np.concatenate([rng.normal(cy, 20, per) for _, cy in centers])
    n = per * len(centers)
    return Localizations({"x_nm": x, "y_nm": y, "frame": np.arange(n),
                          "photons": np.full(n, 200.0),
                          "loc_precision_nm": np.full(n, 11.0)})


def steps(*plugins):
    return [pipeline_module.Step(label=p.name, path=p.path, plugin=p(),
                                 settings=p.Settings())
            for p in plugins]


def a_project():
    project = ROIProject()
    source = project.add_source(table([(0, 0), (1000, 1000)]))
    project.set_filters({})
    project.set_geometry(200)
    rois = [project.add_roi(source.id, center)
            for center in ([0, 0], [1000, 1000])]
    return project, rois


# ----------------------------------------------------------- one ROI at a time

def test_one_roi_hands_back_what_its_evaluators_drew():
    project, rois = a_project()
    record, results, _ = project.evaluate_one(rois[0].id, steps=steps(Drawing))
    assert record["steps"]["Drawing"]["values"] == {"n": 10}
    assert [p.name for p in results["Drawing"].figures()] == ["", "positions"]


def test_looking_at_a_site_records_nothing():
    """A run is a pipeline over every site; clicking down a list is not one."""
    project, rois = a_project()
    project.evaluate_one(rois[0].id, steps=steps(Drawing))
    project.evaluate_one(rois[1].id, steps=steps(Drawing))
    assert project.runs == []
    assert project.results() == []


def test_a_step_that_fails_costs_its_own_figure_and_no_more():
    project, rois = a_project()
    record, results, _ = project.evaluate_one(
        rois[0].id, steps=steps(Failing, Drawing))
    assert results["Failing"] is None
    assert results["Drawing"].figures()
    assert pipeline_module.errors(record) == {"Failing": "ValueError: not this site"}


def test_the_shipped_evaluator_draws_the_site_it_counted():
    """Three numbers cannot tell a ring from a smear; the picture can."""
    project, rois = a_project()
    _, results, _ = project.evaluate_one(
        rois[0].id,
        steps=pipeline_module.resolve(pipeline_module.default_instances()))
    assert results["Statistics"].plot is not None

    from matplotlib.figure import Figure
    ax = Figure().subplots()
    results["Statistics"].plot(ax)
    assert ax.collections and "10 localizations" in ax.get_title()


def test_the_pipeline_on_the_project_is_what_runs_by_default():
    project, rois = a_project()
    project.pipeline = pipeline_module.default_instances()
    record, results, _ = project.evaluate_one(rois[0].id)
    assert "Statistics" in results
    assert record["steps"]["Statistics"]["values"]["n_localizations"] == 10


# ------------------------------------------------------------------ the window

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def a_window(app):
    from smappy.gui.roi_window import ROIManagerWindow
    session = Session(table([(0, 0), (1000, 1000)]))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(200)
    file_id = next(iter(project.sources))
    for center in ([0, 0], [1000, 1000]):
        project.add_roi(file_id, center)
    window = ROIManagerWindow(session)
    return window, project


def two_steps():
    return steps(Drawing, Failing)


def test_the_next_roi_redraws_in_the_window_that_is_already_open(app, monkeypatch):
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    ids = list(project.rois)
    window.select(ids[0])
    site = window._site
    panes = dict(site.figures["Drawing"].panes)
    assert site.isVisible()
    assert [site.tabs.tabText(i) for i in range(site.tabs.count())] == ["Drawing"]
    assert "ROI 1" in site.windowTitle()

    window.select(ids[1])
    assert site.figures["Drawing"].panes == panes     # the same panes, redrawn
    assert "ROI 2" in site.windowTitle()


def test_every_stale_step_is_measured_but_only_the_open_one_is_drawn(app, monkeypatch):
    """The numbers are what the site table needs; the figures are what is looked
    at, and five evaluators with three figures each is why that is not the same
    thing."""
    monkeypatch.setattr(pipeline_module, "resolve",
                        lambda instances: steps(Drawing, Other))
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    window.select(list(project.rois)[0])
    site = window._site
    assert [site.tabs.tabText(i) for i in range(site.tabs.count())] == [
        "Drawing", "Other"]
    assert site.current == "Drawing"
    assert "Drawing, Other evaluated" in window.status.currentMessage()

    assert site.figures["Drawing"].panes[""].figure.axes      # looked at: drawn
    assert not site.figures["Other"].panes[""].figure.axes    # measured, not drawn

    site.tabs.setCurrentIndex(1)                              # now it is
    assert site.figures["Other"].panes[""].figure.axes


def test_an_evaluator_that_fails_says_so_on_its_tab(app, monkeypatch):
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: two_steps())
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    window.select(list(project.rois)[0])
    site = window._site
    assert "not this site" in site.messages["Failing"].text()
    assert site.pages["Failing"].currentWidget() is site.messages["Failing"]


def test_only_the_figure_being_looked_at_is_drawn(app, monkeypatch):
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    window.select(list(project.rois)[0])
    figures = window._site.figures["Drawing"]
    assert [figures.tabs.tabText(i) for i in range(figures.tabs.count())] == [
        "figure", "positions", "All"]
    assert not figures.panes["positions"].figure.axes     # not looked at yet
    figures.tabs.setCurrentIndex(1)
    assert figures.panes["positions"].figure.axes


def test_nothing_runs_until_the_plots_are_asked_for(app, monkeypatch):
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window, project = a_window(app)
    window.select(list(project.rois)[0])
    assert window._site is None


def test_what_is_current_is_not_measured_again(app, monkeypatch):
    """Scrolling back to a site already measured costs nothing."""
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    ids = list(project.rois)
    window.select(ids[0])
    window.select(ids[1])
    runs = len(project.runs)
    window.select(ids[0])                             # back again: stored
    assert len(project.runs) == runs                  # nothing new to store
    assert "drawn from the result already stored" in window.status.currentMessage()


def test_with_re_evaluation_off_a_stale_step_is_left_alone(app, monkeypatch):
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: two_steps())
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    ids = list(project.rois)
    window.select(ids[0])
    window._site.tabs.setCurrentIndex(1)              # both steps measured
    window._site.tabs.setCurrentIndex(0)
    assert not project.stale_steps(ids[0], two_steps())

    window.re_evaluate.setChecked(False)
    project.move_roi(ids[0], [500, 500])              # everything is stale now
    window.select(ids[0])
    stale = [step.label for step in project.stale_steps(ids[0], two_steps())]
    assert stale == ["Failing"]                       # only the open tab ran
    assert project.navigation["re_evaluate"] is False


def test_an_empty_pipeline_says_so_rather_than_drawing(app):
    window, project = a_window(app)
    window.project.pipeline = []
    window.evaluate_live.setChecked(True)
    window.select(list(project.rois)[0])
    assert "no evaluator" in window.status.currentMessage()


def test_whether_to_plot_is_remembered_with_the_project(app):
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    assert project.navigation["plot_evaluation"] is True
    window.refresh()
    assert window.evaluate_live.isChecked()


def test_the_manager_opens_the_pipeline_window(app):
    """Below the ROI list, where the evaluation is being looked at."""
    from smappy.gui.roi_tab import ROIHeader
    session = Session(table([(0, 0)]))
    session.show_grouped(0, False)
    header = ROIHeader(session)
    header.open_manager()
    header.manager.evaluation_button.click()
    assert header.evaluation is not None and header.evaluation.isVisible()


def test_closing_the_site_window_stops_the_measuring(app, monkeypatch):
    """Nobody is looking, so nothing should be computed for it."""
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window, project = a_window(app)
    window.evaluate_live.setChecked(True)
    ids = list(project.rois)
    window.select(ids[0])

    window._site.close()
    assert not window.evaluate_live.isChecked()
    runs = len(project.runs)
    window.select(ids[1])
    assert len(project.runs) == runs
