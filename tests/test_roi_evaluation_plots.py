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
    record, results = project.evaluate_one(rois[0].id, steps=steps(Drawing))
    assert record["steps"]["Drawing"]["values"] == {"n": 10}
    assert [name for name, _ in results["Drawing"].figures()] == ["", "positions"]


def test_looking_at_a_site_records_nothing():
    """A run is a pipeline over every site; clicking down a list is not one."""
    project, rois = a_project()
    project.evaluate_one(rois[0].id, steps=steps(Drawing))
    project.evaluate_one(rois[1].id, steps=steps(Drawing))
    assert project.runs == []
    assert project.results() == []


def test_a_step_that_fails_costs_its_own_figure_and_no_more():
    project, rois = a_project()
    record, results = project.evaluate_one(rois[0].id,
                                           steps=steps(Failing, Drawing))
    assert results["Failing"] is None
    assert results["Drawing"].figures()
    assert pipeline_module.errors(record) == {"Failing": "ValueError: not this site"}


def test_the_shipped_evaluator_draws_the_site_it_counted():
    """Three numbers cannot tell a ring from a smear; the picture can."""
    project, rois = a_project()
    _, results = project.evaluate_one(rois[0].id,
                                      steps=pipeline_module.resolve(
                                          pipeline_module.default_instances()))
    assert results["Statistics"].plot is not None

    from matplotlib.figure import Figure
    ax = Figure().subplots()
    results["Statistics"].plot(ax)
    assert ax.collections and "10 localizations" in ax.get_title()


def test_the_pipeline_on_the_project_is_what_runs_by_default():
    project, rois = a_project()
    project.pipeline = pipeline_module.default_instances()
    record, results = project.evaluate_one(rois[0].id)
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


def test_the_next_roi_redraws_the_figures_it_already_has(app, monkeypatch):
    window, project = a_window(app)
    drawn = []

    def fake(cache, key, title, draw):
        drawn.append((key, title))
        cache[key] = cache.get(key) or object()
        return cache[key]

    monkeypatch.setattr("smappy.gui.figures.draw_figure", fake)
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window.evaluate_live.setChecked(True)
    ids = list(project.rois)
    window.select(ids[0])
    first = dict(window._figures)
    window.select(ids[1])

    # two plots, two windows, and the second ROI drew into the same two
    assert len(window._figures) == 2
    assert window._figures == first
    assert [title for _, title in drawn] == [
        "Drawing: ROI 1", "Drawing: ROI 1: positions",
        "Drawing: ROI 2", "Drawing: ROI 2: positions"]


def test_nothing_is_drawn_until_it_is_asked_for(app, monkeypatch):
    window, project = a_window(app)
    monkeypatch.setattr(pipeline_module, "resolve", lambda instances: steps(Drawing))
    window.select(list(project.rois)[0])
    assert window._figures == {}


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
