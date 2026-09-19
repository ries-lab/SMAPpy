"""What is still current, per step: the data *and* the parameters.

A record used to carry one signature over the ROI's inputs, so editing an
evaluator's parameter left every stored number looking perfectly up to date --
the one case where being out of date matters most.  Now every step carries its
own, and the two things that go out of date do so independently: an ROI that
moves invalidates all of its steps, an evaluator that is edited invalidates
that evaluator over every ROI.
"""
import numpy as np
import pytest

from smappy.locs import Localizations                                  # noqa: E402
from smappy.roi_manager import ROIProject                              # noqa: E402
from smappy.roi_manager import pipeline as pipeline_module             # noqa: E402
from smappy.workspace import Instance                                  # noqa: E402

STATISTICS = "ROIManager/Evaluate/Statistics"


def table(centers, per=4, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(c, 5, per) for c in centers])
    n = per * len(centers)
    return Localizations({"x_nm": x, "y_nm": np.zeros(n), "frame": np.arange(n),
                          "photons": np.full(n, 200.0),
                          "loc_precision_nm": np.full(n, 11.0)})


def a_project(rois=(0, 1000)):
    project = ROIProject()
    source = project.add_source(table(rois))
    project.set_filters({})
    project.set_geometry(200)
    for center in rois:
        project.add_roi(source.id, [center, 0])
    project.pipeline = [Instance(plugin=STATISTICS)]
    return project, list(project.rois)


def states(project, roi_id):
    return {label: state for label, (_, state) in project.entries(roi_id).items()}


# ---------------------------------------------------------- what goes stale

def test_an_edited_parameter_makes_the_stored_numbers_out_of_date():
    project, ids = a_project()
    project.evaluate()
    assert states(project, ids[0]) == {"Statistics": "current"}
    assert len(project.results()) == 2

    project.pipeline[0].values = {"precision_column": "photons"}
    assert states(project, ids[0]) == {"Statistics": "stale"}
    assert project.results() == []                 # not silently reused
    assert set(project.needs_evaluation()) == set(ids)


def test_renaming_a_step_does_not_invalidate_anything():
    """The label names the columns; it is not part of the measurement."""
    project, ids = a_project()
    project.evaluate()
    project.pipeline[0].label = "precision"
    assert states(project, ids[0]) == {"precision": "current"}
    assert len(project.results()) == 2


def test_moving_an_roi_invalidates_that_roi_and_no_other():
    project, ids = a_project()
    project.evaluate()
    project.move_roi(ids[0], [500, 0])
    assert states(project, ids[0]) == {"Statistics": "stale"}
    assert states(project, ids[1]) == {"Statistics": "current"}
    assert project.needs_evaluation() == [ids[0]]


def test_a_step_that_was_never_run_is_missing_rather_than_stale():
    project, ids = a_project()
    assert states(project, ids[0]) == {"Statistics": "missing"}
    assert project.latest(ids[0])[0] is None


def test_a_record_from_before_per_step_signatures_is_trusted_but_labelled():
    """An older file's results are reported; they just cannot be checked."""
    project, ids = a_project()
    project.evaluate()
    for record in project.runs[-1]["records"].values():
        for entry in record["steps"].values():
            del entry["signature"]
    assert states(project, ids[0]) == {"Statistics": "unverified"}
    assert len(project.results()) == 2             # not lost on opening the file
    assert set(project.needs_evaluation()) == set(ids)


# ------------------------------------------------------- running only those

def test_two_steps_go_stale_one_at_a_time():
    project, ids = a_project()
    project.pipeline.append(Instance(plugin=STATISTICS, label="photons"))
    project.evaluate()
    assert states(project, ids[0]) == {"Statistics": "current", "photons": "current"}

    project.pipeline[1].values = {"precision_column": "photons"}
    assert states(project, ids[0]) == {"Statistics": "current", "photons": "stale"}


def test_re_evaluating_runs_the_stale_step_and_copies_the_rest_forward():
    project, ids = a_project()
    project.pipeline.append(Instance(plugin=STATISTICS, label="photons"))
    project.evaluate()
    before = project.runs[-1]["records"][ids[0]]["steps"]["Statistics"]

    project.pipeline[1].values = {"precision_column": "photons"}
    run = project.evaluate(reuse=True)
    after = run["records"][ids[0]]["steps"]
    assert after["Statistics"] == before           # copied, not run again
    assert after["photons"]["values"]["mean_precision_nm"] == 200.0
    assert not project.needs_evaluation()
    assert len(project.results()) == 2


def test_one_site_reuses_what_is_current_and_says_what_it_ran():
    """A list is scrolled through without recomputing what has not changed."""
    project, ids = a_project()
    project.evaluate()
    record, results = project.evaluate_one(ids[0], reuse=True)
    assert results["Statistics"] is None           # kept, so nothing to draw
    assert record["steps"]["Statistics"]["values"]["n_localizations"] == 4

    project.pipeline[0].values = {"precision_column": "photons"}
    record, results = project.evaluate_one(ids[0], reuse=True)
    assert results["Statistics"] is not None       # stale, so it ran
    assert record["steps"]["Statistics"]["values"]["mean_precision_nm"] == 200.0


def test_a_figure_can_be_asked_for_even_when_the_numbers_are_current():
    """A plot is a closure; nothing stored brings it back, so it re-runs."""
    project, ids = a_project()
    project.evaluate()
    _, results = project.evaluate_one(ids[0], reuse=True, force=["Statistics"])
    assert results["Statistics"].figures()


def test_what_a_site_re_evaluation_stores_makes_it_current():
    project, ids = a_project()
    project.evaluate()
    project.pipeline[0].values = {"precision_column": "photons"}
    assert project.needs_evaluation() == list(ids)

    project.evaluate_one(ids[0], reuse=True, store=True)
    assert project.needs_evaluation() == [ids[1]]  # only the one looked at
    assert states(project, ids[0]) == {"Statistics": "current"}
    assert project.runs[-1]["scope"] == "site"     # honest about what it was

    project.evaluate_one(ids[0], reuse=True, store=True)
    assert len(project.runs) == 2                  # nothing ran, nothing stored


def test_a_pipeline_nobody_stored_is_read_back_from_the_run():
    """A script that hands its steps to `evaluate` still gets its results."""
    project, ids = a_project()
    project.pipeline = []
    steps = pipeline_module.resolve([Instance(plugin=STATISTICS)])
    project.evaluate(steps=steps)
    assert len(project.results()) == 2
    assert states(project, ids[0]) == {"Statistics": "current"}


def test_the_evaluation_window_re_runs_only_what_changed(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.evaluation import EvaluationWindow
    from smappy.session import Session
    QApplication.instance() or QApplication([])

    session = Session(table([0, 1000]))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(200)
    file_id = next(iter(project.sources))
    for center in (0, 1000):
        project.add_roi(file_id, [center, 0])

    window = EvaluationWindow(session, [Instance(plugin=STATISTICS)])
    window.run()
    assert not project.needs_evaluation()

    window.run(reuse=True)
    assert "every ROI is up to date" in window.status.text()

    window.instances[0].values = {"precision_column": "photons"}
    window.forms.clear()                           # the value, not a stale form
    window.refresh_counts()
    assert "(2)" in window.changed_button.text()
    window.run(reuse=True)
    assert "2 of them out of date" in window.status.text()
    assert not project.needs_evaluation()
