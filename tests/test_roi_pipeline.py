"""The ROI plugins as real plugins, and the pipeline that runs them per site."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Context, settings_from, settings_values
from smappy.plugins.roi import (DensityPeakSettings, HistogramSettings, Histograms,
                                density_peaks, histograms)
from smappy.plugins.segment import SegmentRegions, SegmentSettings


def peaks(**values):
    """Segment regions' settings for density peaks with these values."""
    return SegmentSettings(method="peaks", peaks=DensityPeakSettings(**values))
from smappy.plugins.statistics import LocalizationStatistics, StatisticsSettings
from smappy.roi_manager import pipeline as pipeline_module
from smappy.chain import Block, Step as ChainStep
from test_roi_session import rois_of                          # noqa: E402


def table(centers, per=10, seed=0):
    rng = np.random.default_rng(seed)
    x, y = [], []
    for cx, cy in centers:
        x.append(rng.normal(cx, 20, per))
        y.append(rng.normal(cy, 20, per))
    n = per * len(centers)
    return Localizations({"x_nm": np.concatenate(x), "y_nm": np.concatenate(y),
                          "frame": np.arange(n), "photons": np.full(n, 200.0),
                          "xy_err_nm": np.full(n, 11.0)})


# ------------------------------------------------------------- the plugins

def test_the_region_plugins_are_registered_under_regions():
    found = plugins.refs("Analysis/Regions/")
    assert set(found) == {"Analysis/Regions/Segment regions",
                          "Analysis/Regions/Find clusters",
                          "Analysis/Regions/Regions to column",
                          "Analysis/Regions/Sort ROIs",
                          "Analysis/Regions/Filter ROIs",
                          "Analysis/Regions/Remove double ROIs",
                          "Analysis/Regions/Centre ROIs",
                          "Analysis/Regions/Average ROIs",
                          "Analysis/Regions/Summarize/Histograms",
                          "Analysis/Regions/Summarize/Export table",
                          "Analysis/Regions/Summarize/NPC labelling efficiency",
                          "Analysis/Regions/Workflow/NPC Analysis"}
    found = {**found, **plugins.refs("Analysis/Measure/")}
    # per_region is what marks an evaluator, not the folder it sits in
    assert found["Analysis/Measure/NPC Corners"].per_region
    assert not found["Analysis/Regions/Segment regions"].per_region


def test_the_defaults_dict_became_a_typed_settings_class():
    specs = plugins.get("Analysis/Regions/Segment regions").specs()
    assert set(specs) == {"preset", "method", "ring", "peaks", "ids", "replace"}
    peak = specs["peaks"].children
    assert set(peak) == {"bin_nm", "sigma_nm", "separation_nm",
                         "count_radius_nm", "min_count"}
    assert peak["bin_nm"].type is float and peak["bin_nm"].info.unit == "nm"
    assert peak["min_count"].type is int and peak["min_count"].info.min == 1


def test_statistics_measures_one_site_through_the_context():
    project, source = rois_of(table([(0, 0), (5000, 0)], per=30), filters={})
    roi = project.add_roi(source.id, [0.0, 0.0])
    result = LocalizationStatistics().run(project.region_context(roi),
                                          StatisticsSettings(precision_fit="none"))
    assert result.values["n_localizations"] == 30       # this site's, not the file's
    assert result.values["mean_precision_nm"] == 11.0
    assert np.isnan(result.values["sigma_c_nm"])        # the fit was switched off


def test_a_bad_setting_is_refused_rather_than_silently_odd():
    with pytest.raises(ValueError, match="must be positive"):
        density_peaks(table([(0, 0)]), DensityPeakSettings(bin_nm=0.0))


def test_histograms_needs_a_site_table():
    with pytest.raises(ValueError, match="nothing measured in the regions yet"):
        Histograms().run(Context(), HistogramSettings())


def test_histograms_picks_the_numeric_columns_itself():
    rows = [{"roi_id": "a", "file_id": "f", "n": 3, "name": "x"},
            {"roi_id": "b", "file_id": "f", "n": 5, "name": "y"}]
    out = histograms(rows, HistogramSettings(bins=2))
    assert set(out) == {"n"}                # roi_id, file_id and text are skipped
    assert out["n"]["counts"].sum() == 2


# ------------------------------------------------------------ the pipeline

def test_a_step_of_the_projects_block_is_a_chains_step():
    instance = ChainStep(plugin="Analysis/Measure/Localization Statistics",
                         values={"bins": 40})
    step, = pipeline_module.block_steps(Block(steps=[instance]))
    assert step.label == "Localization Statistics" and step.version == "5"
    assert step.settings.bins == 40
    assert step.as_record()["plugin"] == "Analysis/Measure/Localization Statistics"


def test_a_specialised_evaluator_is_offered_but_not_in_the_default_pipeline():
    assert plugins.refs()["Analysis/Measure/NPC Corners"].per_region
    assert [i.plugin for i in pipeline_module.default_block().steps] == \
        ["Analysis/Measure/Localization Statistics"]


def test_a_disabled_step_is_skipped():
    block = pipeline_module.default_block()
    block.steps[0].enabled = False
    assert pipeline_module.block_steps(block) == []


def test_a_step_whose_plugin_is_gone_costs_only_itself():
    block = pipeline_module.default_block()
    block.steps.insert(0, ChainStep(plugin="Gone/Away"))
    steps = pipeline_module.block_steps(block)
    assert [s.path for s in steps] == ["Analysis/Measure/Localization Statistics"]


def test_the_same_evaluator_twice_gets_distinct_columns():
    """SMAP's addmodule renames on collision; so does this."""
    one = ChainStep(plugin="Analysis/Measure/Localization Statistics")
    steps = pipeline_module.block_steps(Block(steps=[one, ChainStep(plugin=one.plugin)]))
    assert [s.label for s in steps] == ["Localization Statistics", "Localization Statistics 2"]


def test_running_a_two_step_pipeline_over_a_project():
    project, source = rois_of(table([(0, 0), (1000, 1000)]), filters={})
    project.set_geometry(200)
    for center in ([0, 0], [1000, 1000]):
        project.add_roi(source.id, center)
    one = ChainStep(plugin="Analysis/Measure/Localization Statistics")
    two = ChainStep(plugin=one.plugin, values={"bins": 50})
    run = project.evaluate([one, two])
    assert [step["label"] for step in run["pipeline"]] == ["Localization Statistics", "Localization Statistics 2"]
    rows = project.results()
    assert len(rows) == 2
    # both steps produce every column, so every one is qualified
    assert rows[0]["Localization Statistics.mean_precision_nm"] == 11.0
    assert rows[0]["Localization Statistics 2.mean_precision_nm"] == 11.0


def test_the_block_travels_with_the_project():
    """The columns of a results table are meaningless without it."""
    from smappy.session import Session
    session = Session(table([(0, 0)]))
    project = session.rois
    project.add_roi(next(iter(project.sources)), [0, 0])
    project.block = pipeline_module.default_block()
    project.block.steps[0].label = "my statistics"
    project.block.write_values = ["n_localizations"]
    project.evaluate()
    project.from_dict(project.to_dict())
    assert [s.label for s in project.block.steps] == ["my statistics"]
    assert project.block.write_values == ["n_localizations"]
    assert project.runs[0]["pipeline"][0]["label"] == "my statistics"


def test_a_file_from_before_the_block_reads_its_steps_and_options():
    from smappy.roi_manager.core import block_from
    block = block_from({
        "pipeline": [{"plugin": "Analysis/Measure/Localization Statistics",
                      "label": "stats", "values": {"bins": 50}},
                     {"plugin": "Chain/Block", "label": "inner", "enabled": False,
                      "chain": {"block": "", "regions": "cluster_id", "steps": []}},
                     {"nope": 1}, "text"],
        "block_options": {"regions": "rois", "write_back": ["n_here"], "merge": "mean"}})
    first, inner = block.steps
    assert first.label == "stats" and first.values == {"bins": 50}
    assert isinstance(inner, Block) and inner.label == "inner" and not inner.enabled
    assert block.write_back == ["n_here"] and block.merge == "mean"
    assert block_from({}).steps == []


# ------------------------------------------------------------- the settings

def test_settings_round_trip_through_a_flat_map():
    settings = DensityPeakSettings(bin_nm=5.0, min_count=3)
    values = settings_values(settings)
    assert values["bin_nm"] == 5.0 and values["min_count"] == 3
    assert settings_from(DensityPeakSettings, values) == settings


def test_a_renamed_field_costs_that_field_and_not_the_pipeline():
    rebuilt = settings_from(DensityPeakSettings, {"bin_nm": 9.0, "went_away": 1})
    assert rebuilt.bin_nm == 9.0
    assert rebuilt.min_count == DensityPeakSettings().min_count


def test_a_segmenter_preview_proposes_what_run_adds_and_adds_nothing():
    from smappy.session import Session
    session = Session(table([(0, 0), (3000, 0), (0, 3000)], per=40))
    session.show_grouped(0, False)
    settings = peaks(separation_nm=1000.0, min_count=20)
    finder = SegmentRegions()
    preview = finder.preview(session.context(), settings)
    assert session.rois.rois == {}
    sites = preview.roi_preview["sites"]
    assert len(sites) == 3 and preview.text == "3 candidates previewed"

    # the session puts it where the manager draws it; a run adds the same
    # ROIs and takes the preview away
    session.show_preview(finder, preview)
    assert session.rois.segment_preview["sites"] == sites
    run = finder.run(session.context(), settings)
    session.apply(finder, run)
    assert session.rois.segment_preview is None
    assert sorted(r.center for r in session.rois.rois.values()) == \
        sorted(s["center"] for s in sites)


def test_propose_is_find_without_the_adding():
    project, source = rois_of(table([(0, 0), (3000, 0)], per=40))
    project.set_geometry(300)
    settings = peaks(separation_nm=1000.0, min_count=20)
    origin, proposals = project.propose(source.id, settings=settings)
    assert project.rois == {} and len(proposals) == 2
    assert origin["method"] == "Analysis/Regions/Segment regions"
    assert all(p["origin"]["segment"] == "peaks" for p in proposals)
    found = project.find(source.id, settings=settings)
    assert [r.center for r in found] == [p["center"] for p in proposals]
    # the found ROIs are what a second proposal keeps its distance from,
    # unless they are the ones a run would remove first
    assert project.propose(source.id, settings=settings)[1] == []
    assert len(project.propose(source.id, settings=settings,
                               exclude=[r.id for r in found])[1]) == 2


def test_an_roi_edit_is_undone_and_redone_and_a_no_op_leaves_no_step():
    project, source = rois_of(table([(0, 0)]))
    with project.edit("Add ROI"):
        kept = project.add_roi(source.id, (0.0, 0.0))
    with project.edit("Exclude ROI 1"):
        kept.use = False
    with project.edit("nothing"):
        pass
    assert project.history.undo_label == "Exclude ROI 1"
    assert project.undo() == "Exclude ROI 1" and project.rois[kept.id].use
    assert project.undo() == "Add ROI" and project.rois == {}
    assert project.undo() is None
    assert project.redo() == "Add ROI" and list(project.rois) == [kept.id]
    # a new edit is a new branch: what was undone cannot come back
    with project.edit("Remove ROI 1"):
        project.rois.clear()
    assert not project.history.can_redo


def test_the_roi_undo_keeps_twenty_steps():
    project, source = rois_of(table([(0, 0)]))
    for i in range(25):
        with project.edit(f"Add {i}"):
            project.add_roi(source.id, (float(i), 0.0))
    undone = [project.undo() for _ in range(25)]
    assert undone[:20] == [f"Add {i}" for i in range(24, 4, -1)]
    assert undone[20:] == [None] * 5 and len(project.rois) == 5


def test_a_replacing_run_is_one_undo_step_that_brings_the_old_rois_back_with_their_results():
    from smappy.session import Session
    session = Session(table([(0, 0), (3000, 0), (0, 3000)], per=40))
    session.show_grouped(0, False)
    project = session.rois
    file_id = next(iter(project.sources))
    drawn = project.add_roi(file_id, (1500.0, 1500.0))
    finder = SegmentRegions()
    settings = peaks(separation_nm=1000.0, min_count=20)
    first = finder.run(session.context(), settings).rois
    assert len(first) == 3
    project.evaluate([ChainStep(plugin="Analysis/Measure/Localization Statistics")])
    assert len(project.results()) == 4

    # again without replace: nothing new, and no step for it
    import dataclasses
    assert finder.run(session.context(),
                      dataclasses.replace(settings, replace=False)).rois == []
    assert project.history.undo_label == "Segment regions: 3 ROIs"

    again = finder.run(session.context(), settings).rois     # replace: the default
    assert len(again) == 3 and drawn.id in project.rois
    assert not {r.id for r in first} & set(project.rois)
    assert project.undo() == "Segment regions: 3 ROIs for 3"
    assert set(project.rois) == {drawn.id} | {r.id for r in first}
    assert len(project.results()) == 4          # the numbers came back too


# ---------------------------------------------------------------- the GUI

@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_evaluation_window_offers_only_evaluators(app):
    from smappy.gui.chooser import PluginTree
    tree = PluginTree(per_region=True)
    assert sorted(tree.refs) == ["Analysis/Measure/Blinking",
                                 "Analysis/Measure/Circle fit",
                                 "Analysis/Measure/Cluster shape",
                                 "Analysis/Measure/Histogram",
                                 "Analysis/Measure/Line Profile",
                                 "Analysis/Measure/Localization Statistics",
                                 "Analysis/Measure/NPC Corners",
                                 "Analysis/Measure/Spatial statistics"]
    assert set(tree.refs) < set(PluginTree().refs)


def test_the_region_steps_edit_the_projects_block(app, monkeypatch):
    from PySide6.QtCore import Qt
    from smappy.gui import chooser
    from smappy.gui.region_steps import RegionSteps
    from smappy.session import Session

    session = Session(table([(0, 0)]))
    window = RegionSteps(session)
    project = session.rois
    tree = window.panel.steps.invisibleRootItem()
    assert tree.childCount() == 1                       # the block's steps, at the top
    assert tree.child(0).text(0).startswith("Localization Statistics")
    window.panel.form.set_values({"for_each_roi.localization_statistics.bins": 50})
    assert project.block.steps[0].values["bins"] == 50
    # a step added with nothing selected goes into the block
    window.panel.steps.setCurrentItem(None)
    monkeypatch.setattr(chooser, "choose_plugin",
                        lambda *a, **k: "Analysis/Measure/NPC Corners")
    window.panel.add_step()
    assert [i.plugin for i in project.block.steps] == [
        "Analysis/Measure/Localization Statistics", "Analysis/Measure/NPC Corners"]
    assert project.block.steps[0].values["bins"] == 50     # what the form held survives
    window.panel.steps.invisibleRootItem().child(1).setCheckState(0, Qt.Unchecked)
    assert not project.block.steps[1].enabled
    assert [s.label for s in project.block_steps()] == ["Localization Statistics"]
    # what it repeats over is the block's own field
    window.panel.form.set_values({"for_each_roi.write_back": "n_here"})
    assert project.block.write_back == ["n_here"]


def test_a_tab_pins_a_per_region_plugin_too(app):
    """One that can run per region runs on the view as well."""
    from smappy.gui.chooser import PluginTree
    assert "Analysis/Measure/Localization Statistics" in PluginTree().refs
    assert "Analysis/Regions/Segment regions" in PluginTree().refs


def test_find_evaluate_and_analyse_through_the_session(app):
    """Segment, region steps, analyse -- on one session."""
    from smappy.gui.region_steps import RegionSteps
    from smappy.session import Session

    rng = np.random.default_rng(1)
    n = 400
    x = np.concatenate([rng.normal(c, 30, 100) for c in (0, 3000, 0, 3000)])
    y = np.concatenate([rng.normal(c, 30, 100) for c in (0, 0, 3000, 3000)])
    session = Session(Localizations({
        "x_nm": x, "y_nm": y, "frame": np.arange(n),
        "photons": np.full(n, 300.0), "xy_err_nm": np.full(n, 9.0)}))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(400)
    file_id = next(iter(project.sources))

    found = project.find(file_id, plugin=SegmentRegions(),
                         settings=peaks(sigma_nm=60.0, separation_nm=1000.0,
                                        min_count=20))
    assert len(found) == 4

    window = RegionSteps(session)
    window.run()
    assert "4 ROIs, 1 step(s)" in window.status.text()
    # the session's log says what was evaluated, with each step's settings
    entry = session.history[-1]
    assert entry["what"] == "Analysis/Regions/Per-region steps" and "4 ROIs" in entry["text"]
    assert entry["steps"][0]["plugin"] == "Chain/Block"
    assert entry["steps"][0]["values"]["localization_statistics.precision_fit"] == "histogram"
    rows = project.results()
    assert len(rows) == 4 and rows[0]["n_localizations"] == 100

    analysis = plugins.get("Analysis/Regions/Summarize/Histograms")()(ctx=session.context())
    assert "4 ROIs" in analysis.text
    assert analysis.plot is not None


def test_the_manager_draws_a_preview_and_run_turns_it_into_rois(app):
    """Preview opens the manager with the proposals dashed and no ROI added;
    a second preview replaces the first; Run adds them and the window,
    which used to miss a segmenter's run, lists them."""
    from PySide6.QtCore import QEventLoop, QTimer
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.gui.roi_tab import ROIHeader
    from smappy.session import Session

    session = Session(table([(0, 0), (3000, 0), (0, 3000), (3000, 3000)], per=40))
    session.show_grouped(0, False)
    header = ROIHeader(session)
    panel = PluginPanel(plugins.get("Analysis/Regions/Segment regions"), session)
    panel.form.restore({"method": "peaks", "peaks.separation_nm": 1000.0,
                        "peaks.min_count": 20})

    def wait(start):
        loop = QEventLoop()
        panel.ended.connect(loop.quit)
        start()
        QTimer.singleShot(10000, loop.quit)          # never hang the suite
        loop.exec()
        panel.ended.disconnect(loop.quit)
        app.processEvents()

    wait(panel.preview)
    window = header.manager
    assert window is not None and session.rois.rois == {}
    xs, _ = window.file_pane.shapes["preview"].getData()
    assert np.isnan(xs).sum() == 4                   # four outlines
    assert window.clear_preview_button.isVisibleTo(window)
    assert "previewing 4" in window.status.currentMessage()

    panel.form.restore({"peaks.min_count": 1000})
    wait(panel.preview)
    assert session.rois.segment_preview["sites"] == []

    panel.form.restore({"peaks.min_count": 20})
    wait(panel.run)
    assert len(session.rois.rois) == 4 and window.table.model().rowCount() == 4
    assert session.rois.segment_preview is None
    assert window.file_pane.shapes["preview"].getData()[0] is None or \
        not len(window.file_pane.shapes["preview"].getData()[0])
    assert not window.clear_preview_button.isVisibleTo(window)

    wait(panel.preview)
    window.clear_preview_button.click()
    assert session.rois.segment_preview is None and len(session.rois.rois) == 4


def test_the_manager_undoes_a_removal_with_its_own_undo(app):
    from smappy.gui.roi_window import ROIManagerWindow
    from smappy.session import Session

    session = Session(table([(0, 0), (3000, 0)], per=40))
    session.show_grouped(0, False)
    project = session.rois
    file_id = next(iter(project.sources))
    first = project.add_roi(file_id, (0.0, 0.0))
    project.add_roi(file_id, (3000.0, 0.0))
    window = ROIManagerWindow(session)
    assert not window.undo_button.isEnabled()
    assert window.undo_button.text() == "Undo ROI edit"

    window.select(first.id)
    window._remove()
    assert first.id not in project.rois and window.table.model().rowCount() == 1
    assert window.undo_button.toolTip() == "undo: Remove ROI 1"
    # the button names the edit, so it is not the main window's "Undo RCC"
    assert window.undo_button.text() == "Undo remove ROI 1"
    window.undo_button.click()
    assert first.id in project.rois and window.table.model().rowCount() == 2
    assert "undid Remove ROI 1" in window.status.currentMessage()
    assert window.redo_button.toolTip() == "redo: Remove ROI 1"
    assert window.redo_button.text() == "Redo remove ROI 1"
    assert window.undo_button.text() == "Undo ROI edit"
    # the table's undo is another stack, and has nothing to undo
    assert not session.can_undo
    window.redo_button.click()
    assert window.table.model().rowCount() == 1 and window.active is None
