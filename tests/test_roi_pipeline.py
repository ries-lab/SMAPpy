"""The ROI plugins as real plugins, and the pipeline that runs them per site."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Context, settings_from, settings_values
from smappy.plugins.roi import (DensityPeakSettings, HistogramSettings, Histograms,
                                Statistics, StatisticsSettings, density_peaks,
                                histograms, site_statistics)
from smappy.roi_manager import ROIProject
from smappy.roi_manager import pipeline as pipeline_module
from smappy.workspace import Instance


def table(centers, per=10, seed=0):
    rng = np.random.default_rng(seed)
    x, y = [], []
    for cx, cy in centers:
        x.append(rng.normal(cx, 20, per))
        y.append(rng.normal(cy, 20, per))
    n = per * len(centers)
    return Localizations({"x_nm": np.concatenate(x), "y_nm": np.concatenate(y),
                          "frame": np.arange(n), "photons": np.full(n, 200.0),
                          "loc_precision_nm": np.full(n, 11.0)})


# ------------------------------------------------------------- the plugins

def test_the_three_are_registered_where_smap_puts_them():
    found = plugins.refs("ROIManager/")
    assert set(found) == {"ROIManager/Segment/Density Peaks",
                          "ROIManager/Evaluate/Statistics",
                          "ROIManager/Analyze/Histograms"}
    # scope is what marks an evaluator, not the folder it sits in
    assert found["ROIManager/Evaluate/Statistics"].scope == "site"
    assert found["ROIManager/Segment/Density Peaks"].scope == "locs"


def test_the_defaults_dict_became_a_typed_settings_class():
    specs = plugins.get("ROIManager/Segment/Density Peaks").specs()
    assert set(specs) == {"bin_nm", "sigma_nm", "separation_nm",
                          "count_radius_nm", "min_count"}
    assert specs["bin_nm"].type is float and specs["bin_nm"].info.unit == "nm"
    assert specs["min_count"].type is int and specs["min_count"].info.min == 1


def test_statistics_measures_one_site_through_the_context():
    locs = table([(0, 0)], per=4)
    result = Statistics().run(Context(locs=locs), StatisticsSettings())
    assert result.data["n_localizations"] == 4
    assert result.data["mean_precision_nm"] == 11.0


def test_a_bad_setting_is_refused_rather_than_silently_odd():
    with pytest.raises(ValueError, match="must be positive"):
        density_peaks(table([(0, 0)]), DensityPeakSettings(bin_nm=0.0))


def test_histograms_needs_a_site_table():
    with pytest.raises(ValueError, match="no site table"):
        Histograms().run(Context(), HistogramSettings())


def test_histograms_picks_the_numeric_columns_itself():
    rows = [{"roi_id": "a", "file_id": "f", "n": 3, "name": "x"},
            {"roi_id": "b", "file_id": "f", "n": 5, "name": "y"}]
    out = histograms(rows, HistogramSettings(bins=2))
    assert set(out) == {"n"}                # roi_id, file_id and text are skipped
    assert out["n"]["counts"].sum() == 2


# ------------------------------------------------------------ the pipeline

def test_a_pipeline_step_is_the_same_instance_type_a_tab_pins():
    instance = Instance(plugin="ROIManager/Evaluate/Statistics",
                        values={"precision_column": "loc_precision_nm"})
    step, = pipeline_module.resolve([instance])
    assert step.label == "Statistics" and step.version == "1"
    assert step.settings.precision_column == "loc_precision_nm"
    assert step.as_record()["plugin"] == "ROIManager/Evaluate/Statistics"


def test_a_disabled_step_is_skipped():
    instances = pipeline_module.default_instances()
    instances[0].enabled = False
    assert pipeline_module.resolve(instances) == []


def test_a_step_whose_plugin_is_gone_costs_only_itself():
    instances = [Instance(plugin="Gone/Away")] + pipeline_module.default_instances()
    steps = pipeline_module.resolve(instances)
    assert [s.path for s in steps] == ["ROIManager/Evaluate/Statistics"]


def test_the_same_evaluator_twice_gets_distinct_columns():
    """SMAP's addmodule renames on collision; so does this."""
    one = Instance(plugin="ROIManager/Evaluate/Statistics")
    steps = pipeline_module.resolve([one, one.copy()])
    assert [s.label for s in steps] == ["Statistics", "Statistics 2"]


def test_running_a_two_step_pipeline_over_a_project():
    project = ROIProject()
    source = project.add_source(table([(0, 0), (1000, 1000)]))
    project.set_filters({})
    project.set_geometry(200)
    for center in ([0, 0], [1000, 1000]):
        project.add_roi(source.id, center)
    one = Instance(plugin="ROIManager/Evaluate/Statistics")
    two = one.copy()
    two.values = {"precision_column": "photons"}
    run = project.evaluate([one, two])
    assert [step["label"] for step in run["pipeline"]] == ["Statistics", "Statistics 2"]
    rows = project.results()
    assert len(rows) == 2
    # both steps produce every column, so every one is qualified
    assert rows[0]["Statistics.mean_precision_nm"] == 11.0
    assert rows[0]["Statistics 2.mean_precision_nm"] == 200.0


def test_a_pipeline_file_round_trips(tmp_path):
    instances = pipeline_module.default_instances()
    instances[0].label = "precision only"
    instances[0].values = {"precision_column": "photons"}
    path = pipeline_module.save(instances, tmp_path / "p.yaml")
    back = pipeline_module.load(path)
    assert [i.label for i in back] == ["precision only"]
    assert back[0].values == {"precision_column": "photons"}
    assert pipeline_module.resolve(back)[0].settings.precision_column == "photons"


def test_a_broken_pipeline_file_gives_no_steps_rather_than_raising():
    assert pipeline_module.from_dict(None) == []
    assert pipeline_module.from_dict({"pipeline": [{"nope": 1}, "text"]}) == []


def test_the_pipeline_travels_with_the_project(tmp_path):
    """The columns of a site table are meaningless without it."""
    project = ROIProject()
    source = project.add_source(table([(0, 0)]), path=str(tmp_path / "locs.hdf5"))
    project.set_filters({})
    project.add_roi(source.id, [0, 0])
    project.pipeline = pipeline_module.default_instances()
    project.pipeline[0].label = "my statistics"
    project.evaluate(project.pipeline)

    from smappy.io.hdf5 import save_localizations
    save_localizations(tmp_path / "locs.hdf5", table([(0, 0)]))
    path = project.save(tmp_path / "project.h5")
    loaded = ROIProject.load(path)
    assert [i.label for i in loaded.pipeline] == ["my statistics"]
    assert loaded.runs[0]["pipeline"][0]["label"] == "my statistics"


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


# ---------------------------------------------------------------- the GUI

@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_evaluation_window_offers_only_evaluators(app):
    from smappy.gui.chooser import PluginTree
    tree = PluginTree(scope="site")
    assert list(tree.refs) == ["ROIManager/Evaluate/Statistics"]
    assert set(PluginTree(scope="locs").refs).isdisjoint(tree.refs)


def test_the_window_edits_the_pipeline_and_keeps_the_values(app):
    from smappy.gui.evaluation import EvaluationWindow
    from smappy.session import Session

    window = EvaluationWindow(Session(table([(0, 0)])))
    assert [window.list.item(0).text()] == ["Statistics"]
    window.forms_stack.currentWidget().restore({"precision_column": "photons"})
    window.duplicate()
    assert [window.list.item(i).text() for i in range(2)] == \
        ["Statistics", "Statistics copy"]
    window.move(-1)
    assert window.list.item(0).text() == "Statistics copy"
    instances = window.save_values()
    assert instances[1].values["precision_column"] == "photons"

    from PySide6.QtCore import Qt
    window.list.item(0).setCheckState(Qt.Unchecked)
    assert not window.instances[0].enabled
    assert [s.label for s in pipeline_module.resolve(window.instances)] == ["Statistics"]


def test_a_tab_will_not_pin_an_evaluator(app):
    """A scope="site" plugin has no site in a tab, so its Run could only fail."""
    from smappy.gui.chooser import PluginTree
    assert "ROIManager/Evaluate/Statistics" not in PluginTree(scope="locs").refs
    assert "ROIManager/Segment/Density Peaks" in PluginTree(scope="locs").refs


def test_find_evaluate_and_analyse_through_the_session(app):
    """Segment, pipeline, analyse -- the three kinds, on one session."""
    from smappy.gui.evaluation import EvaluationWindow
    from smappy.plugins.roi import DensityPeaks
    from smappy.session import Session

    rng = np.random.default_rng(1)
    n = 400
    x = np.concatenate([rng.normal(c, 30, 100) for c in (0, 3000, 0, 3000)])
    y = np.concatenate([rng.normal(c, 30, 100) for c in (0, 0, 3000, 3000)])
    session = Session(Localizations({
        "x_nm": x, "y_nm": y, "frame": np.arange(n),
        "photons": np.full(n, 300.0), "loc_precision_nm": np.full(n, 9.0)}))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(400)
    file_id = next(iter(project.sources))

    found = project.find(file_id, plugin=DensityPeaks(),
                         settings=DensityPeakSettings(sigma_nm=60.0,
                                                      separation_nm=1000.0,
                                                      min_count=20))
    assert len(found) == 4
    for roi in project.rois.values():
        roi.reviewed = True

    window = EvaluationWindow(session)
    window.run()
    assert "4 ROIs, 1 step(s)" in window.status.text()
    rows = project.results()
    assert len(rows) == 4 and rows[0]["n_localizations"] == 100

    analysis = plugins.get("ROIManager/Analyze/Histograms")()(ctx=session.context())
    assert "4 ROIs" in analysis.text
    assert analysis.plot is not None
