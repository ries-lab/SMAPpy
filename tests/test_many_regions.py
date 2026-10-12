"""Many regions: the ROI manager costs what it shows, and a save says which
region results its settings leave out."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins.file import Save, left_out, region_counts
from smappy.session import Session


def clusters(n, per=12, spread=20.0, seed=0, field=2e5):
    """``n`` clusters of ``per`` localizations, ``cluster_id`` 1..n."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, field, (n, 2))
    xy = np.repeat(centres, per, axis=0) + rng.normal(0, spread, (n * per, 2))
    m = len(xy)
    session = Session(Localizations({
        "x_nm": xy[:, 0], "y_nm": xy[:, 1], "frame": np.arange(m, dtype=np.int64),
        "photons": np.full(m, 1000.0), "xy_err_nm": np.full(m, 10.0),
        "cluster_id": np.repeat(np.arange(1, n + 1), per)}, {}))
    session.show_grouped(0, False)
    return session


def measured_per_cluster(session, keep_figures="no"):
    from smappy.chain import Block, ChainSpec, Step, chain_class
    cls = chain_class(ChainSpec(steps=[Block(
        regions="cluster_id", keep_figures=keep_figures,
        steps=[Step(plugin="Analysis/Measure/Localization Statistics")])]))
    session.run(cls(), cls.Settings())


# ------------------------------------------------------------------- saving

def test_a_save_says_it_leaves_out_the_region_results_before_and_after(tmp_path):
    session = clusters(3)
    measured_per_cluster(session)
    assert region_counts(session) == {"rois": 0, "regions": 3, "kept": 0}
    ctx = session.context()
    assert Save().notice_for(ctx, Save.Settings()) is None      # all of it saved
    notice = Save().notice_for(ctx, Save.Settings(region_results=False))
    assert "3 regions of id columns will not be saved" in notice.text
    result = session.run(Save(), Save.Settings(path=str(tmp_path / "a.hdf5"),
                                               region_results=False))
    assert "3 regions of id columns were not saved" in result.text


def test_a_save_without_figure_data_counts_the_results_that_kept_some():
    session = clusters(3)
    measured_per_cluster(session, keep_figures="yes")
    settings = Save.Settings(region_figures=False)
    assert left_out(session, settings).startswith(
        "The figure data of 3 region results will not be saved")
    measured_per_cluster(session)            # measured again, nothing kept
    assert left_out(session, settings) is None


# -------------------------------------------------------------- the manager

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_save_panel_shows_the_notice_once_there_are_results(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = clusters(3)
    panel = PluginPanel(Save, session)
    panel.form.set_values({"region_results": False})
    panel.refresh_derived()
    assert panel.notice.isHidden()                       # nothing to leave out
    measured_per_cluster(session)                        # says "rois" changed
    assert not panel.notice.isHidden() and "3 regions" in panel.notice.text()


def test_past_the_limit_a_pane_draws_centres_and_the_selected_outline(app, monkeypatch):
    from smappy.gui import roi_window
    from smappy.plugins.region_column import ColumnToROIs
    monkeypatch.setattr(roi_window, "OUTLINE_LIMIT", 20)
    session = clusters(40)
    session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=1))
    window = roi_window.ROIManagerWindow(session)
    pane = window.file_pane
    xs, _ = pane.shapes["centres"].getData()
    assert len(xs) == 40 and "40 ROIs in view" in pane.label.text()
    assert not len(pane.shapes["others"].getData()[0] or [])
    first = next(iter(session.rois.rois))
    window.select(first)
    assert len(pane.shapes["centres"].getData()[0]) == 39
    assert np.isnan(pane.shapes["roi"].getData()[0]).sum() == 1   # one outline
    # zoomed in on it, the few ROIs in view are outlined
    window.zoom_pane.width_nm = 500.0
    window.redraw()
    assert window.zoom_pane.label.text() == "zoom"
    assert not len(window.zoom_pane.shapes["centres"].getData()[0] or [])


def test_the_list_makes_a_row_only_when_it_is_shown(app, monkeypatch):
    from smappy.gui import roi_window
    from smappy.plugins.region_column import ColumnToROIs
    monkeypatch.setattr(roi_window, "COUNT_LIMIT", 10)
    session = clusters(30)
    session.run(ColumnToROIs(), ColumnToROIs.Settings(min_size=1))
    checked = []
    original = session.rois.stale_steps
    monkeypatch.setattr(session.rois, "stale_steps",
                        lambda roi_id, steps=None: checked.append(roi_id)
                        or original(roi_id, steps))
    window = roi_window.ROIManagerWindow(session)
    model = window.table.model()
    assert model.rowCount() == 30
    assert "to evaluate" not in window.status.currentMessage()   # not counted
    assert len(checked) < 30                     # only the rows drawn, if any
    from PySide6.QtCore import Qt
    assert model.data(model.index(4, 0), Qt.ForegroundRole) is not None  # waiting
    roi_id = model.ids[4]
    model.setData(model.index(4, 2), Qt.Unchecked, Qt.CheckStateRole)
    assert session.rois.rois[roi_id].use is False
