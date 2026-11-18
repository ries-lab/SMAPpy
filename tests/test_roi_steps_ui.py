"""The Regions tab's and ROI manager's smaller controls: Segment's id
attribute chosen from the table's, Sort ROIs' keys one at a time, the range
band's edges, the ROI image's scale bar, the steps' right-click menu and
*Measure this ROI*."""
import numpy as np
import pytest

from smappy.plugins.roi_order import SortROIs, SortROIsSettings
from smappy.plugins.region_column import id_attributes
from smappy.plugins.segment import SegmentRegions, SegmentSettings
from test_roi_membership import track_window


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_segment_offers_the_tables_id_attributes_ids_first():
    from test_roi_membership import crossing_tracks
    from smappy.session import Session
    locs = crossing_tracks()
    locs.columns["cluster_id"] = np.zeros(len(locs), np.int64)
    locs.columns["n_steps"] = np.ones(len(locs), np.int64)
    names = id_attributes(locs)
    assert names[:2] == ["cluster_id", "track_id"] and "n_steps" in names
    assert "frame" not in names
    session = Session(locs)
    context = session.context(grouping=SegmentRegions.grouping)    # as the form asks
    state = SegmentRegions().describe(context, SegmentSettings(method="ids"))
    assert [v for v, _ in state.choices["ids.column"]] == names
    # another method: no list worked out for a part that is not shown
    state = SegmentRegions().describe(context, SegmentSettings(method="circle"))
    assert "ids.column" not in state.choices


def test_sort_rois_shows_one_key_and_the_next_once_the_one_before_is_set():
    shown = SortROIs().describe(None, SortROIsSettings()).shown
    assert not shown["then_by"] and not shown["last_by"]
    shown = SortROIs().describe(None, SortROIsSettings(by="n")).shown
    assert shown["then_by"] and shown["then_descending"] and not shown["last_by"]
    shown = SortROIs().describe(None, SortROIsSettings(by="n", then_by="m")).shown
    assert shown["last_by"]


def test_a_key_after_one_left_empty_sorts_nothing():
    from smappy.plugins import Context
    from test_roi_order import measured_project
    session, project, ids = measured_project()
    SortROIs().run(Context(locs=session.locs, session=session),
                   SortROIsSettings(by="", then_by="n_localizations"))
    assert project.ordered_ids() == ids              # by number, as with none


def test_a_scale_bar_is_a_round_length_of_about_a_fifth():
    from smappy.gui.roi_window import scale_bar_length, scale_bar_text
    assert scale_bar_length(80) == 50 and scale_bar_length(100) == 100
    assert scale_bar_length(260) == 200 and scale_bar_length(0) == 0
    assert scale_bar_text(500) == "500 nm" and scale_bar_text(2000) == "2 µm"


def test_the_roi_image_has_a_scale_bar_and_the_others_none(app):
    session, project, window = track_window()
    roi = next(iter(project.rois.values()))
    window.select(roi.id)
    pane = window.roi_pane
    assert pane.bar.isVisible() and pane.bar_text.toPlainText().endswith("nm")
    xs, ys = pane.bar.getData()
    (x0, x1), _ = pane.view.viewRange()
    assert 0.1 < (xs[1] - xs[0]) / (x1 - x0) <= 0.2 + 1e-9
    assert not window.zoom_pane.bar.isVisible()
    window.close()


def test_a_band_is_moved_by_its_edges_and_the_scale_holds_still(app):
    session, project, window = track_window()
    track = next(r for r in project.rois.values() if r.ids["track_id"] == 1)
    window.select(track.id)
    ranges = window.annotation.ranges
    plot = ranges.plots["frame"]
    band = plot.band
    assert band.dataBounds(0) is None                 # the view never chases it
    assert band.lines[0]._maxMarkerSize >= 6          # an edge is wide to grab

    class Drag:                                       # a drag inside, no Shift
        ignored = False

        def modifiers(self):
            from PySide6.QtCore import Qt
            return Qt.NoModifier

        def ignore(self):
            self.ignored = True
    drag = Drag()
    band.mouseDragEvent(drag)
    assert drag.ignored and not band.moving
    # a bound typed beyond the data is kept, and the plot takes it in
    ranges.fields[("frame", 1)].setText("40")
    ranges.fields[("frame", 1)].editingFinished.emit()
    assert project.rois[track.id].frames == [None, 40.0]
    (lo, hi), _ = plot.getPlotItem().getViewBox().viewRange()
    assert hi > 40 and lo < 1
    assert ranges.fields[("frame", 1)].text() == "40"
    window.close()


def test_the_steps_are_renamed_moved_and_removed_from_a_right_click(app):
    from smappy.gui.region_steps import RegionSteps
    from smappy.session import Session
    from test_roi_membership import crossing_tracks
    from smappy.chain import Step
    session = Session(crossing_tracks())
    session.rois.block.steps = [Step(plugin=p) for p in (
        "Analysis/Measure/Localization Statistics", "Analysis/Measure/Cluster shape",
        "Analysis/Measure/Line Profile")]
    steps = RegionSteps(session)
    panel = steps.panel
    before = [s.plugin for s in session.rois.block.steps]
    panel.select((0, 0))
    actions = {label: (slot, on) for label, slot, on in panel.step_actions()}
    assert set(actions) == {"Rename...", "Move up", "Move down", "Remove"}
    assert not actions["Move up"][1] and actions["Move down"][1]
    actions["Move down"][0]()
    after = [s.plugin for s in session.rois.block.steps]
    assert after[:2] == before[1::-1]
    panel.select((0, 1))
    dict((label, slot) for label, slot, _ in panel.step_actions())["Remove"]()
    assert [s.plugin for s in session.rois.block.steps] == [after[0]] + after[2:]
    from PySide6.QtWidgets import QAbstractButton
    assert not [b for b in panel.findChildren(QAbstractButton)
                if b.text() in ("up", "down", "remove", "rename...")]


def test_measure_this_roi_measures_the_selected_one_with_measuring_off(app):
    from smappy.gui.roi_tab import ROIHeader
    session, project, window = track_window()
    header = ROIHeader(session)
    header.manager = window
    roi = next(iter(project.rois.values()))
    window.select(roi.id)
    header._show_result()
    assert header.measure_now.isEnabled()
    assert project.latest(roi.id)[0] is None
    header.measure_selected()
    record, states = project.latest(roi.id)
    assert record is not None and set(states.values()) == {"current"}
    window.close()


# ------------------------------------------------- chains: sections are the list

def test_a_step_switched_off_keeps_its_section_and_its_settings(app):
    from smappy.chain import Step
    from test_chain_gui import new_chain_tab
    from test_batch_foundations import blinks
    from smappy.session import Session
    tab, instance, panel = new_chain_tab(app, Session(blinks()))
    spec = panel.current_spec()
    spec.steps += [Step(plugin="Analysis/Process/Layers"),
                   Step(plugin="Analysis/Measure/Localization Statistics",
                        values={"bins": 33})]
    panel._apply(spec)
    sections = panel.step_sections()
    assert [s.tick.isChecked() for *_, s in sections] == [True, True]
    sections[1][3].tick.setChecked(False)
    sections = panel.step_sections()
    assert len(sections) == 2 and not sections[1][3].tick.isChecked()
    assert [k for k, _, _ in panel.cls.steps] == ["layers"]          # not run
    assert panel.form.value().localization_statistics.bins == 33     # still there
    assert instance.chain["steps"][1]["enabled"] is False
    assert instance.chain["steps"][1]["values"]["bins"] == 33
    # its file and menu are in the tab section's title bar
    section = tab.sections[0]
    assert panel.menu_button in section.lent_widgets
    assert [a.text() for a in panel.chain_menu.actions()] == [
        "Open chain...", "Save chain", "Save as..."]
    assert panel.build.button.isChecked()                # a new chain: build open


def test_open_chain_takes_a_saved_chains_steps(app, tmp_path):
    from smappy import chain
    from smappy.chain import Block, ChainSpec, Step
    from smappy.gui.region_steps import RegionSteps
    from smappy.session import Session
    from test_roi_membership import crossing_tracks
    saved = tmp_path / "two.chain.yaml"
    chain.write(ChainSpec(steps=[Block(steps=[
        Step(plugin="Analysis/Measure/Cluster shape"),
        Step(plugin="Analysis/Measure/Localization Statistics")])]), saved)
    session = Session(crossing_tracks())
    steps = RegionSteps(session)
    assert steps.panel.open_chain(saved)
    assert [s.plugin for s in session.rois.block.steps] == [
        "Analysis/Measure/Cluster shape", "Analysis/Measure/Localization Statistics"]


def test_the_per_region_front_repeats_over_a_menu_and_marks_stale_steps(app):
    from smappy.gui.roi_tab import ROIHeader
    session, project, window = track_window()
    header = ROIHeader(session)
    header.manager = window
    steps = header.steps
    names = [steps.for_each.itemData(i) for i in range(steps.for_each.count())]
    assert names[0] == "rois" and "track_id" in names and "frame" not in names
    # Run and Measure this ROI side by side, at the top
    assert steps.controls.indexOf(steps.panel.panel.run_button) == 0
    assert steps.controls.indexOf(header.measure_now) == 1
    roi = next(iter(project.rois.values()))
    window.select(roi.id)
    header.measure_selected()
    header._show_result()
    title = steps.panel.step_sections()[0][3].title
    assert "(" not in title                                # current: no mark
    # a setting changed: the step's result for this ROI is out of date
    key = steps.block_key()
    inner = steps.panel.step_sections()[0][1]
    steps.panel.panel.form.set_values({f"{key}.{inner}.bins": 17})
    steps.store()
    header._show_result()
    assert steps.panel.step_sections()[0][3].title.endswith("(stale)")
    steps.for_each.setCurrentIndex(names.index("track_id"))
    steps.for_each.activated.emit(names.index("track_id"))
    assert project.block.regions == "track_id"
    window.close()
