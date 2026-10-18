"""An ROI as a filter: a shape, an id, a frame range and a z range."""

import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.roi_manager.core import IdLookup, ROI, value_range
from test_roi_session import rois_of, set_bounds              # noqa: E402


def crossing_tracks():
    """Two tracks through the origin, one along x and one along y, ten
    frames each, so a circle round the origin holds both."""
    t = np.arange(10)
    x = np.r_[(t - 4.5) * 20, np.zeros(10)]
    y = np.r_[np.zeros(10), (t - 4.5) * 20]
    return Localizations({"x_nm": x, "y_nm": y,
                          "z_nm": np.r_[t * 10.0, -t * 10.0],
                          "frame": np.r_[t, t].astype(np.int64),
                          "photons": np.full(20, 1000.0),
                          "xy_err_nm": np.full(20, 5.0),
                          "track_id": np.r_[np.full(10, 1), np.full(10, 2)]},
                         {"units": "nm"})


def test_a_track_roi_holds_its_own_track_and_not_the_one_crossing_it():
    p, s = rois_of(crossing_tracks(), filters={})
    circle = p.add_roi(s.id, [0, 0])
    assert len(p.indices(circle)) == 20          # the shape takes in both
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False)
    assert p.indices(track).tolist() == list(range(10, 20))
    assert p.region(track) is None
    both = p.add_roi(s.id, [0, 0], ids={"track_id": 1})   # shape and id
    p.set_geometry(50)                           # a circle of radius 25
    assert p.indices(both).tolist() == [4, 5]


def test_frame_and_z_ranges_narrow_an_roi_with_open_ends():
    p, s = rois_of(crossing_tracks(), filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 1}, spatial=False,
                      frames=[2, 5])
    assert p.indices(track).tolist() == [2, 3, 4, 5]
    p.set_ranges([track.id], frames=[7, None])
    assert p.indices(track).tolist() == [7, 8, 9]
    p.set_ranges([track.id], frames=None, z=[None, 15])
    assert p.indices(track).tolist() == [0, 1]
    p.undo()
    assert p.rois[track.id].frames == [7, None] and p.rois[track.id].z is None


def test_a_z_range_on_a_table_without_z_cuts_nothing():
    locs = crossing_tracks()
    del locs.columns["z_nm"]
    p, s = rois_of(locs, filters={})
    roi = p.add_roi(s.id, [0, 0], z=[0, 10])
    assert len(p.indices(roi)) == 20


def test_a_layer_filter_applies_inside_an_id_roi():
    locs = crossing_tracks()
    locs.columns["photons"] = np.arange(20) * 100.0
    p, s = rois_of(locs, filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False)
    set_bounds(p, {"photons": (1500, None)})
    assert p.indices(track).tolist() == list(range(15, 20))


def test_a_measurement_in_an_id_roi_sees_its_rows_and_no_shape():
    p, s = rois_of(crossing_tracks(), filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 1}, spatial=False, frames=[0, 3])
    ctx = p.region_context(track)
    assert ctx.region is None
    assert np.asarray(ctx.locs["track_id"]).tolist() == [1, 1, 1, 1]
    assert np.asarray(ctx.locs["frame"]).tolist() == [0, 1, 2, 3]
    run = p.evaluate()
    assert run["records"][track.id]


def test_blinks_of_a_ranged_roi_are_those_its_localizations_belong_to():
    # one emitter on for frames 0-5, another for 10-12: two blinks
    n = 9
    frame = np.r_[np.arange(6), np.arange(10, 13)].astype(np.int64)
    locs = Localizations({"x_nm": np.r_[np.zeros(6), np.full(3, 30.0)],
                          "y_nm": np.zeros(n), "frame": frame,
                          "photons": np.full(n, 1000.0), "xy_err_nm": np.full(n, 5.0)},
                         {"units": "nm"})
    p, s = rois_of(locs, filters={}, grouped=True)
    roi = p.add_roi(s.id, [0, 0], frames=[4, 11])
    ctx = p.region_context(roi)
    assert np.asarray(ctx.ungrouped()[0]["frame"]).tolist() == [4, 5, 10, 11]
    blinks = ctx.locs
    assert len(blinks) == 2                     # both blinks, each half in range
    assert sorted(np.asarray(blinks["n_in_group"]).tolist()) == [3, 6]


def test_ranges_and_ids_are_part_of_the_signature_and_the_saved_file():
    p, s = rois_of(crossing_tracks(), filters={})
    plain = p.add_roi(s.id, [0, 0])
    assert set(p.geometry(plain)) == {"center", "shape", "size_nm", "direction"}
    assert p.to_dict()["format_version"] == 1  # readable by an older smappy
    p.evaluate()
    assert not p.stale_steps(plain.id)
    p.set_ranges([plain.id], frames=[0, 4])
    assert p.stale_steps(plain.id)
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False, z=[-50, 0])
    doc = p.to_dict()
    assert doc["format_version"] == 2
    p.from_dict(doc)
    again = p.rois[track.id]
    assert (again.ids, again.spatial, again.z) == ({"track_id": 2}, False, [-50.0, 0.0])
    assert p.indices(again).tolist() == list(range(10, 16))
    doc["format_version"] = 3
    with pytest.raises(ValueError, match="newer"):
        p.from_dict(doc)


def test_an_roi_of_an_id_cannot_be_moved_or_made_without_an_id():
    p, s = rois_of(crossing_tracks(), filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False)
    with pytest.raises(ValueError, match="cannot be moved"):
        p.move_roi(track.id, [10, 10])
    with pytest.raises(ValueError, match="needs an id"):
        p.add_roi(s.id, [0, 0], spatial=False)
    with pytest.raises(ValueError, match="whole number"):
        p.add_roi(s.id, [0, 0], ids={"track_id": 1.5})
    with pytest.raises(ValueError, match="low to high"):
        p.add_roi(s.id, [0, 0], frames=[5, 1])
    with pytest.raises(ValueError, match="needs an id"):
        ROI.from_dict({"file_id": s.id, "center": [0, 0], "spatial": False})


def test_value_range_and_the_id_lookup():
    assert value_range(None) is None
    assert value_range([None, None]) is None
    assert value_range([1, None]) == [1.0, None]
    lookup = IdLookup(np.array([3, 1, 3, 2, 1, 3]))
    assert lookup.rows(3).tolist() == [0, 2, 5]
    assert lookup.rows([1, 2]).tolist() == [1, 3, 4]
    assert lookup.rows([7]).tolist() == []


def test_segment_regions_makes_an_roi_per_track_drawn_as_its_line():
    from smappy.plugins.region_column import ColumnROIsSettings
    from smappy.plugins.segment import SegmentRegions, SegmentSettings
    from smappy.session import Session
    session = Session(crossing_tracks())
    session.show_grouped(0, False)
    settings = SegmentSettings(method="ids", ids=ColumnROIsSettings(
        column="track_id", min_size=5, membership="id"))
    made = SegmentRegions().run(session.context(), settings).rois
    project = session.rois
    assert sorted(r.ids["track_id"] for r in made) == [1, 2]
    assert all(not r.spatial and project.draws_track(r) for r in made)
    along_x = next(r for r in made if r.ids["track_id"] == 1)
    xs, ys = project.outline(along_x)
    assert np.allclose(xs, (np.arange(10) - 4.5) * 20) and np.allclose(ys, 0)
    # selected by the line, not by a rim round it
    assert project.rim_distance(along_x, 35, 3) == pytest.approx(3)
    centre, half = project.extent(along_x)
    assert np.allclose(centre, [0, 0]) and half == pytest.approx(90)
    # a cluster column is drawn round, as its hull
    settings.ids.draw = "outline"
    settings.replace = True
    hulls = SegmentRegions().run(session.context(), settings).rois
    assert not any(project.draws_track(r) for r in hulls)


def test_segment_regions_by_shape_is_as_before():
    from smappy.plugins.region_column import ColumnROIsSettings
    from smappy.plugins.segment import SegmentRegions, SegmentSettings
    from smappy.session import Session
    session = Session(crossing_tracks())
    session.show_grouped(0, False)
    settings = SegmentSettings(method="ids", ids=ColumnROIsSettings(column="track_id",
                                                                    min_size=5))
    made = SegmentRegions().run(session.context(), settings).rois
    assert all(r.spatial and not r.ids for r in made)
    # a hull round a line has no area: the ROI manager's shape round its centre
    assert all(r.polygon is None for r in made)


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def track_window():
    from smappy.gui.roi_window import ROIManagerWindow
    from smappy.plugins.region_column import ColumnROIsSettings
    from smappy.plugins.segment import SegmentRegions, SegmentSettings
    from smappy.session import Session
    session = Session(crossing_tracks())
    session.show_grouped(0, False)
    SegmentRegions().run(session.context(), SegmentSettings(
        method="ids", ids=ColumnROIsSettings(column="track_id", min_size=5,
                                             membership="id")))
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    return session, session.rois, window


def test_a_track_is_selected_by_a_click_on_its_line_in_the_manager(app):
    session, project, window = track_window()
    along_x, along_y = sorted(project.rois.values(), key=lambda r: r.ids["track_id"])
    window.zoom_pane.render(project.state(along_x.file_id), -200, 200, -200, 200)
    assert window._roi_at(60, 1) is along_x
    assert window._roi_at(1, -60) is along_y
    assert window._roi_at(60, 60) is None
    window.select(along_y.id)
    xs, ys = window.roi_pane.shapes["roi"].getData()
    assert np.allclose(xs, 0) and len(ys) == 10           # its line, not a box
    assert "track_id 2, 10 localizations in frames 0\u20139, path 180 nm" in \
        window.roi_pane.label.text()
    project.add_roi(along_x.file_id, [500, 500])
    window.kind_box.setCurrentIndex(window.kind_box.findData("ids"))
    assert len(window.rows.ids) == 2
    window.kind_box.setCurrentIndex(window.kind_box.findData("shapes"))
    assert len(window.rows.ids) == 1
    window.close()


def test_a_click_drafts_the_nearest_track_once_the_layer_draws_tracks(app):
    from dataclasses import replace
    from smappy.gui.roi_tab import ROIHeader
    from smappy.gui.roi_window import ROIManagerWindow
    from smappy.session import Session
    session = Session(crossing_tracks())
    session.show_grouped(0, False)
    project = session.rois
    header = ROIHeader(session)
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    assert not project.drafts_tracks and header.shape.currentText() == "circle"
    state = session.layers[0].state
    state.settings = replace(state.settings, mode="tracks")
    session.changed("layer")
    assert project.drafts_tracks and header.shape.currentText() == "track"
    assert not header.size.isEnabled()

    file_id = next(iter(project.sources))
    window.zoom_pane.render(project.state(file_id), -200, 200, -200, 200)
    window._zoom_clicked(3, 70)                 # off both lines, nearer y's
    assert window.draft_track.ids == {"track_id": 2}
    # drawn thicker in its own colours, not covered by a line of one colour
    assert window.zoom_pane.shapes["draft"].getData()[0] is None or \
        len(window.zoom_pane.shapes["draft"].getData()[0]) == 0
    window._add()
    roi = project.rois[window.active]
    assert roi.ids == {"track_id": 2} and not roi.spatial
    assert project.indices(roi).tolist() == list(range(10, 20))
    window._zoom_clicked(900, 900)              # nothing there: no draft
    assert window.draft is None

    header.shape.setCurrentText("square")       # chosen: a shape, tracks or not
    assert not project.drafts_tracks and project.shape == "square"
    window._zoom_clicked(60, 3)
    assert window.draft is not None and window.draft_track is None
    window.close()
    header.deleteLater()


def test_a_selected_track_is_drawn_thicker_in_its_own_colours():
    from dataclasses import replace
    from smappy.gui.roi_window import EMPHASIS, emphasised
    from smappy.render import FieldOfView
    from smappy.session import Session
    session = Session(crossing_tracks())
    session.show_grouped(0, False)
    state = session.layers[0].state
    state.settings = replace(state.settings, mode="tracks", color_field="frame")
    snapshot = state.snapshot()
    fov = FieldOfView(-100, -100, 2.0, 100, 100)
    plain, _ = snapshot.image(fov)
    track_y = lambda locs: np.flatnonzero(np.asarray(locs["track_id"]) == 2)
    bold = emphasised(snapshot, fov, track_y)
    lit = lambda rgb, rows, cols: (rgb[rows, cols].max(axis=-1) > 0).sum()
    # across the y track (x = 0, on a pixel edge): a line w pixels wide
    # lights w + 1 there, so 2 at the layer's width and 4 at EMPHASIS times it
    assert lit(plain, 30, slice(40, 60)) == 2
    assert lit(bold, 30, slice(40, 60)) == EMPHASIS + 1
    # the x track, not chosen, is as it was; and the colours stay the render's
    assert np.array_equal(bold[50, :40], plain[50, :40])
    assert np.allclose(bold[30, 50], plain[30, 50], atol=0.05)


def test_the_frame_and_z_range_are_typed_or_dragged_and_undone(app):
    session, project, window = track_window()
    along_x = next(r for r in project.rois.values() if r.ids["track_id"] == 1)
    window.select(along_x.id)
    ranges = window.annotation.ranges
    ranges.fields[("frames", 0)].setText("3")
    ranges.fields[("frames", 0)].editingFinished.emit()
    assert project.rois[along_x.id].frames == [3.0, None]
    assert project.indices(project.rois[along_x.id]).tolist() == list(range(3, 10))
    # the band, let go between frames 2 and 6, its far end at the data's edge
    band = ranges.frame_plot.band
    band.setRegion((2.2, 10))
    band.sigRegionChangeFinished.emit(band)
    assert project.rois[along_x.id].frames == [2.0, None]
    band = ranges.side_plot.band
    band.setRegion((20, 60))
    band.sigRegionChangeFinished.emit(band)
    assert project.rois[along_x.id].z == [20.0, 60.0]
    ranges._to_all()
    assert all(r.z == [20.0, 60.0] for r in project.rois.values())
    window.undo()
    assert [r.z for r in project.rois.values()].count(None) == 1
    ranges._clear()
    assert project.rois[along_x.id].frames is None and project.rois[along_x.id].z is None
    window.close()


def test_regions_to_column_writes_the_used_tracks_their_range_and_keywords():
    from smappy.plugins.region_column import RegionColumnSettings, RegionsToColumn
    from smappy.session import Session
    session = Session(crossing_tracks())
    session.show_grouped(0, False)
    project = session.rois
    file_id = next(iter(project.sources))
    one = project.add_roi(file_id, [0, 0], ids={"track_id": 1}, spatial=False,
                          frames=[2, 6])
    two = project.add_roi(file_id, [0, 0], ids={"track_id": 2}, spatial=False)
    project.set_annotation_lists({"quality": ["good", "bad"]})
    project.annotate(one.id, "quality", "bad")
    project.rois[two.id].use = False                  # not used: in no region
    result = RegionsToColumn().run(session.context(), RegionColumnSettings())
    locs = result.locs
    roi_id = np.asarray(locs["roi_id"])
    assert roi_id.tolist() == [0, 0] + [one.number] * 5 + [0] * 13
    keyword = np.asarray(locs["quality_keyword"])
    assert np.all(keyword[2:7] == 2) and np.isnan(keyword[[0, 1, 7, 12]]).all()
