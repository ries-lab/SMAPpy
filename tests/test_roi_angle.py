"""An ROI's angle: an annotation that the project, when it rotates its ROIs,
shows and measures them by -- a square turned with the structure, and every
measurement given the ROI's own frame (``x_site_nm``) beside ``x_nm``."""
import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.regions import SITE_X, SITE_Y, Region, SiteFrame
from smappy.roi_manager.core import ROI, normal_angle
from smappy.roi_manager.pipeline import Step
from smappy.session import Session


def _session(points):
    xy = np.asarray(points, float)
    locs = Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(len(xy), dtype=np.int64),
                          "photons": np.full(len(xy), 1000.0),
                          "xy_err_nm": np.full(len(xy), 5.0)}, {})
    session = Session()
    session.add_file(locs, FileInfo(name="f", path="", format="test"))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(200, "square")
    return session, project


# about (0, 0): a point a 45 degree turn takes into a 200 nm square and one it
# takes out, and a far one no square holds
INSIDE_TURNED, INSIDE_UPRIGHT, FAR = (120.0, 0.0), (90.0, 90.0), (5000.0, 5000.0)


class Frames(Plugin):
    """What a measurement in an ROI is given of its frame."""
    changes_locs = False
    per_region = True

    def run(self, ctx, settings=None):
        frame = ctx.frame
        return Result(values={"n": float(len(ctx.locs)),
                              "angle": float("nan") if frame is None else frame.angle,
                              "u_max": float(np.max(ctx.locs[SITE_X])),
                              "x_max": float(np.max(ctx.locs["x_nm"]))})


def _measure(project):
    steps = [Step(label="Frames", path="Test/Frames", plugin=Frames(), settings=None)]
    project.evaluate(steps=steps)
    return {r["roi_id"]: r for r in project.results()}, steps


def test_the_sites_frame_turns_a_structure_at_its_angle_onto_plus_x():
    frame = SiteFrame([10.0, 20.0], 30.0)
    a = np.radians(30)
    uv = frame.to_site([[10 + 5 * np.cos(a), 20 + 5 * np.sin(a)]])
    assert uv == pytest.approx(np.array([[5.0, 0.0]]))
    xy = np.array([[3.0, -4.0], [100.0, 7.0]])
    assert frame.to_data(frame.to_site(xy)) == pytest.approx(xy)
    u, v = frame.columns(xy[:, 0], xy[:, 1])
    assert np.column_stack((u, v)) == pytest.approx(frame.to_site(xy))


def test_an_angle_is_kept_between_minus_and_plus_180():
    assert normal_angle(190) == -170 and normal_angle(-180) == 180
    assert normal_angle(-0.0) == 0.0 and normal_angle(45) == 45
    with pytest.raises(ValueError):
        normal_angle(float("nan"))


def test_an_older_files_direction_line_gives_the_roi_its_angle():
    roi = ROI.from_dict({"file_id": "f", "center": [0, 0],
                         "lines": {"direction": [[0, 0], [0, -50]]}})
    assert roi.angle == -90 and "direction" in roi.lines
    roi = ROI.from_dict({"file_id": "f", "center": [0, 0], "angle": 10,
                         "lines": {"direction": [[0, 0], [0, -50]]}})
    assert roi.angle == 10                       # its own, once it has one


def test_without_rotation_an_roi_is_still_measured_in_its_own_frame():
    """*rotate* is a way of looking: a plugin that wants the structure's axes
    gets them whether or not it is ticked, so an unticked box cannot hand it
    the file's axes instead.  The square stays upright, as it is shown."""
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    roi = project.add_roi(next(iter(project.sources)), [0, 0])
    assert "roi_angle_deg" not in project.sort_rows()[roi.id]    # no angles: no column
    before = project.geometry(roi)
    assert project.frame(roi).angle == 0                 # never turned: the file's axes
    project.set_angle(roi.id, 45)
    assert project.geometry(roi) != before               # the frame changed: out of date
    assert project.region(roi).kind == "rect"            # the square not turned
    ctx = project.region_context(roi)
    assert sorted(ctx.locs["x_nm"]) == [90.0]
    assert ctx.frame.angle == 45                         # its own frame all the same
    assert ctx.locs[SITE_X][0] == pytest.approx(90 * np.sqrt(2))
    assert ctx.locs[SITE_Y][0] == pytest.approx(0.0, abs=1e-9)
    assert project.sort_rows()[roi.id]["roi_angle_deg"] == 45


def test_rotated_a_square_turns_with_its_roi_and_a_polygon_does_not():
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    file_id = next(iter(project.sources))
    roi = project.add_roi(file_id, [0, 0])
    project.set_angle(roi.id, 45)
    project.set_rotate(True)
    ctx = project.region_context(roi)
    assert list(ctx.locs["x_nm"]) == [120.0]             # absolute, as ever
    assert ctx.locs[SITE_X][0] == pytest.approx(120 * np.cos(np.radians(45)))
    assert ctx.locs[SITE_Y][0] == pytest.approx(-120 * np.sin(np.radians(45)))
    assert ctx.frame.angle == 45 and "at 45" in project.describe(roi)
    # a polygon is drawn round localizations: the angle leaves it where it is
    drawn = project.add_roi(file_id, [0, 0], polygon=Region.square([0, 0], 200).vertices())
    project.set_angle(drawn.id, 45)
    assert sorted(project.region_context(drawn).locs["x_nm"]) == [90.0]
    # and the session's own table never carries the frame
    assert SITE_X not in session.locs


def test_turning_a_rotated_roi_puts_its_results_out_of_date_and_is_undone():
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    roi = project.add_roi(next(iter(project.sources)), [0, 0])
    project.set_rotate(True)
    measured, steps = _measure(project)
    assert measured[roi.number]["n"] == 1 and measured[roi.number]["x_max"] == 90
    project.set_angle(roi.id, 45)
    assert project.stale_steps(roi.id, steps)
    measured, _ = _measure(project)
    row = measured[roi.number]
    assert row["angle"] == 45 and row["x_max"] == 120
    assert row["u_max"] == pytest.approx(120 * np.cos(np.radians(45)))
    project.undo()
    assert project.rois[roi.id].angle == 0          # undo puts copies back


def test_switching_rotation_off_keeps_the_frame_and_is_saved_with_the_project():
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    roi = project.add_roi(next(iter(project.sources)), [0, 0])
    project.set_angle(roi.id, 45)
    project.set_rotate(True)
    turned = project.geometry(roi)
    project.set_rotate(False)
    upright = project.geometry(roi)
    # the angle stays in the signature (it is the frame); the square is upright
    assert upright != turned and upright["angle"] == 45 and upright["upright"]
    assert project.frame(roi).angle == 45
    project.set_rotate(True)
    saved = project.to_dict()
    project.from_dict(None)
    assert not project.rotate
    project.from_dict(saved)
    assert project.rotate and project.rois[roi.id].angle == 45
    assert project.geometry(project.rois[roi.id]) == turned        # as before


def test_a_rotated_roi_is_looked_at_along_its_angle_in_3d():
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    roi = project.add_roi(next(iter(project.sources)), [0, 0])
    project.set_angle(roi.id, 30)
    assert project.view_region(roi).kind == "rect"       # not rotated: its shape
    project.set_rotate(True)
    line = project.view_region(roi)
    p0, p1 = line.ends()
    d = p1 - p0
    assert np.degrees(np.arctan2(d[1], d[0])) == pytest.approx(30)
    assert np.hypot(*d) == pytest.approx(200) and line.width == pytest.approx(200)


def test_average_rois_turns_each_roi_by_its_own_angle(tmp_path):
    from smappy.plugins.average_rois import AverageROIs
    from test_average_rois import ARROW, averaged, found_at, turned_arrows
    session, copies = turned_arrows(tmp_path, n=9)
    project = session.rois
    for roi, gamma in zip(project.rois.values(), copies["gamma_deg"]):
        project.set_angle(roi.id, gamma)
    settings = AverageROIs.Settings()
    settings.orientation.by = "roi"
    session.run(AverageROIs(), settings)
    x, y, out = averaged(session)
    assert SITE_X not in out
    n, mx, my = found_at(x, y, ARROW[2])
    assert n > 40 and mx == pytest.approx(50, abs=4) and my == pytest.approx(40, abs=4)


# ------------------------------------------------------------------ the GUI

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_line_starts_with_a_ring_and_ends_with_a_head_of_fixed_size():
    from smappy.gui.roi_window import ARROW_HEAD_PIXELS, arrow
    xs, ys = arrow([[0, 0], [100, 0]], pixel=2.0)
    xs, ys = np.asarray(xs), np.asarray(ys)
    pieces = np.split(np.column_stack((xs, ys)), np.flatnonzero(np.isnan(xs)))
    shaft, head, ring = (p[~np.isnan(p).any(axis=1)] for p in pieces)
    assert shaft[-1] == pytest.approx([100, 0])
    assert 100 - head[:, 0].min() == pytest.approx(ARROW_HEAD_PIXELS * 2)
    assert np.hypot(*ring.T) == pytest.approx(np.full(len(ring), 8.0))   # round the start


def test_the_rotate_box_shows_the_angle_and_the_pane_turns_and_maps_clicks_back(app):
    from smappy.gui.roi_window import ROIManagerWindow
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    roi = project.add_roi(next(iter(project.sources)), [0, 0])
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.select(roi.id)
    assert window.angle_box.isHidden() and window.roi_pane.turn is None
    assert window.roi_pane.handle is None
    window.rotate_box.setChecked(True)
    assert project.rotate and not window.angle_box.isHidden()
    assert window.roi_pane.handle is not None
    # the handle dragged a quarter turn anticlockwise on the data's axes
    window.roi_pane.turning.emit(-30.0, False)
    assert window.roi_pane.turn[1] == pytest.approx(30)
    assert roi.angle == 0                                  # stored at the release
    window.roi_pane.turned.emit(-40.0, True)               # Shift: to 15 degrees
    assert roi.angle == 45 and window.angle_box.value() == 45
    assert window.roi_pane.turn[1] == 45
    # a point of the turned picture is the data's point under it
    from PySide6.QtCore import QPointF
    mapped = window.roi_pane.content.mapToParent(QPointF(120.0, 0.0))
    back = window.roi_pane.content.mapFromParent(mapped)
    assert (mapped.x(), mapped.y()) == (pytest.approx(84.85, abs=0.01),
                                        pytest.approx(-84.85, abs=0.01))
    assert (back.x(), back.y()) == (pytest.approx(120), pytest.approx(0, abs=1e-9))
    window._start_drawing("line")
    window._roi_clicked(-50.0, 0.0)                        # clicks arrive in nm
    window._roi_clicked(50.0, 0.0)
    window.angle_from_line.click()
    assert roi.angle == 0
    window.angle_box.setValue(-60)
    assert roi.angle == -60
    window._step_angle(1.0)
    assert roi.angle == -59
    window.rotate_box.setChecked(False)
    assert window.roi_pane.turn is None and window.angle_box.isHidden()
    assert window.roi_pane.handle is None and roi.angle == -59
    window.close()


def test_a_drafted_roi_takes_its_angle_when_it_is_added(app):
    from smappy.gui.roi_window import ROIManagerWindow
    session, project = _session([INSIDE_TURNED, INSIDE_UPRIGHT, FAR])
    project.set_rotate(True)
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.draft = np.array([0.0, 0.0])
    window.redraw()
    window.set_angle(20)
    assert window.roi_pane.turn[1] == 20
    window._add()
    assert project.rois[window.active].angle == 20
    window.close()


def test_a_range_cuts_the_roi_image_inside_the_roi_only_and_a_draft_keeps_its_own(app):
    from smappy.gui.roi_window import ROIManagerWindow
    # frames 0..3 inside a 200 nm square about (0, 0), 4..7 far outside it
    points = [(10.0, 0.0), (-20.0, 10.0), (0.0, 30.0), (40.0, -40.0),
              (5000.0, 5000.0), (5010.0, 5000.0), (5020.0, 5000.0), (5030.0, 5000.0)]
    session, project = _session(points)
    window = ROIManagerWindow(session)
    window.set_measuring(False)
    window.draft = np.array([0.0, 0.0])
    window.redraw()
    assert window.shown_roi().frames is None                       # a draft: all frames
    window.annotation.ranges.fields[("frame", 0)].setText("1")
    window.annotation.ranges.fields[("frame", 1)].setText("2")
    window.annotation.ranges._typed("frame")
    assert window.draft_ranges == {"frame": [1.0, 2.0]} and not project.rois  # no edit yet
    view = project.view(window.current_file()), project.state(window.current_file())
    layers, _ = window._cut_to_ranges(view)
    shown = np.flatnonzero(next(s for s in layers if s.locs is not None).mask)
    assert shown.tolist() == [1, 2, 4, 5, 6, 7]      # inside: frames 1-2; outside: all
    window._add()
    roi = project.rois[window.active]
    assert roi.frames == [1.0, 2.0]
    window.zoom_center = np.array([1000.0, 1000.0])
    window._zoom_clicked(1000.0, 1000.0)                           # a new draft
    assert window.shown_roi().frames is None
    window.close()
