"""An ROI's properties from what was measured in it: a segmenter that
orients the tracks it keeps (*Segment regions* from an id attribute), and
*ROIs from results* -- angle, ranges, a line, a keyword, the centre -- from
any measured column; what was set by hand left alone, property by property."""
import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.plugins.region_column import ColumnROIsSettings, trace_values
from smappy.plugins.roi_results import ROIsFromResults, new_values
from smappy.plugins.segment import SegmentRegions
from smappy.roi_manager.core import ROI
from smappy.session import Session

# three motors, each 60 localizations walking 8 nm a frame at its angle, and
# three diffusing molecules going nowhere in particular
MOTOR_ANGLES = (30.0, 150.0, -100.0)


def tracks(seed=0):
    rng = np.random.default_rng(seed)
    x, y, frame, tid = [], [], [], []
    for k, angle in enumerate(MOTOR_ANGLES):
        t = np.arange(60)
        a = np.radians(angle)
        x += list(3000 * k + 8 * t * np.cos(a) + rng.normal(0, 3, 60))
        y += list(8 * t * np.sin(a) + rng.normal(0, 3, 60))
        frame += list(t + 100 * k)
        tid += [k] * 60
    for k in range(3, 6):
        steps = rng.normal(0, 20, (60, 2)).cumsum(axis=0)
        x += list(3000 * k + steps[:, 0])
        y += list(steps[:, 1])
        frame += list(np.arange(60) + 100 * k)
        tid += [k] * 60
    n = len(x)
    return Localizations({"x_nm": np.array(x), "y_nm": np.array(y),
                          "frame": np.array(frame, dtype=np.int64),
                          "tid": np.array(tid, dtype=np.int64),
                          "photons": np.full(n, 1000.0), "xy_err_nm": np.full(n, 3.0)}, {})


def off(a, b):
    """The difference of two angles, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def session_of(locs):
    session = Session()
    session.add_file(locs, FileInfo(name="f", path="", format="test"))
    session.show_grouped(0, False)
    return session


def segment(session, **ids):
    values = {"column": "tid", "min_size": 10, "membership": "id", **ids}
    return session.run(SegmentRegions(), SegmentRegions.Settings(
        method="ids", ids=ColumnROIsSettings(**values)))


def test_each_trace_is_measured_with_its_direction_forward_in_time():
    values = trace_values(tracks(), "tid")
    for k, angle in enumerate(MOTOR_ANGLES):
        assert values[k]["eccentricity"] > 5
        assert off(values[k]["angle_deg"], angle) == pytest.approx(0, abs=3)
        assert off(values[k]["ends_angle_deg"], angle) == pytest.approx(0, abs=5)


def test_keep_picks_the_motors_and_the_angle_points_the_way_they_moved():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4 and displacement_nm >= 300", angle="axis")
    rois = sorted(session.rois.rois.values(), key=lambda r: r.ids["tid"])
    assert [r.ids["tid"] for r in rois] == [0, 1, 2]          # the diffusing ones left out
    for roi, angle in zip(rois, MOTOR_ANGLES):
        assert off(roi.angle, angle) == pytest.approx(0, abs=3)
        assert roi.source("angle") == "Segment regions"
        assert roi.origin["values"]["eccentricity"] > 4        # shown as results columns
    # turned onto its angle, each motor runs along +x, left to right in time
    ctx = session.rois.region_context(rois[1])
    order = np.argsort(ctx.locs["frame"])
    along = np.asarray(ctx.locs["x_site_nm"])[order]
    assert along[-1] - along[0] > 400 and np.ptp(ctx.locs["y_site_nm"]) < 60


def test_without_keep_or_angle_the_id_method_measures_nothing():
    session = session_of(tracks())
    segment(session)
    assert len(session.rois.rois) == 6
    assert all(r.angle == 0 and not r.set_by and "values" not in r.origin
               for r in session.rois.rois.values())


def from_results(target, **part):
    """The plugin's settings: ``target`` set from ``part``'s fields."""
    settings = ROIsFromResults.Settings(target=target)
    for name, value in part.items():
        if name in ("which", "used_only"):
            setattr(settings, name, value)
        else:
            setattr(getattr(settings, target), name, value)
    return settings


def test_an_angle_needs_a_measured_column():
    session = session_of(tracks())
    segment(session)                                          # nothing measured
    with pytest.raises(ValueError, match="no ROI has angle_deg"):
        session.run(ROIsFromResults(), from_results("angle", column="angle_deg"))


def test_the_angle_comes_from_a_measured_column_and_hand_turned_rois_stay():
    session = session_of(tracks())
    # keep measures each trace, and its values are the ROIs' columns
    segment(session, keep="eccentricity >= 4")
    project = session.rois
    rois = sorted(project.rois.values(), key=lambda r: r.ids["tid"])
    assert all(r.angle == 0 for r in rois)                    # angle: none
    project.set_angle(rois[0].id, 10.0)                       # corrected by hand
    rois[2].use = False
    result = session.run(ROIsFromResults(), from_results("angle", column="angle_deg"))
    assert project.rois[rois[0].id].angle == 10.0             # by hand: kept
    assert off(project.rois[rois[1].id].angle, 150) == pytest.approx(0, abs=3)
    assert project.rois[rois[1].id].source("angle") == "ROIs from results"
    assert project.rois[rois[2].id].angle == 0                # not used: left
    assert "1 by hand" in result.text and "1 not used" in result.text
    # all of them, the hand-turned one too, in one undo step
    session.run(ROIsFromResults(), from_results("angle", column="angle_deg", which="all",
                                                used_only=False))
    assert off(project.rois[rois[0].id].angle, 30) == pytest.approx(0, abs=3)
    project.undo()
    assert project.rois[rois[0].id].angle == 10.0


def test_only_those_not_set_yet_are_given_one():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4", angle="axis")  # set by the segmenter
    project = session.rois
    values, counts = new_values(project, from_results("angle", column="angle_deg",
                                                      which="unset"))
    assert values == {} and counts["already set"] == 3


def _measured(session, **columns):
    """Values on each ROI as its segmenter's would be: columns of its row."""
    for roi in session.rois.rois.values():
        k = roi.ids["tid"]
        roi.origin = {**roi.origin, "values": {**roi.origin.get("values", {}),
                                               **{c: f(k) for c, f in columns.items()}}}


def test_a_frame_range_from_a_measured_first_frame_cuts_the_tail():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4")
    project = session.rois
    _measured(session, tail_end=lambda k: 100 * k + 10.4)    # the first 10 frames out
    hand = sorted(project.rois.values(), key=lambda r: r.ids["tid"])[2]
    project.set_ranges([hand.id], frames=[205, 250])          # by hand
    session.run(ROIsFromResults(), from_results("range", attribute="frame", low="tail_end"))
    for roi in project.rois.values():
        k = roi.ids["tid"]
        if roi.id == hand.id:
            assert roi.frames == [205.0, 250.0] and roi.source("range:frame") == "hand"
            continue
        assert roi.frames == [100 * k + 10.0, None]           # whole frames, open end
        frames = project.extract(roi)["frame"]
        assert frames.min() == 100 * k + 10 and len(frames) == 50


def test_a_z_range_from_a_centre_and_a_width():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4")
    _measured(session, z_mid=lambda k: 100.0 * k, z_sd=lambda k: 20.0)
    session.run(ROIsFromResults(), from_results("range", attribute="z_nm", mode="centre",
                                                centre="z_mid", width="z_sd", k=2.5))
    for roi in session.rois.rois.values():
        assert roi.z == [100.0 * roi.ids["tid"] - 50, 100.0 * roi.ids["tid"] + 50]


def test_a_line_along_the_measured_axis_points_forward_and_joins_the_names():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4")
    project = session.rois
    _measured(session, cx=lambda k: 3000.0 * k, cy=lambda k: 0.0, length=lambda k: 400.0)
    session.run(ROIsFromResults(), from_results("line", name="axis", x="cx", y="cy",
                                                angle="angle_deg", length="length"))
    assert "axis" in project.line_names
    for roi in project.rois.values():
        (x0, y0), (x1, y1) = roi.lines["axis"]
        angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        assert off(angle, MOTOR_ANGLES[roi.ids["tid"]]) == pytest.approx(0, abs=3)
        assert np.hypot(x1 - x0, y1 - y0) == pytest.approx(400)
        assert roi.source("line:axis") == "ROIs from results"
        assert roi.source("angle") == ""                      # another property


def test_a_keyword_from_a_condition_leaves_hand_given_ones_and_measures_nothing_again():
    session = session_of(tracks())
    segment(session, keep="n_locs >= 10")                     # all six, measured
    project = session.rois
    project.set_annotation_lists({"type": ["motor", "diffusing", "unsure"]})
    rois = sorted(project.rois.values(), key=lambda r: r.ids["tid"])
    project.annotate(rois[5].id, "type", "unsure")            # by hand
    before = {r.id: project.geometry(r) for r in rois}
    session.run(ROIsFromResults(), from_results(
        "keyword", list_name="type", condition="eccentricity >= 4", keyword="motor",
        otherwise="diffusing"))
    got = [project.rois[r.id].annotations.get("type") for r in rois]
    assert got == ["motor"] * 3 + ["diffusing"] * 2 + ["unsure"]
    assert {r.id: project.geometry(project.rois[r.id]) for r in rois} == before
    with pytest.raises(ValueError, match="not in the list"):
        session.run(ROIsFromResults(), from_results(
            "keyword", list_name="type", condition="eccentricity >= 4", keyword="fast"))


def test_a_centre_moves_a_shape_and_its_lines_with_it():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4", membership="shape", outline=False)
    project = session.rois
    roi = next(iter(project.rois.values()))
    project.set_line(roi.id, "line", [[0, 0], [10, 0]])
    start = np.asarray(roi.center)
    roi.origin = {**roi.origin, "values": {**roi.origin["values"],
                                           "fx": start[0] + 30, "fy": start[1] - 20}}
    session.run(ROIsFromResults(), from_results("centre", x="fx", y="fy"))
    moved = project.rois[roi.id]
    assert np.allclose(moved.center, start + [30, -20])
    assert moved.lines["line"] == [[30.0, -20.0], [40.0, -20.0]]
    assert moved.source("centre") == "ROIs from results"


def test_an_angle_from_an_older_file_counts_as_set_by_hand():
    old = ROI.from_dict({"file_id": "f", "center": [0, 0], "angle": 30})
    assert old.source("angle") == "hand" and old.set_by == {"angle": "hand"}
    assert ROI.from_dict({"file_id": "f", "center": [0, 0]}).set_by == {}
    measured = ROI.from_dict({"file_id": "f", "center": [0, 0], "angle": 30,
                              "set_by": {"angle": "Orient ROIs"}})
    assert measured.source("angle") == "Orient ROIs"


def test_who_set_the_angle_is_undone_with_it():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4", angle="axis")
    project = session.rois
    roi = next(iter(project.rois.values()))
    assert roi.source("angle") == "Segment regions"
    project.set_angle(roi.id, 5.0)                            # by hand
    assert project.rois[roi.id].set_by == {"angle": "hand"}
    project.undo()
    assert project.rois[roi.id].source("angle") == "Segment regions"


# ------------------------------------------------------------------ overlays

def test_overlays_are_checked_and_kept_with_a_regions_results(tmp_path):
    from smappy.plugins import Result
    from smappy.region_table import RegionTable
    from smappy.roi_manager.core import overlay_shapes, region_entry
    shapes = [{"kind": "line", "points": [[0, 0], [10, 5]], "label": "axis"},
              {"kind": "circle", "points": [[1, 2]], "radius": 50},
              {"kind": "points", "points": [[3, 4], [5, 6]]}]
    entry = region_entry(Result(values={"n": 3}, overlays=shapes))
    assert entry["overlays"] == overlay_shapes(shapes)
    for wrong, why in (([{"kind": "arrow", "points": [[0, 0]]}], "one of"),
                       ([{"kind": "line", "points": [[0, float("nan")]]}], "finite"),
                       ([{"kind": "circle", "points": [[0, 0]], "radius": 0}], "radius")):
        with pytest.raises(ValueError, match=why):
            overlay_shapes(wrong)
    table = RegionTable()
    table.set("r1", {**entry, "signature": "s"}, step={"label": "x"})
    assert table.entry("r1")["overlays"] == entry["overlays"]
    assert RegionTable.from_dict(table.to_dict()).entry("r1")["overlays"] == entry["overlays"]
    import h5py
    with h5py.File(tmp_path / "t.h5", "w") as f:
        table.to_hdf5(f.create_group("t"))
    with h5py.File(tmp_path / "t.h5", "r") as f:
        assert RegionTable.from_hdf5(f["t"]).entry("r1")["overlays"] == entry["overlays"]


def test_the_trace_inspector_measures_the_axis_in_an_roi_and_draws_it():
    from smappy.plugins.trace_inspector import TraceInspector
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4")
    project = session.rois
    roi = sorted(project.rois.values(), key=lambda r: r.ids["tid"])[1]
    ctx = project.region_context(roi)
    result = TraceInspector().run(ctx, TraceInspector.Settings())
    assert off(result.values["angle_deg"], MOTOR_ANGLES[1]) == pytest.approx(0, abs=3)
    (line,) = result.overlays
    (x0, y0), (x1, y1) = line["points"]
    assert off(np.degrees(np.arctan2(y1 - y0, x1 - x0)), MOTOR_ANGLES[1]) == \
        pytest.approx(0, abs=3)


def test_a_range_on_another_attribute_joins_the_offered_ones():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4")
    project = session.rois
    _measured(session, p_lo=lambda k: 900.0)
    session.run(ROIsFromResults(), from_results("range", attribute="photons", low="p_lo"))
    assert "photons" in project.range_columns
    assert all(r.ranges["photons"] == [900.0, None] for r in project.rois.values())
    assert all(r.source("range:photons") == "ROIs from results" for r in project.rois.values())


def test_a_centre_set_by_a_click_is_overwritten_by_a_measured_one():
    session = session_of(tracks())
    segment(session, keep="eccentricity >= 4", membership="shape", outline=False)
    project = session.rois
    roi = next(iter(project.rois.values()))
    start = np.asarray(roi.center)
    roi.origin = {**roi.origin, "values": {**roi.origin["values"],
                                           "fx": start[0] + 5, "fy": start[1]}}
    roi.set_source("centre", "hand")                          # as an older file may say
    session.run(ROIsFromResults(), from_results("centre", x="fx", y="fy"))
    assert np.allclose(project.rois[roi.id].center, start + [5, 0])
