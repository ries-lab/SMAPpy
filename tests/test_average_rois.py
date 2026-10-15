"""Average ROIs: structures at random orientations superimposed back into one."""
import numpy as np
import pytest
import yaml

from smappy.chain import Block, Step
from smappy.io.formats import FileInfo
from smappy.plugins.average_rois import AverageROIs, superimpose
from smappy.session import Session
from smappy.simulate import simulate
from smappy.simulate.settings import SimulationSettings, StructureSettings

# three labels, no two alike in their distances: an arrow whose turn shows
ARROW = [[-50, 0, 0], [50, 0, 0], [50, 40, 0]]


def turned_arrows(tmp_path, n=25, seed=3):
    """``n`` arrows on a grid, each turned at random: the session, with an ROI
    on each and the copies' truth (positions and turns)."""
    spec = {"name": "arrow", "elements": [{"points": ARROW}],
            "copies": {"n": n, "placement": "grid", "rotation": "random",
                       "field": [0, 0, 5000, 5000]}}
    path = tmp_path / "arrow.yaml"
    path.write_text(yaml.safe_dump(spec))
    locs = simulate(SimulationSettings(n_frames=2000, seed=seed,
                                       structure=StructureSettings(file=str(path))))
    session = Session()
    session.add_file(locs, FileInfo(name="arrows", path="arrows", format="simulated",
                                    n=len(locs)))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(250)
    file_id = next(iter(project.sources))
    copies = locs.metadata["copies"]
    for x, y in zip(copies["x_nm"], copies["y_nm"]):
        project.add_roi(file_id, [x, y])
    return session, copies


def draw_lines(project, copies, turn_by=0.0):
    """A line on each ROI along its copy's turn (plus ``turn_by``)."""
    for roi, gamma in zip(project.rois.values(), copies["gamma_deg"]):
        a = np.radians(gamma + turn_by)
        x, y = roi.center
        project.set_line(roi.id, "direction", [[x, y], [x + 60 * np.cos(a),
                                                        y + 60 * np.sin(a)]])


def averaged(session):
    out = session.locs[np.asarray(session.locs["filenumber"]) == len(session.files) - 1]
    return np.asarray(out["x_nm"]), np.asarray(out["y_nm"]), out


def found_at(x, y, where, within=20):
    near = np.hypot(x - where[0], y - where[1]) < within
    return near.sum(), np.median(x[near]), np.median(y[near])


def test_superimposing_turns_a_direction_onto_plus_x():
    x, y = superimpose([10 + np.cos(np.radians(30))], [20 + np.sin(np.radians(30))],
                       (10, 20), 30.0)
    assert x[0] == pytest.approx(1) and y[0] == pytest.approx(0, abs=1e-12)


def test_turned_arrows_are_averaged_back_into_the_arrow_by_their_lines(tmp_path):
    session, copies = turned_arrows(tmp_path)
    draw_lines(session.rois, copies)
    settings = AverageROIs.Settings()
    settings.orientation.by = "line"
    result = session.run(AverageROIs(), settings)
    assert result.values["n_rois"] == 25
    assert session.files[-1].name == "average of 25 ROIs"
    x, y, out = averaged(session)
    for label in ARROW:
        n, mx, my = found_at(x, y, label)
        assert n > 100
        assert (mx, my) == (pytest.approx(label[0], abs=3), pytest.approx(label[1], abs=3))
    # each ROI's own frames, the file's kept beside them
    roi_id, frame = np.asarray(out["roi_id"]), np.asarray(out["frame"])
    first, second = sorted(set(roi_id))[:2]
    assert frame[roi_id == first].max() < frame[roi_id == second].min()
    assert np.asarray(out["frame_in_file"]).max() < 2000
    from matplotlib.figure import Figure
    figure = Figure()
    result.plot.draw_into(figure)
    assert len(figure.axes) == 2


def test_without_a_turn_the_average_is_a_blur_and_a_reversed_line_flips_it(tmp_path):
    session, copies = turned_arrows(tmp_path)
    session.run(AverageROIs(), AverageROIs.Settings())          # centred only
    x, y, _ = averaged(session)
    assert found_at(x, y, ARROW[2])[0] < 80       # smeared round; aligned, 170
    draw_lines(session.rois, copies, turn_by=180)
    settings = AverageROIs.Settings()
    settings.orientation.by = "line"
    settings.output = "table"
    session.run(AverageROIs(), settings)
    x, y = np.asarray(session.locs["x_nm"]), np.asarray(session.locs["y_nm"])
    n, mx, my = found_at(x, y, (-50, -40))                       # turned half round
    assert n > 100 and mx == pytest.approx(-50, abs=3) and my == pytest.approx(-40, abs=3)


def test_a_measured_centre_and_angle_place_each_roi(tmp_path):
    """Cluster shape's centre is the arrow's mean, and an ROI drawn off it
    is still centred there; a line's point and a measured angle as well."""
    session, copies = turned_arrows(tmp_path, n=9)
    project = session.rois
    for roi in list(project.rois.values()):
        project.move_roi(roi.id, np.asarray(roi.center) + 15)     # off the centre
    draw_lines(project, copies)
    project.block = Block(steps=[Step(plugin="Analysis/Measure/Cluster shape")])
    project.evaluate()
    settings = AverageROIs.Settings()
    settings.centre.by = "measured"
    settings.orientation.by = "measured"
    settings.orientation.angle = "direction_angle_deg"
    session.run(AverageROIs(), settings)
    x, y, _ = averaged(session)
    mean = np.mean(ARROW, axis=0)
    n, mx, my = found_at(x, y, np.array(ARROW[2][:2]) - mean[:2])
    assert n > 30 and mx == pytest.approx(50 - mean[0], abs=4)

    settings = AverageROIs.Settings()
    settings.centre.by, settings.centre.point = "line", "end"
    settings.orientation.by = "line"
    session.run(AverageROIs(), settings)
    x, y, _ = averaged(session)
    # the line starts 15 nm off the copy's centre, so its end is 60 nm
    # along from there: the arrow sits 60 nm back, shifted by the turned 15
    assert found_at(x, y, (-50 - 60, 0), within=40)[0] > 30


def test_rois_without_what_the_settings_ask_for_are_left_out_and_said(tmp_path):
    rng = np.random.default_rng(0)
    from smappy.locs import Localizations
    xy = np.concatenate([rng.normal(c, 10, (50, 2)) for c in ((500, 500), (1500, 500))])
    locs = Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(len(xy)), "photons": np.full(len(xy), 500.0)})
    session = Session()
    session.add_file(locs, FileInfo(name="two", path="two", format="test", n=len(locs)))
    project = session.rois
    file_id = next(iter(project.sources))
    a = project.add_roi(file_id, [500, 500])
    project.add_roi(file_id, [1500, 500])
    project.set_line(a.id, "direction", [[500, 500], [500, 600]])
    settings = AverageROIs.Settings()
    settings.orientation.by = "line"
    result = session.run(AverageROIs(), settings)
    assert result.values == {"n_rois": 1, "n_skipped": 1, "n_localizations": 50}
    assert "1 with no line direction" in result.text
    x, y, _ = averaged(session)
    assert abs(np.median(y)) < 5 and abs(np.median(x)) < 5
    bare = Session(locs)                         # a table, and no file it came from
    bare.rois.add_roi(next(iter(bare.rois.sources)), [500, 500])
    with pytest.raises(ValueError, match="in no file"):
        AverageROIs()(ctx=bare.context())


def test_the_form_shows_only_what_the_choices_read():
    settings = AverageROIs.Settings()
    state = AverageROIs().describe(None, settings)
    assert not state.shown["centre.x"] and not state.shown["orientation.line"]
    settings.centre.by, settings.orientation.by = "measured", "line"
    state = AverageROIs().describe(None, settings)
    assert state.shown["centre.x"] and state.shown["orientation.line"]
